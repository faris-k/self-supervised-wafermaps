import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from lightly.data import LightlyDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from ssl_wafermap.models.evals import LinearClassifier
from ssl_wafermap.utilities.data import TensorDataset, WaferMapDataset
from ssl_wafermap.utilities.transforms import (
    get_base_transforms,
    get_inference_transforms,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pl.seed_everything(0)

torch.set_float32_matmul_precision("high")

# suppress annoying torchmetrics and lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")
warnings.filterwarnings("ignore", ".*logging interval.*")
warnings.filterwarnings("ignore", ".*confusion.*")

num_workers = 2
knn_batch_size = 64
use_amp = True
max_epochs_probe = 1000
max_epochs_resnet = 100

data_path = "../data/processed/WM811K"
train_1_split = pd.read_pickle(os.path.join(data_path, "train_1_split.pkl.xz"))
train_10_split = pd.read_pickle(os.path.join(data_path, "train_10_split.pkl.xz"))
train_20_split = pd.read_pickle(os.path.join(data_path, "train_20_split.pkl.xz"))
train_29_split = pd.read_pickle(os.path.join(data_path, "train_29_split.pkl.xz"))
train_data = pd.read_pickle(os.path.join(data_path, "train_data.pkl.xz"))
val_data = pd.read_pickle(os.path.join(data_path, "val_data.pkl.xz"))
test_data = pd.read_pickle(os.path.join(data_path, "test_data.pkl.xz"))

# First, get the k-NN test metrics
# Use the test_data for the dataloader_kNN passed to KNNBenchmarkModules
dataset_train_kNN = LightlyDataset.from_torch_dataset(
    WaferMapDataset(train_data.waferMap, train_data.failureCode),
    transform=get_inference_transforms(),
)
dataset_test_kNN = LightlyDataset.from_torch_dataset(
    WaferMapDataset(val_data.waferMap, val_data.failureCode),
    transform=get_inference_transforms(),
)

dataloader_train_kNN = DataLoader(
    dataset_train_kNN,
    batch_size=knn_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False,
)
dataloader_test_kNN = DataLoader(
    dataset_test_kNN,
    batch_size=knn_batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True if num_workers > 0 else False,
)

# Then, these dataloaders will be used for validating and testing the linear classifier
# They will be frozen features from the SSL model on these datasets
ssl_val_dataset = LightlyDataset.from_torch_dataset(
    WaferMapDataset(
        val_data.waferMap,
        val_data.failureCode,
    ),
    transform=get_inference_transforms(),
)
ssl_test_dataset = LightlyDataset.from_torch_dataset(
    WaferMapDataset(
        test_data.waferMap,
        test_data.failureCode,
    ),
    transform=get_inference_transforms(),
)

ssl_val_loader = DataLoader(
    ssl_val_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True if num_workers > 0 else False,
)
ssl_test_loader = DataLoader(
    ssl_test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True if num_workers > 0 else False,
)

# These will be the splits used for training
train_splits = {
    "train_1_split": train_1_split,
    "train_10_split": train_10_split,
    "train_20_split": train_20_split,
}

ckpt_dir = "../models/new_knn/"
ckpt_file_end = "checkpoints/epoch=149-step=87450.ckpt"


class SupervisedR18(pl.LightningModule):
    def __init__(self, num_classes=9, weight=None):
        super().__init__()
        # The backbone will only be used for feature extraction
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        # Full model will be used for training
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=num_classes
        )
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.train_acc(y_hat, y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def knn_test():
    # First, test the random init SupervisedR18 model
    model = SupervisedR18(dataloader_kNN=dataloader_train_kNN, num_classes=9)
    trainer = pl.Trainer(
        accelerator="gpu",
        logger=False,
        inference_mode=True,
        precision="16-mixed",
        # enable_progress_bar=False,
    )
    trainer.validate(model, dataloader_test_kNN)

    # Loop over all checkpoints in the ckpt_dir
    for folder in os.listdir(ckpt_dir):
        # Full path, i.e. ../models/new_knn/MAE/checkpoints/epoch=149-step=87450.ckpt
        ckpt_path = os.path.join(ckpt_dir, folder, ckpt_file_end)

        # Get model name, i.e. MAE
        model_name = ckpt_path.split("new_knn/")[-1].split("\\")[0]
        print(model_name)
        if model_name == "MAE2":
            model_name = "MAE"
        # Get the model class using the model_name
        ModelClass = getattr(sys.modules[__name__], model_name)

        # Instantiate a model of this class and load ckpt weights
        model = ModelClass(dataloader_kNN=dataloader_train_kNN, num_classes=9)
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])

        # Test the model
        trainer.validate(model, dataloader_test_kNN)


def train_supervised():
    for split_name, df in train_splits.items():
        X_train, y_train = df.waferMap, df.failureCode

        weight = compute_class_weight(
            class_weight="balanced", classes=np.unique(y_train.values), y=y_train.values
        )
        weight = torch.tensor(weight, dtype=torch.float32)

        train_dataset = LightlyDataset.from_torch_dataset(
            WaferMapDataset(X_train, y_train), transform=get_base_transforms()
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=knn_batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        model = SupervisedR18(weight=weight)

        logger = TensorBoardLogger(
            save_dir="../models/wm811k_evals/supervised",
            name="",
            sub_dir=split_name,
        )
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
        )

        trainer = pl.Trainer(
            accelerator="gpu",
            max_epochs=max_epochs_resnet,
            precision="16-mixed" if use_amp else "32-true",
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5, mode="min"),
                RichProgressBar(),
                checkpoint_callback,
            ],
            logger=logger,
            benchmark=True,
        )

        trainer.fit(
            model, train_dataloaders=train_loader, val_dataloaders=ssl_val_loader
        )
        trainer.test(model, ssl_test_loader)


def linear_probe(
    model,
    model_name,
    train_df,
    train_split_name,
    inference_trainer,
    val_features,
    test_features,
):
    ssl_train_dataset = LightlyDataset.from_torch_dataset(
        WaferMapDataset(train_df.waferMap, train_df.failureCode),
        transform=get_inference_transforms(),
    )
    ssl_train_loader = DataLoader(
        ssl_train_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    train_features = inference_trainer.predict(model, ssl_train_loader)
    train_features = torch.cat(train_features, dim=0)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    val_features = scaler.transform(val_features)
    test_features = scaler.transform(test_features)

    # Now we can create the frozen feature banks
    tensor_train_dataset = TensorDataset(train_features, train_df.failureCode.values)
    tensor_val_dataset = TensorDataset(val_features, val_data.failureCode.values)
    tensor_test_dataset = TensorDataset(test_features, test_data.failureCode.values)

    # Create the dataloaders
    train_loader = DataLoader(
        tensor_train_dataset,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        tensor_val_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        tensor_test_dataset,
        batch_size=1024,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if num_workers > 0 else False,
    )

    # Calculate the weight for the CrossEntropyLoss
    weight = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_df.failureCode.values),
        y=train_df.failureCode.values,
    )
    weight = torch.tensor(weight, dtype=torch.float32)

    # Create the model
    linear_model = LinearClassifier(num_features=train_features.shape[1], weight=weight)

    logger = TensorBoardLogger(
        save_dir=f"../models/wm811k_evals/{model_name}",
        name="",
        sub_dir=f"{train_split_name}",
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
    )
    supervised_trainer = pl.Trainer(
        accelerator="gpu",
        enable_model_summary=False,
        max_epochs=max_epochs_probe,
        precision="16-mixed" if use_amp else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=50, mode="min"),
            RichProgressBar(),
            checkpoint_callback,
        ],
        logger=logger,
        benchmark=True,
    )
    supervised_trainer.fit(
        linear_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    supervised_trainer.test(linear_model, test_loader, ckpt_path="best")


def linear_probe_ssl():
    # Loop over all checkpoints in the ckpt_dir
    for folder in os.listdir(ckpt_dir):
        # Full path, i.e. ../models/new_knn/MAE/checkpoints/epoch=149-step=87450.ckpt
        ckpt_path = os.path.join(ckpt_dir, folder, ckpt_file_end)

        # Get model name, i.e. MAE
        model_name = ckpt_path.split("new_knn/")[-1].split("\\")[0]
        if model_name == "MAE2":
            model_name = "MAE"
        # Get the model class using the model_name
        ModelClass = getattr(sys.modules[__name__], model_name)

        # Instantiate a model of this class and load ckpt weights
        model = ModelClass(dataloader_kNN=dataloader_train_kNN, num_classes=9)
        model.load_from_checkpoint(ckpt_path)

        print(f"\n\n\nLoaded {model_name} from checkpoint")

        inference_trainer = pl.Trainer(
            accelerator="auto",
            precision="16-mixed" if use_amp else "32-true",
            enable_checkpointing=False,
            enable_model_summary=False,
            inference_mode=True,
            logger=False,
            callbacks=[RichProgressBar()],
        )

        print("Predicting on validation set")
        val_features = inference_trainer.predict(model, ssl_val_loader)
        print("Predicting on test set")
        test_features = inference_trainer.predict(model, ssl_test_loader)

        val_features = torch.cat(val_features, dim=0)
        test_features = torch.cat(test_features, dim=0)

        for train_split_name, train_df in train_splits.items():
            print(f"Evaluating using on {train_split_name}")
            linear_probe(
                model,
                model_name,
                train_df,
                train_split_name,
                inference_trainer,
                val_features,
                test_features,
            )


if __name__ == "__main__":
    # knn_test()
    # train_supervised()
    linear_probe_ssl()
