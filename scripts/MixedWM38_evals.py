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
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)

from ssl_wafermap.utilities.data import TensorDataset, WaferMapDataset
from ssl_wafermap.utilities.transforms import (
    get_base_transforms,
    get_inference_transforms,
)

torch.set_float32_matmul_precision("high")

# Ignore annoying pytorch lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")
warnings.filterwarnings("ignore", ".*meaningless.*")
warnings.filterwarnings("ignore", ".*logging interval.*")

train_data = pd.read_pickle("../data/processed/MixedWM38/train_data.pkl.xz")
val_data = pd.read_pickle("../data/processed/MixedWM38/val_data.pkl.xz")
test_data = pd.read_pickle("../data/processed/MixedWM38/test_data.pkl.xz")
train_1_split = pd.read_pickle("../data/processed/MixedWM38/train_1_split.pkl.xz")
train_5_split = pd.read_pickle("../data/processed/MixedWM38/train_5_split.pkl.xz")
train_10_split = pd.read_pickle("../data/processed/MixedWM38/train_10_split.pkl.xz")
train_20_split = pd.read_pickle("../data/processed/MixedWM38/train_20_split.pkl.xz")

num_workers = 2
batch_size = 64
use_amp = True
max_epochs_resnet = 100
max_epochs_probe = 1000


class SupervisedR18(pl.LightningModule):
    def __init__(self, num_classes=8, pos_weight=None):
        super().__init__()
        # The backbone will only be used for feature extraction
        self.backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        # Full model will be used for training
        self.model = timm.create_model(
            "resnet18", pretrained=False, num_classes=num_classes
        )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_acc = MultilabelAccuracy(num_labels=num_classes)
        self.train_auc = MultilabelAUROC(num_labels=num_classes)
        self.train_f1 = MultilabelF1Score(num_labels=num_classes)

        self.val_acc = MultilabelAccuracy(num_labels=num_classes)
        self.val_auc = MultilabelAUROC(num_labels=num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=num_classes)

        self.test_acc = MultilabelAccuracy(num_labels=num_classes)
        self.test_auc = MultilabelAUROC(num_labels=num_classes)
        self.test_f1 = MultilabelF1Score(num_labels=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        self.train_acc(y_hat, y)
        self.train_auc(y_hat, y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        self.val_acc(y_hat, y)
        self.val_auc(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(y_hat, y)
        self.test_auc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        return self.backbone(images)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Create a simple linear classifier for fine-tuning
# This will be for a multi-label classification problem
class MultilabelLinearClassifier(pl.LightningModule):
    def __init__(self, num_features, num_classes=8, pos_weight=None):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.pos_weight = pos_weight

        self.model = nn.Linear(self.num_features, self.num_classes)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_acc = MultilabelAccuracy(num_labels=num_classes)
        self.train_auc = MultilabelAUROC(num_labels=num_classes)
        self.train_f1 = MultilabelF1Score(num_labels=num_classes)

        self.val_acc = MultilabelAccuracy(num_labels=num_classes)
        self.val_auc = MultilabelAUROC(num_labels=num_classes)
        self.val_f1 = MultilabelF1Score(num_labels=num_classes)

        self.test_acc = MultilabelAccuracy(num_labels=num_classes)
        self.test_auc = MultilabelAUROC(num_labels=num_classes)
        self.test_f1 = MultilabelF1Score(num_labels=num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        self.train_acc(y_hat, y)
        self.train_auc(y_hat, y)
        self.train_f1(y_hat, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log(
            "train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_auc", self.train_auc, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("train_f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.float())

        self.val_acc(y_hat, y)
        self.val_auc(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(y_hat, y)
        self.test_auc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_auc", self.test_auc, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2)
        return optimizer


val_dataset = WaferMapDataset(
    val_data.waferMap, val_data.label, transform=get_inference_transforms()
)
test_dataset = WaferMapDataset(
    test_data.waferMap, test_data.label, transform=get_inference_transforms()
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True if num_workers > 0 else False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
    persistent_workers=True if num_workers > 0 else False,
)

train_splits = {
    "train_20_split": train_20_split,
    "train_10_split": train_10_split,
    "train_5_split": train_5_split,
    "train_1_split": train_1_split,
    # "train_data": train_data,
}


def train_supervised():
    for split_name, df in train_splits.items():
        X_train, y_train = df.waferMap, df.label

        label_array = np.vstack(y_train)
        pos_class_freq = np.sum(label_array, axis=0) / label_array.shape[0]
        neg_class_freq = 1 - pos_class_freq
        pos_weight = torch.tensor(neg_class_freq / pos_class_freq)

        train_dataset = WaferMapDataset(
            X_train, y_train, transform=get_base_transforms(denoise=True)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if num_workers > 0 else False,
        )

        model = SupervisedR18(pos_weight=pos_weight)

        logger = TensorBoardLogger(
            save_dir="../models/mixedwm38_finetune/supervised",
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

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(model, test_loader)


ssl_val_dataset = LightlyDataset.from_torch_dataset(
    val_dataset, transform=get_inference_transforms()
)
ssl_test_dataset = LightlyDataset.from_torch_dataset(
    test_dataset, transform=get_inference_transforms()
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
        WaferMapDataset(
            train_df.waferMap, train_df.failureType.factorize(sort=True)[0]
        ),
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
    tensor_train_dataset = TensorDataset(train_features, np.vstack(train_df.label))
    tensor_val_dataset = TensorDataset(val_features, np.vstack(val_data.label))
    tensor_test_dataset = TensorDataset(test_features, np.vstack(test_data.label))

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

    # Calculate the pos_weight for the BCEWithLogitsLoss
    label_array = np.vstack(train_df.label)
    pos_class_freq = np.sum(label_array, axis=0) / label_array.shape[0]
    neg_class_freq = 1 - pos_class_freq
    pos_weight = torch.tensor(neg_class_freq / pos_class_freq)

    # Create the model
    linear_model = MultilabelLinearClassifier(
        num_features=train_features.shape[1], pos_weight=pos_weight
    )

    logger = TensorBoardLogger(
        save_dir=f"../models/mixedwm38_finetune/{model_name}",
        name="",
        sub_dir=train_split_name,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, "checkpoints"),
    )

    supervised_trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=100,
        precision="16-mixed" if use_amp else "32-true",
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=15, mode="min"),
            RichProgressBar(),
            checkpoint_callback,
        ],
        logger=logger,
        benchmark=True,
    )

    supervised_trainer.fit(
        linear_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    supervised_trainer.test(linear_model, test_loader)


def linear_probe_ssl():
    ckpt_dir = "../models/mixed_wm38_pretrain/wafermaps/version_2/"
    for subdir, dirs, files in os.walk(ckpt_dir):
        for file in files:
            if file.endswith(".ckpt"):
                ckpt_path = os.path.join(subdir, file)

                model_name = ckpt_path.split("version_2/")[-1].split("\\")[0]
                print(model_name)

                ModelClass = getattr(sys.modules[__name__], model_name)

                model = ModelClass()
                model.load_state_dict(torch.load(ckpt_path)["state_dict"])

                # model = ModelClass.load_from_checkpoint(ckpt_path)

                print(f"Loaded {model_name} from checkpoint")

                inference_trainer = pl.Trainer(
                    accelerator="auto",
                    precision="16-mixed" if use_amp else "32-true",
                    enable_checkpointing=False,
                    enable_model_summary=False,
                    # enable_progress_bar=False,
                    inference_mode=True,
                    logger=False,
                )

                val_features = inference_trainer.predict(model, ssl_val_loader)
                test_features = inference_trainer.predict(model, ssl_test_loader)

                val_features = torch.cat(val_features, dim=0)
                test_features = torch.cat(test_features, dim=0)

                for train_split_name, train_df in train_data.items():
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
    # train_supervised()
    linear_probe_ssl()
