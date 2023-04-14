import os
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)

from ssl_wafermap.utilities.data import WaferMapDataset
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

num_workers = 0
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


if __name__ == "__main__":
    train_supervised()
