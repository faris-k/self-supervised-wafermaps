import os
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from ssl_wafermap.data import WaferMapDataset
from ssl_wafermap.transforms import get_base_transforms, get_inference_transforms

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pl.seed_everything(0)

torch.set_float32_matmul_precision("high")

# suppress annoying torchmetrics and lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")
warnings.filterwarnings("ignore", ".*logging interval.*")
warnings.filterwarnings("ignore", ".*confusion.*")

num_workers = 2
batch_size = 64
use_amp = True
max_epochs = 100

data_path = "../data/processed/WM811K"
train_data = pd.read_pickle(os.path.join(data_path, "train_data.pkl.xz"))
val_data = pd.read_pickle(os.path.join(data_path, "val_data.pkl.xz"))
test_data = pd.read_pickle(os.path.join(data_path, "test_data.pkl.xz"))

train_dataset = WaferMapDataset(
    X=train_data.waferMap,
    y=train_data.failureCode,
    transform=get_inference_transforms(),
)
val_dataset = WaferMapDataset(
    X=val_data.waferMap, y=val_data.failureCode, transform=get_base_transforms()
)
test_dataset = WaferMapDataset(
    X=test_data.waferMap, y=test_data.failureCode, transform=get_inference_transforms()
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


class SupervisedR18(pl.LightningModule):
    def __init__(self, num_classes=9, weight=None):
        super().__init__()

        backbone = timm.create_model("resnet18", pretrained=False, num_classes=0)
        self.backbone = backbone
        self.classifier = nn.Linear(backbone.num_features, num_classes)

        self.criterion = nn.CrossEntropyLoss(weight=weight)

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.train_f1 = MulticlassF1Score(num_classes=num_classes)

        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes)

        self.test_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes)

    def forward(self, x):
        z = self.backbone(x)
        y_hat = self.classifier(z)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch
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
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

        self.val_acc(y_hat, y)
        self.val_f1(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        images, _ = batch
        return self.backbone(images)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


weight = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_data.failureCode.values),
    y=train_data.failureCode.values,
)
weight = torch.tensor(weight, dtype=torch.float32)
model = SupervisedR18(weight=weight)
logger = TensorBoardLogger(save_dir="../models/wm811k_evals/supervised", name="",)
checkpoint_callback = ModelCheckpoint(
    dirpath=os.path.join(logger.log_dir, "checkpoints"),
)

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=max_epochs,
    precision="16-mixed" if use_amp else "32-true",
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        checkpoint_callback,
    ],
    logger=logger,
)


def main():
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.validate(model, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
