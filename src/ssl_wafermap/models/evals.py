import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch import nn
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelF1Score,
)


class LinearClassifier(pl.LightningModule):
    def __init__(self, num_features, num_classes=9, weight=None):
        super().__init__()

        self.model = nn.Linear(num_features, num_classes)

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        self.test_acc(y_hat, y)
        self.test_f1(y_hat, y)

        self.log("test_acc", self.test_acc, on_epoch=True, prog_bar=True)
        self.log("test_f1", self.test_f1, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return optimizer


class MultilabelLinearClassifier(pl.LightningModule):
    def __init__(self, num_features, num_classes=8, pos_weight=None):
        super().__init__()

        self.model = nn.Linear(num_features, num_classes)

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
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return optimizer


class TwoLayerMultilabelClassifier(MultilabelLinearClassifier):
    def __init__(self, num_features, num_classes=8, pos_weight=None):
        super().__init__(
            num_features=num_features, num_classes=num_classes, pos_weight=pos_weight
        )
        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.Mish(),
            nn.Dropout(p=0.5),
            nn.Linear(256, num_classes),
        )
