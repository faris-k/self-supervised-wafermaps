import copy
import os
import warnings

import lightly
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torchvision
from lightly.data import LightlyDataset
from lightly.models import utils
from lightly.models.modules import heads, masked_autoencoder
from lightly.utils import debug, scheduler
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.preprocessing import StandardScaler
from timm.optim.lars import Lars
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

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from lightly.data import LightlyDataset
from lightly.utils import debug, scheduler
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

# from ssl_wafermap.models.ssl import BYOL, DCLW, MAE, MSN, DINOViT, SwaV, VICReg
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
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.05, momentum=0.9)
        return optimizer


class TwoLayerMultilabelClassifier(MultilabelLinearClassifier):
    def __init__(self, num_features, num_classes=8, pos_weight=None):
        super().__init__(
            num_features=num_features, num_classes=num_classes, pos_weight=pos_weight
        )
        self.model = nn.Sequential(
            nn.Linear(self.num_features, 256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, self.num_classes),
        )


max_epochs = 150
input_size = 224

#  Set to True to enable Distributed Data Parallel training.
distributed = False

# Set to True to enable Mixed Precision training.
use_amp = True

# Set to True to enable Synchronized Batch Norm (requires distributed=True).
# If enabled the batch norm is calculated over all gpus, otherwise the batch
# norm is only calculated from samples on the same gpu.
sync_batchnorm = False

# Set to True to gather features from all gpus before calculating
# the loss (requires distributed=True).
# If enabled then the loss on every gpu is calculated with features from all
# gpus, otherwise only features from the same gpu are used.
gather_distributed = False

n_runs = 1  # optional, increase to create multiple runs and report mean + std
batch_size = 64
lr_factor = batch_size / 256  #  scales the learning rate linearly with batch size


class DINOViT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", pretrained=False
        )
        feature_dim = self.backbone.embed_dim

        self.head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=False
        )
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=False
        )

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = lightly.loss.DINOLoss(output_dim=2048)
        self.warmup_epochs = 40 if max_epochs >= 800 else 20

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.head(y)
        self.log("rep_std", debug.std_of_l2_normalized(y))
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.teacher_backbone, m=0.99)
        utils.update_momentum(self.head, self.teacher_head, m=0.99)
        views, _, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss)
        return loss

    # Trying out AdamW; authors recommend using this with ViT
    def configure_optimizers(self):
        param = list(self.backbone.parameters()) + list(self.head.parameters())
        optim = torch.optim.AdamW(
            param,
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class SwaV(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features

        self.projection_head = heads.SwaVProjectionHead(feature_dim, 2048, 128)
        self.prototypes = heads.SwaVPrototypes(128, 3000)  # use 3000 prototypes

        self.criterion = lightly.loss.SwaVLoss(
            sinkhorn_gather_distributed=gather_distributed
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        self.log("rep_std", debug.std_of_l2_normalized(x))
        x = self.projection_head(x)
        x = nn.functional.normalize(x, dim=1, p=2)
        return self.prototypes(x)

    def training_step(self, batch, batch_idx):
        # normalize the prototypes so they are on the unit sphere
        self.prototypes.normalize()

        # the multi-crop dataloader returns a list of image crops where the
        # first two items are the high resolution crops and the rest are low
        # resolution crops
        multi_crops, _, _ = batch
        multi_crop_features = [self.forward(x) for x in multi_crops]

        # split list of crop features into high and low resolution
        high_resolution_features = multi_crop_features[:2]
        low_resolution_features = multi_crop_features[2:]

        # calculate the SwaV loss
        loss = self.criterion(high_resolution_features, low_resolution_features)

        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=1e-3 * lr_factor,
            weight_decay=1e-6,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class MAE(pl.LightningModule):
    def __init__(self):
        super().__init__()

        decoder_dim = 512
        vit = torchvision.models.vit_b_32()

        self.warmup_epochs = 40 if max_epochs >= 800 else 20
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)
        self.decoder = masked_autoencoder.MAEDecoder(
            seq_length=vit.seq_length,
            num_layers=1,
            num_heads=16,
            embed_input_dim=vit.hidden_dim,
            hidden_dim=decoder_dim,
            mlp_dim=decoder_dim * 4,
            out_dim=vit.patch_size**2 * 3,
            dropout=0,
            attention_dropout=0,
        )
        self.criterion = nn.MSELoss()

    def forward_encoder(self, images, idx_keep=None):
        out = self.backbone.encode(images, idx_keep)
        self.log("rep_std", debug.std_of_l2_normalized(out.flatten(1)))
        return out

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
        images, _, _ = batch

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images, idx_keep)
        x_pred = self.forward_decoder(x_encoded, idx_keep, idx_mask)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_pred, target)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class DCLW(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.DCLWLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        self.log("rep_std", debug.std_of_l2_normalized(x))
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(), lr=6e-2 * lr_factor, momentum=0.9, weight_decay=5e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class VICReg(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features
        self.projection_head = heads.BarlowTwinsProjectionHead(feature_dim, 2048, 2048)
        self.criterion = lightly.loss.VICRegLoss()
        self.warmup_epochs = 40 if max_epochs >= 800 else 20

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        self.log("rep_std", debug.std_of_l2_normalized(x))
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _, _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = Lars(
            self.parameters(), lr=0.3 * lr_factor, weight_decay=1e-4, momentum=0.9
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class BYOL(pl.LightningModule):
    def __init__(self):
        super().__init__()
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features

        # create a byol model based on ResNet
        self.projection_head = heads.BYOLProjectionHead(feature_dim, 4096, 256)
        self.prediction_head = heads.BYOLPredictionHead(256, 4096, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        self.criterion = lightly.loss.NegativeCosineSimilarity()

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        self.log("rep_std", debug.std_of_l2_normalized(y))
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.backbone, self.backbone_momentum, m=0.99)
        utils.update_momentum(
            self.projection_head, self.projection_head_momentum, m=0.99
        )
        (x0, x1), _, _ = batch
        p0 = self.forward(x0)
        z0 = self.forward_momentum(x0)
        p1 = self.forward(x1)
        z1 = self.forward_momentum(x1)
        loss = 0.5 * (self.criterion(p0, z1) + self.criterion(p1, z0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = (
            list(self.backbone.parameters())
            + list(self.projection_head.parameters())
            + list(self.prediction_head.parameters())
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


class MSN(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.warmup_epochs = 15
        #  ViT small configuration (ViT-S/16) = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6)
        #  ViT tiny configuration (ViT-T/16) = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3)
        self.mask_ratio = 0.15
        self.backbone = masked_autoencoder.MAEBackbone(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=6,
            hidden_dim=384,
            mlp_dim=384 * 4,
        )
        self.projection_head = heads.MSNProjectionHead(384)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight
        self.criterion = lightly.loss.MSNLoss()

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, _, _ = batch
        views = [view.to(self.device, non_blocking=True) for view in views]
        targets = views[0]
        anchors = views[1]
        anchors_focal = torch.concat(views[2:], dim=0)

        targets_out = self.backbone(targets)
        targets_out = self.projection_head(targets_out)
        anchors_out = self.encode_masked(anchors)
        anchors_focal_out = self.encode_masked(anchors_focal)
        anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

        loss = self.criterion(anchors_out, targets_out, self.prototypes.data)
        self.log("train_loss_ssl", loss)
        self.log(
            "rep_std",
            debug.std_of_l2_normalized(targets_out.flatten(1)),
        )
        return loss

    def encode_masked(self, anchors):
        batch_size, _, _, width = anchors.shape
        seq_length = (width // self.anchor_backbone.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=self.device,
        )
        out = self.anchor_backbone(anchors, idx_keep)
        return self.anchor_projection_head(out)

    def configure_optimizers(self):
        params = [
            *list(self.anchor_backbone.parameters()),
            *list(self.anchor_projection_head.parameters()),
            self.prototypes,
        ]
        optim = torch.optim.AdamW(
            params=params,
            lr=1.5e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.95),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]

    def predict_step(self, batch, batch_idx):
        images, _, _ = batch
        return self.backbone(images)


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
    two_layer_model = TwoLayerMultilabelClassifier(
        num_features=train_features.shape[1], pos_weight=pos_weight
    )

    logger = TensorBoardLogger(
        save_dir=f"../models/mixedwm38_finetune/{model_name}",
        name="",
        sub_dir=f"{train_split_name}/linear",
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

    logger = TensorBoardLogger(
        save_dir=f"../models/mixedwm38_finetune/{model_name}",
        name="",
        sub_dir=f"{train_split_name}/2layer",
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
        two_layer_model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )
    supervised_trainer.test(two_layer_model, test_loader, ckpt_path="best")


ckpt_dict = {
    "SwaV": (
        SwaV,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/SwaV/checkpoints/epoch=149-step=62250.ckpt",
    ),
    "BYOL": (
        BYOL,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/BYOL/checkpoints/epoch=149-step=62250.ckpt",
    ),
    "DCLW": (
        DCLW,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/DCLW/checkpoints/epoch=149-step=62250.ckpt",
    ),
    "DINOViT": (
        DINOViT,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/DINOViT/checkpoints/epoch=74-step=31125.ckpt",
    ),
    "MAE": (
        MAE,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/MAE/checkpoints/epoch=149-step=62250.ckpt",
    ),
    "MSN": (
        MSN,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/MSN/checkpoints/epoch=149-step=62250.ckpt",
    ),
    "VICReg": (
        VICReg,
        "../models/mixed_wm38_pretrain/wafermaps/version_2/VICReg/checkpoints/epoch=149-step=62250.ckpt",
    ),
}


def linear_probe_ssl():
    for model_name, (model_class, ckpt_path) in ckpt_dict.items():
        model = model_class.load_from_checkpoint(ckpt_path)
        print(f"Loaded {model_name} from checkpoint")

        inference_trainer = pl.Trainer(
            accelerator="auto",
            precision="16-mixed" if use_amp else "32-true",
            enable_checkpointing=False,
            enable_model_summary=False,
            # enable_progress_bar=False,
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
            print(f"Training linear probe on {train_split_name}")
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
    linear_probe_ssl()
    train_supervised()
