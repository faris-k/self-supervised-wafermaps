# Adapted from https://github.com/lightly-ai/lightly/blob/master/docs/source/getting_started/benchmarks/imagenette_benchmark.py
import copy
import os
import time
import warnings

import lightly
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from lightly.models import utils
from lightly.models.modules import heads, masked_autoencoder
from lightly.utils import debug, scheduler
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from timm.optim.lars import Lars
from torch.utils.data import DataLoader

from ssl_wafermap.data import WaferMapDataset
from ssl_wafermap.models.knn import KNNBenchmarkModule
from ssl_wafermap.transforms import (
    BaseViewTransform,
    MultiCropTransform,
    get_base_transforms,
    get_inference_transforms,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.set_float32_matmul_precision("high")

# suppress annoying torchmetrics and lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")
warnings.filterwarnings("ignore", ".*meaningless.*")
warnings.filterwarnings("ignore", ".*confusion.*")

logs_root_dir = "../models/benchmark_logs"

num_workers = 2  # os.cpu_count()
memory_bank_size = 4096

dummy = True  # Whether to perform a test run using a subset of the dataset
max_epochs = 2 if dummy else 150
knn_k = 5  # sweep of knn_k values leads to best performance at k=5
knn_t = 0.1
classes = 9

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

# Use a GPU if available
devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

if distributed:
    strategy = "ddp"
    # reduce batch size for distributed training
    batch_size = batch_size // devices
else:
    strategy = "auto"
    # limit to single gpu if not using distributed training
    devices = min(devices, 1)

if dummy:
    # Create a smaller dataset for benchmarking using one of the training splits
    df = pd.read_pickle("../data/processed/WM811K/train_20_split.pkl.xz")
    X_train, X_val, y_train, y_val = train_test_split(
        df.waferMap,
        df.failureCode,
        test_size=0.2,
        random_state=42,
        stratify=df.failureCode,
    )
    del df
else:
    # Otherwise, use the larger train/val splits
    df_train = pd.read_pickle("../data/processed/WM811K/train_data.pkl.xz")
    df_val = pd.read_pickle("../data/processed/WM811K/val_data.pkl.xz")
    X_train, y_train = df_train.waferMap, df_train.failureCode
    X_val, y_val = df_val.waferMap, df_val.failureCode
    del df_train, df_val


# Use inference transforms to get kNN feature bank (dataset_train_kNN) and then evaluate (dataset_test)
dataset_train_kNN = WaferMapDataset(X_train, y_train, get_inference_transforms())
dataset_test = WaferMapDataset(X_val, y_val, get_inference_transforms())

# Base transform for basic joint embedding frameworks
# FastSiam and MAE use a variant of this with different n_views
base_transform = BaseViewTransform()
fastsiam_transform = BaseViewTransform(n_views=4)
mae_transform = BaseViewTransform(n_views=1)

# DINO, SwAV, MSN, and PMSN use multi-crop transforms
multicrop_transform = MultiCropTransform()


def create_dataset_train_ssl(model):
    """Helper method to create the SSL training dataset for a given model

    Parameters
    ----------
    model : pl.LightningModule
        The model for which to select the correct transform

    Returns
    -------
    WaferMapDataset
        The SSL training dataset for the given model
    """

    model_to_transform = {
        BarlowTwins: base_transform,
        BYOL: base_transform,
        DCLW: base_transform,
        DINO: multicrop_transform,
        DINOViT: multicrop_transform,
        FastSiam: fastsiam_transform,
        MAE: mae_transform,
        MoCo: base_transform,
        MSN: multicrop_transform,
        PMSN: multicrop_transform,
        SimCLR: base_transform,
        SimMIM: mae_transform,
        SimSiam: base_transform,
        SwaV: multicrop_transform,
        VICReg: base_transform,
        # Supervised model will not use a multi-view transform; just use the single transform composition
        SupervisedR18: get_base_transforms(),
    }
    transform = model_to_transform[model]
    return WaferMapDataset(X_train, y_train, transform=transform)


def get_data_loaders(batch_size: int, dataset_train_ssl):
    """Helper method to create dataloaders for SSL, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """

    dataloader_train_ssl = DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    dataloader_train_kNN = DataLoader(
        dataset_train_kNN,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    dataloader_test = DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return dataloader_train_ssl, dataloader_train_kNN, dataloader_test


# All benchmarking models are defined here


# Supervised Learning
class SupervisedR18(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        self.fc = timm.create_model("resnet18", num_classes=9).get_classifier()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        p = self.fc(f)
        self.log("rep_std", debug.std_of_l2_normalized(f))
        return F.log_softmax(p, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters())
        return optim


# Contrastive Learning
class SimCLR(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features
        self.projection_head = heads.SimCLRProjectionHead(feature_dim, feature_dim, 128)
        self.criterion = lightly.loss.NTXentLoss()

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        self.log("rep_std", debug.std_of_l2_normalized(x))
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _ = batch
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


class DCLW(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
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
        (x0, x1), _ = batch
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


class MoCo(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features

        # create a moco model based on ResNet
        self.projection_head = heads.MoCoProjectionHead(feature_dim, 2048, 128)
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_momentum)

        # create our loss with the optional memory bank
        self.criterion = lightly.loss.NTXentLoss(
            temperature=0.1, memory_bank_size=memory_bank_size
        )

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return self.projection_head(x)

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch

        # update momentum
        utils.update_momentum(self.backbone, self.backbone_momentum, 0.99)
        utils.update_momentum(self.projection_head, self.projection_head_momentum, 0.99)

        def step(x0_, x1_):
            x1_, shuffle = utils.batch_shuffle(x1_, distributed=distributed)
            x0_ = self.backbone(x0_).flatten(start_dim=1)
            self.log("rep_std", debug.std_of_l2_normalized(x0_))
            x0_ = self.projection_head(x0_)

            x1_ = self.backbone_momentum(x1_).flatten(start_dim=1)
            x1_ = self.projection_head_momentum(x1_)
            x1_ = utils.batch_unshuffle(x1_, shuffle, distributed=distributed)
            return x0_, x1_

        # We use a symmetric loss (model trains faster at little compute overhead)
        # https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
        loss_1 = self.criterion(*step(x0, x1))
        loss_2 = self.criterion(*step(x1, x0))

        loss = 0.5 * (loss_1 + loss_2)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        params = list(self.backbone.parameters()) + list(
            self.projection_head.parameters()
        )
        optim = torch.optim.SGD(
            params,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


# Redundancy Reduction


class BarlowTwins(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features
        self.projection_head = heads.BarlowTwinsProjectionHead(feature_dim, 2048, 2048)
        self.criterion = lightly.loss.BarlowTwinsLoss(
            gather_distributed=gather_distributed
        )
        self.warmup_epochs = 40 if max_epochs >= 800 else 20

    def forward(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        self.log("rep_std", debug.std_of_l2_normalized(x))
        return z

    def training_step(self, batch, batch_index):
        (x0, x1), _ = batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log("train_loss_ssl", loss)
        return loss

    # Switch from SGD to LARS since SGD diverges; use Lightly's imagenet100 settings
    def configure_optimizers(self):
        optim = Lars(
            self.parameters(), lr=0.2 * lr_factor, weight_decay=1.5e-6, momentum=0.9
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]


class VICReg(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
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
        (x0, x1), _ = batch
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


# Self-Distillation
class BYOL(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
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
        (x0, x1), _ = batch
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


class DINO(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features

        self.head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )

        utils.deactivate_requires_grad(self.teacher_backbone)
        utils.deactivate_requires_grad(self.teacher_head)

        self.criterion = lightly.loss.DINOLoss(output_dim=2048)

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
        views, _ = batch
        views = [view.to(self.device) for view in views]
        global_views = views[:2]
        teacher_out = [self.forward_teacher(view) for view in global_views]
        student_out = [self.forward(view) for view in views]
        loss = self.criterion(teacher_out, student_out, epoch=self.current_epoch)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        param = list(self.backbone.parameters()) + list(self.head.parameters())
        optim = torch.optim.SGD(
            param,
            lr=6e-2 * lr_factor,
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class DINOViT(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        self.backbone = torch.hub.load(
            "facebookresearch/dino:main", "dino_vits16", pretrained=False
        )
        feature_dim = self.backbone.embed_dim

        self.head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
        )
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_head = heads.DINOProjectionHead(
            feature_dim, 2048, 256, 2048, batch_norm=True
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
        views, _ = batch
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


class SimSiam(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
        # create a ResNet backbone and remove the classification head
        self.backbone = timm.create_model("resnet18", num_classes=0, pretrained=False)
        feature_dim = self.backbone.num_features
        self.projection_head = heads.SimSiamProjectionHead(feature_dim, 2048, 2048)
        self.prediction_head = heads.SimSiamPredictionHead(2048, 512, 2048)
        self.criterion = lightly.loss.NegativeCosineSimilarity()

    def forward(self, x):
        f = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(f)
        p = self.prediction_head(z)
        z = z.detach()
        self.log("rep_std", debug.std_of_l2_normalized(f))
        return z, p

    def training_step(self, batch, batch_idx):
        (x0, x1), _ = batch
        z0, p0 = self.forward(x0)
        z1, p1 = self.forward(x1)
        loss = 0.5 * (self.criterion(z0, p1) + self.criterion(z1, p0))
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=6e-2,  #  no lr-scaling, results in better training stability
            momentum=0.9,
            weight_decay=5e-4,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, max_epochs)
        return [optim], [scheduler]


class FastSiam(SimSiam):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

    # Only the training_step is different
    def training_step(self, batch, batch_idx):
        views, _ = batch
        features = [self.forward(view) for view in views]
        zs = torch.stack([z for z, _ in features])
        ps = torch.stack([p for _, p in features])

        loss = 0.0
        for i in range(len(views)):
            mask = torch.arange(len(views), device=self.device) != i
            loss += self.criterion(ps[i], torch.mean(zs[mask], dim=0)) / len(views)

        self.log("train_loss_ssl", loss)
        return loss


# Mask Denoising
class MSN(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

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

        views, _ = batch
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


class PMSN(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

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
        self.criterion = lightly.loss.PMSNLoss(gather_distributed=gather_distributed)

    def training_step(self, batch, batch_idx):
        utils.update_momentum(self.anchor_backbone, self.backbone, 0.996)
        utils.update_momentum(self.anchor_projection_head, self.projection_head, 0.996)

        views, _ = batch
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


# Clustering
class SwaV(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)
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
        multi_crops, _ = batch
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


# Masked Image Modeling
class MAE(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

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
        images, _ = batch
        # BaseTransform returns a list of transformed images; in this case, we only have one view
        images = images[0]

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


class SimMIM(KNNBenchmarkModule):
    def __init__(self, dataloader_kNN=None, num_classes=9, **kwargs):
        super().__init__(dataloader_kNN, num_classes, **kwargs)

        vit = torchvision.models.vit_b_32()
        self.warmup_epochs = 40 if max_epochs >= 800 else 20
        decoder_dim = vit.hidden_dim
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))

        # same backbone as MAE
        self.backbone = masked_autoencoder.MAEBackbone.from_vit(vit)

        # the decoder is a simple linear layer
        self.decoder = nn.Linear(vit.hidden_dim, vit.patch_size**2 * 3)

        # L1 loss as paper suggestion
        self.criterion = nn.L1Loss()

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        tokens = self.backbone.images_to_tokens(images, prepend_class_token=True)
        tokens_masked = utils.mask_at_index(tokens, idx_mask, self.mask_token)
        # self.log("rep_std", debug.std_of_l2_normalized(out.flatten(1)))
        return self.backbone.encoder(tokens_masked)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)

    def training_step(self, batch, batch_idx):
        images, _ = batch
        # BaseTransform returns a list of transformed images; in this case, we only have one view
        images = images[0]

        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        loss = self.criterion(x_out, target)
        self.log("train_loss_ssl", loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=8e-4 * lr_factor,
            weight_decay=0.05,
            betas=(0.9, 0.999),
        )
        cosine_scheduler = scheduler.CosineWarmupScheduler(
            optim, self.warmup_epochs, max_epochs
        )
        return [optim], [cosine_scheduler]


def main():
    models = [
        # Contrastive Learning
        SimCLR,
        MoCo,
        DCLW,
        # Clustering
        SwaV,
        # Self-Distillation
        BYOL,
        SimSiam,
        FastSiam,
        DINO,
        DINOViT,
        # Redundancy Reduction
        VICReg,
        BarlowTwins,
        # Mask Denoising
        MSN,
        PMSN,
        # Masked Image Modeling
        MAE,
        SimMIM,
        # Supervised
        SupervisedR18,
    ]
    bench_results = dict()

    experiment_version = None
    # loop through configurations and train models
    for Benchmark in models:
        runs = []
        model_name = Benchmark.__name__.replace("", "")
        for seed in range(n_runs):
            pl.seed_everything(seed)
            dataset_train_ssl = create_dataset_train_ssl(Benchmark)
            (
                dataloader_train_ssl,
                dataloader_train_kNN,
                dataloader_test,
            ) = get_data_loaders(
                batch_size=batch_size, dataset_train_ssl=dataset_train_ssl
            )
            benchmark_model = Benchmark(
                dataloader_train_kNN, classes, knn_k=knn_k, knn_t=knn_t
            )

            # Save logs to: {CWD}/benchmark_logs/wafermaps/{experiment_version}/{model_name}/
            # If multiple runs are specified a subdirectory for each run is created.
            sub_dir = model_name if n_runs <= 1 else f"{model_name}/run{seed}"
            logger = TensorBoardLogger(
                save_dir=os.path.join(logs_root_dir, "wafermaps"),
                name="",
                sub_dir=sub_dir,
                version=experiment_version,
            )
            if experiment_version is None:
                # Save results of all models under same version directory
                experiment_version = logger.version
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(logger.log_dir, "checkpoints"),
                every_n_epochs=max_epochs // 10,
            )

            trainer = pl.Trainer(
                max_epochs=max_epochs,
                devices=devices,
                accelerator=accelerator,
                default_root_dir=logs_root_dir,
                strategy=strategy,
                sync_batchnorm=sync_batchnorm,
                logger=logger,
                callbacks=[checkpoint_callback],
                enable_progress_bar=True,
                precision="16-mixed" if use_amp else "32-true",
            )
            start = time.time()
            trainer.fit(
                benchmark_model,
                train_dataloaders=dataloader_train_ssl,
                val_dataloaders=dataloader_test,
            )
            end = time.time()
            run = {
                "model": model_name,
                "batch_size": dataloader_train_ssl.batch_size,
                "epochs": trainer.current_epoch,
                "params": sum(
                    p.numel() for p in benchmark_model.parameters() if p.requires_grad
                )
                / 1_000_000,
                "max_accuracy": benchmark_model.max_accuracy,
                "max_f1": benchmark_model.max_f1,
                "runtime": end - start,
                "gpu_memory_usage": torch.cuda.max_memory_allocated() / (1024**3),
                "seed": seed,
            }
            runs.append(run)
            print(run)

            # Save confusion matrix to compressed npz file
            stacked_cm = np.stack(benchmark_model.confusion_matrix)
            np.savez_compressed(
                os.path.join(logger.log_dir, "confusion_matrix.npz"),
                confusion_matrix=stacked_cm,
            )

            # Save the results dictionary to file
            pd.DataFrame(runs).to_csv(
                os.path.join(logger.log_dir, "results.csv"), index=False
            )

            # delete model and trainer + free up cuda memory
            del benchmark_model, trainer, stacked_cm  # , stacked_history
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        bench_results[model_name] = runs

    #  print results table
    header = (
        f"| {'':<13} | {'Batch Size':>10} | {'Epochs':>6} | {'#param.':>9} "
        f"| {'KNN Test Accuracy':>18} | {'KNN Test F1':>16} | {'Time':>10} | {'Peak GPU Usage':>14} |"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for model, results in bench_results.items():
        runtime = np.array([result["runtime"] for result in results])
        runtime = runtime.mean() / 60  # convert to min
        accuracy = np.array([result["max_accuracy"] for result in results])
        f1 = np.array([result["max_f1"] for result in results])
        gpu_memory_usage = np.array([result["gpu_memory_usage"] for result in results])
        gpu_memory_usage = gpu_memory_usage.max()
        epochs = np.array([result["epochs"] for result in results])
        epochs = int(epochs.mean())
        params = results[0]["params"]

        if len(accuracy) > 1:
            accuracy_msg = f"{accuracy.mean():>8.3f} +- {accuracy.std():>4.3f}"
        else:
            accuracy_msg = f"{accuracy.mean():>18.3f}"
        if len(f1) > 1:
            f1_msg = f"{f1.mean():>8.3f} +- {f1.std():>4.3f}"
        else:
            f1_msg = f"{f1.mean():>16.3f}"

        print(
            f"| {model:<13} | {batch_size:>10} | {epochs:>6} | {params:>8.1f}M "
            f"| {accuracy_msg} | {f1_msg} | {runtime:>6.1f} Min "
            f"| {gpu_memory_usage:>8.1f} GByte |",
            flush=True,
        )
    print("-" * len(header))


# To get num_workers > 0 for DataLoaders on Windows, do the following:
# Use a __main__ guard to prevent spawning of multiple processes
# And set pin_memory=True and persistent_workers=True
if __name__ == "__main__":
    main()
