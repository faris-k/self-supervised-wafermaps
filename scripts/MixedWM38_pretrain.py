# Adapted from https://github.com/lightly-ai/lightly/blob/master/docs/source/getting_started/benchmarks/imagenette_benchmark.py
import copy
import os
import time
import warnings

import lightly
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
from pytorch_lightning.loggers import TensorBoardLogger
from timm.optim.lars import Lars
from torch.utils.data import DataLoader

from ssl_wafermap.utilities.data import WaferMapDataset
from ssl_wafermap.utilities.transforms import (
    WaferDINOCOllateFunction,
    WaferImageCollateFunction,
    WaferMAECollateFunction2,
    WaferMSNCollateFunction,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

torch.set_float32_matmul_precision("high")

# suppress annoying torchmetrics and lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")
warnings.filterwarnings("ignore", ".*meaningless.*")
warnings.filterwarnings("ignore", ".*confusion.*")

logs_root_dir = "../models/mixed_wm38_pretrain"

num_workers = 2  # os.cpu_count()

subset = False  # Whether to train using a subset of the dataset
max_epochs = 5 if subset else 150
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

if subset:
    # Use a small subset of the training data for debugging
    df = pd.read_pickle("../data/processed/MixedWM38/train_5_split.pkl.xz")
else:
    # Otherwise, use the full training data
    df = pd.read_pickle("../data/processed/MixedWM38/train_data.pkl.xz")

# SSL training will have no transforms passed to the dataset object; this is handled by collate function
# Note that the labels here aren't actually used at all, we just need to pass in something
dataset_train_ssl = LightlyDataset.from_torch_dataset(
    WaferMapDataset(df.waferMap, df.failureType.factorize(sort=True)[0])
)

# Base collate function for basic joint embedding frameworks
# e.g. SimCLR, MoCo, BYOL, Barlow Twins, DCLW, SimSiam
collate_fn = WaferImageCollateFunction(
    denoise=True,
)

# DINO, MAE, and MSN will use their own collate functions
dino_collate_fn = WaferDINOCOllateFunction(
    global_crop_size=input_size, local_crop_size=input_size // 2, denoise=True
)
mae2_collate_fn = WaferMAECollateFunction2(denoise=True)
msn_collate_fn = WaferMSNCollateFunction(
    random_size=input_size, focal_size=input_size // 2, denoise=True
)


def get_data_loader(batch_size: int, model):
    """Helper method to create dataloaders for SSL, kNN train and kNN test

    Args:
        batch_size: Desired batch size for all dataloaders
    """
    # By default, use the base collate function
    col_fn = collate_fn
    # if the model is any of the DINO models, we use the DINO collate function
    if model == DINOViT:
        col_fn = dino_collate_fn
    elif model == MAE:
        col_fn = mae2_collate_fn
    elif model == MSN:
        col_fn = msn_collate_fn

    dataloader_train_ssl = DataLoader(
        dataset_train_ssl,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=col_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return dataloader_train_ssl


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


def main():
    models = [
        # Mask Denoising
        MSN,
        # Contrastive Learning
        DCLW,
        # Redundancy Reduction
        VICReg,
        # Masked Image Modeling
        MAE,
        # Distillation
        BYOL,
        DINOViT,
    ]
    results = dict()

    experiment_version = None
    # loop through configurations and train models
    for model_class in models:
        runs = []
        model_name = model_class.__name__.replace("", "")
        for seed in range(n_runs):
            pl.seed_everything(seed)
            dataloader_train_ssl = get_data_loader(
                batch_size=batch_size,
                model=model_class,
            )
            model = model_class()

            # Save logs to: {CWD}/benchmark_logs/wafermaps/{experiment_version}/{model_name}/
            # If multiple runs are specified a subdirectory for each run is created.
            sub_dir = model_name
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
                every_n_epochs=max_epochs // 5,
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
                benchmark=True,
            )
            start = time.time()
            trainer.fit(
                model,
                train_dataloaders=dataloader_train_ssl,
            )
            end = time.time()
            run = {
                "model": model_name,
                "batch_size": dataloader_train_ssl.batch_size,  # batch_size of the dataloader, not the global batch size
                "epochs": trainer.current_epoch,
                "params": sum(p.numel() for p in model.parameters() if p.requires_grad)
                / 1_000_000,
                "runtime": end - start,
                "gpu_memory_usage": torch.cuda.max_memory_allocated() / (1024**3),
                "seed": seed,
            }
            runs.append(run)
            print(run)

            # Save the results dictionary to file
            pd.DataFrame(runs).to_csv(
                os.path.join(logger.log_dir, "results.csv"), index=False
            )

            # delete model and trainer + free up cuda memory
            del model, trainer
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        results[model_name] = runs


# To get num_workers > 0 for DataLoaders on Windows, do the following:
# Use a __main__ guard to prevent spawning of multiple processes
# And set pin_memory=True and persistent_workers=True
if __name__ == "__main__":
    # multiprocessing.set_start_method("spawn", True)
    # multiprocessing.freeze_support()
    main()
