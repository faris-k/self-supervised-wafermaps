import os
import sys
import warnings

import pandas as pd
import pytorch_lightning as pl
import torch
from lightly.data import LightlyDataset
from torch.utils.data import DataLoader

from ssl_wafermap.utilities.data import WaferMapDataset
from ssl_wafermap.utilities.transforms import get_inference_transforms

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

pl.seed_everything(0)

torch.set_float32_matmul_precision("high")

# suppress annoying torchmetrics and lightning warnings
warnings.filterwarnings("ignore", ".*many workers.*")


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
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    # persistent_workers=True,
)
dataloader_test_kNN = DataLoader(
    dataset_test_kNN,
    batch_size=64,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    # persistent_workers=True,
)

ckpt_dir = "../models/new_knn/"
ckpt_file_end = "checkpoints/epoch=149-step=87450.ckpt"


def main():
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

        trainer = pl.Trainer(
            accelerator="gpu",
            logger=False,
            inference_mode=True,
            precision="16-mixed",
            # enable_progress_bar=False,
        )
        trainer.validate(model, dataloader_test_kNN)


if __name__ == "__main__":
    main()
