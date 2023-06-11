from typing import List, Tuple, Union

import torchvision.transforms as T
from lightly.transforms.multi_view_transform import MultiViewTransform
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode

from ssl_wafermap.transforms.augmentations import get_base_transforms
from ssl_wafermap.transforms.utils import NORMALIZE_STATS

# Unlike lightly, I've created a single transform for multi-crop augmentations
# It seems odd to create separate ones for SwAV, DINO, MSN, etc. when they all follow the same logic


class MultiCropViewTransform:
    """Creates a *single* view transform for multi-crop augmentations.

    Parameters
    ----------
    img_size : List[int], optional
        Size of the wafer map after resizing to a square shape, by default [224, 224]
    crop_size : int, optional
        Size of the image after performing RandomResizedCrop, by default 224 (square)
    crop_scale : Tuple[float, float], optional
        Scale parameter bound for RandomResizedCrop, by default (0.4, 1.0)
    die_noise_prob : float, optional
        Probability of adding die noise, by default 0.03
    denoise : bool, optional
        Whether to perform median filter denoising, by default False
    hf_prob : float, optional
        Probability of horizontal flip, by default 0.5
    vf_prob : float, optional
        Probability of vertical flip, by default 0.5
    rr_prob : float, optional
        Probability of random rotation, by default 0.5
    normalize : bool, optional
        Whether to normalize the image, by default True
    """

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        crop_size: int = 224,
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        # Augment >> Crop to Size >> ToTensor + Normalize
        augment = get_base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=False,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=False,
            normalize=False,
        )

        crop = T.RandomResizedCrop(
            size=crop_size,
            scale=crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        transform = [
            augment,
            crop,
            T.ToTensor(),
        ]

        if normalize:
            transform.append(T.Normalize(**NORMALIZE_STATS))

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.transform(image)


class MultiCropTransform(MultiViewTransform):
    """Multi-crop transform for wafermap images.
    Creates a certain number of global and local views using the same augmentation parameters
    aside from the cropping performed after augmentation.

    Parameters
    ----------
    img_size : List[int], optional
        Size of the wafer map after resizing to a square shape, by default [224, 224]
    global_crop_size : int, optional
        Size of the global crop, by default 224 (square)
    global_crop_scale : Tuple[float, float], optional
        Scale parameter bound for global crop's RandomResizedCrop, by default (0.6, 1.0)
    local_crop_size : int, optional
        Size of the local crop, by default 96 (square)
    local_crop_scale : Tuple[float, float], optional
        Scale parameter bound for local crop's RandomResizedCrop, by default (0.1, 0.4)
    n_global_views : int, optional
        Number of global views, by default 2
    n_local_views : int, optional
        Number of local views, by default 6
    die_noise_prob : float, optional
        Probability of adding die noise, by default 0.03
    denoise : bool, optional
        Whether to perform median filter denoising, by default False
    hf_prob : float, optional
        Probability of horizontal flip, by default 0.5
    vf_prob : float, optional
        Probability of vertical flip, by default 0.5
    rr_prob : float, optional
        Probability of random rotation, by default 0.5
    normalize : bool, optional
        Whether to normalize the image, by default True
    """

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.6, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.1, 0.4),
        n_global_views: int = 2,
        n_local_views: int = 6,
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        assert n_global_views > 0, "n_global_views must be greater than 0"
        assert n_local_views > 0, "n_local_views must be greater than 0"

        # Create global and local transforms
        global_transform = MultiCropViewTransform(
            img_size=img_size,
            crop_size=global_crop_size,
            crop_scale=global_crop_scale,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )
        local_transform = MultiCropViewTransform(
            img_size=img_size,
            crop_size=local_crop_size,
            crop_scale=local_crop_scale,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )

        # Create list of all the transforms together
        transforms = [global_transform] * n_global_views
        local_transforms = [local_transform] * n_local_views
        transforms.extend(local_transforms)

        super().__init__(transforms=transforms)
