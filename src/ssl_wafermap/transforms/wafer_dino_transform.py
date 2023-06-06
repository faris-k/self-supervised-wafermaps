from typing import List, Optional, Tuple, Union

import PIL
import torchvision.transforms as T
from lightly.transforms.multi_view_transform import MultiViewTransform
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode

from ssl_wafermap.transforms.augmentations import get_base_transforms
from ssl_wafermap.transforms.utils import NORMALIZE_STATS

# This can actually be used for MSN as well, just change n_local_views to 10 and optionally global_crop_scale
# Maybe global_crop_scale should be increased to (0.7, 1.0) for DINO, (1.0, 1.0) for MSN since we already drop patches?


class DINOViewTransform:
    def __init__(
        self,
        img_size: List[int] = [224, 224],
        crop_size: List[int] = [224, 224],
        crop_scale: Tuple[float, float] = (0.4, 1.0),
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):

        # Augment >> Crop to Size >> ToTensor + Normalize
        augmentation = get_base_transforms(
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

        cropping = T.RandomResizedCrop(
            size=crop_size,
            scale=crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        transform = [
            augmentation,
            cropping,
            T.ToTensor(),
        ]

        if normalize:
            transform.extend(T.Normalize(**NORMALIZE_STATS))

        self.transform = T.Compose(transform)

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.transform(image)


class DINOTransform(MultiViewTransform):
    def __init__(
        self,
        img_size: List[int] = [224, 224],
        global_crop_size: int = 224,
        global_crop_scale: Tuple[float, float] = (0.6, 1.0),
        local_crop_size: int = 96,
        local_crop_scale: Tuple[float, float] = (0.1, 0.4),
        n_local_views: int = 6,
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):

        global_transform = DINOViewTransform(
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
        local_transform = DINOViewTransform(
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
        transforms = [global_transform] * 2
        local_transforms = [local_transform] * n_local_views
        transforms.extend(local_transforms)
        super().__init__(transforms=transforms)
