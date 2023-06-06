import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightly.transforms.multi_view_transform import MultiViewTransform
from torchvision.transforms.functional import InterpolationMode


# This needs to be modified to use correct interpolation method
class WaferMultiCropTranform(MultiViewTransform):
    """Implements the multi-crop transformations for SwaV.
    For wafer maps, the ratio of crops is fixed at 1.0 to keep images square.

    Attributes:
        crop_sizes:
            Size of the input image in pixels for each crop category.
        crop_counts:
            Number of crops for each crop category.
        crop_min_scales:
            Min scales for each crop category.
        crop_max_scales:
            Max_scales for each crop category.
        transforms:
            Transforms which are applied to all crops.
    """

    def __init__(
        self,
        crop_sizes: List[int],
        crop_counts: List[int],
        crop_min_scales: List[float],
        crop_max_scales: List[float],
        transforms: T.Compose,
    ):
        if len(crop_sizes) != len(crop_counts):
            raise ValueError(
                "Length of crop_sizes and crop_counts must be equal but are"
                f" {len(crop_sizes)} and {len(crop_counts)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_min_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )
        if len(crop_sizes) != len(crop_min_scales):
            raise ValueError(
                "Length of crop_sizes and crop_max_scales must be equal but are"
                f" {len(crop_sizes)} and {len(crop_min_scales)}."
            )

        crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = T.RandomResizedCrop(
                crop_sizes[i],
                scale=(crop_min_scales[i], crop_max_scales[i]),
                ratio=(1.0, 1.0),
                interpolation=InterpolationMode.NEAREST,
            )

            crop_transforms.extend(
                [
                    T.Compose(
                        [
                            transforms,
                            random_resized_crop,
                            # T.ToTensor(),
                        ]
                    )
                ]
                * crop_counts[i]
            )
        super().__init__(crop_transforms)
