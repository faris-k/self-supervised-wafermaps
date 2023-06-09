# NOTE: Deprecated in favor of ssl_wafermap.transforms (much cleaner 😉)

import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightly.data.collate import BaseCollateFunction, MultiViewCollateFunction
from lightly.transforms.rotation import RandomRotate
from torchvision.transforms.functional import InterpolationMode


class DieNoise:
    """Adds noise to wafermap die by flipping pass to fail and vice-versa with probability p.
    Inspired by DOI: 10.1109/ITC50571.2021.00019 (https://ieeexplore.ieee.org/document/9611304)

    Parameters
    ----------
    p : float, optional
        Probability of flipping on a die-level basis, by default 0.03
    """

    def __init__(self, p=0.03) -> None:
        self.p = p

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # Create a boolean mask of the 128's and 255's in the matrix
        mask = (sample == 128) | (sample == 255)
        # Create a tensor of random numbers between 0 and 1 with the same shape as the matrix
        rand = torch.rand(*sample.shape)
        # Use the mask and the random numbers to determine which elements to flip
        flip = ((rand < self.p) & mask).type(torch.bool)
        # Flip the elements
        sample[flip] = 383 - sample[flip]
        return sample


# Inspired by Albumentations OneOf
# https://albumentations.ai/docs/api_reference/core/composition/#albumentations.core.composition.OneOf
# https://github.com/albumentations-team/albumentations/blob/master/albumentations/core/composition.py#L302
class RandomOneOf:
    """Randomly applies one of the given transforms with a given probability.

    Parameters
    ----------
    transforms : List[torch.nn.Module]
        List of transforms to apply.
    weights : List[float], optional
        List of weights for each transform. If None, all transforms are
        equally likely to be applied. By default None.
    p : float, optional
        Probability of applying the RandomOneOf block at all, by default 1.0.
    """

    def __init__(
        self,
        transforms,
        weights: List[float] = None,
        p: float = 1.0,
    ):
        if weights:
            if len(transforms) != len(weights):
                raise ValueError(
                    "The number of weights must match the number of transforms"
                )
            if not all([w >= 0 for w in weights]):
                raise ValueError("Weights must be non-negative")
            total_weight = sum(weights)
            if total_weight == 0:
                raise ValueError("At least one weight must be greater than 0")
            self.weights = [w / total_weight for w in weights]
        else:
            self.weights = [1.0 / len(transforms)] * len(transforms)

        if p is not None and (p < 0 or p > 1):
            raise ValueError("p must be a float between 0 and 1")

        self.transforms = transforms
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            idx = random.choices(range(len(self.transforms)), weights=self.weights)[0]
            img = self.transforms[idx](img)

        return img


class MedianFilter:
    """Applies a median filter to denoise a wafermap.
    Simply uses OpenCV's medianBlur function.

    Parameters
    ----------
    kernel_size : int, optional
        Size of the median filter kernel, by default 3
    """

    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size

    def __call__(self, sample: torch.Tensor) -> torch.Tensor:
        # OpenCV expects a numpy array
        sample_np = sample.numpy()
        sample_np = cv2.medianBlur(sample_np, self.kernel_size)
        return torch.from_numpy(sample_np)
        # return median(sample)


class DPWTransform:
    """
    Transforms a wafer map to a lower DPW version. Scale param is first
    generated using a power law distribution, then randomization is applied
    by using it as the lower bound of a beta distribution (skewed towards the lower bound).

    Parameters
    ----------
    domain_lower : int, optional
        Lower bound of shapeMaxDim in the dataset, by default 26
    domain_upper : int, optional
        Upper bound of shapeMaxDim in the dataset, by default 212
    out_lower : float, optional
        Lower bound of scale param, by default 0.4
    out_upper : float, optional
        Upper bound of scale param, by default 0.95
    alpha : float, optional
        Alpha param for beta distribution, by default 0.5
    beta : float, optional
        Beta param for beta distribution, by default 1.0
    p : float, optional
        Power param for power law distribution, by default 5.0
    """

    def __init__(
        self,
        domain_lower: int = 26,
        domain_upper: int = 212,
        out_lower: float = 0.4,
        out_upper: float = 0.95,
        alpha: float = 0.5,
        beta: float = 1.5,
        p: float = 5.0,
    ):
        self.domain_lower = domain_lower
        self.domain_upper = domain_upper
        self.out_lower = out_lower
        self.out_upper = out_upper
        self.alpha = alpha
        self.beta = beta
        self.p = p

    def power_law_transform(
        x,
        domain_lower,
        domain_upper,
        out_lower=0.4,
        out_upper=0.95,
        p=5,
    ):
        # Handle edge cases
        if x <= domain_lower:
            return out_upper
        if x >= domain_upper:
            return out_lower

        # Invert input domain
        domain_range = domain_upper - domain_lower
        inverted_x = abs(x - domain_lower)
        normalized_x = inverted_x / domain_range
        # Apply power law transformation
        y = (1 - normalized_x) ** p
        # Map result to output range
        out_range = out_upper - out_lower
        return out_lower + y * out_range

    def generate_skewed_random(lower_bound, upper_bound=0.95, alpha=0.5, beta=1):
        # Generate random number using beta distribution
        x = np.random.beta(alpha, beta)
        # Scale to range [lower_bound, upper_bound]
        return lower_bound + (upper_bound - lower_bound) * x

    def dpw_transform(wafermap: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Transforms a wafer map to a lower DPW version.
        Works by taking the central coordinates of passing and failing die, then
        mapping those coordinates to the new wafer map.

        Parameters
        ----------
        wafermap : torch.Tensor
            Wafermap to transform
        scale : float, optional
            Scale to transform the wafer map to.
            Must be between 0 and 1.
        """

        assert 0.0 < scale <= 1.0, "Scale must be between 0 and 1."

        # Calculate the new dimensions of the wafer after scaling down
        h, w = wafermap.shape
        new_h = int(h * scale)
        new_w = int(w * scale)

        # Find the indices of the passing elements in the original wafer
        # passing_indices = torch.argwhere(wafermap == 128)
        passing_indices = (wafermap == 128).nonzero()

        # Find the indices of the failing elements in the original wafer
        # failing_indices = torch.argwhere(wafermap == 255)
        failing_indices = (wafermap == 255).nonzero()

        # Calculate the relative central coordinate of the passing and failing elements in the original wafer
        pass_coords = (passing_indices + 0.5) / torch.tensor(wafermap.shape)
        fail_coords = (failing_indices + 0.5) / torch.tensor(wafermap.shape)

        # Calculate the central coordinates of the passing and failing elements in the new wafer map
        new_pass_coords = (pass_coords * torch.tensor([new_h, new_w])).long()
        new_fail_coords = (fail_coords * torch.tensor([new_h, new_w])).long()

        # Create a tensor for storing transformed data with shape (new_h,new_w)
        new_wafer = torch.zeros((new_h, new_w), dtype=torch.uint8)

        # Set values for passing and failing elements in transformed tensor
        new_wafer[new_pass_coords[:, 0], new_pass_coords[:, 1]] = 128
        new_wafer[new_fail_coords[:, 0], new_fail_coords[:, 1]] = 255

        return new_wafer

    def __call__(self, img):
        # Calculate initial scale parameter using power_law_transform
        max_dim = max(img.shape)
        scale_init = DPWTransform.power_law_transform(
            max_dim,
            self.domain_lower,
            self.domain_upper,
            self.out_lower,
            self.out_upper,
            self.p,
        )

        # Generate a skewed random scale parameter
        lower_bound = scale_init
        scale = DPWTransform.generate_skewed_random(
            lower_bound, self.out_upper, alpha=self.alpha, beta=self.beta
        )

        # Apply the DPW transform using the scale parameter
        new_img = DPWTransform.dpw_transform(img, scale=scale)

        return new_img


NORMALIZE_STATS = {
    "mean": [0.4496, 0.4496, 0.4496],
    "std": [0.2926, 0.2926, 0.2926],
}


# def get_base_transforms(
#     img_size: List[int] = [224, 224],
#     die_noise_prob: float = 0.03,
#     rr_prob: float = 0.5,
#     hf_prob: float = 0.5,
#     vf_prob: float = 0.5,
#     to_tensor: bool = True,
#     normalize: bool = True,
# ) -> T.Compose:
#     """Base image transforms for self-supervised training.
#     Applies randomized die noise, converts to PIL Image, resizes, rotates, flips, and optionally converts to tensor.

#     Parameters
#     ----------
#     img_size : List[int], optional
#         Size of image, by default [224, 224]
#     die_noise_prob : float, optional
#         Probability of adding die noise, by default 0.03
#     rr_prob : float, optional
#         Probability of rotating image 90 degrees, by default 0.5
#     hf_prob : float, optional
#         Probability of flipping image horizontally, by default 0.5
#     vf_prob : float, optional
#         Probability of flipping image vertically, by default 0.5
#     to_tensor : bool, optional
#         Whether to convert to tensor, by default True.
#         Use False if you need further augmentations like global/patch cropping.
#     """

#     transforms = [
#         # Randomly perform either DieNoise or DPWTransform with a 50% chance
#         RandomOneOf(
#             [
#                 DieNoise(die_noise_prob),
#                 DPWTransform(),
#             ]
#         ),
#         # Convert to PIL Image, then perform all torchvision transforms except cropping
#         T.ToPILImage(),
#         T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
#         RandomRotate(rr_prob),
#         T.RandomVerticalFlip(vf_prob),
#         T.RandomHorizontalFlip(hf_prob),
#         # Finally, create a 3-channel image and convert to tensor
#         T.Grayscale(num_output_channels=3),  # R == G == B
#     ]

#     # Optionally convert to tensor
#     if to_tensor:
#         transforms.append(T.ToTensor())

#     # Optionally normalize
#     if normalize:
#         transforms.append(T.Normalize(**NORMALIZE_STATS))

#     return T.Compose(transforms)


def get_base_transforms(
    img_size: List[int] = [224, 224],
    die_noise_prob: float = 0.03,
    denoise: bool = False,
    crop: bool = False,
    rr_prob: float = 0.5,
    hf_prob: float = 0.5,
    vf_prob: float = 0.5,
    to_tensor: bool = True,
    normalize: bool = True,
) -> T.Compose:
    """Base image transforms for self-supervised training.
    Applies randomized die noise, converts to PIL Image, resizes, rotates, flips, and optionally converts to tensor.

    Parameters
    ----------
    img_size : List[int], optional
        Size of image, by default [224, 224]
    die_noise_prob : float, optional
        Probability of adding die noise, by default 0.03
    denoise : bool, optional
        Whether to apply denoising via median filtering, by default False.
        If False, DPWTransform is applied instead. Note that denoising may
        remove salient defect patterns, so it should be used with care.
    rr_prob : float, optional
        Probability of rotating image 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability of flipping image horizontally, by default 0.5
    vf_prob : float, optional
        Probability of flipping image vertically, by default 0.5
    to_tensor : bool, optional
        Whether to convert to tensor, by default True.
        Use False if you need further augmentations like global/patch cropping.
    """

    transforms = [
        # Randomly perform either DieNoise or DPWTransform with a 50% chance
        RandomOneOf(
            [
                DieNoise(die_noise_prob),
                MedianFilter() if denoise else DPWTransform(),
            ]
        ),
        # Convert to PIL Image, then perform all torchvision transforms except cropping
        T.ToPILImage(),
        T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
        RandomRotate(rr_prob),
        T.RandomVerticalFlip(vf_prob),
        T.RandomHorizontalFlip(hf_prob),
        # Finally, create a 3-channel image and convert to tensor
        T.Grayscale(num_output_channels=3),  # R == G == B
    ]

    # If cropping, add a random crop transform with a 50% chance
    if crop:
        transforms.append(
            T.RandomApply(
                torch.nn.ModuleList(
                    [
                        T.RandomResizedCrop(
                            size=img_size,
                            scale=(0.4, 1.0),
                            ratio=(1.0, 1.0),
                            interpolation=InterpolationMode.NEAREST,
                        )
                    ]
                ),
                p=0.5,
            )
        )

    # Optionally convert to tensor
    if to_tensor:
        transforms.append(T.ToTensor())

    # Optionally normalize
    if normalize:
        transforms.append(T.Normalize(**NORMALIZE_STATS))

    return T.Compose(transforms)


class WaferImageCollateFunction(BaseCollateFunction):
    """Implements augmentations for self-supervised training on wafermaps.
    Works for "generic" joint-embedding methods like SimCLR, MoCo-v2, BYOL, SimSiam, etc.

    Parameters
    ----------
    img_size : List[int], optional
        Size of augmented images, by default [224, 224]
    die_noise_prob : float, optional
        Probability of applying die noise on a per-die basis, by default 0.03
    hf_prob : float, optional
        Probability of horizontally flipping the image, by default 0.5
    vf_prob : float, optional
        Probability of vertically flipping the image, by default 0.5
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    """

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        die_noise_prob: float = 0.03,
        crop: bool = False,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        transforms = get_base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=crop,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=True,
            normalize=normalize,
        )
        super().__init__(transforms)


class WaferFastSiamCollateFunction(MultiViewCollateFunction):
    """Implements augmentations for FastSiam training on wafermaps.

    Parameters
    ----------
    img_size : List[int], optional
        Size of augmented images, by default [224, 224]
    die_noise_prob : float, optional
        Probability of applying die noise on a per-die basis, by default 0.03
    hf_prob : float, optional
        Probability of horizontally flipping the image, by default 0.5
    vf_prob : float, optional
        Probability of vertically flipping the image, by default 0.5
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    """

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        crop: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        transforms = get_base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=crop,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=True,
            normalize=normalize,
        )
        super().__init__([transforms] * 4)


class WaferDINOCOllateFunction(MultiViewCollateFunction):
    """Custom collate function for DINO training on wafermaps."""

    def __init__(
        self,
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
    ):
        """Implements augmentations for DINO training on wafermaps.

        Parameters
        ----------
        global_crop_size : int, optional
            Size of global crop, by default 224
        global_crop_scale : tuple, optional
            Minimum and maximum size of the global crops relative to global_crop_size,
            by default (0.6, 1.0)
        local_crop_size : int, optional
            Size of local crop, by default 96
        local_crop_scale : tuple, optional
            Minimum and maximum size of the local crops relative to global_crop_size,
            by default (0.1, 0.4)
        n_local_views : int, optional
            Number of generated local views, by default 6
        die_noise_prob : float, optional
            Probability of applying die noise on a per-die basis, by default 0.03
        hf_prob : float, optional
            Probability of horizontally flipping, by default 0.5
        vf_prob : float, optional
            Probability of vertically flipping, by default 0.5
        rr_prob : float, optional
            Probability of rotating by 90 degrees, by default 0.5
        """

        base_transform = get_base_transforms(
            img_size=[global_crop_size, global_crop_size],
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=False,  # we already use multi-crop
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=False,
            normalize=False,
        )

        global_crop = T.RandomResizedCrop(
            global_crop_size,
            scale=global_crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        local_crop = T.RandomResizedCrop(
            local_crop_size,
            scale=local_crop_scale,
            ratio=(1.0, 1.0),
            interpolation=InterpolationMode.NEAREST,
        )

        global_transform = T.Compose(
            [
                base_transform,
                global_crop,
                T.ToTensor(),
                T.Normalize(**NORMALIZE_STATS),
            ]
        )

        local_transform = T.Compose(
            [
                base_transform,
                local_crop,
                T.ToTensor(),
                T.Normalize(**NORMALIZE_STATS),
            ]
        )

        # Create 2 global transforms and n_local_views local transforms
        global_transforms = [global_transform] * 2
        local_transforms = [local_transform] * n_local_views
        transforms = global_transforms + local_transforms

        super().__init__(transforms)


class WaferMSNCollateFunction(MultiViewCollateFunction):
    """
    Implements MSN transformations for wafermaps.
    Modified from https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py#L855

    Parameters
    ----------
    random_size : int, optional
        Size of the global/random image views, by default 224
    focal_size : int, optional
        Size of the focal image views, by default 96
    random_views : int, optional
        Number of global/random views to generate, by default 2
    focal_views : int, optional
        Number of focal views to generate, by default 10
    random_crop_scale : Tuple[float, float], optional
        Minimum and maximum size of the randomized crops relative to random_size,
        by default (0.3, 1.0)
    focal_crop_scale : Tuple[float, float], optional
        Minimum and maximum size of the focal crops relative to focal_size,
        by default (0.05, 0.3)
    die_noise_prob : float, optional
        Probability of adding randomized die noise at a die-level, by default 0.03
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.5
    vf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.0
    """

    def __init__(
        self,
        random_size: int = 224,
        focal_size: int = 96,
        random_views: int = 2,
        focal_views: int = 10,
        random_crop_scale: Tuple[float, float] = (0.6, 1.0),
        focal_crop_scale: Tuple[float, float] = (0.1, 0.4),
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        rr_prob: float = 0.5,
        hf_prob: float = 0.5,
        vf_prob: float = 0.0,
    ) -> None:
        base_transform = get_base_transforms(
            img_size=[random_size, random_size],
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=False,  # we already use multi-crop
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=False,
            normalize=False,
        )

        # Create separate transforms for random and focal views
        random_crop = T.Compose(
            [
                T.RandomResizedCrop(
                    size=random_size,
                    scale=random_crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_STATS),
            ]
        )
        focal_crop = T.Compose(
            [
                T.RandomResizedCrop(
                    size=focal_size,
                    scale=focal_crop_scale,
                    ratio=(1.0, 1.0),
                    interpolation=InterpolationMode.NEAREST,
                ),
                T.ToTensor(),
                T.Normalize(**NORMALIZE_STATS),
            ]
        )

        # Combine base transforms with random and focal crops
        transform = T.Compose([base_transform, random_crop])
        focal_transform = T.Compose([base_transform, focal_crop])

        # Put all transforms together
        transforms = [transform] * random_views
        transforms += [focal_transform] * focal_views
        super().__init__(transforms=transforms)


class WaferMAECollateFunction(MultiViewCollateFunction):
    """Implements the view augmentation for MAE.
    Unlike original paper, no cropping is performed, and we randomly rotate the image by 90 degrees.

    Parameters
    ----------
    img_size : List[int], optional
        Size of the image views, by default [224, 224]
    rr_prob : float, optional
        Probability of rotating the image by 90 degrees, by default 0.5
    hf_prob : float, optional
        Probability that horizontal flip is applied, by default 0.5
    """

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        rr_prob: float = 0.5,
        hf_prob: float = 0.5,
    ):
        transforms = [
            T.ToPILImage(),
            T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
            RandomRotate(rr_prob),
            T.RandomHorizontalFlip(hf_prob),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(**NORMALIZE_STATS),
        ]

        super().__init__([T.Compose(transforms)])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
        return views[0], labels, fnames


class WaferMAECollateFunction2(MultiViewCollateFunction):
    """WaferMAECollateFunction with transforms."""

    def __init__(
        self,
        img_size: List[int] = [224, 224],
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        crop: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        transforms = get_base_transforms(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=crop,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            to_tensor=True,
            normalize=normalize,
        )

        super().__init__([transforms])

    def forward(self, batch: List[tuple]):
        views, labels, fnames = super().forward(batch)
        # Return only first view as MAE needs only a single view per image.
        return views[0], labels, fnames


class WaferMultiCropCollateFunction(MultiViewCollateFunction):
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


class WaferSwaVCollateFunction(WaferMultiCropCollateFunction):
    """Implements the multi-crop transformations for SwaV training on wafer maps.

    Parameters
    ----------
    crop_sizes : List[int], optional
        Size of the input image in pixels for each crop category, by default [224, 96]
    crop_counts : List[int], optional
        Number of crops for each crop category, by default [2, 6]
    crop_min_scales : List[float], optional
        Min scales for each crop category, by default [0.6, 0.1]
    crop_max_scales : List[float], optional
        Max scales for each crop category, by default [1.0, 0.4]
    die_noise_prob : float, optional
        Probability of adding randomized die noise, by default 0.03
    hf_prob : float, optional
        Probability of horizontally flipping, by default 0.5
    vf_prob : float, optional
        Probability of vertically flipping, by default 0.5
    rr_prob : float, optional
        Probability of rotating by 90 degrees, by default 0.5
    """

    def __init__(
        self,
        crop_sizes: List[int] = [224, 96],
        crop_counts: List[int] = [2, 6],
        crop_min_scales: List[float] = [0.6, 0.1],
        crop_max_scales: List[float] = [1.0, 0.4],
        die_noise_prob: float = 0.03,
        denoise: bool = False,
        hf_prob: float = 0.5,
        vf_prob: float = 0.5,
        rr_prob: float = 0.5,
        normalize: bool = True,
    ):
        transforms = get_base_transforms(
            img_size=[crop_sizes[0], crop_sizes[0]],
            die_noise_prob=die_noise_prob,
            denoise=denoise,
            crop=False,  # we already use multi-crop
            rr_prob=rr_prob,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            normalize=normalize,
        )

        super(WaferSwaVCollateFunction, self).__init__(
            crop_sizes=crop_sizes,
            crop_counts=crop_counts,
            crop_min_scales=crop_min_scales,
            crop_max_scales=crop_max_scales,
            transforms=transforms,
        )


def get_inference_transforms(img_size: List[int] = [224, 224], normalize: bool = True):
    """Image transforms for inference.
    Simply converts to PIL Image, resizes, and converts to tensor.

    Parameters
    ----------
    img_size : List[int], optional
        Size of image, by default [224, 224]
    """
    transforms = [
        # Convert to PIL Image, then perform all torchvision transforms
        T.ToPILImage(),
        T.Resize(img_size, interpolation=InterpolationMode.NEAREST),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
    ]

    if normalize:
        transforms.append(T.Normalize(**NORMALIZE_STATS))

    return T.Compose(transforms)


# Inspired by https://github.com/sparks-baird/xtal2png/blob/main/src/xtal2png/utils/data.py#L138
def rgb_scale(
    X,
    feature_range=[0, 255],
    data_range=None,
):
    """Scales array to RGB domain [0, 255]"""
    import numpy as np

    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if data_range is None:
        data_range = [np.min(X), np.max(X)]
    if feature_range is None:
        feature_range = [np.min(X), np.max(X)]

    data_min, data_max = data_range
    feature_min, feature_max = feature_range
    X_std = (X - data_min) / (data_max - data_min)
    X_scaled = X_std * (feature_max - feature_min) + feature_min
    X_scaled = np.round(X_scaled).astype(np.uint8)
    return X_scaled
