import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from lightly.transforms.rotation import (
    RandomRotate,  # FIXME: from lightly.transforms.rotation import random_rotation_transform
)
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
        self, transforms, weights: List[float] = None, p: float = 1.0,
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
        x, domain_lower, domain_upper, out_lower=0.4, out_upper=0.95, p=5,
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
    as_list: bool = False,
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
            [DieNoise(die_noise_prob), MedianFilter() if denoise else DPWTransform(),]
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

    if as_list:
        return transforms
    else:
        T.Compose(transforms)
