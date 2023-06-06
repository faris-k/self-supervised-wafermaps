from typing import List

from lightly.transforms.multi_view_transform import MultiViewTransform

from ssl_wafermap.transforms.augmentations import get_base_transforms


class BaseTransform(MultiViewTransform):
    """Base transform for wafermap images.

    Parameters
    ----------
    img_size : List[int], optional
        Size of the wafer map after resizing to a square shape, by default [224, 224]
    die_noise_prob : float, optional
        Probability of adding die noise, by default 0.03
    crop : bool, optional
        Whether to perform randomized cropping, by default False
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
    n_views : int, optional
        Number of views to generate, by default 2
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
        n_views: int = 2,
    ):
        view_transform = get_base_transforms(
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
        transforms = [view_transform] * n_views

        super().__init__(transforms=transforms)
