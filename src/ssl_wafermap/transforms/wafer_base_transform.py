from typing import List, Union

from lightly.transforms.multi_view_transform import MultiViewTransform
from PIL.Image import Image
from torch import Tensor

from ssl_wafermap.transforms.augmentations import get_base_transforms

# Notation convention:
# {Model}ViewTransform refers to a transformation of a single view for a given model
# {Model}Transform refers to the multi-view transform to be passed to a dataset for a given model, i.e. num_views {Model}ViewTransform

# FastSiam can use BaseTransform with n_views=4, MAE with n_views=1


class BaseViewTransform:
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

        self.transform = get_base_transforms(
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

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        return self.transform(image)


class BaseTransform(MultiViewTransform):
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

        view_transform = BaseViewTransform(
            img_size=img_size,
            die_noise_prob=die_noise_prob,
            crop=crop,
            denoise=denoise,
            hf_prob=hf_prob,
            vf_prob=vf_prob,
            rr_prob=rr_prob,
            normalize=normalize,
        )
        transforms = [view_transform] * n_views
        super().__init__(transforms=transforms)
