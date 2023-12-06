import warnings
from pathlib import Path
from typing import *

import cv2
import numpy as np
from diskcache import FanoutCache
from einops import rearrange
from opencv_transforms import transforms
from scipy import ndimage

from inv3d_util.image import scale_image, tight_crop_image
from inv3d_util.load import load_array, load_image
from inv3d_util.mapping import scale_map, tight_crop_map
from inv3d_util.mask import scale_mask, tight_crop_mask

cache = FanoutCache(
    directory="/tmp/inv3d-model-cache", shards=128, size_limit=512 * 1024 * 1024 * 1024
)

warnings.simplefilter("ignore", UserWarning)

# (cached) function wrappers


def prepare_masked_image_jitter(*args, **kwargs):
    return _prepare_masked_image(*args, **kwargs, color_jitter=True)


def prepare_image_jitter(*args, **kwargs):
    return _prepare_image(*args, **kwargs, color_jitter=True)


@cache.memoize(typed=True)
def prepare_masked_image(*args, **kwargs):
    return _prepare_masked_image(*args, **kwargs, color_jitter=False)


@cache.memoize(typed=True)
def prepare_image(*args, **kwargs):
    return _prepare_image(*args, **kwargs, color_jitter=False)


@cache.memoize(typed=True)
def prepare_template(*args, **kwargs):
    return _prepare_template(*args, **kwargs)


@cache.memoize(typed=True)
def prepare_bm(*args, **kwargs):
    return _prepare_bm(*args, **kwargs)


@cache.memoize(typed=True)
def prepare_wc(*args, **kwargs):
    return _prepare_wc(*args, **kwargs)


@cache.memoize(typed=True)
def prepare_mask(*args, **kwargs):
    return _prepare_mask(*args, **kwargs)


# raw functions


def _prepare_masked_image(
    image_file: Path, uv_file: Path, color_jitter: bool, **scale_settings
):
    mask = load_array(uv_file)[..., :1]

    image = load_image(image_file)
    image = tight_crop_image(image, mask.squeeze())
    image = scale_image(image, **scale_settings)
    image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(image) if color_jitter else image
    image = rearrange(image, "h w c -> c h w")
    image = image.astype("float32") / 255
    return image


def _prepare_image(file: Path, color_jitter: bool, **scale_settings):
    image = load_image(file)
    image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.2)(image) if color_jitter else image
    image = scale_image(image, **scale_settings)
    image = rearrange(image, "h w c -> c h w")
    image = image.astype("float32") / 255
    return image


def _prepare_template(
    template_file: Path,
    delta_file: Path,
    text_file: Path,
    variation: str,
    **scale_settings,
):
    template = load_image(template_file)
    delta = load_image(delta_file)
    text = load_image(text_file)

    if variation == "full":
        pass
    elif variation == "white":
        template = np.ones_like(template, dtype=np.uint8) * 255
    elif variation == "struct":
        mask = np.any(text != 255, axis=-1, keepdims=True)
        mask = np.repeat(mask, 3, axis=-1)
        ind = ndimage.distance_transform_edt(
            input=mask, return_distances=False, return_indices=True
        )
        template = template[tuple(ind)]
    elif variation == "text":
        template = np.copy(text)
        template[np.any(delta != 255, axis=-1), :] = 255
    else:
        raise ValueError(f"Unknown template variation '{variation}'")

    template = scale_image(template, **scale_settings)
    template = rearrange(template, "h w c -> c h w")
    template = template.astype("float32") / 255
    return template


def _prepare_bm(file: Path, resolution: Optional[int] = None):
    assert file.suffix in [".npz", ".mat"]

    bm = load_array(file).astype("float32")

    if file.suffix == ".mat":
        bm = rearrange(bm, "c h w -> w h c")
        bm = np.roll(bm, shift=1, axis=-1)
        bm /= 448

    bm = tight_crop_map(bm)
    bm = scale_map(bm, resolution) if resolution is not None else bm
    bm = rearrange(bm, "h w c -> c h w")
    return bm


def _prepare_wc(
    file: Path, min_values: Tuple, max_values: Tuple, resolution: Optional[int] = None
):
    assert file.suffix in [".npz", ".exr"]

    wc = load_array(file).astype("float32")
    wc = wc[:, :, :3]  # cut off 4th channel if existent

    mask = np.any(wc != 0, axis=-1)
    wc = tight_crop_image(wc, mask)

    mask = np.any(wc != 0, axis=-1)
    wc = scale_image(wc, mask=mask, resolution=resolution)

    mask = np.any(wc != 0, axis=-1)
    wc = (wc - np.array(min_values)) / (np.array(max_values) - np.array(min_values))
    wc = cv2.bitwise_and(wc, wc, mask=mask.astype("uint8")).astype("float32")

    wc = rearrange(wc, "h w c -> c h w")

    return wc


def _prepare_mask(uv_file: Path, resolution: Optional[int] = None):
    assert uv_file.suffix in [".npz", ".exr"]

    mask = load_array(uv_file)[..., 0]

    mask = tight_crop_mask(mask)
    mask = scale_mask(mask, resolution=resolution)

    return mask
