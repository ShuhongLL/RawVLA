import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


def validate_image_array(img: np.ndarray) -> np.ndarray:
    """Validate the public image contract: HWC uint8 or float32 in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected HWC image with 1/3/4 channels, got shape={arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype != np.float32:
        raise TypeError(f"Image dtype must be uint8 or float32, got {arr.dtype}")
    if not np.isfinite(arr).all():
        raise ValueError("Float image contains NaN or infinity")
    if arr.size and (float(arr.min()) < 0.0 or float(arr.max()) > 1.0):
        raise ValueError(
            f"Float image values must be in [0, 1], got [{float(arr.min())}, {float(arr.max())}]"
        )
    return arr


def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    arr = validate_image_array(img)
    if arr.dtype == np.float32:
        arr = np.rint(255.0 * arr).astype(np.uint8)
    return arr


def _resize_float_image(image: np.ndarray, height: int, width: int, method: int) -> np.ndarray:
    channels = [
        np.asarray(Image.fromarray(image[..., index], mode="F").resize((width, height), resample=method))
        for index in range(image.shape[-1])
    ]
    return np.stack(channels, axis=-1).astype(np.float32, copy=False)


def resize_image(image: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Resize one HWC image while preserving uint8 or float32 dtype and range."""
    arr = validate_image_array(image)
    if arr.shape[:2] == (height, width):
        return arr
    if arr.dtype == np.float32:
        return _resize_float_image(arr, height, width, method)
    return np.asarray(Image.fromarray(arr).resize((width, height), resample=method))


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    """Replicates tf.image.resize_with_pad for multiple images using PIL. Resizes a batch of images to a target height.

    Args:
        images: A batch of images in [..., height, width, channel] format.
        height: The target height of the image.
        width: The target width of the image.
        method: The interpolation method to use. Default is bilinear.

    Returns:
        The resized images in [..., height, width, channel].
    """
    images = np.asarray(images)
    if images.ndim < 3:
        raise ValueError(f"Expected image batch ending in HWC, got shape={images.shape}")
    validate_image_array(images.reshape(-1, *images.shape[-3:])[0])
    # If the images are already the correct size, return them as is.
    if images.shape[-3:-1] == (height, width):
        return images

    original_shape = images.shape

    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack([_resize_with_pad(im, height, width, method=method) for im in images])
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def _resize_with_pad(image: np.ndarray, height: int, width: int, method: int) -> np.ndarray:
    arr = validate_image_array(image)
    cur_height, cur_width = arr.shape[:2]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized = resize_image(arr, resized_height, resized_width, method=method)
    output = np.zeros((height, width, arr.shape[-1]), dtype=arr.dtype)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    output[pad_height : pad_height + resized_height, pad_width : pad_width + resized_width] = resized
    return output


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method: int) -> Image.Image:
    """Replicates tf.image.resize_with_pad for one image using PIL. Resizes an image to a target height and
    width without distortion by padding with zeros.

    Unlike the jax version, note that PIL uses [width, height, channel] ordering instead of [batch, h, w, c].
    """
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image  # No need to resize if the image is already the correct size.

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)

    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    assert zero_image.size == (width, height)
    return zero_image


from typing import Any


def to_pil_preserve(images: Any, scale_float: bool = True):
    """
    Convert (possibly nested) numpy image arrays back to PIL.Image WITHOUT changing spatial shape
    or nesting structure.

    Accepts:
      - np.ndarray with shape (H, W, C), C in {1,3,4}, dtype uint8 or float
      - PIL.Image.Image (returned as-is)
      - Nested list / tuple structures containing the above

    Guarantees:
      - No resize / pad / crop performed
      - Returns an object with the SAME nesting layout (list -> list, tuple -> tuple)
      - Only dtype (float -> uint8) and channel-mode adaptation may happen
        * float arrays assumed in [0,1] if scale_float=True (scaled *255 + clip)
    Args:
      images: input object / sequence
      scale_float: whether to scale float images in [0,1] to uint8
    Returns:
      Mirrored structure with all leaf nodes as PIL.Image.Image
    """

    def _convert(obj):
        # Nested containers
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)

        # PIL stays
        if isinstance(obj, Image.Image):
            return obj

        # numpy -> PIL
        if isinstance(obj, np.ndarray):
            arr = validate_image_array(obj)
            if arr.dtype == np.float32:
                if scale_float:
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    raise TypeError("Float array provided but scale_float=False")

            # Single channel -> 'L'
            if arr.shape[2] == 1:
                arr = arr[:, :, 0]
                return Image.fromarray(arr, mode="L")
            # 3 channels -> RGB, 4 -> RGBA
            mode = "RGB" if arr.shape[2] == 3 else "RGBA"
            return Image.fromarray(arr, mode=mode)

        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)


def preserve_float_or_convert_uint8_to_pil(images: Any):
    """Keep float32 image leaves exact and convert uint8 leaves to PIL."""

    def _convert(obj):
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, tuple):
            return tuple(_convert(x) for x in obj)
        if isinstance(obj, Image.Image):
            return obj
        if isinstance(obj, np.ndarray):
            arr = validate_image_array(obj)
            return arr if arr.dtype == np.float32 else to_pil_preserve(arr)
        raise TypeError(f"Unsupported element type: {type(obj)}")

    return _convert(images)


def resize_model_images(images: Any, target_size: tuple[int, int]):
    """Resize nested PIL/NumPy model images without quantizing float32 leaves."""
    width, height = target_size
    if isinstance(images, list):
        return [resize_model_images(x, target_size) for x in images]
    if isinstance(images, tuple):
        return tuple(resize_model_images(x, target_size) for x in images)
    if isinstance(images, Image.Image):
        return images.resize((width, height))
    if isinstance(images, np.ndarray):
        return resize_image(images, height, width)
    raise TypeError(f"Unsupported image type: {type(images)}")


_AUDITED_STAGES: set[str] = set()


def audit_image_once(stage: str, image: Any) -> None:
    """Append a one-shot image integrity record when STARVLA_IMAGE_AUDIT_PATH is set."""
    path = os.environ.get("STARVLA_IMAGE_AUDIT_PATH")
    if not path or stage in _AUDITED_STAGES:
        return
    arr = np.asarray(image)
    record = {
        "stage": stage,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "unique_values": int(np.unique(arr).size),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }
    audit_path = Path(path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    _AUDITED_STAGES.add(stage)
