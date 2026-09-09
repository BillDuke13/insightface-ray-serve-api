"""Decode raw bytes into normalized BGR images for inference.

Synchronous by design: callers run :func:`decode_image` in a worker thread so
the event loop never blocks on pixel work. EXIF orientation is applied so
phone photos enter detection upright.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

from faceapi.errors import BadImageError, ImageTooLargeError

logger = logging.getLogger(__name__)


def decode_image(
    data: bytes, *, max_dimension: int = 1920, max_pixels: int = 4096 * 4096
) -> npt.NDArray[Any]:
    """Decode bytes to a BGR uint8 array, downscaled to ``max_dimension``.

    Args:
        data: Raw encoded image bytes (JPEG/PNG/WebP/...).
        max_dimension: Longest edge cap in pixels; aspect ratio is preserved.
        max_pixels: Total pixel cap, checked against the image header before
            any pixel is decoded, so crafted files cannot force huge
            allocations.

    Raises:
        BadImageError: If the bytes are not a decodable image.
        ImageTooLargeError: If the declared pixel count exceeds ``max_pixels``.
    """
    if not data:
        raise BadImageError("Image data is empty.")
    try:
        handle = Image.open(io.BytesIO(data))
    except Exception as exc:
        raise BadImageError("Could not decode image data.") from exc
    with handle:
        if handle.size[0] * handle.size[1] > max_pixels:
            raise ImageTooLargeError(f"Image exceeds the {max_pixels} pixel limit.")
        try:
            rgb = np.asarray(ImageOps.exif_transpose(handle.convert("RGB")))
        except Exception as exc:
            raise BadImageError("Could not decode image data.") from exc
    longest = max(rgb.shape[0], rgb.shape[1])
    if longest > max_dimension:
        scale = max_dimension / longest
        rgb = cv2.resize(
            rgb,
            (int(rgb.shape[1] * scale), int(rgb.shape[0] * scale)),
            interpolation=cv2.INTER_AREA,
        )
    bgr: npt.NDArray[Any] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    logger.debug("decoded image: shape=%s", bgr.shape)
    return bgr
