"""Preprocessing tests with locally generated images (no fixtures)."""

from __future__ import annotations

import io
import struct
import zlib

import pytest
from PIL import Image

from faceapi.errors import BadImageError, ImageTooLargeError
from faceapi.preprocess import decode_image


def _jpeg_bytes(width: int, height: int, exif_orientation: int | None = None) -> bytes:
    img = Image.new("RGB", (width, height), (200, 30, 40))
    buf = io.BytesIO()
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        img.save(buf, format="JPEG", exif=exif)
    else:
        img.save(buf, format="JPEG")
    return buf.getvalue()


def test_decode_returns_bgr() -> None:
    arr = decode_image(_jpeg_bytes(32, 16))
    assert arr.shape == (16, 32, 3)
    assert arr.dtype.name == "uint8"
    # Red channel dominant in RGB becomes last channel in BGR.
    assert int(arr[0, 0, 2]) > int(arr[0, 0, 0])


def test_decode_applies_exif_orientation() -> None:
    arr = decode_image(_jpeg_bytes(32, 16, exif_orientation=6))
    assert arr.shape == (32, 16, 3)


def test_decode_downscales_long_edge() -> None:
    arr = decode_image(_jpeg_bytes(4000, 2000), max_dimension=1000)
    assert arr.shape == (500, 1000, 3)


def test_decode_converts_grayscale() -> None:
    buf = io.BytesIO()
    Image.new("L", (8, 8)).save(buf, format="PNG")
    assert decode_image(buf.getvalue()).shape == (8, 8, 3)


@pytest.mark.parametrize("data", [b"", b"not-an-image", b"\x89PNGgarbage"])
def test_decode_rejects_bad_bytes(data: bytes) -> None:
    with pytest.raises(BadImageError):
        decode_image(data)


def test_decode_rejects_oversized_pixel_count() -> None:
    """A tiny file whose header claims huge dimensions is refused before decode.

    The IHDR width/height of a real 8x8 PNG are patched to 5000x5000 (CRC
    recomputed so Pillow still identifies the file): above the 4096x4096
    pixel cap but below Pillow's own decompression-bomb threshold, so the
    refusal provably comes from our cap, on the header, before any pixel
    allocation.
    """
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    header = bytearray(buf.getvalue())
    struct.pack_into(">II", header, 16, 5_000, 5_000)
    struct.pack_into(">I", header, 29, zlib.crc32(bytes(header[12:29])))
    with pytest.raises(ImageTooLargeError):
        decode_image(bytes(header), max_pixels=4096 * 4096)
