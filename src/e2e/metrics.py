"""Image metrics on RGB screenshots: baseline difference, missing-texture pixels, non-sky content."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# BlueMap's missing texture (core resourceExtensions `assets/bluemap/textures/block/missing.png`, also
# `Texture.MISSING`) is a 16x16 checkerboard of 8x8 #FF00FF and #000000 quadrants; its lowres colour is
# (0.5, 0, 0.5). Lighting scales all channels alike, so the magenta half shows as "red == blue, no green" at any
# brightness. The black half cannot be told apart from shadows, so the missing-texture area is estimated as twice
# the magenta area: unbiased for hires faces at any zoom (nearest-filtered textures sample both halves equally
# often), an overestimate where only lowres tiles (solid purple) are visible.
MAGENTA_MIN_BRIGHTNESS = 40
SKY_PIXEL_THRESHOLD = 12


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def save_rgb(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, "RGB").save(path)


def _channel_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)


def diff_mask(current: np.ndarray, baseline: np.ndarray, pixel_threshold: int) -> np.ndarray:
    if current.shape != baseline.shape:
        raise ValueError(f"size mismatch: current {current.shape[1]}x{current.shape[0]}, "
                         f"baseline {baseline.shape[1]}x{baseline.shape[0]}")
    return _channel_delta(current, baseline) > pixel_threshold


def diff_image(current: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Differing pixels in red over a dimmed greyscale copy of the current screenshot."""
    grey = (current.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) * 0.4
    out = np.repeat(grey[..., None], 3, axis=2).astype(np.uint8)
    out[mask] = (255, 0, 0)
    return out


def magenta_mask(image: np.ndarray) -> np.ndarray:
    r, g, b = (image[..., i].astype(np.int32) for i in range(3))
    low, high = np.minimum(r, b), np.maximum(r, b)
    return (high >= MAGENTA_MIN_BRIGHTNESS) & (4 * g <= low) & (4 * (high - low) <= high)


def content_mask(image: np.ndarray, sky: np.ndarray) -> np.ndarray:
    """Pixels that differ from a frame of the same camera with only the sky drawn."""
    return _channel_delta(image, sky) > SKY_PIXEL_THRESHOLD


@dataclass(frozen=True)
class ImageMetrics:
    pixels: int
    content_ratio: float  # share of pixels showing something other than sky
    magenta_ratio: float  # share of pixels showing the magenta half of the missing texture
    missing_ratio: float  # estimated share of pixels showing the missing texture (2 x magenta, capped by content)
    missing_of_content: float  # estimated missing-texture pixels / content pixels


def measure(image: np.ndarray, sky: np.ndarray) -> ImageMetrics:
    pixels = image.shape[0] * image.shape[1]
    content = content_mask(image, sky)
    content_pixels = int(content.sum())
    magenta = int((magenta_mask(image) & content).sum())
    missing = min(2 * magenta, content_pixels)
    return ImageMetrics(
        pixels=pixels,
        content_ratio=content_pixels / pixels,
        magenta_ratio=magenta / pixels,
        missing_ratio=missing / pixels,
        missing_of_content=missing / content_pixels if content_pixels else 0.0,
    )
