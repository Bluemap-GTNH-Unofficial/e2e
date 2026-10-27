from pathlib import Path

import numpy as np
import pytest

from e2e import metrics, paths

MISSING_PNG = paths.REPO / "core/src/main/resourceExtensions/assets/bluemap/textures/block/missing.png"


SKY = np.array((125, 171, 255), dtype=np.uint8)


def _missing_texture() -> np.ndarray:
    return metrics.load_rgb(MISSING_PNG)


def _sky_like(image: np.ndarray) -> np.ndarray:
    return np.broadcast_to(SKY, image.shape).copy()


@pytest.mark.parametrize("brightness", [1.0, 0.6, 0.3])
@pytest.mark.parametrize("scale", [1, 4])
def test_lit_missing_texture_is_fully_counted(brightness: float, scale: int) -> None:
    big = _missing_texture().repeat(scale, axis=0).repeat(scale, axis=1)
    image = (big.astype(np.float32) * brightness).round().astype(np.uint8)
    measured = metrics.measure(image, _sky_like(image))
    assert measured.magenta_ratio == pytest.approx(0.5)
    assert measured.missing_of_content == pytest.approx(1.0)


@pytest.mark.parametrize("texels_per_pixel", [2.7, 5.3, 11.1])
def test_minified_missing_texture_is_estimated_without_bias(texels_per_pixel: float) -> None:
    # a far camera: nearest-filtered sampling of a field of missing-texture blocks, many texels per screen pixel
    field = np.tile(_missing_texture(), (64, 64, 1))
    coords = (np.arange(300) * texels_per_pixel + 0.37).astype(int) % field.shape[0]
    image = field[np.ix_(coords, coords)]
    measured = metrics.measure(image, _sky_like(image))
    assert measured.missing_of_content == pytest.approx(1.0, abs=0.1)


def test_lowres_missing_colour_is_magenta() -> None:
    # Texture.MISSING colour (0.5, 0, 0.5) as written to lowres tiles
    assert metrics.magenta_mask(np.array([[[128, 0, 128]]], dtype=np.uint8)).all()


@pytest.mark.parametrize("rgb", [
    (189, 68, 179),   # magenta wool
    (123, 47, 190),   # purple wool
    (169, 125, 169),  # purpur
    (125, 171, 255),  # default sky colour #7dabff
    (20, 0, 18),      # nearly black: too dark to tell
])
def test_real_block_colours_are_not_missing(rgb: tuple[int, int, int]) -> None:
    assert not metrics.magenta_mask(np.array([[rgb]], dtype=np.uint8)).any()


def test_diff_counts_pixels_beyond_threshold() -> None:
    baseline = np.full((10, 10, 3), 100, dtype=np.uint8)
    current = baseline.copy()
    current[0, :5] = (100, 100, 125)  # 25 > 24: differs
    current[1, :] = (124, 100, 100)   # 24: within threshold
    mask = metrics.diff_mask(current, baseline, pixel_threshold=24)
    assert mask.sum() == 5


def test_diff_rejects_size_mismatch() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        metrics.diff_mask(np.zeros((2, 2, 3), np.uint8), np.zeros((2, 3, 3), np.uint8), 0)


def test_measure_reports_content_and_missing_shares() -> None:
    sky = np.zeros((10, 10, 3), dtype=np.uint8)
    sky[...] = SKY
    image = sky.copy()
    image[:5, :] = (90, 140, 60)        # half the frame: grass-like content
    image[:1, :5] = (255, 0, 255)       # 5 magenta pixels inside the content
    image[9, 9] = (200, 0, 200)         # magenta-like but indistinguishable from the sky frame: ignored
    sky[9, 9] = (200, 0, 200)
    measured = metrics.measure(image, sky)
    assert measured.content_ratio == pytest.approx(0.5)
    assert measured.magenta_ratio == pytest.approx(0.05)
    assert measured.missing_ratio == pytest.approx(0.1)
    assert measured.missing_of_content == pytest.approx(0.2)


def test_diff_image_marks_differences_red(tmp_path: Path) -> None:
    current = np.full((4, 4, 3), 200, dtype=np.uint8)
    mask = np.zeros((4, 4), dtype=bool)
    mask[2, 3] = True
    out = metrics.diff_image(current, mask)
    assert tuple(out[2, 3]) == (255, 0, 0)
    assert tuple(out[0, 0]) == (80, 80, 80)
    metrics.save_rgb(out, tmp_path / "d.png")
    assert (metrics.load_rgb(tmp_path / "d.png") == out).all()
