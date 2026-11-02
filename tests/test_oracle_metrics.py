import math

import numpy as np
import pytest

from e2e.oracle import metrics

# Sharma, Wu, Dalal (2005), "The CIEDE2000 color-difference formula", Table 1 test data (pairs 1, 2, 3, 7, 17, 25)
SHARMA = [
    ((50.0, 2.6772, -79.7751), (50.0, 0.0, -82.7485), 2.0425),
    ((50.0, 3.1571, -77.2803), (50.0, 0.0, -82.7485), 2.8615),
    ((50.0, 2.8361, -74.0200), (50.0, 0.0, -82.7485), 3.4412),
    ((50.0, 0.0, 0.0), (50.0, -1.0, 2.0), 2.3669),
    ((50.0, 2.5, 0.0), (73.0, 25.0, -18.0), 27.1492),
    ((60.2574, -34.0099, 36.2677), (60.4626, -34.1751, 39.4387), 1.2644),
]


@pytest.mark.parametrize(("lab1", "lab2", "expected"), SHARMA)
def test_ciede2000_matches_published_pairs(lab1, lab2, expected) -> None:
    got = metrics.ciede2000(np.array(lab1), np.array(lab2))
    assert float(got) == pytest.approx(expected, abs=1e-4)
    assert float(metrics.ciede2000(np.array(lab2), np.array(lab1))) == pytest.approx(expected, abs=1e-4)


def test_srgb_to_lab_reference_points() -> None:
    lab = metrics.srgb_to_lab(np.array([[255, 255, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8))
    assert lab[0] == pytest.approx([100.0, 0.0, 0.0], abs=0.01)
    assert lab[1] == pytest.approx([0.0, 0.0, 0.0], abs=0.01)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.05)


def _textured(seed: int = 1, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    texture = rng.integers(0, 256, (size // 4, size // 4, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)
    return np.concatenate([texture, np.full((size, size, 1), 255, np.uint8)], axis=2)


def test_identical_images_are_perfect() -> None:
    image = _textured()
    m = metrics.compare_region(image, image.copy())
    assert (m.iou, m.delta_e, m.delta_e_p95) == (1.0, 0.0, 0.0)
    assert m.ssim == pytest.approx(1.0)
    assert m.ref_pixels == m.test_pixels == m.both_pixels == 64 * 64


def test_regions_and_coverage() -> None:
    ref = _textured()
    test = ref.copy()
    test[:, 32:, 3] = 0  # right half transparent in the test image
    m = metrics.compare_region(ref, test)
    assert m.iou == pytest.approx(1 - 31 * 64 / 4096)  # the column next to the test's edge is within 1 px
    assert m.delta_e == 0.0
    left = np.zeros((64, 64), bool)
    left[:, :32] = True
    m = metrics.compare_region(ref, test, left)
    assert m.iou == 1.0 and m.both_pixels == 64 * 32
    # separate regions: the test attributes the left half's top rows to another face
    test_region = left.copy()
    test_region[:16] = False
    m = metrics.compare_region(ref, test, left, test_region)
    assert (m.ref_pixels, m.test_pixels, m.both_pixels) == (64 * 32, 48 * 32, 48 * 32)
    assert m.iou == pytest.approx(1 - 15 * 32 / (64 * 32))


def test_iou_tolerates_a_one_pixel_edge_shift_only() -> None:
    ref = np.zeros((40, 40, 4), np.uint8)
    ref[10:30, 10:30] = 255
    one = np.roll(ref, 1, axis=1)
    assert metrics.compare_region(ref, one).iou == 1.0
    three = np.roll(ref, 3, axis=1)
    assert metrics.compare_region(ref, three).iou == pytest.approx(1 - 2 * 2 * 20 / (23 * 20))


def test_empty_union_is_not_judged() -> None:
    empty = np.zeros((32, 32, 4), np.uint8)
    m = metrics.compare_region(empty, empty)
    assert m.iou == 1.0 and math.isnan(m.delta_e) and math.isnan(m.ssim) and m.union_pixels == 0
    t = metrics.Thresholds(0.9, 3.0, 0.8)
    assert not t.judged(m) and t.check(m) == []


def test_structure_and_colour_errors_are_separated() -> None:
    ref = _textured(1)
    shuffled = _textured(2)  # same statistics, different structure
    tinted = ref.copy()
    tinted[..., :3] = (tinted[..., :3] * np.array([1.0, 0.8, 0.8])).astype(np.uint8)
    s = metrics.compare_region(ref, shuffled)
    t = metrics.compare_region(ref, tinted)
    assert s.ssim < 0.3 < t.ssim
    assert t.delta_e > 3
    assert metrics.compare_region(ref, ref).ssim > t.ssim


def test_thresholds_check() -> None:
    t = metrics.Thresholds(iou_min=0.9, delta_e_max=5.0, ssim_min=0.7)
    good = metrics.RegionMetrics(100, 100, 95, 0.95, 2.0, 4.0, 0.9)
    bad = metrics.RegionMetrics(100, 100, 50, 50 / 150, 9.0, 20.0, 0.2)
    assert t.check(good) == []
    assert [c.split()[0] for c in t.check(bad)] == ["iou", "delta_e", "ssim"]
    assert metrics.Thresholds.from_json(t.to_json()) == t


def test_calibrate_uses_tukey_fences() -> None:
    rng = np.random.default_rng(3)
    samples = [metrics.RegionMetrics(200, 200, 190, float(i), float(d), float(d) * 2, float(s))
               for i, d, s in zip(rng.uniform(0.96, 1.0, 400), rng.uniform(0.0, 2.0, 400), rng.uniform(0.9, 1.0, 400))]
    samples.append(metrics.RegionMetrics(5, 5, 0, 0.0, math.nan, math.nan, math.nan))  # below min_pixels
    cal = metrics.calibrate(samples)
    assert cal.samples == 400
    iou = np.array([m.iou for m in samples[:-1]])
    q1, q3 = np.percentile(iou, [25, 75])
    assert cal.thresholds.iou_min == pytest.approx(q1 - 3 * (q3 - q1), abs=1e-4)
    de = np.array([m.delta_e for m in samples[:-1]])
    q1, q3 = np.percentile(de, [25, 75])
    assert cal.thresholds.delta_e_max == pytest.approx(q3 + 3 * (q3 - q1), abs=1e-3)
    assert sum(cal.distributions["delta_e"]["histogram"]["counts"]) == 400
    assert cal.to_json()["rule"].startswith("Tukey far-out fences")


def test_calibrate_leaves_exact_agreements_out_and_takes_iou_from_geometry_samples() -> None:
    rng = np.random.default_rng(4)
    exact = [metrics.RegionMetrics(200, 200, 200, 1.0, 0.0, 0.0, 1.0)] * 600  # most regions agree exactly
    noisy = [metrics.RegionMetrics(200, 200, 190, float(i), float(d), float(d), float(s))
             for i, d, s in zip(rng.uniform(0.96, 0.999, 200), rng.uniform(0.1, 2.0, 200), rng.uniform(0.9, 0.999, 200))]
    cal = metrics.calibrate(exact + noisy)
    de = np.array([m.delta_e for m in noisy])
    q1, q3 = np.percentile(de, [25, 75])
    assert cal.thresholds.delta_e_max == pytest.approx(q3 + 3 * (q3 - q1), abs=1e-3)  # not 0 (all-sample IQR is 0)
    assert cal.fences["delta_e"]["exact"] == 600 and cal.fences["delta_e"]["inexact"] == 200
    geometry = [metrics.RegionMetrics(200, 200, 150, float(i), 0.0, 0.0, 1.0) for i in rng.uniform(0.7, 0.8, 50)]
    wide = metrics.calibrate(exact + noisy, geometry_samples=geometry)
    assert wide.thresholds.iou_min < 0.7 and wide.thresholds.delta_e_max == cal.thresholds.delta_e_max
    assert wide.to_json()["geometrySamples"] == 50
    only_exact = metrics.calibrate(exact)
    assert (only_exact.thresholds.iou_min, only_exact.thresholds.delta_e_max, only_exact.thresholds.ssim_min) == (1.0, 0.0, 1.0)


def test_summarize_ignores_nan() -> None:
    s = metrics.summarize([1.0, 2.0, math.nan, 3.0, None])
    assert s["count"] == 3 and s["p50"] == 2.0 and s["max"] == 3.0
    assert metrics.summarize([]) == {"count": 0}
