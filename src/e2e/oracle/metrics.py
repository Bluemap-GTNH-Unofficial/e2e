"""Region metrics for comparing a reference rendering (the real client) with BlueMap's rendering.

Both images are straight-alpha RGBA ``uint8`` arrays of the same size. A *region* is a boolean mask of pixels that
belong to the compared thing (one block face, a whole block view, a face projected into a screenshot); the
*coverage* of an image in a region is the region's pixels whose alpha is at least ``alpha_min``. Reference and test
may have their own region (e.g. the pixels each renderer attributes to the block's up face).

Per region:

- ``iou``: intersection over union of the two coverages (geometry/silhouette agreement), 1.0 if both are empty.
- ``delta_e``: mean CIEDE2000 colour difference over pixels covered in both, ``delta_e_p95`` its 95th percentile.
- ``ssim``: structural similarity of the luminance (Wang et al. 2004, 7x7 uniform window, K1 0.01, K2 0.03) on the
  bounding box of the union of both coverages, block-mean downsampled by ``ssim_downsample``; each image is first
  composited over mid-grey with everything outside its coverage set to mid-grey, and the SSIM map is averaged over
  the downsampled union. NaN if the box is smaller than the window after downsampling.

[Thresholds] turns metrics into pass/fail; [calibrate] derives thresholds from a sample distribution (the vanilla
blocks of the default pack) with Tukey's far-out fences, so they are never tuned by hand.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

import numpy as np

ALPHA_MIN = 128
SSIM_WINDOW = 7
SSIM_DOWNSAMPLE = 2
_SSIM_C1 = (0.01 * 255) ** 2
_SSIM_C2 = (0.03 * 255) ** 2
_GREY = 128.0

# sRGB (D65) -> CIE XYZ, IEC 61966-2-1
_RGB_TO_XYZ = np.array([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_WHITE = np.array([0.95047, 1.0, 1.08883])
_LINEAR = np.array([c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in np.arange(256) / 255.0])


@dataclass(frozen=True)
class RegionMetrics:
    ref_pixels: int
    test_pixels: int
    both_pixels: int
    iou: float
    delta_e: float
    delta_e_p95: float
    ssim: float

    @property
    def union_pixels(self) -> int:
        return self.ref_pixels + self.test_pixels - self.both_pixels

    def to_json(self) -> dict:
        return {k: (None if isinstance(v, float) and math.isnan(v) else (round(v, 4) if isinstance(v, float) else v))
                for k, v in asdict(self).items()}


def srgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """``(..., 3)`` uint8 sRGB -> ``(..., 3)`` float CIE L*a*b* (D65)."""
    linear = _LINEAR[rgb.astype(np.intp)]
    xyz = linear @ _RGB_TO_XYZ.T / _WHITE
    delta = 6 / 29
    f = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], axis=-1)


def ciede2000(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference (Sharma, Wu, Dalal 2005; kL = kC = kH = 1) of ``(..., 3)`` Lab arrays."""
    l1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    l2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    c_bar = (np.hypot(a1, b1) + np.hypot(a2, b2)) / 2
    g = 0.5 * (1 - np.sqrt(c_bar ** 7 / (c_bar ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + g) * a1, (1 + g) * a2
    c1p, c2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    chroma = c1p * c2p
    zero = chroma == 0

    dlp = l2 - l1
    dcp = c2p - c1p
    dh = h2p - h1p
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(zero, 0.0, dh)
    dhp = 2 * np.sqrt(chroma) * np.sin(np.radians(dh) / 2)

    lp_bar = (l1 + l2) / 2
    cp_bar = (c1p + c2p) / 2
    h_sum = h1p + h2p
    hp_bar = np.where(np.abs(h1p - h2p) <= 180, h_sum / 2, np.where(h_sum < 360, (h_sum + 360) / 2, (h_sum - 360) / 2))
    hp_bar = np.where(zero, h_sum, hp_bar)

    t = (1 - 0.17 * np.cos(np.radians(hp_bar - 30)) + 0.24 * np.cos(np.radians(2 * hp_bar))
         + 0.32 * np.cos(np.radians(3 * hp_bar + 6)) - 0.20 * np.cos(np.radians(4 * hp_bar - 63)))
    d_theta = 30 * np.exp(-(((hp_bar - 275) / 25) ** 2))
    r_c = 2 * np.sqrt(cp_bar ** 7 / (cp_bar ** 7 + 25.0 ** 7))
    s_l = 1 + 0.015 * (lp_bar - 50) ** 2 / np.sqrt(20 + (lp_bar - 50) ** 2)
    s_c = 1 + 0.045 * cp_bar
    s_h = 1 + 0.015 * cp_bar * t
    r_t = -np.sin(np.radians(2 * d_theta)) * r_c
    tl, tc, th = dlp / s_l, dcp / s_c, dhp / s_h
    return np.sqrt(np.maximum(tl ** 2 + tc ** 2 + th ** 2 + r_t * tc * th, 0.0))


def _box_mean(image: np.ndarray, size: int) -> np.ndarray:
    """Mean over every ``size`` x ``size`` window (valid positions only), via an integral image."""
    s = np.pad(image, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (s[size:, size:] - s[:-size, size:] - s[size:, :-size] + s[:-size, :-size]) / (size * size)


def _downsample(image: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return image
    h, w = image.shape[0] // factor * factor, image.shape[1] // factor * factor
    return image[:h, :w].reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def _luminance(rgba: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    rgb = rgba[..., :3].astype(np.float64)
    alpha = rgba[..., 3:4].astype(np.float64) / 255
    luma = (rgb * alpha + _GREY * (1 - alpha)) @ np.array([0.299, 0.587, 0.114])
    return np.where(coverage, luma, _GREY)


def ssim(ref_luma: np.ndarray, test_luma: np.ndarray, weight: np.ndarray | None = None) -> float:
    """Mean SSIM of two luminance images (0..255), averaged over ``weight`` > 0.5 at the window centres."""
    w = SSIM_WINDOW
    if ref_luma.shape[0] < w or ref_luma.shape[1] < w:
        return math.nan
    mu_x, mu_y = _box_mean(ref_luma, w), _box_mean(test_luma, w)
    n = w * w
    correction = n / (n - 1)  # sample covariance, as skimage
    var_x = (_box_mean(ref_luma * ref_luma, w) - mu_x ** 2) * correction
    var_y = (_box_mean(test_luma * test_luma, w) - mu_y ** 2) * correction
    cov = (_box_mean(ref_luma * test_luma, w) - mu_x * mu_y) * correction
    smap = ((2 * mu_x * mu_y + _SSIM_C1) * (2 * cov + _SSIM_C2)) / (
        (mu_x ** 2 + mu_y ** 2 + _SSIM_C1) * (var_x + var_y + _SSIM_C2))
    if weight is not None:
        centre = weight[w // 2: w // 2 + smap.shape[0], w // 2: w // 2 + smap.shape[1]] > 0.5
        if not centre.any():
            return math.nan
        return float(smap[centre].mean())
    return float(smap.mean())


def compare_region(ref_rgba: np.ndarray, test_rgba: np.ndarray, ref_region: np.ndarray | None = None,
                   test_region: np.ndarray | None = None, *, alpha_min: int = ALPHA_MIN,
                   ssim_downsample: int = SSIM_DOWNSAMPLE) -> RegionMetrics:
    """Compares two straight-alpha RGBA images inside their regions (``test_region`` defaults to ``ref_region``,
    ``None`` is the whole image)."""
    if ref_rgba.shape != test_rgba.shape or ref_rgba.ndim != 3 or ref_rgba.shape[2] != 4:
        raise ValueError(f"need two RGBA images of the same size, got {ref_rgba.shape} and {test_rgba.shape}")
    if test_region is None:
        test_region = ref_region
    ref_cov = ref_rgba[..., 3] >= alpha_min
    test_cov = test_rgba[..., 3] >= alpha_min
    if ref_region is not None:
        ref_cov &= ref_region
    if test_region is not None:
        test_cov &= test_region
    both = ref_cov & test_cov
    union = ref_cov | test_cov
    n_ref, n_test, n_both, n_union = int(ref_cov.sum()), int(test_cov.sum()), int(both.sum()), int(union.sum())
    iou = n_both / n_union if n_union else 1.0

    if n_both:
        de = ciede2000(srgb_to_lab(ref_rgba[..., :3][both]), srgb_to_lab(test_rgba[..., :3][both]))
        delta_e, delta_e_p95 = float(de.mean()), float(np.percentile(de, 95))
    else:
        delta_e = delta_e_p95 = math.nan

    structural = math.nan
    if n_union:
        rows, cols = np.nonzero(union)
        box = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        ref_l = _downsample(_luminance(ref_rgba[box], ref_cov[box]), ssim_downsample)
        test_l = _downsample(_luminance(test_rgba[box], test_cov[box]), ssim_downsample)
        weight = _downsample(union[box].astype(np.float64), ssim_downsample)
        structural = ssim(ref_l, test_l, weight)
    return RegionMetrics(n_ref, n_test, n_both, iou, delta_e, delta_e_p95, structural)


@dataclass(frozen=True)
class Thresholds:
    """Pass criteria; regions whose coverage union is smaller than ``min_pixels`` are not judged."""

    iou_min: float
    delta_e_max: float
    ssim_min: float
    min_pixels: int = 16

    def check(self, m: RegionMetrics) -> list[str]:
        """The failed criteria (empty: pass). NaN metrics (nothing to compare) skip their check."""
        if m.union_pixels < self.min_pixels:
            return []
        failed = []
        if m.iou < self.iou_min:
            failed.append(f"iou {m.iou:.3f} < {self.iou_min:.3f}")
        if not math.isnan(m.delta_e) and m.delta_e > self.delta_e_max:
            failed.append(f"delta_e {m.delta_e:.2f} > {self.delta_e_max:.2f}")
        if not math.isnan(m.ssim) and m.ssim < self.ssim_min:
            failed.append(f"ssim {m.ssim:.3f} < {self.ssim_min:.3f}")
        return failed

    def judged(self, m: RegionMetrics) -> bool:
        return m.union_pixels >= self.min_pixels

    def to_json(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_json(data: dict) -> Thresholds:
        return Thresholds(float(data["iou_min"]), float(data["delta_e_max"]), float(data["ssim_min"]),
                          int(data.get("min_pixels", 16)))


def finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray([v for v in values if v is not None], dtype=np.float64)
    return array[np.isfinite(array)]


def summarize(values: Iterable[float]) -> dict:
    """Count, mean and quantiles of the finite values."""
    v = finite(values)
    if not v.size:
        return {"count": 0}
    q = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    return {"count": int(v.size), "mean": float(v.mean()), "min": float(v.min()), "p01": float(q[0]),
            "p05": float(q[1]), "p25": float(q[2]), "p50": float(q[3]), "p75": float(q[4]), "p95": float(q[5]),
            "p99": float(q[6]), "max": float(v.max())}


def histogram(values: Iterable[float], edges: Sequence[float]) -> dict:
    """Counts of the finite values per bin (``edges`` ascending; values outside fall into the first/last bin)."""
    v = np.clip(finite(values), edges[0], edges[-1])
    counts, _ = np.histogram(v, bins=np.asarray(edges, dtype=np.float64))
    return {"edges": [float(e) for e in edges], "counts": [int(c) for c in counts]}


@dataclass
class Calibration:
    thresholds: Thresholds
    samples: int
    rule: str
    distributions: dict = field(default_factory=dict)
    fences: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"thresholds": self.thresholds.to_json(), "samples": self.samples, "rule": self.rule,
                "fences": self.fences, "distributions": self.distributions}


IOU_EDGES = [0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 1.0]
DELTA_E_EDGES = [0, 0.5, 1, 1.5, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50, 100]
SSIM_EDGES = [-1, 0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 1.0]


def calibrate(samples: Sequence[RegionMetrics], *, fence: float = 3.0, min_pixels: int = 16) -> Calibration:
    """Thresholds from Tukey's fences over a sample distribution, e.g. every vanilla face with the default pack.

    ``iou_min = Q1 - fence * IQR`` and ``ssim_min`` likewise (clipped to their range), ``delta_e_max = Q3 + fence *
    IQR``. With ``fence = 3`` (Tukey's "far out") a region fails only when it is an outlier against the bulk of the
    samples, whatever produced the bulk's residual error (rasterisation edges, sampling, shading).
    """
    judged = [m for m in samples if m.union_pixels >= min_pixels]
    columns = {"iou": [m.iou for m in judged], "delta_e": [m.delta_e for m in judged], "ssim": [m.ssim for m in judged]}
    fences = {}
    for name, values in columns.items():
        v = finite(values)
        if not v.size:
            raise ValueError(f"no finite {name} values to calibrate on")
        q1, q3 = (float(x) for x in np.percentile(v, [25, 75]))
        fences[name] = {"q1": q1, "q3": q3, "iqr": q3 - q1}
    iou_min = max(0.0, fences["iou"]["q1"] - fence * fences["iou"]["iqr"])
    ssim_min = max(-1.0, fences["ssim"]["q1"] - fence * fences["ssim"]["iqr"])
    delta_e_max = fences["delta_e"]["q3"] + fence * fences["delta_e"]["iqr"]
    thresholds = Thresholds(round(iou_min, 4), round(delta_e_max, 3), round(ssim_min, 4), min_pixels)
    rule = (f"Tukey fences with k = {fence} over {len(judged)} regions with at least {min_pixels} covered pixels: "
            f"iou_min = Q1 - k*IQR, ssim_min = Q1 - k*IQR, delta_e_max = Q3 + k*IQR")
    distributions = {
        "iou": {"summary": summarize(columns["iou"]), "histogram": histogram(columns["iou"], IOU_EDGES)},
        "delta_e": {"summary": summarize(columns["delta_e"]), "histogram": histogram(columns["delta_e"], DELTA_E_EDGES)},
        "ssim": {"summary": summarize(columns["ssim"]), "histogram": histogram(columns["ssim"], SSIM_EDGES)},
    }
    return Calibration(thresholds, len(judged), rule, distributions, fences)
