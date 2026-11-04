"""Per-pixel block faces of a rendered view, the alignment check between two renderings and the per-face metrics.

A pixel shows the surface its view ray hits first. The hit point lies in the block cell that was hit: when it is
within the depth error of a block plane (a cube face), it belongs to the cell on the far side of that plane; otherwise
it is pushed a little further along the ray, into the cell. The cell face through which the ray *enters* that cell is
the pixel's face. For a block of the scene's dump this is exactly "the block's six faces projected with the camera,
kept where the depth buffer says that block is the visible surface": a front-facing face's projection holds the
pixels whose rays enter the cell through it, and the depth decides whether the visible surface lies in the cell. It
also works for blocks that are not full cubes (slabs, pipes, plants): their pixels are grouped by the cell face their
rays enter through.

Both renderings are labelled from their own depth with the same (client) matrices, so a face's reference region is
where the client sees that block through that face and its test region is where BlueMap does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from e2e.oracle import metrics

FACES = ("down", "up", "north", "south", "west", "east")
# ray direction > 0 along the axis enters the cell through its negative face, < 0 through its positive face
_ENTRY = {0: (FACES.index("west"), FACES.index("east")), 1: (FACES.index("down"), FACES.index("up")),
          2: (FACES.index("north"), FACES.index("south"))}
_XZ_OFFSET = 1 << 20
_Y_OFFSET = 1 << 12
INSIDE_EPSILON = 2e-3  # blocks: push along the ray, and the least snapping distance to a block plane
SNAP_FLOOR = 4e-3  # blocks: hit points this close to a block plane (or closer than their push) lie on it
PLANAR_EPSILON = 1e-2  # blocks: a hit point this close to its entry plane lies on the cell face (full cube face)


def encode(cells: np.ndarray, face: np.ndarray) -> np.ndarray:
    """int64 keys of (cell x, y, z, face); x and z within +-2^20, y within +-2^12."""
    x = cells[..., 0].astype(np.int64) + _XZ_OFFSET
    z = cells[..., 2].astype(np.int64) + _XZ_OFFSET
    y = cells[..., 1].astype(np.int64) + _Y_OFFSET
    return (((x << 21 | z) << 13 | y) << 3) | face.astype(np.int64)


def decode(key: int) -> tuple[int, int, int, str]:
    face = key & 7
    y = (key >> 3) & ((1 << 13) - 1)
    z = (key >> 16) & ((1 << 21) - 1)
    x = key >> 37
    return int(x - _XZ_OFFSET), int(y - _Y_OFFSET), int(z - _XZ_OFFSET), FACES[face]


@dataclass
class Labels:
    """Per-pixel labelling of one rendering."""

    key: np.ndarray  # (H, W) int64, -1 = no surface
    residual: np.ndarray  # (H, W) distance (blocks) of the hit point from its entry plane, NaN = no surface
    distance: np.ndarray  # (H, W) distance (blocks) from the eye, NaN = no surface


def window_depth_epsilon(distance: np.ndarray, near: float, far: float, bits: int = 24) -> np.ndarray:
    """Depth tolerance of hit points read back from a ``bits``-bit window depth buffer: the default push plus four
    times the depth quantisation error ``z^2 (f - n) / (f n 2^bits)`` at view depth ``z``."""
    return INSIDE_EPSILON + 4.0 * distance ** 2 * (far - near) / (far * near * 2.0 ** bits)


def label(world: np.ndarray, valid: np.ndarray, eye: np.ndarray, epsilon: float | np.ndarray = INSIDE_EPSILON) -> Labels:
    """Labels the hit points ``world`` (H, W, 3) seen from ``eye`` (3,), where ``valid``. ``epsilon`` (blocks, scalar
    or per pixel) is the depth tolerance, see [window_depth_epsilon]: a hit point within ``max(epsilon, SNAP_FLOOR)``
    of a block plane is snapped to it (the plane nearest to it decides), others are pushed ``epsilon`` along the
    ray. Snapping keeps grazing views of cube faces in the right cell, where a push along the ray would barely move
    the point across the plane."""
    world = np.where(valid[..., None], world, eye)
    ray = world - eye
    distance = np.linalg.norm(ray, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ray = np.where(valid[..., None], ray / distance[..., None], 0.0)
    epsilon = np.broadcast_to(np.asarray(epsilon, dtype=np.float64), distance.shape)
    cells = np.floor(world + epsilon[..., None] * ray)
    offsets = world - np.round(world)
    nearest = np.argmin(np.abs(offsets), axis=-1)
    offset = np.take_along_axis(offsets, nearest[..., None], -1)[..., 0]
    snap = np.abs(offset) < np.maximum(epsilon, SNAP_FLOOR)
    plane_coord = np.round(np.take_along_axis(world, nearest[..., None], -1)[..., 0])
    toward = np.take_along_axis(ray, nearest[..., None], -1)[..., 0]
    snapped = np.where(toward > 0, plane_coord, plane_coord - 1.0)  # the cell beyond the plane, seen from the eye
    for axis in range(3):
        cells[..., axis] = np.where(snap & (nearest == axis), snapped, cells[..., axis])
    t = np.full(world.shape[:2] + (3,), -np.inf)
    plane = np.zeros_like(t)
    for axis in range(3):
        r = ray[..., axis]
        lower = cells[..., axis]
        candidate = np.where(r > 0, lower, lower + 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            t[..., axis] = np.where(r != 0, (candidate - eye[axis]) / r, -np.inf)
        plane[..., axis] = candidate
    axis = np.argmax(t, axis=-1)
    r_axis = np.take_along_axis(ray, axis[..., None], -1)[..., 0]
    negative = np.array([_ENTRY[a][0] for a in range(3)])[axis]
    positive = np.array([_ENTRY[a][1] for a in range(3)])[axis]
    face = np.where(r_axis > 0, negative, positive)
    hit_plane = np.take_along_axis(plane, axis[..., None], -1)[..., 0]
    hit_coord = np.take_along_axis(world, axis[..., None], -1)[..., 0]
    key = np.where(valid, encode(cells.astype(np.int64), face), -1)
    residual = np.where(valid, np.abs(hit_coord - hit_plane), np.nan)
    return Labels(key, residual, np.where(valid, distance, np.nan))


def group(keys: np.ndarray) -> dict[int, np.ndarray]:
    """Flat pixel indices per key (keys >= 0)."""
    flat = keys.reshape(-1)
    order = np.argsort(flat, kind="stable")
    sorted_keys = flat[order]
    start = np.searchsorted(sorted_keys, 0)
    order, sorted_keys = order[start:], sorted_keys[start:]
    if not len(sorted_keys):
        return {}
    boundaries = np.flatnonzero(np.diff(sorted_keys)) + 1
    return {int(sorted_keys[s]): idx for s, idx in zip(np.r_[0, boundaries], np.split(order, boundaries))}


def _shift(array: np.ndarray, dx: int, dy: int, fill) -> np.ndarray:
    """``array`` moved by (dx, dy) pixels (x right, y down), uncovered pixels set to ``fill``."""
    out = np.full_like(array, fill)
    h, w = array.shape
    ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
    xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
    out[yd, xd] = array[ys, xs]
    return out


def boundaries(keys: np.ndarray) -> np.ndarray:
    """Pixels whose right or lower neighbour has another label (both labelled or one of them unlabelled)."""
    edge = np.zeros(keys.shape, dtype=bool)
    edge[:, :-1] |= keys[:, :-1] != keys[:, 1:]
    edge[:-1, :] |= keys[:-1, :] != keys[1:, :]
    return edge


def _distance_to(mask: np.ndarray, limit: int) -> np.ndarray:
    """Chessboard distance (pixels) to the nearest ``mask`` pixel, capped at ``limit``."""
    distance = np.full(mask.shape, limit, dtype=np.int32)
    reached = mask.copy()
    distance[reached] = 0
    for step in range(1, limit):
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown[1:, 1:] |= reached[:-1, :-1]
        grown[:-1, :-1] |= reached[1:, 1:]
        grown[1:, :-1] |= reached[:-1, 1:]
        grown[:-1, 1:] |= reached[1:, :-1]
        distance[grown & ~reached] = step
        reached = grown
    return distance


EDGE_LIMIT = 8  # pixels: farther edge distances are counted as this


@dataclass
class Alignment:
    shift: tuple[float, float]  # best global shift of the test labels onto the reference labels, sub-pixel
    agreement_at_zero: float  # share of the reference's dump-face pixels with the same label in the test
    agreement_at_shift: float
    edge_pixels: int  # reference block-edge pixels between full-cube faces
    edge_error_median: float  # pixels to the nearest test label edge
    edge_error_p90: float
    edge_error_mean: float
    reference_plane_residual_px: float  # median distance of the reference's cube-face hit points from their plane
    passed: bool

    def to_json(self) -> dict:
        def r(v):
            return None if isinstance(v, float) and math.isnan(v) else (round(v, 4) if isinstance(v, float) else v)

        return {"shift": [r(self.shift[0]), r(self.shift[1])], "agreement_at_zero": r(self.agreement_at_zero),
                "agreement_at_shift": r(self.agreement_at_shift), "edge_pixels": self.edge_pixels,
                "edge_error_median_px": r(self.edge_error_median), "edge_error_p90_px": r(self.edge_error_p90),
                "edge_error_mean_px": r(self.edge_error_mean),
                "reference_plane_residual_px": r(self.reference_plane_residual_px), "passed": self.passed}


# alignment must be within these before face metrics are trusted
MAX_SHIFT_PX = 1.0
MAX_EDGE_MEDIAN_PX = 1.0
MAX_EDGE_P90_PX = 2.0


def alignment(reference: Labels, test: Labels, in_dump: np.ndarray, pixels_per_block: np.ndarray,
              search: int = 3) -> Alignment:
    """How well ``test`` lines up with ``reference``.

    - Global shift: the integer shift of the test labels within +-``search`` px that maximises label agreement on the
      reference's dump-face pixels, refined to sub-pixel with a parabola through the neighbours.
    - Block-edge reprojection error: for reference edge pixels between two full-cube faces (faces whose hit points
      lie on their cell face plane), the distance to the nearest test label edge (chessboard px, capped at 8).
    - Reference self-consistency: median distance of cube-face hit points from their plane, in pixels.
    """
    ref_keys = np.where(in_dump, reference.key, -1)
    mask = ref_keys >= 0
    total = int(mask.sum())
    scores = np.zeros((2 * search + 1, 2 * search + 1))
    for iy, dy in enumerate(range(-search, search + 1)):
        for ix, dx in enumerate(range(-search, search + 1)):
            shifted = _shift(test.key, dx, dy, -2)
            scores[iy, ix] = (shifted[mask] == ref_keys[mask]).sum() / total if total else 0.0
    iy, ix = np.unravel_index(np.argmax(scores), scores.shape)

    def refine(values: np.ndarray, i: int) -> float:
        if 0 < i < len(values) - 1:
            a, b, c = values[i - 1], values[i], values[i + 1]
            denominator = a - 2 * b + c
            if denominator < 0:
                return i + 0.5 * (a - c) / denominator
        return float(i)

    shift = (refine(scores[iy, :], ix) - search, refine(scores[:, ix], iy) - search)

    planar = mask & (reference.residual < PLANAR_EPSILON)
    planar_keys = np.where(planar, reference.key, -1)
    edges = boundaries(planar_keys)
    # only edges between two planar dump faces: both neighbours labelled and planar
    both = np.zeros_like(edges)
    both[:, :-1] |= (planar_keys[:, :-1] >= 0) & (planar_keys[:, 1:] >= 0) & (planar_keys[:, :-1] != planar_keys[:, 1:])
    both[:-1, :] |= (planar_keys[:-1, :] >= 0) & (planar_keys[1:, :] >= 0) & (planar_keys[:-1, :] != planar_keys[1:, :])
    edges &= both
    distance = _distance_to(boundaries(test.key), EDGE_LIMIT)[edges].astype(np.float64)
    if distance.size:
        median, p90, mean = float(np.median(distance)), float(np.percentile(distance, 90)), float(distance.mean())
    else:
        median = p90 = mean = math.nan
    residual_px = reference.residual * pixels_per_block
    plane_px = float(np.median(residual_px[planar])) if planar.any() else math.nan
    passed = (bool(distance.size) and abs(shift[0]) <= MAX_SHIFT_PX and abs(shift[1]) <= MAX_SHIFT_PX
              and median <= MAX_EDGE_MEDIAN_PX and p90 <= MAX_EDGE_P90_PX)
    return Alignment(shift, float(scores[search, search]), float(scores[iy, ix]), int(edges.sum()), median, p90, mean,
                     plane_px, passed)


@dataclass
class FaceResult:
    key: int
    x: int
    y: int
    z: int
    face: str
    metrics: metrics.RegionMetrics
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive) of the union of both regions
    planar: float  # share of the region's pixels (reference, else test) on the cell face plane
    distance: float  # median distance (blocks) of those pixels from the eye


OCCLUSION_TOLERANCE = 0.02  # blocks (plus 0.2 % of the distance): a surface this much nearer occludes


def _region(own: Labels, other: Labels, own_index: np.ndarray) -> np.ndarray:
    """``own_index`` (each pixel's face in the compared keys, -1 = none) without the pixels where ``other`` shows a
    nearer surface of something else: occluded in ``other``, that difference belongs to the occluder's own face,
    which has these pixels in its region."""
    tolerance = OCCLUSION_TOLERANCE + 0.002 * own.distance
    with np.errstate(invalid="ignore"):
        nearer = other.distance < own.distance - tolerance  # NaN (nothing drawn) is never nearer
    return np.where((own_index >= 0) & ((other.key == own.key) | ~nearer), own_index, -1)


def sorted_keys(keys: np.ndarray) -> np.ndarray:
    """``keys`` as [index_of] wants them: sorted and unique (as they are, when they already are)."""
    keys = np.asarray(keys, dtype=np.int64).reshape(-1)
    return keys if keys.size < 2 or bool((keys[1:] > keys[:-1]).all()) else np.unique(keys)


def index_of(labels: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """The position of each of ``labels`` in ``keys`` ([sorted_keys]), -1 for the labels not in them: ``np.isin`` in
    a binary search, without sorting the labels (a view has a million) and the keys (a scene up to 1.4 million)."""
    if not keys.size:
        return np.full(labels.shape, -1, dtype=np.intp)
    position = np.minimum(np.searchsorted(keys, labels), keys.size - 1)
    return np.where(keys[position] == labels, position, -1)


def _surface(reference: Labels, test: Labels, ref_id: np.ndarray, test_id: np.ndarray,
             count: int) -> tuple[np.ndarray, np.ndarray]:
    """Per face: the share of its region's pixels on the cell face plane and their median distance from the eye, of
    the reference region, else (a face only BlueMap shows) of the test region."""
    ref_flat, test_flat = ref_id.reshape(-1), test_id.reshape(-1)
    ref_pixels = np.flatnonzero(ref_flat >= 0)
    test_pixels = np.flatnonzero(test_flat >= 0)
    test_pixels = test_pixels[np.bincount(ref_flat[ref_pixels], minlength=count)[test_flat[test_pixels]] == 0]
    ids = np.concatenate([ref_flat[ref_pixels], test_flat[test_pixels]])
    residual = np.concatenate([reference.residual.reshape(-1)[ref_pixels], test.residual.reshape(-1)[test_pixels]])
    distance = np.concatenate([reference.distance.reshape(-1)[ref_pixels], test.distance.reshape(-1)[test_pixels]])
    size = np.bincount(ids, minlength=count)
    planar = np.bincount(ids[residual < PLANAR_EPSILON], minlength=count) / size
    return planar, metrics.segment_median(distance[np.lexsort((distance, ids))], size)


def cell_keys(cells: np.ndarray) -> np.ndarray:
    """Keys of whole cells (face bits 0) of ``cells`` (..., 3): the cell part of [encode]."""
    return encode(cells, np.zeros(cells.shape[:-1], dtype=np.int64))


def decode_cells(keys: np.ndarray) -> np.ndarray:
    """The cells (n, 3) of face or cell keys, vectorised [decode]."""
    keys = np.asarray(keys, dtype=np.int64)
    return np.stack([(keys >> 37) - _XZ_OFFSET, ((keys >> 3) & ((1 << 13) - 1)) - _Y_OFFSET,
                     ((keys >> 16) & ((1 << 21) - 1)) - _XZ_OFFSET], axis=-1)


BEHIND_STEPS = 64  # cells a view ray is followed beyond a translucent surface


def behind(labels: Labels, world: np.ndarray, eye: np.ndarray, pixels: np.ndarray, cells: np.ndarray,
           see_through: np.ndarray, steps: int = BEHIND_STEPS) -> np.ndarray:
    """What a view ray shows through a translucent surface: for each labelled pixel of ``pixels`` (H, W bool), the
    cell key ([cell_keys]) of the first cell beyond the pixel's own cell along the ray that is in ``cells`` and not in
    ``see_through`` (cell keys; translucent cells in a row, such as a glass wall, are looked through), -1 when there is
    none within ``steps`` cells. ``world`` (H, W, 3) are the labelled hit points, ``eye`` the camera position."""
    out = np.full(labels.key.shape, -1, dtype=np.int64)
    cells, see_through = sorted_keys(cells), sorted_keys(see_through)
    index = np.flatnonzero(pixels & (labels.key >= 0))
    if not index.size:
        return out
    point = world.reshape(-1, 3)[index]
    ray = point - eye
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    cell = decode_cells(labels.key.reshape(-1)[index])
    step = np.sign(ray).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):  # Amanatides-Woo grid traversal
        t_next = np.maximum(np.where(ray != 0, (cell + (step > 0) - point) / ray, np.inf), 0.0)
        t_delta = np.where(ray != 0, 1.0 / np.abs(ray), np.inf)
    found = np.full(index.size, -1, dtype=np.int64)
    pending = np.arange(index.size)
    for _ in range(steps):
        axis = np.argmin(t_next[pending], axis=-1)
        cell[pending, axis] += step[pending, axis]
        t_next[pending, axis] += t_delta[pending, axis]
        key = cell_keys(cell[pending])
        hit = (index_of(key, cells) >= 0) & (index_of(key, see_through) < 0)
        found[pending[hit]] = key[hit]
        pending = pending[~hit]
        if not pending.size:
            break
    out.reshape(-1)[index] = found
    return out


def compare_faces(reference_rgba: np.ndarray, test_rgba: np.ndarray, reference: Labels, test: Labels,
                  keys: np.ndarray, colour_excluded: np.ndarray | None = None) -> list[FaceResult]:
    """Metrics of every face in ``keys`` that either rendering shows, in key order. The reference region is where the
    client shows the face, the test region where BlueMap does; a region's pixels where the *other* rendering shows a
    nearer surface of something else are left out (occluded there: that difference is the occluder's), so a face is
    judged on its own surface only. A face only one rendering shows has an empty region in the other (IoU 0). A
    face's metrics are [metrics.compare_region] of its two regions, all faces computed at once
    ([metrics.compare_regions]); its box is the bounding box of the union of both regions. ``colour_excluded``
    (H, W) pixels count for the coverage but not for the colour metrics."""
    keys = sorted_keys(keys)
    ref_index = _region(reference, test, index_of(reference.key, keys))
    test_index = _region(test, reference, index_of(test.key, keys))
    shown = np.zeros(keys.size + 1, dtype=bool)  # the last slot takes the -1s
    shown[ref_index] = True
    shown[test_index] = True
    shown = shown[:-1]
    found = keys[shown]
    if not found.size:
        return []
    count = found.size
    rank = np.cumsum(shown) - 1
    ref_id = np.where(ref_index >= 0, rank[ref_index], -1)
    test_id = np.where(test_index >= 0, rank[test_index], -1)
    region_metrics = metrics.compare_regions(reference_rgba, test_rgba, ref_id, test_id, count,
                                             colour_excluded=colour_excluded)
    y0, y1, x0, x1 = (b.tolist() for b in metrics.region_bounds(count, ref_id, test_id))
    planar, distance = _surface(reference, test, ref_id, test_id, count)
    cells = decode_cells(found).tolist()
    return [FaceResult(key, *cell, FACES[face], m, box, p, d) for key, cell, face, m, box, p, d in zip(
        found.tolist(), cells, (found & 7).tolist(), region_metrics, zip(x0, y0, x1, y1), planar.tolist(),
        distance.tolist())]
