"""Unit tests of the world oracle: camera conventions, unprojection, face labelling, alignment, depth decoding."""

from __future__ import annotations

import math

import numpy as np
import pytest

from e2e import scenes
from e2e.oracle import accepted, camera, faces, metrics, webapp, world

W, H = 160, 90
FAR = 384.0


def _client(spec: dict, depth: np.ndarray | None = None) -> camera.ClientView:
    projection, modelview = camera.expected_matrices(spec, W, H, FAR)
    image = np.zeros((H, W, 3), dtype=np.uint8)
    meta = {**spec, "near": 0.05, "far": FAR, "width": W, "height": H}
    return camera.ClientView("test", image, depth if depth is not None else np.ones((H, W), dtype=np.float32),
                             projection, modelview, np.asarray(spec["eye"], dtype=np.float64), meta)


def _rays(client: camera.ClientView) -> np.ndarray:
    near = camera.eye_from_linear_depth(client.projection, np.ones((H, W)))
    world = camera.eye_to_world(client.view, near)
    directions = world - client.eye
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def _raycast(eye: np.ndarray, rays: np.ndarray, cubes: list[tuple[int, int, int]]) -> np.ndarray:
    """Distance along each ray to the nearest unit cube (inf = miss)."""
    best = np.full(rays.shape[:2], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for cube in cubes:
            low = (np.asarray(cube, dtype=np.float64) - eye) / rays
            high = (np.asarray(cube, dtype=np.float64) + 1.0 - eye) / rays
            t_min = np.nanmax(np.minimum(low, high), axis=-1)
            t_max = np.nanmin(np.maximum(low, high), axis=-1)
            hit = (t_min <= t_max) & (t_min > 0)
            best = np.where(hit & (t_min < best), t_min, best)
    return best


def _window_depth(client: camera.ClientView, points: np.ndarray, hit: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate([points, np.ones(points.shape[:2] + (1,))], axis=-1)
    clip = homogeneous @ (client.projection @ client.view).T
    depth = (clip[..., 2] / clip[..., 3] + 1.0) / 2.0
    return np.where(hit, depth, 1.0).astype(np.float32)


CUBES = [(0, 5, 0), (2, 5, 0), (0, 5, 2), (2, 5, 2), (1, 4, 1)]
SPEC = {"eye": [-3.3, 10.2, -2.7], "fov": 70.0, **dict(zip(("yaw", "pitch"), camera.look_at((-3.3, 10.2, -2.7), (1.5, 5.5, 1.5))))}


def _scene() -> tuple[camera.ClientView, np.ndarray, np.ndarray]:
    probe = _client(SPEC)
    rays = _rays(probe)
    t = _raycast(probe.eye, rays, CUBES)
    hit = np.isfinite(t)
    points = probe.eye + rays * np.where(hit, t, 0.0)[..., None]
    return _client(SPEC, _window_depth(probe, points, hit)), points, hit


def test_look_at_and_direction_agree():
    eye, target = (3.0, 20.0, -7.0), (10.0, 4.0, 12.0)
    yaw, pitch = camera.look_at(eye, target)
    expected = np.subtract(target, eye) / np.linalg.norm(np.subtract(target, eye))
    assert np.allclose(camera.direction(yaw, pitch), expected)
    assert camera.look_at((0, 0, 0), (0, 0, 1)) == pytest.approx((0.0, 0.0))  # south
    assert camera.look_at((0, 0, 0), (-1, 0, 0))[0] == pytest.approx(90.0)  # west
    assert camera.look_at((0, 0, 0), (0, -1, 1e-9))[1] == pytest.approx(90.0)  # straight down


def test_webapp_views_become_minecraft_cameras():
    view = scenes.View(name="v", map="overworld", x=18, y=5, z=17, distance=45, rotation=0.6, angle=0.9)
    cam = camera.from_view(view, 70.0, (1280, 720))
    to_target = np.array([18, 5, 17]) - np.array(cam.eye)
    assert np.linalg.norm(to_target) == pytest.approx(45.0)
    assert np.allclose(camera.direction(cam.yaw, cam.pitch), to_target / 45.0)
    free = scenes.View(name="f", map="overworld", x=1, y=2, z=3, distance=0, rotation=math.pi, angle=1.35, mode="free")
    cam = camera.from_view(free, 70.0, (1280, 720))
    assert cam.eye == (1.0, 2.0, 3.0)
    assert cam.yaw == pytest.approx(0.0) or cam.yaw == pytest.approx(360.0)  # rotation pi looks south
    assert cam.pitch == pytest.approx(90.0 - math.degrees(1.35))


def test_scene_cameras_skip_orthographic_views_and_add_listed_ones():
    cams = {c.name: c for c in camera.scene_cameras("vanilla-zoo")}
    assert "top-flat" not in cams and "perspective" in cams and "south-west" in cams
    sw = cams["south-west"]
    assert np.allclose(camera.direction(sw.yaw, sw.pitch),
                       (np.array([18, 5, 17]) - sw.eye) / np.linalg.norm(np.array([18, 5, 17]) - sw.eye))
    assert (sw.width, sw.height, sw.fov) == (1280, 720, 70.0)


def test_excluded_states_of_the_gtnh_zoo():
    states = camera.excluded_states("gtnh-zoo")
    assert camera.is_excluded(states, "Natura:Natura.workbench", 14)
    assert not camera.is_excluded(states, "Natura:Natura.workbench", 0)
    assert not camera.is_excluded(states, "minecraft:stone", 0)
    assert camera.excluded_states("vanilla-zoo") == {}
    assert camera.is_excluded({"a:b": None}, "a:b", 7)


def test_look_target_projects_to_the_image_centre():
    client = _client(SPEC)
    centre = camera.project(client.view, client.projection, W, H, np.array([[1.5, 5.5, 1.5]]))[0]
    assert centre == pytest.approx((W / 2, H / 2), abs=1e-6)


def test_window_depth_unprojects_to_the_hit_points():
    client, points, hit = _scene()
    world = camera.eye_to_world(client.view, camera.eye_from_window_depth(client.projection, client.depth))
    assert hit.mean() > 0.02
    assert np.abs(world[hit] - points[hit]).max() < 1e-3


def test_linear_depth_unprojects_to_the_hit_points():
    client, points, hit = _scene()
    eye_points = (np.concatenate([points, np.ones((H, W, 1))], axis=-1) @ client.view.T)[..., :3]
    world = camera.eye_to_world(client.view, camera.eye_from_linear_depth(client.projection, -eye_points[..., 2]))
    assert np.abs(world[hit] - points[hit]).max() < 1e-9


def test_labels_name_the_hit_cube_and_its_visible_faces():
    client, points, hit = _scene()
    labels = faces.label(points, hit, client.eye)
    keys, counts = np.unique(labels.key[labels.key >= 0], return_counts=True)
    seen = {faces.decode(int(k)): int(n) for k, n in zip(keys, counts)}
    # rays grazing a cube edge may land in the neighbouring cell: at most a stray pixel per silhouette
    assert sum(n for s, n in seen.items() if s[:3] in CUBES) >= 0.99 * hit.sum()
    # the eye is north-west of and above every cube: only up, north and west faces can face it
    assert {s[3] for s in seen if s[:3] in CUBES} <= {"up", "north", "west"}
    assert {(0, 5, 0, "up"), (0, 5, 0, "north"), (0, 5, 0, "west")} <= set(seen)
    assert np.nanpercentile(labels.residual, 99) < 1e-6  # full cubes: hit points lie on their entry face plane


def test_encode_decode_roundtrip():
    cells = np.array([[-5, 3, 1000000], [123456, -40, -7], [0, 255, 0]])
    for cell in cells:
        for f in range(6):
            assert faces.decode(int(faces.encode(cell, np.array(f)))) == (*map(int, cell), faces.FACES[f])


def test_alignment_passes_identical_labels_and_reports_a_shift():
    client, points, hit = _scene()
    labels = faces.label(points, hit, client.eye)
    in_dump = labels.key >= 0
    ppb = np.full((H, W), 10.0)
    same = faces.alignment(labels, labels, in_dump, ppb)
    assert same.passed and same.shift == pytest.approx((0.0, 0.0)) and same.edge_error_median == 0
    shifted_key = faces._shift(labels.key, 2, 0, -1)
    shifted = faces.Labels(shifted_key, labels.residual, labels.distance)
    moved = faces.alignment(labels, shifted, in_dump, ppb)
    assert not moved.passed
    assert moved.shift[0] == pytest.approx(-2.0, abs=0.5)


def test_compare_faces_of_identical_renderings():
    client, points, hit = _scene()
    labels = faces.label(points, hit, client.eye)
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    rgba[..., 0] = (labels.key % 200).astype(np.uint8)
    keys = np.unique(labels.key[labels.key >= 0])
    results = faces.compare_faces(rgba, rgba, labels, labels, keys)
    assert len(results) == len(keys)
    assert all(r.metrics.iou == 1.0 and r.metrics.delta_e == 0.0 for r in results)
    empty = faces.Labels(np.full_like(labels.key, -1), labels.residual, labels.distance)
    missing = faces.compare_faces(rgba, rgba, labels, empty, keys)
    assert all(r.metrics.iou == 0.0 and r.metrics.test_pixels == 0 for r in missing)


def _cube_distances(eye: np.ndarray, rays: np.ndarray, cubes: list) -> np.ndarray:
    """(len(cubes), H, W) distance along each ray to each cube (inf = miss)."""
    return np.stack([_raycast(eye, rays, [c]) for c in cubes])


@pytest.mark.parametrize("see_through", [[(0, 5, 0)], [(0, 5, 0), (1, 4, 1)]])
def test_behind_finds_the_first_opaque_cell_beyond_a_translucent_one(see_through):
    client, points, hit = _scene()
    labels = faces.label(points, hit, client.eye)
    own = np.where(labels.key >= 0, labels.key & ~np.int64(7), -1)
    translucent_keys = faces.cell_keys(np.asarray(see_through))
    pixels = np.isin(own, translucent_keys)
    found = faces.behind(labels, points, client.eye, pixels, faces.cell_keys(np.asarray(CUBES)), translucent_keys)
    # brute force: the nearest cube along the ray that is not translucent and lies beyond the visible surface
    opaque = [c for c in CUBES if c not in see_through]
    t = _cube_distances(client.eye, _rays(client), opaque)
    t_hit = np.linalg.norm(points - client.eye, axis=-1)
    t = np.where(t > t_hit + 1e-6, t, np.inf)
    nearest = np.argmin(t, axis=0)
    expected = np.where(np.isfinite(t.min(axis=0)), faces.cell_keys(np.asarray(opaque))[nearest], -1)
    assert pixels.sum() > 100
    assert (found[pixels] == expected[pixels]).mean() > 0.98
    # (1, 4, 1) is behind part of (0, 5, 0); looked through as well, nothing is behind
    assert (found[pixels] >= 0).any() == ((1, 4, 1) not in see_through) and (found[pixels] < 0).any()
    assert (found[~pixels] == -1).all()


def test_colour_excluded_pixels_count_for_coverage_only():
    ref = np.zeros((32, 32, 4), dtype=np.uint8)
    ref[..., 3] = 255
    test = ref.copy()
    test[:, :16, :3] = 200  # the left half differs in colour
    region = np.ones((32, 32), bool)
    plain = metrics.compare_region(ref, test, region)
    excluded = np.zeros((32, 32), bool)
    excluded[:, :16] = True
    m = metrics.compare_region(ref, test, region, colour_excluded=excluded)
    assert plain.delta_e > 10 and m.delta_e == 0.0 and m.ssim == pytest.approx(1.0)
    assert m.iou == plain.iou == 1.0 and m.both_pixels == plain.both_pixels == 32 * 32
    everything = metrics.compare_region(ref, test, region, colour_excluded=np.ones((32, 32), bool))
    assert math.isnan(everything.delta_e) and math.isnan(everything.ssim) and everything.iou == 1.0


def test_depth_pass_decoding_matches_the_shader_packing():
    depth = np.array([0.0, 0.5, 3.14159, 17.000061, 250.25, 1023.99])
    i = np.floor(depth)
    f = depth - i
    rgba = np.stack([np.floor(i / 4), np.mod(i, 4) * 64 + np.floor(f * 64), np.floor(np.mod(f * 64, 1) * 256),
                     np.full_like(depth, 255)], axis=-1).astype(np.uint8)[None]
    decoded = webapp.decode_depth(rgba)[0]
    assert np.abs(decoded - depth).max() <= 1 / 16384
    assert np.isnan(webapp.decode_depth(np.zeros((1, 1, 4), dtype=np.uint8))[0, 0])


def test_webapp_camera_reproduces_the_client_view():
    client = _client(SPEC)
    cam = camera.webapp_camera(client)
    x, y, z, w = cam["quaternion"]
    rotation = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    world = np.eye(4)
    world[:3, :3] = rotation
    world[:3, 3] = cam["position"]
    assert np.allclose(np.linalg.inv(world), client.view, atol=1e-9)
    assert np.allclose(np.asarray(cam["projection"]).reshape(4, 4).T, client.projection)
    assert cam["fov"] == pytest.approx(70.0) and cam["aspect"] == pytest.approx(W / H)


def _face(block: str, meta: int, iou: float) -> dict:
    return {"view": "v", "block": block, "meta": meta, "label": f"{block}:{meta}", "mod": block.split(":")[0],
            "ref_pixels": 100, "test_pixels": 100, "both_pixels": round(100 * iou), "iou": iou, "delta_e": 0.5,
            "delta_e_p95": 1.0, "ssim": 0.99, "magenta": 0.0, "face": "up"}


def test_judge_reports_accepted_faces_apart(tmp_path):
    file = tmp_path / "accepted.yaml"
    file.write_text("reasons: {random_variant: r}\naccepted:\n  - {states: 'minecraft:stone[meta=0]', reason: random_variant}\n")
    states = {"minecraft:stone[meta=0]": {"name": "minecraft:stone", "meta": 0, "textures": {}, "randomOffset": False,
                                          "randomVariant": True, "flatteningDrop": False}}
    accepted_list = accepted.load({"states": states}, set(), file)
    view = world.ViewOutcome("v", "perspective", [_face("minecraft:stone", 0, 0.5), _face("mod:x", 0, 1.0),
                                                  _face("mod:y", 0, 0.5)], {"passed": True})
    world.judge([view], metrics.Thresholds(0.9, 2.0, 0.9, 64), accepted_list)
    stone, good, bad = view.faces
    assert stone["accepted"].startswith("accepted (1.13)") and stone["failed"] == [] and stone["accepted_checks"]
    assert good["passed"] and not bad["passed"] and bad["failed"]
    s = world.summarize([view])
    assert (s["judged"], s["passed"], s["accepted"]) == (2, 1, 1)
    assert s["per_mod"]["minecraft"]["accepted"] == 1 and s["per_mod"]["minecraft"]["pass_rate"] is None
