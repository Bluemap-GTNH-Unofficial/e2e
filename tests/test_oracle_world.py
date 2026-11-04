"""Unit tests of the world oracle: camera conventions, unprojection, face labelling, alignment, depth decoding,
judging schedule."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import astuple
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from e2e import metrics as image_metrics
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


def test_compare_faces_is_compare_region_of_each_face():
    client, points, hit = _scene()
    reference = faces.label(points, hit, client.eye)
    # BlueMap's labels a pixel to the right, its left third a block nearer: occlusion in both directions
    test = faces.Labels(faces._shift(reference.key, 1, 0, -1), faces._shift(reference.residual, 1, 0, np.nan),
                        faces._shift(reference.distance, 1, 0, np.nan))
    test.distance[:, :W // 3] -= 1.0
    rng = np.random.default_rng(5)
    ref_rgba = np.concatenate([rng.integers(0, 256, (H, W, 3)), np.full((H, W, 1), 255)], axis=-1).astype(np.uint8)
    test_rgba = ref_rgba.copy()
    test_rgba[..., :3] = np.clip(ref_rgba[..., :3] + rng.integers(-15, 15, (H, W, 3)), 0, 255)
    excluded = rng.random((H, W)) < 0.1
    keys = np.unique(reference.key[reference.key >= 0])
    results = faces.compare_faces(ref_rgba, test_rgba, reference, test, keys[::-1], excluded)  # any key order
    regions = {}
    for key in keys.tolist():
        own_regions = []
        for own, other in ((reference, test), (test, reference)):
            with np.errstate(invalid="ignore"):
                nearer = other.distance < own.distance - (faces.OCCLUSION_TOLERANCE + 0.002 * own.distance)
            own_regions.append((own.key == key) & ((other.key == key) | ~nearer))
        if own_regions[0].any() or own_regions[1].any():
            regions[key] = own_regions
    assert [r.key for r in results] == list(regions) and len(results) > 10
    for r in results:
        ref_region, test_region = regions[r.key]
        expected = metrics.compare_region(ref_rgba, test_rgba, ref_region, test_region, colour_excluded=excluded)
        assert [np.float64(v).tobytes() for v in astuple(r.metrics)] == [np.float64(v).tobytes()
                                                                         for v in astuple(expected)]
        rows, cols = np.nonzero(ref_region | test_region)
        assert r.bbox == (cols.min(), rows.min(), cols.max() + 1, rows.max() + 1)
        source, labels = (ref_region, reference) if ref_region.any() else (test_region, test)
        assert r.planar == float((labels.residual[source] < faces.PLANAR_EPSILON).mean())
        assert r.distance == float(np.median(labels.distance[source]))
        assert (r.x, r.y, r.z, r.face) == faces.decode(r.key)


def test_face_records_count_magenta_and_excluded_pixels_in_the_face_box():
    client, points, hit = _scene()
    reference = faces.label(points, hit, client.eye)
    test = faces.Labels(faces._shift(reference.key, 1, 0, -1), faces._shift(reference.residual, 1, 0, np.nan),
                        faces._shift(reference.distance, 1, 0, np.nan))
    test.distance[:, :W // 3] -= 1.0
    rng = np.random.default_rng(9)
    client.image[...] = rng.integers(0, 256, (H, W, 3))
    test_rgb = rng.integers(0, 256, (H, W, 3)).astype(np.uint8)
    test_rgb[rng.random((H, W)) < 0.2] = (200, 0, 200)  # missing texture
    excluded = rng.random((H, W)) < 0.2
    blocks = {cube: world.Block(f"test:block{i}", i, None) for i, cube in enumerate(CUBES)}
    records = world.face_records(client, reference, test, test_rgb, blocks, world.dump_keys(blocks), excluded)
    magenta = image_metrics.magenta_mask(test_rgb)
    assert len(records) > 10 and any(r["magenta"] for r in records) and any(r["background_excluded"] for r in records)
    for r in records:
        key = int(faces.encode(np.array([r["x"], r["y"], r["z"]]), np.array(faces.FACES.index(r["face"]))))
        x0, y0, x1, y1 = r["bbox"]
        box = (slice(y0, y1), slice(x0, x1))
        test_region = test.key[box] == key  # occluded pixels too
        assert r["magenta"] == round(float(magenta[box][test_region].mean()) if test_region.any() else 0.0, 3)
        assert r["background_excluded"] == int((excluded[box] & ((reference.key[box] == key) | test_region)).sum())
        assert (r["block"], r["meta"]) == (blocks[r["x"], r["y"], r["z"]].name, blocks[r["x"], r["y"], r["z"]].meta)


def test_see_through_labels_read_each_render_once_per_capture(tmp_path, monkeypatch):
    monkeypatch.setattr(world.variants, "oracle_dir", lambda v: tmp_path / (v if isinstance(v, str) else v.name))
    client = tmp_path / "default" / "states" / "client"
    client.mkdir(parents=True)
    alpha = {"glass": 120, "stone": 255}
    for name, a in alpha.items():
        Image.new("RGBA", (8, 8), (90, 90, 90, a)).save(client / f"{name}.png")
    states_index = [{"name": f"test:{n}", "meta": 0, "path": n} for n in alpha]
    (client / "index.json").write_text(json.dumps({"states": states_index}))
    variant = SimpleNamespace(name="default")
    labels = {"test:glass:0", "test:stone:0", "test:unknown:0"}
    assert world.see_through_labels(variant, labels) == {"test:glass:0"}
    (client / "glass.png").unlink()  # the verdicts of this capture are kept
    assert world.see_through_labels(variant, labels) == {"test:glass:0"}
    (client / "index.json").write_text(json.dumps({"states": states_index[1:]}))  # another capture: read again
    assert world.see_through_labels(variant, labels) == set()


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


class _ThreadPool(ThreadPoolExecutor):
    """The judge processes as threads: the fakes below cannot be pickled to a spawned process."""

    def __init__(self, workers: int, mp_context=None, initializer=None) -> None:
        super().__init__(workers)


class _FakeCli:
    def __init__(self, group: world.RenderGroup) -> None:
        self.url = f"http://{group.variant.name}/"
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def _schedule(tmp_path, monkeypatch, judge, clis: list[_FakeCli]) -> tuple[list, list[str]]:
    """``judge_parallel`` over two groups with fake CLIs (appended to ``clis``) and judge jobs: default (vanilla-zoo,
    which calibrates, and gt-ores), modernity (vanilla-zoo, and gt-pipes without a reference)."""
    default, modernity = SimpleNamespace(name="default"), SimpleNamespace(name="modernity")
    client = tmp_path / "client"
    client.mkdir()
    for name in ("a", "b", "c"):
        (client / f"{name}.json").write_text("{}")
    cams = [SimpleNamespace(name=n, source="perspective") for n in ("a", "b", "c", "missing")]
    key = SimpleNamespace(key=SimpleNamespace(short="k"), log=None, seconds=None)

    def run(variant, scene: str) -> world.SceneRun:
        return world.SceneRun(variant, scene, tmp_path / variant.name / scene, cams, {}, [], [], key, [])

    runs = [run(default, "vanilla-zoo"), run(default, "gt-ores"), run(modernity, "vanilla-zoo"),
            run(modernity, "gt-pipes")]
    groups = [world.RenderGroup(default, [], runs[:2], tmp_path), world.RenderGroup(modernity, [], runs[2:], tmp_path)]
    client_dirs = {id(r): None if r.scene == "gt-pipes" else client for r in runs}
    closed: list = []

    def start(group, jars):
        clis.append(_FakeCli(group))
        return clis[-1]

    def close(s, plan, summaries, failed):
        closed.append((s.run.variant.name, s.run.scene, [(v.name, v.error) for v in s.views], s.cli.stopped))

    monkeypatch.setattr(world, "ProcessPoolExecutor", _ThreadPool)
    monkeypatch.setattr(world, "start_bluemap", start)
    monkeypatch.setattr(world, "_judge_process_view", judge)
    monkeypatch.setattr(world, "close_scene", close)
    failed: list[str] = []
    world.judge_parallel(groups, client_dirs, None, world.CalibrationPlan(True, None, None), {}, failed, 4)
    return closed, failed


def _judged(job: world.ViewJob) -> tuple[world.ViewOutcome, float]:
    alignment = {"passed": True, "shift": [0.0, 0.0], "edge_error_median_px": 0.0, "edge_error_p90_px": 0.0}
    return world.ViewOutcome(job.camera, job.source, alignment=alignment, seconds={"webapp": 0, "compare": 0}), 0.0


def test_parallel_judging_reports_the_calibration_scene_first(tmp_path, monkeypatch):
    def judge(job):  # the calibration scene's views are the slowest: the others are done first and wait
        time.sleep(0.3 if (job.variant.name, job.scene) == ("default", "vanilla-zoo") else 0.01)
        return _judged(job)

    clis: list[_FakeCli] = []
    reports, failed = _schedule(tmp_path, monkeypatch, judge, clis)
    assert [(variant, scene) for variant, scene, _, _ in reports][0] == ("default", "vanilla-zoo")
    assert sorted((variant, scene) for variant, scene, _, _ in reports) == [
        ("default", "gt-ores"), ("default", "vanilla-zoo"), ("modernity", "vanilla-zoo")]
    for _, _, views, stopped in reports:  # camera order; the CLI serves until its group's last scene is reported
        assert views == [("a", None), ("b", None), ("c", None), ("missing", "no reference capture")]
        assert not stopped
    assert failed == ["modernity/gt-pipes"]
    assert len(clis) == 2 and all(cli.stopped for cli in clis)


def test_parallel_judging_stops_every_cli_when_a_view_fails(tmp_path, monkeypatch):
    def judge(job):
        if job.camera == "b":
            raise RuntimeError("compare failed")
        return _judged(job)

    clis: list[_FakeCli] = []
    with pytest.raises(RuntimeError, match="compare failed"):
        _schedule(tmp_path, monkeypatch, judge, clis)
    assert clis and all(cli.stopped for cli in clis)


def test_update_summary_reads_each_report_once(tmp_path, monkeypatch):
    monkeypatch.setattr(world.variants, "oracle_dir", lambda variant: tmp_path)
    for scene in ("a", "b"):
        (tmp_path / "worlds" / scene).mkdir(parents=True)
        (tmp_path / "worlds" / scene / "report.json").write_text(json.dumps(
            {"scene": scene, "summary": {"passed": 1}, "timings": {"total": 2.0}, "created": "then", "faces": []}))
    known = {tmp_path / "worlds" / "b" / "report.json": world.summary_entry(
        {"scene": "b", "summary": {"passed": 5}, "timings": {"total": 1.0}, "created": "now"})}
    variant = SimpleNamespace(name="default")
    world.update_summary(variant, known)
    (tmp_path / "worlds" / "a" / "report.json").write_text("not read again")
    path = world.update_summary(variant, known)
    data = json.loads(path.read_text())
    assert data["scenes"] == {"a": {"summary": {"passed": 1}, "timings": {"total": 2.0}, "created": "then"},
                              "b": {"summary": {"passed": 5}, "timings": {"total": 1.0}, "created": "now"}}


def test_jobs_option():
    parser = argparse.ArgumentParser()
    world.add_parser(parser.add_subparsers())
    assert parser.parse_args(["world"]).jobs == world.JOBS >= 1
    assert parser.parse_args(["world", "--jobs", "1"]).jobs == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["world", "--jobs", "0"])
