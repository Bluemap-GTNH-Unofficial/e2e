"""`e2e oracle world`: BlueMap's webapp against screenshots of the real GTNH client, per visible block face.

Per scene and variant (outputs in ``out/oracle/<variant>/worlds/<scene>/``, never committed):

1. ``cameras.json``: the scene's oracle cameras (``oracle.camera``).
2. Reference: ``client/`` links to the client's captures in the reference cache (``oracle.reference``), keyed by
   everything the client sees (pack version, client mods and packs, scene content, cameras, capture settings). The
   misses of all scenes of a variant are captured with one ``gtnh-client cameras --jobs`` (see the launcher README).
3. BlueMap: every scene of a variant rendered by one BlueMap CLI process with the variant's packs (one map per scene,
   ``worlds/_bluemap/``), served, and captured at the client's exact matrices with the capture conditions of the
   client (``oracle.webapp``): ``bluemap/<view>.png`` and ``bluemap/<view>.depth.bin`` (float32 view-space depth,
   NaN = nothing drawn). ``--jobs N`` judge processes, each with its own headless Chromium, capture and compare the
   views of every scene of the run at once (the next variant renders meanwhile); ``--jobs 1`` judges one view after
   another in this process.
4. Faces: both renderings are labelled per pixel with the dump block and cell face they show (``oracle.faces``);
   alignment is checked first (block-edge reprojection error, global shift), then every face of a dump block that
   either rendering shows is compared with the shared region metrics (``oracle.metrics``).
5. ``report.json`` and ``report.html`` (worst faces first, grouped by mod, with crops in ``report/``).

Pass thresholds come from a calibration on the faces of ``vanilla-zoo`` (default variant) whose vanilla states look
the same in 1.7.10 and 1.13 (``oracle.vanilla``, from the default state run): Tukey's far-out fences over their
inexact samples, never tuned by hand. The calibration is stored in the reference cache, keyed by the references it
was computed from (vanilla-zoo's world reference and the default variant's state reference), and reused until they
change or ``--recalibrate``; ``out/oracle/default/worlds/calibration.json`` is the active copy. Faces of states
``oracle/accepted.yaml`` names (when the data backs the reason) are reported as ``accepted (1.13)``, neither passing
nor failing. ``--rejudge`` / ``--report-dir`` judge an existing ``report.json`` again without capturing or rendering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import multiprocessing.util
import os
import shutil
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import sync_playwright

from e2e import bluemap, paths, runner, scenes
from e2e import metrics as image_metrics
from e2e.browser import CHROMIUM_ARGS
from e2e.oracle import camera as cameras
from e2e.oracle import accepted, faces, metrics, reference, vanilla, variants, webapp, world_report

CALIBRATION_SCENE = "vanilla-zoo"
CALIBRATION_VARIANT = "default"
MIN_PIXELS = 64  # faces whose coverage union is smaller are reported but not judged
CROPS = 60  # worst faces per scene that get a crop image
# judge processes (--jobs): one per 4 CPUs, a judge's Chromium (SwiftShader) and comparison keep about 4 busy. On the
# 32-CPU container the default variant (81 views) took 291 s with 8, 302 s with 16, 426 s with 4 (README, run time)
JOBS = max(1, (os.cpu_count() or 4) // 4)


def calibration_file() -> Path:
    return variants.oracle_dir(CALIBRATION_VARIANT) / "worlds" / "calibration.json"


def scene_dir(variant: variants.Variant, scene: str) -> Path:
    return variants.oracle_dir(variant) / "worlds" / scene


@dataclass
class Block:
    name: str
    meta: int
    te: dict | None

    @property
    def mod(self) -> str:
        return self.name.split(":", 1)[0]

    @property
    def label(self) -> str:
        """`modid:name:meta`, plus `#<mID>` for GregTech meta tile entities."""
        label = f"{self.name}:{self.meta}"
        if self.te and "mID" in self.te:
            label += f"#{self.te['mID']}"
        return label


def load_dump(scene: str) -> dict[tuple[int, int, int], Block]:
    path = paths.scenes_dir() / scene / "dump.ndjson"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; build the scene with "
                                f"`uv run orch --pack-version {paths.pack().version} scene {scene}`")  # fmt: skip
    blocks = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            entry = json.loads(line)
            if entry["name"] != "minecraft:air":
                blocks[(entry["x"], entry["y"], entry["z"])] = Block(entry["name"], entry["meta"], entry.get("te"))
    return blocks


def dump_keys(blocks: dict[tuple[int, int, int], Block]) -> np.ndarray:
    """The face keys of the dump's cells (six per cell), sorted: [faces.index_of] looks labels up in them."""
    cells = np.asarray(list(blocks), dtype=np.int64)
    return np.sort(np.concatenate([faces.encode(cells, np.full(len(cells), f)) for f in range(len(faces.FACES))]))


# ---------------------------------------------------------------- scenes of a run


def oracle_scenes(variant: variants.Variant, only: list[str] | None = None) -> list[str]:
    """The scenes ``--all`` runs for a variant: every scene with oracle cameras and a dump whose packs fit the variant
    (scenes on the addons export belong to ``addons``, the others to ``default`` and ``modernity``); with ``only``
    (``--all --scene ...``) just those of them, the others printed as skipped."""
    found = []
    for name in scenes.names():
        if only is not None and name not in only:
            continue
        spec = scenes.load(name)
        if (variant.pack.addons_export in spec.extra_packs) != (variant.name == "addons"):
            if only is not None:
                print(f"[oracle] --all: {variant.name}/{name} skipped, a scene of another variant", flush=True)
            continue
        if not (paths.scenes_dir() / name / "dump.ndjson").is_file():
            print(f"[oracle] --all: {variant.name}/{name} skipped, no dump in {paths.scenes_dir() / name}", flush=True)
            continue
        try:
            cameras.scene_cameras(name)
        except ValueError as e:
            print(f"[oracle] --all: {variant.name}/{name} skipped ({e})", flush=True)
            continue
        found.append(name)
    for name in sorted(set(only or ()) - set(scenes.names())):
        print(f"[oracle] --all: {name} skipped, no scenes/{name}.yaml", flush=True)
    return found


@dataclass
class SceneRun:
    """One scene of one variant in a run: cameras, dump and reference."""

    variant: variants.Variant
    scene: str
    out: Path
    camera_list: list[cameras.Camera]
    blocks: dict
    removed: list
    removed_states: list[str]
    job: reference.WorldJob
    packs: list[Path]

    @property
    def calibrating_pair(self) -> bool:
        return self.variant.name == CALIBRATION_VARIANT and self.scene == CALIBRATION_SCENE


def prepare(variant: variants.Variant, scene: str, inputs: reference.Inputs) -> SceneRun:
    """Writes ``cameras.json`` and computes the scene's reference key."""
    out = scene_dir(variant, scene)
    out.mkdir(parents=True, exist_ok=True)
    (out / "client.log").unlink(missing_ok=True)  # launcher log of the pre-cache captures; now under oracle-ref/logs
    camera_list = cameras.scene_cameras(scene)
    blocks, removed, removed_states = scene_blocks(scene)
    data = cameras.cameras_json(scene, camera_list, removed)
    cameras_file = out / "cameras.json"
    cameras_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    spec = scenes.load(scene)
    if not spec.world.is_dir():
        raise FileNotFoundError(f"scene world {spec.world} does not exist; build it with `orch scene {scene}`")
    job = reference.WorldJob(variant, scene, spec.world, cameras_file, data, inputs.world(variant, scene, data))
    packs = list(dict.fromkeys([*spec.extra_packs, *variant.bluemap_packs]))
    return SceneRun(variant, scene, out, camera_list, blocks, removed, removed_states, job, packs)


# ---------------------------------------------------------------- comparison


@dataclass
class ViewOutcome:
    name: str
    source: str
    faces: list[dict] = field(default_factory=list)
    alignment: dict | None = None
    checks: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    error: str | None = None


def _rgba(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb[..., :3], np.full(rgb.shape[:2] + (1,), 255, dtype=np.uint8)], axis=-1)


def suspect(face: dict) -> str:
    """A first guess at why a face failed, from its metrics and pixels (to be confirmed by looking at the crop)."""
    if face["test_pixels"] == 0:
        return "BlueMap draws nothing there (missing block, invisible model, or a surface behind it shows instead)"
    if face["ref_pixels"] == 0:
        return "BlueMap draws a surface the client does not (shape too large / full cube for a smaller block)"
    reasons = []
    if face["iou"] is not None and face["failed_iou"]:
        reasons.append("shape differs (bounds, connections, model variant or facing)")
    if face["magenta"] > 0.05:
        reasons.append("missing texture (magenta)")
    if face["failed_delta_e"] and face["ssim"] is not None and not face["failed_ssim"]:
        reasons.append("same structure, other colour (tint, brightness, overlay or colour multiplier)")
    elif face["failed_delta_e"] or face["failed_ssim"]:
        reasons.append("different texture (wrong icon, rotation/flip, CTM or animation frame)")
    return "; ".join(reasons) or "borderline"


def _face_record(view: str, result: faces.FaceResult, block: dict, magenta: float) -> dict:
    """A face's measurements; ``judge`` gives the verdict (``judged`` ... ``suspect``) with the run's thresholds."""
    m = result.metrics
    return {"view": view, "x": result.x, "y": result.y, "z": result.z, "face": result.face, **block,
            **m.to_json(), "planar": round(result.planar, 3), "distance": round(result.distance, 2),
            "bbox": list(result.bbox), "magenta": round(magenta, 3), "judged": m.union_pixels >= MIN_PIXELS,
            "passed": False, "failed": [], "failed_iou": False, "failed_delta_e": False, "failed_ssim": False,
            "severity": 0.0, "suspect": ""}


def severity(m: metrics.RegionMetrics, t: metrics.Thresholds) -> float:
    """Sum of the relative exceedances of the failed criteria (worst faces first)."""
    total = max(0.0, t.iou_min - m.iou) / max(t.iou_min, 1e-6)
    if not math.isnan(m.delta_e):
        total += max(0.0, m.delta_e - t.delta_e_max) / max(t.delta_e_max, 1e-6)
    if not math.isnan(m.ssim):
        total += max(0.0, t.ssim_min - m.ssim) / max(t.ssim_min + 1.0, 1e-6)
    return round(total, 4)


def reference_labels(client: cameras.ClientView) -> tuple[faces.Labels, np.ndarray, np.ndarray]:
    """The client's labels, eye-space points and world points (pushed by the 24-bit depth buffer's error)."""
    eye_points = cameras.eye_from_window_depth(client.projection, client.depth)
    world = cameras.eye_to_world(client.view, eye_points)
    epsilon = faces.window_depth_epsilon(np.abs(eye_points[..., 2]), float(client.meta.get("near", 0.05)),
                                         float(client.meta["far"]))
    return faces.label(world, client.depth < 1.0, client.eye, epsilon), eye_points, world


def bluemap_labels(client: cameras.ClientView, distance: np.ndarray) -> tuple[faces.Labels, np.ndarray]:
    """BlueMap's labels and world points from its view-space depth (NaN = nothing drawn) and the client's matrices."""
    valid = ~np.isnan(distance)
    test_eye = cameras.eye_from_linear_depth(client.projection, np.where(valid, distance, 1.0))
    world = cameras.eye_to_world(client.view, test_eye)
    return faces.label(world, valid, client.eye), world


SEE_THROUGH_ALPHA = 250  # client state render: a covered pixel below this alpha (over black and white) is translucent
SEE_THROUGH_SHARE = 0.5  # a state is translucent when at least this share of its covered pixels is
SEE_THROUGH_READERS = 16  # threads reading state renders (gtnh-zoo's 16,000: 113 s one after another, 44 s with 16)


def _translucent_render(path: Path | None) -> bool:
    if path is None or not path.is_file():
        return False
    with Image.open(path) as image:
        alpha = np.asarray(image.convert("RGBA"))[..., 3]
    covered = alpha > 0
    return bool(covered.any() and (alpha[covered] < SEE_THROUGH_ALPHA).mean() >= SEE_THROUGH_SHARE)


def see_through_labels(variant: variants.Variant, labels: set[str]) -> set[str]:
    """The ``labels`` whose isolated client render of the state oracle (the variant's, else the default variant's)
    is translucent: at least ``SEE_THROUGH_SHARE`` of its covered pixels have an alpha below ``SEE_THROUGH_ALPHA``
    (stained glass, ice, water; cut-out blocks such as leaves and glass are opaque where they draw). The verdicts
    are kept per state capture (its reference key and ``index.json``) in ``see-through.json`` next to the world
    reports of the variant whose renders are read, and a run reads only the renders of labels without one: reading
    gtnh-zoo's 16,000 renders over the bind mount takes 44 s with ``SEE_THROUGH_READERS`` threads, and the BlueMap
    render it overlapped took 168 s instead of 63 s."""
    from e2e.oracle import states

    if not labels:
        return set()
    source: variants.Variant | str = variant
    client_dir = states.states_dir(source) / "client"
    if not (client_dir / "index.json").is_file():
        source = states.CALIBRATION_VARIANT
        client_dir = states.states_dir(source) / "client"
    if not (client_dir / "index.json").is_file():
        return set()
    index = (client_dir / "index.json").read_bytes()
    capture = f"{reference.recorded_key(states.states_dir(source))}:{hashlib.sha256(index).hexdigest()}"
    cache = variants.oracle_dir(source) / "worlds" / "see-through.json"
    known: dict[str, bool] = {}
    if cache.is_file():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if stored.get("capture") == capture:
            known = stored["translucent"]
    missing = [label for label in labels if label not in known]
    if missing:
        paths_by_label: dict[str, str] = {}
        for s in json.loads(index)["states"]:
            if s.get("error") or s.get("blank"):
                continue
            if s.get("mte") is not None:
                paths_by_label.setdefault(f"{s['name']}:{s.get('worldMeta', s['meta'])}#{s['mte']}", s["path"])
            else:
                paths_by_label.setdefault(f"{s['name']}:{s['meta']}", s["path"])
        renders = [client_dir / f"{paths_by_label[label]}.png" if label in paths_by_label else None
                   for label in missing]
        with ThreadPoolExecutor(SEE_THROUGH_READERS, thread_name_prefix="see-through") as readers:
            known.update(zip(missing, readers.map(_translucent_render, renders)))
        cache.parent.mkdir(parents=True, exist_ok=True)
        partial = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
        partial.write_text(json.dumps({"capture": capture, "translucent": known}, sort_keys=True) + "\n",
                           encoding="utf-8")
        os.replace(partial, cache)
    return {label for label in labels if known[label]}


@dataclass
class Backgrounds:
    """What translucent faces show through them: the dump's cells (excluded cells included, BlueMap still draws
    them), its translucent cells and the cells whose appearance is known to differ (accepted 1.13 states where the
    data backs the reason, excluded cells)."""

    cells: np.ndarray  # cell keys (faces.cell_keys), sorted as each of the three
    see_through: np.ndarray
    differing: np.ndarray


def backgrounds(variant: variants.Variant, blocks: dict, removed: list, accepted_list: accepted.Accepted,
                translucent: set[str] | None = None) -> Backgrounds:
    """``translucent``: the blocks' [see_through_labels] when they are known already."""
    def keys_of(cells: list) -> np.ndarray:
        return np.unique(faces.cell_keys(np.asarray(cells, dtype=np.int64).reshape(-1, 3)))

    if translucent is None:
        translucent = see_through_labels(variant, {b.label for b in blocks.values()})
    differing = list(removed)
    for p, b in blocks.items():
        entry, backed = accepted_list.check(b.name, b.meta)
        if entry is not None and backed:
            differing.append(p)
    return Backgrounds(keys_of([*blocks, *removed]), keys_of([p for p, b in blocks.items() if b.label in translucent]),
                       keys_of(differing))


def colour_excluded(client: cameras.ClientView, reference: faces.Labels, ref_world: np.ndarray, test: faces.Labels,
                    test_world: np.ndarray, bg: Backgrounds) -> np.ndarray:
    """Pixels whose colour is not judged: where either rendering shows a translucent dump cell through which the view
    ray does not reach a judged dump cell, but a cell whose appearance is known to differ (the difference is that
    cell's, accepted or excluded there) or nothing of the dump (terrain around the scene and sky, never judged).
    Coverage (IoU) is still judged."""
    out = np.zeros(reference.key.shape, dtype=bool)
    if not bg.see_through.size:
        return out
    for labels, world in ((reference, ref_world), (test, test_world)):
        own = np.where(labels.key >= 0, labels.key & ~np.int64(7), -1)
        translucent = faces.index_of(own, bg.see_through) >= 0
        beyond = faces.behind(labels, world, client.eye, translucent, bg.cells, bg.see_through)
        out |= translucent & ((beyond < 0) | (faces.index_of(beyond, bg.differing) >= 0))
    return out


def _in_box(ids: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray,
            pixels: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The ``pixels`` (flat indices, default all) whose face ``ids`` (H, W, -1 = none) is one with its box (x0, y0,
    x1, y1 per face) around the pixel, and their faces."""
    w = ids.shape[1]
    flat = ids.reshape(-1)
    pixels = np.flatnonzero(flat >= 0) if pixels is None else pixels[flat[pixels] >= 0]
    face = flat[pixels]
    rows, cols = np.divmod(pixels, w)
    inside = (rows >= y0[face]) & (rows < y1[face]) & (cols >= x0[face]) & (cols < x1[face])
    return pixels[inside], face[inside]


def face_records(client: cameras.ClientView, reference: faces.Labels, test: faces.Labels, test_rgb: np.ndarray,
                 blocks: dict, keys: np.ndarray, excluded: np.ndarray | None = None) -> list[dict]:
    """One record per dump face either rendering shows (``faces.compare_faces``), to be judged by ``judge``;
    ``excluded`` pixels are left out of the colour metrics ([colour_excluded]). A face's ``magenta`` share is that of
    the test pixels of the face in its box (occluded ones too), ``background_excluded`` counts the ``excluded``
    pixels of the face in either rendering in its box."""
    results = faces.compare_faces(_rgba(client.image), _rgba(test_rgb), reference, test, keys, excluded)
    if not results:
        return []
    count = len(results)
    found = np.array([r.key for r in results], dtype=np.int64)
    x0, y0, x1, y1 = np.array([r.bbox for r in results], dtype=np.int64).T
    test_id = faces.index_of(test.key, found)
    pixels, face = _in_box(test_id, x0, y0, x1, y1)
    shown = np.bincount(face, minlength=count)
    magenta = np.bincount(face[image_metrics.magenta_mask(test_rgb[..., :3]).reshape(-1)[pixels]], minlength=count)
    share = np.divide(magenta, shown, out=np.zeros(count), where=shown > 0).tolist()
    background = [None] * count
    if excluded is not None:
        candidates = np.flatnonzero(excluded)
        ref_id = faces.index_of(reference.key, found)
        _, ref_face = _in_box(ref_id, x0, y0, x1, y1, candidates)
        test_pixels, test_face = _in_box(test_id, x0, y0, x1, y1, candidates)
        also = test_face != ref_id.reshape(-1)[test_pixels]  # a pixel of the face in both renderings counts once
        background = (np.bincount(ref_face, minlength=count) + np.bincount(test_face[also], minlength=count)).tolist()
    records = []
    for result, magenta_share, background_excluded in zip(results, share, background):
        block = blocks[(result.x, result.y, result.z)]
        record = _face_record(client.name, result, {"block": block.name, "meta": block.meta, "label": block.label,
                                                    "mod": block.mod}, magenta_share)
        if excluded is not None:
            record["background_excluded"] = background_excluded
        records.append(record)
    return records


def compare_view(client: cameras.ClientView, capture: webapp.WebappCapture, blocks: dict, keys: np.ndarray,
                 out: Path, bg: Backgrounds) -> tuple[list[dict], dict, dict]:
    """Face records (unjudged), the alignment and the matrix/consistency checks of one view (``keys``: [dump_keys])."""
    reference, ref_eye, ref_world = reference_labels(client)
    test, test_world = bluemap_labels(client, capture.distance)

    in_dump = faces.index_of(reference.key, keys) >= 0
    focal = client.height / 2.0 * client.projection[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels_per_block = focal / np.maximum(-ref_eye[..., 2], 1e-6)
    alignment = faces.alignment(reference, test, in_dump, pixels_per_block)
    excluded = colour_excluded(client, reference, ref_world, test, test_world, bg)
    records = face_records(client, reference, test, capture.rgba, blocks, keys, excluded)
    expected_p, expected_mv = cameras.expected_matrices(
        {"fov": client.meta["fov"], "yaw": client.meta["yaw"], "pitch": client.meta["pitch"]},
        client.width, client.height, float(client.meta["far"]))
    checks = {
        "client_settled": client.meta.get("settle", {}).get("settled"),
        "client_stable": client.meta.get("stability", {}).get("stable"),
        "client_projection_vs_vanilla": float(np.abs(client.projection - expected_p).max()),
        "client_modelview_vs_vanilla": float(np.abs(client.modelview - expected_mv).max()),
        "client_eye_vs_requested": float(np.abs(client.camera - np.asarray(client.meta["eye"])).max()),
        "webapp_projection_error": float(np.abs(capture.projection - client.projection).max()),
        "webapp_view_error": float(np.abs(capture.view - client.view).max()),
        "webapp_settled": capture.settled,
        "webapp_stable": capture.stable,
        "webapp_frames": capture.frames,
        "webapp_console_errors": capture.console_errors[:5],
        "reference_dump_pixels": int(in_dump.sum()),
        "test_dump_pixels": int((faces.index_of(test.key, keys) >= 0).sum()),
    }
    world_report.save_view_images(out, client, capture, reference, test, keys)
    return records, alignment.to_json(), checks


def area(client: cameras.ClientView, keys: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Centre and radius (blocks) of the dump cells (``keys``: [dump_keys]) the client shows, for the webapp's hires
    tile loading."""
    labels, _, world = reference_labels(client)
    shown = faces.index_of(labels.key, keys) >= 0
    if not shown.any():
        x, y, z = client.camera
        return (float(x), float(y), float(z)), 64.0
    points = world[shown]
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    return (float(centre[0]), float(centre[1]), float(centre[2])), float(np.linalg.norm(high - low) / 2.0 + 48.0)


@dataclass
class CalibrationPlan:
    """Where a run's thresholds come from: ``compute`` (the calibration scene of this run calibrates and stores the
    result under ``artifact`` and as the active ``calibration_file()``), else the stored ``artifact``, installed as
    the active file. ``artifact`` is None when the references are not known or not current (then only the active
    file is written)."""

    compute: bool
    artifact: Path | None
    references: dict | None


def calibration_references(world_key: str | None, states_key: str | None) -> tuple[dict, Path | None]:
    references = {"world": world_key, "states": states_key}
    if world_key is None:
        return references, None
    return references, reference.calibration_path("world", reference.calibration_key("world", references))


def plan_calibration(runs: list[SceneRun], inputs: reference.Inputs, recalibrate: bool) -> CalibrationPlan:
    """The stored calibration for the current references of the calibration scene (``vanilla-zoo``, default variant)
    and of the default variant's state oracle, unless ``recalibrate`` or there is none: then the calibration scene
    must be part of the run and calibrates."""
    from e2e.oracle import states

    default = variants.load(CALIBRATION_VARIANT)
    own = next((r for r in runs if r.calibrating_pair), None)
    world_key = own.job.key if own else inputs.world(
        default, CALIBRATION_SCENE, cameras.cameras_json(CALIBRATION_SCENE, cameras.scene_cameras(CALIBRATION_SCENE),
                                                         scene_blocks(CALIBRATION_SCENE)[1]))
    states_key = reference.recorded_key(states.states_dir(default)) or inputs.states(default).digest
    references, artifact = calibration_references(world_key.digest, states_key)
    if not recalibrate and artifact is not None and artifact.is_file():
        reference.install_calibration(artifact, calibration_file())
        print(f"[oracle] world calibration {artifact.name} (references unchanged)", flush=True)
        return CalibrationPlan(False, artifact, references)
    if own is None:
        why = "--recalibrate" if recalibrate else f"no world calibration for the current references ({artifact.name})"
        raise SystemExit(f"{why}: add `--variant {CALIBRATION_VARIANT} --scene {CALIBRATION_SCENE}` (or use --all) so "
                         f"that it is calibrated first")
    print(f"[oracle] world calibration: {CALIBRATION_VARIANT}/{CALIBRATION_SCENE} calibrates "
          f"({'--recalibrate' if recalibrate else 'no stored calibration for its references'})", flush=True)
    return CalibrationPlan(True, artifact, references)


def region_metrics(face: dict) -> metrics.RegionMetrics:
    """The metrics of a face record (NaN where the record has null)."""
    def value(key: str) -> float:
        return math.nan if face[key] is None else face[key]

    return metrics.RegionMetrics(face["ref_pixels"], face["test_pixels"], face["both_pixels"], value("iou"),
                                 value("delta_e"), value("delta_e_p95"), value("ssim"))


def calibration_samples(views: list[ViewOutcome], labels: set[str]) -> list[dict]:
    """Every face of the ``labels`` block states with at least ``MIN_PIXELS`` pixels in the aligned views."""
    return [f for v in views if v.alignment and v.alignment["passed"] for f in v.faces
            if f["label"] in labels and f["ref_pixels"] + f["test_pixels"] - f["both_pixels"] >= MIN_PIXELS]


def calibration_labels(classes: dict, accepted_list: accepted.Accepted) -> dict[str, set[str]]:
    """The ``texture`` and ``geometry`` sets of the 1.7.10 vs 1.13 classification as face labels, without the states
    ``accepted.yaml`` names."""
    return {name: {vanilla.label(k) for k in classes["sets"][name]
                   if accepted_list.entry(classes["states"][k]["name"], classes["states"][k]["meta"]) is None}
            for name in ("texture", "geometry")}


def false_failures(views: list[ViewOutcome], thresholds: metrics.Thresholds, labels: set[str]) -> dict:
    """Faces and block states of a calibration set that fail ``thresholds`` (they all look the same in 1.7.10 and
    1.13)."""
    faces = calibration_samples(views, labels)
    failing = [f for f in faces if thresholds.check(region_metrics(f))]
    states = {f["label"] for f in faces}
    failing_states = sorted({f["label"] for f in failing})
    return {"faces": len(faces), "faces_fail": len(failing), "states": len(states), "states_fail": len(failing_states),
            "rate": round(len(failing_states) / len(states), 4) if states else None, "failing": failing_states}


def calibrate(views: list[ViewOutcome], classes: dict, accepted_list: accepted.Accepted,
              plan: CalibrationPlan) -> tuple[metrics.Thresholds, dict]:
    """Thresholds from the faces of the calibration run's vanilla states that look the same in 1.7.10 and 1.13: IoU
    over the ``geometry`` set, delta E and SSIM over the ``texture`` set (``oracle.vanilla``). Stored as the
    artifact of ``plan`` (keyed by the references) and as the active ``calibration_file()``."""
    labels = calibration_labels(classes, accepted_list)
    texture = calibration_samples(views, labels["texture"])
    geometry = calibration_samples(views, labels["geometry"])
    calibration = metrics.calibrate([region_metrics(f) for f in texture],
                                    geometry_samples=[region_metrics(f) for f in geometry],
                                    groups=[f["block"] for f in texture], min_pixels=MIN_PIXELS)
    data = {**calibration.to_json(), "scene": CALIBRATION_SCENE, "variant": CALIBRATION_VARIANT,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "version": reference.CALIBRATION_VERSION,
            "references": plan.references, "artifact": str(plan.artifact) if plan.artifact else None,
            "classification": str(vanilla.file()),
            "unit": f"every face of a calibration state with at least {MIN_PIXELS} pixels in an aligned view (IoU); "
                    f"each block's worst inexact face (delta E, SSIM)",
            "block_states": {"texture": len({f["label"] for f in texture}),
                             "geometry": len({f["label"] for f in geometry})},
            "falseFailures": {name: false_failures(views, calibration.thresholds, labels[name])
                              for name in ("texture", "geometry")}}
    reference.store_calibration(data, plan.artifact, calibration_file())
    return calibration.thresholds, data


def judge(views: list[ViewOutcome], thresholds: metrics.Thresholds, accepted_list: accepted.Accepted) -> None:
    """(Re)applies thresholds and ``accepted.yaml`` to every face record. Faces of accepted states (the data backs
    the reason) are neither passes nor failures; ``accepted_checks`` keeps what they would fail."""
    for view in views:
        for f in view.faces:
            m = region_metrics(f)
            judged = m.union_pixels >= thresholds.min_pixels
            failed = thresholds.check(m) if judged else []
            for key in ("accepted", "accepted_checks", "accepted_unbacked", "crop"):
                f.pop(key, None)
            entry, backed = accepted_list.check(f["block"], f["meta"])
            if entry is not None and backed:
                f.update(judged=judged, passed=False, failed=[], failed_iou=False, failed_delta_e=False,
                         failed_ssim=False, severity=0.0, suspect="", accepted=accepted_list.describe(entry),
                         accepted_checks=failed)
                continue
            if entry is not None:
                f["accepted_unbacked"] = entry.states
            f.update(judged=judged, passed=judged and not failed, failed=failed,
                     failed_iou=any(x.startswith("iou") for x in failed),
                     failed_delta_e=any(x.startswith("delta_e") for x in failed),
                     failed_ssim=any(x.startswith("ssim") for x in failed),
                     severity=severity(m, thresholds) if failed else 0.0)
            f["suspect"] = suspect(f) if failed else ""


def summarize(views: list[ViewOutcome]) -> dict:
    """Face pass rate (every judged face of the aligned views that is not accepted) and block states (labels) with
    every judged face passing, overall and per mod; accepted (1.13) faces are counted apart."""
    trusted = [f for v in views if v.alignment and v.alignment["passed"] for f in v.faces]
    judged = [f for f in trusted if f["judged"] and not f.get("accepted")]
    accepted_faces = [f for f in trusted if f["judged"] and f.get("accepted")]
    passed = [f for f in judged if f["passed"]]
    per_mod: dict[str, dict] = {}
    labels: dict[str, bool] = {}
    for f in judged:
        entry = per_mod.setdefault(f["mod"], {"judged": 0, "passed": 0, "accepted": 0, "states": set(),
                                              "states_failing": set()})
        entry["judged"] += 1
        entry["passed"] += f["passed"]
        entry["states"].add(f["label"])
        if not f["passed"]:
            entry["states_failing"].add(f["label"])
        labels[f["label"]] = labels.get(f["label"], True) and f["passed"]
    for f in accepted_faces:
        per_mod.setdefault(f["mod"], {"judged": 0, "passed": 0, "accepted": 0, "states": set(),
                                      "states_failing": set()})["accepted"] += 1
    for entry in per_mod.values():
        entry["pass_rate"] = round(entry["passed"] / entry["judged"], 4) if entry["judged"] else None
        entry["states"], entry["states_failing"] = len(entry["states"]), len(entry["states_failing"])
    return {
        "views": len(views),
        "views_aligned": sum(1 for v in views if v.alignment and v.alignment["passed"]),
        "faces": sum(len(v.faces) for v in views),
        "faces_in_aligned_views": len(trusted),
        "judged": len(judged),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(judged), 4) if judged else None,
        "accepted": len(accepted_faces),
        "accepted_states": len({f["label"] for f in accepted_faces}),
        "states": len(labels),
        "states_all_faces_pass": sum(labels.values()),
        "per_mod": dict(sorted(per_mod.items(), key=lambda kv: (kv[1]["pass_rate"] is None,
                                                                kv[1]["pass_rate"] or 0.0, kv[0]))),
    }


@dataclass
class RenderGroup:
    """Scenes of one variant with the same BlueMap packs: one CLI process renders all of them (one map each, map id =
    scene name) and serves them while their views are captured."""

    variant: variants.Variant
    packs: list[Path]
    runs: list[SceneRun]
    root: Path


def render_groups(runs: list[SceneRun]) -> list[RenderGroup]:
    """``runs`` grouped by variant and packs, in the order of their first scene (the calibration scene stays first)."""
    groups: dict[tuple, RenderGroup] = {}
    for run in runs:
        key = (run.variant.name, tuple(run.packs))
        if key not in groups:
            n = sum(1 for g in groups.values() if g.variant.name == run.variant.name)
            root = variants.oracle_dir(run.variant) / "worlds" / ("_bluemap" if n == 0 else f"_bluemap-{n + 1}")
            groups[key] = RenderGroup(run.variant, run.packs, [], root)
        groups[key].runs.append(run)
    return list(groups.values())


def start_bluemap(group: RenderGroup, jars: bluemap.Jars) -> bluemap.BlueMapCli:
    """Writes the group's CLI config (every scene's overworld as one map, id ``bluemap.map_id(scene)``) and starts
    rendering and serving."""
    shutil.rmtree(group.root, ignore_errors=True)
    group.root.mkdir(parents=True)
    maps = []
    for run in group.runs:
        world = scenes.load(run.scene).world
        dims = [d for d in bluemap.detect_dimensions(world) if d.map_id == "overworld"]
        if not dims:
            raise FileNotFoundError(f"{world} has no overworld region files")
        maps.append(bluemap.MapSource(bluemap.map_id(run.scene), world, dims[0]))
    ids = [m.map_id for m in maps]
    if len(set(ids)) != len(ids):
        raise ValueError(f"scenes {[r.scene for r in group.runs]} give ambiguous BlueMap map ids {ids}")
    port = bluemap.free_port()
    bluemap.write_config(group.root, maps, jars, port, group.packs)
    (group.root / "packs.txt").write_text("".join(f"{p}\n" for p in group.packs), encoding="utf-8")
    cli = bluemap.BlueMapCli(jars, group.root, port)
    cli.start()
    served = json.loads((group.root / "web" / "settings.json").read_text(encoding="utf-8")).get("maps", [])
    if sorted(served) != sorted(ids):
        cli.stop()
        raise bluemap.BlueMapError(f"BlueMap serves maps {served}, expected {ids}; log {cli.log}")
    print(f"[oracle] {group.variant.name}: BlueMap rendered {len(maps)} map(s) in {cli.render_seconds:.1f}s "
          f"({group.root})", flush=True)
    return cli


@dataclass
class SceneData:
    """What comparing a scene's views needs besides the two captures."""

    blocks: dict
    keys: np.ndarray
    bg: Backgrounds


def scene_data(variant: variants.Variant, packs: list, blocks: dict, removed: list,
               translucent: set[str] | None = None) -> SceneData:
    return SceneData(blocks, dump_keys(blocks), backgrounds(variant, blocks, removed, load_accepted(packs), translucent))


@dataclass(frozen=True)
class ViewJob:
    """One camera of a scene, captured from its group's running CLI and compared (pickled to a judge process)."""

    variant: variants.Variant
    scene: str
    packs: tuple[Path, ...]
    out: Path
    client_dir: Path
    url: str
    index: int  # in the scene's camera list
    camera: str
    source: str
    translucent: frozenset[str]  # the scene's see_through_labels, read once by the run (not by every judge process)


def judge_view(job: ViewJob, data: SceneData, browser) -> tuple[ViewOutcome, float]:
    """The webapp capture of one camera compared with the client's (``error`` when the capture failed) and when its
    judging started (``time.monotonic``)."""
    view = ViewOutcome(job.camera, job.source)
    t0 = time.monotonic()
    client = cameras.ClientView.load(job.client_dir, job.camera)
    centre, radius = area(client, data.keys)
    try:
        capture = webapp.capture(browser, job.url, bluemap.map_id(job.scene), cameras.webapp_camera(client), centre,
                                 radius, client.width, client.height)
    except PlaywrightError as e:  # e.g. the map not loaded within the load timeout: this view only
        view.error = f"webapp capture failed: {str(e).splitlines()[0]}"
        return view, t0
    t1 = time.monotonic()
    view.faces, view.alignment, view.checks = compare_view(client, capture, data.blocks, data.keys, job.out, data.bg)
    view.seconds = {"webapp": round(t1 - t0, 1), "compare": round(time.monotonic() - t1, 1),
                    "client": round(float(client.meta.get("seconds", 0.0)), 1)}
    return view, t0


def view_line(job: ViewJob, view: ViewOutcome) -> str:
    where = f"[oracle] {job.variant.name}/{job.scene}/{view.name}"
    if view.error:
        return f"{where}: {view.error}"
    a = view.alignment
    return (f"{where}: {len(view.faces)} faces, alignment {'ok' if a['passed'] else 'FAILED'} (shift {a['shift']}, "
            f"edge median {a['edge_error_median_px']} px, p90 {a['edge_error_p90_px']} px), "
            f"webapp {view.seconds['webapp']}s, compare {view.seconds['compare']}s")


_judge_process: dict = {}  # in a judge process: its Chromium and the data of its latest scenes


def _judge_process_start() -> None:
    """Initializer of a judge process: its own Playwright and Chromium (the sync API is per process), closed when the
    process exits."""
    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=True, args=CHROMIUM_ARGS)

    def close() -> None:
        browser.close()
        playwright.stop()

    multiprocessing.util.Finalize(None, close, exitpriority=10)
    _judge_process.update(browser=browser, scenes={})


def _judge_process_view(job: ViewJob) -> tuple[ViewOutcome, float]:
    known: dict = _judge_process["scenes"]
    key = (job.variant.name, job.scene)
    data = known.pop(key, None)
    if data is None:
        blocks, removed, _ = scene_blocks(job.scene)
        data = scene_data(job.variant, list(job.packs), blocks, removed, set(job.translucent))
    known[key] = data  # the latest last; jobs come scene after scene, two cover a scene boundary
    if len(known) > 2:
        del known[next(iter(known))]
    return judge_view(job, data, _judge_process["browser"])


@dataclass
class SceneJudging:
    """A scene being judged from its group's running CLI: its views in camera order (cameras the reference lacks
    carry that error), filled in as the jobs of the others are done."""

    run: SceneRun
    group: RenderGroup
    cli: bluemap.BlueMapCli
    views: list[ViewOutcome]
    jobs: list[ViewJob]
    pending: int = 0
    started: float = math.inf  # the first view's start
    ended: float = 0.0  # the last view's end

    def __post_init__(self) -> None:
        self.pending = len(self.jobs)

    def judged(self, job: ViewJob, view: ViewOutcome, started: float) -> None:
        self.views[job.index] = view
        self.pending -= 1
        self.started = min(self.started, started)
        self.ended = time.monotonic()


def open_scene(run: SceneRun, client_dir: Path | None, cli: bluemap.BlueMapCli, group: RenderGroup,
               plan: CalibrationPlan, failed: list[str], translucent: frozenset[str]) -> SceneJudging | None:
    """One job per camera the scene's reference has (``translucent``: the scene's [see_through_labels]); without any
    the scene fails (the calibration scene the run calibrates on ends the run)."""
    out, scene, variant = run.out, run.scene, run.variant
    captured = [c for c in run.camera_list if client_dir is not None and (client_dir / f"{c.name}.json").is_file()]
    if client_dir is None or not captured:
        why = (f"{variant.name}/{scene}: no reference captures (key {run.job.key.short}"
               f"{', log ' + str(run.job.log) if run.job.log else ''})")
        if run.calibrating_pair and plan.compute:
            raise SystemExit(f"[oracle] the calibration scene failed, nothing can be judged: {why}")
        print(f"[oracle] {why}", file=sys.stderr, flush=True)
        failed.append(f"{variant.name}/{scene}")
        return None
    root = out / "bluemap"
    for sub in ("config", "data", "web"):  # per-scene renders of earlier versions
        shutil.rmtree(root / sub, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    (root / "packs.txt").write_text("".join(f"{p}\n" for p in run.packs), encoding="utf-8")
    views: list[ViewOutcome] = []
    jobs: list[ViewJob] = []
    for index, cam in enumerate(run.camera_list):
        views.append(ViewOutcome(cam.name, cam.source))
        if cam not in captured:
            views[-1].error = "no reference capture"
        else:
            jobs.append(ViewJob(variant, scene, tuple(run.packs), out, client_dir, cli.url, index, cam.name,
                                cam.source, translucent))
    return SceneJudging(run, group, cli, views, jobs)


def close_scene(s: SceneJudging, plan: CalibrationPlan, summaries: dict[Path, dict], failed: list[str]) -> None:
    """Judges a scene whose views are all done (calibrating first if it is the run's calibration scene), writes its
    report and the variant's summary."""
    t0 = time.monotonic()
    run, cli = s.run, s.cli
    out, scene, variant = run.out, run.scene, run.variant
    # bluemap_render: the group's CLI (all scenes of the batch); total: from this scene's first view to its last
    timings: dict[str, float | None] = {"bluemap_render": round(cli.render_seconds, 1)}
    if run.job.seconds is not None:
        timings["reference_capture"] = round(float(run.job.seconds), 1)
    timings["total"] = round(max(s.ended - s.started, 0.0), 1)
    ref = json.loads((out / reference.REFERENCE).read_text(encoding="utf-8")) if (out / reference.REFERENCE).is_file() else {}
    base = {
        "scene": scene,
        "variant": variant.name,
        "bluemap_packs": [str(p) for p in run.packs],
        "bluemap_batch": [r.scene for r in s.group.runs],
        "client": {"dir": str(variant.client_dir), "resource_packs": [str(p) for p in variant.resource_packs],
                   "extra_mods": [str(p) for p in variant.extra_mods]},
        "reference": {k: ref.get(k) for k in ("key", "entry", "cached", "complete", "stale", "wanted_key", "log")
                      if k in ref},
        "excluded": {"cells": len(run.removed), "states": run.removed_states},
        "timings": timings,
        "bluemap_log_errors": cli.log_lines("ERROR")[:20],
    }
    report = finish(out, base, s.views, calibrate_into=plan if run.calibrating_pair and plan.compute else None)
    summaries[out / "report.json"] = summary_entry(report)
    update_summary(variant, summaries)
    sm = report["summary"]
    rate = "-" if sm["pass_rate"] is None else f"{sm['pass_rate']:.2%}"
    print(f"[oracle] {variant.name}/{scene}: {sm['passed']}/{sm['judged']} faces pass ({rate}), "
          f"{sm['views_aligned']}/{sm['views']} views aligned, {timings['total']:.0f}s, report "
          f"{time.monotonic() - t0:.0f}s; {out / 'report.html'}", flush=True)
    if sm["views_aligned"] < sm["views"]:
        failed.append(f"{variant.name}/{scene}")


def judge_serial(groups: list[RenderGroup], client_dirs: dict[int, Path | None], jars: bluemap.Jars,
                 plan: CalibrationPlan, summaries: dict[Path, dict], failed: list[str]) -> None:
    """``--jobs 1``: one group after another, one view after another, in this process with one Chromium."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        try:
            for group in groups:
                cli = start_bluemap(group, jars)
                try:
                    for run in group.runs:
                        translucent = see_through_labels(run.variant, {b.label for b in run.blocks.values()})
                        s = open_scene(run, client_dirs[id(run)], cli, group, plan, failed, frozenset(translucent))
                        if s is None:
                            continue
                        data = scene_data(run.variant, run.packs, run.blocks, run.removed, translucent)
                        for job in s.jobs:
                            view, started = judge_view(job, data, browser)
                            s.judged(job, view, started)
                            print(view_line(job, view), flush=True)
                        close_scene(s, plan, summaries, failed)
                finally:
                    cli.stop()
        finally:
            browser.close()


def judge_parallel(groups: list[RenderGroup], client_dirs: dict[int, Path | None], jars: bluemap.Jars,
                   plan: CalibrationPlan, summaries: dict[Path, dict], failed: list[str], jobs: int) -> None:
    """``--jobs N``: N judge processes, each with its own Chromium, capture and compare the views of every scene of
    the run, in camera order per scene, as soon as the group's CLI serves them; the groups render one after another
    meanwhile, the next while the views of the previous are judged. A scene is judged and reported as soon as its
    last view is done, the run's calibration scene before any other; a group's CLI stops after its last scene."""
    pool = ProcessPoolExecutor(jobs, mp_context=multiprocessing.get_context("spawn"),
                               initializer=_judge_process_start)
    renderer = ThreadPoolExecutor(1, thread_name_prefix="bluemap")
    renders: dict[Future, RenderGroup] = {renderer.submit(start_bluemap, group, jars): group for group in groups}
    # every scene's translucent states, read once here while the first group renders, not by each judge process
    lookups = ThreadPoolExecutor(1, thread_name_prefix="see-through")
    translucent = {id(run): lookups.submit(see_through_labels, run.variant, {b.label for b in run.blocks.values()})
                   for group in groups for run in group.runs}
    views: dict[Future, tuple[SceneJudging, ViewJob]] = {}
    clis: list[bluemap.BlueMapCli] = []
    open_scenes: dict[int, int] = {}  # per group: scenes not yet reported
    done_scenes: list[SceneJudging] = []  # every view judged, not yet reported
    calibrated = not plan.compute
    try:
        while renders or views:
            finished, _ = wait([*renders, *views], return_when=FIRST_COMPLETED)
            for future in finished:
                if future in renders:
                    group = renders.pop(future)
                    cli = future.result()
                    clis.append(cli)
                    opened = [open_scene(run, client_dirs[id(run)], cli, group, plan, failed,
                                         frozenset(translucent[id(run)].result())) for run in group.runs]
                    opened = [s for s in opened if s is not None]
                    open_scenes[id(group)] = len(opened)
                    for s in opened:
                        views.update({pool.submit(_judge_process_view, job): (s, job) for job in s.jobs})
                    if not opened:
                        cli.stop()
                else:
                    s, job = views.pop(future)
                    view, started = future.result()
                    s.judged(job, view, started)
                    print(view_line(job, view), flush=True)
                    if not s.pending:
                        done_scenes.append(s)
            while ready := [s for s in done_scenes if calibrated or s.run.calibrating_pair]:
                s = ready[0]
                done_scenes.remove(s)
                close_scene(s, plan, summaries, failed)
                calibrated = calibrated or s.run.calibrating_pair
                open_scenes[id(s.group)] -= 1
                if not open_scenes[id(s.group)]:
                    s.cli.stop()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        renderer.shutdown(wait=True, cancel_futures=True)
        lookups.shutdown(wait=True, cancel_futures=True)
        clis += [f.result() for f in renders if not f.cancelled() and f.exception() is None]
        for cli in clis:
            cli.stop()


def load_accepted(packs: list) -> accepted.Accepted:
    """``accepted.yaml`` against the classification, with the textures BlueMap's ``packs`` replace."""
    return accepted.load(vanilla.load(), vanilla.texture_overrides(Path(p) for p in packs))


def finish(out: Path, base: dict, views: list[ViewOutcome], *, calibrate_into: CalibrationPlan | None) -> dict:
    """Calibrates (``calibrate_into``: the calibration scene) or reads the active thresholds, judges every face with
    them and ``accepted.yaml``, writes ``report.json`` and ``report.html`` to ``out``."""
    classes = vanilla.load()
    accepted_list = load_accepted(base["bluemap_packs"])
    if calibrate_into is not None:
        thresholds, calibration = calibrate(views, classes, accepted_list, calibrate_into)
    else:
        calibration = json.loads(calibration_file().read_text(encoding="utf-8"))
        thresholds = metrics.Thresholds.from_json(calibration["thresholds"])
    judge(views, thresholds, accepted_list)
    faces_all = [f for v in views for f in v.faces]
    report = {
        **base,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "thresholds": thresholds.to_json(),
        "calibration": calibration,
        "summary": summarize(views),
        "accepted": {"file": str(accepted.FILE), "entries": len(accepted_list.entries),
                     "states": sorted({f["label"] for f in faces_all if f.get("accepted")}),
                     "unbacked": sorted({f["label"] for f in faces_all if f.get("accepted_unbacked")})},
        "views": [{"name": v.name, "source": v.source, "error": v.error, "alignment": v.alignment,
                   "checks": v.checks, "seconds": v.seconds,
                   "summary": summarize([v]) if v.alignment else None} for v in views],
        "faces": faces_all,
    }
    world_report.write(out, report, CROPS)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    return report


def rejudge(variant: variants.Variant, out: Path, *, recalibrate: bool = False) -> dict:
    """Judges an existing ``out/report.json`` again with the current metrics, calibration and ``accepted.yaml``,
    without capturing or rendering anything. Every view's face metrics are recomputed from the saved captures
    (``client/<view>.{png,depth.bin,json}``, ``bluemap/<view>.{png,depth.bin}``) and the scene dump; the alignment
    is kept. A view whose BlueMap depth was not saved (a partial copy) keeps its stored metrics, and the report says
    so under ``stored_metric_views``. With ``recalibrate``, the calibration scene (vanilla-zoo of the default variant,
    in its own scene directory) calibrates again, stored under the references recorded in its ``reference.json``."""
    data = json.loads((out / "report.json").read_text(encoding="utf-8"))
    if data["variant"] != variant.name:
        raise ValueError(f"{out}: report of variant {data['variant']}, not {variant.name}")
    blocks, removed, _ = scene_blocks(data["scene"])
    keys = dump_keys(blocks)
    bg = backgrounds(variant, blocks, removed, load_accepted(data["bluemap_packs"]))
    stored = {}
    for f in data["faces"]:
        stored.setdefault(f["view"], []).append(f)
    views, kept = [], []
    for v in data["views"]:
        view = ViewOutcome(v["name"], v["source"], stored.get(v["name"], []), v["alignment"], v["checks"],
                           v["seconds"], v["error"])
        views.append(view)
        depth = out / "bluemap" / f"{v['name']}.depth.bin"
        if v["error"] or not v["alignment"]:
            continue
        if not depth.is_file():
            kept.append(v["name"])
            continue
        client = cameras.ClientView.load(out / "client", v["name"])
        with Image.open(out / "bluemap" / f"{v['name']}.png") as image:
            test_rgb = np.asarray(image.convert("RGB"))
        distance = np.fromfile(depth, dtype="<f4").reshape(client.height, client.width)
        reference, _, ref_world = reference_labels(client)
        test, test_world = bluemap_labels(client, distance)
        view.faces = face_records(client, reference, test, test_rgb, blocks, keys,
                                  colour_excluded(client, reference, ref_world, test, test_world, bg))
    base = {k: data.get(k) for k in ("scene", "variant", "bluemap_packs", "client", "excluded", "timings",
                                     "bluemap_log_errors")}
    base["excluded"] = base["excluded"] or {"cells": 0, "states": []}
    base["rejudged_from"] = data.get("rejudged_from") or data["created"]
    base["stored_metric_views"] = kept
    calibrate_into = None
    if (recalibrate and variant.name == CALIBRATION_VARIANT and data["scene"] == CALIBRATION_SCENE
            and out.resolve() == scene_dir(variant, CALIBRATION_SCENE).resolve()):
        from e2e.oracle import states

        references, artifact = calibration_references(reference.recorded_key(out),
                                                      reference.recorded_key(states.states_dir(CALIBRATION_VARIANT)))
        calibrate_into = CalibrationPlan(True, artifact, references)
    return finish(out, base, views, calibrate_into=calibrate_into)


def scene_blocks(scene: str) -> tuple[dict, list, list[str]]:
    """The scene dump's blocks without its ``oracle.exclude`` cells (air in the client's world, never judged), the
    removed cells and their state labels."""
    blocks = load_dump(scene)
    excluded = cameras.excluded_states(scene)
    removed = sorted(p for p, b in blocks.items() if cameras.is_excluded(excluded, b.name, b.meta))
    removed_states = sorted({blocks[p].label for p in removed})
    for p in removed:
        del blocks[p]
    return blocks, removed, removed_states


def summary_entry(report: dict) -> dict:
    return {"scene": report["scene"], "summary": report["summary"], "timings": report["timings"],
            "created": report["created"]}


def update_summary(variant: variants.Variant, known: dict[Path, dict] | None = None) -> Path:
    """``out/oracle/<variant>/worlds/summary.json``: the summary of every scene report of the variant. ``known``
    (report path: ``summary_entry``) keeps the reports a run already read or wrote from being read again (the
    end-game report alone is about 750 MB)."""
    root = variants.oracle_dir(variant) / "worlds"
    known = {} if known is None else known
    scenes_summary = {}
    for report in sorted(root.glob("*/report.json")):
        if report not in known:
            known[report] = summary_entry(json.loads(report.read_text(encoding="utf-8")))
        entry = known[report]
        scenes_summary[entry["scene"]] = {k: entry[k] for k in ("summary", "timings", "created")}
    path = root / "summary.json"
    path.write_text(json.dumps({"variant": variant.name, "scenes": scenes_summary}, indent=2) + "\n", encoding="utf-8")
    return path


def cmd_rejudge(args: argparse.Namespace, variant_list: list[variants.Variant]) -> int:
    if args.report_dir and len(variant_list) != 1:
        raise SystemExit("--report-dir takes exactly one --variant")
    if args.recalibrate and not (CALIBRATION_SCENE in (args.scene or [])
                                 and any(v.name == CALIBRATION_VARIANT for v in variant_list)):
        raise SystemExit(f"--rejudge --recalibrate needs --variant {CALIBRATION_VARIANT} --scene {CALIBRATION_SCENE}")
    variant_list = sorted(variant_list, key=lambda v: v.name != CALIBRATION_VARIANT)  # calibrate first
    for variant in variant_list:
        targets = [scene_dir(variant, s) for s in args.scene or []] + list(args.report_dir or [])
        if not targets:
            raise SystemExit("--rejudge needs --scene or --report-dir")
        targets.sort(key=lambda t: t != scene_dir(variant, CALIBRATION_SCENE))
        for out in targets:
            s = rejudge(variant, out, recalibrate=args.recalibrate)["summary"]
            rate = "-" if s["pass_rate"] is None else f"{s['pass_rate']:.2%}"
            print(f"[oracle] {variant.name}/{out.name} rejudged: {s['passed']}/{s['judged']} faces pass ({rate}), "
                  f"{s['accepted']} faces accepted (1.13); {out / 'report.html'}", flush=True)
        if args.scene:
            update_summary(variant)
    return 0


def cmd_world(args: argparse.Namespace) -> int:
    started = time.monotonic()
    variant_list = [variants.load(v) for v in dict.fromkeys(args.variant or ["default"])]
    if args.rejudge or args.report_dir:
        return cmd_rejudge(args, variant_list)
    if not args.scene and not args.all:
        raise SystemExit("--scene or --all is required")
    for variant in variant_list:
        missing = variant.missing()
        if missing:
            raise FileNotFoundError(f"variant {variant.name}: missing {', '.join(str(p) for p in missing)}")

    # references: keys, cache hits, one client batch per variant for the misses
    inputs = reference.Inputs.open()
    selected = {v.name: oracle_scenes(v, args.scene) if args.all else args.scene for v in variant_list}
    runs = [prepare(v, s, inputs) for v in variant_list for s in selected[v.name]]
    if not runs:
        print("[oracle] no scene to run", flush=True)
        return 0
    runs.sort(key=lambda r: not r.calibrating_pair)  # the calibration scene is judged first
    plan = plan_calibration(runs, inputs, args.recalibrate)
    inputs.save()
    t_keys = time.monotonic()
    reference.capture_world([r.job for r in runs], recapture=args.recapture, capture=not args.no_capture,
                            options=reference.LaunchOptions(args.mod_jar, args.parallel))
    t_refs = time.monotonic()
    client_dirs = {id(r): reference.use_world(r.job, r.out) for r in runs}
    own = next((r for r in runs if r.calibrating_pair), None)
    if plan.compute and own is not None and not own.job.complete:
        print(f"[oracle] the calibration scene's reference is not complete or not current: its calibration is "
              f"written to {calibration_file()} only, not stored for the references", file=sys.stderr, flush=True)
        plan.artifact = None
    hits = sum(r.job.cached for r in runs)
    print(f"[oracle] references: {hits}/{len(runs)} cached, keys {t_keys - started:.1f}s, "
          f"capture {t_refs - t_keys:.1f}s", flush=True)

    jars = runner.prepare_jars(build=not args.no_build)
    failed: list[str] = []
    summaries: dict[Path, dict] = {}
    if args.jobs == 1:
        judge_serial(render_groups(runs), client_dirs, jars, plan, summaries, failed)
    else:
        judge_parallel(render_groups(runs), client_dirs, jars, plan, summaries, failed, args.jobs)
    print(f"[oracle] {len(runs)} scene runs in {time.monotonic() - started:.0f}s (references {t_refs - started:.0f}s)"
          + (f"; failed: {', '.join(failed)}" if failed else ""), flush=True)
    return 1 if failed else 0


def _jobs(value: str) -> int:
    jobs = int(value)
    if jobs < 1:
        raise argparse.ArgumentTypeError("--jobs must be at least 1")
    return jobs


def add_parser(oracle_sub: argparse._SubParsersAction) -> None:
    p = oracle_sub.add_parser("world", help="BlueMap's webapp vs real-client screenshots of scene worlds, per block face")
    p.add_argument("--variant", action="append", choices=variants.NAMES,
                   help="client/pack variant (repeatable, default: default)")
    p.add_argument("--scene", action="append", help="scene (repeatable)")
    p.add_argument("--all", action="store_true",
                   help="every scene with oracle cameras and a dump that fits the variant (addons-* scenes: addons); "
                        "with --scene only those of them, the others skipped")
    p.add_argument("--rejudge", action="store_true",
                   help="judge the existing report.json of each --scene again (current metrics, calibration and "
                        "accepted.yaml) without capturing or rendering")
    p.add_argument("--report-dir", type=Path, action="append",
                   help="rejudge the report.json in this directory (a copy of a scene directory; repeatable)")
    p.add_argument("--recapture", action="store_true",
                   help="capture the references with the client even if they are cached")
    p.add_argument("--no-capture", action="store_true",
                   help="never launch a client: scenes without a cached reference use their existing client/ "
                        "captures (reported as stale) or fail")
    p.add_argument("--recalibrate", action="store_true",
                   help=f"calibrate the thresholds again ({CALIBRATION_VARIANT}/{CALIBRATION_SCENE} must be in the run) "
                        f"instead of reusing the calibration stored for the current references")
    p.add_argument("--parallel", type=int, help="client instances for the reference captures (gtnh-client --parallel)")
    p.add_argument("--jobs", type=_jobs, default=JOBS,
                   help=f"views judged at once: processes that each capture BlueMap's webapp in their own headless "
                        f"Chromium and compare it with the client (default {JOBS}); 1 judges one view after another "
                        f"in this process")
    p.add_argument("--mod-jar", type=Path, help="exporter jar for the reference capture (default: the launcher builds it)")
    p.add_argument("--no-build", action="store_true", help="use the existing BlueMap CLI/addon jars")
    p.set_defaults(func=cmd_world)
