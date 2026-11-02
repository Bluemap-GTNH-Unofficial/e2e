"""`e2e oracle world`: BlueMap's webapp against screenshots of the real GTNH client, per visible block face.

Per scene and variant (outputs in ``out/oracle/<variant>/worlds/<scene>/``, never committed):

1. ``cameras.json``: the scene's oracle cameras (``oracle.camera``).
2. Reference: ``client/<view>.{png,depth.bin,json}`` from ``gtnh-client cameras`` (the client joins a server running
   the scene world; see the launcher README). Captured again when missing, when the camera list changed, or with
   ``--capture always``.
3. BlueMap: the scene rendered with the BlueMap CLI and the variant's packs (``bluemap/``), served, and captured at
   the client's exact matrices with the capture conditions of the client (``oracle.webapp``):
   ``bluemap/<view>.png`` and ``bluemap/<view>.depth.bin`` (float32 view-space depth, NaN = nothing drawn).
4. Faces: both renderings are labelled per pixel with the dump block and cell face they show (``oracle.faces``);
   alignment is checked first (block-edge reprojection error, global shift), then every face of a dump block that
   either rendering shows is compared with the shared region metrics (``oracle.metrics``).
5. ``report.json`` and ``report.html`` (worst faces first, grouped by mod, with crops in ``report/``).

Pass thresholds come from a calibration on the vanilla blocks of ``vanilla-zoo`` with the default variant
(``out/oracle/default/worlds/calibration.json``, Tukey's far-out fences over that distribution; never tuned by hand);
that run writes it, every other run reads it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

from e2e import bluemap, paths, runner, scenes
from e2e import metrics as image_metrics
from e2e.browser import CHROMIUM_ARGS
from e2e.oracle import camera as cameras
from e2e.oracle import faces, metrics, variants, webapp, world_report

CALIBRATION_SCENE = "vanilla-zoo"
CALIBRATION_VARIANT = "default"
CALIBRATION_NAMESPACE = "minecraft"
MIN_PIXELS = 64  # faces whose coverage union is smaller are reported but not judged
CROPS = 60  # worst faces per scene that get a crop image


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
    path = paths.workspace() / "out" / "scenes" / scene / "dump.ndjson"
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; build the scene with `uv run orch scene {scene}`")
    blocks = {}
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            entry = json.loads(line)
            if entry["name"] != "minecraft:air":
                blocks[(entry["x"], entry["y"], entry["z"])] = Block(entry["name"], entry["meta"], entry.get("te"))
    return blocks


def dump_keys(blocks: dict[tuple[int, int, int], Block]) -> np.ndarray:
    cells = np.asarray(list(blocks), dtype=np.int64)
    return np.concatenate([faces.encode(cells, np.full(len(cells), f)) for f in range(len(faces.FACES))])


# ---------------------------------------------------------------- reference capture


def client_complete(client_dir: Path, data: dict) -> bool:
    """True when every camera of ``data`` has a capture made with exactly this camera definition and removals."""
    spec = client_dir / "spec.json"
    if not spec.is_file():
        return False
    spec_data = json.loads(spec.read_text(encoding="utf-8"))
    if spec_data.get("cameras") != data["cameras"] or spec_data.get("remove", []) != data["remove"]:
        return False
    return all((client_dir / f"{c['name']}{s}").is_file() for c in data["cameras"] for s in (".png", ".depth.bin", ".json"))


def capture_reference(variant: variants.Variant, scene: str, cameras_file: Path, client_dir: Path, log: Path,
                      mod_jar: Path | None) -> float:
    """Runs `gtnh-client cameras` in the sibling launcher checkout; returns its wall time."""
    launcher = paths.workspace() / "gtnh-exporter" / "launcher"
    cmd = ["uv", "run", "gtnh-client", "cameras", "--scene", scene, "--variant", variant.name,
           "--cameras", str(cameras_file), "--out", str(client_dir), "--client-dir", str(variant.client_dir),
           "--server-dir", variant.server, "--port", str(variant.server_port), "--rcon-port", str(variant.rcon_port)]
    cmd += [f"--resource-pack={p}" for p in variant.resource_packs]
    cmd += [f"--extra-mod={m}" for m in variant.extra_mods]
    if mod_jar:
        cmd += ["--mod-jar", str(mod_jar)]
    print(f"[oracle] reference capture: {subprocess.list2cmdline(cmd)}", flush=True)
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as out:
        proc = subprocess.Popen(cmd, cwd=launcher, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            if line.startswith(b"[cameras]") or line.startswith(b"[run]") or b"Camera " in line:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
        code = proc.wait()
    if code != 0:
        print(f"[oracle] reference capture exited with {code} (log {log}); comparing the cameras that were captured",
              file=sys.stderr)
    return time.monotonic() - started


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


def _face_record(view: str, result: faces.FaceResult, block: dict, thresholds: metrics.Thresholds | None,
                 magenta: float) -> dict:
    m = result.metrics
    record = {"view": view, "x": result.x, "y": result.y, "z": result.z, "face": result.face, **block,
              **m.to_json(), "planar": round(result.planar, 3), "distance": round(result.distance, 2),
              "bbox": list(result.bbox), "magenta": round(magenta, 3)}
    judged = m.union_pixels >= (thresholds.min_pixels if thresholds else MIN_PIXELS)
    failed = thresholds.check(m) if thresholds and judged else []
    record["judged"] = judged
    record["passed"] = judged and thresholds is not None and not failed
    record["failed"] = failed
    record["failed_iou"] = any(f.startswith("iou") for f in failed)
    record["failed_delta_e"] = any(f.startswith("delta_e") for f in failed)
    record["failed_ssim"] = any(f.startswith("ssim") for f in failed)
    record["severity"] = severity(m, thresholds) if failed else 0.0
    record["suspect"] = suspect(record) if failed else ""
    return record


def severity(m: metrics.RegionMetrics, t: metrics.Thresholds | None) -> float:
    """Sum of the relative exceedances of the failed criteria (worst faces first)."""
    if t is None:
        return 0.0
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


def compare_view(client: cameras.ClientView, capture: webapp.WebappCapture, blocks: dict, keys: np.ndarray,
                 thresholds: metrics.Thresholds | None, out: Path) -> tuple[list[dict], dict, dict]:
    """Face records, the alignment and the matrix/consistency checks of one view."""
    reference, ref_eye, _ = reference_labels(client)
    valid = ~np.isnan(capture.distance)
    test_eye = cameras.eye_from_linear_depth(client.projection, np.where(valid, capture.distance, 1.0))
    test = faces.label(cameras.eye_to_world(client.view, test_eye), valid, client.eye)

    in_dump = np.isin(reference.key, keys)
    focal = client.height / 2.0 * client.projection[1, 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        pixels_per_block = focal / np.maximum(-ref_eye[..., 2], 1e-6)
    alignment = faces.alignment(reference, test, in_dump, pixels_per_block)

    ref_rgba = _rgba(client.image)
    test_rgba = _rgba(capture.rgba)
    results = faces.compare_faces(ref_rgba, test_rgba, reference, test, keys)
    magenta = image_metrics.magenta_mask(capture.rgba[..., :3])
    records = []
    for result in results:
        block = blocks[(result.x, result.y, result.z)]
        x0, y0, x1, y1 = result.bbox
        test_region = test.key[y0:y1, x0:x1] == result.key
        share = float(magenta[y0:y1, x0:x1][test_region].mean()) if test_region.any() else 0.0
        records.append(_face_record(client.name, result, {"block": block.name, "meta": block.meta, "label": block.label,
                                                          "mod": block.mod}, thresholds, share))

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
        "test_dump_pixels": int(np.isin(test.key, keys).sum()),
    }
    world_report.save_view_images(out, client, capture, reference, test, keys)
    return records, alignment.to_json(), checks


def area(client: cameras.ClientView, keys: np.ndarray) -> tuple[tuple[float, float, float], float]:
    """Centre and radius (blocks) of the dump cells the client shows, for the webapp's hires tile loading."""
    labels, _, world = reference_labels(client)
    shown = np.isin(labels.key, keys)
    if not shown.any():
        x, y, z = client.camera
        return (float(x), float(y), float(z)), 64.0
    points = world[shown]
    low, high = points.min(axis=0), points.max(axis=0)
    centre = (low + high) / 2.0
    return (float(centre[0]), float(centre[1]), float(centre[2])), float(np.linalg.norm(high - low) / 2.0 + 48.0)


def load_thresholds(variant: variants.Variant, scene: str) -> metrics.Thresholds | None:
    if variant.name == CALIBRATION_VARIANT and scene == CALIBRATION_SCENE:
        return None  # this run produces the calibration
    path = calibration_file()
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist; run `uv run e2e oracle world --variant {CALIBRATION_VARIANT} "
                                f"--scene {CALIBRATION_SCENE}` first (it calibrates the thresholds)")
    return metrics.Thresholds.from_json(json.loads(path.read_text(encoding="utf-8"))["thresholds"])


def region_metrics(face: dict) -> metrics.RegionMetrics:
    """The metrics of a face record (NaN where the record has null)."""
    def value(key: str) -> float:
        return math.nan if face[key] is None else face[key]

    return metrics.RegionMetrics(face["ref_pixels"], face["test_pixels"], face["both_pixels"], value("iou"),
                                 value("delta_e"), value("delta_e_p95"), value("ssim"))


def calibration_samples(views: list[ViewOutcome]) -> list[dict]:
    """One face record per vanilla block state and face direction: its largest region over the aligned views. Every
    block state weighs the same (the stone floor of the zoo would otherwise be most of the samples)."""
    best: dict[tuple[str, str], dict] = {}
    for view in views:
        if not (view.alignment and view.alignment["passed"]):
            continue
        for f in view.faces:
            if f["mod"] != CALIBRATION_NAMESPACE:
                continue
            union = f["ref_pixels"] + f["test_pixels"] - f["both_pixels"]
            key = (f["label"], f["face"])
            if union >= MIN_PIXELS and (key not in best or union > best[key]["_union"]):
                best[key] = {**f, "_union": union}
    return list(best.values())


def calibrate(views: list[ViewOutcome]) -> tuple[metrics.Thresholds, dict]:
    """Thresholds from the vanilla faces of the calibration run (``calibration_samples``)."""
    samples = calibration_samples(views)
    calibration = metrics.calibrate([region_metrics(f) for f in samples], min_pixels=MIN_PIXELS)
    data = {**calibration.to_json(), "scene": CALIBRATION_SCENE, "variant": CALIBRATION_VARIANT,
            "namespace": CALIBRATION_NAMESPACE,
            "unit": "one sample per vanilla block state and face direction: its largest visible region in any view",
            "block_states": len({f["label"] for f in samples})}
    path = calibration_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return calibration.thresholds, data


def judge(views: list[ViewOutcome], thresholds: metrics.Thresholds) -> None:
    """(Re)applies thresholds to every face record (the calibration run judges after calibrating)."""
    for view in views:
        for f in view.faces:
            m = region_metrics(f)
            judged = m.union_pixels >= thresholds.min_pixels
            failed = thresholds.check(m) if judged else []
            f.update(judged=judged, passed=judged and not failed, failed=failed,
                     failed_iou=any(x.startswith("iou") for x in failed),
                     failed_delta_e=any(x.startswith("delta_e") for x in failed),
                     failed_ssim=any(x.startswith("ssim") for x in failed),
                     severity=severity(m, thresholds) if failed else 0.0)
            f["suspect"] = suspect(f) if failed else ""


def summarize(views: list[ViewOutcome]) -> dict:
    """Face pass rate (every judged face of the aligned views) and block states (labels) with every judged face
    passing, overall and per mod."""
    trusted = [f for v in views if v.alignment and v.alignment["passed"] for f in v.faces]
    judged = [f for f in trusted if f["judged"]]
    passed = [f for f in judged if f["passed"]]
    per_mod: dict[str, dict] = {}
    labels: dict[str, bool] = {}
    for f in judged:
        entry = per_mod.setdefault(f["mod"], {"judged": 0, "passed": 0, "states": set(), "states_failing": set()})
        entry["judged"] += 1
        entry["passed"] += f["passed"]
        entry["states"].add(f["label"])
        if not f["passed"]:
            entry["states_failing"].add(f["label"])
        labels[f["label"]] = labels.get(f["label"], True) and f["passed"]
    for entry in per_mod.values():
        entry["pass_rate"] = round(entry["passed"] / entry["judged"], 4)
        entry["states"], entry["states_failing"] = len(entry["states"]), len(entry["states_failing"])
    return {
        "views": len(views),
        "views_aligned": sum(1 for v in views if v.alignment and v.alignment["passed"]),
        "faces": sum(len(v.faces) for v in views),
        "faces_in_aligned_views": len(trusted),
        "judged": len(judged),
        "passed": len(passed),
        "pass_rate": round(len(passed) / len(judged), 4) if judged else None,
        "states": len(labels),
        "states_all_faces_pass": sum(labels.values()),
        "per_mod": dict(sorted(per_mod.items(), key=lambda kv: (kv[1]["pass_rate"], kv[0]))),
    }


def run_scene(variant: variants.Variant, scene: str, jars: bluemap.Jars, browser, args: argparse.Namespace) -> dict:
    started = time.monotonic()
    out = scene_dir(variant, scene)
    out.mkdir(parents=True, exist_ok=True)
    camera_list = cameras.scene_cameras(scene)
    blocks = load_dump(scene)
    excluded = cameras.excluded_states(scene)
    removed = sorted(p for p, b in blocks.items() if cameras.is_excluded(excluded, b.name, b.meta))
    removed_states = sorted({blocks[p].label for p in removed})
    for p in removed:  # air in the client's world: never judged
        del blocks[p]
    data = cameras.cameras_json(scene, camera_list, removed)
    cameras_file = out / "cameras.json"
    cameras_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    client_dir = out / "client"
    timings: dict[str, float] = {}

    if args.capture == "always" or (args.capture == "auto" and not client_complete(client_dir, data)):
        timings["reference_capture"] = round(capture_reference(variant, scene, cameras_file, client_dir,
                                                               out / "client.log", args.mod_jar), 1)
    captured = [c for c in camera_list if (client_dir / f"{c.name}.json").is_file()]
    if not captured:
        raise RuntimeError(f"no reference captures in {client_dir}; see {out / 'client.log'}")

    keys = dump_keys(blocks)
    thresholds = load_thresholds(variant, scene)

    # BlueMap render with the variant's packs
    spec = scenes.load(scene)
    root = out / "bluemap"
    for sub in ("config", "data", "web"):
        shutil.rmtree(root / sub, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    dims = [d for d in bluemap.detect_dimensions(spec.world) if d.map_id == "overworld"]
    port = bluemap.free_port()
    packs = list(dict.fromkeys([*spec.extra_packs, *variant.bluemap_packs]))
    bluemap.write_config(root, spec.world, dims, jars, port, packs)
    (root / "packs.txt").write_text("".join(f"{p}\n" for p in packs), encoding="utf-8")
    views: list[ViewOutcome] = []
    with bluemap.BlueMapCli(jars, root, port) as cli:
        timings["bluemap_render"] = round(cli.render_seconds, 1)
        for cam in camera_list:
            view = ViewOutcome(cam.name, cam.source)
            views.append(view)
            if cam not in captured:
                view.error = "no reference capture"
                continue
            t0 = time.monotonic()
            client = cameras.ClientView.load(client_dir, cam.name)
            centre, radius = area(client, keys)
            capture = webapp.capture(browser, cli.url, "overworld", cameras.webapp_camera(client), centre, radius,
                                     client.width, client.height)
            t1 = time.monotonic()
            view.faces, view.alignment, view.checks = compare_view(client, capture, blocks, keys, thresholds, out)
            view.seconds = {"webapp": round(t1 - t0, 1), "compare": round(time.monotonic() - t1, 1),
                            "client": round(float(client.meta.get("seconds", 0.0)), 1)}
            print(f"[oracle] {scene}/{cam.name}: {len(view.faces)} faces, alignment "
                  f"{'ok' if view.alignment['passed'] else 'FAILED'} (shift {view.alignment['shift']}, "
                  f"edge median {view.alignment['edge_error_median_px']} px, p90 {view.alignment['edge_error_p90_px']} px), "
                  f"webapp {view.seconds['webapp']}s, compare {view.seconds['compare']}s", flush=True)
        log_errors = cli.log_lines("ERROR")

    calibration = None
    if thresholds is None:
        thresholds, calibration = calibrate(views)
        judge(views, thresholds)
    else:
        calibration = json.loads(calibration_file().read_text(encoding="utf-8"))
    timings["total"] = round(time.monotonic() - started, 1)
    report = {
        "scene": scene,
        "variant": variant.name,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bluemap_packs": [str(p) for p in packs],
        "client": {"dir": str(variant.client_dir), "resource_packs": [str(p) for p in variant.resource_packs],
                   "extra_mods": [str(p) for p in variant.extra_mods]},
        "thresholds": thresholds.to_json(),
        "calibration": calibration,
        "summary": summarize(views),
        "excluded": {"cells": len(removed), "states": removed_states},
        "timings": timings,
        "bluemap_log_errors": log_errors[:20],
        "views": [{"name": v.name, "source": v.source, "error": v.error, "alignment": v.alignment,
                   "checks": v.checks, "seconds": v.seconds,
                   "summary": summarize([v]) if v.alignment else None} for v in views],
        "faces": [f for v in views for f in v.faces],
    }
    world_report.write(out, report, CROPS)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    update_summary(variant)
    return report


def update_summary(variant: variants.Variant) -> Path:
    """``out/oracle/<variant>/worlds/summary.json``: the summary of every scene report of the variant."""
    root = variants.oracle_dir(variant) / "worlds"
    scenes_summary = {}
    for report in sorted(root.glob("*/report.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        scenes_summary[data["scene"]] = {"summary": data["summary"], "timings": data["timings"], "created": data["created"]}
    path = root / "summary.json"
    path.write_text(json.dumps({"variant": variant.name, "scenes": scenes_summary}, indent=2) + "\n", encoding="utf-8")
    return path


def cmd_world(args: argparse.Namespace) -> int:
    variant = variants.load(args.variant)
    # BlueMap packs are always needed; the client, its packs and mods only for a reference capture
    needed = variant.bluemap_packs if args.capture == "never" else variant.missing()
    missing = [p for p in needed if not p.exists()]
    if missing:
        raise FileNotFoundError(f"variant {variant.name}: missing {', '.join(str(p) for p in missing)}")
    jars = runner.prepare_jars(build=not args.no_build)
    failed = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=CHROMIUM_ARGS)
        try:
            for scene in args.scene:
                report = run_scene(variant, scene, jars, browser, args)
                s = report["summary"]
                rate = "-" if s["pass_rate"] is None else f"{s['pass_rate']:.2%}"
                print(f"[oracle] {variant.name}/{scene}: {s['passed']}/{s['judged']} faces pass ({rate}), "
                      f"{s['views_aligned']}/{s['views']} views aligned, {report['timings']['total']:.0f}s; "
                      f"{scene_dir(variant, scene) / 'report.html'}", flush=True)
                if s["views_aligned"] < s["views"]:
                    failed.append(scene)
        finally:
            browser.close()
    return 1 if failed else 0


def add_parser(oracle_sub: argparse._SubParsersAction) -> None:
    p = oracle_sub.add_parser("world", help="BlueMap's webapp vs real-client screenshots of scene worlds, per block face")
    p.add_argument("--variant", default="default", choices=variants.NAMES, help="client/pack variant (default %(default)s)")
    p.add_argument("--scene", action="append", required=True, help="scene (repeatable)")
    p.add_argument("--capture", choices=("auto", "always", "never"), default="auto",
                   help="reference capture with gtnh-client cameras: when missing or the cameras changed (auto), "
                        "always, or never (default %(default)s)")
    p.add_argument("--mod-jar", type=Path, help="exporter jar for the reference capture (default: the launcher builds it)")
    p.add_argument("--no-build", action="store_true", help="use the existing BlueMap CLI/addon jars")
    p.set_defaults(func=cmd_world)
