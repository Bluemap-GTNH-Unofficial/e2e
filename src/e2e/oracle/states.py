"""`e2e oracle states`: BlueMap's rendering of every block state against the real GTNH client's, in isolation.

Per variant (outputs in ``out/oracle/<variant>/states/``, never committed):

1. Reference: ``client/`` from ``gtnh-client states`` (sibling gtnh-exporter launcher): every block state the
   exporter knows (vanilla ones from the fork's flattening table), drawn alone by the real client at two fixed
   orthographic views, with per-pixel face IDs and ``client/index.json``.
2. BlueMap: ``bluemap/`` from the fork module ``oracle`` (``./gradlew :oracle:renderStates``): the same states
   through BlueMap's hires pipeline (legacy chunk reader and flattening, gtnh-render renderers, the variant's packs),
   rasterised with the same camera.
3. Comparison per view: the whole view (``view`` thresholds) and each axis face, by the face IDs of both sides
   (``face`` thresholds), with the shared region metrics (``oracle.metrics``).
4. ``report.json`` and ``report.html`` (pass rates per mod, worst states first with reference / BlueMap / diff).
   States ``oracle/accepted.yaml`` names, where the 1.7.10 vs 1.13 data (``oracle.vanilla``) backs the reason, are
   reported as ``accepted (1.13)``, neither passing nor failing.

Thresholds come from the vanilla states of the default variant that look the same in 1.7.10 and 1.13
(``out/oracle/default/states/calibration.json``, Tukey's far-out fences over their inexact samples,
``metrics.calibrate``): that run classifies the vanilla states (``vanilla.json``) and writes the calibration, every
other run reads both. ``e2e oracle swap-test`` measures how many genuinely different renders the thresholds flag.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from e2e import bluemap, paths
from e2e.oracle import accepted, metrics, states_report, vanilla, variants

VIEWS = ("se", "nw")
VIEW_SIZE = 128
FACES = {1: "down", 2: "up", 3: "north", 4: "south", 5: "west", 6: "east", 7: "other"}
JUDGED_FACES = (1, 2, 3, 4, 5, 6)
CALIBRATION_VARIANT = "default"
MIN_PIXELS = 32  # regions whose coverage union is smaller are reported but not judged
WORST = 100  # worst states with images in the report


def states_dir(variant: variants.Variant | str) -> Path:
    return variants.oracle_dir(variant) / "states"


def calibration_file() -> Path:
    return states_dir(CALIBRATION_VARIANT) / "calibration.json"


# --- BlueMap side ------------------------------------------------------------------------------------------------

def render_bluemap(variant: variants.Variant, out: Path, log: Path, threads: int | None = None) -> None:
    """Runs the fork's ``:oracle:renderStates`` for the client index of ``out``."""
    repo = paths.bluemap_repo()
    args = ["--index", str(out / "client" / "index.json"), "--out", str(out / "bluemap"),
            "--mc-cache", str(paths.download_cache())]
    for pack in variant.bluemap_packs:
        args += ["--pack", str(pack)]
    if threads:
        args += ["--threads", str(threads)]
    cmd = ["mise", "exec", "--", "./gradlew", "--console=plain", ":oracle:renderStates", f"--args={subprocess.list2cmdline(args)}"]
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[states] BlueMap side: {subprocess.list2cmdline(cmd)} (log {log})", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(cmd, cwd=repo, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = "".join(log.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-30:])
        raise bluemap.BlueMapError(f":oracle:renderStates failed (exit {result.returncode}), log {log}:\n{tail}")


# --- comparison --------------------------------------------------------------------------------------------------

def load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def load_faces(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode != "P":
            raise ValueError(f"{path}: face IDs must be a palette PNG, got mode {image.mode}")
        return np.asarray(image, dtype=np.uint8)


def _mean_lab(rgba: np.ndarray, mask: np.ndarray) -> list[float] | None:
    if not mask.any():
        return None
    return [round(float(v), 2) for v in metrics.srgb_to_lab(rgba[..., :3][mask]).mean(axis=0)]


def compare_images(ref: np.ndarray, test: np.ndarray, ref_faces: np.ndarray, test_faces: np.ndarray) -> dict:
    """Raw metrics of both views: the whole view and every face ID either side shows."""
    views = {}
    for v, name in enumerate(VIEWS):
        cols = slice(v * VIEW_SIZE, (v + 1) * VIEW_SIZE)
        r, t, rf, tf = ref[:, cols], test[:, cols], ref_faces[:, cols], test_faces[:, cols]
        overall = metrics.compare_region(r, t)
        both = (r[..., 3] >= metrics.ALPHA_MIN) & (t[..., 3] >= metrics.ALPHA_MIN)
        faces = {}
        for face in FACES:
            rm, tm = rf == face, tf == face
            if rm.any() or tm.any():
                faces[face] = metrics.compare_region(r, t, rm, tm)
        views[name] = {"overall": overall, "faces": faces, "lab": [_mean_lab(r, both), _mean_lab(t, both)]}
    return views


def _compare_job(job: tuple[str, str, str]) -> tuple[str, dict | None, str | None]:
    path, client_root, bluemap_root = job
    try:
        ref = load_rgba(Path(client_root) / f"{path}.png")
        test = load_rgba(Path(bluemap_root) / f"{path}.png")
        ref_faces = load_faces(Path(client_root) / f"{path}.faces.png")
        test_faces = load_faces(Path(bluemap_root) / f"{path}.faces.png")
        return path, compare_images(ref, test, ref_faces, test_faces), None
    except (OSError, ValueError) as e:
        return path, None, str(e)


@dataclass
class Judge:
    view: metrics.Thresholds
    face: metrics.Thresholds

    def check(self, views: dict) -> list[str]:
        failed = []
        for name, data in views.items():
            failed += [f"{name}: {c}" for c in self.view.check(data["overall"])]
            for face in JUDGED_FACES:
                m = data["faces"].get(face)
                if m is not None:
                    failed += [f"{name} {FACES[face]}: {c}" for c in self.face.check(m)]
        return failed

    def to_json(self) -> dict:
        return {"view": self.view.to_json(), "face": self.face.to_json()}


def score(views: dict) -> float:
    """Badness for ordering, 0 = identical: per view (1 - IoU) + mean CIEDE2000 / 50 + (1 - SSIM) / 2, NaN as worst."""
    total = 0.0
    for data in views.values():
        m = data["overall"]
        de = 50.0 if math.isnan(m.delta_e) else min(m.delta_e, 50.0)
        ss = -1.0 if math.isnan(m.ssim) else m.ssim
        total += (1 - m.iou) + de / 50 + (1 - ss) / 2
    return total / max(1, len(views))


def suspected_cause(result: dict) -> str:
    """A heuristic first guess from the metrics and both indices (the report's worst list is reviewed by hand)."""
    client, bm = result["client"], result["bluemap"]
    if bm.get("error"):
        return f"BlueMap failed: {bm['error']}"
    if bm.get("blank"):
        return "BlueMap draws nothing"
    if bm.get("missingTexture"):
        return "BlueMap draws the missing model or texture"
    views = result["views"]
    worst_iou = min(v["overall"]["iou"] for v in views.values())
    if client.get("tesr") and not client.get("quads"):
        return "tile-entity special renderer only (no block geometry captured)"
    if worst_iou < 0.8:
        return "silhouette differs (geometry)"
    checks = " ".join(result["checks"])
    lab = [v.get("meanLab") or [None, None] for v in views.values()]
    if "delta_e" in checks:
        shifts = [(t[0] - r[0], math.hypot(t[1] - r[1], t[2] - r[2])) for r, t in lab if r and t]
        if shifts and all(abs(dl) > 2 * dc for dl, dc in shifts):
            return "brightness differs (" + ("BlueMap darker" if shifts[0][0] < 0 else "BlueMap brighter") + ")"
        return "colour or tint differs"
    if "ssim" in checks:
        return "texture pattern differs (texture, rotation or UV)"
    if "iou" in checks:
        return "face geometry or orientation differs"
    return ""


# --- report data -------------------------------------------------------------------------------------------------

def _views_json(views: dict) -> dict:
    return {
        name: {
            "overall": data["overall"].to_json(),
            "faces": {FACES[f]: m.to_json() for f, m in data["faces"].items()},
            "meanLab": data["lab"],
        }
        for name, data in views.items()
    }


def _regions(rs: list[dict], raw: dict[str, dict]) -> dict[str, tuple[list, list[str]]]:
    """View and face regions of ``rs`` with the block each belongs to."""
    out: dict[str, tuple[list, list[str]]] = {"view": ([], []), "face": ([], [])}
    for r in rs:
        for data in raw[r["path"]].values():
            out["view"][0].append(data["overall"])
            out["view"][1].append(r["name"])
            for f, m in data["faces"].items():
                if f in JUDGED_FACES:
                    out["face"][0].append(m)
                    out["face"][1].append(r["name"])
    return out


def calibration_sets(results: list[dict], classes: dict, accepted_list: accepted.Accepted) -> dict[str, list[dict]]:
    """The ``texture`` and ``geometry`` sets of the 1.7.10 vs 1.13 classification (``oracle.vanilla``), limited to
    states both sides drew (status still ``pending``) that ``accepted.yaml`` does not name."""
    by_key = {r["key"]: r for r in results}
    return {name: [by_key[k] for k in classes["sets"][name] if k in by_key and by_key[k]["status"] == "pending"
                   and accepted_list.entry(by_key[k]["name"], by_key[k]["meta"]) is None]
            for name in ("texture", "geometry")}


def false_failures(judge: Judge, sets: dict[str, list[dict]], raw: dict[str, dict]) -> dict:
    """Per calibration set: the states the thresholds fail (every one of them looks the same in 1.7.10 and 1.13)."""
    out = {}
    for name, rs in sets.items():
        failing = [(r["key"], judge.check(raw[r["path"]])) for r in rs]
        failing = [(k, c) for k, c in failing if c]
        out[name] = {"states": len(rs), "fail": len(failing), "rate": len(failing) / len(rs) if rs else None,
                     "failing": [{"key": k, "checks": c[:4]} for k, c in failing]}
    return out


def calibrate(results: list[dict], raw: dict[str, dict], classes: dict,
              accepted_list: accepted.Accepted) -> tuple[Judge, dict]:
    """Thresholds from the vanilla states that look the same in 1.7.10 and 1.13: IoU over the regions of the
    ``geometry`` set, delta E and SSIM over the ``texture`` set with one sample per block, its worst inexact region
    (``calibration_sets``, ``metrics.calibrate``)."""
    sets = calibration_sets(results, classes, accepted_list)
    geometry, texture = _regions(sets["geometry"], raw), _regions(sets["texture"], raw)
    view_cal, face_cal = (metrics.calibrate(texture[level][0], geometry_samples=geometry[level][0],
                                            groups=texture[level][1], min_pixels=MIN_PIXELS)
                          for level in ("view", "face"))
    judge = Judge(view_cal.thresholds, face_cal.thresholds)
    data = {
        "format": 2,
        "variant": CALIBRATION_VARIANT,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "states": len(sets["texture"]),
        "geometryStates": len(sets["geometry"]),
        "classification": str(vanilla.file()),
        "geometryBand": classes["geometryBand"],
        "sets": {name: [r["key"] for r in rs] for name, rs in sets.items()},
        "excluded": {f"{n}[meta={m}]": why for (n, m), why in vanilla.NOT_IN_1_7_10.items()},
        "view": view_cal.to_json(),
        "face": face_cal.to_json(),
        "falseFailures": false_failures(judge, sets, raw),
    }
    return judge, data


def load_judge(file: Path) -> Judge:
    data = json.loads(file.read_text(encoding="utf-8"))
    return Judge(metrics.Thresholds.from_json(data["view"]["thresholds"]), metrics.Thresholds.from_json(data["face"]["thresholds"]))


def diff_image(ref: np.ndarray, test: np.ndarray) -> np.ndarray:
    """RGB: CIEDE2000 heat (black 0 .. red 25 .. yellow 50+) where both draw, magenta = only the client draws,
    green = only BlueMap draws, dark grey = neither."""
    rc, tc = ref[..., 3] >= metrics.ALPHA_MIN, test[..., 3] >= metrics.ALPHA_MIN
    out = np.full(ref.shape[:2] + (3,), 32, np.uint8)
    both = rc & tc
    if both.any():
        de = metrics.ciede2000(metrics.srgb_to_lab(ref[..., :3][both]), metrics.srgb_to_lab(test[..., :3][both]))
        heat = np.clip(de / 25.0, 0, 2)
        out[both] = np.stack([np.clip(heat, 0, 1) * 255, np.clip(heat - 1, 0, 1) * 255, np.zeros_like(heat)], axis=1)
    out[rc & ~tc] = (255, 0, 255)
    out[tc & ~rc] = (0, 200, 0)
    return out


def run(variant: variants.Variant, *, render: bool, only: str | None, workers: int, recalibrate: bool) -> int:
    started = time.monotonic()
    out = states_dir(variant)
    client_index_file = out / "client" / "index.json"
    if not client_index_file.is_file():
        raise FileNotFoundError(
            f"{client_index_file} missing: run `uv run gtnh-client states --variant {variant.name} ...` in "
            f"gtnh-exporter/launcher first (see the e2e README)")
    missing = [p for p in variant.bluemap_packs if not p.exists()]
    if missing:
        raise FileNotFoundError(f"variant {variant.name}: missing {', '.join(map(str, missing))}")
    t_render = time.monotonic()
    if render:
        render_bluemap(variant, out, out / "bluemap.log")
    t_compare = time.monotonic()
    client_index = json.loads(client_index_file.read_text(encoding="utf-8"))
    bluemap_index_file = out / "bluemap" / "index.json"
    bluemap_index = json.loads(bluemap_index_file.read_text(encoding="utf-8")) if bluemap_index_file.is_file() else {"states": []}
    bm_states = {s["path"]: s for s in bluemap_index.get("states", [])}
    pattern = re.compile(only) if only else None
    states = [s for s in client_index["states"] if not pattern or pattern.search(s["key"]) or pattern.search(s["name"])]

    jobs = [(s["path"], str(out / "client"), str(out / "bluemap")) for s in states if s["path"] in bm_states]
    raw: dict[str, dict] = {}
    load_errors: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, views, error in pool.map(_compare_job, jobs, chunksize=64):
            if views is None:
                load_errors[path] = error or "unreadable"
            else:
                raw[path] = views

    results = []
    for s in states:
        bm = bm_states.get(s["path"], {})
        name = s["name"]
        result = {
            "key": s["key"], "path": s["path"], "name": name, "meta": s["meta"], "mod": name.partition(":")[0],
            "client": {k: s.get(k) for k in ("renderType", "class", "tileEntity", "tesr", "quads", "passes", "blank",
                                             "error", "teError", "worldMeta", "mte") if s.get(k) is not None},
            "bluemap": {k: bm.get(k) for k in ("blockState", "blockEntity", "variants", "triangles", "missingTexture",
                                               "blank", "error") if bm.get(k) is not None},
            "checks": [],
        }
        if (name, s["meta"]) in vanilla.NOT_IN_1_7_10:
            result["note"] = vanilla.NOT_IN_1_7_10[(name, s["meta"])]
        views = raw.get(s["path"])
        if s.get("error"):
            result["status"] = "client_error"
        elif not bm:
            result["status"] = "bluemap_missing"
        elif s["path"] in load_errors:
            result["status"] = "bluemap_missing"
            result["checks"] = [load_errors[s["path"]]]
        elif bm.get("error"):
            result["status"] = "fail"
            result["checks"] = [f"BlueMap error: {bm['error']}"]
        elif s.get("blank"):
            result["status"] = "pass" if bm.get("blank") else "client_blank"
        else:
            result["status"] = "pending"
        if views is not None:
            result["views"] = _views_json(views)
            result["score"] = round(score(views), 4)
        results.append(result)

    classes = classification(variant, {**client_index, "states": states}, bluemap_index, out, write=not only)
    accepted_list = accepted.load(classes, vanilla.texture_overrides(variant.bluemap_packs))
    calibration_written = False
    provisional = False
    if variant.name == CALIBRATION_VARIANT and not only and (recalibrate or not calibration_file().is_file()):
        judge, calibration = calibrate(results, raw, classes, accepted_list)
        calibration_file().write_text(json.dumps(calibration, indent=1), encoding="utf-8")
        calibration_written = True
    elif calibration_file().is_file():
        judge = load_judge(calibration_file())
    elif variant.name == CALIBRATION_VARIANT:
        # a subset run before any full run: judge with thresholds from the subset's vanilla states, write nothing
        judge, _ = calibrate(results, raw, classes, accepted_list)
        provisional = True
        print("[states] no calibration yet: provisional thresholds from this subset's vanilla states", flush=True)
    else:
        raise FileNotFoundError(f"{calibration_file()} missing: run `e2e oracle states --variant {CALIBRATION_VARIANT}` first")

    for r in results:
        if r["status"] != "pending":
            continue
        r["checks"] = judge.check(raw[r["path"]])
        entry, backed = accepted_list.check(r["name"], r["meta"])
        if entry is not None and backed:
            r["status"] = "accepted"
            r["accepted"] = accepted_list.describe(entry)
            continue
        if entry is not None:
            r["acceptedUnbacked"] = entry.states
        r["status"] = "fail" if r["checks"] else "pass"
    for r in results:
        if r["status"] in ("fail", "client_blank", "bluemap_missing"):
            r["cause"] = suspected_cause(r) if "views" in r else r["status"].replace("_", " ")

    t_end = time.monotonic()
    report = summarize(variant, results, judge, client_index, bluemap_index, {
        "bluemap": round(t_compare - t_render, 1) if render else None,
        "compare": round(t_end - t_compare, 1),
        "total": round(t_end - started, 1),
        "client": client_index.get("seconds"),
        "bluemapRender": bluemap_index.get("seconds"),
    })
    report["calibrationWritten"] = calibration_written
    report["provisionalThresholds"] = provisional
    report["accepted"] = {
        "file": str(accepted.FILE), "entries": len(accepted_list.entries),
        "states": sum(r["status"] == "accepted" for r in results),
        "unbacked": [r["key"] for r in results if r.get("acceptedUnbacked")],
        "unmatched": accepted_list.unmatched(),
    }
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    states_report.write(out, report, results, diff_image)
    totals = report["total"]
    print(f"[states] {variant.name}: {totals['pass']}/{totals['judged']} judged states pass "
          f"({totals['passRate']:.2%}), {totals['accepted']} {accepted.LABEL}; {totals['clientBlank']} client-blank, "
          f"{totals['clientError']} client errors, {totals['bluemapMissing']} without BlueMap output; compare "
          f"{t_end - t_compare:.0f}s; report {out / 'report.html'}")
    return 0 if totals["fail"] == 0 else 1


def summarize(variant: variants.Variant, results: list[dict], judge: Judge, client_index: dict, bluemap_index: dict,
              seconds: dict) -> dict:
    def counts(rs: list[dict]) -> dict:
        c = Counter(r["status"] for r in rs)
        judged = c["pass"] + c["fail"]
        return {"states": len(rs), "judged": judged, "pass": c["pass"], "fail": c["fail"],
                "passRate": c["pass"] / judged if judged else None, "accepted": c["accepted"],
                "clientBlank": c["client_blank"], "clientError": c["client_error"],
                "bluemapMissing": c["bluemap_missing"]}

    by_mod = defaultdict(list)
    for r in results:
        by_mod[r["mod"]].append(r)
    failing = sorted((r for r in results if r["status"] == "fail"), key=lambda r: -r.get("score", 0))
    causes = Counter(r.get("cause", "") for r in failing)
    return {
        "format": 2,
        "variant": variant.name,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bluemapPacks": [str(p) for p in variant.bluemap_packs],
        "packRoots": bluemap_index.get("packRoots"),
        "camera": client_index.get("image"),
        "conditions": client_index.get("conditions"),
        "thresholds": judge.to_json(),
        "calibration": str(calibration_file()),
        "seconds": seconds,
        "total": counts(results),
        "mods": {mod: counts(rs) for mod, rs in sorted(by_mod.items(), key=lambda e: -len(e[1]))},
        "causes": dict(causes.most_common()),
        "worst": [{k: r.get(k) for k in ("key", "mod", "score", "checks", "cause", "client", "bluemap")} for r in failing[:WORST]],
        "worstBlocks": [{k: r.get(k) for k in ("key", "mod", "score", "checks", "cause", "client", "bluemap")}
                        for r in states_report.worst_per_block(failing)[:WORST]],
        "states": [{k: r.get(k) for k in ("key", "status", "score", "cause", "note", "accepted", "acceptedUnbacked")
                    if r.get(k) is not None} | {"checks": r["checks"][:6]} for r in results],
    }


def classification(variant: variants.Variant, client_index: dict, bluemap_index: dict, out: Path, *,
                   write: bool) -> dict:
    """The 1.7.10 vs 1.13 classification of the vanilla states (``oracle.vanilla``): computed by the default variant
    (written with the full state set), read by every other variant."""
    if variant.name != CALIBRATION_VARIANT:
        return vanilla.load()
    classes = vanilla.classify(client_index, bluemap_index, out / "client", out / "bluemap", vanilla.sources(),
                               len(VIEWS))
    if write:
        vanilla.file().write_text(json.dumps(classes, indent=1), encoding="utf-8")
    return classes


# --- swap test -----------------------------------------------------------------------------------------------------

SWAP_MIN_DIFFERENCE = 1.0  # mean absolute RGBA difference from which two client renders count as different


def _swap_job(job: tuple[str, str, str]) -> tuple[str, dict | None]:
    path, reference_root, test_root = job
    try:
        ref = load_rgba(Path(reference_root) / f"{path}.png")
        test = load_rgba(Path(test_root) / f"{path}.png")
        if np.abs(ref.astype(np.int16) - test.astype(np.int16)).mean() < SWAP_MIN_DIFFERENCE:
            return path, None
        return path, compare_images(ref, test, load_faces(Path(reference_root) / f"{path}.faces.png"),
                                    load_faces(Path(test_root) / f"{path}.faces.png"))
    except (OSError, ValueError):
        return path, None


def swap_test(judges: dict[str, Judge], workers: int) -> dict:
    """Detection power of thresholds: the Modernity client's render of every state is the reference and the default
    client's render of the same state the test, for the states whose two renders differ (mean absolute RGBA difference
    at least ``SWAP_MIN_DIFFERENCE``). Every such state looks different, so a discriminating oracle flags all of them.
    Also the false failures of each judge on the current calibration sets (states that look the same)."""
    reference, test = states_dir("modernity") / "client", states_dir(CALIBRATION_VARIANT) / "client"
    index = json.loads((test / "index.json").read_text(encoding="utf-8"))["states"]
    jobs = [(s["path"], str(reference), str(test)) for s in index if not s.get("error") and not s.get("blank")]
    differing: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for path, views in pool.map(_swap_job, jobs, chunksize=64):
            if views is not None:
                differing[path] = views
    calibration = json.loads(calibration_file().read_text(encoding="utf-8"))
    keys = {name: set(calibration["sets"][name]) for name in ("texture", "geometry")}
    by_key = {s["key"]: s["path"] for s in index}
    out = states_dir(CALIBRATION_VARIANT)
    calibration_raw = {}
    for path, views, _ in map(_compare_job, [(by_key[k], str(out / "client"), str(out / "bluemap"))
                                             for k in set().union(*keys.values())]):
        if views is not None:
            calibration_raw[path] = views
    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "differingStates": len(differing),
              "minDifference": SWAP_MIN_DIFFERENCE, "judges": {}}
    for name, judge in judges.items():
        flagged = sum(bool(judge.check(views)) for views in differing.values())
        ff = {}
        for set_name, members in keys.items():
            failing = [k for k in members if by_key[k] in calibration_raw and judge.check(calibration_raw[by_key[k]])]
            ff[set_name] = {"states": len(members), "fail": len(failing),
                            "rate": len(failing) / len(members) if members else None}
        result["judges"][name] = {"thresholds": judge.to_json(), "flagged": flagged,
                                  "detection": flagged / len(differing) if differing else None, "falseFailures": ff}
    return result


def cmd_swap_test(args: argparse.Namespace) -> int:
    judges = {"current": load_judge(calibration_file())}
    if args.before:
        judges = {"before": load_judge(args.before), **judges}
    result = swap_test(judges, args.workers)
    target = states_dir(CALIBRATION_VARIANT) / "swap-test.json"
    target.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for name, j in result["judges"].items():
        ff = j["falseFailures"]
        print(f"[swap-test] {name}: flags {j['flagged']}/{result['differingStates']} differing states "
              f"({j['detection']:.1%}); false failures on the calibration sets: texture {ff['texture']['fail']}/"
              f"{ff['texture']['states']}, geometry {ff['geometry']['fail']}/{ff['geometry']['states']}")
    print(f"[swap-test] {target}")
    return 0


def add_parser(oracle_sub: argparse._SubParsersAction) -> None:
    p = oracle_sub.add_parser("states", help="BlueMap vs the real client, every block state in isolation")
    p.add_argument("--variant", default="default", choices=variants.NAMES, help="client/pack variant (default %(default)s)")
    p.add_argument("--no-render", action="store_true", help="compare the existing bluemap/ output instead of rendering it")
    p.add_argument("--only", help="only states whose key or registry name matches this regex")
    p.add_argument("--recalibrate", action="store_true",
                   help=f"recompute the thresholds from this run's vanilla states (variant {CALIBRATION_VARIANT} only)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="comparison processes")
    p.set_defaults(func=cmd_states)
    s = oracle_sub.add_parser("swap-test", help="detection power of the state thresholds: Modernity client renders "
                                                "against default client renders of the same states")
    s.add_argument("--before", type=Path, help="also evaluate the thresholds of this earlier calibration.json")
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2), help="comparison processes")
    s.set_defaults(func=cmd_swap_test)


def cmd_states(args: argparse.Namespace) -> int:
    if args.recalibrate and args.variant != CALIBRATION_VARIANT:
        raise SystemExit(f"--recalibrate only applies to --variant {CALIBRATION_VARIANT}")
    if args.only and args.recalibrate:
        raise SystemExit("--recalibrate needs the full state set (no --only)")
    return run(variants.load(args.variant), render=not args.no_render, only=args.only, workers=args.workers,
               recalibrate=args.recalibrate)
