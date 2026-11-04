"""`e2e oracle world-swap-test`: how well the world oracle's thresholds discriminate.

Per scene and direction, BlueMap's captures of one variant are judged against the *other* variant's client captures
of the same cameras (default BlueMap against the Modernity client, and Modernity BlueMap against the default client).
The faces counted are those that pass in the test variant's own report (its BlueMap matches its own client there)
and that the two clients draw differently (mean absolute RGB difference of the two client captures over the pixels
either client shows of the face, at least ``MIN_DIFFERENCE``): each of them looks different in the reference client,
so a discriminating oracle flags every one; the share flagged is the detection rate. Faces that pass in their own
report and that the two clients draw alike (difference below ``SAME_DIFFERENCE``) should still pass; the share that
fails is the swap's false-failure rate. Writes ``out/oracle/default/worlds/swap-test.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from e2e.oracle import camera as cameras
from e2e.oracle import faces, metrics, variants, world

MIN_DIFFERENCE = 1.0  # mean absolute RGB difference (0..255): the two clients draw the face differently
SAME_DIFFERENCE = 0.25  # below this the two clients draw the face alike
DIFFERENCE_BINS = [1.0, 2.0, 5.0, 10.0, 20.0]
SCENES = ("vanilla-zoo", "gt-multiblocks", "gtnh-zoo")
DIRECTIONS = (("default", "modernity"), ("modernity", "default"))  # (BlueMap variant, client variant)
MATRIX_TOLERANCE = 1e-4


def _bluemap(out: Path, name: str, client: cameras.ClientView) -> tuple[np.ndarray, np.ndarray] | None:
    depth = out / "bluemap" / f"{name}.depth.bin"
    if not depth.is_file():
        return None
    with Image.open(out / "bluemap" / f"{name}.png") as image:
        rgb = np.asarray(image.convert("RGB"))
    return rgb, np.fromfile(depth, dtype="<f4").reshape(client.height, client.width)


def _same_camera(a: cameras.ClientView, b: cameras.ClientView) -> bool:
    return (a.image.shape == b.image.shape and np.abs(a.projection - b.projection).max() < MATRIX_TOLERANCE
            and np.abs(a.view - b.view).max() < MATRIX_TOLERANCE)


def _client_difference(a: cameras.ClientView, a_labels: faces.Labels, b: cameras.ClientView,
                       b_labels: faces.Labels, keys: list[int]) -> dict[int, float]:
    """Per face key: mean absolute RGB difference of the two client captures over the pixels either shows it on."""
    diff = np.abs(a.image[..., :3].astype(np.int16) - b.image[..., :3].astype(np.int16)).mean(axis=-1)
    wanted = np.asarray(keys, dtype=np.int64)
    a_key = np.where(np.isin(a_labels.key, wanted), a_labels.key, -1)
    b_key = np.where(np.isin(b_labels.key, wanted), b_labels.key, -1)
    a_groups, b_groups = faces.group(a_key), faces.group(b_key)
    flat = diff.reshape(-1)
    out = {}
    for key in keys:
        index = np.union1d(a_groups.get(key, np.empty(0, np.int64)), b_groups.get(key, np.empty(0, np.int64)))
        if index.size:
            out[key] = float(flat[index].mean())
    return out


def direction(scene: str, bluemap_variant: str, client_variant: str, thresholds: dict[str, metrics.Thresholds]) -> dict:
    test_v, ref_v = variants.load(bluemap_variant), variants.load(client_variant)
    test_dir, ref_dir = world.scene_dir(test_v, scene), world.scene_dir(ref_v, scene)
    own = json.loads((test_dir / "report.json").read_text(encoding="utf-8"))
    ref_report = json.loads((ref_dir / "report.json").read_text(encoding="utf-8"))
    ref_aligned = {v["name"] for v in ref_report["views"] if v["alignment"] and v["alignment"]["passed"]}
    own_passed = {(f["view"], f["x"], f["y"], f["z"], f["face"]) for f in own["faces"] if f["passed"]}
    blocks, removed, _ = world.scene_blocks(scene)
    keys = world.dump_keys(blocks)
    accepted_list = world.load_accepted(own["bluemap_packs"])
    bg = world.backgrounds(test_v, blocks, removed, accepted_list)
    views, differences, skipped = [], {}, []
    for v in own["views"]:
        name = v["name"]
        if v["error"] or not v["alignment"] or not v["alignment"]["passed"] or name not in ref_aligned:
            skipped.append({"view": name, "reason": "not aligned in both reports"})
            continue
        ref_client = cameras.ClientView.load(ref_dir / "client", name)
        own_client = cameras.ClientView.load(test_dir / "client", name)
        if not _same_camera(ref_client, own_client):
            skipped.append({"view": name, "reason": "the two clients' cameras differ"})
            continue
        capture = _bluemap(test_dir, name, own_client)
        if capture is None:
            skipped.append({"view": name, "reason": "no saved BlueMap depth"})
            continue
        rgb, distance = capture
        reference, _, ref_world = world.reference_labels(ref_client)
        own_labels, _, _ = world.reference_labels(own_client)
        test, test_world = world.bluemap_labels(ref_client, distance)
        excluded = world.colour_excluded(ref_client, reference, ref_world, test, test_world, bg)
        records = world.face_records(ref_client, reference, test, rgb, blocks, keys, excluded)
        by_key = {faces.encode(np.array([f["x"], f["y"], f["z"]]), np.array(faces.FACES.index(f["face"]))): f
                  for f in records}
        for key, d in _client_difference(ref_client, reference, own_client, own_labels, list(by_key)).items():
            differences[id(by_key[key])] = d
        views.append(world.ViewOutcome(name, v["source"], records, v["alignment"]))
    result = {"scene": scene, "bluemap": bluemap_variant, "client": client_variant, "skipped_views": skipped,
              "judges": {}}
    for judge_name, t in thresholds.items():
        world.judge(views, t, accepted_list)
        counted = [f for view in views for f in view.faces
                   if f["judged"] and not f.get("accepted") and id(f) in differences
                   and (f["view"], f["x"], f["y"], f["z"], f["face"]) in own_passed]
        differing = [f for f in counted if differences[id(f)] >= MIN_DIFFERENCE]
        same = [f for f in counted if differences[id(f)] < SAME_DIFFERENCE]
        flagged = [f for f in differing if not f["passed"]]
        bins = {}
        edges = [*DIFFERENCE_BINS, float("inf")]
        for low, high in zip(edges, edges[1:]):
            members = [f for f in differing if low <= differences[id(f)] < high]
            bins[f"{low:g}-{high:g}"] = {"faces": len(members), "flagged": sum(not f["passed"] for f in members)}
        missed: dict[str, int] = {}
        for f in differing:
            if f["passed"]:
                missed[f["label"]] = missed.get(f["label"], 0) + 1
        result["judges"][judge_name] = {
            "thresholds": t.to_json(),
            "own_passing_faces": len(counted),
            "differing": len(differing), "flagged": len(flagged),
            "differing_colour_excluded": sum(f["delta_e"] is None and f["both_pixels"] > 0 for f in differing),
            "detection": round(len(flagged) / len(differing), 4) if differing else None,
            "same": len(same), "same_failing": sum(not f["passed"] for f in same),
            "false_failure_rate": round(sum(not f["passed"] for f in same) / len(same), 4) if same else None,
            "by_difference": bins,
            "missed_states": dict(sorted(missed.items(), key=lambda kv: -kv[1])[:30]),
        }
    return result


def cmd_world_swap_test(args: argparse.Namespace) -> int:
    thresholds = {"current": metrics.Thresholds.from_json(
        json.loads(world.calibration_file().read_text(encoding="utf-8"))["thresholds"])}
    if args.before:
        thresholds = {"before": metrics.Thresholds.from_json(
            json.loads(args.before.read_text(encoding="utf-8"))["thresholds"]), **thresholds}
    scenes = args.scene or list(SCENES)
    results = []
    for scene in scenes:
        for bluemap_variant, client_variant in DIRECTIONS:
            r = direction(scene, bluemap_variant, client_variant, thresholds)
            results.append(r)
            for name, j in r["judges"].items():
                rate = "-" if j["detection"] is None else f"{j['detection']:.1%}"
                ff = "-" if j["false_failure_rate"] is None else f"{j['false_failure_rate']:.1%}"
                print(f"[world-swap] {scene}: {bluemap_variant} BlueMap vs {client_variant} client, {name}: flags "
                      f"{j['flagged']}/{j['differing']} differing faces ({rate}); {j['same_failing']}/{j['same']} "
                      f"alike faces fail ({ff})", flush=True)
    totals = {}
    for name in thresholds:
        js = [r["judges"][name] for r in results]
        differing, flagged = sum(j["differing"] for j in js), sum(j["flagged"] for j in js)
        same, same_failing = sum(j["same"] for j in js), sum(j["same_failing"] for j in js)
        bins = {b: {"faces": sum(j["by_difference"][b]["faces"] for j in js),
                    "flagged": sum(j["by_difference"][b]["flagged"] for j in js)} for b in js[0]["by_difference"]}
        totals[name] = {"differing": differing, "flagged": flagged,
                        "detection": round(flagged / differing, 4) if differing else None,
                        "differing_colour_excluded": sum(j["differing_colour_excluded"] for j in js),
                        "same": same, "same_failing": same_failing,
                        "false_failure_rate": round(same_failing / same, 4) if same else None,
                        "by_difference": bins}
        print(f"[world-swap] total, {name}: flags {flagged}/{differing} differing faces "
              f"({totals[name]['detection']:.1%}); {same_failing}/{same} alike faces fail; by client difference "
              + ", ".join(f"{b}: {v['flagged']}/{v['faces']}" for b, v in bins.items()), flush=True)
    target = world.calibration_file().parent / "swap-test.json"
    target.write_text(json.dumps({"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "min_difference": MIN_DIFFERENCE,
                                  "same_difference": SAME_DIFFERENCE, "totals": totals, "results": results},
                                 indent=1) + "\n", encoding="utf-8")
    print(f"[world-swap] {target}")
    return 0


def add_parser(oracle_sub: argparse._SubParsersAction) -> None:
    p = oracle_sub.add_parser("world-swap-test", help="detection power of the world thresholds: each variant's BlueMap "
                                                      "captures judged against the other variant's client captures")
    p.add_argument("--scene", action="append", help=f"scene (repeatable, default {', '.join(SCENES)})")
    p.add_argument("--before", type=Path, help="also evaluate the thresholds of this earlier calibration.json")
    p.set_defaults(func=cmd_world_swap_test)
