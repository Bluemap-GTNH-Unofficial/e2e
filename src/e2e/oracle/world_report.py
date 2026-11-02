"""Images and the static HTML report of the world oracle (``out/oracle/<variant>/worlds/<scene>/report.html``)."""

from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image

from e2e.oracle import faces

CROP_TARGET_HEIGHT = 120  # crops are scaled up (nearest) to at least this height


def _save(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def diff_image(reference: np.ndarray, test: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Per-pixel max channel difference, x3, in red over the dimmed reference; outside ``mask`` only the dimmed
    reference (pixels where neither rendering drew anything, i.e. sky)."""
    delta = np.abs(reference[..., :3].astype(np.int16) - test[..., :3].astype(np.int16)).max(axis=-1)
    if mask is not None:
        delta = np.where(mask, delta, 0)
    grey = (reference[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)) * 0.35
    out = np.repeat(grey[..., None], 3, axis=-1)
    out[..., 0] = np.maximum(out[..., 0], np.minimum(delta * 3, 255))
    return out.astype(np.uint8)


def label_image(reference: faces.Labels, test: faces.Labels, keys: np.ndarray) -> np.ndarray:
    """Dump-face label agreement: grey = same face in both, red = the client's face differs from BlueMap's, blue =
    a dump face only BlueMap shows there; dark grey = surfaces outside the dump."""
    ref_in = np.isin(reference.key, keys)
    test_in = np.isin(test.key, keys)
    out = np.zeros(reference.key.shape + (3,), dtype=np.uint8)
    out[(reference.key >= 0) | (test.key >= 0)] = (60, 60, 60)
    same = ref_in & (reference.key == test.key)
    out[same] = (170, 170, 170)
    out[ref_in & ~same] = (230, 40, 40)
    out[test_in & ~ref_in] = (40, 90, 230)
    return out


def save_view_images(out: Path, client, capture, reference: faces.Labels, test: faces.Labels, keys: np.ndarray) -> None:
    name = client.name
    _save(capture.rgba[..., :3], out / "bluemap" / f"{name}.png")
    capture.distance.astype("<f4").tofile(out / "bluemap" / f"{name}.depth.bin")
    drawn = (reference.key >= 0) | (test.key >= 0)
    _save(diff_image(client.image, capture.rgba, drawn), out / "report" / f"{name}-diff.png")
    _save(label_image(reference, test, keys), out / "report" / f"{name}-labels.png")
    np.savez_compressed(out / "report" / f"{name}-labels.npz", reference=reference.key, test=test.key)


def _outline(region: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(region)
    edge[1:, :] |= region[1:, :] & ~region[:-1, :]
    edge[:-1, :] |= region[:-1, :] & ~region[1:, :]
    edge[:, 1:] |= region[:, 1:] & ~region[:, :-1]
    edge[:, :-1] |= region[:, :-1] & ~region[:, 1:]
    return edge


def crop(out: Path, face: dict, labels_cache: dict, images_cache: dict) -> str:
    """``report/crops/<view>-<x>_<y>_<z>-<face>.png``: client | BlueMap | difference around the face, regions
    outlined (yellow: the client's face region, cyan: BlueMap's)."""
    view = face["view"]
    if view not in images_cache:
        images_cache[view] = (np.asarray(Image.open(out / "client" / f"{view}.png").convert("RGB")),
                              np.asarray(Image.open(out / "bluemap" / f"{view}.png").convert("RGB")))
        labels = np.load(out / "report" / f"{view}-labels.npz")
        labels_cache[view] = (labels["reference"], labels["test"])
    reference, test = images_cache[view]
    ref_keys, test_keys = labels_cache[view]
    h, w = ref_keys.shape
    x0, y0, x1, y1 = face["bbox"]
    margin = max(4, int(0.35 * max(x1 - x0, y1 - y0)))
    x0, y0, x1, y1 = max(0, x0 - margin), max(0, y0 - margin), min(w, x1 + margin), min(h, y1 + margin)
    key = int(faces.encode(np.array([face["x"], face["y"], face["z"]]), np.array(faces.FACES.index(face["face"]))))
    box = (slice(y0, y1), slice(x0, x1))
    a, b = reference[box].copy(), test[box].copy()
    a[_outline(ref_keys[box] == key)] = (255, 230, 0)
    b[_outline(test_keys[box] == key)] = (0, 230, 255)
    d = diff_image(reference[box], test[box])
    gap = np.full((y1 - y0, 2, 3), 255, dtype=np.uint8)
    strip = np.concatenate([a, gap, b, gap, d], axis=1)
    scale = max(1, min(8, math.ceil(CROP_TARGET_HEIGHT / max(1, y1 - y0))))
    strip = strip.repeat(scale, axis=0).repeat(scale, axis=1)
    name = f"{view}-{face['x']}_{face['y']}_{face['z']}-{face['face']}.png"
    _save(strip, out / "report" / "crops" / name)
    return f"report/crops/{name}"


def _fmt(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def _rate(passed: int, judged: int) -> str:
    return f"{passed / judged:.1%}" if judged else "-"


CSS = """
body { font-family: -apple-system, Helvetica, Arial, sans-serif; margin: 20px; color: #222; }
table { border-collapse: collapse; margin: 8px 0 16px; font-size: 13px; }
th, td { border: 1px solid #ccc; padding: 3px 7px; text-align: right; }
th { background: #f0f0f0; } td.l, th.l { text-align: left; }
.bad { color: #b00; font-weight: bold; } .ok { color: #070; }
.views img { width: 24%; margin-right: 0.5%; }
.face { display: inline-block; vertical-align: top; margin: 6px 10px 12px 0; font-size: 12px; max-width: 760px; }
.face img { image-rendering: pixelated; display: block; border: 1px solid #999; }
h2 { margin-top: 28px; } code { font-size: 12px; }
"""


def write(out: Path, report: dict, crops: int) -> None:
    """Crops of the worst faces (stored in the report's face records) and ``report.html``."""
    failing = sorted((f for f in report["faces"] if f["failed"]), key=lambda f: -f["severity"])
    labels_cache: dict = {}
    images_cache: dict = {}
    for face in failing[:crops]:
        face["crop"] = crop(out, face, labels_cache, images_cache)

    s = report["summary"]
    t = report["thresholds"]
    e = html.escape
    parts = [f"<!doctype html><meta charset='utf-8'><title>World oracle {e(report['variant'])}/{e(report['scene'])}</title>",
             f"<style>{CSS}</style>",
             f"<h1>World oracle: {e(report['scene'])} ({e(report['variant'])})</h1>",
             f"<p>{e(report['created'])}. Reference: the GTNH client (<code>{e(report['client']['dir'])}</code>"
             f"{', packs ' + e(', '.join(report['client']['resource_packs'])) if report['client']['resource_packs'] else ''}"
             f"{', mods ' + e(', '.join(report['client']['extra_mods'])) if report['client']['extra_mods'] else ''}). "
             f"Test: the BlueMap webapp with packs <code>{e(', '.join(Path(p).name for p in report['bluemap_packs']))}</code>, "
             f"camera pinned to the client's matrices.</p>",
             f"<p><b>{s['passed']} / {s['judged']} judged faces pass ({_rate(s['passed'], s['judged'])})</b> in "
             f"{s['views_aligned']} / {s['views']} aligned views; {s['faces']} faces shown, faces in views that failed "
             f"the alignment check are not judged. Thresholds: IoU &ge; {t['iou_min']}, mean CIEDE2000 &le; "
             f"{t['delta_e_max']}, SSIM &ge; {t['ssim_min']}, faces with fewer than {t['min_pixels']} pixels are not judged.</p>"]
    excluded = report.get("excluded") or {}
    if excluded.get("cells"):
        parts.append(f"<p>Excluded (<code>oracle.exclude</code>: the real client throws rendering them; set to air for "
                     f"the capture, not judged): {excluded['cells']} cells, states "
                     f"<code>{e(', '.join(excluded['states']))}</code>.</p>")
    calibration = report.get("calibration") or {}
    if calibration:
        parts.append(f"<p>Calibration: {e(calibration.get('rule', ''))} ({e(str(calibration.get('scene')))}, "
                     f"{e(str(calibration.get('variant')))}, <code>{e(str(calibration.get('namespace')))}</code> faces).</p>")
        dists = calibration.get("distributions", {})
        rows = "".join(
            f"<tr><td class=l>{e(k)}</td>" + "".join(f"<td>{_fmt(v['summary'].get(q))}</td>"
                                                    for q in ("count", "p01", "p05", "p25", "p50", "p75", "p95", "p99"))
            + "</tr>" for k, v in dists.items())
        parts.append("<table><tr><th class=l>calibration</th><th>n</th><th>p1</th><th>p5</th><th>p25</th><th>p50</th>"
                     f"<th>p75</th><th>p95</th><th>p99</th></tr>{rows}</table>")
    parts.append("<h2>Views</h2><table><tr><th class=l>view</th><th class=l>source</th><th>aligned</th><th>shift px</th>"
                 "<th>edge median px</th><th>edge p90 px</th><th>edge pixels</th><th>faces</th><th>judged</th>"
                 "<th>pass</th><th>client settled/stable</th><th>webapp stable</th><th>seconds client/webapp/compare</th></tr>")
    for v in report["views"]:
        if v["error"]:
            parts.append(f"<tr><td class=l>{e(v['name'])}</td><td class=l colspan=12 class=bad>{e(v['error'])}</td></tr>")
            continue
        a = v["alignment"]
        vs = v["summary"]
        c = v["checks"]
        sec = v["seconds"]
        parts.append(
            f"<tr><td class=l>{e(v['name'])}</td><td class=l>{e(v['source'])}</td>"
            f"<td class={'ok' if a['passed'] else 'bad'}>{'yes' if a['passed'] else 'NO'}</td>"
            f"<td>{_fmt(a['shift'][0], 2)}, {_fmt(a['shift'][1], 2)}</td><td>{_fmt(a['edge_error_median_px'], 2)}</td>"
            f"<td>{_fmt(a['edge_error_p90_px'], 2)}</td><td>{a['edge_pixels']}</td><td>{vs['faces']}</td>"
            f"<td>{vs['judged']}</td><td>{_rate(vs['passed'], vs['judged'])}</td>"
            f"<td>{c.get('client_settled')}/{c.get('client_stable')}</td><td>{c.get('webapp_stable')}</td>"
            f"<td>{sec.get('client')}/{sec.get('webapp')}/{sec.get('compare')}</td></tr>")
    parts.append("</table><div class=views>")
    for v in report["views"]:
        if v["error"]:
            continue
        n = e(v["name"])
        parts.append(f"<p><b>{n}</b>: client | BlueMap | difference | face labels (grey: same face, red: client "
                     f"only, blue: BlueMap only)<br>"
                     f"<a href='client/{n}.png'><img src='client/{n}.png'></a>"
                     f"<a href='bluemap/{n}.png'><img src='bluemap/{n}.png'></a>"
                     f"<a href='report/{n}-diff.png'><img src='report/{n}-diff.png'></a>"
                     f"<a href='report/{n}-labels.png'><img src='report/{n}-labels.png'></a></p>")
    parts.append("</div><h2>Per mod</h2><table><tr><th class=l>mod</th><th>judged</th><th>passed</th><th>pass rate</th></tr>")
    for mod, m in s["per_mod"].items():
        parts.append(f"<tr><td class=l>{e(mod)}</td><td>{m['judged']}</td><td>{m['passed']}</td>"
                     f"<td>{_rate(m['passed'], m['judged'])}</td></tr>")
    parts.append("</table><h2>Worst faces, by mod</h2><p>Crops: client | BlueMap | difference; yellow and cyan outline "
                 "the face region in each rendering.</p>")
    by_mod: dict[str, list[dict]] = {}
    for face in failing:
        by_mod.setdefault(face["mod"], []).append(face)
    for mod, mod_faces in sorted(by_mod.items(), key=lambda kv: -kv[1][0]["severity"]):
        parts.append(f"<h3>{e(mod)} ({len(mod_faces)} failing)</h3>")
        for face in mod_faces[:20]:
            image = f"<img src='{e(face['crop'])}'>" if face.get("crop") else ""
            parts.append(
                f"<div class=face>{image}<b>{e(face['label'])}</b> {e(face['view'])} ({face['x']}, {face['y']}, "
                f"{face['z']}) {e(face['face'])}: IoU {_fmt(face['iou'])}, &Delta;E {_fmt(face['delta_e'], 2)}, "
                f"SSIM {_fmt(face['ssim'])}, px {face['ref_pixels']}/{face['test_pixels']}<br>"
                f"<span class=bad>{e('; '.join(face['failed']))}</span><br>{e(face['suspect'])}</div>")
        if len(mod_faces) > 20:
            parts.append(f"<p>... {len(mod_faces) - 20} more in report.json</p>")
    (out / "report.html").write_text("\n".join(parts) + "\n", encoding="utf-8")
