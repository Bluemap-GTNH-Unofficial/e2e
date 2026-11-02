"""Static ``report.html`` of ``e2e oracle states``: pass rates per mod, calibration, worst states with images."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import numpy as np
from PIL import Image

PER_MOD_IMAGES = 12  # failing states per mod shown with images
PER_MOD_ROWS = 200  # failing states per mod listed

CSS = """
body { font: 13px/1.4 system-ui, sans-serif; margin: 1.5em; color: #222; }
table { border-collapse: collapse; margin: .5em 0 1.5em; }
th, td { border: 1px solid #ccc; padding: 2px 6px; text-align: right; vertical-align: top; }
th { background: #eee; } td.l, th.l { text-align: left; }
.imgs img { width: 256px; height: 128px; image-rendering: pixelated; margin-right: 4px;
  background: repeating-conic-gradient(#bbb 0 25%, #eee 0 50%) 0 0 / 16px 16px; }
.state { border-top: 1px solid #ddd; padding: .4em 0; }
.bad { color: #b00; } .ok { color: #070; } code { font-size: 12px; }
.bar { display: inline-block; background: #58a; height: 10px; vertical-align: middle; }
details { margin: .3em 0; } summary { cursor: pointer; }
"""


def _url(*parts: str) -> str:
    return "/".join(quote(p, safe="") for part in parts for p in part.split("/"))


def _fmt(v, digits: int = 3) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return html.escape(str(v))


def _rate(v) -> str:
    return "-" if v is None else f"{v:.1%}"


def _write_diffs(out: Path, results: list[dict], diff_image: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> None:
    for r in results:
        client = out / "client" / f"{r['path']}.png"
        bluemap = out / "bluemap" / f"{r['path']}.png"
        target = out / "diff" / f"{r['path']}.png"
        if not client.is_file() or not bluemap.is_file():
            continue
        with Image.open(client) as a, Image.open(bluemap) as b:
            diff = diff_image(np.asarray(a.convert("RGBA")), np.asarray(b.convert("RGBA")))
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(diff, "RGB").save(target)


def _state_block(r: dict, images: bool) -> str:
    status = r["status"]
    cls = "ok" if status in ("pass", "accepted") else "bad"
    bm = r.get("bluemap", {})
    variants = ", ".join(f"{v.get('renderer', '?')} {v.get('model', '')}" for v in bm.get("variants", []) or [])
    client = r.get("client", {})
    facts = [f"client: render type {_fmt(client.get('renderType'))}"]
    if client.get("tileEntity"):
        facts.append(f"TE {html.escape(client['tileEntity'].rsplit('.', 1)[-1])}{' (TESR)' if client.get('tesr') else ''}")
    if client.get("mte") is not None:
        facts.append(f"MTE {client['mte']} (block meta {client.get('worldMeta')})")
    if client.get("error"):
        facts.append(f"client error: {html.escape(client['error'])}")
    facts.append(f"BlueMap: <code>{html.escape(str(bm.get('blockState', '?')))}</code> {html.escape(variants)}")
    if bm.get("error"):
        facts.append(f"BlueMap error: {html.escape(bm['error'])}")
    if r.get("note"):
        facts.append(html.escape(r["note"]))
    if r.get("accepted"):
        facts.append(html.escape(r["accepted"]))
    if r.get("acceptedUnbacked"):
        facts.append(f"accepted.yaml names it ({html.escape(r['acceptedUnbacked'])}) but the 1.7.10 vs 1.13 data does "
                     f"not show that difference: judged")
    overall = []
    for name, v in (r.get("views") or {}).items():
        m = v["overall"]
        overall.append(f"{name}: IoU {_fmt(m['iou'])}, &Delta;E {_fmt(m['delta_e'], 2)}, SSIM {_fmt(m['ssim'])}")
    shown = "accepted (1.13)" if status == "accepted" else status
    parts = [f'<div class="state"><b class="{cls}">{html.escape(shown)}</b> <code>{html.escape(r["key"])}</code> '
             f'score {_fmt(r.get("score"))}<br>{"; ".join(facts)}<br>{" | ".join(overall)}']
    if r.get("cause"):
        parts.append(f"<br>suspected: <i>{html.escape(r['cause'])}</i>")
    if r.get("checks"):
        parts.append(f"<br><small>{html.escape('; '.join(r['checks'][:8]))}{' ...' if len(r['checks']) > 8 else ''}</small>")
    if images:
        parts.append(
            '<div class="imgs">'
            f'<img loading="lazy" title="client" src="{_url("client", r["path"])}.png">'
            f'<img loading="lazy" title="BlueMap" src="{_url("bluemap", r["path"])}.png">'
            f'<img loading="lazy" title="diff" src="{_url("diff", r["path"])}.png"></div>'
        )
    parts.append("</div>")
    return "".join(parts)


def _histogram(title: str, hist: dict) -> str:
    edges, counts = hist["edges"], hist["counts"]
    peak = max(counts) or 1
    rows = "".join(
        f'<tr><td>{edges[i]:g} .. {edges[i + 1]:g}</td><td>{c}</td>'
        f'<td class="l"><span class="bar" style="width:{200 * c / peak:.0f}px"></span></td></tr>'
        for i, c in enumerate(counts)
    )
    return f"<table><tr><th colspan=3>{html.escape(title)}</th></tr>{rows}</table>"


def _calibration(report: dict) -> str:
    file = Path(report["calibration"])
    if not file.is_file():
        return "<p>No calibration file.</p>"
    cal = json.loads(file.read_text(encoding="utf-8"))
    parts = [f"<p>From <code>{html.escape(str(file))}</code>, {html.escape(cal['created'])}: vanilla states of variant "
             f"{html.escape(cal['variant'])} that look the same in 1.7.10 and 1.13 (<code>{html.escape(str(cal.get('classification')))}</code>): "
             f"delta E and SSIM over {cal['states']} states, one sample per block (identical textures, no random "
             f"offset or variant, not rotated, no custom UV or shade:false element, consistent state mapping, same "
             f"geometry, drawn by 1.7.10's box or crossed-squares renderers), IoU over the regions of "
             f"{cal.get('geometryStates', cal['states'])} states (the same without the renderer condition). States "
             f"accepted.yaml names and {html.escape(', '.join(cal['excluded']))} (not in 1.7.10) are left out.</p>"]
    ff = cal.get("falseFailures") or {}
    if ff:
        parts.append("<p>False failures on the calibration sets (states that look the same in both versions): "
                     + "; ".join(f"{html.escape(k)} {v['fail']} / {v['states']} ({_rate(v['rate'])})" for k, v in ff.items())
                     + "</p>")
        failing = (ff.get("geometry") or {}).get("failing", [])
        if failing:
            parts.append("<details><summary>failing calibration states</summary>" + "".join(
                f"<div><code>{html.escape(f['key'])}</code> {html.escape('; '.join(f['checks']))}</div>" for f in failing)
                + "</details>")
    for level in ("view", "face"):
        c = cal[level]
        t = c["thresholds"]
        parts.append(f"<h3>{level} regions: IoU &ge; {t['iou_min']}, &Delta;E &le; {t['delta_e_max']}, "
                     f"SSIM &ge; {t['ssim_min']} (judged if &ge; {t['min_pixels']} px)</h3><p>{html.escape(c['rule'])}</p>")
        rows = []
        for metric in ("iou", "delta_e", "ssim"):
            s = c["distributions"][metric]["summary"]
            f = c["fences"][metric]
            rows.append(f"<tr><td class=l>{metric}</td>" + "".join(
                f"<td>{_fmt(s.get(k))}</td>" for k in ("count", "mean", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max"))
                + f"<td>{f.get('inexact', '-')}</td><td>{_fmt(f.get('q1'))}</td><td>{_fmt(f.get('q3'))}</td></tr>")
        parts.append("<table><tr><th class=l>metric</th><th>n</th><th>mean</th><th>p1</th><th>p5</th><th>p25</th>"
                     "<th>p50</th><th>p75</th><th>p95</th><th>p99</th><th>max</th><th>inexact</th><th>Q1 inexact</th>"
                     "<th>Q3 inexact</th></tr>" + "".join(rows) + "</table>")
        parts.append("<div style='display:flex;gap:2em'>" + "".join(
            _histogram(f"{level} {metric}", c["distributions"][metric]["histogram"]) for metric in ("iou", "delta_e", "ssim"))
            + "</div>")
    return "".join(parts)


def worst_per_block(failing: list[dict]) -> list[dict]:
    """The worst failing state of every block (``failing`` sorted worst first), so one block's metas do not fill a list."""
    seen, out = set(), []
    for r in failing:
        if r["name"] not in seen:
            seen.add(r["name"])
            out.append(r)
    return out


def write(out: Path, report: dict, results: list[dict], diff_image: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Path:
    failing = sorted((r for r in results if r["status"] == "fail"), key=lambda r: -r.get("score", 0))
    by_mod = defaultdict(list)
    for r in failing:
        by_mod[r["mod"]].append(r)
    worst_blocks = worst_per_block(failing)
    with_images = {r["path"] for r in worst_blocks[:100]}
    for rs in by_mod.values():
        with_images.update(r["path"] for r in rs[:PER_MOD_IMAGES])
    _write_diffs(out, [r for r in results if r["path"] in with_images], diff_image)

    t = report["total"]
    sec = report["seconds"]
    body = [
        f"<h1>Per-state rendering oracle: {html.escape(report['variant'])}</h1>",
        f"<p>{html.escape(report['created'])}. BlueMap packs: {html.escape(', '.join(Path(p).name for p in report['bluemapPacks']))}. "
        f"Runtime: client render {_fmt(sec.get('client'), 0)} s, BlueMap render {_fmt(sec.get('bluemapRender'), 0)} s"
        f" (with Gradle {_fmt(sec.get('bluemap'), 0)} s), comparison {_fmt(sec.get('compare'), 0)} s.</p>",
        f"<p>Conditions: {html.escape(str(report.get('conditions')))}</p>",
        f"<h2>Result: {t['pass']} / {t['judged']} judged states pass ({_rate(t['passRate'])}), "
        f"{t.get('accepted', 0)} accepted (1.13)</h2>",
        f"<p>{t['states']} states; accepted (1.13): {t.get('accepted', 0)} vanilla states "
        f"<code>{html.escape(str((report.get('accepted') or {}).get('file')))}</code> names and the 1.7.10 vs 1.13 data "
        f"backs (not passes, not failures); not judged: {t['clientBlank']} drawn only by BlueMap while the client draws "
        f"nothing in isolation, {t['clientError']} with a client render error, {t['bluemapMissing']} without BlueMap output."
        + (" Thresholds are provisional (subset run before a calibration)." if report.get("provisionalThresholds") else "")
        + "</p>",
        "<h2>Per mod</h2><table><tr><th class=l>mod</th><th>states</th><th>judged</th><th>pass</th><th>fail</th>"
        "<th>pass rate</th><th>accepted (1.13)</th><th>client blank</th><th>client error</th><th>no BlueMap</th></tr>",
    ]
    for mod, c in report["mods"].items():
        body.append(f"<tr><td class=l><a href='#mod-{html.escape(mod)}'>{html.escape(mod)}</a></td><td>{c['states']}</td>"
                    f"<td>{c['judged']}</td><td>{c['pass']}</td><td>{c['fail']}</td><td>{_rate(c['passRate'])}</td>"
                    f"<td>{c.get('accepted', 0)}</td><td>{c['clientBlank']}</td><td>{c['clientError']}</td>"
                    f"<td>{c['bluemapMissing']}</td></tr>")
    body.append("</table><h2>Suspected causes of failures (heuristic)</h2><table><tr><th class=l>cause</th><th>states</th></tr>")
    body += [f"<tr><td class=l>{html.escape(k or '(none)')}</td><td>{v}</td></tr>" for k, v in report["causes"].items()]
    body.append("</table><h2>Thresholds and calibration</h2>")
    body.append(_calibration(report))
    body.append("<h2>Worst 100 blocks</h2><p>The worst failing state of each block. Images: client | BlueMap | "
                "difference (CIEDE2000 black 0, red 25, yellow 50+; magenta: only the client draws; green: only BlueMap "
                "draws).</p>")
    body += [_state_block(r, True) for r in worst_blocks[:100]]
    body.append("<h2>Failing states per mod</h2>")
    for mod, rs in sorted(by_mod.items(), key=lambda e: -len(e[1])):
        c = report["mods"][mod]
        body.append(f"<details id='mod-{html.escape(mod)}'><summary><b>{html.escape(mod)}</b>: {c['fail']} failing of "
                    f"{c['judged']} judged ({_rate(c['passRate'])} pass)</summary>")
        body += [_state_block(r, i < PER_MOD_IMAGES) for i, r in enumerate(rs[:PER_MOD_ROWS])]
        if len(rs) > PER_MOD_ROWS:
            body.append(f"<p>... {len(rs) - PER_MOD_ROWS} more in report.json</p>")
        body.append("</details>")
    for status, title in (("accepted", "Accepted (1.13): differences of the 1.13 assets BlueMap keeps on purpose"),
                          ("client_blank", "Not judged: the client draws nothing in isolation"),
                          ("client_error", "Not judged: client render errors"),
                          ("bluemap_missing", "No BlueMap output")):
        rs = [r for r in results if r["status"] == status]
        if rs:
            body.append(f"<details><summary><b>{title}</b> ({len(rs)})</summary>")
            body += [_state_block(r, False) for r in rs[:2000]]
            body.append("</details>")
    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>states oracle {html.escape(report['variant'])}</title>"
            f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>")
    target = out / "report.html"
    target.write_text(page, encoding="utf-8")
    return target
