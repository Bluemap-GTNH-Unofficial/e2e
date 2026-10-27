"""One scene end to end: render the world with the CLI, serve it, capture every view, compare, report."""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from e2e import bluemap, metrics, paths
from e2e.browser import Capture, Chromium
from e2e.scenes import Scene, View


@dataclass
class ViewResult:
    view: str
    url: str
    shot: Path
    sky_shot: Path
    baseline: Path | None
    diff_image: Path | None
    diff_ratio: float | None
    metrics: metrics.ImageMetrics
    settled: bool
    stable: bool
    load_seconds: float
    stable_seconds: float
    stable_samples: int
    webgl_renderer: str
    console_errors: list[str]
    failed_requests: list[str]
    missing_tiles: int
    failures: list[str] = field(default_factory=list)


@dataclass
class SceneResult:
    scene: str
    out: Path
    port: int
    maps: list[str]
    render_seconds: float  # CLI start until the webserver answers (resources, render, webapp)
    log: Path
    log_errors: list[str]
    log_warnings: int
    total_seconds: float = 0.0
    views: list[ViewResult] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)  # scene level: errors in the BlueMap log

    @property
    def ok(self) -> bool:
        return not self.failures and all(not v.failures for v in self.views)


def prepare_jars(build: bool) -> bluemap.Jars:
    if build:
        bluemap.build(paths.out_dir() / "build.log")
    return bluemap.locate_jars()


def _select_dimensions(scene: Scene) -> list[bluemap.Dimension]:
    if not scene.world.is_dir():
        raise FileNotFoundError(f"scene world {scene.world} does not exist; build it with `orch scene {scene.name}`")
    dims = bluemap.detect_dimensions(scene.world)
    available = {d.map_id for d in dims}
    if scene.maps is not None:
        unknown = set(scene.maps) - available
        if unknown:
            raise ValueError(f"{scene.name}: maps {sorted(unknown)} have no region files; available {sorted(available)}")
        dims = [d for d in dims if d.map_id in scene.maps]
    rendered = {d.map_id for d in dims}
    for view in scene.views:
        if view.map not in rendered:
            raise ValueError(f"{scene.name}/{view.name}: map {view.map!r} is not rendered; rendered {sorted(rendered)}")
    return dims


def _evaluate(scene: Scene, view: View, capture: Capture, out: Path, url: str) -> ViewResult:
    shot = out / "shots" / f"{view.name}.png"
    sky_shot = out / "shots" / "sky" / f"{view.name}.png"
    for path, png in ((shot, capture.png), (sky_shot, capture.sky_png)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    image = metrics.load_rgb(shot)
    measured = metrics.measure(image, metrics.load_rgb(sky_shot))
    result = ViewResult(
        view=view.name, url=url, shot=shot, sky_shot=sky_shot, baseline=None, diff_image=None, diff_ratio=None,
        metrics=measured, settled=capture.settled, stable=capture.stable, load_seconds=capture.load_seconds,
        stable_seconds=capture.stable_seconds, stable_samples=capture.stable_samples,
        webgl_renderer=capture.webgl_renderer,
        console_errors=capture.console_errors,
        failed_requests=capture.failed_requests, missing_tiles=capture.missing_tiles,
    )

    baseline = paths.baseline(scene.name, view.name)
    if baseline.is_file():
        result.baseline = baseline
        try:
            mask = metrics.diff_mask(image, metrics.load_rgb(baseline), view.tolerance.pixel)
        except ValueError as e:
            result.failures.append(f"baseline {baseline}: {e}")
        else:
            result.diff_ratio = float(mask.mean())
            if result.diff_ratio > view.tolerance.ratio:
                result.diff_image = out / "diff" / f"{view.name}.png"
                metrics.save_rgb(metrics.diff_image(image, mask), result.diff_image)
                result.failures.append(
                    f"{result.diff_ratio:.4%} of pixels differ from the baseline by more than "
                    f"{view.tolerance.pixel} (allowed {view.tolerance.ratio:.4%}); diff {result.diff_image}")
    else:
        result.failures.append(f"no baseline {baseline}; approve with `uv run e2e approve {scene.name} {view.name}`")

    if measured.content_ratio < view.min_content:
        result.failures.append(f"only {measured.content_ratio:.2%} of pixels differ from the bare sky "
                               f"(minimum {view.min_content:.2%})")
    if view.max_missing is not None and measured.missing_of_content > view.max_missing:
        result.failures.append(f"missing texture on {measured.missing_of_content:.2%} of the rendered pixels "
                               f"(allowed {view.max_missing:.2%})")
    if not capture.settled:
        result.failures.append(f"tiles still loading after {capture.load_seconds:.0f}s")
    if not capture.stable:
        result.failures.append("frames did not settle within the stable-frame bound")
    if capture.console_errors:
        result.failures.append(f"{len(capture.console_errors)} console errors: {capture.console_errors[:3]}")
    if capture.failed_requests:
        result.failures.append(f"{len(capture.failed_requests)} failed requests: {capture.failed_requests[:3]}")
    return result


def run_scene(scene: Scene, jars: bluemap.Jars, chromium: Chromium, extra_packs: tuple[Path, ...] = ()) -> SceneResult:
    started = time.monotonic()
    out = paths.scene_out(scene.name)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    dims = _select_dimensions(scene)
    port = bluemap.free_port()
    bluemap.write_config(out, scene.world, dims, jars, port, [*scene.extra_packs, *extra_packs])
    with bluemap.BlueMapCli(jars, out, port) as cli:
        result = SceneResult(scene=scene.name, out=out, port=port, maps=[d.map_id for d in dims],
                             render_seconds=cli.render_seconds, log=cli.log, log_errors=[], log_warnings=0)
        for view in scene.views:
            url = f"{cli.url}#{view.hash}"
            capture = chromium.capture(cli.url, view, scene.viewport)
            result.views.append(_evaluate(scene, view, capture, out, url))
        result.log_errors = cli.log_lines("ERROR")
        result.log_warnings = len(cli.log_lines("WARNING"))
    if result.log_errors:
        result.failures.append(f"{len(result.log_errors)} errors in {result.log}: {result.log_errors[:3]}")

    result.total_seconds = time.monotonic() - started
    (out / "report.json").write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
    return result


def approve(scene: Scene, views: list[str]) -> list[Path]:
    """Copies the latest screenshots of ``views`` (all views when empty) to the baselines."""
    known = [v.name for v in scene.views]
    selected = views or known
    unknown = set(selected) - set(known)
    if unknown:
        raise ValueError(f"{scene.name}: unknown views {sorted(unknown)}; known {known}")
    approved = []
    for name in selected:
        shot = paths.scene_out(scene.name) / "shots" / f"{name}.png"
        if not shot.is_file():
            raise FileNotFoundError(f"no screenshot {shot}; run `uv run e2e run {scene.name}` first")
        target = paths.baseline(scene.name, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(shot, target)
        approved.append(target)
    return approved
