"""Headless Chromium captures of the BlueMap webapp."""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass, field
from urllib.parse import urlparse

from playwright.sync_api import Browser, ConsoleMessage, Page, Playwright, Request, Response
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from e2e.scenes import View

# Software WebGL (SwiftShader): identical pixels on every machine, independent of the host GPU.
CHROMIUM_ARGS = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]

# Every tile manager (hires + each lowres LOD) has been pointed at the camera, has nothing in flight and holds a
# settled tile (loaded or known-missing) for every cell that `TileManager.loadNextTile` would still request (the
# same spiral walk; it does not cover the full view square).
TILES_SETTLED_JS = """
(mapId) => {
    const map = window.bluemap?.mapViewer?.map;
    if (!map || !map.isLoaded || map.data.id !== mapId) return false;
    for (const manager of [map.hiresTileManager, ...map.lowresTileManager]) {
        if (manager.unloaded || manager.currentlyLoading !== 0) return false;
        const cx = manager.centerTile.x, cz = manager.centerTile.y;
        const vx = manager.viewDistanceX, vz = manager.viewDistanceZ;
        const settled = (x, z) => {
            if (Math.abs(x) > vx || Math.abs(z) > vz) return true;
            const tile = manager.tiles.get(`x${cx + x}z${cz + z}`);
            return tile !== undefined && !tile.loading;
        };
        let x = 0, z = 0, d = 1, m = 1;
        while (m < Math.max(vx, vz) * 2 + 1) {
            while (2 * x * d < m) { if (!settled(x, z)) return false; x += d; }
            while (2 * z * d < m) { if (!settled(x, z)) return false; z += d; }
            d = -d;
            m++;
        }
    }
    return true;
}
"""

# Pins every texture animation (water, lava, fire, portals, ...) to its first frame so frames can settle.
FREEZE_ANIMATIONS_JS = """
() => {
    for (const animation of window.bluemap.mapViewer.map?.animations ?? []) {
        animation.step = function () {
            const u = this.uniforms;
            u.animationFrameIndex.value = this.data.frames ? this.data.frames[0].index : 0;
            u.animationInterpolationFrameIndex.value = u.animationFrameIndex.value;
            u.animationInterpolation.value = 0;
        };
        animation.step(0);
    }
    window.bluemap.mapViewer.redraw();
}
"""

# The tile set the webapp ends up with depends on how the camera got there: it re-centres only after moving half a
# view distance, and `removeFarTiles` keeps tiles anywhere in the view square while the spiral loader only requests
# part of it, so tiles loaded around earlier centres survive at the far edge. Dropping every tile and loading
# around the camera's final position makes the set a function of that position alone.
RESET_TILES_JS = """
() => {
    const viewer = window.bluemap.mapViewer;
    const map = viewer.map;
    for (const manager of [map.hiresTileManager, ...map.lowresTileManager]) manager.removeAllTiles();
    const position = viewer.controlsManager.position;
    viewer.loadMapArea(position.x, position.z);
}
"""

# MapHeightControls eases the camera's target and minimum heights towards the terrain height under the camera
# (raycast against whatever tiles are loaded) and snaps once within 0.001. Converged = snapped to the current tiles.
HEIGHTS_CONVERGED_JS = """
() => {
    const app = window.bluemap;
    const manager = app.mapViewer.controlsManager;
    const map = app.mapViewer.map;
    if (manager.controls !== app.mapControls) return true;  // free flight: no height following
    const heights = app.mapControls.mapHeight;
    const target = map.terrainHeightAt(manager.position.x, manager.position.z) + 3 || 0;
    if (heights.targetHeight !== target) return false;
    if (app.mapControls.constructor.getMaxPerspectiveAngleForDistance(manager.distance) < 0.1) return true;
    const camera = manager.camera.position;
    return heights.cameraHeight === (map.terrainHeightAt(camera.x, camera.z) || 0);
}
"""

# Re-applies the view's camera. Height following only ever raises `position.y`, so heights eased towards terrain of
# transient tile sets (initial camera, lowres before hires) would otherwise stay in the final frame. The webapp
# rewrites the URL hash with rounded values on every camera change, so the exact view hash is restored first
# (`replaceState`: no hashchange event).
REAPPLY_CAMERA_JS = """
async (hash) => {
    history.replaceState(undefined, undefined, "#" + hash);
    await window.bluemap.loadPageAddress();
}
"""

# Waits until `stableFrames` consecutive samples (one per `frameGap` animation frames) are pixel-identical while
# the tiles stay settled. Bounded by `timeoutMs`; returns the last frame either way.
STABLE_FRAME_JS = """
async ([mapId, stableFrames, frameGap, timeoutMs]) => {
    const settled = %s;
    const canvas = window.bluemap.mapViewer.renderer.domElement;
    const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
    const started = performance.now();
    let previous = null, same = 0, samples = 0;
    while (performance.now() - started < timeoutMs) {
        for (let i = 0; i < frameGap; i++) await nextFrame();
        samples++;
        if (!settled(mapId)) { previous = null; same = 0; continue; }
        const frame = canvas.toDataURL("image/png");
        same = frame === previous ? same + 1 : 0;
        previous = frame;
        if (same + 1 >= stableFrames) return {stable: true, frame, samples, ms: performance.now() - started};
    }
    return {stable: false, frame: canvas.toDataURL("image/png"), samples, ms: performance.now() - started};
}
""" % TILES_SETTLED_JS.strip()

# The same camera with only the skybox drawn, for the "is anything rendered at all" metric.
SKY_FRAME_JS = """
() => {
    const viewer = window.bluemap.mapViewer;
    viewer.renderer.clear();
    viewer.skyboxCamera.rotation.copy(viewer.camera.rotation);
    viewer.renderer.render(viewer.skyboxScene, viewer.skyboxCamera);
    const frame = viewer.renderer.domElement.toDataURL("image/png");
    viewer.redraw();
    return frame;
}
"""


WEBGL_RENDERER_JS = """
() => {
    const gl = window.bluemap.mapViewer.renderer.getContext();
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    return info ? gl.getParameter(info.UNMASKED_RENDERER_WEBGL) : gl.getParameter(gl.RENDERER);
}
"""


@dataclass
class Capture:
    png: bytes
    sky_png: bytes
    settled: bool  # all tiles loaded within the load bound
    stable: bool  # identical consecutive frames within the stable-frame bound
    load_seconds: float  # navigation until all tiles settled
    stable_seconds: float  # settled until the stable frame
    stable_samples: int  # frames sampled until stable (or the bound)
    webgl_renderer: str
    console_errors: list[str] = field(default_factory=list)
    failed_requests: list[str] = field(default_factory=list)
    missing_tiles: int = 0  # 404s for tiles outside the rendered area (expected)


def _decode(data_url: str) -> bytes:
    return base64.b64decode(data_url.split(",", 1)[1])


def _is_tile(url: str) -> bool:
    return "/tiles/" in urlparse(url).path


class Chromium:
    def __init__(self, playwright: Playwright) -> None:
        self.browser: Browser = playwright.chromium.launch(headless=True, args=CHROMIUM_ARGS)

    def close(self) -> None:
        self.browser.close()

    def capture(self, base_url: str, view: View, viewport: tuple[int, int], load_timeout: float = 120,
                stable_frames: int = 3, frame_gap: int = 2, stable_timeout: float = 60) -> Capture:
        width, height = viewport
        context = self.browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
        page = context.new_page()
        console_errors: list[str] = []
        failed: list[str] = []
        missing_tiles = 0

        def on_console(message: ConsoleMessage) -> None:
            if message.type == "error":
                console_errors.append(message.text)

        def on_request_failed(request: Request) -> None:
            failed.append(f"{request.method} {request.url}: {request.failure}")

        def on_response(response: Response) -> None:
            nonlocal missing_tiles
            if response.status < 400:
                return
            if response.status == 404 and _is_tile(response.url):
                missing_tiles += 1
            else:
                failed.append(f"{response.request.method} {response.url}: HTTP {response.status}")

        page.on("console", on_console)
        page.on("pageerror", lambda error: console_errors.append(f"pageerror: {error}"))
        page.on("requestfailed", on_request_failed)
        page.on("response", on_response)
        try:
            started = time.monotonic()
            deadline = started + load_timeout
            page.goto(f"{base_url}#{view.hash}", wait_until="load")
            # 1. the webapp's own loading, 2. a fresh tile set around the final camera position, 3. heights snapped to
            # that tile set, 4. the hash camera re-applied on top (see the JS constants)
            settled = _wait(page, TILES_SETTLED_JS, view.map, deadline)
            if settled:
                page.evaluate(RESET_TILES_JS)
                settled = _wait(page, TILES_SETTLED_JS, view.map, deadline)
            if settled:
                settled = _wait(page, HEIGHTS_CONVERGED_JS, None, deadline)
                page.evaluate(REAPPLY_CAMERA_JS, view.hash)
            loaded = time.monotonic()
            page.evaluate(FREEZE_ANIMATIONS_JS)
            result = page.evaluate(STABLE_FRAME_JS, [view.map, stable_frames, frame_gap, stable_timeout * 1000])
            stable_seconds = time.monotonic() - loaded
            sky = page.evaluate(SKY_FRAME_JS)
            webgl_renderer = page.evaluate(WEBGL_RENDERER_JS)
        finally:
            context.close()
        # Chromium echoes every HTTP 404 as a console error; the responses themselves are classified above
        # (tile 404s = unrendered area, others = failed requests)
        console_errors = [e for e in console_errors if not _is_resource_404(e)]
        return Capture(
            png=_decode(result["frame"]),
            sky_png=_decode(sky),
            settled=settled,
            stable=result["stable"],
            load_seconds=loaded - started,
            stable_seconds=stable_seconds,
            stable_samples=result["samples"],
            webgl_renderer=webgl_renderer,
            console_errors=console_errors,
            failed_requests=failed,
            missing_tiles=missing_tiles,
        )


def _wait(page: Page, js: str, arg: object, deadline: float) -> bool:
    """Polls `js` every animation frame until it is truthy or the deadline passes."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    try:
        page.wait_for_function(js, arg=arg, timeout=remaining * 1000, polling="raf")
    except PlaywrightTimeoutError:
        return False
    return True


def _is_resource_404(message: str) -> bool:
    return message.startswith("Failed to load resource: the server responded with a status of 404")
