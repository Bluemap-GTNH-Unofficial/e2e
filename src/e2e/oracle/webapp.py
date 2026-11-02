"""BlueMap webapp captures at a camera pinned to the client's matrices (world oracle).

Per camera, in a fresh headless Chromium context of the client image size:

1. The WebGL context is created without MSAA (an init script forces ``antialias: false``), so edges are as hard as
   the client's; colour and depth are read from the WebGL drawing buffer, so no DOM UI overlay is ever captured.
2. Once the map is loaded, the map controls are detached and the camera gets the client's eye position, orientation
   and projection matrix (``updateProjectionMatrix`` is replaced so BlueMap's far-plane juggling cannot change it).
3. Capture conditions as on the client: ``ambientLight`` and ``sunlightStrength`` 1 (full brightness), no ambient
   occlusion and Minecraft's flat face shade (up 1.0, down 0.5, north/south 0.8, west/east 0.6; vertical diagonal
   faces such as cross plants unshaded, which takes a tolerance of 0.02 on the normal components: BlueMap's normals of
   a 45 degree rotated element are not exactly diagonal) instead of BlueMap's directional light, no mipmaps (nearest
   texel sampling like the client with ``mipmapLevels:0``), animations pinned to their first frame, no chunk borders.
4. Hires tiles are loaded around the given centre (dropping every earlier tile) and the frame is drawn until two
   consecutive frames are identical. A second pass draws the same camera with the hires materials writing their
   view-space depth instead of their colour (24 bit fixed point, 1/16384 block resolution, alpha 0 where nothing is
   drawn; blending off so the nearest surface wins; sky and lowres tiles hidden).
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import numpy as np
from playwright.sync_api import Browser

from e2e.browser import FREEZE_ANIMATIONS_JS, TILES_SETTLED_JS, _wait

NO_MSAA_INIT_JS = """
(() => {
    const original = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attributes) {
        if (type === "webgl2" || type === "webgl") attributes = Object.assign({}, attributes, {antialias: false});
        return original.call(this, type, attributes);
    };
})();
"""

# GLSL patches of BlueMap's hires shaders (common/webapp/src/js/map/hires/Hires*Shader.js); every anchor must exist,
# otherwise the webapp changed and the capture fails instead of silently rendering with BlueMap's own lighting.
PIN_CAMERA_JS = """
([cam, center, hiresDistance, lowresDistance]) => {
    const viewer = window.bluemap.mapViewer;
    const map = viewer.map;
    const manager = viewer.controlsManager;
    const oracle = window.__oracle = window.__oracle || {depth: {value: false}};

    // the camera belongs to the oracle from now on
    manager.controls = null;
    manager.update = () => {};
    manager.updateCamera = () => {};
    viewer.superSampling = 1;

    // lighting and texture sampling as on the client
    const uniforms = viewer.data.uniforms;
    uniforms.ambientLight.value = 1;
    uniforms.sunlightStrength.value = 1;
    uniforms.chunkBorders.value = false;
    const replace = (source, anchor, replacement, what) => {
        if (!source.includes(anchor)) throw new Error(`BlueMap ${what} shader changed: anchor not found: ${anchor}`);
        return source.replace(anchor, replacement);
    };
    for (const material of map.hiresMaterial) {
        if (material.userData.oraclePatched) continue;
        let vs = material.vertexShader;
        vs = replace(vs, "varying float vBlocklight;", "varying float vBlocklight;\\nvarying float vOracleViewDepth;", "hires vertex");
        vs = replace(vs, "vAo = ao;", "vAo = 1.0;", "hires vertex");
        const start = vs.indexOf("// apply directional lighting");
        const last = vs.indexOf("vAo *= 1.0 - max(0.0, -vNormal.y)", start);
        const end = vs.indexOf("}", last);
        if (start < 0 || last < 0 || end < 0) throw new Error("BlueMap hires vertex shader changed: directional lighting block not found");
        vs = vs.slice(0, start) +
            "// oracle: Minecraft's flat-lighting face shade (no ambient occlusion)\\n" +
            "    if (abs(vNormal.y) > 0.02 || abs(abs(vNormal.x) - abs(vNormal.z)) > 0.02) {\\n" +
            "        vAo = min(vNormal.x * vNormal.x * 0.6 + vNormal.y * vNormal.y * ((3.0 + vNormal.y) / 4.0) + vNormal.z * vNormal.z * 0.8, 1.0);\\n" +
            "    }" + vs.slice(end + 1);
        vs = replace(vs, "gl_Position = projectionMatrix", "vOracleViewDepth = -(viewMatrix * modelMatrix * vec4(position, 1)).z;\\n    gl_Position = projectionMatrix", "hires vertex");
        let fs = material.fragmentShader;
        fs = replace(fs, "varying float vBlocklight;", "varying float vBlocklight;\\nvarying float vOracleViewDepth;\\nuniform bool oracleDepth;", "hires fragment");
        fs = replace(fs, "gl_FragColor = color;",
            "if (oracleDepth) {\\n" +
            "        float d = clamp(vOracleViewDepth, 0.0, 1023.999);\\n" +
            "        float i = floor(d);\\n" +
            "        float f = d - i;\\n" +
            "        gl_FragColor = vec4(floor(i / 4.0) / 255.0, (mod(i, 4.0) * 64.0 + floor(f * 64.0)) / 255.0, floor(fract(f * 64.0) * 256.0) / 255.0, 1.0);\\n" +
            "    } else {\\n" +
            "        gl_FragColor = color;\\n" +
            "    }", "hires fragment");
        material.vertexShader = vs;
        material.fragmentShader = fs;
        material.uniforms.oracleDepth = oracle.depth;
        const texture = material.uniforms.textureImage.value;
        texture.generateMipmaps = false;
        texture.minFilter = 1003;  // THREE.NearestFilter
        texture.needsUpdate = true;
        material.userData.oraclePatched = true;
        material.needsUpdate = true;
    }

    // the client's camera
    const camera = viewer.camera;
    camera.ortho = 0;
    camera.distance = 1;  // < 1000: hires drawn, lowres discarded under hires tiles
    camera.near = cam.near;
    camera.far = cam.far;
    camera.fov = cam.fov;
    camera.aspect = cam.aspect;
    camera.position.set(cam.position[0], cam.position[1], cam.position[2]);
    camera.quaternion.set(cam.quaternion[0], cam.quaternion[1], cam.quaternion[2], cam.quaternion[3]);
    camera.updateMatrixWorld(true);
    camera.updateProjectionMatrix = function () {
        this.projectionMatrix.fromArray(cam.projection);
        this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
    };
    camera.updateProjectionMatrix();

    // a fresh tile set around the compared area; one tile more than asked for, because the webapp's spiral loader
    // (TileManager.loadNextTile) stops one ring short on one side of each axis and would leave out tiles within
    // hiresDistance (seen with the camera outside the area: the tile under the camera was never loaded)
    manager.position.set(center[0], center[1], center[2]);
    for (const tiles of [map.hiresTileManager, ...map.lowresTileManager]) tiles.removeAllTiles();
    const hiresTile = Math.max(map.data.hires.tileSize.x, map.data.hires.tileSize.z);
    viewer.loadMapArea(center[0], center[2], hiresDistance + hiresTile, lowresDistance);
    viewer.redraw();
}
"""

CAPTURE_JS = """
async ([width, height, maxFrames]) => {
    const viewer = window.bluemap.mapViewer;
    const renderer = viewer.renderer;
    const gl = renderer.getContext();
    const canvas = renderer.domElement;
    if (canvas.width !== width || canvas.height !== height)
        throw new Error(`canvas is ${canvas.width}x${canvas.height}, expected ${width}x${height}`);
    const read = () => {
        const pixels = new Uint8Array(width * height * 4);
        gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
        return pixels;
    };
    const same = (a, b) => {
        if (!a || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
        return true;
    };
    const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
    const base64 = bytes => {
        let s = "";
        for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
        return btoa(s);
    };

    let previous = null, colour = null, stable = false, frames = 0;
    while (frames < maxFrames) {
        await nextFrame();
        await nextFrame();
        frames++;
        viewer.render(0);
        colour = read();
        if (same(previous, colour)) { stable = true; break; }
        previous = colour;
    }

    // depth pass
    const materials = viewer.map.hiresMaterial;
    const blending = materials.map(m => m.blending);
    const lowres = viewer.map.lowresTileManager.map(m => m.sceneParent.visible);
    const sky = viewer.skyboxScene.visible;
    const clearColor = renderer.getClearColor(viewer.data.uniforms.skyColor.value.clone());
    const clearAlpha = renderer.getClearAlpha();
    materials.forEach(m => { m.blending = 0; });  // THREE.NoBlending: the nearest surface wins
    viewer.map.lowresTileManager.forEach(m => { m.sceneParent.visible = false; });
    viewer.skyboxScene.visible = false;
    renderer.setClearColor(0x000000, 0);
    window.__oracle.depth.value = true;
    let depth;
    try {
        viewer.render(0);
        depth = read();
    } finally {
        window.__oracle.depth.value = false;
        materials.forEach((m, i) => { m.blending = blending[i]; });
        viewer.map.lowresTileManager.forEach((m, i) => { m.sceneParent.visible = lowres[i]; });
        viewer.skyboxScene.visible = sky;
        renderer.setClearColor(clearColor, clearAlpha);
        viewer.render(0);
    }
    const camera = viewer.camera;
    return {
        colour: base64(colour), depth: base64(depth), stable, frames,
        projection: Array.from(camera.projectionMatrix.elements),
        view: Array.from(camera.matrixWorldInverse.elements),
    };
}
"""


@dataclass
class WebappCapture:
    rgba: np.ndarray  # (H, W, 4) uint8, rows top to bottom
    distance: np.ndarray  # (H, W) float64 view-space depth in blocks, NaN where nothing is drawn
    stable: bool
    frames: int
    settled: bool
    load_seconds: float
    capture_seconds: float
    projection: np.ndarray  # (4, 4) as applied by three.js
    view: np.ndarray  # (4, 4) world -> eye as applied by three.js
    console_errors: list[str]


def decode_depth(rgba: np.ndarray) -> np.ndarray:
    """View-space depth of the depth pass: integer part 10 bits (R, top 2 bits of G), fraction 14 bits."""
    r, g, b, a = (rgba[..., i].astype(np.float64) for i in range(4))
    depth = (r * 4 + np.floor(g / 64)) + (np.mod(g, 64) / 64.0) + b / 16384.0
    return np.where(a > 0, depth, np.nan)


def _pixels(data: str, width: int, height: int) -> np.ndarray:
    # gl.readPixels rows are bottom to top
    return np.frombuffer(base64.b64decode(data), dtype=np.uint8).reshape(height, width, 4)[::-1].copy()


def capture(browser: Browser, base_url: str, map_id: str, camera: dict, center: tuple[float, float, float],
            hires_distance: float, width: int, height: int, load_timeout: float = 180, max_frames: int = 60) -> WebappCapture:
    """One pinned capture; ``camera`` as built by ``oracle.camera.webapp_camera``."""
    context = browser.new_context(viewport={"width": width, "height": height}, device_scale_factor=1)
    context.add_init_script(NO_MSAA_INIT_JS)
    page = context.new_page()
    errors: list[str] = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "status of 404" not in m.text else None)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    try:
        started = time.monotonic()
        deadline = started + load_timeout
        x, y, z = center
        page.goto(f"{base_url}#{map_id}:{x:g}:{y:g}:{z:g}:50:0:0.8:0:0:perspective", wait_until="load")
        settled = _wait(page, TILES_SETTLED_JS, map_id, deadline)
        page.evaluate(PIN_CAMERA_JS, [camera, list(center), hires_distance, 500])
        settled = _wait(page, TILES_SETTLED_JS, map_id, deadline) and settled
        loaded = time.monotonic()
        page.evaluate(FREEZE_ANIMATIONS_JS)
        result = page.evaluate(CAPTURE_JS, [width, height, max_frames])
        done = time.monotonic()
    finally:
        context.close()
    depth_rgba = _pixels(result["depth"], width, height)
    return WebappCapture(
        rgba=_pixels(result["colour"], width, height),
        distance=decode_depth(depth_rgba),
        stable=result["stable"],
        frames=result["frames"],
        settled=settled,
        load_seconds=loaded - started,
        capture_seconds=done - loaded,
        projection=np.asarray(result["projection"], dtype=np.float64).reshape(4, 4).T,
        view=np.asarray(result["view"], dtype=np.float64).reshape(4, 4).T,
        console_errors=errors,
    )
