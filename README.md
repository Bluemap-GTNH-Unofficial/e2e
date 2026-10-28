# e2e

Browser end-to-end checks of rendered BlueMap GTNH maps (plan.md §4 "浏览器 e2e", §12). For each scene it renders
a scene world with the BlueMap CLI, serves it, opens camera views in headless Chromium, and compares screenshots
with approved baselines.

## Setup

```bash
uv sync --group dev
uv run playwright install chromium   # once per machine (~100 MB, ~/Library/Caches/ms-playwright)
```

Scene worlds come from the orchestrator (`uv run orch scene vanilla-zoo` etc. in `testing/orchestrator`) and live in
`<workspace>/worlds/<scene>/`. The workspace defaults to the parent of `BlueMap-GTNH`; override with
`BMGTNH_WORKSPACE`. Java: `mise where java@temurin-25.0.5+8.0.LTS`, override with `BMGTNH_JAVA=/path/to/java`.

## Commands

```bash
uv run e2e run [scene...]            # all scenes by default; exit code 1 if any check fails
uv run e2e run vanilla-zoo --no-build   # reuse the jars already built
uv run e2e run gtnh-zoo --packs /path/to/exported-packs   # extra resource packs (file, or directory of packs)
uv run e2e approve <scene> [view...]  # copy the latest screenshots to baselines/<scene>/<view>.png
uv run e2e list                       # scenes, views and their URL hashes
uv run --group dev pytest             # unit tests + one e2e test per scene view and per scene log
uv run --group dev pytest -m "not e2e"                         # unit tests only (seconds)
uv run --group dev pytest --scene vanilla-zoo --no-build       # e2e for one scene, existing jars
```

## What `run` does per scene

1. **Jars.** Unless `--no-build`: `./gradlew :cli:shadowJar :legacy:addonJar` (plus `:gtnh-render:addonJar` once
   `gtnh-render/build.gradle.kts` exists) in `BlueMap-GTNH`, log in `out/e2e/build.log`. Then picks the newest
   `implementations/cli/build/libs/cli-*-shadow.jar` (refused unless it contains the webapp) and the newest
   `*-addon.jar` of each addon project.
2. **Config.** Wipes `out/e2e/<scene>/` and writes `config/` there: `accept-download: true`, data and webroot
   inside the scene folder, file storage, webserver on the first free port of `127.0.0.1:8200..8209`, addon jars and
   `--packs`/`extra_packs` in `config/packs/`, and one map per dimension that has a non-empty region file (D17;
   ids `overworld`, `the_nether`, `the_end`, `dim_<n>`; loader `gtnh:anvil_1_7`). Downloaded Minecraft client jars
   are kept in `out/e2e/.cache/` and hard-linked into each scene's `data/`.
3. **Render and serve.** One CLI process `-v 1.7.10 -r -f -g -w` (log `bluemap.log`). Ready once the log shows the
   render finished and saved ("Your maps are now all up-to-date!", then "Stopped.") and `settings.json` answers.
4. **Capture**, per view, in a fresh Chromium context (SwiftShader WebGL, so pixels do not depend on the host GPU):
   open `http://127.0.0.1:<port>/#<map>:<x>:<y>:<z>:<distance>:<rotation>:<angle>:<tilt>:<ortho>:<mode>` and wait
   until every tile manager (hires and each lowres LOD) has `currentlyLoading === 0` and holds a finished tile for
   every cell its spiral loader would request. The webapp's tile set and camera height then still depend on how
   the camera got there (tiles loaded around earlier centres survive at the far edge of the view square; height
   following only ever raises the camera), so the capture makes both a function of the view alone: drop every
   tile and load around the final camera position, wait until settled again, wait until the height controls
   have snapped to that tile set, and re-apply the exact view hash (the webapp rewrites the URL hash rounded). All
   of this shares one bound of 120 s. Then pin texture animations to frame 0, wait for 3 identical consecutive
   frames 2 animation frames apart (bound 60 s) and take `canvas.toDataURL()` (`preserveDrawingBuffer` is on).
   Repeated runs give pixel-identical shots. A second frame of the same camera with only the skybox drawn is
   the "sky" reference. Console errors (`console.error`, uncaught exceptions) and failed requests (network
   failures and HTTP errors other than 404 on tiles outside the rendered area) are collected.
5. **Check and report.** Writes `shots/<view>.png`, `shots/sky/<view>.png`, `diff/<view>.png` (failing views only:
   differing pixels in red over the dimmed shot) and `report.json`; prints a table.

A view fails when: there is no baseline; more than `tolerance.ratio` of its pixels differ from the baseline by more
than `tolerance.pixel` in any RGB channel; less than `min_content` of its pixels differ from the sky reference;
`max_missing` is set and exceeded; tiles or frames did not settle within their bounds; there were console errors or
failed requests. A scene also fails on `ERROR` lines in `bluemap.log`.

## Metrics

| Column | Meaning |
|---|---|
| `content` | share of pixels that differ from the sky-only frame (> 12 in some channel) |
| `magenta` | share of content pixels with BlueMap's missing-texture magenta: red ≈ blue (within 25 %), green ≤ 25 % of them, brightness ≥ 40 — any lighting level of `#FF00FF`, and the lowres missing colour (0.5, 0, 0.5) |
| `missing` | estimated share of pixels showing the missing texture: 2 × magenta, capped by content (the texture, `core/src/main/resourceExtensions/assets/bluemap/textures/block/missing.png`, is half `#FF00FF`, half `#000000`, and black cannot be told from shadow; overestimates areas where only lowres tiles are visible) |
| `miss/cont` | `missing` / `content` |
| `diff` | share of pixels differing from the baseline beyond `tolerance.pixel` |
| `load` | navigation until all tiles settled |
| `stable` | tiles settled until the stable frame |

## Scenes

`scenes/<scene>.yaml`:

```yaml
world: vanilla-zoo          # folder under <workspace>/worlds (default: the scene name)
maps: [overworld]           # optional subset of the auto-detected maps to render
extra_packs: [out/packs]    # optional, relative to the workspace
viewport: [1280, 720]
defaults:                   # merged into every view
  map: overworld
  tolerance: {pixel: 24, ratio: 0.005}
  min_content: 0.05
  max_missing: null         # share of content pixels; null = report only
views:
  top-flat:
    position: [18, 5, 17]   # x, y, z of the camera target
    distance: 30
    rotation: 0             # radians around the vertical axis
    angle: 0                # radians from straight down
    tilt: 0
    ortho: 1                # 0 perspective .. 1 orthographic
    mode: flat              # perspective | flat | free
```

Baselines in `baselines/<scene>/` are smoke baselines of the renderer state when they were approved; re-approve
after intentional rendering changes and review the new PNGs in the diff.
