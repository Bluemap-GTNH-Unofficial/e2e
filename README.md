# e2e: browser screenshot tests for BlueMap GTNH

> [!WARNING]
> Unofficial: not affiliated with BlueMap or GT New Horizons. Vibe coded (written largely by an AI coding agent), no
> quality guarantee, use at your own risk. See the
> [umbrella repository](https://github.com/Bluemap-GTNH-Unofficial/BlueMap-GTNH) for details.

Part of [Bluemap-GTNH-Unofficial/BlueMap-GTNH](https://github.com/Bluemap-GTNH-Unofficial/BlueMap-GTNH). Clone the
umbrella with `git clone --recursive`; this repository expects its sibling checkouts (`BlueMap/`, `orchestrator/` and
the workspace folders) next to it.

Browser end-to-end checks of rendered BlueMap GTNH maps. For each scene it renders a scene world with the BlueMap CLI,
serves it, opens camera views in headless Chromium, and compares screenshots with locally approved baselines.

The baselines are a BlueMap-vs-BlueMap regression check only: they show that a render did not change since it was
approved by eye, not that it is correct. Correctness comes from the oracle, which compares BlueMap with the real GTNH
client: every block state in isolation (`uv run e2e oracle states`) and scene worlds per visible block face
(`uv run e2e oracle world`), see below. The results of the last full run are summarised in
`<workspace>/out/oracle-2.9.0/SUMMARY.md` (not committed). Baselines contain Minecraft and mod textures, so they are
never committed: they live in `<workspace>/baselines-2.9.0/<scene>/<view>.png` (currently 8 scenes, 50 views, all
approved by eye; override with `BMGTNH_BASELINES`) and are regenerated on demand. `e2e run` fails a view that has no
baseline; review its screenshot in `<workspace>/out/e2e-2.9.0/<scene>/shots/` and then run
`uv run e2e approve <scene> [view...]`.

## Setup

```bash
uv sync --group dev
uv run playwright install chromium   # once per machine (~100 MB, ~/Library/Caches/ms-playwright)
```

Scene worlds come from the orchestrator (`uv run orch scene vanilla-zoo` etc. in the sibling `orchestrator/` checkout)
and live in `<workspace>/worlds-2.9.0/<scene>/`. The workspace defaults to the parent of this `e2e/` checkout; the
BlueMap fork is expected next to it in `<workspace>/BlueMap`. Override the workspace with `BMGTNH_WORKSPACE`. Java:
`mise where java@temurin-25.0.5+8.0.LTS`, override with `BMGTNH_JAVA=/path/to/java`.

**Pack version.** GTNH 2.9.0 is the only supported version. `uv run e2e --pack-version 2.9.0 <command>` (a root
option, before the command; it sets `BMGTNH_PACK_VERSION` for `gtnh-client` and `orch` too) accepts only `2.9.0`, the
default (also `$BMGTNH_PACK_VERSION`). It selects the per-version directories (`src/e2e/paths.py`), which end in
`-2.9.0`:

| | 2.9.0 |
|---|---|
| scene worlds / orchestrator outputs | `worlds-2.9.0/`, `out/scenes-2.9.0/` |
| `run` outputs / baselines | `out/e2e-2.9.0/`, `baselines-2.9.0/` |
| exported packs (`{export}`, `{addons_export}`) | `out/exporter-2.9.0/bluemap-gtnh-resources-2.9.0.zip`, `out/exporter-addons-2.9.0/…-2.9.0-addons.zip` |
| oracle outputs | `out/oracle-2.9.0/<variant>/` |
| clients / servers (ports) | `clients/gtnh-2.9.0`, `servers/gtnh-2.9.0` (25569/25579), `gtnh-addons-2.9.0` (25572/25582) |

The BlueMap build log and the Minecraft client jar cache are version-independent, in `out/e2e/`.

## Commands

```bash
uv run e2e run [scene...]            # all scenes by default; exit code 1 if any check fails
uv run e2e run vanilla-zoo --no-build   # reuse the jars already built
uv run e2e run gtnh-zoo --packs /path/to/exported-packs   # extra resource packs (file, or directory of packs)
uv run e2e approve <scene> [view...]  # copy the latest screenshots to <workspace>/baselines-2.9.0/<scene>/<view>.png
uv run e2e list                       # scenes, views and their URL hashes
uv run --group dev pytest             # unit tests + one e2e test per scene view and per scene log
uv run --group dev pytest -m "not e2e"                         # unit tests only (seconds)
uv run --group dev pytest --scene vanilla-zoo --no-build       # e2e for one scene, existing jars
uv run e2e oracle world --scene vanilla-zoo   # world oracle: BlueMap vs the real client (see below)
uv run e2e oracle world --variant modernity --scene gtnh-zoo --capture always --no-build
```

## What `run` does per scene

1. **Jars.** Unless `--no-build`: `./gradlew :cli:shadowJar :legacy:addonJar` (plus `:gtnh-render:addonJar` once
   `gtnh-render/build.gradle.kts` exists) in `<workspace>/BlueMap`, log in `out/e2e/build.log`. Then picks the newest
   `implementations/cli/build/libs/cli-*-shadow.jar` (refused unless it contains the webapp) and the newest
   `*-addon.jar` of each addon project.
2. **Config.** Wipes `out/e2e-2.9.0/<scene>/` and writes `config/` there: `accept-download: true`, data and webroot
   inside the scene folder, file storage, webserver on the first free port of `127.0.0.1:8200..8209`, addon jars and
   `--packs`/`extra_packs` in `config/packs/`, and one map per dimension that has a non-empty region file
   (ids `overworld`, `the_nether`, `the_end`, `dim_<n>`; loader `gtnh:anvil_1_7`). Downloaded Minecraft client jars
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
world: vanilla-zoo          # folder under the pack version's worlds directory (default: the scene name)
maps: [overworld]           # optional subset of the auto-detected maps to render
extra_packs: ["{export}"]   # optional: "{export}" / "{addons_export}" (the pack version's exported packs) or
                            # paths relative to the workspace
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
oracle:                     # optional, world oracle cameras (see "World oracle")
  size: [1280, 720]         # default: viewport
  fov: 70                   # vertical, degrees (default 70)
  views: [perspective]      # views to derive cameras from (default: every non-orthographic view)
  cameras:                  # extra cameras: eye + look_at, or eye + yaw + pitch (Minecraft degrees)
    south-west: {eye: [2, 23, 33], look_at: [18, 5, 17]}
  exclude:                  # block states the real client throws rendering (metas or `all`): air for the capture,
    "Natura:Natura.workbench": [13, 14, 15]   # not judged (one such block blanks its whole chunk in the client)
```

Baselines in `<workspace>/baselines-2.9.0/<scene>/` record the renderer state when they were approved; re-approve after
intentional rendering changes, after reviewing the new screenshots.

## State oracle (`oracle states`)

Compares BlueMap's rendering of every block state the exporter knows (62,659 with the default pack on 2.9.0: 1,191
vanilla states of Mojang's 1.13 flattening table plus the 61,468 states of the pack's `gtnh/blocks.ndjson`, GregTech
machines included) with the real GTNH client drawing the same state in isolation. Outputs go to
`<workspace>/out/oracle-2.9.0/<variant>/states/` and are never committed (they hold Mojang and mod textures).

```bash
cd ../gtnh-exporter/launcher && uv run gtnh-client states                 # reference: client/ (about 70 s)
cd ../../e2e && uv run e2e oracle states --variant default                # BlueMap side, compare, report (about 100 s)
# modernity: gtnh-client states --variant modernity --client-dir ../../clients/gtnh-modernity-2.9.0 \
#   --resource-pack <Modernity zip> --extra-mod <MyCTMLib jar>, then e2e oracle states --variant modernity
```

1. **Reference** (`client/`, exporter mod mode `states`, see the exporter and launcher READMEs): each state alone in
   a single-block world, drawn by the client's own `RenderBlocks.renderBlockByRenderType` (passes 0/1) and its
   tile-entity special renderer, into an offscreen framebuffer.
2. **BlueMap** (`bluemap/`, fork module `oracle`, `./gradlew :oracle:renderStates`): the state placed at
   (0, 64, 0) of a synthetic 1.7.10 chunk read by the real legacy chunk loader and flattening, with the client's
   tile-entity NBT, rendered by every BlueMap render pass with the variant's packs (loaded like the server mod:
   packs by reverse file name, `resourceExtensions.zip`, the 1.13 client jar), written as a hires tile, decoded like
   the webapp and rasterised in software.
3. **Camera** (both sides): two orthographic 128 x 128 views in one 256 x 128 image, looking at the block centre
   from 30 degrees above, `se` (yaw 45: up/south/east) and `nw` (yaw 225: up/north/west), 2 blocks across.
4. **Capture conditions** (both sides): full brightness (sky light 15, white lightmap), no ambient occlusion (BlueMap
   AO is 1 with air neighbours), no fog or clouds, noon, plains, animated textures at their first frame, nearest
   texel sampling without mipmaps. Vanilla's directional face shade is applied identically: the client draws it
   itself (non-AO path), the BlueMap rasteriser multiplies by down 0.5, up 1.0, north/south 0.8, west/east 0.6 (the
   webapp's own distance-dependent directional light is not applied). Cut-out alpha test 0.5 on the client (pass 0),
   texel discard at alpha 0.01 in BlueMap (the webapp's rule). Visual alpha on the client from drawing over black
   and over white.
5. **Face IDs** per pixel (palette PNGs `*.faces.png`): 1 down, 2 up, 3 north, 4 south, 5 west, 6 east (normal
   component at least 0.95), 7 other. Client: plane normals from its depth buffer; BlueMap: the triangle normals.
6. **Comparison** (`oracle/metrics.py`, per view): the whole view with the `view` thresholds and each axis face
   (pixels each side attributes to that face) with the `face` thresholds: alpha-coverage IoU with a 1 px boundary
   tolerance (a covered pixel matches when the other coverage has one within 1 px, so the one-pixel edge
   differences of the two rasterisers do not count), mean CIEDE2000 over pixels both draw, luminance SSIM (7 x 7
   window, 2x downsampled) on the pixels both draw (coverage differences are the IoU's). Regions under 32 covered
   pixels are not judged.
   A state passes when both views and all their faces pass. Not judged (listed separately): states the client
   draws nothing for in isolation while BlueMap draws something (upper halves, markings without support, ...), and
   states whose client render threw.
7. **Calibration**: the full runs of variant `default` classify every vanilla state against 1.7.10
   (`src/e2e/oracle/vanilla.py`, written to `out/oracle-2.9.0/default/states/vanilla.json`; other variants and the world
   oracle read it) and the first one (or `--recalibrate`) writes `out/oracle-2.9.0/default/states/calibration.json`.
   BlueMap draws vanilla blocks from the 1.13 client jar, so the thresholds are calibrated only on vanilla states
   that look the same in both versions, decided from data:
   - **textures**: every texture of the 1.13 models BlueMap draws the state with is pixel-identical (frame 0) to the
     1.7.10 texture it replaces, through the `copy` entries of `modernity/data/renames-1.7.10-to-1.13.json` and the
     client jars (`downloads/mc/minecraft-client-1.7.10.jar`, BlueMap's `out/e2e/.cache/minecraft-client-1.13.jar`);
     tinted faces also need identical grass/foliage colormaps;
   - **no model difference**: no random offset (the fork's `blockProperties.json`), no random model or rotation in
     the 1.13 blockstate (stone, grass, ...), no blockstate rotation at all (its textures turn with the model, with
     or without `uvlock`), no unrotated element face whose `uv` is not the element-bounds UV or that has a texture
     `rotation` (1.7.10's box renderer clips the texture to the bounds), no unrotated element with `shade: false`
     (BlueMap ignores the flag, see below), no tile-entity renderer on the client, no BlueMap entity model, a state
     1.7.10 has (not stone 1..6, dirt 1, sponge 1);
   - **consistent state mapping**: within the block, the client draws two metas alike exactly when BlueMap draws
     them with the same 1.13 model (farmland moisture 1-6 is wet in 1.7.10 but dry in 1.13, for example);
   - **same geometry**: per view, the pixels whose face direction (from depth) or coverage differ, per pixel of the
     client's face edges, no worse than for full cubes (1.13 model one 16^3 element, 1.7.10 render type 0; the
     largest cube value within Tukey's far-out fence).

   IoU is calibrated on these states (`geometry` set), delta E and SSIM on those of them 1.7.10 draws with its box
   or crossed-squares renderers (render types 0, 1, 31, 39; `texture` set), whose texture mapping is the 1.13
   cube/cross mapping; states `oracle/accepted.yaml` names are left out. Thresholds are Tukey's far-out fences
   (k = 3, `metrics.calibrate`) over the **inexact** samples: most regions of identical renderings agree exactly
   (IoU 1, delta E 0, SSIM 1), which makes the IQR 0, so the spread is taken from the samples that differ. IoU takes
   every inexact region; delta E and SSIM take one sample per block, its worst inexact region (a block's metas and
   faces share one model and texture set, so this weighs cut-out and translucent blocks like the rest instead of
   letting many nearly exact faces of a few blocks set the fence). Fewer than 8 inexact samples: exact agreement.
   `iou_min`/`ssim_min` = Q1 - 3 IQR, `delta_e_max` = Q3 + 3 IQR. They are not tuned by hand; `calibration.json`
   and `report.html` carry the distributions, the sets and the calibration states the thresholds fail
   (`falseFailures`). The 2.9.0 default run calibrated: view IoU 1.0, delta E max 1.465, SSIM min 0.9913; face
   IoU 0.9753, delta E max 2.894, SSIM min 0.9867. With them, 48,266 of the 61,513 judged states pass (78.5 %; 120
   accepted (1.13), 789 client blank, 237 client error); Modernity: 47,138 of 61,582 (76.5 %).

   BlueMap never reads a model element's `shade: false` (`Element.shade` is parsed but unused, the hires tile has no
   flag, and the webapp's vertex shader lights every face by its normal), so crops, torches and other
   `shade: false` elements are shaded in BlueMap but not in the client. The state rasteriser applies the face shade
   by direction just as the webapp does, so the state oracle reproduces BlueMap's behaviour there.
8. **Accepted 1.13 differences** (`oracle/accepted.yaml`, committed): vanilla state patterns with a reason
   (`texture`, `random_offset`, `random_variant` (the stone rotation), `flattening_drop`). A named state is reported
   as `accepted (1.13)`, neither pass nor failure, only where `vanilla.json` shows that reason for it in the
   variant (a texture the variant's BlueMap packs replace, e.g. Modernity's, is not a 1.13 texture there);
   otherwise it is judged and listed as unbacked. Keep the list minimal: only states that fail and whose failure
   the data explains.
9. **Report**: `report.json` (totals and pass rates per mod, thresholds, causes, the 100 worst states and the worst
   state of the 100 worst blocks, every state's status, score and failed checks, `accepted`) and `report.html` (the
   same with client | BlueMap | difference images, failing states per mod, accepted states).

`uv run e2e oracle swap-test [--before <old calibration.json>]` measures how well the thresholds discriminate: the
Modernity client's render of every state is the reference and the default client's render the test, for the
states whose two renders differ (mean absolute RGBA difference at least 1); every one of them looks different, so
the share flagged is the detection rate. It also reports each threshold set's failures on the calibration sets
(states that look the same), and writes `out/oracle-2.9.0/default/states/swap-test.json`.

The suspected causes in the report are heuristics from the metrics (silhouette, colour, brightness, texture
pattern, BlueMap draws nothing or the missing model, tile-entity-only blocks); review the images before acting.
Unit tests: `tests/test_oracle_metrics.py` (CIEDE2000 against published pairs, SSIM, regions, thresholds, Tukey
fences over inexact samples), `tests/test_oracle_states.py` (classification, judging, calibration, accepted states
and report on synthetic renders and jars), `tests/test_oracle_accepted.py` (patterns, data backing) and
`tests/test_oracle_vanilla.py` (UV, state-mapping and geometry-band rules, per-block calibration).

## World oracle (`oracle world`)

Compares BlueMap's webapp with screenshots of the real GTNH client at exactly the same camera, per visible block
face, so neighbour-dependent appearance (connected textures, pipes, cables, multiblocks, stairs and fences) and the
webapp's own shaders are checked against the game itself. Everything it writes goes to
`<workspace>/out/oracle-2.9.0/<variant>/worlds/<scene>/` and is never committed (it holds Mojang and mod textures).

```bash
uv run e2e oracle world --variant default --scene vanilla-zoo [--scene ...] [--capture auto|always|never] [--mod-jar JAR] [--no-build]
uv run e2e oracle world --variant default --rejudge --scene vanilla-zoo --scene gtnh-zoo   # judge existing reports again
uv run e2e oracle world --variant modernity --report-dir <copy of a scene directory>       # e.g. a saved earlier run
```

Variants (`src/e2e/oracle/variants.py`, directories with the `-2.9.0` suffix): `default` (client `clients/gtnh-2.9.0`,
server `gtnh-2.9.0`, BlueMap with the default export), `modernity` (client `clients/gtnh-modernity-2.9.0` with the
Modernity release zip and its MyCTMLib jar for the run; BlueMap with the default export and both `modernity` packs
of that version from `out/modernity/`), `addons` (client `clients/gtnh-addons-2.9.0` with the jars listed for 2.9.0
in `downloads/addons/versions.json`: 5 addons plus GTNLLib, which GT: Not Leisure requires; server
`gtnh-addons-2.9.0`, BlueMap with the addons export).

Per scene:

1. **Cameras.** `cameras.json`: one Minecraft camera per perspective or free view of the scene (same eye and view
   direction as the webapp view; orthographic views are skipped) plus `oracle.cameras` (scene YAML above), and as
   `remove` the dump cells of `oracle.exclude` states. Their faces are never judged (BlueMap still draws them; where
   they hide a neighbour, the occlusion rule of step 6 keeps that from counting); the report lists them under
   `excluded`.
2. **Reference.** With `--capture auto` (default) when a camera has no capture or the camera list or `remove`
   changed: `gtnh-client cameras` in the sibling `gtnh-exporter/launcher` (see its README) restores the scene world
   on the variant's server, turns random ticks off, sets the `remove` cells to air, joins it with the client and
   writes `client/<view>.{png,depth.bin,json}` (colour, depth buffer, projection and modelview matrices, eye
   position). Log: `client.log`.
3. **BlueMap.** Renders the scene world with the BlueMap CLI and the variant's packs (`bluemap/config`, `packs.txt`),
   serves it and captures each camera in headless Chromium (`src/e2e/oracle/webapp.py`): MSAA off; map controls
   detached and the three.js camera set to the client's eye, orientation and projection matrix; hires tiles loaded
   around the area the client shows; the capture conditions of the client: full brightness (`ambientLight` and
   `sunlightStrength` 1), no ambient occlusion and Minecraft's flat face shade (up 1.0, down 0.5, north/south 0.8,
   west/east 0.6; BlueMap's own directional light is 1.0 / 0.4 / 0.82 / 0.64; vertical diagonal faces such as cross
   plants unshaded like the client's crossed squares, with a tolerance of 0.02 on the normal components because
   BlueMap's normals of a 45 degree rotated element are not exactly diagonal), no mipmaps, animations at frame 0.
   The colour frame is read from the WebGL buffer once two consecutive frames are identical; a second pass writes
   the hires surfaces' view-space depth (`bluemap/<view>.png`, `bluemap/<view>.depth.bin`).
4. **Faces** (`src/e2e/oracle/faces.py`). Both renderings are labelled per pixel from their own depth and the client's
   matrices: the block cell the visible surface lies in (a hit point within the depth error of a block plane belongs
   to the cell beyond it, others are pushed a little along the ray) and the cell face the view ray enters it through.
   This equals projecting each dump block's six faces and keeping the parts the depth buffer shows, and also works
   for non-cube blocks. Only cells of the scene's dump (`out/scenes-2.9.0/<scene>/dump.ndjson`) count.
5. **Alignment** before anything is trusted: the global shift of BlueMap's labels onto the client's (best of +-3 px,
   refined to sub-pixel) and the block-edge reprojection error (distance from every client edge pixel between two
   full-cube faces to the nearest BlueMap label edge). A view passes with |shift| <= 1 px, median edge error <= 1 px
   and 90th percentile <= 2 px; faces of views that fail are reported but not judged.
6. **Metrics** per face with the shared region metrics (`src/e2e/oracle/metrics.py`): the client's region against
   BlueMap's region of the same face, leaving out pixels where the other rendering shows a nearer surface of
   something else (that difference is the occluder's). IoU, mean CIEDE2000 and SSIM; faces with fewer than 64 pixels
   are not judged. **Translucent faces** show what is behind them: where either rendering shows a translucent dump
   cell (a state whose isolated client render in the state oracle has at least half of its covered pixels below
   alpha 250: stained glass, ice, water, ...) and the view ray, followed through the dump grid past translucent
   cells, reaches an accepted (1.13) or excluded cell or nothing of the dump (the terrain around the scene, sky),
   the pixel counts for the IoU but not for delta E and SSIM (`background_excluded` in the report): the colour there
   is the background's, which is judged (or accepted) as its own face. On vanilla-zoo this is the randomly rotated
   1.13 stone floor under every glass block.
7. **Thresholds** come from Tukey's far-out fences (k = 3) over the inexact samples of the `vanilla-zoo` faces
   (default variant, every face with at least 64 pixels in an aligned view) whose states look the same in 1.7.10 and
   1.13: IoU over the faces of the `geometry` set, delta E and SSIM over the `texture` set with one sample per block
   (its worst inexact face) of `vanilla.json` (state oracle, step
   7 above; run `oracle states --variant default` first). That run writes `out/oracle-2.9.0/default/worlds/calibration.json`
   with the distributions and the calibration faces the thresholds fail; every other run reads it, so run
   vanilla-zoo first. Faces of states `oracle/accepted.yaml` names (where the data backs the reason) are reported as
   `accepted (1.13)` and counted apart.
8. **Report.** `report.json` (every face: block, mod, metrics, failed criteria, a first guess at the cause, accepted),
   `report.html` (views side by side: client, BlueMap, difference, label agreement; per-mod pass rates; the worst
   faces by mod with crops in `report/crops/`), and `../summary.json` over the variant's scenes. Exit code 1 if a
   view failed the alignment check. `--rejudge` (with `--scene`) or `--report-dir` judge an existing `report.json`
   again without capturing or rendering: the face metrics are recomputed from the saved captures
   (`client/<view>.*`, `bluemap/<view>.png`, `bluemap/<view>.depth.bin`) with the current metrics, calibration and
   `accepted.yaml` (views without a saved BlueMap depth keep their stored metrics and are listed under
   `stored_metric_views`); rejudging vanilla-zoo of the default variant in its scene directory recalibrates.

`uv run e2e oracle world-swap-test [--scene ...] [--before <old calibration.json>]` (`src/e2e/oracle/world_swap.py`)
measures how well the world thresholds discriminate: per scene (vanilla-zoo, gt-multiblocks and gtnh-zoo by
default) BlueMap's captures of each variant are judged against the other variant's client captures of the same
cameras (default BlueMap vs the Modernity client and the reverse). Counted are the faces that pass in the BlueMap
variant's own report and that the two clients draw differently (mean absolute RGB difference of the two client
captures over the face's pixels at least 1): every one of them looks different in the reference client, so the share
flagged is the detection rate, also reported by the size of the client difference. Faces whose two client captures
agree (difference below 0.25) should still pass; the share that fails is the swap's false-failure rate. Writes
`out/oracle-2.9.0/default/worlds/swap-test.json`.
