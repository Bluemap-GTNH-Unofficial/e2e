"""Which vanilla block states look the same in the real 1.7.10 client and in the 1.13 assets BlueMap draws them from.

BlueMap keeps the 1.13 client jar's textures and models for vanilla blocks. The thresholds of both oracles are
calibrated only on vanilla states whose 1.7.10 and 1.13 appearance is the same, decided from data, and
``oracle/accepted.yaml`` may only accept differences this data shows. The default state run classifies every vanilla
state (``out/oracle/default/states/vanilla.json``; other variants and the world oracle read it):

- ``textures``: every texture of the 1.13 models BlueMap draws the state with (frame 0, RGBA) against the 1.7.10
  texture it replaces, through the ``copy`` entries of the rename table (``modernity/data/renames-1.7.10-to-1.13.json``)
  and the two client jars (``downloads/mc/minecraft-client-1.7.10.jar``, BlueMap's downloaded 1.13 jar): ``identical``,
  ``differs``, ``no 1.7.10 source``, ``converted (<op>)``. Tinted faces also need identical grass/foliage colormaps.
- ``randomOffset``: BlueMap moves the model by a position hash (``randomOffset`` in the fork's ``blockProperties.json``).
- ``randomVariant``: the 1.13 blockstate picks one of several models or rotations by position (stone, grass, ...).
- ``rotated``: the 1.13 blockstate rotates the model (``x``/``y``, with or without ``uvlock``): which texture faces
  where, and how it is turned, then follows 1.13's blockstate.
- ``customUv``: an unrotated element face with a texture ``rotation`` or a ``uv`` that is not the element-bounds UV
  (1.7.10's box renderer clips the texture to the bounds).
- ``unshadedFaces``: an unrotated element with ``shade: false``; BlueMap ignores the flag and shades it anyway.
- ``mappingConsistent``: within the block, the client draws two metas alike exactly when BlueMap draws them with
  the same 1.13 model (otherwise the flattening maps the meta to a state 1.13 draws differently, e.g. farmland).
- ``geometry``: per view, the pixels whose face direction (from depth) or coverage differ between the client and
  BlueMap, per pixel of the client's face edges (the worse view). The geometry is the same when this is no worse than
  for any full cube (1.13 model one 16x16x16 element, 1.7.10 render type 0): rasterisation band, not shape.
- ``flatteningDrop``: the client draws the state, BlueMap draws nothing (the 1.13 upgrade turns it into air).
- ``tesr`` (the client adds a tile-entity special renderer) and ``entityRenderer`` (BlueMap draws a 1.13 entity
  model instead of a block model).

Calibration sets (``sets``):

- ``geometry``: comparable states (both draw, the state exists in 1.7.10, no TESR, BlueMap's block-model renderer)
  with identical textures, no random offset or variant, not rotated, no custom UV, no ``shade: false`` element, a
  consistent state mapping, the same geometry. IoU is calibrated on these.
- ``texture``: the ``geometry`` states 1.7.10 draws with its box or crossed-squares renderers (render types 0, 1, 31,
  39), whose texture mapping is the 1.13 cube/cross mapping. Delta E and SSIM are calibrated on these.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image

from e2e import paths
from e2e.oracle import metrics

FORMAT = 1
BOX_RENDER_TYPES = (0, 1, 31, 39)  # 1.7.10 standard block, crossed squares, log, quartz pillar
FULL_CUBE = ([0, 0, 0], [16, 16, 16])

# Vanilla metas that Mojang's 1.13 flattening table knows but 1.7.10 does not have (the 1.7.10 client draws the
# block's base state for them): never in a calibration set.
NOT_IN_1_7_10 = {
    **{("minecraft:stone", m): "granite/diorite/andesite were added in 1.8" for m in range(1, 7)},
    ("minecraft:dirt", 1): "coarse dirt was added in 1.8",
    ("minecraft:sponge", 1): "wet sponge was added in 1.8",
}

# accepted.yaml reasons this data can back (see `differences`)
REASONS = ("texture", "random_offset", "random_variant", "flattening_drop")
AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air")
TEXTURE_DIFFERS = ("differs", "no 1.7.10 source", "converted", "grass/foliage colormap differs")


@dataclass(frozen=True)
class Sources:
    jar_1_7_10: Path
    jar_1_13: Path
    renames: Path
    block_properties: Path

    def missing(self) -> list[Path]:
        return [p for p in (self.jar_1_7_10, self.jar_1_13, self.renames, self.block_properties) if not p.is_file()]


def sources() -> Sources:
    ws = paths.workspace()
    return Sources(
        jar_1_7_10=ws / "downloads" / "mc" / "minecraft-client-1.7.10.jar",
        jar_1_13=paths.download_cache() / "minecraft-client-1.13.jar",
        renames=ws / "modernity" / "data" / "renames-1.7.10-to-1.13.json",
        block_properties=paths.bluemap_repo() / "core" / "src" / "main" / "resourceExtensions" / "assets" / "minecraft"
        / "blockProperties.json",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_state(state: str) -> tuple[str, dict[str, str]]:
    """``minecraft:oak_log[axis=y]`` -> (``minecraft:oak_log``, ``{"axis": "y"}``)."""
    match = re.fullmatch(r"([^\[]+)(?:\[(.*)\])?", state)
    if not match:
        raise ValueError(f"not a block state: {state!r}")
    props = dict(kv.split("=", 1) for kv in match.group(2).split(",")) if match.group(2) else {}
    return match.group(1), props


class Assets:
    """The two client jars: 1.13 blockstates and models, frame 0 of block textures of both versions."""

    def __init__(self, src: Sources):
        self.src = src
        self.old = zipfile.ZipFile(src.jar_1_7_10)
        self.new = zipfile.ZipFile(src.jar_1_13)
        renames = json.loads(src.renames.read_text(encoding="utf-8"))
        self.sources: dict[str, list[tuple[str, str]]] = {}
        for old, entries in renames["mapped"].items():
            for entry in entries:
                self.sources.setdefault(entry["to"], []).append((old, entry["op"]))
        self.properties = json.loads(src.block_properties.read_text(encoding="utf-8"))
        self._models: dict[str, dict | None] = {}
        self._textures: dict[str, dict] = {}

    def close(self) -> None:
        self.old.close()
        self.new.close()

    @staticmethod
    def _json(jar: zipfile.ZipFile, name: str) -> dict | None:
        try:
            return json.loads(jar.read(name))
        except KeyError:
            return None

    @staticmethod
    def frame0(jar: zipfile.ZipFile, name: str) -> np.ndarray | None:
        """The first animation frame (``frames[0]`` of the ``.mcmeta``, else the top square) as RGBA."""
        try:
            data = jar.read(name)
        except KeyError:
            return None
        with Image.open(io.BytesIO(data)) as image:
            rgba = np.asarray(image.convert("RGBA"))
        width = rgba.shape[1]
        if rgba.shape[0] > width:
            index = 0
            meta = Assets._json(jar, name + ".mcmeta")
            frames = (meta or {}).get("animation", {}).get("frames")
            if frames:
                index = frames[0] if isinstance(frames[0], int) else frames[0]["index"]
            rgba = rgba[index * width:(index + 1) * width]
        return rgba

    @cached_property
    def colormaps_identical(self) -> dict[str, bool]:
        out = {}
        for name in ("grass", "foliage"):
            path = f"assets/minecraft/textures/colormap/{name}.png"
            a, b = self.frame0(self.old, path), self.frame0(self.new, path)
            out[name] = a is not None and b is not None and a.shape == b.shape and bool((a == b).all())
        return out

    def model(self, name: str) -> dict | None:
        """The 1.13 model ``name`` (``minecraft:block/x`` or ``block/x``): parent chain, elements, textures, tint."""
        key = name.split(":")[-1]
        if key in self._models:
            return self._models[key]
        chain, layers, elements = [key], [], None
        current = self._json(self.new, f"assets/minecraft/models/{key}.json")
        if current is None:
            self._models[key] = None
            return None
        while current is not None:
            layers.append(current.get("textures", {}))
            if elements is None and "elements" in current:
                elements = current["elements"]
            parent = current.get("parent")
            if not parent:
                break
            parent = parent.split(":")[-1]
            chain.append(parent)
            current = self._json(self.new, f"assets/minecraft/models/{parent}.json")
        variables: dict[str, str] = {}
        for layer in reversed(layers):
            variables.update(layer)

        def resolve(value: str | None) -> str | None:
            for _ in range(10):
                if not (isinstance(value, str) and value.startswith("#")):
                    break
                value = variables.get(value[1:])
            return value

        textures, tinted = set(), False
        for element in elements or []:
            for face in element.get("faces", {}).values():
                texture = resolve(face.get("texture"))
                if texture and not texture.startswith("#"):
                    textures.add(texture.split(":")[-1])
                tinted |= "tintindex" in face
        result = {"chain": chain, "elements": elements or [], "textures": sorted(textures), "tinted": tinted}
        self._models[key] = result
        return result

    def texture(self, name: str) -> dict:
        """``block/x`` of 1.13 against its 1.7.10 source(s) in the rename table."""
        if name in self._textures:
            return self._textures[name]
        short = name.split("/")[-1]
        new = self.frame0(self.new, f"assets/minecraft/textures/{name}.png")
        result: dict = {"status": "no 1.7.10 source", "from": []}
        if new is None:
            result["status"] = "not in the 1.13 jar"
        for old_name, op in self.sources.get(short, []):
            result["from"].append(old_name)
            old = self.frame0(self.old, f"assets/minecraft/textures/blocks/{old_name}.png")
            if old is None:
                status = "1.7.10 texture missing"
            elif op != "copy":
                status = f"converted ({op})"
            elif old.shape != new.shape or not (old == new).all():
                status = "differs"
            else:
                status = "identical"
            if result["status"] != "identical" and new is not None:
                result["status"] = status
        self._textures[name] = result
        return result

    def blockstate_entries(self, state: str) -> tuple[list[dict], bool] | None:
        """The 1.13 blockstate entries that apply to ``state`` and whether one of them is a random choice."""
        name, props = parse_state(state)
        data = self._json(self.new, f"assets/minecraft/blockstates/{name.split(':')[-1]}.json")
        if data is None:
            return None
        entries: list[dict] = []
        random = False

        def take(value) -> None:
            nonlocal random
            options = value if isinstance(value, list) else [value]
            if len({(o.get("model"), o.get("x", 0), o.get("y", 0)) for o in options}) > 1:
                random = True
            entries.extend(options)

        if "variants" in data:
            for key, value in data["variants"].items():
                condition = dict(kv.split("=", 1) for kv in key.split(",") if kv)
                if all(props.get(k) == v for k, v in condition.items()):
                    take(value)
                    break
        for part in data.get("multipart", []):
            if _applies(part.get("when"), props):
                take(part["apply"])
        return entries, random


def _applies(when: dict | None, props: dict[str, str]) -> bool:
    if when is None:
        return True
    if "OR" in when:
        return any(_applies(w, props) for w in when["OR"])
    return all(props.get(k) in str(v).lower().split("|") for k, v in when.items())  # JSON booleans: "false"


def _faces(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def _alpha(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"))[..., 3] >= metrics.ALPHA_MIN


def geometry_disagreement(ref_faces: np.ndarray, ref_cover: np.ndarray, test_faces: np.ndarray,
                          test_cover: np.ndarray, views: int) -> float:
    """Worst view's count of pixels whose face direction or coverage differ, per pixel of the reference's face edges."""
    width = ref_faces.shape[1] // views
    worst = 0.0
    for v in range(views):
        cols = slice(v * width, (v + 1) * width)
        a = np.where(ref_cover[:, cols], ref_faces[:, cols], 0).astype(np.int16)
        b = np.where(test_cover[:, cols], test_faces[:, cols], 0).astype(np.int16)
        differ = int((a != b).sum())
        p = np.pad(a, 1, mode="edge")
        centre = p[1:-1, 1:-1]
        edges = int(((centre != p[:-2, 1:-1]) | (centre != p[2:, 1:-1]) | (centre != p[1:-1, :-2])
                     | (centre != p[1:-1, 2:])).sum())
        worst = max(worst, differ / max(edges, 1))
    return worst


def geometry_band(cube_ratios: list[float], fence: float = 3.0) -> float:
    """The rasterisation band of identical geometry: the largest full-cube ratio that is not a far-out outlier.
    Most full cubes share one ratio (the same opaque cube drawn by both renderers); Tukey's fence is taken over the
    others (cut-out, translucent cubes), so one cube BlueMap draws wrongly cannot widen the band. Fewer than
    ``metrics.MIN_INEXACT`` other cubes: the most common ratio."""
    if not cube_ratios:
        return 0.0
    values = np.asarray(cube_ratios, dtype=np.float64)
    common, _ = Counter(cube_ratios).most_common(1)[0]
    others = values[values != common]
    if others.size < metrics.MIN_INEXACT:
        return float(common)
    q1, q3 = np.percentile(others, [25, 75])
    return float(values[values <= q3 + fence * (q3 - q1)].max())


def _is_full_cube(model: dict) -> bool:
    elements = model["elements"]
    return (len(elements) == 1 and (elements[0]["from"], elements[0]["to"]) == FULL_CUBE
            and not elements[0].get("rotation"))


def _mark_mapping(records: dict[str, dict], looks: dict[str, str]) -> None:
    """``mappingConsistent``: within a block, the 1.7.10 client draws two metas alike exactly when BlueMap draws them
    with the same 1.13 state model. A look the client shares between metas that map to different 1.13 models (or one
    1.13 model for metas the client draws differently) means the flattening maps the meta to a state 1.13 draws
    otherwise (e.g. farmland moisture 1-6 is wet in 1.7.10, dry in 1.13). States the client does not draw: False."""
    by_look: dict[tuple[str, str], set] = {}
    by_model: dict[tuple[str, tuple], set] = {}
    for key, r in records.items():
        if key in looks:
            model = (r["state"].split("[")[0], tuple(r["models"]))
            by_look.setdefault((r["name"], looks[key]), set()).add(model)
            by_model.setdefault((r["name"], model), set()).add(looks[key])
    for key, r in records.items():
        model = (r["state"].split("[")[0], tuple(r["models"]))
        r["mappingConsistent"] = (key in looks and len(by_look[(r["name"], looks[key])]) == 1
                                  and len(by_model[(r["name"], model)]) == 1)


def classify(client_index: dict, bluemap_index: dict, client_root: Path, bluemap_root: Path, src: Sources,
             views: int) -> dict:
    """Every vanilla state of the client index (see the module doc)."""
    missing = src.missing()
    if missing:
        raise FileNotFoundError("1.7.10 vs 1.13 classification needs " + ", ".join(map(str, missing)))
    assets = Assets(src)
    try:
        bm_states = {s["path"]: s for s in bluemap_index.get("states", [])}
        records: dict[str, dict] = {}
        cube_ratios: list[float] = []
        looks: dict[str, str] = {}  # key -> hash of the client's render
        for s in client_index["states"]:
            if not s["name"].startswith("minecraft:"):
                continue
            bm = bm_states.get(s["path"], {})
            r = _record(s, bm, assets)
            ref_png, test_png = client_root / f"{s['path']}.png", bluemap_root / f"{s['path']}.png"
            drawn = (not s.get("blank") and not s.get("error") and bm and not bm.get("error") and not bm.get("blank")
                     and test_png.is_file())
            r["flatteningDrop"] = bool(not s.get("blank") and not s.get("error") and bm
                                       and parse_state(r["state"] or "minecraft:air")[0] in AIR)
            if drawn:
                r["geometry"] = round(geometry_disagreement(
                    _faces(client_root / f"{s['path']}.faces.png"), _alpha(ref_png),
                    _faces(bluemap_root / f"{s['path']}.faces.png"), _alpha(test_png), views), 4)
            r["comparable"] = bool(drawn and (s["name"], s["meta"]) not in NOT_IN_1_7_10 and not r["tesr"]
                                   and not r["entityRenderer"])
            models = [assets.model(m) for m in r["models"]]
            r["fullCube"] = bool(models) and all(m is not None and _is_full_cube(m) for m in models)
            r["boxRenderer"] = r["renderType"] in BOX_RENDER_TYPES
            if ref_png.is_file() and not s.get("blank") and not s.get("error"):
                with Image.open(ref_png) as image:
                    looks[s["key"]] = hashlib.sha1(np.asarray(image.convert("RGBA")).tobytes()).hexdigest()
            if (r["comparable"] and r["texturesIdentical"] and r["fullCube"] and r["renderType"] == 0
                    and not r["rotated"] and not r["randomVariant"] and r.get("geometry") is not None):
                cube_ratios.append(r["geometry"])
            records[s["key"]] = r
        band = geometry_band(cube_ratios)
        _mark_mapping(records, looks)
        geometry_set, texture_set = [], []
        for key, r in records.items():
            r["sameGeometry"] = None if r.get("geometry") is None else r["geometry"] <= band
            if (r["comparable"] and r["texturesIdentical"] and not r["randomOffset"] and not r["randomVariant"]
                    and not r["rotated"] and not r["customUv"] and not r["unshadedFaces"] and r["sameGeometry"]
                    and r["mappingConsistent"]):
                geometry_set.append(key)
                if r["boxRenderer"]:
                    texture_set.append(key)
        return {
            "format": FORMAT,
            "sources": {"jar_1_7_10": str(src.jar_1_7_10), "jar_1_7_10_sha256": _sha256(src.jar_1_7_10),
                        "jar_1_13": str(src.jar_1_13), "jar_1_13_sha256": _sha256(src.jar_1_13),
                        "renames": str(src.renames), "blockProperties": str(src.block_properties)},
            "colormapsIdentical": assets.colormaps_identical,
            "geometryBand": {"value": band, "fullCubes": len(cube_ratios),
                             "rule": "largest ratio over full cubes (1.13 model one 16^3 element, 1.7.10 render "
                                     "type 0, identical textures, not rotated) within Tukey's far-out fence of the "
                                     "cubes that differ from the most common ratio; the ratio is the worse view's "
                                     "face-direction/coverage disagreement per client face-edge pixel"},
            "sets": {"geometry": geometry_set, "texture": texture_set},
            "states": records,
        }
    finally:
        assets.close()


def _record(s: dict, bm: dict, assets: Assets) -> dict:
    variants = bm.get("variants") or []
    state = bm.get("blockState") or ""
    models = [v["model"] for v in variants if v.get("model")]
    r = {"path": s["path"], "name": s["name"], "meta": s["meta"], "state": state, "models": models,
         "renderType": s.get("renderType"), "tesr": bool(s.get("tesr")),
         "entityRenderer": any(v.get("renderer") != "bluemap:default" or ":entity/" in (v.get("model") or "")
                               for v in variants)}
    textures: dict[str, str] = {}
    tinted = False
    for name in models:
        if ":entity/" in name or parse_state(state or "minecraft:air")[0] in AIR:
            continue  # BlueMap's own entity models, or nothing drawn
        model = assets.model(name)
        if model is None:
            textures[name] = "model not in the 1.13 jar"
            continue
        tinted |= model["tinted"]
        for texture in model["textures"]:
            textures[texture] = assets.texture(texture)["status"]
    if tinted and not all(assets.colormaps_identical.values()):
        textures["colormap"] = "grass/foliage colormap differs"
    r["textures"] = textures
    r["texturesIdentical"] = bool(textures) and all(v == "identical" for v in textures.values())
    block = parse_state(state)[0] if state else ""
    r["randomOffset"] = bool(assets.properties.get(block, {}).get("randomOffset"))
    entries = assets.blockstate_entries(state) if state else None
    r["randomVariant"] = bool(entries and entries[1])
    r["rotated"] = bool(entries and not entries[1] and any(e.get("x", 0) or e.get("y", 0) for e in entries[0]))
    block_models = [m for m in (assets.model(n) for n in models if ":entity/" not in n) if m is not None]
    r["customUv"] = any(_custom_uv(e) for m in block_models for e in m["elements"])
    r["unshadedFaces"] = any(e.get("shade") is False and not e.get("rotation") for m in block_models
                             for e in m["elements"])
    return r


def auto_uv(face: str, low: list[float], high: list[float]) -> list[float]:
    """The UV Minecraft derives for an element face without ``uv`` (the texture clipped to the element's bounds,
    like 1.7.10's box renderer clips it to the block bounds)."""
    x1, y1, z1 = low
    x2, y2, z2 = high
    return {"down": [x1, 16 - z2, x2, 16 - z1], "up": [x1, z1, x2, z2], "north": [16 - x2, 16 - y2, 16 - x1, 16 - y1],
            "south": [x1, 16 - y2, x2, 16 - y1], "west": [z1, 16 - y2, z2, 16 - y1],
            "east": [16 - z2, 16 - y2, 16 - z1, 16 - y1]}[face]


def _custom_uv(element: dict) -> bool:
    """An unrotated element with a face whose texture is rotated or not clipped to the element bounds."""
    if element.get("rotation"):
        return False  # crossed planes: the full texture on each plane, like 1.7.10's crossed squares
    for name, face in element.get("faces", {}).items():
        if face.get("rotation"):
            return True
        if "uv" in face and [float(v) for v in face["uv"]] != [float(v) for v in auto_uv(name, element["from"],
                                                                                         element["to"])]:
            return True
    return False


def texture_overrides(packs: Iterable[Path]) -> set[str]:
    """``block/<name>`` textures of the ``minecraft`` namespace that a variant's BlueMap packs replace."""
    out = set()
    pattern = re.compile(r"assets/minecraft/textures/(block/[^/]+)\.png$")
    for pack in packs:
        if pack.is_file() and zipfile.is_zipfile(pack):
            with zipfile.ZipFile(pack) as z:
                names = z.namelist()
        elif pack.is_dir():
            names = [p.relative_to(pack).as_posix() for p in pack.rglob("*.png")]
        else:
            continue
        out.update(m.group(1) for m in map(pattern.match, names) if m)
    return out


def differences(record: dict, overrides: set[str]) -> set[str]:
    """The 1.13 differences (``REASONS``) the data shows for a state, for a variant whose packs replace ``overrides``.
    A differing texture the variant's packs replace is drawn from the pack, not the 1.13 jar."""
    out = set()
    if any(name not in overrides and status.startswith(TEXTURE_DIFFERS) for name, status in record["textures"].items()):
        out.add("texture")
    if record["randomOffset"]:
        out.add("random_offset")
    if record["randomVariant"]:
        out.add("random_variant")
    if record.get("flatteningDrop"):
        out.add("flattening_drop")
    return out


def file() -> Path:
    from e2e.oracle import states  # the default variant's state oracle directory

    return states.states_dir(states.CALIBRATION_VARIANT) / "vanilla.json"


def load() -> dict:
    path = file()
    if not path.is_file():
        raise FileNotFoundError(f"{path} missing: run `e2e oracle states --variant default` first (it classifies the "
                                f"vanilla states the calibration and accepted.yaml rely on)")
    return json.loads(path.read_text(encoding="utf-8"))


def label(key: str) -> str:
    """``minecraft:stone[meta=0]`` -> the world oracle's ``minecraft:stone:0``."""
    name, _, meta = key.partition("[meta=")
    return f"{name}:{meta.rstrip(']')}"
