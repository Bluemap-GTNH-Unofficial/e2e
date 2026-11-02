import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from e2e.oracle import states, vanilla, variants

PALETTE = [0, 0, 0, 64, 64, 64, 255, 255, 255, 255, 0, 0, 0, 255, 255, 0, 0, 255, 255, 255, 0, 255, 0, 255]


def _block(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A box-like test image (both views): up face on top, south and east below, 4x4-texel random texture."""
    rng = np.random.default_rng(seed)
    rgba = np.zeros((128, 256, 4), np.uint8)
    faces = np.zeros((128, 256), np.uint8)
    for x0 in (0, 128):
        texture = rng.integers(40, 220, (32, 32, 3), dtype=np.uint8).repeat(4, 0).repeat(4, 1)
        for rows, cols, face in ((slice(10, 50), slice(10, 118), 2), (slice(50, 110), slice(10, 64), 4),
                                 (slice(50, 110), slice(64, 118), 6)):
            cols = slice(x0 + cols.start, x0 + cols.stop)
            h, w = rows.stop - rows.start, cols.stop - cols.start
            rgba[rows, cols, :3] = texture[:h, :w]
            rgba[rows, cols, 3] = 255
            faces[rows, cols] = face
    return rgba, faces


def _save(root, path: str, rgba: np.ndarray, faces: np.ndarray) -> None:
    target = root / f"{path}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(target)
    image = Image.fromarray(faces, "P")
    image.putpalette(PALETTE)
    image.save(root / f"{path}.faces.png", transparency=0)


def _png(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, "PNG")
    return buffer.getvalue()


def _cube(texture: str) -> str:
    faces = {f: {"texture": "#all"} for f in ("down", "up", "north", "south", "west", "east")}
    return json.dumps({"textures": {"all": texture}, "elements": [{"from": [0, 0, 0], "to": [16, 16, 16], "faces": faces}]})


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("BMGTNH_WORKSPACE", str(tmp_path))
    (tmp_path / variants.DEFAULT_PACK).parent.mkdir(parents=True)
    (tmp_path / variants.DEFAULT_PACK).write_bytes(b"")
    # the 1.7.10 vs 1.13 sources (oracle.vanilla): `block` and `shifted` share a texture identical in both jars,
    # white wool differs
    src = vanilla.sources()
    for path in (src.jar_1_7_10, src.jar_1_13, src.renames, src.block_properties):
        path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    same = rng.integers(0, 256, (16, 16, 4), dtype=np.uint8)
    same[..., 3] = 255
    wool_old, wool_new = same.copy(), same.copy()
    wool_new[..., 0] //= 2
    with zipfile.ZipFile(src.jar_1_7_10, "w") as z:
        z.writestr("assets/minecraft/textures/blocks/block.png", _png(same))
        z.writestr("assets/minecraft/textures/blocks/wool_colored_white.png", _png(wool_old))
    with zipfile.ZipFile(src.jar_1_13, "w") as z:
        z.writestr("assets/minecraft/textures/block/block.png", _png(same))
        z.writestr("assets/minecraft/textures/block/white_wool.png", _png(wool_new))
        for name, texture in ([(f"block{j}", "block/block") for j in range(10)]
                              + [("shifted", "block/block"), ("white_wool", "block/white_wool")]):
            z.writestr(f"assets/minecraft/models/block/{name}.json", _cube(texture))
            z.writestr(f"assets/minecraft/blockstates/{name}.json",
                       json.dumps({"variants": {"": {"model": f"block/{name}"}}}))
    src.renames.write_text(json.dumps({"mapped": {"block": [{"to": "block", "op": "copy"}],
                                                  "wool_colored_white": [{"to": "white_wool", "op": "copy"}]}}))
    src.block_properties.write_text("{}")
    return tmp_path


def test_run_judges_reports_and_calibrates(workspace):
    out = states.states_dir("default")
    client, bluemap = out / "client", out / "bluemap"
    entries, bm_entries = [], []
    rng = np.random.default_rng(7)

    def add(name, meta, ref, test, client_extra=None, bm_extra=None, state=None):
        ns, value = name.split(":")
        path = f"{ns}/{value}/{meta}"
        entries.append({"key": f"{name}[meta={meta}]", "name": name, "meta": meta, "path": path, "renderType": 0,
                        **(client_extra or {})})
        model = {"blockState": f"minecraft:{state}", "variants": [
            {"renderer": "bluemap:default", "model": f"minecraft:block/{state}"}]} if state else {}
        bm_entries.append({"key": f"{name}[meta={meta}]", "path": path, **model, **(bm_extra or {})})
        _save(client, path, *ref)
        _save(bluemap, path, *test)

    for i in range(40):  # ten vanilla blocks of four metas each (the client draws a block's metas alike), BlueMap with
        rgba, faces = _block(i // 4)  # small, varying noise: the calibration bulk
        noisy = rgba.copy()
        amplitude = 1 + (i // 4) % 4  # per block: the blocks' residuals spread like real ones
        noise = rng.integers(-amplitude, amplitude + 1, rgba[..., :3].shape)
        noisy[..., :3] = np.clip(rgba[..., :3].astype(int) + noise, 0, 255)
        noisy[..., 3] = rgba[..., 3]
        add(f"minecraft:block{i // 4}", i % 4, (rgba, faces), (noisy, faces), state=f"block{i // 4}")
    rgba, faces = _block(100)
    add("minecraft:shifted", 0, (rgba, faces), (np.roll(rgba, 24, axis=1), np.roll(faces, 24, axis=1)), state="shifted")
    rgba, faces = _block(101)
    red = rgba.copy()
    red[..., 1:3] //= 3
    add("mod:tinted", 0, (rgba, faces), (red, faces))
    add("minecraft:wool", 0, (rgba, faces), (red, faces), state="white_wool")  # 1.13 texture: accepted.yaml
    empty = (np.zeros((128, 256, 4), np.uint8), np.zeros((128, 256), np.uint8))
    add("mod:invisible", 0, empty, empty, {"blank": True}, {"blank": True})
    add("mod:upper_half", 8, empty, _block(102), {"blank": True})
    add("mod:broken", 0, _block(103), _block(103), {"error": "tesr pass 0: NullPointerException"})

    (client / "index.json").write_text(json.dumps({"format": 1, "states": entries, "seconds": 1.0}))
    (bluemap / "index.json").write_text(json.dumps({"format": 1, "states": bm_entries, "seconds": 2.0}))

    code = states.run(variants.load("default"), render=False, only=None, workers=2, recalibrate=False)
    assert code == 1  # two states fail

    report = json.loads((out / "report.json").read_text())
    assert report["calibrationWritten"] is True and states.calibration_file().is_file()
    statuses = {w["key"]: w for w in report["worst"]}
    assert set(statuses) == {"minecraft:shifted[meta=0]", "mod:tinted[meta=0]"}
    assert any("delta_e" in c for c in statuses["mod:tinted[meta=0]"]["checks"])
    assert report["mods"]["minecraft"]["pass"] == 40 and report["mods"]["minecraft"]["fail"] == 1
    assert report["mods"]["minecraft"]["accepted"] == 1 and report["accepted"]["unbacked"] == []
    wool = next(s for s in report["states"] if s["key"] == "minecraft:wool[meta=0]")
    assert wool["status"] == "accepted" and "1.13" in wool["accepted"] and wool["checks"]
    assert report["mods"]["mod"] == {"states": 4, "judged": 2, "pass": 1, "fail": 1, "passRate": 0.5, "accepted": 0,
                                     "clientBlank": 1, "clientError": 1, "bluemapMissing": 0}
    classes = json.loads(vanilla.file().read_text())
    assert classes["states"]["minecraft:wool[meta=0]"]["textures"] == {"block/white_wool": "differs"}
    assert classes["states"]["minecraft:shifted[meta=0]"]["sameGeometry"] is False
    calibration = json.loads(states.calibration_file().read_text())
    assert calibration["states"] == 40 and calibration["geometryStates"] == 40  # identical-looking vanilla states only
    assert calibration["falseFailures"]["texture"]["fail"] == 0
    assert calibration["face"]["thresholds"]["iou_min"] > 0.9
    html = (out / "report.html").read_text()
    assert "minecraft:shifted[meta=0]" in html and (out / "diff" / "mod" / "tinted" / "0.png").is_file()

    # a second run reads the calibration instead of writing it
    assert states.run(variants.load("default"), render=False, only="mod:", workers=1, recalibrate=False) == 1
    assert json.loads((out / "report.json").read_text())["calibrationWritten"] is False
