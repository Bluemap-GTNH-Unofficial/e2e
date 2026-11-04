from __future__ import annotations

import math

import numpy as np
import pytest
import yaml

from e2e.oracle import camera as cameras
from e2e.oracle import machine_cameras as mc

SIZE = (1280, 720)
FOV = 70.0


def _projected(eye: np.ndarray, target: np.ndarray, box: list[int]) -> list[tuple[float, float]]:
    """(horizontal, vertical) tangents of the box corners in a camera at ``eye`` looking at ``target``."""
    forward = (target - eye) / np.linalg.norm(target - eye)
    right = np.cross(forward, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    x0, y0, z0, x1, y1, z1 = box
    out = []
    for x in (x0, x1 + 1):
        for y in (y0, y1 + 1):
            for z in (z0, z1 + 1):
                p = np.array([x, y, z], dtype=float) - eye
                depth = p @ forward
                out.append((float(p @ right / depth), float(p @ up / depth)))
    return out


@pytest.mark.parametrize("key, box", [
    ("mega-ebf", [54, 5, 97, 68, 24, 111]),
    ("mega-dt", [108, 5, 97, 122, 58, 105]),  # tall and thin
    ("lhc", [164, 5, 258, 278, 20, 382]),  # flat
    ("fusion-mk1", [91, 5, 11, 105, 7, 25]),  # flat
])
def test_machine_camera_fits_the_box_within_the_margin(key, box):
    eye, target = mc.machine_camera({"key": key, "box": box}, FOV, SIZE)
    framing = mc.FLAT_VIEW if mc.FLAT_MACHINES.match(key) else mc.MACHINE_VIEW
    assert np.allclose(target, mc.centre(box))
    tan_v = math.tan(math.radians(FOV) / 2) * framing.margin
    tan_h = tan_v * SIZE[0] / SIZE[1]
    corners = _projected(eye, target, box)
    assert all(abs(h) <= tan_h + 1e-9 and abs(v) <= tan_v + 1e-9 for h, v in corners)
    # pulled back no further than needed: some corner touches the margin
    assert max(max(abs(h) / tan_h, abs(v) / tan_v) for h, v in corners) == pytest.approx(1.0)
    # from the north-west, looking down at the demo's angle
    yaw, pitch = cameras.look_at(tuple(eye), tuple(target))
    assert eye[0] < target[0] and eye[2] < target[2]
    assert pitch == pytest.approx(90.0 - math.degrees(framing.angle))
    assert yaw == pytest.approx((180.0 + math.degrees(framing.rotation)) % 360.0)


def test_machine_camera_stays_over_the_plaza():
    godforge = {"key": "godforge", "box": [25, 5, 258, 151, 33, 444]}
    plaza = [0, 4, 0, 304, 58, 455]
    free, _ = mc.machine_camera(godforge, FOV, SIZE)
    eye, target = mc.machine_camera(godforge, FOV, SIZE, plaza)
    assert free[0] < 0  # the fitted eye is west of the plaza
    assert eye[0] == pytest.approx(0.0) and eye[2] > 0
    # moved along the same line of sight
    assert cameras.look_at(tuple(eye), tuple(target)) == pytest.approx(cameras.look_at(tuple(free), tuple(target)))
    inside, _ = mc.machine_camera({"key": "eec", "box": [246, 5, 97, 250, 11, 101]}, FOV, SIZE, plaza)
    assert np.allclose(inside, mc.machine_camera({"key": "eec", "box": [246, 5, 97, 250, 11, 101]}, FOV, SIZE)[0])


def test_update_replaces_only_the_generated_block():
    text = (
        "world: endgame\nviewport: [1280, 720]\nviews: {}\noracle:\n  size: [1280, 720]\n  cameras:\n"
        f"    {mc.BEGIN}\n    stale: {{eye: [0, 0, 0], yaw: 0, pitch: 0}}\n    {mc.END}\n  exclude:\n    \"a:b\": [1]\n"
    )
    scene = {"machines": [{"key": "eec", "label": "Extreme Entity Crusher", "tier": 11, "box": [246, 5, 97, 250, 11, 101]}]}
    updated = mc.update(text, scene)
    raw = yaml.safe_load(updated)
    assert list(raw["oracle"]["cameras"]) == ["eec"]
    assert raw["oracle"]["exclude"] == {"a:b": [1]}
    eye, target = mc.machine_camera(scene["machines"][0], FOV, SIZE)
    assert raw["oracle"]["cameras"]["eec"]["eye"] == pytest.approx(list(eye), abs=0.01)
    assert raw["oracle"]["cameras"]["eec"]["look_at"] == pytest.approx(list(target), abs=0.01)
    assert mc.update(updated, scene) == updated


def test_update_needs_the_markers():
    with pytest.raises(ValueError):
        mc.update("views: {}\noracle:\n  cameras: {}\n", {"machines": []})


def test_endgame_scene_has_one_camera_per_generated_line():
    names = [c.name for c in cameras.scene_cameras("endgame")]
    assert names and len(names) == len(set(names))
    assert {"fusion-mk1", "godforge", "lhc"} <= set(names)
