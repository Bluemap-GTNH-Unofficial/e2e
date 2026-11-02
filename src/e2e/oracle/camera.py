"""World-oracle cameras: definitions per scene, Minecraft camera conventions and the matrices the client recorded.

Cameras come from the scene YAML (``scenes/<scene>.yaml``): every perspective or free e2e view is turned into the
Minecraft camera that sees the same thing (orthographic views have no Minecraft equivalent and are skipped), plus the
cameras listed under ``oracle.cameras``::

    oracle:
      size: [1280, 720]      # default: the scene viewport
      fov: 70                # vertical, degrees (default 70)
      views: [perspective]   # e2e views to derive cameras from (default: every non-orthographic view)
      cameras:
        south-east: {eye: [40, 20, 40], look_at: [18, 5, 17]}   # or yaw/pitch instead of look_at
      exclude:               # block states the real client cannot render (it throws), metas or `all`
        "Natura:Natura.workbench": [13, 14, 15]

Excluded cells are set to air on the server before the client joins (``remove`` in ``cameras.json``) and their faces
are not judged: the client's chunk mesher fails a whole chunk on the first block that throws, so it would draw nothing.

Minecraft angles (degrees): yaw 0 looks south (+z), 90 west (-x), 180 north, 270 east; pitch 90 looks straight down.
The client's camera is at the player's eye; the projection, the modelview (rotation and the 0.1 block step back of
``EntityRenderer.orientCamera``) and the eye position are recorded per view, and world -> eye is
``modelview @ translate(-camera)``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from e2e import paths, scenes

CAMERAS_FORMAT = 1
DEFAULT_FOV = 70.0


@dataclass(frozen=True)
class Camera:
    name: str
    eye: tuple[float, float, float]
    yaw: float
    pitch: float
    fov: float
    width: int
    height: int
    source: str  # "view <name>" or "oracle.cameras"

    def to_json(self) -> dict:
        return {"name": self.name, "eye": [round(v, 4) for v in self.eye], "yaw": round(self.yaw, 4),
                "pitch": round(self.pitch, 4), "fov": self.fov, "width": self.width, "height": self.height,
                "source": self.source}


def direction(yaw: float, pitch: float) -> np.ndarray:
    """Unit view direction of a Minecraft yaw/pitch (degrees)."""
    y, p = math.radians(yaw), math.radians(pitch)
    return np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])


def look_at(eye: tuple[float, float, float], target: tuple[float, float, float]) -> tuple[float, float]:
    """Minecraft (yaw, pitch) in degrees looking from ``eye`` at ``target``."""
    dx, dy, dz = (t - e for t, e in zip(target, eye))
    yaw = math.degrees(math.atan2(-dx, dz)) % 360.0
    pitch = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    return yaw, pitch


def from_view(view: scenes.View, fov: float, size: tuple[int, int]) -> Camera:
    """The Minecraft camera of an e2e webapp view (perspective orbit around ``position``, or free flight at it).

    The webapp's view direction is ``(sin a sin r, -cos a, -sin a cos r)`` for rotation ``r`` (0 = north) and angle
    ``a`` (0 = straight down); the camera sits ``distance`` back along it (0 in free flight)."""
    r, a = view.rotation, view.angle
    d = np.array([math.sin(a) * math.sin(r), -math.cos(a), -math.sin(a) * math.cos(r)])
    distance = 0.0 if view.mode == "free" else view.distance
    eye = np.array([view.x, view.y, view.z]) - distance * d
    return Camera(view.name, (float(eye[0]), float(eye[1]), float(eye[2])), (180.0 + math.degrees(r)) % 360.0,
                  90.0 - math.degrees(a), fov, size[0], size[1], f"view {view.name}")


def scene_cameras(name: str) -> list[Camera]:
    """Every oracle camera of a scene (derived views first, then ``oracle.cameras``)."""
    scene = scenes.load(name)
    raw = yaml.safe_load((paths.SCENES_DIR / f"{name}.yaml").read_text(encoding="utf-8")).get("oracle") or {}
    size = tuple(raw.get("size", scene.viewport))
    fov = float(raw.get("fov", DEFAULT_FOV))
    wanted = raw.get("views")
    by_name = {v.name: v for v in scene.views}
    if wanted is not None:
        unknown = set(wanted) - set(by_name)
        if unknown:
            raise ValueError(f"{name}: oracle.views {sorted(unknown)} are not views of the scene")
    cameras = []
    for view in scene.views:
        if wanted is not None and view.name not in wanted:
            continue
        if view.mode == "flat" or view.ortho > 0:
            if wanted is not None:
                raise ValueError(f"{name}: view {view.name} is orthographic; the client has no such camera")
            continue
        cameras.append(from_view(view, fov, size))
    for camera_name, spec in (raw.get("cameras") or {}).items():
        eye = tuple(float(v) for v in spec["eye"])
        if "look_at" in spec:
            yaw, pitch = look_at(eye, tuple(float(v) for v in spec["look_at"]))
        else:
            yaw, pitch = float(spec["yaw"]), float(spec["pitch"])
        cameras.append(Camera(camera_name, eye, yaw, pitch, float(spec.get("fov", fov)), size[0], size[1], "oracle.cameras"))
    names = [c.name for c in cameras]
    if len(set(names)) != len(names):
        raise ValueError(f"{name}: duplicate camera names {names}")
    if not cameras:
        raise ValueError(f"{name}: no oracle cameras (only orthographic views and no oracle.cameras)")
    return cameras


def excluded_states(name: str) -> dict[str, frozenset[int] | None]:
    """``oracle.exclude`` of a scene: block name -> excluded metas (None: every meta)."""
    raw = yaml.safe_load((paths.SCENES_DIR / f"{name}.yaml").read_text(encoding="utf-8")).get("oracle") or {}
    states: dict[str, frozenset[int] | None] = {}
    for block, metas in (raw.get("exclude") or {}).items():
        if metas == "all":
            states[block] = None
        elif isinstance(metas, list) and metas and all(isinstance(m, int) and 0 <= m <= 15 for m in metas):
            states[block] = frozenset(metas)
        else:
            raise ValueError(f"{name}: oracle.exclude[{block!r}] must be `all` or a list of metas 0..15, got {metas!r}")
    return states


def is_excluded(states: dict[str, frozenset[int] | None], block: str, meta: int) -> bool:
    if block not in states:
        return False
    metas = states[block]
    return metas is None or meta in metas


def scene_box(name: str) -> list[int] | None:
    """The scene's dump box ``[x0, y0, z0, x1, y1, z1]`` (inclusive) from ``out/scenes/<scene>/scene.json``."""
    path = paths.workspace() / "out" / "scenes" / name / "scene.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("dump_box")


def cameras_json(name: str, cameras: list[Camera], remove: list[tuple[int, int, int]] = ()) -> dict:
    """The camera list for `gtnh-client cameras` (format 1); ``remove``: cells set to air before the client joins."""
    scene = scenes.load(name)
    return {"format": CAMERAS_FORMAT, "scene": name, "world": scene.world.name, "levelType": "FLAT",
            "box": scene_box(name), "cameras": [c.to_json() for c in cameras], "remove": [list(p) for p in remove]}


@dataclass
class ClientView:
    """One reference capture of the client (`<view>.png`, `<view>.depth.bin`, `<view>.json`)."""

    name: str
    image: np.ndarray  # (H, W, 3) uint8, rows top to bottom
    depth: np.ndarray  # (H, W) float32 OpenGL window depth, 1.0 = nothing drawn
    projection: np.ndarray  # (4, 4)
    modelview: np.ndarray  # (4, 4), camera-relative
    camera: np.ndarray  # (3,) world position the modelview is relative to
    meta: dict

    @property
    def width(self) -> int:
        return self.image.shape[1]

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def view(self) -> np.ndarray:
        """World -> eye."""
        translate = np.eye(4)
        translate[:3, 3] = -self.camera
        return self.modelview @ translate

    @property
    def eye(self) -> np.ndarray:
        """The eye in world coordinates (0.1 behind the player's eye position)."""
        return np.linalg.inv(self.view)[:3, 3]

    @staticmethod
    def load(directory: Path, name: str) -> ClientView:
        meta = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
        from PIL import Image

        with Image.open(directory / f"{name}.png") as im:
            image = np.asarray(im.convert("RGB"))
        h, w = image.shape[:2]
        if (w, h) != (meta["width"], meta["height"]):
            raise ValueError(f"{name}: image {w}x{h} but metadata {meta['width']}x{meta['height']}")
        depth = np.fromfile(directory / f"{name}.depth.bin", dtype="<f4").reshape(h, w)
        return ClientView(name, image, depth, column_major(meta["projection"]), column_major(meta["modelview"]),
                          np.asarray(meta["camera"], dtype=np.float64), meta)


def column_major(values: list[float]) -> np.ndarray:
    """A 4x4 matrix from OpenGL's column-major 16 floats."""
    return np.asarray(values, dtype=np.float64).reshape(4, 4).T


def ndc_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """NDC x and y of the pixel centres, rows top to bottom: (H, W) arrays."""
    x = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    y = 1.0 - (np.arange(height) + 0.5) / height * 2.0
    return np.broadcast_to(x, (height, width)), np.broadcast_to(y[:, None], (height, width))


def eye_from_window_depth(projection: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Eye-space points (H, W, 3) of OpenGL window depths (depth range 0..1)."""
    h, w = depth.shape
    x, y = ndc_grid(w, h)
    clip = np.stack([x, y, 2.0 * depth.astype(np.float64) - 1.0, np.ones_like(x)], axis=-1)
    eye = clip @ np.linalg.inv(projection).T
    return eye[..., :3] / eye[..., 3:4]


def eye_from_linear_depth(projection: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Eye-space points (H, W, 3) at ``distance`` along the view axis (eye z = -distance)."""
    h, w = distance.shape
    x, y = ndc_grid(w, h)
    near = np.stack([x, y, -np.ones_like(x), np.ones_like(x)], axis=-1) @ np.linalg.inv(projection).T
    near = near[..., :3] / near[..., 3:4]
    return near * (distance[..., None] / -near[..., 2:3])


def eye_to_world(view: np.ndarray, eye_points: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(view)
    return eye_points @ inverse[:3, :3].T + inverse[:3, 3]


def project(view: np.ndarray, projection: np.ndarray, width: int, height: int, points: np.ndarray) -> np.ndarray:
    """Pixel coordinates (N, 2) (x right, y down, pixel centres at .5) of world points (N, 3); NaN behind the eye."""
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    clip = homogeneous @ (projection @ view).T
    with np.errstate(divide="ignore", invalid="ignore"):
        ndc = clip[:, :2] / clip[:, 3:4]
    pixels = np.stack([(ndc[:, 0] + 1.0) / 2.0 * width, (1.0 - ndc[:, 1]) / 2.0 * height], axis=1)
    pixels[clip[:, 3] <= 0] = np.nan
    return pixels


def webapp_camera(client: ClientView) -> dict:
    """The three.js camera that reproduces the client's: world position, orientation quaternion (x, y, z, w) and the
    projection matrix (column-major, as three's Matrix4.fromArray reads it)."""
    world = np.linalg.inv(client.view)
    rotation = world[:3, :3]
    projection = client.projection
    fov_y = math.degrees(2.0 * math.atan(1.0 / projection[1, 1]))
    near = float(client.meta.get("near", 0.05))
    far = float(client.meta.get("far", 1000.0))
    return {
        "position": [float(v) for v in world[:3, 3]],
        "quaternion": quaternion(rotation),
        "projection": [float(v) for v in projection.T.reshape(-1)],
        "fov": fov_y,
        "aspect": float(projection[1, 1] / projection[0, 0]),
        "near": near,
        "far": far,
    }


def quaternion(rotation: np.ndarray) -> list[float]:
    """(x, y, z, w) of a rotation matrix (Shepperd's method)."""
    m = rotation
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w, x, y, z = 0.25 / s, (m[2, 1] - m[1, 2]) * s, (m[0, 2] - m[2, 0]) * s, (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    return [x / norm, y / norm, z / norm, w / norm]


def expected_matrices(camera: dict, width: int, height: int, far: float) -> tuple[np.ndarray, np.ndarray]:
    """Projection and modelview that vanilla 1.7.10 sets up for a camera spec (gluPerspective(fov, aspect, 0.05,
    far); translate(0, 0, -0.1), rotate pitch about x, rotate yaw + 180 about y), to cross-check the recorded ones."""
    fov = math.radians(camera["fov"])
    f = 1.0 / math.tan(fov / 2.0)
    near = 0.05
    projection = np.array([
        [f / (width / height), 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0],
    ])
    def rotation(axis: int, degrees: float) -> np.ndarray:
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        m = np.eye(4)
        i, j = [(1, 2), (2, 0), (0, 1)][axis]
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        return m
    step_back = np.eye(4)
    step_back[2, 3] = -0.1
    modelview = step_back @ rotation(0, camera["pitch"]) @ rotation(1, camera["yaw"] + 180.0)
    return projection, modelview
