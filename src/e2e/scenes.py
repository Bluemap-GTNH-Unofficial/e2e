"""Scene specs: which world to render and which camera views to check (``scenes/<scene>.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from e2e import paths

MODES = ("perspective", "flat", "free")


@dataclass(frozen=True)
class Tolerance:
    pixel: int = 24  # a pixel differs when any RGB channel differs by more than this
    ratio: float = 0.005  # a view fails when more than this share of its pixels differ


@dataclass(frozen=True)
class View:
    name: str
    map: str
    x: float
    y: float
    z: float
    distance: float
    rotation: float = 0.0
    angle: float = 0.0
    tilt: float = 0.0
    ortho: float = 0.0
    mode: str = "perspective"
    tolerance: Tolerance = field(default_factory=Tolerance)
    min_content: float = 0.05  # share of pixels that must differ from the bare sky
    max_missing: float | None = None  # share of content pixels showing the missing texture; None = report only

    @property
    def hash(self) -> str:
        """The webapp location hash `#<map>:<x>:<y>:<z>:<distance>:<rotation>:<angle>:<tilt>:<ortho>:<mode>`."""
        values = (self.x, self.y, self.z, self.distance, self.rotation, self.angle, self.tilt, self.ortho)
        return ":".join([self.map, *(f"{v:g}" for v in values), self.mode])


@dataclass(frozen=True)
class Scene:
    name: str
    world: Path
    maps: tuple[str, ...] | None  # None: every dimension with region files
    viewport: tuple[int, int]
    views: tuple[View, ...]
    extra_packs: tuple[Path, ...] = ()


def names() -> list[str]:
    return sorted(p.stem for p in paths.SCENES_DIR.glob("*.yaml"))


def _view(name: str, raw: dict, defaults: dict) -> View:
    merged = {**defaults, **raw}
    tolerance = Tolerance(**{**defaults.get("tolerance", {}), **raw.get("tolerance", {})})
    x, y, z = merged.pop("position")
    merged.pop("tolerance", None)
    view = View(name=name, x=x, y=y, z=z, tolerance=tolerance, **merged)
    if view.mode not in MODES:
        raise ValueError(f"view {name}: mode must be one of {MODES}")
    return view


def load(name: str) -> Scene:
    path = paths.SCENES_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"unknown scene {name!r}; known: {', '.join(names())}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    views = tuple(_view(view_name, view, defaults) for view_name, view in raw["views"].items())
    maps = raw.get("maps")
    width, height = raw.get("viewport", (1280, 720))
    packs = tuple(paths.workspace() / p for p in raw.get("extra_packs", []))
    return Scene(
        name=name,
        world=paths.worlds_dir() / raw.get("world", name),
        maps=tuple(maps) if maps is not None else None,
        viewport=(width, height),
        views=views,
        extra_packs=packs,
    )
