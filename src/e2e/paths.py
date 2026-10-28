"""Workspace layout shared by the e2e runner, the CLI and the pytest suite."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # src/e2e/paths.py -> src -> e2e repo
SCENES_DIR = PROJECT / "scenes"
BASELINES_DIR = PROJECT / "baselines"


def workspace() -> Path:
    """The workspace holding this repo and its siblings (`BlueMap/`, `worlds/`, ...); override with BMGTNH_WORKSPACE."""
    env = os.environ.get("BMGTNH_WORKSPACE")
    return Path(env) if env else PROJECT.parent


def bluemap_repo() -> Path:
    """The BlueMap fork checkout (sibling `BlueMap/`) whose CLI and addon jars are rendered with."""
    return workspace() / "BlueMap"


def worlds_dir() -> Path:
    return workspace() / "worlds"


def out_dir() -> Path:
    return workspace() / "out" / "e2e"


def scene_out(scene: str) -> Path:
    return out_dir() / scene


def download_cache() -> Path:
    """Minecraft client jars downloaded by BlueMap, shared between scene renders."""
    return out_dir() / ".cache"


def baseline(scene: str, view: str) -> Path:
    return BASELINES_DIR / scene / f"{view}.png"
