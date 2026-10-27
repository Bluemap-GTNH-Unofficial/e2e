"""Workspace layout shared by the e2e runner, the CLI and the pytest suite."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # testing/e2e
SCENES_DIR = PROJECT / "scenes"
BASELINES_DIR = PROJECT / "baselines"
REPO = PROJECT.parents[1]  # BlueMap-GTNH


def workspace() -> Path:
    env = os.environ.get("BMGTNH_WORKSPACE")
    # testing/e2e -> testing -> BlueMap-GTNH -> workspace
    return Path(env) if env else REPO.parent


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
