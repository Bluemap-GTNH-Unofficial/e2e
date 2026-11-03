"""Workspace layout shared by the e2e runner, the CLI and the pytest suite."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]  # src/e2e/paths.py -> src -> e2e repo
SCENES_DIR = PROJECT / "scenes"

DEFAULT_PACK_VERSION = "2.9.0"


def workspace() -> Path:
    """The workspace holding this repo and its siblings (`BlueMap/`, `worlds/`, ...); override with BMGTNH_WORKSPACE."""
    env = os.environ.get("BMGTNH_WORKSPACE")
    return Path(env) if env else PROJECT.parent


@dataclass(frozen=True)
class Pack:
    """The workspace directories of one GTNH pack version: every per-version directory is ``<base><suffix>`` (scene
    outputs and world snapshots as the orchestrator's ``packs.py`` writes them), the suffix ``-<version>``."""

    version: str
    suffix: str
    server_port: int  # ports the oracle runs the pack's scene servers on
    rcon_port: int
    addon_server_port: int
    addon_rcon_port: int

    def dir(self, base: str) -> Path:
        """``<workspace>/<base><suffix>``, e.g. ``out/scenes-2.9.0``."""
        return workspace() / f"{base}{self.suffix}"

    @property
    def client(self) -> Path:
        return workspace() / "clients" / f"gtnh-{self.version}"

    @property
    def server(self) -> str:
        """Directory under ``<workspace>/servers`` with the plain pack."""
        return f"gtnh-{self.version}"

    @property
    def addon_server(self) -> str:
        """Directory under ``<workspace>/servers`` with the pack plus the community addons."""
        return f"gtnh-addons{self.suffix}"

    @property
    def export(self) -> Path:
        """The exporter's resource pack of the plain pack (``gtnh-client export``)."""
        return self.dir("out/exporter") / f"bluemap-gtnh-resources-{self.version}.zip"

    @property
    def addons_export(self) -> Path:
        """The exporter's resource pack of the pack plus the addons (``--pack-suffix addons``)."""
        return self.dir("out/exporter-addons") / f"bluemap-gtnh-resources-{self.version}-addons.zip"


# add a version here (and to the orchestrator's and launcher's tables) to test it
PACKS = {
    "2.9.0": Pack("2.9.0", "-2.9.0", 25569, 25579, 25572, 25582),
}


def pack(version: str | None = None) -> Pack:
    """The layout of ``version``; without one the version the CLI selected (``e2e --pack-version``, environment
    variable BMGTNH_PACK_VERSION), else DEFAULT_PACK_VERSION."""
    version = version or os.environ.get("BMGTNH_PACK_VERSION") or DEFAULT_PACK_VERSION
    if version not in PACKS:
        raise ValueError(f"unknown pack version {version!r}; known: {', '.join(PACKS)}")
    return PACKS[version]


def bluemap_repo() -> Path:
    """The BlueMap fork checkout (sibling `BlueMap/`) whose CLI and addon jars are rendered with."""
    return workspace() / "BlueMap"


def worlds_dir() -> Path:
    """World snapshots of the selected pack version's scenes."""
    return pack().dir("worlds")


def scenes_dir() -> Path:
    """The orchestrator's outputs (dump.ndjson, scene.json, manifests) of the selected pack version's scenes."""
    return pack().dir("out/scenes")


def out_dir() -> Path:
    """Version-independent e2e files: the BlueMap build log and the Minecraft client jar cache."""
    return workspace() / "out" / "e2e"


def scene_out(scene: str) -> Path:
    """`e2e run` outputs of a scene of the selected pack version."""
    return pack().dir("out/e2e") / scene


def download_cache() -> Path:
    """Minecraft client jars downloaded by BlueMap, shared between scene renders."""
    return out_dir() / ".cache"


def baselines_dir() -> Path:
    """Local BlueMap-vs-BlueMap regression baselines of the selected pack version, never committed; override with
    BMGTNH_BASELINES."""
    env = os.environ.get("BMGTNH_BASELINES")
    return Path(env) if env else pack().dir("baselines")


def baseline(scene: str, view: str) -> Path:
    return baselines_dir() / scene / f"{view}.png"
