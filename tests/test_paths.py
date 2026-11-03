from pathlib import Path

import pytest

from e2e import paths, scenes
from e2e.oracle import variants


@pytest.fixture
def workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.delenv("BMGTNH_BASELINES", raising=False)
    monkeypatch.delenv("BMGTNH_PACK_VERSION", raising=False)
    monkeypatch.setenv("BMGTNH_WORKSPACE", str(tmp_path))
    return tmp_path


def test_baselines_live_in_the_workspace(workspace: Path) -> None:
    assert paths.baseline("gt-pipes", "top") == workspace / "baselines-2.9.0" / "gt-pipes" / "top.png"


def test_baselines_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BMGTNH_BASELINES", str(tmp_path / "b"))
    assert paths.baseline("gt-pipes", "top") == tmp_path / "b" / "gt-pipes" / "top.png"


def test_workspace_layout(workspace: Path) -> None:
    assert paths.pack().version == paths.DEFAULT_PACK_VERSION == "2.9.0"
    assert (paths.worlds_dir(), paths.scenes_dir(), paths.scene_out("gtnh-zoo")) == (
        workspace / "worlds-2.9.0", workspace / "out" / "scenes-2.9.0", workspace / "out" / "e2e-2.9.0" / "gtnh-zoo")
    assert paths.pack().export == workspace / "out" / "exporter-2.9.0" / "bluemap-gtnh-resources-2.9.0.zip"
    assert paths.download_cache() == workspace / "out" / "e2e" / ".cache"
    assert scenes.load("gt-multiblocks").extra_packs == (paths.pack().export,)
    assert scenes.load("gt-multiblocks").world == workspace / "worlds-2.9.0" / "gt-multiblocks"
    assert scenes.pack_path("out/other.zip") == workspace / "out" / "other.zip"


def test_variants(workspace: Path) -> None:
    default = variants.load("default")
    assert (default.client_dir, default.server, default.server_port, default.rcon_port) == (
        workspace / "clients" / "gtnh-2.9.0", "gtnh-2.9.0", 25569, 25579)
    assert variants.oracle_dir(default) == workspace / "out" / "oracle-2.9.0" / "default"
    assert variants.load("modernity").client_dir == workspace / "clients" / "gtnh-modernity-2.9.0"
    addons = variants.load("addons")
    assert (addons.client_dir, addons.server, addons.server_port) == (
        workspace / "clients" / "gtnh-addons-2.9.0", "gtnh-addons-2.9.0", 25572)
