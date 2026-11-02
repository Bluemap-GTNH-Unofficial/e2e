from pathlib import Path

import pytest

from e2e import paths


def test_baselines_live_in_the_workspace(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("BMGTNH_BASELINES", raising=False)
    monkeypatch.setenv("BMGTNH_WORKSPACE", str(tmp_path))
    assert paths.baseline("gt-pipes", "top") == tmp_path / "baselines" / "gt-pipes" / "top.png"


def test_baselines_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("BMGTNH_BASELINES", str(tmp_path / "b"))
    assert paths.baseline("gt-pipes", "top") == tmp_path / "b" / "gt-pipes" / "top.png"
