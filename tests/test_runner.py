import contextlib
import threading
from types import SimpleNamespace

import pytest

from e2e import runner


@pytest.fixture
def fake_browsers(monkeypatch):
    """No Playwright: every worker thread gets a dummy browser; counts the browsers opened and closed."""
    opened, closed = [], []

    @contextlib.contextmanager
    def sync_playwright():
        yield SimpleNamespace()

    def chromium(_playwright):
        opened.append(threading.get_ident())
        return SimpleNamespace(close=lambda: closed.append(threading.get_ident()))

    monkeypatch.setattr(runner, "sync_playwright", sync_playwright)
    monkeypatch.setattr(runner, "Chromium", chromium)
    return opened, closed


def _specs(names: str) -> list:
    return [SimpleNamespace(name=n) for n in names]


def test_scenes_run_side_by_side_and_come_back_in_order(fake_browsers, monkeypatch):
    both = threading.Barrier(2, timeout=10)  # a and b only pass it when they run at the same time

    def run_scene(spec, jars, chromium, extra):
        if spec.name in "ab":
            both.wait()
        return SimpleNamespace(scene=spec.name)

    monkeypatch.setattr(runner, "run_scene", run_scene)
    done = []
    results = runner.run_scenes(_specs("abcde"), None, (), 2, lambda r: done.append(r.scene))
    assert [r.scene for r in results] == list("abcde") and sorted(done) == list("abcde")
    opened, closed = fake_browsers
    assert len(opened) == 2 and sorted(opened) == sorted(closed)  # one browser per worker thread, all closed


def test_the_first_error_stops_further_scenes_and_is_raised(fake_browsers, monkeypatch):
    ran = []

    def run_scene(spec, jars, chromium, extra):
        ran.append(spec.name)
        if spec.name == "b":
            raise RuntimeError("boom")
        return SimpleNamespace(scene=spec.name)

    monkeypatch.setattr(runner, "run_scene", run_scene)
    with pytest.raises(RuntimeError, match="boom"):
        runner.run_scenes(_specs("abcd"), None, (), 1, lambda r: None)
    assert ran == ["a", "b"]
    opened, closed = fake_browsers
    assert opened == closed
