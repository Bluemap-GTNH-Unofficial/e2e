from __future__ import annotations

import os
from collections.abc import Iterator

import pytest
from playwright.sync_api import sync_playwright

from e2e import bluemap, runner, scenes
from e2e.browser import Chromium


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--scene", action="append", default=[], help="e2e scenes to run (default: all)")
    parser.addoption("--no-build", action="store_true", help="use existing CLI/addon jars instead of running Gradle")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "scene_name" not in metafunc.fixturenames:
        return
    selected = metafunc.config.getoption("--scene") or scenes.names()
    if "view_name" in metafunc.fixturenames:
        params = [(s, v.name) for s in selected for v in scenes.load(s).views]
        metafunc.parametrize(("scene_name", "view_name"), params, ids=[f"{s}/{v}" for s, v in params])
    else:
        metafunc.parametrize("scene_name", selected)


@pytest.fixture(scope="session")
def jars(pytestconfig: pytest.Config) -> bluemap.Jars:
    no_build = pytestconfig.getoption("--no-build") or os.environ.get("E2E_NO_BUILD") == "1"
    return runner.prepare_jars(build=not no_build)


@pytest.fixture(scope="session")
def chromium() -> Iterator[Chromium]:
    with sync_playwright() as playwright:
        browser = Chromium(playwright)
        yield browser
        browser.close()


class SceneResults(dict[str, runner.SceneResult]):
    """Renders and captures each scene once per session, on first access."""

    def __init__(self, jars: bluemap.Jars, chromium: Chromium) -> None:
        super().__init__()
        self.jars = jars
        self.chromium = chromium

    def __missing__(self, name: str) -> runner.SceneResult:
        result = runner.run_scene(scenes.load(name), self.jars, self.chromium)
        self[name] = result
        return result


@pytest.fixture(scope="session")
def scene_results(jars: bluemap.Jars, chromium: Chromium) -> SceneResults:
    return SceneResults(jars, chromium)
