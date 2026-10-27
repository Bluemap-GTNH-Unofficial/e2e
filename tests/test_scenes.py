"""Browser end-to-end checks: one test per scene view (scenes/*.yaml), plus each scene's BlueMap log."""

import pytest

pytestmark = pytest.mark.e2e


def test_bluemap_log_has_no_errors(scene_results, scene_name: str) -> None:
    scene = scene_results[scene_name]
    assert not scene.failures, f"{scene_name}: " + "\n".join(scene.failures)


def test_view(scene_results, scene_name: str, view_name: str) -> None:
    scene = scene_results[scene_name]
    view = next(v for v in scene.views if v.view == view_name)
    m = view.metrics
    summary = (f"content {m.content_ratio:.2%}, missing {m.missing_of_content:.2%} of content, "
               f"diff {view.diff_ratio}, shot {view.shot}")
    assert not view.failures, f"{scene_name}/{view_name} ({summary}):\n" + "\n".join(view.failures)
