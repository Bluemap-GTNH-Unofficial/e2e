"""`e2e` command line: run scenes, approve baselines, list scenes and views."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from e2e import paths, runner, scenes
from e2e.browser import Chromium


def _print_scene(result: runner.SceneResult) -> None:
    print(f"\n== {result.scene}: maps {','.join(result.maps)} | render {result.render_seconds:.1f}s | "
          f"total {result.total_seconds:.1f}s | {result.out}")
    for failure in result.failures:
        print(f"   FAIL {failure}")
    print(f"   {'view':<22}{'content':>9}{'magenta':>9}{'missing':>9}{'miss/cont':>10}{'diff':>9}"
          f"{'load':>7}{'stable':>7}  status")
    for v in result.views:
        m = v.metrics
        diff = "-" if v.diff_ratio is None else f"{v.diff_ratio:.3%}"
        status = "ok" if not v.failures else "FAIL"
        print(f"   {v.view:<22}{m.content_ratio:>9.2%}{m.magenta_ratio:>9.2%}{m.missing_ratio:>9.2%}"
              f"{m.missing_of_content:>10.2%}{diff:>9}{v.load_seconds:>6.1f}s{v.stable_seconds:>6.1f}s  {status}")
        for failure in v.failures:
            print(f"      - {failure}")
        print(f"      shot {v.shot}")


def cmd_run(args: argparse.Namespace) -> int:
    names = args.scenes or scenes.names()
    specs = [scenes.load(name) for name in names]
    jars = runner.prepare_jars(build=not args.no_build)
    print(f"cli {jars.cli}\naddons {', '.join(str(a) for a in jars.addons)}")
    extra = tuple(Path(p).resolve() for p in args.packs)
    results = []
    with sync_playwright() as playwright:
        chromium = Chromium(playwright)
        try:
            for spec in specs:
                result = runner.run_scene(spec, jars, chromium, extra)
                _print_scene(result)
                results.append(result)
        finally:
            chromium.close()
    failed = [r.scene for r in results if not r.ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenes passed" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


def cmd_approve(args: argparse.Namespace) -> int:
    for target in runner.approve(scenes.load(args.scene), args.views):
        print(f"approved {target}")
    return 0


def cmd_list(_: argparse.Namespace) -> int:
    for name in scenes.names():
        spec = scenes.load(name)
        print(f"{name}  (world {spec.world})")
        for view in spec.views:
            print(f"   {view.name:<22}#{view.hash}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e2e", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="render scenes, capture views, compare with baselines")
    run.add_argument("scenes", nargs="*", help=f"scenes (default: all in {paths.SCENES_DIR})")
    run.add_argument("--no-build", action="store_true", help="use the existing CLI/addon jars instead of running Gradle")
    run.add_argument("--packs", action="append", default=[], metavar="PATH",
                     help="extra resource pack file or directory of packs to copy into the CLI packs/ folder")
    run.set_defaults(func=cmd_run)

    approve = sub.add_parser("approve", help="copy the latest screenshots of a scene to its baselines")
    approve.add_argument("scene")
    approve.add_argument("views", nargs="*", help="views to approve (default: all)")
    approve.set_defaults(func=cmd_approve)

    sub.add_parser("list", help="list scenes and their view URLs").set_defaults(func=cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
