"""A stand-in for `uv run gtnh-client` in the reference cache tests (``BMGTNH_GTNH_CLIENT``): writes fake captures in
the launcher's output layout and logs every invocation.

``cameras --jobs FILE ...``: per job, ``<camera>.png|.depth.bin|.json`` and ``cameras-report.json`` (with
``capture_version`` $FAKE_CAPTURE_VERSION) in its ``out``; scenes in $FAKE_FAIL_SCENES report a failed camera.
``states --out DIR ...``: ``DIR/index.json`` and ``DIR/../states.txt`` (a copy of $FAKE_STATES_TXT).
"""

import json
import os
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as log:
        log.write(json.dumps(argv) + "\n")
    command = argv[0]
    if command == "cameras":
        jobs = json.loads(Path(argv[argv.index("--jobs") + 1]).read_text())["jobs"]
        failing = set(filter(None, os.environ.get("FAKE_FAIL_SCENES", "").split(",")))
        code = 0
        for job in jobs:
            out = Path(job["out"])
            out.mkdir(parents=True, exist_ok=True)
            cams = json.loads(Path(job["cameras"]).read_text())["cameras"]
            fail = job["scene"] in failing
            done = cams[:-1] if fail else cams
            for cam in done:
                for suffix in (".png", ".depth.bin"):
                    (out / f"{cam['name']}{suffix}").write_bytes(job["scene"].encode())
                (out / f"{cam['name']}.json").write_text(json.dumps({"name": cam["name"], "remove": job["remove"]}))
            (out / "cameras-report.json").write_text(json.dumps({
                "scene": job["scene"], "cameras": len(cams), "captured": len(done), "failed": len(cams) - len(done),
                "capture_version": int(os.environ["FAKE_CAPTURE_VERSION"]), "seconds": 1.5}))
            code = code or int(fail)
        return code
    if command == "states":
        out = Path(argv[argv.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        listed = Path(os.environ["FAKE_STATES_TXT"]).read_text()
        (out.parent / "states.txt").write_text(listed)
        (out / "index.json").write_text(json.dumps({"states": [{"key": line} for line in listed.splitlines()]}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
