"""Building, configuring and running the BlueMap CLI for a scene world."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from e2e import paths

JAVA_TOOL = "java@temurin-25.0.5+8.0.LTS"
MC_VERSION = "1.7.10"
LOADER = "gtnh:anvil_1_7"
PORTS = range(8200, 8210)
WEBAPP_ENTRY = "de/bluecolored/bluemap/webapp.zip"


class BlueMapError(RuntimeError):
    pass


@dataclass(frozen=True)
class Jars:
    cli: Path
    addons: tuple[Path, ...]


@dataclass(frozen=True)
class Dimension:
    map_id: str
    key: str
    folder: Path


def java() -> str:
    env = os.environ.get("BMGTNH_JAVA")
    if env:
        return env
    home = subprocess.run(["mise", "where", JAVA_TOOL], check=True, capture_output=True, text=True).stdout.strip()
    return str(Path(home) / "bin" / "java")


def _repo() -> Path:
    repo = paths.bluemap_repo()
    if not (repo / "settings.gradle.kts").is_file():
        raise BlueMapError(f"BlueMap checkout not found at {repo}; clone it next to e2e or set BMGTNH_WORKSPACE")
    return repo


def _optional_addon_projects(repo: Path) -> list[str]:
    """Main-build projects that produce a BlueMap addon jar next to `:legacy` (present once they exist)."""
    return [name for name in ("gtnh-render",) if (repo / name / "build.gradle.kts").is_file()]


def build(log: Path) -> None:
    """Builds the CLI shadow jar (with the webapp) and every addon jar in one Gradle invocation."""
    repo = _repo()
    tasks = [":cli:shadowJar", ":legacy:addonJar", *(f":{p}:addonJar" for p in _optional_addon_projects(repo))]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as out:
        result = subprocess.run(
            ["mise", "exec", "--", "./gradlew", "--console=plain", *tasks],
            cwd=repo, stdout=out, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        tail = "".join(log.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)[-40:])
        raise BlueMapError(f"gradle {' '.join(tasks)} failed (exit {result.returncode}), log {log}:\n{tail}")


def _newest(directory: Path, pattern: str) -> Path | None:
    candidates = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
    return candidates[-1] if candidates else None


def locate_jars() -> Jars:
    repo = _repo()
    cli = _newest(repo / "implementations" / "cli" / "build" / "libs", "cli-*-shadow.jar")
    if cli is None:
        raise BlueMapError("no CLI shadow jar; run without --no-build or `./gradlew :cli:shadowJar`")
    with zipfile.ZipFile(cli) as jar:
        if WEBAPP_ENTRY not in jar.namelist():
            raise BlueMapError(f"{cli} was built without the webapp (-x :common:buildWebapp); rebuild it")
    legacy = _newest(repo / "legacy" / "build" / "libs", "*-addon.jar")
    if legacy is None:
        raise BlueMapError("no :legacy addon jar; run without --no-build or `./gradlew :legacy:addonJar`")
    # optional addons are used when built (always the case after `build`)
    optional = (_newest(repo / p / "build" / "libs", "*-addon.jar") for p in _optional_addon_projects(repo))
    return Jars(cli, (legacy, *(a for a in optional if a is not None)))


_REGION = re.compile(r"r\.-?\d+\.-?\d+\.mca")


def _has_regions(folder: Path) -> bool:
    region = folder / "region"
    return region.is_dir() and any(_REGION.fullmatch(f.name) and f.stat().st_size > 0 for f in region.iterdir())


def detect_dimensions(world: Path) -> list[Dimension]:
    """Every dimension with at least one non-empty region file (D17), keyed like LegacyDimensions."""
    found: list[tuple[int, Path]] = []
    if _has_regions(world):
        found.append((0, world))
    for folder in world.glob("DIM*"):
        suffix = folder.name[3:]
        if re.fullmatch(r"-?\d+", suffix) and _has_regions(folder):
            found.append((int(suffix), folder))
    dims = []
    for dim_id, folder in sorted(found, key=lambda e: (e[0] != 0, e[0])):
        if dim_id == 0:
            dims.append(Dimension("overworld", "minecraft:overworld", folder))
        elif dim_id == -1:
            dims.append(Dimension("the_nether", "minecraft:the_nether", folder))
        elif dim_id == 1:
            dims.append(Dimension("the_end", "minecraft:the_end", folder))
        else:
            dims.append(Dimension(f"dim_{dim_id}", f"gtnh:dim_{dim_id}", folder))
    return dims


def _hocon(value: object) -> str:
    # JSON is valid HOCON
    return json.dumps(value)


def _map_conf(world: Path, dim: Dimension, sorting: int) -> str:
    settings: dict[str, object] = {
        "world": str(world),
        "loader": LOADER,
        "dimension": dim.key,
        "name": f"{dim.map_id} ({world.name})",
        "sorting": sorting,
        "storage": "file",
    }
    if dim.key == "minecraft:overworld":
        settings |= {"sky-color": "#7dabff", "ambient-light": 0.1, "remove-caves-below-y": 55,
                     "cave-detection-ocean-floor": -5}
    elif dim.key == "minecraft:the_nether":
        settings |= {"sky-color": "#290000", "void-color": "#150000", "sky-light": 0, "ambient-light": 0.6,
                     "remove-caves-below-y": -10000}
    elif dim.key == "minecraft:the_end":
        settings |= {"sky-color": "#080010", "void-color": "#080010", "sky-light": 0, "ambient-light": 0.6,
                     "remove-caves-below-y": -10000}
    else:
        settings |= {"ambient-light": 0.1, "remove-caves-below-y": -10000}
    return "".join(f"{key}: {_hocon(value)}\n" for key, value in settings.items())


def write_config(root: Path, world: Path, dims: list[Dimension], jars: Jars, port: int,
                 extra_packs: list[Path]) -> Path:
    """A fresh CLI config under ``root/config``; data and webroot live next to it in ``root``."""
    config = root / "config"
    packs = config / "packs"
    for sub in (config / "maps", config / "storages", packs):
        sub.mkdir(parents=True, exist_ok=True)
    data = root / "data"
    web = root / "web"
    (config / "core.conf").write_text(
        f"accept-download: true\n"
        f"data: {_hocon(str(data))}\n"
        f"render-thread-count: {max(1, (os.cpu_count() or 2) - 2)}\n"
        f"scan-for-mod-resources: false\n"
        f"metrics: false\n"
        f"log: {{ file: {_hocon(str(data / 'logs' / 'debug.log'))}, append: false }}\n",
        encoding="utf-8",
    )
    (config / "webapp.conf").write_text(
        f"enabled: true\nwebroot: {_hocon(str(web))}\nupdate-settings-file: true\nuse-cookies: false\n",
        encoding="utf-8",
    )
    (config / "webserver.conf").write_text(
        f"enabled: true\nwebroot: {_hocon(str(web))}\nip: \"127.0.0.1\"\nport: {port}\n"
        f"log: {{ file: {_hocon(str(data / 'logs' / 'webserver.log'))}, append: false }}\n",
        encoding="utf-8",
    )
    (config / "storages" / "file.conf").write_text(
        f"storage-type: file\nroot: {_hocon(str(web / 'maps'))}\ncompression: gzip\n", encoding="utf-8"
    )
    for i, dim in enumerate(dims):
        (config / "maps" / f"{dim.map_id}.conf").write_text(_map_conf(world, dim, i), encoding="utf-8")
    for jar in jars.addons:
        shutil.copy2(jar, packs / jar.name)
    for source in extra_packs:
        entries = sorted(source.iterdir()) if source.is_dir() else [source]
        for entry in entries:
            target = packs / entry.name
            if entry.is_dir():
                shutil.copytree(entry, target)
            else:
                shutil.copy2(entry, target)
    return config


def _seed_downloads(data: Path) -> None:
    cache = paths.download_cache()
    data.mkdir(parents=True, exist_ok=True)
    for jar in cache.glob("minecraft-client-*.jar") if cache.is_dir() else []:
        target = data / jar.name
        if not target.exists():
            try:
                os.link(jar, target)
            except OSError:
                shutil.copy2(jar, target)


def _store_downloads(data: Path) -> None:
    cache = paths.download_cache()
    cache.mkdir(parents=True, exist_ok=True)
    for jar in data.glob("minecraft-client-*.jar"):
        target = cache / jar.name
        if not target.exists():
            shutil.copy2(jar, target)


def free_port() -> int:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise BlueMapError(f"no free port in {PORTS.start}..{PORTS.stop - 1}")


class BlueMapCli:
    """One CLI process: force-render every configured map, generate the webapp and serve it (`-r -f -g -w`).

    Rendering and serving share the process like on a server, so the webserver has the loaded maps (live SSE
    endpoint included). The webserver starts while the render runs; the CLI is ready once the render finished and
    saved its state (`RENDER_DONE`, then `RENDER_SAVED`) and the webserver answers.
    """

    RENDER_DONE = "Your maps are now all up-to-date!"
    RENDER_SAVED = " INFO] Stopped."

    def __init__(self, jars: Jars, root: Path, port: int) -> None:
        self.jars = jars
        self.root = root
        self.url = f"http://127.0.0.1:{port}/"
        self.log = root / "bluemap.log"
        self.proc: subprocess.Popen[bytes] | None = None
        self.render_seconds = 0.0

    def start(self, timeout: float = 3600) -> None:
        data = self.root / "data"
        _seed_downloads(data)
        with self.log.open("wb") as log:
            self.proc = subprocess.Popen(
                [java(), "-jar", str(self.jars.cli), "-c", str(self.root / "config"), "-v", MC_VERSION,
                 "-r", "-f", "-g", "-w"],
                cwd=self.root, stdout=log, stderr=subprocess.STDOUT,
            )
        started = time.monotonic()
        try:
            self._wait_serving(started + timeout)
        except BaseException:
            self.stop()
            raise
        self.render_seconds = time.monotonic() - started  # resources + render + save + webserver up
        _store_downloads(data)

    def _rendered(self) -> bool:
        text = self.log.read_text(encoding="utf-8", errors="replace")
        done = text.find(self.RENDER_DONE)
        return done >= 0 and text.find(self.RENDER_SAVED, done) >= 0

    def _serving(self) -> bool:
        try:
            with urllib.request.urlopen(self.url + "settings.json", timeout=2) as response:
                return response.status == 200
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def _wait_serving(self, deadline: float) -> None:
        assert self.proc is not None
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise BlueMapError(f"BlueMap CLI exited with {self.proc.returncode}; log {self.log}")
            if self._rendered() and self._serving():
                return
            # block on the process instead of sleeping blindly: returns at once if it dies
            try:
                self.proc.wait(timeout=0.25)
            except subprocess.TimeoutExpired:
                pass
        raise BlueMapError(f"BlueMap CLI did not finish rendering and serve {self.url} in time; log {self.log}")

    def log_lines(self, level: str) -> list[str]:
        """Log lines of a level (`ERROR`, `WARNING`) written so far."""
        marker = f" {level}]"
        return [line for line in self.log.read_text(encoding="utf-8", errors="replace").splitlines()
                if marker in line]

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()

    def __enter__(self) -> BlueMapCli:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
