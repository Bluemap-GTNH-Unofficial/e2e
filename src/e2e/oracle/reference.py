"""Content-addressed cache of the real client's reference captures (world oracle and state oracle).

A reference capture depends only on what the client sees, so it is captured once per combination of inputs and
reused until one of them changes. The cache key is the SHA-256 of a canonical JSON of:

- the pack version and the launcher's capture-settings version (``CAPTURE_VERSION`` in the launcher's
  ``gtnh_client/cameras.py``, ``STATES_CAPTURE_VERSION`` in ``gtnh_client/states.py``; bumped there whenever the
  capture conditions, the exporter mod's capture code or anything else that changes the client's pixels changes);
- the client's mods (``.minecraft/mods/*.{jar,zip}`` and ``mods/1.7.10/*``, file name and SHA-256, without the
  exporter jar, which the capture version covers) and the variant's extra mods and resource packs (name and SHA-256);
- world references: the harness mod jar the launcher loads into the client (``bmh-*.jar`` of the pack's reference
  server), the scene's content (``dump.ndjson``: every non-air block's position, name, meta and tile entity in
  canonical form; not file times, numeric ids or line order) and ``cameras.json`` (cameras, dump box, ``remove``);
- state references: the state list the launcher draws (the flattening table's vanilla states, then the exported
  pack's ``gtnh/blocks.ndjson`` states).

Layout (local, never committed; ``out/oracle-ref/<pack version>/``)::

    <variant>/<scene>/<key>/      <view>.png, .depth.bin, .json, cameras-report.json, key.json
    <variant>/states/<key>/       client/ (index.json, images), states.txt, key.json
    calibration/{world,states}-<key>.json   thresholds, keyed by the reference keys they were computed from
    logs/                         launcher logs
    hashes.json                   file hash memo (path, size, mtime -> SHA-256), only to avoid re-reading jars

An entry appears atomically (staged in ``<key>.partial/`` and renamed when the launcher reports every camera / state
captured); its files are read-only. The oracle's output directories link to it (``worlds/<scene>/client`` and
``states/client`` are symlinks) and record the key in ``reference.json``. Exception: when the cache is on a file system
shared with the host (the dev container's bind mount of the checkout, ``HOST_SHARED``) and the oracle output is not,
``states/client`` is a local copy of the state reference instead (``copy_reference``: one copy per capture; its ~125k
images are read by every state oracle run, and over the bind mount each file costs milliseconds), and the state
entry's files are left writable (nothing links to it; clearing the write bits of 125k files there took 13 minutes).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from e2e import paths
from e2e.oracle import variants

KEY_FORMAT = 1
CALIBRATION_VERSION = 1  # bump when a calibration method (world.calibrate, states.calibrate, metrics.calibrate) changes
KEY_CHARS = 16  # hex digits of the SHA-256 used in directory names
EXPORTER_MOD = re.compile(r"bluemap-gtnh-exporter.*\.jar")
MOD_SUFFIXES = (".jar", ".zip")
LAUNCHER_ENV = "BMGTNH_GTNH_CLIENT"  # command that replaces `uv run gtnh-client` (tests, other checkouts)
FLATTENING_TSV = "legacy/src/main/resources/de/bluecolored/bluemap/gtnh/legacy/flattening_1_7_10.tsv"
BLOCKS_NDJSON = "gtnh/blocks.ndjson"
CAMERA_FILES = (".png", ".depth.bin", ".json")
REFERENCE = "reference.json"  # in the oracle output directory next to the client link
ECHO = (b"[cameras]", b"[states]", b"[run]", b"[lock]", b"[jobs]", b"[worker", b"[bridge]")
# File system types reached through the host: the dev container's bind mount of the checkout (virtiofs from Windows),
# Docker Desktop's file sharing, network shares. Every open, stat or chmod there is a round trip to the host.
HOST_SHARED = frozenset({"virtiofs", "9p", "drvfs", "fakeowner", "fuse.grpcfuse", "osxfs", "nfs", "nfs4", "cifs", "smb3"})
COPY_MARKER = ".reference.json"  # in a local copy of a cache entry: the key and the promotion of the entry it copies
COPY_THREADS = 16
MOUNTS = Path("/proc/mounts")


class ReferenceError(RuntimeError):
    pass


def root(pack: paths.Pack | None = None) -> Path:
    pack = pack or paths.pack()
    return paths.workspace() / "out" / "oracle-ref" / pack.version


def launcher_dir() -> Path:
    return paths.workspace() / "gtnh-exporter" / "launcher"


# ---------------------------------------------------------------- hashing


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


class Hasher:
    """SHA-256 of files and scene dumps, memoised by (path, size, mtime) in ``hashes.json`` so that unchanged jars and
    dumps are not read again; the hashes themselves only depend on content."""

    def __init__(self, memo: Path | None) -> None:
        self.memo = memo
        self.entries: dict[str, dict] = {}
        self.dirty = False
        if memo is not None and memo.is_file():
            try:
                self.entries = json.loads(memo.read_text(encoding="utf-8"))
            except ValueError:
                self.entries = {}

    def _memoised(self, kind: str, path: Path, compute) -> str:
        path = path.resolve()
        st = path.stat()
        tag = f"{kind}:{path}"
        stamp = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        entry = self.entries.get(tag)
        if entry and all(entry.get(k) == v for k, v in stamp.items()):
            return entry["sha256"]
        digest = compute(path)
        self.entries[tag] = {**stamp, "sha256": digest}
        self.dirty = True
        return digest

    def file(self, path: Path) -> str:
        def compute(p: Path) -> str:
            h = hashlib.sha256()
            with p.open("rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
            return h.hexdigest()

        return self._memoised("file", path, compute)

    def dump(self, path: Path) -> str:
        return self._memoised("dump", path, dump_digest)

    def save(self) -> None:
        if self.memo is None or not self.dirty:
            return
        self.memo.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.memo.with_name(f"{self.memo.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.entries, indent=0, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, self.memo)
        self.dirty = False


def dump_digest(path: Path) -> str:
    """SHA-256 of a scene dump's content: every non-air block as (x, y, z, name, meta, tile entity), sorted by position,
    tile entities as canonical JSON. Line order, numeric block ids and file times do not count."""
    cells = []
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if not line.strip():
                continue
            e = json.loads(line)
            if e["name"] == "minecraft:air":
                continue
            cells.append((e["x"], e["y"], e["z"], e["name"], e["meta"], _canonical(e.get("te")).decode("utf-8")))
    cells.sort()
    h = hashlib.sha256()
    for cell in cells:
        h.update(_canonical(list(cell)) + b"\n")
    return h.hexdigest()


def _constant(source: Path, name: str) -> int:
    """A module-level ``NAME = <int>`` of a launcher source file (read, not imported: another environment)."""
    if not source.is_file():
        raise ReferenceError(f"{source} does not exist; the gtnh-exporter checkout is expected next to e2e/")
    for node in ast.parse(source.read_text(encoding="utf-8")).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets) and node.value is not None:
            value = ast.literal_eval(node.value)
            if isinstance(value, int):
                return value
    raise ReferenceError(f"{source} has no integer {name}; update the gtnh-exporter checkout")


def capture_version() -> int:
    return _constant(launcher_dir() / "src" / "gtnh_client" / "cameras.py", "CAPTURE_VERSION")


def states_capture_version() -> int:
    return _constant(launcher_dir() / "src" / "gtnh_client" / "states.py", "STATES_CAPTURE_VERSION")


# ---------------------------------------------------------------- keys


@dataclass(frozen=True)
class Key:
    digest: str  # full SHA-256 of the canonical inputs
    inputs: dict

    @property
    def short(self) -> str:
        return self.digest[:KEY_CHARS]


def make_key(inputs: dict) -> Key:
    return Key(hashlib.sha256(_canonical(inputs)).hexdigest(), inputs)


def bmh_jar(pack: paths.Pack) -> Path:
    """The harness mod the launcher loads into the client for the capture (``bmh random-ticks``, ``bmh set``, frozen
    tile entities): the newest ``bmh-*.jar`` of the pack's reference server (``servers/gtnh-<version>/mods``), as the
    launcher picks it for every variant."""
    mods = paths.workspace() / "servers" / pack.server / "mods"
    found = sorted(mods.glob("bmh-*.jar"), key=lambda p: p.stat().st_mtime) if mods.is_dir() else []
    if not found:
        raise ReferenceError(f"no bmh-*.jar in {mods} (the harness mod the reference capture loads)")
    return found[-1]


def client_mods(client_dir: Path, hasher: Hasher) -> list[list[str]]:
    """``[name, sha256]`` of every mod Forge loads from the client directory: ``mods/*.{jar,zip}`` and
    ``mods/1.7.10/*.{jar,zip}`` without the exporter jar (the launcher installs its own, covered by the capture
    version), sorted by name."""
    mods = client_dir / ".minecraft" / "mods"
    if not mods.is_dir():
        raise ReferenceError(f"{mods} does not exist; prepare the client first (gtnh-client prepare)")
    found = []
    for folder, prefix in ((mods, ""), (mods / "1.7.10", "1.7.10/")):
        if not folder.is_dir():
            continue
        for p in folder.iterdir():
            if p.is_file() and p.suffix in MOD_SUFFIXES and not EXPORTER_MOD.fullmatch(p.name):
                found.append([prefix + p.name, hasher.file(p)])
    return sorted(found)


def _files(files, hasher: Hasher) -> list[list[str]]:
    return [[Path(p).name, hasher.file(Path(p))] for p in files]


@dataclass
class Inputs:
    """Key inputs shared by the references of one run (client mods are hashed once per variant)."""

    hasher: Hasher
    _clients: dict[str, dict] = field(default_factory=dict)

    @staticmethod
    def open(pack: paths.Pack | None = None) -> Inputs:
        return Inputs(Hasher(root(pack) / "hashes.json"))

    def client(self, variant: variants.Variant) -> dict:
        """The client part of a key: mods, extra mods and resource packs (order kept: first = highest priority)."""
        if variant.name not in self._clients:
            self._clients[variant.name] = {
                "client_mods": client_mods(variant.client_dir, self.hasher),
                "extra_mods": _files(variant.extra_mods, self.hasher),
                "resource_packs": _files(variant.resource_packs, self.hasher),
            }
        return self._clients[variant.name]

    def world(self, variant: variants.Variant, scene: str, cameras: dict) -> Key:
        dump = paths.scenes_dir() / scene / "dump.ndjson"
        if not dump.is_file():
            raise ReferenceError(f"{dump} does not exist; build the scene with "
                                 f"`uv run orch --pack-version {variant.pack.version} scene {scene}`")
        return make_key({"kind": "world", "format": KEY_FORMAT, "pack": variant.pack.version,
                         "capture_version": capture_version(), **self.client(variant),
                         "harness": _files([bmh_jar(variant.pack)], self.hasher),
                         "scene": {"name": scene, "dump": self.hasher.dump(dump)}, "cameras": cameras,
                         "freeze_random_ticks": True})

    def states(self, variant: variants.Variant) -> Key:
        listed = state_list(variant)
        return make_key({"kind": "states", "format": KEY_FORMAT, "pack": variant.pack.version,
                         "capture_version": states_capture_version(), **self.client(variant),
                         "states": {"count": listed.count("\n"),
                                    "sha256": hashlib.sha256(listed.encode("utf-8")).hexdigest()}})

    def save(self) -> None:
        self.hasher.save()


def state_list(variant: variants.Variant) -> str:
    """The ``<modid:name> <meta>`` lines ``gtnh-client states`` draws for the variant (``states.txt``): the vanilla
    states Mojang's 1.13 flattening table knows (``registered`` 1, air excluded), then every state of the exported
    pack's ``gtnh/blocks.ndjson``, first occurrence kept. Checked against the launcher's ``states.txt`` after a capture."""
    tsv = paths.bluemap_repo() / FLATTENING_TSV
    if not tsv.is_file():
        raise ReferenceError(f"{tsv} does not exist (BlueMap fork checkout)")
    if not variant.states_from.is_file():
        raise ReferenceError(f"{variant.states_from} does not exist; export the pack first (gtnh-client export)")
    states: list[tuple[str, int]] = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if line.strip():
            block_id, name, meta, _state, registered = line.split("\t")
            if registered == "1" and block_id != "0":
                states.append((name, int(meta)))
    with zipfile.ZipFile(variant.states_from) as z:
        try:
            blocks = z.read(BLOCKS_NDJSON).decode("utf-8").splitlines()
        except KeyError:
            raise ReferenceError(f"{variant.states_from} has no {BLOCKS_NDJSON}") from None
    for line in blocks:
        if line.strip():
            block = json.loads(line)
            states.extend((block["name"], int(m["meta"])) for m in block.get("metas", []))
    seen: set[tuple[str, int]] = set()
    lines = []
    for state in states:
        if state not in seen:
            seen.add(state)
            lines.append(f"{state[0]} {state[1]}\n")
    return "".join(lines)


def explain(key: Key, entries: Path) -> str:
    """Which inputs differ from the newest cached entry of the same reference (for the log of a cache miss)."""
    previous = sorted(entries.glob("*/key.json"), key=lambda p: p.stat().st_mtime)
    if not previous:
        return "no earlier capture"
    old = json.loads(previous[-1].read_text(encoding="utf-8"))["inputs"]
    changed = []
    for name in sorted(set(old) | set(key.inputs)):
        a, b = old.get(name), key.inputs.get(name)
        if a == b:
            continue
        if name in ("client_mods", "extra_mods", "resource_packs") and isinstance(a, list) and isinstance(b, list):
            before, after = {n: h for n, h in a}, {n: h for n, h in b}
            diff = sorted(n for n in set(before) | set(after) if before.get(n) != after.get(n))
            changed.append(f"{name} ({len(diff)}: {', '.join(diff[:3])}{', ...' if len(diff) > 3 else ''})")
        elif name == "cameras" and isinstance(a, dict) and isinstance(b, dict):
            changed.append("cameras (" + ", ".join(k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)) + ")")
        else:
            changed.append(name)
    return "changed since the newest capture: " + ", ".join(changed)


# ---------------------------------------------------------------- cache entries


def world_entries(variant: variants.Variant, scene: str) -> Path:
    return root(variant.pack) / variant.name / scene


def states_entries(variant: variants.Variant) -> Path:
    return root(variant.pack) / variant.name / "states"


def staging(entry: Path) -> Path:
    return entry.with_name(entry.name + ".partial")


def _read_only(directory: Path) -> int:
    """Clears the write bits of every file below ``directory``; returns the number of files."""
    count = 0
    for dirpath, _, files in os.walk(directory):
        for name in files:
            p = Path(dirpath) / name
            p.chmod(p.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
            count += 1
    return count


def promote(stage: Path, entry: Path, key: Key, extra: dict, *, read_only: bool = True) -> None:
    """Turns a complete staging directory into the cache entry ``entry`` (replacing an earlier one atomically), its
    files read-only unless ``read_only`` is False."""
    started = time.monotonic()
    (stage / "key.json").write_text(json.dumps({"key": key.digest, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                **extra, "inputs": key.inputs}, indent=1) + "\n", encoding="utf-8")
    files = _read_only(stage) if read_only else 0
    t_read_only = time.monotonic()
    old = None
    if entry.exists():
        old = entry.with_name(f"{entry.name}.old-{os.getpid()}")
        os.replace(entry, old)
    os.replace(stage, entry)
    if old is not None:
        shutil.rmtree(old)
    print(f"[oracle] cache entry {entry}: " + (f"{files} files made read-only in {t_read_only - started:.1f}s, "
                                               if read_only else "") +
          f"promoted in {time.monotonic() - t_read_only:.1f}s", flush=True)


def _mounts_above(path: Path) -> list[tuple[str, str]]:
    """(mount point, file system type) of every mount ``path`` (existing or not) is on or below, from ``/proc/mounts``
    (none where there is no such file), innermost last."""
    try:
        lines = MOUNTS.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    target = str(path.resolve())
    mounts = []
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        mount = fields[1].replace("\\040", " ")
        if target == mount or target.startswith(mount.rstrip("/") + "/"):
            mounts.append((mount, fields[2]))
    return sorted(mounts, key=lambda m: len(m[0]))


def file_system(path: Path) -> str | None:
    """The type of the file system ``path`` (existing or not) is on (None where ``/proc/mounts`` is missing)."""
    mounts = _mounts_above(path)
    return mounts[-1][1] if mounts else None


def host_shared(path: Path) -> bool:
    return file_system(path) in HOST_SHARED


def crosses_host_shared(path: Path) -> bool:
    """Whether looking ``path`` up passes a file system shared with the host: it is on one, or on a volume mounted
    below one (``out/oracle-2.9.0`` below the dev container's ``/ws``), whose components above it each cost a round
    trip to the host on every lookup of an absolute path."""
    return any(kind in HOST_SHARED for _, kind in _mounts_above(path))


def _copy_files(source: int, target: int, names: list[str]) -> None:
    """Copies the files ``names`` from the directory ``source`` to the directory ``target`` (both file descriptors)."""
    for name in names:
        src = os.open(name, os.O_RDONLY, dir_fd=source)
        try:
            dst = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644, dir_fd=target)
            try:
                while chunk := os.read(src, 1 << 20):
                    os.write(dst, chunk)
            finally:
                os.close(dst)
        finally:
            os.close(src)


def _copy_directory(source_root: int, target_root: int, relative: str, names: list[str]) -> None:
    flags = os.O_RDONLY | os.O_DIRECTORY
    source = os.open(relative, flags, dir_fd=source_root)
    try:
        target = os.open(relative, flags, dir_fd=target_root)
        try:
            _copy_files(source, target, names)
        finally:
            os.close(target)
    finally:
        os.close(source)


def copy_reference(target: Path, source: Path, stamp: dict, threads: int = COPY_THREADS) -> tuple[int, float] | None:
    """Makes ``target`` a local copy of the directory ``source`` unless it already is a copy made with ``stamp``
    (recorded in ``COPY_MARKER``; ``use_states_reference``: the entry's key and when it was promoted, so an entry
    captured again under the same key is copied again); replaces an earlier link or copy. Returns (files, seconds) of a
    copy made, None when the copy was current.

    Every path component the kernel looks up on a host-shared file system is a round trip to the host (about 0.25 ms
    in the dev container, no caching), so files are opened relative to directory descriptors (one lookup per file
    instead of one per component of its absolute path) by ``threads`` threads, one directory at a time each."""
    marker = target / COPY_MARKER
    if not target.is_symlink() and marker.is_file():
        try:
            recorded = json.loads(marker.read_text(encoding="utf-8"))
            if {name: recorded.get(name) for name in stamp} == stamp:
                return None
        except ValueError:
            pass
    started = time.monotonic()
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.copy")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()
    flags = os.O_RDONLY | os.O_DIRECTORY
    target_root = os.open(tmp, flags)
    try:
        directories: list[tuple[str, list[str]]] = []
        for dirpath, _, names, _ in os.fwalk(source):
            relative = os.path.relpath(dirpath, source)
            if relative != ".":
                os.mkdir(relative, dir_fd=target_root)
            directories.append((relative, names))
        source_root = os.open(source, flags)
        try:
            with ThreadPoolExecutor(threads) as pool:
                for future in [pool.submit(_copy_directory, source_root, target_root, relative, names)
                               for relative, names in directories]:
                    future.result()
        finally:
            os.close(source_root)
    finally:
        os.close(target_root)
    files = sum(len(names) for _, names in directories)
    (tmp / COPY_MARKER).write_text(json.dumps({**stamp, "source": str(source), "files": files}) + "\n",
                                   encoding="utf-8")
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    os.replace(tmp, target)
    return files, time.monotonic() - started


def copies_states(out: Path, entry: Path) -> bool:
    """Whether the state oracle output ``out`` gets a local copy of the cache entry ``entry`` instead of a link: the
    cache is on a file system shared with the host and ``out`` is not (see the module doc)."""
    return host_shared(entry) and not host_shared(out)


def use_states_reference(out: Path, entry: Path, key: str) -> None:
    """Makes ``out/client`` the state reference of ``entry``: a link, or a local copy (``copies_states``)."""
    if not copies_states(out, entry):
        link(out / "client", entry / "client")
        return
    try:  # promote writes key.json once per capture
        promoted = (entry / "key.json").stat().st_mtime_ns
    except OSError:
        promoted = None
    copied = copy_reference(out / "client", entry / "client", {"key": key, "promoted": promoted})
    if copied is None:
        print(f"[states] local copy of the reference {key[:KEY_CHARS]} is current ({out / 'client'})", flush=True)
    else:
        print(f"[states] reference copied from the host-shared cache: {copied[0]} files in {copied[1]:.0f}s "
              f"({out / 'client'})", flush=True)


def link(target: Path, entry: Path) -> None:
    """Points ``target`` (an oracle output directory such as ``worlds/<scene>/client``) at ``entry``, replacing what
    was there (an earlier link, or a directory of a capture made before the cache)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.is_file():
        target.unlink()
    elif target.is_dir():
        shutil.rmtree(target)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.link")
    tmp.unlink(missing_ok=True)
    tmp.symlink_to(entry.resolve(), target_is_directory=True)
    os.replace(tmp, target)


def record(out: Path, data: dict) -> None:
    (out / REFERENCE).write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def recorded_key(out: Path) -> str | None:
    """The reference key an oracle output directory was last linked to (``reference.json``), if any."""
    path = out / REFERENCE
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("key")


# ---------------------------------------------------------------- launcher


@dataclass
class LaunchOptions:
    mod_jar: Path | None = None
    parallel: int | None = None


def launcher_command() -> tuple[list[str], Path]:
    custom = os.environ.get(LAUNCHER_ENV)
    if custom:
        return shlex.split(custom), Path.cwd()
    return ["uv", "run", "gtnh-client"], launcher_dir()


def client_flags(variant: variants.Variant, options: LaunchOptions) -> list[str]:
    flags = ["--pack-version", variant.pack.version, "--client-dir", str(variant.client_dir)]
    flags += [f"--resource-pack={p}" for p in variant.resource_packs]
    flags += [f"--extra-mod={m}" for m in variant.extra_mods]
    if options.mod_jar:  # the launcher runs in its own checkout: resolve here
        flags += ["--mod-jar", str(options.mod_jar.resolve())]
    if options.parallel:
        flags += ["--parallel", str(options.parallel)]
    return flags


def run_launcher(args: list[str], log: Path) -> tuple[int, float]:
    """Runs ``gtnh-client <args>``; the output goes to ``log``, progress lines also to stdout. Returns the exit code
    and the wall time."""
    base, cwd = launcher_command()
    cmd = [*base, *args]
    print(f"[oracle] {subprocess.list2cmdline(cmd)} (log {log})", flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("wb") as out:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert proc.stdout is not None
        for line in proc.stdout:
            out.write(line)
            if line.lstrip().startswith(ECHO) or b"Camera " in line:
                sys.stdout.buffer.write(line)
                sys.stdout.flush()
        code = proc.wait()
    return code, time.monotonic() - started


def log_file(kind: str, variant: variants.Variant) -> Path:
    return root(variant.pack) / "logs" / f"{kind}-{variant.name}-{time.strftime('%Y%m%d-%H%M%S')}.log"


# ---------------------------------------------------------------- world references


@dataclass
class WorldJob:
    """One scene of one variant: its cameras and reference key; after ``capture_world``, where its captures are."""

    variant: variants.Variant
    scene: str
    world: Path  # the scene's world snapshot
    cameras_file: Path
    cameras: dict  # cameras.json content
    key: Key
    cached: bool = False  # the captures come from the cache (no client this run)
    captures: Path | None = None  # cache entry, or the staging directory of a partly failed capture
    complete: bool = False  # every camera captured
    seconds: float | None = None  # client time of this job when captured in this run
    log: Path | None = None
    note: str = ""

    @property
    def entry(self) -> Path:
        return world_entries(self.variant, self.scene) / self.key.short


def world_complete(directory: Path, cameras: dict) -> bool:
    return all((directory / f"{c['name']}{s}").is_file() for c in cameras["cameras"] for s in CAMERA_FILES)


def _job_status(stage: Path, job: WorldJob, version: int) -> tuple[bool, str]:
    """Whether a job's staging directory is a complete capture, and why not."""
    report = stage / "cameras-report.json"
    if not report.is_file():
        return False, "no cameras-report.json"
    data = json.loads(report.read_text(encoding="utf-8"))
    if data.get("capture_version") != version:
        return False, f"capture_version {data.get('capture_version')!r} in the report, expected {version}"
    if data.get("failed", 1) != 0 or data.get("error"):
        return False, f"{data.get('failed')} cameras failed{': ' + str(data['error']) if data.get('error') else ''}"
    if not world_complete(stage, job.cameras):
        return False, "camera files missing"
    return True, ""


def capture_world(jobs: list[WorldJob], *, recapture: bool, capture: bool, options: LaunchOptions) -> None:
    """Resolves every job's captures: cache hits are used as they are; the misses of each variant are captured with
    one ``gtnh-client cameras --jobs`` (variants one after another: one client set at a time). With ``capture``
    False nothing is launched and misses keep ``captures`` None."""
    misses: dict[str, list[WorldJob]] = {}
    for job in jobs:
        if not recapture and job.entry.is_dir() and world_complete(job.entry, job.cameras):
            job.cached, job.captures, job.complete = True, job.entry, True
            continue
        job.note = "recapture requested" if recapture else explain(job.key, world_entries(job.variant, job.scene))
        print(f"[oracle] {job.variant.name}/{job.scene}: reference {job.key.short} not cached ({job.note})", flush=True)
        misses.setdefault(job.variant.name, []).append(job)
    if not capture:
        return
    version = capture_version()
    for group in misses.values():
        variant = group[0].variant
        spec = []
        for job in group:
            stage = staging(job.entry)
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True)
            spec.append({"scene": job.scene, "world": str(job.world.resolve()),
                         "cameras": str(job.cameras_file.resolve()), "out": str(stage.resolve()),
                         "remove": job.cameras["remove"], "freeze_random_ticks": True})
        log = log_file("cameras", variant)
        jobs_file = log.with_suffix(".jobs.json")
        jobs_file.parent.mkdir(parents=True, exist_ok=True)
        jobs_file.write_text(json.dumps({"jobs": spec}, indent=1) + "\n", encoding="utf-8")
        code, seconds = run_launcher(["cameras", "--jobs", str(jobs_file), *client_flags(variant, options)], log)
        print(f"[oracle] {variant.name}: {len(group)} reference capture(s) in {seconds:.0f}s (exit {code})", flush=True)
        for job in group:
            stage = staging(job.entry)
            job.log = log
            ok, why = _job_status(stage, job, version)
            report = stage / "cameras-report.json"
            if report.is_file():
                job.seconds = json.loads(report.read_text(encoding="utf-8")).get("seconds")
            if ok:
                promote(stage, job.entry, job.key, {"scene": job.scene, "variant": variant.name, "log": str(log)})
                job.captures, job.complete = job.entry, True
            else:
                print(f"[oracle] {variant.name}/{job.scene}: capture incomplete ({why}); comparing the cameras that "
                      f"were captured, not cached (log {log})", file=sys.stderr, flush=True)
                job.captures = stage


def use_world(job: WorldJob, out: Path) -> Path | None:
    """Links ``out/client`` to the job's captures and records the reference in ``out/reference.json``. Without
    captures (a miss with ``--no-capture``) the existing ``out/client`` is kept if it has any capture (the report
    marks the reference stale). Returns the client directory or None."""
    client = out / "client"
    if job.captures is None:
        if client.is_dir() and any(client.glob("*.json")):
            print(f"[oracle] {job.variant.name}/{job.scene}: no client launched (--no-capture); comparing against the "
                  f"existing captures in {client}, which may be stale", file=sys.stderr, flush=True)
            data = json.loads((out / REFERENCE).read_text(encoding="utf-8")) if (out / REFERENCE).is_file() else {}
            record(out, {**data, "stale": True, "wanted_key": job.key.digest})
            return client
        return None
    link(client, job.captures)
    record(out, {"key": job.key.digest, "entry": str(job.captures), "cached": job.cached, "complete": job.complete,
                 "stale": False, "log": str(job.log) if job.log else None, "note": job.note,
                 "inputs": {k: v for k, v in job.key.inputs.items() if k not in ("client_mods", "cameras")}
                 | {"client_mods": len(job.key.inputs["client_mods"])}})
    return client


# ---------------------------------------------------------------- state references


def ensure_states(variant: variants.Variant, out: Path, inputs: Inputs, *, recapture: bool, capture: bool,
                  options: LaunchOptions) -> Key:
    """Makes ``out/client`` (the state oracle's reference directory) the cached capture of the variant's states,
    running ``gtnh-client states`` on a miss. Returns the key."""
    started = time.monotonic()
    key = inputs.states(variant)
    inputs.save()
    print(f"[states] reference key {key.short} computed in {time.monotonic() - started:.1f}s", flush=True)
    entry = states_entries(variant) / key.short
    if not recapture and (entry / "client" / "index.json").is_file():
        print(f"[states] reference {key.short} cached ({entry})", flush=True)
        use_states_reference(out, entry, key.digest)
        record(out, {"key": key.digest, "entry": str(entry), "cached": True, "stale": False})
        return key
    note = "recapture requested" if recapture else explain(key, states_entries(variant))
    print(f"[states] reference {key.short} not cached ({note})", flush=True)
    if not capture:
        if (out / "client" / "index.json").is_file():
            print(f"[states] no client launched (--no-capture); comparing against the existing {out / 'client'}, which "
                  f"may be stale", file=sys.stderr, flush=True)
            data = json.loads((out / REFERENCE).read_text(encoding="utf-8")) if (out / REFERENCE).is_file() else {}
            record(out, {**data, "stale": True, "wanted_key": key.digest})
            return key
        raise ReferenceError(f"no cached state reference {key.short} and --no-capture")
    stage = staging(entry)
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    log = log_file("states", variant)
    code, seconds = run_launcher(["states", "--variant", variant.name, "--out", str((stage / "client").resolve()),
                                  "--states-from", str(variant.states_from.resolve()),
                                  *client_flags(variant, options)], log)
    print(f"[states] reference capture {seconds:.0f}s (exit {code})", flush=True)
    if code != 0:
        raise ReferenceError(f"gtnh-client states failed (exit {code}); log {log}")
    listed = stage / "states.txt"
    if not listed.is_file() or listed.read_text(encoding="utf-8") != state_list(variant):
        raise ReferenceError(f"{listed} differs from the state list e2e computed for the key (reference.state_list "
                             f"and the launcher's states.write_states disagree); log {log}")
    # read through a local copy (use_states_reference): nothing links to the entry, no read-only marking needed
    promote(stage, entry, key, {"variant": variant.name, "log": str(log), "seconds": round(seconds, 1)},
            read_only=not copies_states(out, entry))
    use_states_reference(out, entry, key.digest)
    record(out, {"key": key.digest, "entry": str(entry), "cached": False, "stale": False, "log": str(log),
                 "seconds": round(seconds, 1)})
    return key


# ---------------------------------------------------------------- calibration artifacts


def calibration_key(kind: str, references: dict) -> str:
    """Key of a calibration: its kind (``world`` or ``states``), the calibration method version and the reference
    keys it is computed from."""
    return hashlib.sha256(_canonical({"kind": kind, "version": CALIBRATION_VERSION, "references": references})).hexdigest()


def calibration_path(kind: str, key: str, pack: paths.Pack | None = None) -> Path:
    return root(pack) / "calibration" / f"{kind}-{key[:KEY_CHARS]}.json"


def store_calibration(data: dict, artifact: Path | None, active: Path) -> None:
    """Writes a calibration to its artifact (when keyed) and to the active copy the reports and swap tests read."""
    text = json.dumps(data, indent=1) + "\n"
    for path in ([artifact] if artifact else []) + [active]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def install_calibration(artifact: Path, active: Path) -> dict:
    """Makes a stored calibration the active one; returns it."""
    text = artifact.read_text(encoding="utf-8")
    active.parent.mkdir(parents=True, exist_ok=True)
    if not active.is_file() or active.read_text(encoding="utf-8") != text:
        active.write_text(text, encoding="utf-8")
    return json.loads(text)
