"""Reference cache of the oracles: key stability, cache hits and misses with a fake launcher, calibration artifacts."""

from __future__ import annotations

import dataclasses
import json
import os
import random
import sys
import zipfile
from pathlib import Path

import pytest

from e2e import paths
from e2e.oracle import reference, variants, world

FAKE = Path(__file__).with_name("fake_gtnh_client.py")
SCENES = ("alpha", "beta")
DUMP = [
    {"x": 0, "y": 4, "z": 0, "id": 1, "name": "minecraft:stone", "meta": 0, "te": None},
    {"x": 1, "y": 5, "z": 0, "id": 2242, "name": "gregtech:gt.blockmachines", "meta": 0,
     "te": {"mID": 1, "mFacing": 2, "Inventory": [], "id": "BaseMetaTileEntity"}},
    {"x": 2, "y": 4, "z": 1, "id": 0, "name": "minecraft:air", "meta": 0, "te": None},
    {"x": 3, "y": 4, "z": 1, "id": 35, "name": "minecraft:wool", "meta": 14, "te": None},
]


def _write_dump(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")


def _jar(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content.encode())
    return path


@pytest.fixture
def ws(tmp_path, monkeypatch):
    """A workspace with a launcher checkout (capture versions), a client, the harness jar, scene dumps and worlds, the
    flattening table and an export; the fake launcher replaces gtnh-client."""
    monkeypatch.setenv("BMGTNH_WORKSPACE", str(tmp_path))
    monkeypatch.delenv("BMGTNH_PACK_VERSION", raising=False)
    src = tmp_path / "gtnh-exporter" / "launcher" / "src" / "gtnh_client"
    src.mkdir(parents=True)
    (src / "cameras.py").write_text('"""launcher"""\nCAPTURE_VERSION = 3\n')
    (src / "states.py").write_text("STATES_CAPTURE_VERSION: int = 1\n")
    mods = tmp_path / "clients" / "gtnh-2.9.0" / ".minecraft" / "mods"
    _jar(mods / "gregtech.jar", "gt")
    _jar(mods / "cached.zip", "zip mod")
    _jar(mods / "1.7.10" / "old.jar", "old")
    _jar(mods / "bluemap-gtnh-exporter-0.1.0.jar", "exporter, covered by the capture version")
    _jar(mods / "notes.txt", "not a mod")
    _jar(mods / "carpentersblocks" / "CarpentersBlocksCachedResources.zip", "a mod's cache, not loaded")
    _jar(tmp_path / "servers" / "gtnh-2.9.0" / "mods" / "bmh-0.3.0.jar", "harness")
    for scene in SCENES:
        _write_dump(tmp_path / "out" / "scenes-2.9.0" / scene / "dump.ndjson", DUMP)
        (tmp_path / "worlds-2.9.0" / scene / "region").mkdir(parents=True)
    tsv = tmp_path / "BlueMap" / reference.FLATTENING_TSV
    tsv.parent.mkdir(parents=True)
    tsv.write_text("0\tminecraft:air\t0\tminecraft:air\t1\n1\tminecraft:stone\t0\tminecraft:stone\t1\n"
                   "1\tminecraft:stone\t7\tminecraft:stone\t0\n")
    export = paths.pack().export
    export.parent.mkdir(parents=True)
    with zipfile.ZipFile(export, "w") as z:
        z.writestr(reference.BLOCKS_NDJSON, json.dumps({"name": "gregtech:gt.blockmachines", "metas": [{"meta": 0}]}))
    monkeypatch.setenv(reference.LAUNCHER_ENV, f"{sys.executable} {FAKE}")
    monkeypatch.setenv("FAKE_LOG", str(tmp_path / "launches.jsonl"))
    monkeypatch.setenv("FAKE_CAPTURE_VERSION", "3")
    return tmp_path


def _launches(ws: Path) -> list[list[str]]:
    log = ws / "launches.jsonl"
    return [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []


def _cameras(scene: str, eye: float = 1.0, remove: list | None = None) -> dict:
    return {"format": 1, "scene": scene, "world": scene, "levelType": "FLAT", "box": [0, 4, 0, 3, 5, 1],
            "cameras": [{"name": "north", "eye": [eye, 9.0, 0.0], "yaw": 0.0, "pitch": 30.0, "fov": 70.0,
                         "width": 64, "height": 36, "source": "oracle.cameras"},
                        {"name": "south", "eye": [eye, 9.0, 9.0], "yaw": 180.0, "pitch": 30.0, "fov": 70.0,
                         "width": 64, "height": 36, "source": "oracle.cameras"}],
            "remove": remove or []}


def _key(variant: variants.Variant | None = None, scene: str = "alpha", cameras: dict | None = None) -> str:
    inputs = reference.Inputs.open()
    return inputs.world(variant or variants.load("default"), scene, cameras or _cameras(scene)).digest


# ---------------------------------------------------------------- keys


def test_dump_digest_ignores_order_ids_air_and_file_times(ws):
    path = ws / "out" / "scenes-2.9.0" / "alpha" / "dump.ndjson"
    before = reference.dump_digest(path)
    shuffled = [dict(e, id=e["id"] + 1000) for e in DUMP if e["name"] != "minecraft:air"]
    random.Random(4).shuffle(shuffled)
    shuffled[0]["te"] = dict(reversed(list(shuffled[0]["te"].items()))) if shuffled[0]["te"] else None
    _write_dump(path, shuffled)
    os.utime(path, (1_000_000, 1_000_000))
    assert reference.dump_digest(path) == before
    for change in ({"meta": 1}, {"te": {"mID": 2, "mFacing": 2, "Inventory": [], "id": "BaseMetaTileEntity"}},
                   {"name": "minecraft:dirt"}, {"y": 6}):
        _write_dump(path, [dict(DUMP[1], **change), *DUMP[:1], *DUMP[2:]])
        assert reference.dump_digest(path) != before, change


def test_world_key_is_stable_over_timestamps_and_rebuilds(ws):
    first = _key()
    for p in [*ws.rglob("*.jar"), *ws.rglob("*.zip"), *ws.rglob("dump.ndjson")]:
        os.utime(p, (2_000_000, 2_000_000))  # every input touched: hashes are read again, the key stays
    assert _key() == first
    dump = ws / "out" / "scenes-2.9.0" / "alpha" / "dump.ndjson"
    _write_dump(dump, list(reversed(DUMP)))  # the scene rebuilt with the same content
    assert _key() == first
    memo = json.loads((reference.root() / "hashes.json").read_text()) if (reference.root() / "hashes.json").is_file() else {}
    assert all("sha256" in v for v in memo.values())
    assert _key(scene="beta", cameras=_cameras("beta")) != first  # other scene name


@pytest.mark.parametrize("change", ["mod", "new mod", "mod 1.7.10", "extra mod", "resource pack", "pack order",
                                    "dump", "camera", "remove", "capture version", "harness", "box"])
def test_every_input_changes_the_world_key(ws, change):
    mods = ws / "clients" / "gtnh-2.9.0" / ".minecraft" / "mods"
    a, b = _jar(ws / "downloads" / "a.zip", "pack a"), _jar(ws / "downloads" / "b.zip", "pack b")
    extra = _jar(ws / "downloads" / "extra.jar", "extra")
    variant = dataclasses.replace(variants.load("default"), resource_packs=(a, b), extra_mods=(extra,))
    cameras = _cameras("alpha")
    before = _key(variant, cameras=cameras)
    if change == "mod":
        _jar(mods / "gregtech.jar", "gt 2")
    elif change == "new mod":
        _jar(mods / "another.jar", "x")
    elif change == "mod 1.7.10":
        _jar(mods / "1.7.10" / "old.jar", "old 2")
    elif change == "extra mod":
        _jar(extra, "extra 2")
    elif change == "resource pack":
        _jar(b, "pack b 2")
    elif change == "pack order":
        variant = dataclasses.replace(variant, resource_packs=(b, a))
    elif change == "dump":
        _write_dump(ws / "out" / "scenes-2.9.0" / "alpha" / "dump.ndjson", DUMP[:2])
    elif change == "camera":
        cameras = _cameras("alpha", eye=1.5)
    elif change == "remove":
        cameras = _cameras("alpha", remove=[[3, 4, 1]])
    elif change == "capture version":
        (ws / "gtnh-exporter" / "launcher" / "src" / "gtnh_client" / "cameras.py").write_text("CAPTURE_VERSION = 4\n")
    elif change == "harness":
        _jar(ws / "servers" / "gtnh-2.9.0" / "mods" / "bmh-0.3.0.jar", "harness 2")
    elif change == "box":
        cameras = dict(cameras, box=[0, 4, 0, 3, 6, 1])
    assert _key(variant, cameras=cameras) != before


def test_mods_that_do_not_count(ws):
    mods = ws / "clients" / "gtnh-2.9.0" / ".minecraft" / "mods"
    before = _key()
    _jar(mods / "bluemap-gtnh-exporter-0.1.0.jar", "rebuilt exporter")
    _jar(mods / "carpentersblocks" / "CarpentersBlocksCachedResources.zip", "rewritten cache")
    _jar(mods / "readme.txt", "x")
    assert _key() == before
    listed = [name for name, _ in reference.client_mods(ws / "clients" / "gtnh-2.9.0", reference.Hasher(None))]
    assert listed == ["1.7.10/old.jar", "cached.zip", "gregtech.jar"]


def test_capture_version_is_read_from_the_launcher_source(ws):
    assert (reference.capture_version(), reference.states_capture_version()) == (3, 1)
    (ws / "gtnh-exporter" / "launcher" / "src" / "gtnh_client" / "states.py").write_text("X = 1\n")
    with pytest.raises(reference.ReferenceError, match="STATES_CAPTURE_VERSION"):
        reference.states_capture_version()


def test_states_key_follows_the_state_list(ws):
    variant = variants.load("default")
    listed = reference.state_list(variant)
    assert listed == "minecraft:stone 0\ngregtech:gt.blockmachines 0\n"
    before = reference.Inputs.open().states(variant).digest
    os.utime(variant.states_from, (3_000_000, 3_000_000))
    with zipfile.ZipFile(variant.states_from, "a") as z:  # another file of the export: same state list
        z.writestr("assets/x.png", b"png")
    assert reference.Inputs.open().states(variant).digest == before
    with zipfile.ZipFile(variant.states_from, "w") as z:
        z.writestr(reference.BLOCKS_NDJSON, json.dumps({"name": "gregtech:gt.blockmachines", "metas": [{"meta": 1}]}))
    assert reference.Inputs.open().states(variant).digest != before


# ---------------------------------------------------------------- world references with the fake launcher


def _jobs(ws: Path, variant: variants.Variant, cams: dict[str, dict] | None = None) -> list[reference.WorldJob]:
    inputs = reference.Inputs.open()
    jobs = []
    for scene in SCENES:
        out = world.scene_dir(variant, scene)
        out.mkdir(parents=True, exist_ok=True)
        data = (cams or {}).get(scene) or _cameras(scene)
        (out / "cameras.json").write_text(json.dumps(data))
        jobs.append(reference.WorldJob(variant, scene, ws / "worlds-2.9.0" / scene, out / "cameras.json", data,
                                       inputs.world(variant, scene, data)))
    return jobs


def _capture(jobs, **kwargs):
    options = dict(recapture=False, capture=True, options=reference.LaunchOptions(parallel=2))
    reference.capture_world(jobs, **(options | kwargs))
    return {j.scene: reference.use_world(j, world.scene_dir(j.variant, j.scene)) for j in jobs}


def test_second_run_launches_no_client(ws):
    variant = variants.load("default")
    clients = _capture(_jobs(ws, variant))
    launches = _launches(ws)
    assert len(launches) == 1  # one batch for both scenes
    argv = launches[0]
    assert argv[:2] == ["cameras", "--jobs"] and argv[argv.index("--parallel") + 1] == "2"
    spec = json.loads(Path(argv[2]).read_text())["jobs"]
    assert [j["scene"] for j in spec] == list(SCENES)
    assert all(j["freeze_random_ticks"] and j["world"].endswith(f"worlds-2.9.0/{j['scene']}") for j in spec)
    for scene, client in clients.items():
        assert client.is_symlink() and (client / "north.png").read_bytes() == scene.encode()
        entry = client.resolve()
        assert entry.parent == reference.world_entries(variant, scene) and (entry / "key.json").is_file()
        assert not (entry / "north.png").stat().st_mode & 0o222  # cache entries are read-only (os.access: root can write)
        ref = json.loads((world.scene_dir(variant, scene) / reference.REFERENCE).read_text())
        assert ref["cached"] is False and ref["complete"] is True

    jobs = _jobs(ws, variant)
    _capture(jobs)
    assert len(_launches(ws)) == 1 and all(j.cached for j in jobs)
    assert json.loads((world.scene_dir(variant, "alpha") / reference.REFERENCE).read_text())["cached"] is True


def test_only_changed_references_are_captured_again(ws):
    variant = variants.load("default")
    _capture(_jobs(ws, variant))
    old = (world.scene_dir(variant, "beta") / "client").resolve()
    jobs = _jobs(ws, variant, {"beta": _cameras("beta", eye=2.0)})
    _capture(jobs)
    launches = _launches(ws)
    assert len(launches) == 2
    assert [j["scene"] for j in json.loads(Path(launches[1][2]).read_text())["jobs"]] == ["beta"]
    assert "cameras (cameras)" in jobs[1].note  # the miss says what changed
    assert (world.scene_dir(variant, "beta") / "client").resolve() != old and old.is_dir()  # old entry kept

    jobs = _jobs(ws, variant, {"beta": _cameras("beta", eye=2.0)})
    _capture(jobs, recapture=True)
    assert len(_launches(ws)) == 3 and not any(j.cached for j in jobs)


def test_failed_capture_is_compared_but_not_cached(ws, monkeypatch):
    variant = variants.load("default")
    monkeypatch.setenv("FAKE_FAIL_SCENES", "beta")
    jobs = _jobs(ws, variant)
    clients = _capture(jobs)
    assert jobs[0].complete and not jobs[1].complete
    assert (clients["beta"] / "north.json").is_file() and not (clients["beta"] / "south.json").exists()
    assert not reference.world_entries(variant, "beta").joinpath(jobs[1].key.short).exists()
    monkeypatch.delenv("FAKE_FAIL_SCENES")
    jobs = _jobs(ws, variant)
    _capture(jobs)
    assert [j.cached for j in jobs] == [True, False] and jobs[1].complete


def test_wrong_capture_version_is_not_cached(ws, monkeypatch):
    monkeypatch.setenv("FAKE_CAPTURE_VERSION", "2")
    jobs = _jobs(ws, variants.load("default"))
    _capture(jobs)
    assert not any(j.complete for j in jobs)


def test_one_batch_per_variant(ws):
    modernity = dataclasses.replace(variants.load("modernity"), client_dir=ws / "clients" / "gtnh-2.9.0",
                                    resource_packs=(_jar(ws / "downloads" / "modernity.zip", "m"),), extra_mods=())
    jobs = _jobs(ws, variants.load("default")) + _jobs(ws, modernity)
    _capture(jobs)
    launches = _launches(ws)
    assert len(launches) == 2
    assert any(a.startswith("--resource-pack=") and a.endswith("modernity.zip") for a in launches[1])


def test_no_capture_keeps_existing_captures_as_stale(ws):
    variant = variants.load("default")
    legacy = world.scene_dir(variant, "alpha") / "client"
    legacy.mkdir(parents=True)
    (legacy / "north.json").write_text("{}")
    clients = _capture(_jobs(ws, variant), capture=False)
    assert _launches(ws) == []
    assert clients["alpha"] == legacy and not legacy.is_symlink() and clients["beta"] is None
    assert json.loads((legacy.parent / reference.REFERENCE).read_text())["stale"] is True
    _capture(_jobs(ws, variant))  # a capture replaces the pre-cache directory with the link
    assert legacy.is_symlink()


# ---------------------------------------------------------------- state references


def test_state_reference_is_cached(ws, monkeypatch):
    variant = variants.load("default")
    listed = ws / "states-fake.txt"
    listed.write_text(reference.state_list(variant))
    monkeypatch.setenv("FAKE_STATES_TXT", str(listed))
    out = ws / "out" / "oracle-2.9.0" / "default" / "states"
    options = reference.LaunchOptions()
    key = reference.ensure_states(variant, out, reference.Inputs.open(), recapture=False, capture=True, options=options)
    assert len(_launches(ws)) == 1 and (out / "client").is_symlink() and (out / "client" / "index.json").is_file()
    argv = _launches(ws)[0]
    assert argv[argv.index("--states-from") + 1] == str(variant.states_from)
    assert reference.recorded_key(out) == key.digest
    assert reference.ensure_states(variant, out, reference.Inputs.open(), recapture=False, capture=True,
                                   options=options) == key
    assert len(_launches(ws)) == 1

    listed.write_text("minecraft:stone 0\n")  # the launcher draws another list than the key says
    with pytest.raises(reference.ReferenceError, match="differs"):
        reference.ensure_states(variant, out, reference.Inputs.open(), recapture=True, capture=True, options=options)


def test_file_system_is_the_longest_mount_prefix(tmp_path, monkeypatch):
    mounts = tmp_path / "mounts"
    mounts.write_text("overlay / overlay rw 0 0\ndrvfs /ws virtiofs rw 0 0\n/dev/sdc /ws/out/oracle-2.9.0 ext4 rw 0 0\n")
    monkeypatch.setattr(reference, "MOUNTS", mounts)
    assert reference.file_system(Path("/ws/out/oracle-ref/2.9.0")) == "virtiofs"
    assert reference.file_system(Path("/ws/out/oracle-2.9.0/default/states")) == "ext4"
    assert reference.file_system(Path("/ws/out/oracle-2.9.0x")) == "virtiofs"
    assert reference.host_shared(Path("/ws/x")) and not reference.host_shared(Path("/tmp/x"))
    # a volume below the bind mount is not shared itself, but its absolute paths cross the bind mount
    assert not reference.host_shared(Path("/ws/out/oracle-2.9.0/x"))
    assert reference.crosses_host_shared(Path("/ws/out/oracle-2.9.0/x"))
    assert not reference.crosses_host_shared(Path("/tmp/x"))


def test_host_shared_state_reference_is_copied_once_per_capture(ws, monkeypatch):
    """With the cache on the host-shared bind mount, states/client is a local copy (no link into the cache, so the
    entry's files are not made read-only), made once per captured entry and replacing an earlier link or copy."""
    variant = variants.load("default")
    listed = ws / "states-fake.txt"
    listed.write_text(reference.state_list(variant))
    monkeypatch.setenv("FAKE_STATES_TXT", str(listed))
    cache = reference.root()
    monkeypatch.setattr(reference, "host_shared", lambda path: path.resolve().is_relative_to(cache.resolve()))
    out = ws / "out" / "oracle-2.9.0" / "default" / "states"
    out.mkdir(parents=True)
    (out / "client").symlink_to(ws, target_is_directory=True)  # an earlier link
    options = reference.LaunchOptions()
    key = reference.ensure_states(variant, out, reference.Inputs.open(), recapture=False, capture=True, options=options)
    client = out / "client"
    assert not client.is_symlink() and (client / "index.json").is_file()
    assert json.loads((client / reference.COPY_MARKER).read_text())["key"] == key.digest
    entry = reference.states_entries(variant) / key.short / "client"
    assert (entry / "index.json").stat().st_mode & 0o200  # left writable

    copy = reference.copy_reference
    results = []
    monkeypatch.setattr(reference, "copy_reference", lambda *a, **k: results.append(copy(*a, **k)) or results[-1])
    reference.ensure_states(variant, out, reference.Inputs.open(), recapture=False, capture=True, options=options)
    assert len(_launches(ws)) == 1 and results == [None]  # cached, and the local copy is current
    reference.ensure_states(variant, out, reference.Inputs.open(), recapture=True, capture=True, options=options)
    assert len(_launches(ws)) == 2 and results[-1] is not None  # the same key captured again: copied again
    files, _ = results[-1]
    assert files == sum(len(names) for _, _, names in os.walk(entry))


# ---------------------------------------------------------------- world calibration


@dataclasses.dataclass
class _Run:
    calibrating_pair: bool
    job: object


class _Inputs:
    def __init__(self, world_key: str, states_key: str) -> None:
        self.keys = (world_key, states_key)

    def world(self, *_):
        return reference.make_key({"world": self.keys[0]})

    def states(self, *_):
        return reference.make_key({"states": self.keys[1]})


def test_world_calibration_is_reused_until_its_references_change(ws, monkeypatch):
    monkeypatch.setattr(world.cameras, "scene_cameras", lambda scene: [])
    monkeypatch.setattr(world, "scene_blocks", lambda scene: ({}, [], []))
    own = _Run(True, reference.WorldJob(variants.load("default"), world.CALIBRATION_SCENE, ws, ws, {},
                                        reference.make_key({"world": "v1"})))
    inputs = _Inputs("v1", "s1")
    plan = world.plan_calibration([own], inputs, recalibrate=False)
    assert plan.compute and plan.artifact is not None and not plan.artifact.exists()
    reference.store_calibration({"thresholds": {"x": 1}, "references": plan.references}, plan.artifact,
                                world.calibration_file())

    # same references: reused, also by a run without the calibration scene
    world.calibration_file().write_text("{}")
    plan2 = world.plan_calibration([], inputs, recalibrate=False)
    assert not plan2.compute and plan2.artifact == plan.artifact
    assert json.loads(world.calibration_file().read_text())["thresholds"] == {"x": 1}  # installed as active
    assert world.plan_calibration([own], inputs, recalibrate=True).compute

    # another state reference: a new calibration, which needs the calibration scene in the run
    with pytest.raises(SystemExit, match="vanilla-zoo"):
        world.plan_calibration([], _Inputs("v1", "s2"), recalibrate=False)
    plan3 = world.plan_calibration([own], _Inputs("v1", "s2"), recalibrate=False)
    assert plan3.compute and plan3.artifact != plan.artifact
