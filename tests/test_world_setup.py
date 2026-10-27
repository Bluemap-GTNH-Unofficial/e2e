from pathlib import Path

from e2e import bluemap
from e2e.scenes import View


def _region(folder: Path, name: str = "r.0.0.mca", size: int = 8192) -> None:
    (folder / "region").mkdir(parents=True, exist_ok=True)
    (folder / "region" / name).write_bytes(b"\0" * size)


def test_maps_for_every_dimension_with_region_files(tmp_path: Path) -> None:
    world = tmp_path / "world"
    _region(world)
    _region(world / "DIM-1")
    _region(world / "DIM1", "r.-1.0.mca")
    _region(world / "DIM64")
    _region(world / "DIM7", size=0)                  # empty region file: no chunks
    (world / "DIM2" / "region").mkdir(parents=True)  # no region files
    _region(world / "DIM112", "r.0.0.mcr")           # not an anvil region
    dims = bluemap.detect_dimensions(world)
    assert [(d.map_id, d.key, d.folder.name) for d in dims] == [
        ("overworld", "minecraft:overworld", "world"),
        ("the_nether", "minecraft:the_nether", "DIM-1"),
        ("the_end", "minecraft:the_end", "DIM1"),
        ("dim_64", "gtnh:dim_64", "DIM64"),
    ]


def test_view_hash_has_the_ten_fields_the_webapp_parses() -> None:
    view = View(name="v", map="dim_64", x=18, y=5.5, z=-17, distance=30, rotation=0.6, angle=0.9, ortho=1,
                mode="flat")
    assert view.hash == "dim_64:18:5.5:-17:30:0.6:0.9:0:1:flat"
    assert len(view.hash.split(":")) == 10
