"""Oracle variants: which client draws the reference and which resource packs BlueMap renders with.

| Variant | Client | Client packs / mods (this run only) | BlueMap packs | Scene server |
|---|---|---|---|---|
| `default` | `clients/gtnh-2.9.0-RC-2` | - | the default export | `gtnh-2.9.0-RC-2` (25565 / 25575) |
| `modernity` | `clients/gtnh-modernity` | the Modernity release zip / its MyCTMLib jar | default export + both `modernity` packs | `gtnh-2.9.0-RC-2` |
| `addons` | `clients/gtnh-addons` | - / the six addon jars of `downloads/addons/versions.json` | the addons export | `gtnh-addons` (25568 / 25578) |

Every path is under the workspace (`paths.workspace()`); the newest matching file is taken where a name carries a
version or release id. `missing()` lists the inputs that do not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from e2e import paths

NAMES = ("default", "modernity", "addons")
DEFAULT_PACK = Path("out/exporter/bluemap-gtnh-resources-2.9.0-RC-2.zip")


@dataclass(frozen=True)
class Variant:
    name: str
    client_dir: Path
    resource_packs: tuple[Path, ...]  # client resource packs, first = highest priority
    extra_mods: tuple[Path, ...]  # client mods installed for the run only
    bluemap_packs: tuple[Path, ...]  # files for BlueMap's packs/ (BlueMap orders them by reverse file name)
    server: str  # directory under <workspace>/servers that runs the scene worlds
    server_port: int
    rcon_port: int

    def missing(self) -> list[Path]:
        """Inputs (client directory, packs, mods) that do not exist."""
        return [p for p in (self.client_dir, *self.resource_packs, *self.extra_mods, *self.bluemap_packs) if not p.exists()]


def _newest(directory: Path, pattern: str) -> Path:
    """The newest match by modification time, or the pattern itself (a path that does not exist) if none matches."""
    found = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime) if directory.is_dir() else []
    return found[-1] if found else directory / pattern


def _modernity_downloads(ws: Path) -> tuple[Path, Path]:
    """The release zip and MyCTMLib jar recorded by `modernity fetch` in release.json, else the newest downloads."""
    folder = ws / "downloads" / "modernity"
    release = folder / "release.json"
    if release.is_file():
        files = json.loads(release.read_text(encoding="utf-8")).get("files", {})
        if "pack" in files and "myctmlib" in files:
            return folder / files["pack"]["name"], folder / files["myctmlib"]["name"]
    return _newest(folder, "Modernity-*.zip"), _newest(folder, "MyCTMLib-*.jar")


def _addon_jars(ws: Path) -> tuple[Path, ...]:
    folder = ws / "downloads" / "addons"
    versions = folder / "versions.json"
    if not versions.is_file():
        return (versions,)
    return tuple(folder / addon["file"] for addon in json.loads(versions.read_text(encoding="utf-8"))["addons"])


def load(name: str) -> Variant:
    ws = paths.workspace()
    if name == "default":
        return Variant(
            name=name,
            client_dir=ws / "clients" / "gtnh-2.9.0-RC-2",
            resource_packs=(),
            extra_mods=(),
            bluemap_packs=(ws / DEFAULT_PACK,),
            server="gtnh-2.9.0-RC-2",
            server_port=25565,
            rcon_port=25575,
        )
    if name == "modernity":
        pack, myctmlib = _modernity_downloads(ws)
        out = ws / "out" / "modernity"
        return Variant(
            name=name,
            client_dir=ws / "clients" / "gtnh-modernity",
            resource_packs=(pack,),
            extra_mods=(myctmlib,),
            bluemap_packs=(ws / DEFAULT_PACK, _newest(out, "modernity-*-modded-*.zip"), _newest(out, "modernity-*-vanilla-*.zip")),
            server="gtnh-2.9.0-RC-2",
            server_port=25565,
            rcon_port=25575,
        )
    if name == "addons":
        return Variant(
            name=name,
            client_dir=ws / "clients" / "gtnh-addons",
            resource_packs=(),
            extra_mods=_addon_jars(ws),
            bluemap_packs=(_newest(ws / "out" / "exporter-addons", "bluemap-gtnh-resources-*-addons.zip"),),
            server="gtnh-addons",
            server_port=25568,
            rcon_port=25578,
        )
    raise ValueError(f"unknown variant {name!r}; known: {', '.join(NAMES)}")


def oracle_dir(variant: Variant | str) -> Path:
    """`<workspace>/out/oracle/<variant>`: every oracle output of a variant (never committed)."""
    name = variant.name if isinstance(variant, Variant) else variant
    if name not in NAMES:
        raise ValueError(f"unknown variant {name!r}; known: {', '.join(NAMES)}")
    return paths.workspace() / "out" / "oracle" / name
