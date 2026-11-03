"""Oracle variants: which client draws the reference and which resource packs BlueMap renders with, per GTNH pack
version (``e2e --pack-version``, default 2.9.0; ``paths.Pack``: per-version directories end in ``-<version>``).

| Variant | Client | Client packs / mods (this run only) | BlueMap packs | Scene server |
|---|---|---|---|---|
| `default` | `clients/gtnh-<version>` | - | the default export | `gtnh-<version>` (2.9.0: 25569 / 25579) |
| `modernity` | `clients/gtnh-modernity-<version>` | the Modernity release zip / its MyCTMLib jar | default export + both `modernity` packs | `gtnh-<version>` |
| `addons` | `clients/gtnh-addons-<version>` | - / the addon jars of the version in `downloads/addons/versions.json` | the addons export | `gtnh-addons-<version>` (2.9.0: 25572 / 25582) |

Every path is under the workspace (`paths.workspace()`); the newest matching file is taken where a name carries a
release id. `missing()` lists the inputs that do not exist yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from e2e import paths

NAMES = ("default", "modernity", "addons")


@dataclass(frozen=True)
class Variant:
    name: str
    pack: paths.Pack  # the GTNH pack version the variant runs on
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


def _addon_jars(ws: Path, version: str) -> tuple[Path, ...]:
    """The addon jars `downloads/addons/versions.json` lists for pack ``version`` (``packs.<version>.addons``)."""
    folder = ws / "downloads" / "addons"
    versions = folder / "versions.json"
    if not versions.is_file():
        return (versions,)
    addons = json.loads(versions.read_text(encoding="utf-8")).get("packs", {}).get(version, {}).get("addons")
    if addons is None:
        return (versions,)  # reported as missing: no addon set for this pack version
    return tuple(folder / addon["file"] for addon in addons)


def load(name: str, pack_version: str | None = None) -> Variant:
    """Variant ``name`` on ``pack_version`` (default: the selected pack version, ``paths.pack()``)."""
    ws = paths.workspace()
    pack = paths.pack(pack_version)
    if name == "default":
        return Variant(
            name=name,
            pack=pack,
            client_dir=pack.client,
            resource_packs=(),
            extra_mods=(),
            bluemap_packs=(pack.export,),
            server=pack.server,
            server_port=pack.server_port,
            rcon_port=pack.rcon_port,
        )
    if name == "modernity":
        modpack, myctmlib = _modernity_downloads(ws)
        out = ws / "out" / "modernity"
        return Variant(
            name=name,
            pack=pack,
            client_dir=ws / "clients" / f"gtnh-modernity{pack.suffix}",
            resource_packs=(modpack,),
            extra_mods=(myctmlib,),
            bluemap_packs=(pack.export, _newest(out, f"modernity-*-modded-{pack.version}.zip"),
                           _newest(out, "modernity-*-vanilla-*.zip")),  # fmt: skip
            server=pack.server,
            server_port=pack.server_port,
            rcon_port=pack.rcon_port,
        )
    if name == "addons":
        return Variant(
            name=name,
            pack=pack,
            client_dir=ws / "clients" / f"gtnh-addons{pack.suffix}",
            resource_packs=(),
            extra_mods=_addon_jars(ws, pack.version),
            bluemap_packs=(pack.addons_export,),
            server=pack.addon_server,
            server_port=pack.addon_server_port,
            rcon_port=pack.addon_rcon_port,
        )
    raise ValueError(f"unknown variant {name!r}; known: {', '.join(NAMES)}")


def oracle_dir(variant: Variant | str) -> Path:
    """`<workspace>/out/oracle[-<version>]/<variant>`: every oracle output of a variant (never committed); a variant
    given by name is on the selected pack version."""
    name = variant.name if isinstance(variant, Variant) else variant
    if name not in NAMES:
        raise ValueError(f"unknown variant {name!r}; known: {', '.join(NAMES)}")
    pack = variant.pack if isinstance(variant, Variant) else paths.pack()
    return pack.dir("out/oracle") / name
