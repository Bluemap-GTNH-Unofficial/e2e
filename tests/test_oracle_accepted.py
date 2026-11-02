import pytest

from e2e.oracle import accepted, vanilla


def _record(name: str, meta: int, **kw) -> dict:
    return {"name": name, "meta": meta, "textures": kw.get("textures", {}), "randomOffset": kw.get("offset", False),
            "randomVariant": kw.get("variant", False), "flatteningDrop": kw.get("drop", False)}


def test_parse_pattern() -> None:
    assert accepted.parse_pattern("minecraft:wool") == ("minecraft:wool", None)
    assert accepted.parse_pattern("minecraft:vine[meta=1-3,7]") == ("minecraft:vine", frozenset({1, 2, 3, 7}))
    with pytest.raises(ValueError):
        accepted.parse_pattern("wool[1]")


def test_the_committed_list_parses_and_names_known_reasons() -> None:
    data = {"states": {}}
    entries = accepted.load(data, set()).entries
    assert entries and all(e.reason in vanilla.REASONS for e in entries)
    assert any(e.name == "minecraft:stone" and e.metas == frozenset({0}) and e.reason == "random_variant"
               for e in entries)


def test_only_data_backed_states_are_accepted(tmp_path) -> None:
    file = tmp_path / "accepted.yaml"
    file.write_text("reasons: {texture: t, random_offset: o}\naccepted:\n"
                    "  - {states: 'minecraft:wool', reason: texture}\n"
                    "  - {states: 'minecraft:red_flower[meta=0]', reason: random_offset}\n"
                    "  - {states: 'minecraft:gone', reason: texture}\n")
    states = {
        "minecraft:wool[meta=0]": _record("minecraft:wool", 0, textures={"block/white_wool": "differs"}),
        "minecraft:wool[meta=1]": _record("minecraft:wool", 1, textures={"block/orange_wool": "identical"}),
        "minecraft:red_flower[meta=0]": _record("minecraft:red_flower", 0, offset=True),
    }
    a = accepted.load({"states": states}, set(), file)
    assert a.check("minecraft:wool", 0)[1] is True
    assert a.check("minecraft:wool", 1) == (a.entries[0], False)  # named, but its texture is identical: judged
    assert a.check("minecraft:red_flower", 0)[1] is True
    assert a.check("minecraft:stone", 0) == (None, False)
    assert a.unmatched() == ["minecraft:gone"]
    # a variant whose packs replace the differing texture draws it from the pack, not the 1.13 jar
    assert accepted.load({"states": states}, {"block/white_wool"}, file).check("minecraft:wool", 0)[1] is False


def test_bad_reason_is_rejected(tmp_path) -> None:
    file = tmp_path / "accepted.yaml"
    file.write_text("reasons: {shape: s}\naccepted:\n  - {states: 'minecraft:lever', reason: shape}\n")
    with pytest.raises(ValueError):
        accepted.load({"states": {}}, set(), file)
