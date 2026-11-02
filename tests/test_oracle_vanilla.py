import numpy as np
import pytest

from e2e.oracle import metrics, vanilla


def test_custom_uv_is_a_uv_other_than_the_element_bounds() -> None:
    button = {"from": [5, 0, 6], "to": [11, 2, 10],
              "faces": {"down": {"uv": [5, 6, 11, 10]}, "north": {"uv": [5, 12, 11, 16]}}}
    assert vanilla._custom_uv(button)  # north auto UV is [5, 14, 11, 16]
    slab = {"from": [0, 0, 0], "to": [16, 8, 16], "faces": {"north": {"uv": [0, 8, 16, 16]}, "up": {}}}
    assert not vanilla._custom_uv(slab)
    assert vanilla._custom_uv({**slab, "faces": {"up": {"rotation": 90}}})
    cross = {"from": [0.8, 0, 8], "to": [15.2, 16, 8], "rotation": {"angle": 45},
             "faces": {"north": {"uv": [0, 0, 16, 16]}}}
    assert not vanilla._custom_uv(cross)


def test_mapping_is_consistent_only_when_looks_and_models_pair_up() -> None:
    def record(name, meta, state, model):
        return {"name": name, "meta": meta, "state": state, "models": [model]}

    records = {
        "f0": record("minecraft:farmland", 0, "minecraft:farmland[moisture=0]", "block/farmland"),
        "f1": record("minecraft:farmland", 1, "minecraft:farmland[moisture=1]", "block/farmland"),
        "f7": record("minecraft:farmland", 7, "minecraft:farmland[moisture=7]", "block/farmland_moist"),
        "w0": record("minecraft:wool", 0, "minecraft:white_wool", "block/white_wool"),
        "w1": record("minecraft:wool", 1, "minecraft:orange_wool", "block/orange_wool"),
        "s0": record("minecraft:stone", 0, "minecraft:stone", "block/stone"),
    }
    looks = {"f0": "dry", "f1": "wet", "f7": "wet", "w0": "white", "w1": "orange"}  # s0: the client draws nothing
    vanilla._mark_mapping(records, looks)
    assert {k: r["mappingConsistent"] for k, r in records.items()} == {
        "f0": False, "f1": False, "f7": False, "w0": True, "w1": True, "s0": False}


def test_geometry_band_ignores_a_far_out_cube() -> None:
    ratios = [0.007] * 100 + [0.02, 0.03, 0.03, 0.04, 0.04, 0.045, 0.05, 0.05] + [0.9]
    assert vanilla.geometry_band(ratios) == 0.05
    assert vanilla.geometry_band([0.0] * 40 + [0.5]) == 0.0  # too few other cubes: the common ratio


def test_grouped_calibration_takes_each_groups_worst_inexact_region() -> None:
    rng = np.random.default_rng(5)
    samples, groups = [], []
    for g in range(20):
        worst = 0.5 + g * 0.1
        for d in (0.0, 0.01, worst):  # exact, nearly exact, the group's worst
            samples.append(metrics.RegionMetrics(200, 200, 200, 1.0, d, d, 1.0 - d / 100))
            groups.append(f"b{g}")
    for _ in range(400):  # one group with many nearly exact regions must not set the fence
        samples.append(metrics.RegionMetrics(200, 200, 200, 1.0, float(rng.uniform(0.001, 0.01)), 0.0, 0.9999))
        groups.append("many")
    cal = metrics.calibrate(samples, groups=groups)
    worst = np.array([0.5 + g * 0.1 for g in range(20)] + [max(m.delta_e for m, g in zip(samples, groups) if g == "many")])
    q1, q3 = np.percentile(worst, [25, 75])
    assert cal.thresholds.delta_e_max == pytest.approx(q3 + 3 * (q3 - q1), abs=1e-3)
    assert cal.fences["delta_e"]["unit"] == "group" and cal.fences["delta_e"]["inexact"] == 21
    ungrouped = metrics.calibrate(samples)
    assert ungrouped.thresholds.delta_e_max < cal.thresholds.delta_e_max
