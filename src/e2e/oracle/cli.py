"""`e2e oracle ...`: BlueMap compared with the real GTNH client (`states`: every block state in isolation, `world`:
scene worlds per visible block face)."""

from __future__ import annotations

import argparse

from e2e.oracle import states, world, world_swap


def register(sub: argparse._SubParsersAction) -> None:
    oracle = sub.add_parser("oracle", help="compare BlueMap with the real GTNH client (reference renders)")
    oracle_sub = oracle.add_subparsers(dest="oracle_command", required=True)
    states.add_parser(oracle_sub)
    world.add_parser(oracle_sub)
    world_swap.add_parser(oracle_sub)
