# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""`exclude_no_placement_params` must run after the Block Swap decisions (Auto bail, Count collapse)."""

import ast
import os
import sys

import unsloth  # noqa: F401
from unsloth.models.loader_utils import exclude_no_placement_params

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_no_placement_params_device_map import Model, check  # noqa: E402

VISION = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "unsloth", "models", "vision.py"
)


def _lines(tree, predicate):
    lines = [n.lineno for n in ast.walk(tree) if predicate(n)]
    assert lines
    return lines


def _calls(name):
    return lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", None) == name


def test_no_placement_exclusion_runs_after_every_block_swap_decision():
    tree = ast.parse(open(VISION, encoding = "utf-8").read())
    (exclude,) = _lines(tree, _calls("exclude_no_placement_params"))
    for decision in ("refuse_block_swap_load", "resolve_auto_block_swap"):
        assert max(_lines(tree, _calls(decision))) < exclude, decision
    pins = _lines(
        tree,
        lambda n: isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", None) == "device_map"
        and isinstance(n.value, ast.Dict)
        and any(getattr(v, "id", None) == "_block_swap_device" for v in n.value.values),
    )
    assert max(pins) < exclude


def test_excluded_map_is_not_the_bare_map_block_swap_needs():
    # resolve_auto_block_swap plans only `{"": N}`; this is an explicit map, so Auto bailed.
    out = exclude_no_placement_params({"": 0}, Model, None)
    check(out, 0)
    assert set(out) != {""}
