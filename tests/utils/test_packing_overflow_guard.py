# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present the Unsloth team. All rights reserved.

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

_SOURCE = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "rl_replacements.py"


def _setup_source():
    tree = ast.parse(_SOURCE.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "_WRAPPED_PACKING_SETUP" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("_WRAPPED_PACKING_SETUP not found")


def _skips_truncation(packing, strategy):
    def pack_dataset(
        dataset,
        seq_length,
        strategy = "bfd",
        map_kwargs = None,
    ):
        pass

    body = (
        "def _f(packing, args, pack_dataset):\n"
        + _setup_source()
        + "    return _unsloth_wrapped_packing\n"
    )
    namespace = {}
    exec(body, namespace)
    return namespace["_f"](packing, SimpleNamespace(packing_strategy = strategy), pack_dataset)


# TRL packs without truncating first; only plain bfd truncates overflow itself, so pre-truncating it is equivalent.
@pytest.mark.parametrize("strategy", ["wrapped", "bfd_split", "bfd-requeue"])
def test_overflow_keeping_strategies_skip_early_truncation(strategy):
    assert _skips_truncation(True, strategy) is True


@pytest.mark.parametrize("strategy", ["bfd", None])
def test_bfd_still_truncates_early(strategy):
    assert not _skips_truncation(True, strategy)


@pytest.mark.parametrize("strategy", ["wrapped", "bfd_split", "bfd-requeue"])
def test_no_packing_always_truncates(strategy):
    assert not _skips_truncation(False, strategy)
