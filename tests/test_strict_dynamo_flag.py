# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Strict Dynamo regions (block compile's first call, the prefix grouper's checked call) turn suppress_errors off,
and still run on torch builds without that flag (AMD's ROCm torch 2.11): patching a missing flag raised, which
block compile took as a compile failure and demoted every block to the stock forward."""

import pathlib
import re
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)
import unsloth.models._block_compile as BC
import unsloth.utils.prefix_grouper_kernel as PG


@pytest.mark.skipif(
    not hasattr(torch._dynamo.config, "suppress_errors"),
    reason = "this torch build has no suppress_errors flag",
)
@pytest.mark.parametrize("module", [BC, PG], ids = ["block_compile", "prefix_grouper"])
def test_strict_region_turns_suppress_errors_off(module, monkeypatch):
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", True)
    with module._strict_dynamo():
        assert torch._dynamo.config.suppress_errors is False
    assert torch._dynamo.config.suppress_errors is True


@pytest.mark.parametrize("module", [BC, PG], ids = ["block_compile", "prefix_grouper"])
def test_strict_region_runs_without_the_flag(module, monkeypatch):
    # A config without the flag, whose patch() raises the way torch's does for an unknown key.
    def patch(**changes):
        raise AttributeError(f"torch._dynamo.config.{next(iter(changes))} does not exist")

    monkeypatch.setattr(torch._dynamo, "config", types.SimpleNamespace(patch = patch))
    with module._strict_dynamo():
        pass


def test_no_unguarded_suppress_errors_patch_left():
    root = pathlib.Path(unsloth.__file__).parent
    hits = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if re.search(r"config\.patch\(\s*suppress_errors", p.read_text(encoding = "utf-8"))
        and "def _strict_dynamo" not in p.read_text(encoding = "utf-8")
    ]
    assert not hits, hits
