# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Under torchrun every rank runs the CLI; only global rank 0 may write the activation log."""

import importlib.util
from pathlib import Path

import pytest

_CLI = Path(__file__).resolve().parents[1] / "unsloth-cli.py"


def _cli():
    spec = importlib.util.spec_from_file_location("unsloth_cli_under_test", _CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("rank, writes", [(None, True), ("0", True), ("1", False), ("3", False)])
def test_only_global_rank_zero_writes_activation_capture(monkeypatch, rank, writes):
    if rank is None:
        monkeypatch.delenv("RANK", raising = False)
    else:
        monkeypatch.setenv("RANK", rank)
    assert _cli()._writes_activation_capture() is writes


def test_the_capture_block_is_gated_on_the_rank():
    source = _CLI.read_text(encoding = "utf-8")
    assert "if args.capture_activations and _writes_activation_capture():" in source
