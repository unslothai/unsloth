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


def test_capture_is_rejected_on_mlx_before_the_torch_module_loads(monkeypatch):
    import sys
    import types
    from argparse import Namespace

    built = []
    fake = types.ModuleType("unsloth")
    fake._IS_MLX = True
    fake.FastLanguageModel = fake.RawTextDataLoader = object
    fake.is_bfloat16_supported = lambda: False
    capture = types.ModuleType("unsloth.activation_capture")
    capture.ActivationCaptureConfig = lambda **kw: built.append(kw)
    trl = types.ModuleType("trl")
    trl.SFTTrainer = trl.SFTConfig = object
    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda *a, **k: None
    tutils = types.ModuleType("transformers.utils")
    tutils.strtobool = bool
    for name, mod in {
        "unsloth": fake,
        "unsloth.activation_capture": capture,
        "trl": trl,
        "datasets": datasets,
        "transformers.utils": tutils,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)
    args = Namespace(
        capture_activations = True,
        capture_output_dir = "out",
        capture_interval = 1,
        capture_max_channels = 8,
    )
    with pytest.raises(SystemExit, match = "MLX"):
        _cli().run(args)
    assert built == []
