# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys
import types

import pytest

import unsloth  # noqa: F401  (must precede transformers/trl)
import torch

OUT = os.path.join(os.environ.get("TMPDIR", "."), "unsloth-flash-adamw-test")


@pytest.fixture
def stub_flashoptim(monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    calls = []

    class FlashAdamW(torch.optim.Optimizer):
        def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-8,
                     weight_decay = 1e-2, master_weight_bits = 24, **kwargs):
            calls.append(dict(lr = lr, betas = betas, eps = eps,
                              master_weight_bits = master_weight_bits, **kwargs))
            super().__init__(params, dict(lr = lr, weight_decay = weight_decay))

    module = types.ModuleType("flashoptim")
    module.FlashAdamW = FlashAdamW
    monkeypatch.setitem(sys.modules, "flashoptim", module)
    monkeypatch.setattr(flash_adamw, "_FLASH_ADAMW_CLASS", None)
    yield FlashAdamW, calls
    flash_adamw._FLASH_ADAMW_CLASS = None


def _sft_config(**kwargs):
    from trl import SFTConfig

    return SFTConfig(output_dir = OUT, report_to = [], **kwargs)


def _unsloth_args(**kwargs):
    from unsloth.trainer import UnslothTrainingArguments

    return UnslothTrainingArguments(output_dir = OUT, report_to = [], **kwargs)


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_selects_flashoptim(stub_flashoptim, make_args, monkeypatch):
    from transformers import Trainer

    FlashAdamW, calls = stub_flashoptim
    args = make_args(
        optim = "adamw_flash",
        learning_rate = 3e-4,
        adam_beta2 = 0.95,
        optim_args = "master_weight_bits=24, quantize=True",
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(args)
    assert issubclass(optimizer_cls, FlashAdamW)
    assert optimizer_kwargs["lr"] == 3e-4
    assert optimizer_kwargs["betas"] == (0.9, 0.95)
    assert optimizer_kwargs["master_weight_bits"] == 24
    assert optimizer_kwargs["quantize"] is True

    bf16 = torch.nn.Parameter(torch.zeros(4, dtype = torch.bfloat16))
    optimizer_cls([{"params": [bf16], "weight_decay": 0.1}], **optimizer_kwargs)
    assert calls[-1]["master_weight_bits"] == 24
    # flashoptim rejects master_weight_bits on all-fp32 params (LoRA adapters).
    fp32 = torch.nn.Parameter(torch.zeros(4))
    optimizer_cls([{"params": [fp32], "weight_decay": 0.0}], **optimizer_kwargs)
    assert calls[-1]["master_weight_bits"] is None


def test_adamw_flash_missing_dependency(monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw
    from transformers import Trainer

    monkeypatch.setitem(sys.modules, "flashoptim", None)
    monkeypatch.setattr(flash_adamw, "_FLASH_ADAMW_CLASS", None)
    args = _sft_config(optim = "adamw_flash")
    with pytest.raises(ImportError, match = "pip install flashoptim"):
        Trainer.get_optimizer_cls_and_kwargs(args)


def test_other_optimizers_untouched():
    from transformers import Trainer

    args = _sft_config(optim = "adamw_torch")
    assert not getattr(args, "_unsloth_flash_adamw", False)
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(args)
    assert optimizer_cls is torch.optim.AdamW


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_survives_dataclasses_replace(stub_flashoptim, make_args, monkeypatch):
    import dataclasses
    from transformers import Trainer

    FlashAdamW, _ = stub_flashoptim
    args = make_args(optim = "adamw_flash")
    copied = dataclasses.replace(args, learning_rate = 1e-5)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(copied)
    assert issubclass(optimizer_cls, FlashAdamW)
    assert optimizer_kwargs["lr"] == 1e-5
    # Default: full 32 bit master semantics and unquantized moments.
    assert optimizer_kwargs["master_weight_bits"] == 32
    assert optimizer_kwargs["quantize"] is False


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_marker_parses_in_transformers(make_args):
    trainer_module = __import__("transformers.trainer", fromlist = ["_parse_optim_args"])
    parse = getattr(trainer_module, "_parse_optim_args", None)
    if parse is None:
        pytest.skip("transformers has no _parse_optim_args")
    args = make_args(optim = "adamw_flash", optim_args = "quantize=True")
    parsed = parse(args.optim_args)
    assert parsed["quantize"] == "True"
