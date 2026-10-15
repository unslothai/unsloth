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
            # flashoptim's own defaults carry initial_lr = the constructor lr.
            super().__init__(params, dict(lr = lr, initial_lr = lr, weight_decay = weight_decay))

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


def _flash_kwargs_args(**kwargs):
    defaults = dict(
        learning_rate = 1e-4, adam_beta1 = 0.9, adam_beta2 = 0.999, adam_epsilon = 1e-8,
        optim_args = None, fsdp = [], fsdp_config = {},
    )
    defaults.update(kwargs)
    return types.SimpleNamespace(**defaults)


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_restored_from_to_dict(stub_flashoptim, make_args, monkeypatch):
    import dataclasses
    import inspect
    from transformers import Trainer

    FlashAdamW, _ = stub_flashoptim
    saved = make_args(optim = "adamw_flash").to_dict()
    assert saved["optim"] == "adamw_torch"
    # A fresh process has the stock Trainer method until a marked config is built.
    patched = Trainer.__dict__["get_optimizer_cls_and_kwargs"].__func__
    original = inspect.getclosurevars(patched).nonlocals["original"]
    monkeypatch.setattr(Trainer, "get_optimizer_cls_and_kwargs", staticmethod(original))
    cls = type(make_args(optim = "adamw_torch"))
    init_fields = {f.name for f in dataclasses.fields(cls) if f.init}
    rebuilt = cls(**{k: v for k, v in saved.items() if k in init_fields and not k.endswith("_token")})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(rebuilt)
    assert issubclass(optimizer_cls, FlashAdamW)


def test_adamw_flash_keeps_per_group_initial_lr(stub_flashoptim, monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
        _flash_kwargs_args(optim_args = "decouple_lr=True")
    )
    groups = [
        {"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 1e-4},
        {"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 5e-6},
        {"params": [torch.nn.Parameter(torch.zeros(4))], "lr": 1.6e-3},
    ]
    optimizer = optimizer_cls(groups, **optimizer_kwargs)
    assert [g["initial_lr"] for g in optimizer.param_groups] == [1e-4, 5e-6, 1.6e-3]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    assert scheduler.base_lrs == [1e-4, 5e-6, 1.6e-3]


def test_adamw_flash_rejects_mixed_dtype_uncompressed_quantized(stub_flashoptim, monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    params = [
        torch.nn.Parameter(torch.zeros(4, dtype = torch.bfloat16)),
        torch.nn.Parameter(torch.zeros(4)),
    ]
    optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
        _flash_kwargs_args(optim_args = "quantize=True,compress_state_dict=False")
    )
    with pytest.raises(ValueError, match = "compress_state_dict"):
        optimizer_cls([{"params": params}], **optimizer_kwargs)
    for optim_args in (None, "quantize=True"):
        optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
            _flash_kwargs_args(optim_args = optim_args)
        )
        optimizer_cls([{"params": params}], **optimizer_kwargs)


@pytest.mark.parametrize(
    "fsdp_config, env, rejected",
    [
        ({"version": 1}, {}, True),
        ({"version": 2}, {}, False),
        (None, {"ACCELERATE_USE_FSDP": "true"}, True),
        (None, {"ACCELERATE_USE_FSDP": "true", "FSDP_VERSION": "2"}, False),
        (None, {}, False),
    ],
)
def test_adamw_flash_rejects_fsdp1(stub_flashoptim, monkeypatch, fsdp_config, env, rejected):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    args = _flash_kwargs_args(
        fsdp = ["full_shard"] if fsdp_config else [], fsdp_config = fsdp_config or {}
    )
    if rejected:
        with pytest.raises(ValueError, match = "FSDP1"):
            flash_adamw.get_flash_adamw_cls_and_kwargs(args)
    else:
        flash_adamw.get_flash_adamw_cls_and_kwargs(args)


@pytest.mark.parametrize("hip, platform", [("6.2.0", "linux"), (None, "win32")])
def test_adamw_flash_rejects_unsupported_platform(stub_flashoptim, monkeypatch, hip, platform):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", hip, raising = False)
    monkeypatch.setattr(sys, "platform", platform)
    with pytest.raises(NotImplementedError, match = "NVIDIA CUDA"):
        flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args())
