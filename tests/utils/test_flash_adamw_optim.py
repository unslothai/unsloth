# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

import os
import sys
import types

import pytest

from real_accelerator import has_real_cuda  # tests/_shared, on sys.path via tests/conftest.py
import unsloth  # noqa: F401  (must precede transformers/trl)
import torch

OUT = os.path.join(os.environ.get("TMPDIR", "."), "unsloth-flash-adamw-test")


@pytest.fixture
def stub_flashoptim(monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    calls = []

    class FlashAdamW(torch.optim.Optimizer):
        def __init__(
            self,
            params,
            lr = 1e-3,
            betas = (0.9, 0.999),
            eps = 1e-8,
            weight_decay = 1e-2,
            decouple_lr = False,
            quantize = True,
            compress_state_dict = True,
            master_weight_bits = 24,
            **kwargs,
        ):
            calls.append(
                dict(lr = lr, betas = betas, eps = eps, master_weight_bits = master_weight_bits, **kwargs)
            )
            if quantize is not True:
                calls[-1]["quantize"] = quantize
            if compress_state_dict is not True:
                calls[-1]["compress_state_dict"] = compress_state_dict
            # flashoptim 0.1.4's own defaults carry initial_lr = the constructor lr and quantize,
            # and a group without the key falls back to self._quantize.
            self._quantize = quantize
            super().__init__(
                params, dict(lr = lr, initial_lr = lr, weight_decay = weight_decay, quantize = quantize)
            )

        # flashoptim: step() calls step_param() for every parameter with a gradient, and
        # gradient release calls step_param() directly.
        def step(self, closure = None):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        self.step_param(p, group)

        def step_param(
            self,
            p,
            group = None,
        ):
            return None

    module = types.ModuleType("flashoptim")
    module.FlashAdamW = FlashAdamW
    module.__version__ = "0.1.4"
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = _sft_config(optim = "adamw_flash")
    with pytest.raises(ImportError, match = "pip install 'flashoptim"):
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
        learning_rate = 1e-4,
        adam_beta1 = 0.9,
        adam_beta2 = 0.999,
        adam_epsilon = 1e-8,
        optim_args = None,
        fsdp = [],
        fsdp_config = {},
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
    rebuilt = cls(
        **{k: v for k, v in saved.items() if k in init_fields and not k.endswith("_token")}
    )
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
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION", "FSDP_OFFLOAD_PARAMS"):
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


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_later_optim_change_wins(stub_flashoptim, make_args, monkeypatch):
    import dataclasses
    from transformers import Trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = make_args(optim = "adamw_flash", optim_args = "quantize=True")
    replaced = dataclasses.replace(args, optim = "rmsprop")
    assert "unsloth_adamw_flash" not in str(replaced.optim_args)
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(replaced)
    assert optimizer_cls is torch.optim.RMSprop
    args.optim = "sgd"
    args.optim_args = None
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(args)
    assert optimizer_cls is torch.optim.SGD


def test_adamw_flash_keeps_one_shot_param_iterables(stub_flashoptim, monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    model = torch.nn.Linear(4, 4)
    optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
        _flash_kwargs_args()
    )
    optimizer = optimizer_cls([{"params": model.parameters(), "lr": 1e-4}], **optimizer_kwargs)
    assert len(optimizer.param_groups[0]["params"]) == 2
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)
    assert len(optimizer.param_groups[0]["params"]) == 2


@pytest.mark.parametrize(
    "fsdp, fsdp_config, env",
    [
        (["full_shard"], {"version": 2, "cpu_offload": True}, {}),
        ("full_shard offload", {"version": 2}, {}),
        (
            [],
            None,
            {"ACCELERATE_USE_FSDP": "true", "FSDP_VERSION": "2", "FSDP_OFFLOAD_PARAMS": "true"},
        ),
    ],
)
def test_adamw_flash_rejects_fsdp2_cpu_offload(
    stub_flashoptim, monkeypatch, fsdp, fsdp_config, env
):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for key in ("ACCELERATE_USE_FSDP", "FSDP_VERSION", "FSDP_OFFLOAD_PARAMS"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    args = _flash_kwargs_args(fsdp = fsdp, fsdp_config = fsdp_config or {})
    with pytest.raises(ValueError, match = "CPU offloaded"):
        flash_adamw.get_flash_adamw_cls_and_kwargs(args)


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_accepted_by_hf_argument_parser(make_args):
    from transformers import HfArgumentParser

    cls = type(make_args(optim = "adamw_torch"))
    (args,) = HfArgumentParser(cls).parse_args_into_dataclasses(
        ["--output_dir", OUT, "--report_to", "none", "--optim", "adamw_flash"]
    )
    assert getattr(args, "_unsloth_flash_adamw", False)
    with pytest.raises(SystemExit):
        HfArgumentParser(cls).parse_args_into_dataclasses(
            ["--output_dir", OUT, "--report_to", "none", "--optim", "not_an_optim"]
        )


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_switch_back_to_adamw(stub_flashoptim, make_args, monkeypatch):
    import dataclasses
    from transformers import Trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = make_args(optim = "adamw_flash")
    replaced = dataclasses.replace(args, optim = "adamw_torch", optim_args = None)
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(replaced)
    assert optimizer_cls is torch.optim.AdamW


@pytest.mark.parametrize("make_args", [_sft_config, _unsloth_args])
def test_adamw_flash_switch_back_in_place(stub_flashoptim, make_args, monkeypatch):
    from transformers import Trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    args = make_args(optim = "adamw_flash")
    args.optim = "adamw_torch"
    args.optim_args = None
    optimizer_cls, _ = Trainer.get_optimizer_cls_and_kwargs(args)
    assert optimizer_cls is torch.optim.AdamW


def test_adamw_flash_platform_checked_before_import(monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setitem(sys.modules, "flashoptim", None)
    monkeypatch.setattr(flash_adamw, "_FLASH_ADAMW_CLASS", None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(NotImplementedError, match = "NVIDIA CUDA"):
        flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args())


@pytest.mark.parametrize(
    "deepspeed, env",
    [
        ({"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}}}, {}),
        ({"zero_optimization": {"stage": 3, "offload_optimizer": {"device": "nvme"}}}, {}),
        (
            None,
            {
                "ACCELERATE_USE_DEEPSPEED": "true",
                "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE": "cpu",
            },
        ),
    ],
)
def test_adamw_flash_rejects_deepspeed_optimizer_offload(
    stub_flashoptim, monkeypatch, deepspeed, env
):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for key in ("ACCELERATE_USE_DEEPSPEED", "ACCELERATE_DEEPSPEED_OFFLOAD_OPTIMIZER_DEVICE"):
        monkeypatch.delenv(key, raising = False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError, match = "offload_optimizer"):
        flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args(deepspeed = deepspeed))
    no_offload = {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}}}
    monkeypatch.delenv("ACCELERATE_USE_DEEPSPEED", raising = False)
    flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args(deepspeed = no_offload))


@pytest.mark.parametrize(
    "optim_args", ["quantize=ture", "compress_state_dict=tru", "check_numerics=2"]
)
def test_adamw_flash_rejects_invalid_booleans(optim_args):
    import unsloth.optimizers.flash_adamw as flash_adamw
    with pytest.raises(ValueError, match = "not a boolean"):
        flash_adamw._parse_optim_args(optim_args)
    assert flash_adamw._parse_optim_args("quantize=False,decouple_lr=no") == {
        "quantize": False,
        "decouple_lr": False,
    }


def test_adamw_flash_requires_compatible_flashoptim(stub_flashoptim, monkeypatch):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys.modules["flashoptim"], "__version__", "0.1.1")
    with pytest.raises(ImportError, match = "flashoptim>=0.1.4, found 0.1.1"):
        flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args())


def test_adamw_flash_rejects_fp16_params_with_unquantized_moments(stub_flashoptim, monkeypatch):
    # flashoptim stores unquantized moments in the param dtype: an fp16 variance underflows to 0
    # for |grad| < ~2e-4, and the next step divides the momentum by eps alone.
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    fp16 = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))
    fp32 = torch.nn.Parameter(torch.zeros(4))
    bf16 = torch.nn.Parameter(torch.zeros(4, dtype = torch.bfloat16))
    for optim_args in (None, "quantize=False", "master_weight_bits=none"):
        optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
            _flash_kwargs_args(optim_args = optim_args)
        )
        # bf16 is the suggested dtype because its variance is widened to fp32.
        with pytest.raises(ValueError, match = r"bfloat16 \(its variance is kept in float32\)"):
            optimizer_cls([{"params": [fp32, fp16]}], **optimizer_kwargs)
        # bf16 / fp32 keep working with unquantized moments.
        optimizer_cls([{"params": [fp32, bf16]}], **optimizer_kwargs)
    # quantize=True stores the variance as sqrt with its own scales, so fp16 is accepted.
    optimizer_cls, optimizer_kwargs = flash_adamw.get_flash_adamw_cls_and_kwargs(
        _flash_kwargs_args(optim_args = "quantize=True")
    )
    optimizer_cls([{"params": [fp32, fp16]}], **optimizer_kwargs)


def test_adamw_flash_fp16_check_covers_positional_late_and_unfrozen_groups(
    stub_flashoptim, monkeypatch
):
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    optimizer_cls, _ = flash_adamw.get_flash_adamw_cls_and_kwargs(_flash_kwargs_args())
    fp32 = torch.nn.Parameter(torch.zeros(4))
    fp16 = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))
    # quantize passed positionally (lr, betas, eps, weight_decay, decouple_lr, quantize).
    with pytest.raises(ValueError, match = "float16"):
        optimizer_cls([fp32, fp16], 1e-3, (0.9, 0.999), 1e-8, 0.0, False, False)
    # A group added after construction (progressive unfreezing) is checked too.
    optimizer = optimizer_cls([fp32], quantize = False)
    with pytest.raises(ValueError, match = "float16"):
        optimizer.add_param_group({"params": [fp16]})
    # A frozen fp16 base (fp32 LoRA on top) is fine until it is unfrozen and gets a gradient.
    frozen = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16), requires_grad = False)
    optimizer = optimizer_cls([fp32, frozen], quantize = False)
    fp32.grad = torch.ones(4)
    optimizer.step()
    frozen.requires_grad_(True)
    frozen.grad = torch.ones(4, dtype = torch.float16)
    with pytest.raises(ValueError, match = "float16"):
        optimizer.step()
    # Gradient release updates through step_param directly, without step().
    with pytest.raises(ValueError, match = "float16"):
        optimizer.step_param(frozen)
    # Frozen again after backward: .grad stays attached, but flashoptim skips the param.
    frozen.requires_grad_(False)
    optimizer.step()
    optimizer.step_param(frozen)
    # Resuming a checkpoint whose group had quantize=False switches an accepted fp16 group.
    trainable = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))
    optimizer = optimizer_cls([trainable], quantize = True)
    saved = optimizer.state_dict()
    saved["param_groups"][0]["quantize"] = False
    optimizer.load_state_dict(saved)
    trainable.grad = torch.ones(4, dtype = torch.float16)
    with pytest.raises(ValueError, match = "float16"):
        optimizer.step()
    # A vanilla AdamW state dict has no quantize key: the optimizer-wide quantize=False applies.
    trainable = torch.nn.Parameter(torch.zeros(4, dtype = torch.float16))
    optimizer = optimizer_cls([trainable], quantize = True)
    saved = optimizer.state_dict()
    saved["param_groups"][0].pop("quantize")
    optimizer = optimizer_cls([trainable], quantize = True)
    optimizer._quantize = False
    optimizer.load_state_dict(saved)
    trainable.grad = torch.ones(4, dtype = torch.float16)
    with pytest.raises(ValueError, match = "float16"):
        optimizer.step()
    # A refused group is not kept, so the caller can add the params again with quantize=True.
    optimizer = optimizer_cls([fp32], quantize = False)
    with pytest.raises(ValueError, match = "float16"):
        optimizer.add_param_group({"params": [fp16]})
    assert len(optimizer.param_groups) == 1
    optimizer.add_param_group({"params": [fp16], "quantize": True})
    assert len(optimizer.param_groups) == 2
    fp16.grad = torch.ones(4, dtype = torch.float16)
    optimizer.step()


def _real_flash_adamw(monkeypatch):
    pytest.importorskip("flashoptim")
    import unsloth.optimizers.flash_adamw as flash_adamw

    monkeypatch.setattr(flash_adamw, "_FLASH_ADAMW_CLASS", None)
    return flash_adamw._get_flash_adamw_class()


def _scaled_noise(step, drop_at):
    # The gradient scale falls 10x at drop_at, as when a finetune moves past its early large
    # gradients. An fp32 EMA follows it down; a variance that cannot decay stays at its peak.
    return torch.randn(1024) * (1.0 if step < drop_at else 0.1)


def _steady(step, drop_at):
    # A steady gradient: the fp32 EMA rises to grad**2, a bf16 one stalls near a third of it.
    return torch.full((1024,), 0.01)


def _train(
    cls,
    device,
    gradient = _scaled_noise,
    steps = 2000,
    drop_at = 500,
    **kwargs,
):
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.zeros(1024, dtype = torch.bfloat16, device = device))
    opt = cls([p], lr = 1e-4, weight_decay = 0.0, quantize = False, **kwargs)
    ema = torch.zeros(1024, device = device)
    for step in range(steps):
        grad = gradient(step, drop_at).to(device = device, dtype = torch.bfloat16)
        ema.mul_(0.999).addcmul_(grad.float(), grad.float(), value = 0.001)
        p.grad = grad
        opt.step()
    return opt, p, ema


@pytest.mark.parametrize("gradient", [_scaled_noise, _steady])
def test_adamw_flash_bf16_variance_follows_fp32(monkeypatch, gradient):
    # With bf16 storage and round-to-nearest the variance froze near 0.39 after the drop
    # (fp32: 0.096) and stalled at 0.31x of grad**2 under the steady gradient: at beta2 =
    # 0.999 a step moves it by at most 0.1% of the gap, under half a bf16 ulp.
    cls = _real_flash_adamw(monkeypatch)
    opt, p, ema = _train(cls, "cpu", gradient, master_weight_bits = None, fused = False)
    var = opt.state[p]["exp_avg_sq"].materialize()
    assert var.dtype == torch.float32
    torch.testing.assert_close(var, ema, rtol = 1e-5, atol = 1e-12)


def test_adamw_flash_bf16_variance_survives_a_state_dict_round_trip(monkeypatch):
    cls = _real_flash_adamw(monkeypatch)
    opt, p, _ = _train(cls, "cpu", steps = 50, drop_at = 25, master_weight_bits = None, fused = False)
    saved = opt.state[p]["exp_avg_sq"].materialize().clone()
    fresh = cls(
        [p], lr = 1e-4, weight_decay = 0.0, quantize = False, master_weight_bits = None, fused = False
    )
    fresh.load_state_dict(opt.state_dict())
    loaded = fresh.state[p]["exp_avg_sq"].materialize()
    assert loaded.dtype == torch.float32
    torch.testing.assert_close(loaded, saved, rtol = 0, atol = 0)


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda(), reason = "flashoptim's fused Triton step needs CUDA")
@pytest.mark.parametrize("gradient", [_scaled_noise, _steady])
def test_adamw_flash_fused_bf16_variance_follows_fp32(monkeypatch, gradient):
    # The fused kernel stores its variance update rounded to the param dtype; the fp32 copy
    # computed beside it is what is kept.
    cls = _real_flash_adamw(monkeypatch)
    opt, p, ema = _train(cls, "cuda", gradient, master_weight_bits = 32)
    var = opt.state[p]["exp_avg_sq"].materialize()
    assert var.dtype == torch.float32
    # fp32 rounding differs between the kernel and the torch EMA (about 2e-5 after 2000 steps).
    torch.testing.assert_close(var, ema, rtol = 1e-4, atol = 1e-12)
