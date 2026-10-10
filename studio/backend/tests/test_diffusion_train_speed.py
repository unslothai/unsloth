# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Diffusion trainer speed defaults: optimizer factory, nf4 compile gate, checkpointing modes."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

import core.training.diffusion_dit_trainer as dit
import core.training.diffusion_lora_trainer as sdxl
import core.training.diffusion_train_common as dtc
from core.training import diffusion_checkpoint as dc
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    PermutationBatchSampler,
    make_lora_optimizer,
    recorded_optimizer_class,
    restore_resume_state,
    write_resume_checkpoint,
)

BNB_KEY = "bitsandbytes.optim.adamw.AdamW8bit"


class _Fake8bit(torch.optim.AdamW):
    """Stands in for bnb.optim.AdamW8bit."""


@pytest.fixture
def fake_bnb(monkeypatch):
    mod = ModuleType("bitsandbytes")
    optim = ModuleType("bitsandbytes.optim")
    optim.AdamW8bit = _Fake8bit
    mod.optim = optim
    monkeypatch.setitem(sys.modules, "bitsandbytes", mod)
    monkeypatch.setitem(sys.modules, "bitsandbytes.optim", optim)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FP32_OPTIM", raising = False)
    monkeypatch.setattr(dtc, "bitsandbytes_optimizer_supported", lambda: True)
    return mod


def _params():
    p = torch.nn.Parameter(torch.zeros(2, 2))
    p.grad = torch.ones(2, 2)
    return [p]


@pytest.mark.parametrize("factory", [dit._make_optimizer, sdxl._make_lora_optimizer])
def test_fresh_run_builds_torch_adamw_even_with_bitsandbytes(fake_bnb, factory):
    opt = factory(_params(), 1e-4)
    assert type(opt) is torch.optim.AdamW
    opt.step()


@pytest.mark.parametrize("factory", [dit._make_optimizer, sdxl._make_lora_optimizer])
def test_resuming_an_8bit_bundle_rebuilds_adamw8bit(fake_bnb, factory):
    opt = factory(_params(), 1e-4, BNB_KEY)
    assert isinstance(opt, _Fake8bit)
    assert type(factory(_params(), 1e-4, "torch.optim.adamw.AdamW")) is torch.optim.AdamW
    assert type(factory(_params(), 1e-4, None)) is torch.optim.AdamW


def test_8bit_resume_on_xpu_falls_back_to_torch_adamw(fake_bnb, monkeypatch):
    # XPU: refused later by restore_resume_state with a readable message.
    monkeypatch.setattr(dtc, "bitsandbytes_optimizer_supported", lambda: False)
    opt = make_lora_optimizer(_params(), 1e-4, BNB_KEY)
    assert type(opt) is torch.optim.AdamW
    opt.step()


def test_fp32_override_still_forces_plain_adamw(fake_bnb, monkeypatch):
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FP32_OPTIM", "1")
    for resume in (None, BNB_KEY):
        opt = make_lora_optimizer(_params(), 1e-4, resume)
        assert type(opt) is torch.optim.AdamW
        assert not opt.defaults.get("fused")


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fused AdamW needs CUDA")
def test_cuda_default_is_fused(monkeypatch):
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FP32_OPTIM", raising = False)
    p = torch.nn.Parameter(torch.zeros(4, 4, device = "cuda"))
    opt = make_lora_optimizer([p], 1e-4)
    assert type(opt) is torch.optim.AdamW and opt.defaults.get("fused") is True


def _identity() -> dc.CheckpointIdentity:
    return dc.CheckpointIdentity(
        family = "sdxl",
        base_model = "stabilityai/sdxl-turbo",
        lora_target_modules = ("to_k", "to_q", "to_v", "to_out.0"),
        lora_rank = 16,
        lora_alpha = 16,
        precision = "bf16",
        base_precision = "nf4",
        resolution = 1024,
        base_revision = "rev-deadbeef",
        dataset_fingerprint = "ds-3-cafe",
    )


def _cfg(out: Path, **kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(
        base_model = "stabilityai/sdxl-turbo",
        data_dir = "unused",
        output_dir = str(out),
        train_steps = 500,
        **kw,
    )


@pytest.fixture
def run_dir():
    from utils.paths import outputs_root

    d = outputs_root() / "speed-run"
    d.mkdir(parents = True, exist_ok = True)
    return d


def _write(
    out: Path,
    model,
    optimizer,
    step: int = 3,
):
    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: 1.0)
    written, error = write_resume_checkpoint(
        _cfg(out),
        step = step,
        model = model,
        optimizer = optimizer,
        lr_scheduler = sched,
        identity = _identity(),
        sampler = PermutationBatchSampler(7, random.Random(0)),
        rng_streams = {"loop": random.Random(0), "variant": random.Random(1)},
    )
    assert error is None and written
    return sched


def test_recorded_optimizer_class_reads_the_bundle_being_resumed(run_dir):
    tmp_path = run_dir
    model = torch.nn.Linear(4, 4, bias = False)
    _write(tmp_path, model, torch.optim.AdamW(model.parameters(), lr = 1e-3))
    cfg = _cfg(tmp_path, resume_from_checkpoint = str(tmp_path / "checkpoint-3"))
    assert recorded_optimizer_class(cfg, _identity()) == "torch.optim.adamw.AdamW"
    manifest_path = tmp_path / "checkpoint-3" / dc.TRAINER_STATE_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding = "utf-8"))
    manifest["optimizer_class"] = BNB_KEY
    manifest_path.write_text(json.dumps(manifest), encoding = "utf-8")
    assert recorded_optimizer_class(cfg, _identity()) == BNB_KEY
    assert recorded_optimizer_class(_cfg(tmp_path), _identity()) is None
    bogus = _cfg(tmp_path, resume_from_checkpoint = str(tmp_path / "nope"))
    assert recorded_optimizer_class(bogus, _identity()) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "fused AdamW needs CUDA")
def test_a_fused_bundle_resumed_under_the_fp32_override_stays_non_fused(run_dir, monkeypatch):
    # load_state_dict restores the saved fused flag; the reference-optimizer override must survive it.
    model = torch.nn.Linear(4, 4, bias = False).cuda()
    fused = torch.optim.AdamW(model.parameters(), lr = 1e-3, fused = True)
    model.weight.grad = torch.ones_like(model.weight)
    fused.step()
    _write(run_dir, model, fused)
    monkeypatch.setenv("UNSLOTH_DIFFUSION_FP32_OPTIM", "1")
    live = torch.nn.Linear(4, 4, bias = False).cuda()
    opt = make_lora_optimizer(list(live.parameters()), 1e-3)
    assert not opt.param_groups[0].get("fused")
    restore_resume_state(
        _cfg(run_dir, resume_from_checkpoint = str(run_dir / "checkpoint-3")),
        model = live,
        optimizer = opt,
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
        identity = _identity(),
    )
    assert not opt.param_groups[0].get("fused")
    assert opt.state[live.weight]["exp_avg"].abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "bitsandbytes AdamW8bit needs CUDA")
def test_old_8bit_bundle_resumes_and_continues_identically(run_dir, monkeypatch):
    tmp_path = run_dir
    bnb = pytest.importorskip("bitsandbytes")
    if not hasattr(getattr(bnb, "optim", None), "AdamW8bit"):
        pytest.skip("bitsandbytes is stubbed out in this environment")
    monkeypatch.delenv("UNSLOTH_DIFFUSION_FP32_OPTIM", raising = False)
    monkeypatch.setattr(dtc, "bitsandbytes_optimizer_supported", lambda: True)

    def _model():
        torch.manual_seed(0)
        # Above bitsandbytes' min_8bit_size, so the moments really are 8-bit.
        return torch.nn.Linear(128, 64, bias = False).cuda()

    def _step(model, opt, i):
        opt.zero_grad(set_to_none = True)
        model.weight.grad = torch.full_like(model.weight, 0.01 * (i + 1))
        opt.step()

    ref = _model()
    ref_opt = bnb.optim.AdamW8bit(ref.parameters(), lr = 1e-3)
    for i in range(3):
        _step(ref, ref_opt, i)
    _write(tmp_path, ref, ref_opt)

    cfg = _cfg(tmp_path, resume_from_checkpoint = str(tmp_path / "checkpoint-3"))
    live = _model()
    opt = make_lora_optimizer(
        list(live.parameters()), 1e-3, recorded_optimizer_class(cfg, _identity())
    )
    assert dc.optimizer_key(opt) == BNB_KEY
    restored = restore_resume_state(
        cfg,
        model = live,
        optimizer = opt,
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 1.0),
        identity = _identity(),
    )
    assert restored.step == 3
    for i in range(3, 6):
        _step(ref, ref_opt, i)
        _step(live, opt, i)
        assert torch.equal(live.weight, ref.weight)


def _ccfg(mode = "auto"):
    return DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image",
        data_dir = "unused",
        output_dir = "unused",
        compile_transformer = mode,
    )


def test_auto_compiles_nf4_only_when_the_versions_are_supported(monkeypatch):
    monkeypatch.setattr(dit, "_nf4_compile_supported", lambda: True)
    assert dit._should_compile(_ccfg(), True, "cuda", "nf4") is True
    monkeypatch.setattr(dit, "_nf4_compile_supported", lambda: False)
    assert dit._should_compile(_ccfg(), True, "cuda", "nf4") is False
    assert dit._should_compile(_ccfg("on"), True, "cuda", "nf4") is True
    assert dit._should_compile(_ccfg("off"), True, "cuda", "nf4") is False
    monkeypatch.setattr(dit, "_nf4_compile_supported", lambda: True)
    assert dit._should_compile(_ccfg(), True, "xpu", "nf4") is False
    assert dit._should_compile(_ccfg(), True, "cpu", "nf4") is False


@pytest.mark.parametrize(
    "torch_version, bnb_version, rocm, expected",
    [
        ("2.13.0+cu130", "0.50.2", False, True),
        ("2.10.0+cu128", "0.46.1", False, True),
        ("2.9.1+cu128", "0.50.2", False, False),
        ("2.13.0", "0.45.5", False, False),
        ("2.13.0", "0.46.0", False, False),
        ("2.13.0+rocm7.0", "0.50.2", True, False),
        ("2.11.0.dev20260101+cu130", "0.49.0.dev0", False, True),
        ("2.10.0rc1", "0.50.2", False, False),
        ("2.13.0", "0.46.1rc1", False, False),
        ("2.10.0.dev20250901+cu128", "0.50.2", False, False),
    ],
)
def test_nf4_compile_version_floor(monkeypatch, torch_version, bnb_version, rocm, expected):
    import importlib.metadata

    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(dit, "torch_is_rocm", lambda: rocm)
    real = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: bnb_version if name == "bitsandbytes" else real(name),
    )
    assert dit._nf4_compile_supported() is expected


def test_nf4_compile_gate_fails_closed_without_bitsandbytes(monkeypatch):
    import importlib.metadata

    def _missing(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _missing)
    assert dit._nf4_compile_supported() is False


def _tiny_zimage():
    diffusers = pytest.importorskip("diffusers")
    peft = pytest.importorskip("peft")
    torch.manual_seed(0)
    model = diffusers.ZImageTransformer2DModel(
        dim = 64,
        n_layers = 4,
        n_refiner_layers = 1,
        n_heads = 2,
        n_kv_heads = 2,
        cap_feat_dim = 32,
        axes_dims = [8, 12, 12],
        axes_lens = [64, 32, 32],
    )
    model.requires_grad_(False)
    model.add_adapter(
        peft.LoraConfig(r = 4, lora_alpha = 4, init_lora_weights = False, target_modules = ["to_q", "to_v"])
    )
    return model


def _lora_grads(model):
    torch.manual_seed(1)
    x, cap, t = [torch.randn(16, 1, 8, 8)], [torch.randn(6, 32)], torch.tensor([0.3])
    model.train()
    out = model(x, t, cap, return_dict = False)[0]
    sum((o**2).mean() for o in out).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}


@pytest.mark.parametrize("mode", ["plain", "partial"])
def test_checkpoint_modes_match_uncheckpointed_grads(mode, monkeypatch):
    monkeypatch.delenv(dtc.GC_MODE_ENV, raising = False)
    reference = _lora_grads(_tiny_zimage())
    model = _tiny_zimage()
    calls = []
    real_checkpoint = torch.utils.checkpoint.checkpoint
    monkeypatch.setattr(
        torch.utils.checkpoint,
        "checkpoint",
        lambda fn, *a, **k: calls.append(k.get("use_reentrant")) or real_checkpoint(fn, *a, **k),
    )
    assert dtc.enable_diffusion_gradient_checkpointing(model, mode) == mode
    grads = _lora_grads(model)
    assert grads.keys() == reference.keys()
    for name in reference:
        torch.testing.assert_close(grads[name], reference[name], rtol = 1e-5, atol = 1e-6)
    # 6 ZImageTransformerBlocks (4 layers + 1 noise + 1 context refiner): partial skips every 2nd.
    assert calls and set(calls) == {False}
    assert len(calls) == (6 if mode == "plain" else 3)


def test_checkpoint_mode_comes_from_the_env_and_defaults_to_plain(monkeypatch):
    seen = {}

    class _Model(torch.nn.Module):
        _repeated_blocks = ["Linear"]

        def __init__(self):
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])

        def enable_gradient_checkpointing(self, gradient_checkpointing_func = None):
            seen["func"] = gradient_checkpointing_func

    monkeypatch.delenv(dtc.GC_MODE_ENV, raising = False)
    assert dtc.enable_diffusion_gradient_checkpointing(_Model()) == "plain"
    for value, expected in (("partial", "partial"), ("PARTIAL ", "partial"), ("bogus", "plain")):
        monkeypatch.setenv(dtc.GC_MODE_ENV, value)
        assert dtc.enable_diffusion_gradient_checkpointing(_Model()) == expected
        assert callable(seen["func"])
    monkeypatch.setattr(_Model, "_repeated_blocks", [])
    monkeypatch.setenv(dtc.GC_MODE_ENV, "partial")
    assert dtc.enable_diffusion_gradient_checkpointing(_Model()) == "plain"
