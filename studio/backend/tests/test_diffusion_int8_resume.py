# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Resuming an int8 LoRA keeps the frozen-base kernel its bundle trained on, through real bundles and the real resume
preflight: a bundle written before base_precision_effective / base_int8_kernel existed trained torchao weight-only and
must resume weight-only (trainer gate and block swap alike), not switch to the W8A8 default."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
import torch

from core.training import diffusion_block_swap as bs
from core.training import diffusion_checkpoint as dc
from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_train_common import (
    DiffusionLoraConfig,
    PermutationBatchSampler,
    write_resume_checkpoint,
)

INT8_ENV = "UNSLOTH_DIFFUSION_INT8_TRAIN"


def _identity(**overrides) -> dc.CheckpointIdentity:
    fields = {
        "family": "qwen-image-2.1",
        "base_model": "Qwen/Qwen-Image-2.1",
        "lora_target_modules": ("to_k", "to_q", "to_v", "to_out.0"),
        "lora_rank": 16,
        "lora_alpha": 16,
        "precision": "bf16",
        "base_precision": "int8",
        "resolution": 512,
        "base_revision": "rev-deadbeef",
        "dataset_fingerprint": "ds-16-cafe",
    }
    fields.update(overrides)
    return dc.CheckpointIdentity(**fields)


@pytest.fixture(autouse = True)
def _healthy_diffusers(healthy_diffusers):
    """The resume preflight asserts the family's pipeline class; these tests are about the int8 kernel."""


@pytest.fixture
def run_dir(monkeypatch):
    from utils.paths import outputs_root

    monkeypatch.delenv(INT8_ENV, raising = False)
    d = outputs_root() / "int8-resume-run"
    d.mkdir(parents = True, exist_ok = True)
    yield d
    import shutil

    shutil.rmtree(d, ignore_errors = True)


def _write_bundle(
    run_dir: Path,
    identity: dc.CheckpointIdentity,
    drop: tuple = (),
) -> Path:
    """A real bundle (adapter tensor, AdamW state, LR schedule, sampler, RNG streams); ``drop`` removes identity
    fields from the written manifest, as a bundle from before those fields existed has them."""
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 4, bias = False)
    opt = torch.optim.AdamW(model.parameters(), lr = 1e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda s: 1.0)
    rng = __import__("random").Random(0)
    variant = __import__("random").Random(1)
    cfg = DiffusionLoraConfig(
        base_model = "Qwen/Qwen-Image-2.1",
        data_dir = "unused",
        output_dir = str(run_dir),
        train_steps = 500,
    )
    model.weight.grad = torch.ones_like(model.weight)
    opt.step()
    path, err = write_resume_checkpoint(
        cfg,
        step = 3,
        model = model,
        optimizer = opt,
        lr_scheduler = sched,
        identity = identity,
        sampler = PermutationBatchSampler(7, rng),
        rng_streams = {"loop": rng, "variant": variant},
    )
    assert err is None and path, err
    manifests = list(Path(path).glob("*.json"))
    hit = False
    for m in manifests:
        data = json.loads(m.read_text())
        if isinstance(data, dict) and isinstance(data.get("identity"), dict):
            for key in drop:
                data["identity"].pop(key, None)
            m.write_text(json.dumps(data))
            hit = True
    assert hit, f"no manifest with an identity in {manifests}"
    return Path(path)


def _resume_cfg(run_dir, **kw):
    return DiffusionLoraConfig(
        base_model = "Qwen/Qwen-Image-2.1",
        model_family = "qwen-image-2.1",
        data_dir = "unused",
        output_dir = str(run_dir),
        train_steps = 500,
        base_precision = "int8",
        resume_from_checkpoint = str(run_dir),
        **kw,
    )


@pytest.mark.parametrize(
    "case, kernel, drop, want_w8a8",
    [
        ("legacy: neither record", None, ("base_precision_effective", "base_int8_kernel"), False),
        ("effective int8, no kernel record", None, ("base_int8_kernel",), False),
        ("recorded weight-only", "weight-only", (), False),
        ("recorded W8A8", "w8a8", (), True),
    ],
)
def test_an_int8_resume_keeps_the_kernel_its_bundle_trained_on(
    run_dir, case, kernel, drop, want_w8a8
):
    saved = _identity(base_precision_effective = "int8")
    if kernel is not None:
        saved = dc.with_int8_kernel(saved, kernel == "w8a8")
    _write_bundle(run_dir, saved, drop)
    # The real resume preflight accepts the bundle for an int8 run identity (missing fields are not mismatches).
    incoming = dataclasses.replace(
        _identity(), base_precision_effective = None, base_int8_kernel = None
    )
    path, step = dc.preflight_resume(str(run_dir), identity = incoming, target_steps = 500)
    assert step == 3 and Path(path).name == "checkpoint-3"
    assert dc.recorded_int8_resume_kernel(str(run_dir)) == ("w8a8" if want_w8a8 else "weight-only")
    cfg = _resume_cfg(run_dir)
    assert dit._int8_w8a8(cfg) is want_w8a8, case
    # Block swap reads the same decision: a weight-only bundle does not stream as W8A8.
    assert bs.swap_lowp_mode("int8", cfg) == ("int8_w8a8" if want_w8a8 else None)
    err = bs.block_swap_error("auto", "int8", "cuda", True, cfg = cfg)
    assert (err is None) is want_w8a8
    if not want_w8a8:
        assert "weight-only" in err
    # The identity the run then records matches the bundle's kernel, so validation accepts it.
    ran = dc.with_int8_kernel(
        dataclasses.replace(_identity(), base_precision_effective = "int8"), want_w8a8
    )
    dc.preflight_resume(str(run_dir), identity = ran, target_steps = 500)


def test_a_non_int8_bundle_says_nothing_about_the_int8_kernel(run_dir):
    _write_bundle(run_dir, _identity(base_precision = "bf16", base_precision_effective = "bf16"))
    assert dc.recorded_int8_resume_kernel(str(run_dir)) is None
    assert dit._int8_w8a8(_resume_cfg(run_dir)) is True


def test_the_env_still_overrides_a_recorded_kernel(run_dir, monkeypatch):
    _write_bundle(run_dir, _identity(base_precision_effective = "int8"), ("base_int8_kernel",))
    monkeypatch.setenv(INT8_ENV, "w8a8")
    assert dit._int8_w8a8(_resume_cfg(run_dir)) is True
