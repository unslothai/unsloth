# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-family accuracy defaults of the DiT LoRA trainer, and that runs started under the previous
defaults still resume with what they were trained with. CPU only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from core.training import diffusion_checkpoint as dc
from core.training.diffusion_dit_trainer import _SPECS, _select_lora_targets
from core.training.diffusion_train_common import DEFAULT_LORA_TARGETS, DiffusionLoraConfig

_OLD_ZIMAGE_TARGETS = ("to_q", "to_k", "to_v", "to_out.0")


def _zimage_cfg(**kw) -> DiffusionLoraConfig:
    return DiffusionLoraConfig(
        base_model = "Tongyi-MAI/Z-Image-Turbo", data_dir = "d", output_dir = "o", **kw
    ).normalized()


def test_zimage_default_targets_cover_attention_and_feed_forward():
    targets = _SPECS["z-image"].lora_targets
    assert set(_OLD_ZIMAGE_TARGETS) <= set(targets)
    assert {"w1", "w2", "w3"} <= set(targets)
    cfg = _zimage_cfg()
    assert _select_lora_targets(cfg.lora_target_modules, targets) == targets
    assert dc.identity_for_config(cfg).lora_target_modules == targets


def test_zimage_feed_forward_names_match_the_diffusers_model():
    diffusers = pytest.importorskip("diffusers")
    model_cls = getattr(diffusers, "ZImageTransformer2DModel", None)
    if model_cls is None:
        pytest.skip("diffusers has no Z-Image")
    model = model_cls(
        all_patch_size = (2,),
        all_f_patch_size = (1,),
        in_channels = 4,
        dim = 64,
        n_layers = 1,
        n_refiner_layers = 1,
        n_heads = 2,
        n_kv_heads = 2,
        cap_feat_dim = 32,
        axes_dims = [8, 12, 12],
        axes_lens = [64, 32, 32],
    )
    leaves = {
        n.rsplit(".", 1)[-1] for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)
    }
    assert {"w1", "w2", "w3"} <= leaves


@pytest.fixture
def run_dir():
    from utils.paths import outputs_root

    d = outputs_root() / "zimage-old-run"
    d.mkdir(parents = True, exist_ok = True)
    return d


def _write_bundle(run_dir: Path, identity: dc.CheckpointIdentity) -> None:
    param = torch.nn.Parameter(torch.zeros(2, 2))
    optimizer = torch.optim.AdamW([param], lr = 1e-3)
    dc.save_checkpoint(
        output_dir = str(run_dir),
        step = 11,
        adapter_state = {"w": torch.zeros(2, 2)},
        identity = identity,
        target_steps = 500,
        optimizer = optimizer,
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0),
        rng = dc.capture_rng_state({}),
        sampler_state = {"n": 1, "order": [0], "pos": 0},
    )


def _old_zimage_identity(**overrides) -> dc.CheckpointIdentity:
    fresh = dc.identity_for_config(_zimage_cfg())
    fields = {**fresh.as_dict(), "lora_target_modules": _OLD_ZIMAGE_TARGETS, "flow_shift": "1.0"}
    fields.update(overrides)
    return dc.CheckpointIdentity.from_dict(fields)


def test_resuming_an_old_zimage_run_keeps_its_recorded_targets(run_dir):
    saved = _old_zimage_identity()
    _write_bundle(run_dir, saved)
    cfg = _zimage_cfg(resume_from_checkpoint = str(run_dir))
    assert cfg.lora_target_modules == _OLD_ZIMAGE_TARGETS
    # The model's own schedule: 1.0 before the numeric flow_shift became the effective shift, None after.
    assert cfg.flow_shift in (None, 1.0)
    assert _select_lora_targets(cfg.lora_target_modules, _SPECS["z-image"].lora_targets) == (
        _OLD_ZIMAGE_TARGETS
    )
    incoming = dc.identity_for_config(cfg)
    assert incoming.lora_target_modules == saved.lora_target_modules
    assert incoming.flow_shift == saved.flow_shift
    cfg = _zimage_cfg(resume_from_checkpoint = str(run_dir / "checkpoint-11"))
    assert cfg.lora_target_modules == _OLD_ZIMAGE_TARGETS


def test_a_save_killed_mid_promotion_still_resumes_on_its_targets(run_dir):
    # A crash mid-promotion leaves only the aged staging bundle, which the preflight restores.
    import os
    import time

    _write_bundle(run_dir, _old_zimage_identity())
    aside = run_dir / f"{dc._STAGING_PREFIX}replaced-11-cafebabe"
    os.replace(run_dir / "checkpoint-11", aside)
    old = time.time() - (dc._LIVE_REPLACEMENT_GRACE_SECONDS + 60)
    os.utime(aside, (old, old))
    cfg = _zimage_cfg(resume_from_checkpoint = str(run_dir))
    assert cfg.lora_target_modules == _OLD_ZIMAGE_TARGETS


def test_a_run_folder_named_like_a_bundle_still_resumes_on_its_targets():
    from utils.paths import outputs_root

    run_dir = outputs_root() / "checkpoint-2026"
    run_dir.mkdir(parents = True, exist_ok = True)
    _write_bundle(run_dir, _old_zimage_identity())
    cfg = _zimage_cfg(resume_from_checkpoint = str(run_dir))
    assert cfg.lora_target_modules == _OLD_ZIMAGE_TARGETS


def test_explicit_request_targets_win_over_the_recorded_ones(run_dir):
    _write_bundle(run_dir, _old_zimage_identity())
    cfg = _zimage_cfg(
        resume_from_checkpoint = str(run_dir),
        lora_target_modules = ("to_q", "to_v"),
    )
    assert cfg.lora_target_modules == ("to_q", "to_v")


def test_unreadable_resume_path_falls_back_to_the_family_defaults(tmp_path):
    cfg = _zimage_cfg(resume_from_checkpoint = str(tmp_path / "nowhere"))
    assert cfg.lora_target_modules == DEFAULT_LORA_TARGETS
    assert dc.recorded_resume_targets(str(tmp_path / "nowhere")) is None


def test_zimage_default_keeps_the_checkpoints_built_in_shift():
    # Shift 2-3 (DiffSynth / musubi) hurt likeness, so keep the scheduler's static shift (3 Turbo, 6 base).
    diffusers = pytest.importorskip("diffusers")
    from core.training.diffusion_dit_trainer import _training_sigma_table
    for shift, base_model in ((3.0, "Tongyi-MAI/Z-Image-Turbo"), (6.0, "Tongyi-MAI/Z-Image")):
        cfg = DiffusionLoraConfig(base_model = base_model, data_dir = "d", output_dir = "o").normalized()
        assert cfg.resolved_family == "z-image" and cfg.flow_shift in (None, 1.0)
        sched = diffusers.FlowMatchEulerDiscreteScheduler(num_train_timesteps = 1000, shift = shift)
        table = _training_sigma_table(sched, cfg.flow_shift)
        assert table is sched.sigmas
        t = torch.linspace(1.0, 1.0 / 1000, 1000)
        assert torch.allclose(table.float(), shift * t / (1 + (shift - 1) * t), atol = 1e-5)
        u = torch.sigmoid(torch.randn(200_000, generator = torch.Generator().manual_seed(0)))
        sig = table[(u * 1000).long().clamp(0, 999)].float()
        expected = float((shift * u / (1 + (shift - 1) * u) > 0.8).float().mean())
        assert abs(float((sig > 0.8).float().mean()) - expected) < 0.01
