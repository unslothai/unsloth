# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import contextlib
import random
import sys
from types import SimpleNamespace

import pytest

from core.training import diffusion_dit_trainer as dit
from core.training.diffusion_download_progress import TrainingDownloadProgress
from core.training.diffusion_train_common import DiffusionLoraConfig


@pytest.mark.parametrize("warm_preview", [False, True])
def test_dit_resumes_download_progress_after_preparation(monkeypatch, tmp_path, warm_preview):
    # Warm route can still need encoders for preview prompts.
    monkeypatch.setitem(
        sys.modules, "diffusers", SimpleNamespace(FlowMatchEulerDiscreteScheduler = object)
    )
    monkeypatch.setitem(
        sys.modules, "diffusers.optimization", SimpleNamespace(get_scheduler = object)
    )
    monkeypatch.setitem(
        sys.modules, "diffusers.training_utils", SimpleNamespace(cast_training_params = object)
    )
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(LoraConfig = object))
    monkeypatch.setitem(
        sys.modules, "peft.utils", SimpleNamespace(get_peft_model_state_dict = object)
    )
    monkeypatch.setattr("loggers.config.quiet_third_party_progress_bars", lambda: None)
    monkeypatch.setattr(dit, "plan_image_canvases", lambda *_a: (None, False))
    monkeypatch.setattr(dit, "_plan_cache_variants", lambda *_a, **_k: object())
    monkeypatch.setattr(dit, "_free_text_encoders", lambda *_a: None)
    monkeypatch.setattr(dit, "_encode_prompts_cached", lambda *_a: [object()])
    monkeypatch.setattr(dit, "_resolve_base_precision", lambda *_a: "bf16")
    monkeypatch.setattr(dit, "isolated_sampling", lambda *_a: contextlib.nullcontext())
    monkeypatch.delenv("UNSLOTH_DIFFUSION_NO_LATENT_CACHE", raising = False)
    events = []
    monitor = TrainingDownloadProgress(
        events.append,
        sample = lambda: {"bytes_done": 1, "bytes_total": 2, "scope": "active_files"},
    )
    monitor.report({"type": "model_load_started"})

    class ReachedDownload(Exception):
        pass

    def download(*_args):
        monitor.poll()
        assert any(event["type"] == "preparing" for event in events)
        assert [event["type"] for event in events[-2:]] == [
            "model_load_started",
            "model_download_progress",
        ]
        raise ReachedDownload

    vae = SimpleNamespace(to = lambda *_a: None)
    spec = SimpleNamespace(
        family = "flux.1",
        lora_targets = ("to_q",),
        decode_latents = warm_preview,
        load_conditioners = download if warm_preview else lambda *_a: (SimpleNamespace(), vae),
        load_transformer = download,
    )
    cfg = DiffusionLoraConfig(
        base_model = "unsloth/FLUX.1-dev",
        data_dir = str(tmp_path),
        output_dir = str(tmp_path / "out"),
        cache_latents = True,
        cond_cache_dir = str(tmp_path / "cache") if warm_preview else None,
    )
    if warm_preview:
        monkeypatch.setattr(
            dit, "plan_samples", lambda *_a: SimpleNamespace(encode_texts = ["preview"])
        )
        monkeypatch.setattr(
            "core.training.diffusion_train_extras.source_revision", lambda *_a: "local"
        )
        monkeypatch.setattr(dit, "PersistentConditioningCache", lambda *_a: object())
        monkeypatch.setattr(
            dit, "_load_warm_conditioning", lambda *_a: ({"caption": object()}, [object()])
        )
    else:

        def cache(*args, **kwargs):
            args[6]({"type": "preparing", "stage": "cache_latents", "done": 1, "total": 1})
            return [object()]

        monkeypatch.setattr(dit, "_build_latent_cache", cache)
    with pytest.raises(ReachedDownload):
        dit._train_dit(
            cfg,
            spec,
            [(str(tmp_path / "image.png"), "caption")],
            random.Random(0),
            "cpu",
            None,
            monitor.report,
            lambda: False,
            lambda: True,
            None,
        )
