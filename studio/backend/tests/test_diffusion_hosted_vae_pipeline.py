# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys

import pytest

from core.inference.diffusion import DiffusionBackend
from .test_diffusion_backend import (
    _FakePipeline,
    _FakeTransformer,
    _force_cuda_target,
    fake_runtime as fake_runtime,
)


@pytest.mark.parametrize(
    "token,hub_token,expected",
    [
        (False, None, False),
        (None, None, "omitted"),
        ("test-token", None, "test-token"),
        (None, False, False),
    ],
    ids = ["anonymous-direct", "ambient", "explicit", "anonymous-worker"],
)
def test_hosted_vae_staged_fallback_keeps_pipeline_token(
    fake_runtime, monkeypatch, token, hub_token, expected
):
    import huggingface_hub
    from core.inference import diffusion as dmod
    from core.inference import diffusion_te_prequant

    class ReachedPipeline(BaseException):
        pass

    backend = DiffusionBackend()
    _force_cuda_target(backend, monkeypatch)
    diffusers = sys.modules["diffusers"]
    monkeypatch.setattr(diffusers, "QwenImage21Pipeline", _FakePipeline, raising = False)
    monkeypatch.setattr(diffusers, "QwenImage21Transformer2DModel", _FakeTransformer, raising = False)
    monkeypatch.setattr(dmod, "prefer_ungated_mirror", lambda base, *_a: base)
    monkeypatch.setattr(diffusion_te_prequant, "te_prequant_mirror_path", lambda *_a, **_k: None)
    seen = []

    def download(repo, filename, **kwargs):
        seen.append((filename, kwargs.get("token", "omitted")))
        raise FileNotFoundError(filename)

    def construct(base, **kwargs):
        seen.append(("pipeline", kwargs.get("token", "omitted")))
        raise ReachedPipeline

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(_FakePipeline, "from_pretrained", construct)
    with pytest.raises(ReachedPipeline):
        backend.load_pipeline(
            "Qwen/Qwen-Image-2.1",
            model_kind = "pipeline",
            hf_token = token,
            _hub_token = hub_token,
            speed_mode = "off",
            text_encoder_quant = "off",
            transformer_quant = "off",
        )
    assert (
        "vae/qwen_image_2.1_vae_bf16.safetensors",
        None if expected == "omitted" else expected,
    ) in seen
    assert seen[-1] == ("pipeline", expected)
