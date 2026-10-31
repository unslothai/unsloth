# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real MLX load of one precision folder of a repo (#13418), on Apple Silicon only."""

from __future__ import annotations

import platform
import shutil
import sys

import pytest

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "darwin" or platform.machine() != "arm64",
        reason = "MLX runs on Apple Silicon only",
    ),
    pytest.mark.allow_network,
]

_TINY = "mlx-community/SmolLM-135M-Instruct-4bit"
_REPO = "example/SmolLM-135M-Instruct-MLX"


def test_precision_folder_loads_and_generates(monkeypatch, tmp_path):
    pytest.importorskip("mlx_lm")
    pytest.importorskip("unsloth_zoo.mlx.loader")
    from huggingface_hub import snapshot_download

    from core.inference.mlx_inference import MLXInferenceBackend
    from utils import hf_cache_settings
    from utils.models.model_config import ModelConfig

    tiny = snapshot_download(_TINY)
    hub = tmp_path / "hub"
    repo_dir = hub / f"models--{_REPO.replace('/', '--')}"
    snapshot = repo_dir / "snapshots" / ("a" * 40)
    shutil.copytree(tiny, snapshot / "4bit")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_text("a" * 40)
    monkeypatch.setattr(
        hf_cache_settings,
        "get_hf_cache_paths",
        lambda: hf_cache_settings.HuggingFaceCachePaths(tmp_path, hub, tmp_path / "xet", "studio"),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")

    config = ModelConfig.from_identifier(_REPO, gguf_variant = "4bit")
    assert config.identifier == _REPO
    assert config.gguf_variant == "4bit"
    assert config.is_gguf is False

    backend = MLXInferenceBackend()
    assert backend.load_model(config, max_seq_length = 512)
    assert backend.active_model_name == _REPO
    assert backend.models[_REPO]["gguf_variant"] == "4bit"
    assert str(getattr(backend._model, "_src_path", "")).endswith("4bit")

    from mlx_lm import generate

    text = generate(
        backend._model, backend._tokenizer, prompt = "The capital of France is", max_tokens = 8
    )
    assert isinstance(text, str) and text.strip()
