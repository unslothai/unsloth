# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real MLX: incremental reply snapshots equal the full re-decode ones, on Apple Silicon only."""

from __future__ import annotations

import platform
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
_GREEDY = dict(temperature = 0.0, top_p = 1.0, top_k = 0, min_p = 0.0, max_new_tokens = 300)


@pytest.fixture
def backend():
    pytest.importorskip("mlx_lm")
    pytest.importorskip("unsloth_zoo.mlx.loader")
    from core.inference.mlx_inference import MLXInferenceBackend
    from utils.models.model_config import ModelConfig

    backend = MLXInferenceBackend()
    assert backend.load_model(ModelConfig.from_identifier(_TINY), max_seq_length = 1024)
    return backend


def _run(monkeypatch, generate, *, incremental):
    from transformers import PreTrainedTokenizerFast
    from core.inference import incremental_decode

    decoded = [0]
    original = PreTrainedTokenizerFast._decode

    def _counting(self, token_ids, *args, **kwargs):
        decoded[0] += len(token_ids) if isinstance(token_ids, list) else 1
        return original(self, token_ids, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(PreTrainedTokenizerFast, "_decode", _counting)
        if not incremental:
            patch.setattr(incremental_decode, "split_profile", lambda _tokenizer: None)
        snapshots = list(generate())
    return snapshots, decoded[0]


@pytest.mark.parametrize("surface", ["chat", "completion"])
def test_real_mlx_snapshots_match_a_full_redecode_with_bounded_work(backend, monkeypatch, surface):
    from core.inference.incremental_decode import split_profile

    assert split_profile(backend._tokenizer) is not None
    if surface == "chat":
        messages = [{"role": "user", "content": "Count from 1 to 400, separated by commas."}]
        generate = lambda: backend.generate_chat_response(messages, **_GREEDY)
    else:
        ids = backend.resolve_completion_prompt("1, 2, 3, 4, 5, 6, 7, 8,")["ids"]
        generate = lambda: backend.generate_completion(ids, **_GREEDY)

    full, full_work = _run(monkeypatch, generate, incremental = False)
    fast, fast_work = _run(monkeypatch, generate, incremental = True)
    assert len(full) > 100
    assert fast == full
    assert fast_work * 4 < full_work, (fast_work, full_work)
