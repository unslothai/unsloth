# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real MLX raw-prompt completion (/v1/completions), on Apple Silicon only."""

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
_PROMPT = "The capital of France is"


@pytest.fixture
def backend():
    pytest.importorskip("mlx_lm")
    pytest.importorskip("unsloth_zoo.mlx.loader")
    from core.inference.mlx_inference import MLXInferenceBackend
    from utils.models.model_config import ModelConfig

    backend = MLXInferenceBackend()
    assert backend.load_model(ModelConfig.from_identifier(_TINY), max_seq_length = 512)
    return backend


def _greedy(backend, ids, **kw):
    text = ""
    for text in backend.generate_completion(
        ids, temperature = 0.0, top_p = 1.0, top_k = 0, min_p = 0.0, max_new_tokens = 8, **kw
    ):
        pass
    return text


def test_raw_prompt_matches_a_direct_untemplated_mlx_lm_run(backend):
    from mlx_lm import stream_generate

    resolved = backend.resolve_completion_prompt(_PROMPT)
    assert resolved["ids"] == backend._tokenizer.encode(_PROMPT, add_special_tokens = True)
    assert resolved["text"] == _PROMPT

    # Decoded from the ids: mlx_lm's streaming detokenizer trims the first piece's leading
    # space, which a raw continuation (and llama-server) keeps.
    tokens = [
        r.token
        for r in stream_generate(
            backend._model, backend._tokenizer, prompt = resolved["ids"], max_tokens = 8
        )
    ]
    direct = backend._tokenizer.decode(
        tokens, skip_special_tokens = True, clean_up_tokenization_spaces = False
    )
    assert direct.strip()
    assert _greedy(backend, resolved["ids"]) == direct
    # Token ids resolve to themselves and decode as the same greedy text.
    assert backend.resolve_completion_prompt(resolved["ids"])["ids"] == resolved["ids"]


def test_raw_prompt_grammar_constrains_the_completion(backend):
    pytest.importorskip("llguidance")
    from core.inference.grammar_constraint import grammar_from_gbnf

    text = _greedy(
        backend,
        backend.resolve_completion_prompt("Answer yes or no:")["ids"],
        grammar = grammar_from_gbnf('root ::= " yes" | " no"'),
    )
    assert text in (" yes", " no")


def test_raw_prompt_rejects_out_of_vocabulary_ids(backend):
    with pytest.raises(ValueError):
        backend.resolve_completion_prompt([10**9])
