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


def test_raw_vision_prompt_matches_the_decoded_sampled_ids():
    pytest.importorskip("mlx_vlm")
    pytest.importorskip("unsloth_zoo.mlx.loader")
    import mlx.core as mx
    from mlx_vlm import stream_generate

    from core.inference.mlx_inference import MLXInferenceBackend
    from utils.models.model_config import ModelConfig

    backend = MLXInferenceBackend()
    assert backend.load_model(
        ModelConfig.from_identifier("mlx-community/Qwen3.5-0.8B-4bit"), max_seq_length = 512
    )
    if not backend._is_vlm:
        pytest.skip("loaded as a text model")
    ids = backend.resolve_completion_prompt(_PROMPT)["ids"]
    # Keyed by count: the closing summary repeats the last token.
    by_count = {
        r.generation_tokens: int(r.token)
        for r in stream_generate(
            backend._model, backend._processor, "", None, input_ids = mx.array([ids]), max_tokens = 8
        )
        if r.token is not None
    }
    tokens = [by_count[n] for n in sorted(by_count)]
    # Leading space included: the runtime's own text trims it, a raw continuation keeps it.
    direct = backend._tokenizer.decode(
        tokens, skip_special_tokens = True, clean_up_tokenization_spaces = False
    )
    assert direct.strip() and _greedy(backend, ids) == direct


def _direct_logprobs(backend, ids):
    """mlx_lm's log_softmax for the first generated position with no processors, under the
    scopes Studio decodes in: the logits llama-server reads ``n_probs`` from."""
    from mlx_lm import stream_generate

    with backend._prefill_decision_scopes(), backend._generation_scopes():
        (first,) = list(
            stream_generate(backend._model, backend._tokenizer, prompt = ids, max_tokens = 1)
        )
    return first.logprobs


# mlx_lm reduces its log_softmax in the model's dtype, so its reference carries a few ulps of
# that dtype (bf16 for a 4-bit checkpoint); the recorder reads float32 like llama-server.
_REF_EPS = {"bfloat16": 2.0**-8, "float16": 2.0**-11, "float32": 2.0**-24}


def _ref_approx(direct, value):
    eps = _REF_EPS.get(str(direct.dtype).rsplit(".", 1)[-1], 2.0**-8)
    return pytest.approx(value, abs = max(1e-3, 4 * eps * abs(value)))


def _scored(backend, ids, **kw):
    for _ in backend.generate_completion(
        ids, temperature = 0.0, top_p = 1.0, top_k = 0, min_p = 0.0, max_new_tokens = 1, **kw
    ):
        pass
    return backend.last_generation_stats["token_logprobs"]


def test_raw_prompt_logprobs_match_a_direct_forward_pass(backend):
    ids = backend.resolve_completion_prompt(_PROMPT)["ids"]
    direct = _direct_logprobs(backend, ids)
    greedy = int(direct.argmax().item())
    (entry,) = _scored(backend, ids, n_probs = 3)
    assert entry["id"] == greedy
    assert entry["logprob"] == _ref_approx(direct, direct[greedy].item())
    assert entry["top_logprobs"][0]["id"] == greedy and len(entry["top_logprobs"]) == 3


def test_raw_prompt_logprobs_report_a_forced_token_unbiased(backend):
    import mlx.core as mx

    ids = backend.resolve_completion_prompt(_PROMPT)["ids"]
    direct = _direct_logprobs(backend, ids)
    forced = int(mx.argsort(-direct)[100].item())
    (entry,) = _scored(backend, ids, n_probs = 2, logit_bias = {forced: 100.0})
    assert entry["id"] == forced
    assert entry["logprob"] == _ref_approx(direct, direct[forced].item())
    assert entry["logprob"] < -1.0
    assert entry["top_logprobs"][0]["id"] == int(direct.argmax().item())
