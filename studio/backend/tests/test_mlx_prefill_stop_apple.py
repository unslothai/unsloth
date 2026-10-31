# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stop during prefill on real mlx-lm (random init, no download), on Apple Silicon only."""

from __future__ import annotations

import platform
import sys
import threading

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or platform.machine() != "arm64",
    reason = "MLX runs on Apple Silicon only",
)

_VOCAB = 320
_PROMPT = list(range(2, 202))
_STEP = 16
# 199 prompt tokens go through chunked prefill before the last one is sampled from.
_PREFILL_CHUNKS = -(-(len(_PROMPT) - 1) // _STEP)


def _tiny_llama():
    pytest.importorskip("mlx_lm")
    from mlx_lm.models import llama

    class Counting(llama.Model):
        def __call__(self, *args, **kwargs):
            self.calls += 1
            if self.on_call is not None:
                self.on_call(self.calls)
            return super().__call__(*args, **kwargs)

    model = Counting(
        llama.ModelArgs(
            model_type = "llama",
            hidden_size = 64,
            num_hidden_layers = 1,
            intermediate_size = 128,
            num_attention_heads = 4,
            rms_norm_eps = 1e-5,
            vocab_size = _VOCAB,
            num_key_value_heads = 4,
        )
    )
    model.calls, model.on_call = 0, None
    return model


def _tokenizer():
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {f"t{i}": i for i in range(_VOCAB)}
    raw = Tokenizer(models.WordLevel(vocab, unk_token = "t0"))
    raw.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    # No eos: a sampled id cannot end the reply early.
    return PreTrainedTokenizerFast(tokenizer_object = raw, unk_token = "t0")


def _stream(model, tokenizer, cancel):
    from core.inference.mlx_inference import _stream_until_cancelled
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    kwargs = dict(
        prompt = list(_PROMPT),
        max_tokens = 4,
        sampler = make_sampler(temp = 0.0),
        prefill_step_size = _STEP,
    )
    if cancel is None:
        return list(stream_generate(model, tokenizer, **kwargs))
    return list(_stream_until_cancelled(stream_generate, cancel, model, tokenizer, **kwargs))


def test_stop_set_during_prefill_ends_after_the_current_chunk():
    model, tokenizer, cancel = _tiny_llama(), _tokenizer(), threading.Event()
    model.on_call = lambda n: cancel.set()

    assert _stream(model, tokenizer, cancel) == []
    assert model.calls == 1


def test_a_plain_stream_reads_the_whole_prompt_before_a_stop_can_land():
    model, tokenizer = _tiny_llama(), _tokenizer()
    responses = _stream(model, tokenizer, None)

    assert responses
    assert model.calls > _PREFILL_CHUNKS


def test_the_runtime_generates_again_after_a_stopped_prefill():
    model, tokenizer, cancel = _tiny_llama(), _tokenizer(), threading.Event()
    model.on_call = lambda n: cancel.set()
    assert _stream(model, tokenizer, cancel) == []

    model.on_call, model.calls = None, 0
    assert len(_stream(model, tokenizer, threading.Event())) == 4
    assert model.calls > _PREFILL_CHUNKS
