# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Nemotron-H with fast_inference = True falls back to Unsloth inference when the installed unsloth_zoo cannot rebuild it from vLLM (unslothai/unsloth-zoo#565)."""

import importlib.util

import pytest
from real_accelerator import (
    has_real_accelerator,
)  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.gpu

MODEL_NAME = "trl-internal-testing/tiny-NemotronHForCausalLM-nano"


@pytest.mark.skipif(not has_real_accelerator(), reason = "needs a GPU")
def test_nemotron_h_needs_a_zoo_that_rebuilds_it(monkeypatch):
    import unsloth_zoo.empty_model as zoo_empty_model
    from unsloth.models.loader import _vllm_unsupported_model_types

    monkeypatch.setattr(zoo_empty_model, "extract_nemotron_h_mixer", object(), raising = False)
    assert _vllm_unsupported_model_types(["nemotron_h"]) == []
    assert _vllm_unsupported_model_types(["gpt_oss"]) == ["gpt_oss"]
    monkeypatch.delattr(zoo_empty_model, "extract_nemotron_h_mixer")
    assert _vllm_unsupported_model_types(["nemotron_h"]) == ["nemotron_h"]
    assert _vllm_unsupported_model_types(["llama"]) == []


@pytest.mark.skipif(not has_real_accelerator(), reason = "needs a GPU")
@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason = "needs vLLM installed")
def test_nemotron_h_fast_inference_falls_back_on_an_older_zoo(capfd, monkeypatch):
    import unsloth_zoo.empty_model as zoo_empty_model
    from unsloth import FastLanguageModel

    monkeypatch.delattr(zoo_empty_model, "extract_nemotron_h_mixer", raising = False)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 256,
        load_in_4bit = False,
        fast_inference = True,
        gpu_memory_utilization = 0.1,
        # vLLM engine knobs reach transformers' from_pretrained unless the fallback drops them.
        enforce_eager = True,
        max_num_seqs = 8,
    )
    out = capfd.readouterr().out
    assert "does not support nemotron_h yet - will switch to Unsloth inference" in out
    assert "Ignoring vLLM-only arguments ['enforce_eager', 'max_num_seqs']" in out
    assert getattr(model, "vllm_engine", None) is None

    inputs = tokenizer(["Hello"], return_tensors = "pt").to(model.device)
    out = model.fast_generate(**inputs, max_new_tokens = 4, min_new_tokens = 4, do_sample = False)
    assert out.shape[1] == inputs["input_ids"].shape[1] + 4
