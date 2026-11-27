# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""gpt-oss with fast_inference = True loads on Unsloth inference instead of crashing in vLLM weight sharing (unslothai/unsloth#4541)."""

import importlib.util

import pytest
from real_accelerator import (
    has_real_accelerator,
)  # tests/_shared, on sys.path via tests/conftest.py

pytestmark = pytest.mark.gpu

MODEL_NAME = "trl-internal-testing/tiny-GptOssForCausalLM"


@pytest.mark.skipif(not has_real_accelerator(), reason = "needs a GPU")
@pytest.mark.skipif(importlib.util.find_spec("vllm") is None, reason = "needs vLLM installed")
def test_gpt_oss_fast_inference_falls_back(capfd):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = MODEL_NAME,
        max_seq_length = 256,
        load_in_4bit = False,
        fast_inference = True,
        gpu_memory_utilization = 0.1,
    )
    assert (
        "does not support gpt_oss yet - will switch to Unsloth inference" in capfd.readouterr().out
    )
    assert getattr(model, "vllm_engine", None) is None

    inputs = tokenizer(["Hello"], return_tensors = "pt").to(model.device)
    out = model.fast_generate(**inputs, max_new_tokens = 4, min_new_tokens = 4, do_sample = False)
    assert out.shape[1] == inputs["input_ids"].shape[1] + 4
