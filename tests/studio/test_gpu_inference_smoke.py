# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real GPU generation smoke that skips on GitHub-hosted runners; select it with -k gpu_generation."""

from __future__ import annotations

import pytest
from real_accelerator import (
    has_real_cuda,
)

torch = pytest.importorskip("torch")

MODEL_ID = "unsloth/gemma-3-270m-it"
MIN_NEW_TOKENS = 4
MAX_NEW_TOKENS = 16


@pytest.mark.skipif(not has_real_cuda(), reason = "requires a CUDA GPU")
def test_gpu_generation_smoke():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - env without transformers
        pytest.skip(f"transformers unavailable: {exc}")

    # Gemma emits only <pad> in fp16; use bf16 where supported, else fp32.
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype = dtype).to("cuda")
    except Exception as exc:
        pytest.skip(f"could not fetch/load {MODEL_ID}: {exc}")

    model.eval()
    messages = [{"role": "user", "content": "Say hello in one word."}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt = True, return_dict = True, return_tensors = "pt"
    ).to("cuda")
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output = model.generate(
            **inputs,
            min_new_tokens = MIN_NEW_TOKENS,
            max_new_tokens = MAX_NEW_TOKENS,
            do_sample = False,
        )

    assert output.shape[1] > prompt_len, "no tokens were generated on the GPU"
    reply = tokenizer.decode(output[0][prompt_len:], skip_special_tokens = True)
    assert reply.strip(), "expected a non-empty GPU generation"
