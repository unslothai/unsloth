# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from eval.inference_adapter import make_generate, collect_generation


class _FakeBackend:
    def __init__(self):
        self.active_model_name = None
    def generate_chat_response(self, messages, system_prompt, **kw):
        text = "hello world"
        acc = ""
        for tok in text.split():
            acc = (acc + " " + tok).strip()
            yield acc


def test_collect_generation_takes_final_cumulative_chunk():
    gen = (c for c in ["a", "a b", "a b c"])
    assert collect_generation(gen) == "a b c"


def test_make_generate_returns_text():
    backend = _FakeBackend()
    generate = make_generate(backend, max_new_tokens=16, temperature=0.0)
    out = generate([{"role": "user", "content": "hi"}], "")
    assert out == "hello world"
