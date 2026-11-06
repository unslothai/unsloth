# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

MODEL = "unsloth/gemma-4-E2B-it"
NATIVE = (_BACKEND / "assets" / "chat_templates" / "gemma-4-edge.jinja").read_text(encoding = "utf-8")
RESULT = "Estonia won 0 gold, 1 silver and 0 bronze medals."
WEB_SEARCH = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
}
MESSAGES = [
    {"role": "user", "content": "What medals did Estonia win at the 2026 Winter Olympics?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": {"query": "Estonia medals 2026 Winter Olympics"},
                },
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call_0", "name": "web_search", "content": RESULT},
]


def _tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<|turn>": 4, "<turn|>": 5}
    backend = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = backend,
        unk_token = "<unk>",
        bos_token = "<bos>",
        eos_token = "<eos>",
        pad_token = "<pad>",
    )
    tokenizer.chat_template = NATIVE
    return tokenizer


def _prompt(monkeypatch, tools):
    try:
        from core.inference import inference
    except (ImportError, RuntimeError) as exc:
        pytest.skip(f"full inference backend unavailable ({type(exc).__name__}: {exc})")
    if getattr(inference.get_chat_template, "__module__", None) != "unsloth.chat_templates":
        pytest.skip("inference module is bound to a stubbed unsloth.chat_templates")
    backend = inference.InferenceBackend.__new__(inference.InferenceBackend)
    backend.active_model_name = MODEL
    backend.models = {
        MODEL: {"tokenizer": _tokenizer(), "is_vision": False, "native_chat_template": NATIVE}
    }
    prompts = []
    monkeypatch.setattr(
        backend,
        "generate_stream",
        lambda prompt, *a, **k: prompts.append(prompt) or iter(()),
        raising = False,
    )
    list(
        backend._generate_chat_response_inner(
            messages = MESSAGES, system_prompt = "Be brief.", tools = tools
        )
    )
    return prompts[0]


@pytest.mark.parametrize("tools", [[WEB_SEARCH], None], ids = ["tools", "tool_budget_spent"])
def test_gemma4_tool_result_reaches_the_model(monkeypatch, tools):
    prompt = _prompt(monkeypatch, tools)
    assert RESULT in prompt
    assert "Be brief." in prompt
    assert "What medals did Estonia win" in prompt
