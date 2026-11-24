# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expected strings from the official Phi-3 / Phi-3.5 templates (tokenizer_config.json), hard coded for offline runs."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<s>": 1, "<|endoftext|>": 2, "<pad>": 3, "<|end|>": 4}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<s>",
        eos_token = "<|endoftext|>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, name)


TEMPLATES = ["phi-3", "phi-35", "phi-3.5"]

CONVERSATIONS = {
    "sys": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ],
    "no_sys": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
    ],
}


@pytest.mark.parametrize("name", TEMPLATES)
@pytest.mark.parametrize("conversation", list(CONVERSATIONS.values()), ids = list(CONVERSATIONS))
def test_phi3_render_ends_with_eos(name, conversation):
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = False)
    assert out.endswith("<|endoftext|>")


@pytest.mark.parametrize("name", TEMPLATES)
def test_phi3_render_matches_official_string(name):
    conversation = CONVERSATIONS["sys"]
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(conversation, tokenize = False, add_generation_prompt = False)
    expected = (
        "<|system|>\nBe brief.<|end|>\n"
        "<|user|>\nHi<|end|>\n"
        "<|assistant|>\nHello!<|end|>\n"
        "<|endoftext|>"
    )
    assert out == expected


@pytest.mark.parametrize("name", TEMPLATES)
def test_phi3_generation_prompt_still_works(name):
    tokenizer = _tokenizer(name)
    out = tokenizer.apply_chat_template(
        CONVERSATIONS["sys"], tokenize = False, add_generation_prompt = True
    )
    assert out.endswith("<|assistant|>\n")
    assert "<|endoftext|>" not in out
