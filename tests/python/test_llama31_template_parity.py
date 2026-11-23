# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expected strings rendered by the official meta-llama/Llama-3.1-8B-Instruct chat
template (tokenizer_config.json, mirrored identically by mlx-community and RedHatAI
repos): default date "26 Jul 2024" and |trim on system, first-user and turn content,
hard coded for offline runs."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD; isolate it per test and xdist worker.
    monkeypatch.chdir(tmp_path)


def _tokenizer(name):
    vocab = {"<unk>": 0, "<|begin_of_text|>": 1, "<|eot_id|>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<|begin_of_text|>",
        eos_token = "<|eot_id|>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, name)


# llama-3.1, llama-31, llama-3.2, llama-3.3, llama-32, llama-33 share one template object in
# Unsloth. The expected strings below are the official Llama 3.1 / 3.3 renders. The official
# Llama 3.2 template differs (it renders the live date with strftime_now), so for llama-3.2
# these cases only pin the shared object, not parity with the official 3.2 template.
TEMPLATES = ["llama-3.1", "llama-3.2", "llama-3.3"]

SYSTEM_HEADER = (
    "<|start_header_id|>system<|end_header_id|>\n\n"
    "Cutting Knowledge Date: December 2023\n"
    "Today Date: 26 Jul 2024\n\n"
)

EXPECTED = {
    "sys": (
        "<|begin_of_text|>" + SYSTEM_HEADER + "Be brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
    ),
    "no_sys": (
        "<|begin_of_text|>" + SYSTEM_HEADER + "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
    ),
    "padded": (
        "<|begin_of_text|>" + SYSTEM_HEADER + "Be brief.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>"
    ),
}

CONVERSATIONS = {
    "sys": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ],
    "no_sys": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ],
    # Official trims |trim on system/turn content; padded input must render the same as clean.
    "padded": [
        {"role": "system", "content": "  Be brief.  "},
        {"role": "user", "content": "  Hi  "},
        {"role": "assistant", "content": "  Hello!  "},
    ],
}


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize("scenario", list(CONVERSATIONS))
@pytest.mark.parametrize("name", TEMPLATES)
def test_llama31_template_matches_official(name, scenario, add_generation_prompt):
    tokenizer = _tokenizer(name)
    expected = EXPECTED[scenario]
    if add_generation_prompt:
        expected += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    out = tokenizer.apply_chat_template(
        CONVERSATIONS[scenario],
        tokenize = False,
        add_generation_prompt = add_generation_prompt,
    )
    assert out == expected
