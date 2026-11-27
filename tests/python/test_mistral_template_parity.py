# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
"""Expected strings = the official Mistral-7B-Instruct-v0.2 chat_template output, inlined for offline runs."""

import pytest
from tokenizers import Tokenizer, models
from transformers import PreTrainedTokenizerFast

from unsloth.chat_templates import get_chat_template


@pytest.fixture(autouse = True)
def _scratch_cwd(tmp_path, monkeypatch):
    # get_chat_template writes a scratch dir under CWD.
    monkeypatch.chdir(tmp_path)


def _tokenizer():
    vocab = {"<unk>": 0, "<s>": 1, "</s>": 2, "<pad>": 3}
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object = Tokenizer(models.WordLevel(vocab, unk_token = "<unk>")),
        bos_token = "<s>",
        eos_token = "</s>",
        unk_token = "<unk>",
        pad_token = "<pad>",
    )
    return get_chat_template(tokenizer, "mistral")


EXPECTED = {
    (
        ("system", "Be brief."),
        ("user", "Hi"),
        ("assistant", "Hello!"),
    ): "<s> [INST] Be brief.\n\nHi [/INST] Hello!</s>",
    (
        ("user", "Hi"),
        ("assistant", "Hello!"),
    ): "<s> [INST] Hi [/INST] Hello!</s>",
    (
        ("system", "Be brief."),
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "Bye"),
        ("assistant", "Bye!"),
    ): "<s> [INST] Be brief.\n\nHi [/INST] Hello!</s> [INST] Bye [/INST] Bye!</s>",
    (
        ("user", "Hi"),
        ("assistant", "Hello!"),
        ("user", "Bye"),
        ("assistant", "Bye!"),
    ): "<s> [INST] Hi [/INST] Hello!</s> [INST] Bye [/INST] Bye!</s>",
}


@pytest.mark.parametrize("add_generation_prompt", [False, True])
@pytest.mark.parametrize(
    "conversation", list(EXPECTED), ids = ["sys", "no_sys", "sys_multi", "multi"]
)
def test_mistral_template_matches_official(conversation, add_generation_prompt):
    tokenizer = _tokenizer()
    messages = [{"role": r, "content": c} for r, c in conversation]
    out = tokenizer.apply_chat_template(
        messages, tokenize = False, add_generation_prompt = add_generation_prompt
    )
    assert out == EXPECTED[conversation]
