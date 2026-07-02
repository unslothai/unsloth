# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""chat_eos: resolve assistant-turn-end stop tokens from the chat_template and
repair generation_config so a chat model whose eos is a bare document terminator
(Qwen3.5: config eos <|endoftext|>, turns end with <|im_end|>) stops at the turn
boundary instead of running past it and looping. Dependency-light: imported here
without the full inference stack.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.chat_eos import (  # noqa: E402
    chat_eos_repair,
    resolve_chat_turn_end_eos_ids,
)


class _FakeTokenizer:
    def __init__(self, eos_id, chat_template="", token_ids=None, unk_token_id=None):
        self.eos_token_id = eos_id
        self.chat_template = chat_template
        self.unk_token_id = unk_token_id
        self._ids = dict(token_ids or {})

    def convert_tokens_to_ids(self, tok):
        return self._ids.get(tok, self.unk_token_id)


# ---- resolve_chat_turn_end_eos_ids ---------------------------------------

_CHATML = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}"


def test_qwen35_adds_im_end_from_template():
    # eos synced to <|endoftext|> (248044); template uses <|im_end|> (248046).
    tok = _FakeTokenizer(248044, chat_template=_CHATML, token_ids={"<|im_end|>": 248046})
    assert resolve_chat_turn_end_eos_ids(tok) == [248044, 248046]


def test_marker_in_vocab_but_not_in_template_is_ignored():
    # Base/coder model: <|im_end|> exists in the shared vocab but the template
    # does not use it, so it must not become a stop token.
    tok = _FakeTokenizer(248044, chat_template="{{ messages }}", token_ids={"<|im_end|>": 248046})
    assert resolve_chat_turn_end_eos_ids(tok) == [248044]


def test_harmony_template_is_left_untouched():
    # gpt-oss/harmony: <|end|> is a channel delimiter, not the turn end.
    harmony = "<|start|>assistant<|channel|>analysis<|message|>...<|end|>"
    tok = _FakeTokenizer(200002, chat_template=harmony, token_ids={"<|end|>": 200007})
    assert resolve_chat_turn_end_eos_ids(tok) == [200002]


def test_llama3_eot_id_from_template():
    tok = _FakeTokenizer(128001, chat_template="...<|eot_id|>...", token_ids={"<|eot_id|>": 128009})
    assert resolve_chat_turn_end_eos_ids(tok) == [128001, 128009]


def test_list_eos_preserved():
    tok = _FakeTokenizer([1, 2], chat_template=_CHATML, token_ids={"<|im_end|>": 2})
    assert resolve_chat_turn_end_eos_ids(tok) == [1, 2]


def test_missing_marker_maps_to_unk_and_is_skipped():
    tok = _FakeTokenizer(7, chat_template=_CHATML, token_ids={}, unk_token_id=0)
    assert resolve_chat_turn_end_eos_ids(tok) == [7]


# ---- chat_eos_repair ------------------------------------------------------


def test_repair_adds_missing_turn_end():
    assert chat_eos_repair(248044, [248044, 248046]) == [248044, 248046]


def test_repair_from_missing_generation_config_eos():
    assert chat_eos_repair(None, [248046]) == [248046]


def test_repair_noop_when_already_covered():
    assert chat_eos_repair([248046, 248044], [248046]) is None


def test_repair_noop_when_no_turn_end_ids():
    assert chat_eos_repair(248044, []) is None
