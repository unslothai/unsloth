# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_chat_turn_end_eos_ids: chat generation must stop on the assistant-turn-end
token even when a loader synced tokenizer.eos to a bare document terminator
(Qwen sets eos to <|endoftext|> though turns end with <|im_end|>). Missing the
turn-end token let a small model run past its turn and loop, re-emitting tool
calls / hallucinating <|im_start|> turns.
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference.inference import _chat_turn_end_eos_ids  # noqa: E402


class _FakeTokenizer:
    def __init__(
        self,
        eos_id,
        vocab,
        unk_token_id = None,
    ):
        self.eos_token_id = eos_id
        self.unk_token_id = unk_token_id
        self._vocab = dict(vocab)

    def get_vocab(self):
        return self._vocab

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, self.unk_token_id)


def test_qwen_chatml_eos_synced_to_endoftext_still_stops_on_im_end():
    # The reproduced bug: eos synced to <|endoftext|>; <|im_end|> ends the turn.
    tok = _FakeTokenizer(eos_id = 248044, vocab = {"<|endoftext|>": 248044, "<|im_end|>": 248046})
    assert _chat_turn_end_eos_ids(tok) == [248044, 248046]


def test_llama_turn_end_tokens_added():
    tok = _FakeTokenizer(eos_id = 128001, vocab = {"<|eot_id|>": 128009, "<|eom_id|>": 128008})
    assert _chat_turn_end_eos_ids(tok) == [128001, 128008, 128009]


def test_plain_model_without_chat_markers_is_unchanged():
    tok = _FakeTokenizer(eos_id = 2, vocab = {"</s>": 2, "hello": 5})
    assert _chat_turn_end_eos_ids(tok) == [2]


def test_list_eos_is_preserved_and_deduped():
    tok = _FakeTokenizer(eos_id = [1, 2], vocab = {"<|im_end|>": 2})
    assert _chat_turn_end_eos_ids(tok) == [1, 2]


def test_unk_marker_is_not_added():
    # A marker that maps to unk (not really in vocab) must be ignored.
    tok = _FakeTokenizer(eos_id = 7, vocab = {"<|end|>": 0}, unk_token_id = 0)
    assert _chat_turn_end_eos_ids(tok) == [7]
