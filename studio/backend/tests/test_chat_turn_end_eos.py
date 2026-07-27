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
    resolve_chat_turn_end_eos_ids_using,
)


class _FakeTokenizer:
    def __init__(
        self,
        eos_id,
        chat_template = "",
        token_ids = None,
        unk_token_id = None,
    ):
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
    tok = _FakeTokenizer(248044, chat_template = _CHATML, token_ids = {"<|im_end|>": 248046})
    assert resolve_chat_turn_end_eos_ids(tok) == [248044, 248046]


def test_marker_in_vocab_but_not_in_template_is_ignored():
    # Base/coder model: <|im_end|> is in the vocab but the template does not use
    # it, so it must not become a stop token.
    tok = _FakeTokenizer(248044, chat_template = "{{ messages }}", token_ids = {"<|im_end|>": 248046})
    assert resolve_chat_turn_end_eos_ids(tok) == [248044]


def test_harmony_template_is_left_untouched():
    # gpt-oss/harmony: <|end|> is a channel delimiter, not the turn end.
    harmony = "<|start|>assistant<|channel|>analysis<|message|>...<|end|>"
    tok = _FakeTokenizer(200002, chat_template = harmony, token_ids = {"<|end|>": 200007})
    assert resolve_chat_turn_end_eos_ids(tok) == [200002]


def test_llama3_eot_id_from_template():
    tok = _FakeTokenizer(128001, chat_template = "...<|eot_id|>...", token_ids = {"<|eot_id|>": 128009})
    assert resolve_chat_turn_end_eos_ids(tok) == [128001, 128009]


def test_gemma4_turn_marker_from_template():
    # Gemma-4 ends turns with <turn|> while keeping a document eos, so <turn|> must
    # be added as a stop token.
    tok = _FakeTokenizer(
        1, chat_template = "...<start_of_turn>...<turn|>...", token_ids = {"<turn|>": 106}
    )
    assert resolve_chat_turn_end_eos_ids(tok) == [1, 106]


def test_resolve_using_reads_markers_from_template_but_ids_from_generation_tokenizer():
    # map_eos_token=True: the mapped template remaps <|im_end|> onto the doc-eos id,
    # but the original keeps it atomic. Reading marker STRINGS from the template but
    # IDS on the original recovers the real turn-end id (7), not the doc-eos id (2).
    template_tok = _FakeTokenizer(2, chat_template = _CHATML, token_ids = {"<|im_end|>": 2})
    id_tok = _FakeTokenizer(2, chat_template = "", token_ids = {"<|im_end|>": 7})
    assert resolve_chat_turn_end_eos_ids_using(template_tok, id_tok) == [2, 7]
    # Same tokenizer for both reproduces the plain resolve (load-time behaviour).
    assert resolve_chat_turn_end_eos_ids_using(template_tok, template_tok) == [2]


def test_list_eos_preserved():
    tok = _FakeTokenizer([1, 2], chat_template = _CHATML, token_ids = {"<|im_end|>": 2})
    assert resolve_chat_turn_end_eos_ids(tok) == [1, 2]


def test_missing_marker_maps_to_unk_and_is_skipped():
    tok = _FakeTokenizer(7, chat_template = _CHATML, token_ids = {}, unk_token_id = 0)
    assert resolve_chat_turn_end_eos_ids(tok) == [7]


def test_starling_barred_end_of_turn_from_template():
    # OpenChat/Starling end turns with the BARRED <|end_of_turn|> (distinct from
    # Gemma's <end_of_turn>). eos synced to </s>=2, turn marker at 32000.
    starling = "GPT4 Correct Assistant: hi<|end_of_turn|>"
    tok = _FakeTokenizer(2, chat_template = starling, token_ids = {"<|end_of_turn|>": 32000})
    assert resolve_chat_turn_end_eos_ids(tok) == [2, 32000]


def test_dict_chat_template_scans_all_variants():
    # Hermes-3 style: chat_template is a {name: template} dict. Detection must scan
    # every variant, not bail because the container is not a plain str.
    tmpl = {"default": "{{ messages }}", "tool_use": _CHATML}
    tok = _FakeTokenizer(2, chat_template = tmpl, token_ids = {"<|im_end|>": 5})
    assert resolve_chat_turn_end_eos_ids(tok) == [2, 5]


def test_list_of_dicts_chat_template_scans_all_variants():
    # tokenizer_config.json stores multi-templates as a list of {name, template}.
    tmpl = [{"name": "default", "template": _CHATML}]
    tok = _FakeTokenizer(2, chat_template = tmpl, token_ids = {"<|im_end|>": 5})
    assert resolve_chat_turn_end_eos_ids(tok) == [2, 5]


def test_dict_harmony_template_left_untouched():
    # A multi-variant container whose variant is harmony must still be left alone.
    tmpl = {"default": "<|start|>assistant<|channel|>analysis<|message|>...<|end|>"}
    tok = _FakeTokenizer(200002, chat_template = tmpl, token_ids = {"<|end|>": 200007})
    assert resolve_chat_turn_end_eos_ids(tok) == [200002]


# ---- chat_eos_repair ------------------------------------------------------


def test_repair_adds_missing_turn_end():
    assert chat_eos_repair(248044, [248044, 248046]) == [248044, 248046]


def test_repair_from_missing_generation_config_eos():
    assert chat_eos_repair(None, [248046]) == [248046]


def test_repair_noop_when_already_covered():
    assert chat_eos_repair([248046, 248044], [248046]) is None


def test_repair_noop_when_no_turn_end_ids():
    assert chat_eos_repair(248044, []) is None
