# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mapper models whose own tokenizer ships no chat_template have their turn-end
eos resolved at LOAD from an empty template (document eos only). The effective
template is installed later, at generate time, via get_chat_template, so the
turn-end-eos cache must be refreshed then; otherwise generate_stream runs past
the ChatML <|im_end|> boundary and loops (the exact bug this PR fixes).
"""

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from core.inference import inference as inf_mod  # noqa: E402
from core.inference.inference import InferenceBackend  # noqa: E402

_CHATML = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}"


class _FakeTokenizer:
    def __init__(
        self,
        eos_id,
        chat_template = "",
        token_ids = None,
    ):
        self.eos_token_id = eos_id
        self.chat_template = chat_template
        self.pad_token_id = eos_id
        self.unk_token_id = None
        self._ids = dict(token_ids or {})

    def convert_tokens_to_ids(self, tok):
        return self._ids.get(tok)


def test_turn_end_eos_refreshed_after_generate_time_template(monkeypatch):
    import utils.datasets as ds

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "unsloth/qwen2.5-0.5b"

    # Loaded tokenizer ships NO chat_template, so the load-time cache saw an
    # empty template and stored only the document eos.
    bare_tok = _FakeTokenizer(151643, chat_template = "")
    model_info = {
        "tokenizer": bare_tok,
        "is_vision": False,
        "chat_turn_end_eos_ids": [151643],
    }
    backend.models = {backend.active_model_name: model_info}

    # The mapper installs a ChatML template (ends turns with <|im_end|>) at
    # generate time.
    templated_tok = _FakeTokenizer(151643, chat_template = _CHATML, token_ids = {"<|im_end|>": 151645})
    monkeypatch.setattr(inf_mod, "get_chat_template", lambda tok, chat_template = None: templated_tok)
    monkeypatch.setattr(
        ds, "MODEL_TO_TEMPLATE_MAPPER", {backend.active_model_name: "qwen-2.5"}, raising = False
    )

    # Stub the tail so the generator runs through the refresh without a real model.
    monkeypatch.setattr(backend, "_normalize_top_k", lambda k: k, raising = False)
    monkeypatch.setattr(
        backend, "_apply_chat_template_for_generation", lambda *a, **k: "PROMPT", raising = False
    )
    monkeypatch.setattr(backend, "generate_stream", lambda *a, **k: iter(()), raising = False)

    list(backend._generate_chat_response_inner(messages = [{"role": "user", "content": "hi"}]))

    # After the effective template is applied the cache must include the ChatML
    # turn-end id, not just the stale document eos.
    assert model_info["chat_turn_end_eos_ids"] == [151643, 151645]
