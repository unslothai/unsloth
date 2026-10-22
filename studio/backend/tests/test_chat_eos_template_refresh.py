# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Refresh the turn-end eos cache at generate time, since the chat template is only installed then."""

import sys
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# These tests construct InferenceBackend, pulling the full stack. CI may lack
# unsloth/unsloth_zoo (ImportError) or have a broken CUDA/bitsandbytes setup
# (RuntimeError); skip at module level so collection is not aborted (exit 2).
try:
    from core.inference import inference as inf_mod  # noqa: E402
    from core.inference.inference import InferenceBackend  # noqa: E402
except (ImportError, RuntimeError) as exc:  # pragma: no cover - env-dependent
    pytest.skip(
        f"full inference backend unavailable ({type(exc).__name__}: {exc})",
        allow_module_level = True,
    )

_CHATML = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}"
_GEMMA = "{% for m in messages %}<start_of_turn>{{m.role}}\n{{m.content}}<end_of_turn>{% endfor %}"


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

    # No chat_template at load, so the cache held only the document eos.
    bare_tok = _FakeTokenizer(151643, chat_template = "", token_ids = {"<|im_end|>": 151645})
    model_info = {
        "tokenizer": bare_tok,
        "is_vision": False,
        "chat_turn_end_eos_ids": [151643],
    }
    backend.models = {backend.active_model_name: model_info}

    templated_tok = _FakeTokenizer(151643, chat_template = _CHATML, token_ids = {"<|im_end|>": 151645})
    monkeypatch.setattr(inf_mod, "get_chat_template", lambda tok, chat_template = None: templated_tok)
    monkeypatch.setattr(
        ds, "MODEL_TO_TEMPLATE_MAPPER", {backend.active_model_name: "qwen-2.5"}, raising = False
    )

    monkeypatch.setattr(backend, "_normalize_top_k", lambda k: k, raising = False)
    monkeypatch.setattr(
        backend, "_apply_chat_template_for_generation", lambda *a, **k: "PROMPT", raising = False
    )
    monkeypatch.setattr(backend, "generate_stream", lambda *a, **k: iter(()), raising = False)

    list(backend._generate_chat_response_inner(messages = [{"role": "user", "content": "hi"}]))

    assert model_info["chat_turn_end_eos_ids"] == [151643, 151645]


def test_turn_end_eos_refresh_preserves_load_time_ids_on_destructive_swap(monkeypatch):
    # Regression: get_chat_template can return a remapped tokenizer (Gemma: <end_of_turn>
    # folded onto the eos id) while generate_stream re-reads the original. Resolving on
    # the swap yields a narrower set, so the refresh must UNION, never overwrite.
    import utils.datasets as ds

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "unsloth/gemma-2b-it"

    orig_tok = _FakeTokenizer(1, chat_template = _GEMMA, token_ids = {"<end_of_turn>": 107})
    model_info = {
        "tokenizer": orig_tok,
        "is_vision": False,
        "chat_turn_end_eos_ids": [1, 107],
    }
    backend.models = {backend.active_model_name: model_info}

    swapped_tok = _FakeTokenizer(1, chat_template = _GEMMA, token_ids = {"<end_of_turn>": 1})
    monkeypatch.setattr(inf_mod, "get_chat_template", lambda tok, chat_template = None: swapped_tok)
    monkeypatch.setattr(
        ds, "MODEL_TO_TEMPLATE_MAPPER", {backend.active_model_name: "gemma-3"}, raising = False
    )

    monkeypatch.setattr(backend, "_normalize_top_k", lambda k: k, raising = False)
    monkeypatch.setattr(
        backend, "_apply_chat_template_for_generation", lambda *a, **k: "PROMPT", raising = False
    )
    monkeypatch.setattr(backend, "generate_stream", lambda *a, **k: iter(()), raising = False)

    list(backend._generate_chat_response_inner(messages = [{"role": "user", "content": "hi"}]))

    assert model_info["chat_turn_end_eos_ids"] == [1, 107]


def test_turn_end_eos_refresh_resolves_marker_id_on_original_not_remapped(monkeypatch):
    # Yi-style map_eos_token=True: the original carries <|im_end|> at its own id, but
    # get_chat_template folds it onto the doc-eos id. generate_stream uses the original,
    # so read marker strings from the mapped template but ids from the original.
    import utils.datasets as ds

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "01-ai/yi-6b"

    orig_tok = _FakeTokenizer(2, chat_template = "", token_ids = {"<|im_end|>": 7})
    model_info = {
        "tokenizer": orig_tok,
        "is_vision": False,
        "chat_turn_end_eos_ids": [2],
    }
    backend.models = {backend.active_model_name: model_info}

    remapped_tok = _FakeTokenizer(2, chat_template = _CHATML, token_ids = {"<|im_end|>": 2})
    monkeypatch.setattr(inf_mod, "get_chat_template", lambda tok, chat_template = None: remapped_tok)
    monkeypatch.setattr(
        ds, "MODEL_TO_TEMPLATE_MAPPER", {backend.active_model_name: "chatml"}, raising = False
    )

    monkeypatch.setattr(backend, "_normalize_top_k", lambda k: k, raising = False)
    monkeypatch.setattr(
        backend, "_apply_chat_template_for_generation", lambda *a, **k: "PROMPT", raising = False
    )
    monkeypatch.setattr(backend, "generate_stream", lambda *a, **k: iter(()), raising = False)

    list(backend._generate_chat_response_inner(messages = [{"role": "user", "content": "hi"}]))

    assert model_info["chat_turn_end_eos_ids"] == [2, 7]


class _FakeProcessor:
    """A ProcessorMixin-like container: carries the chat_template itself and
    wraps the real text tokenizer as ``.tokenizer`` (the vision layout)."""

    def __init__(self, chat_template, tokenizer):
        self.chat_template = chat_template
        self.tokenizer = tokenizer


def test_resolve_chat_eos_reads_vision_processor_template():
    # Vision model: the chat_template lives on the processor while the inner tokenizer
    # ships none. _resolve_chat_eos must read the marker from the processor but resolve
    # its id on the inner tokenizer, and repair generation_config.
    from types import SimpleNamespace

    inner_tok = _FakeTokenizer(1, chat_template = "", token_ids = {"<end_of_turn>": 107})
    processor = _FakeProcessor(_GEMMA, inner_tok)
    model = SimpleNamespace(generation_config = SimpleNamespace(eos_token_id = 1))

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = "unsloth/gemma-3-4b-it"
    model_info = {"model": model, "tokenizer": processor, "processor": processor, "is_vision": True}
    backend.models = {backend.active_model_name: model_info}

    backend._resolve_chat_eos(backend.active_model_name)

    assert model_info["chat_turn_end_eos_ids"] == [1, 107]
    assert model.generation_config.eos_token_id == [1, 107]
