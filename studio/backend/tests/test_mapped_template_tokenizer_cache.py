# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The mapped chat template is applied once per loaded tokenizer, not on every message."""

import sys
import threading
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

try:
    from core.inference import inference as inf_mod  # noqa: E402
    from core.inference.inference import InferenceBackend  # noqa: E402
except (ImportError, RuntimeError) as exc:  # pragma: no cover - env-dependent
    pytest.skip(
        f"full inference backend unavailable ({type(exc).__name__}: {exc})",
        allow_module_level = True,
    )

_CHATML = "{% for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{% endfor %}"
_NAME = "unsloth/gemma-2-2b-it"


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


def _backend(monkeypatch, model_info, get_chat_template):
    import utils.datasets as ds

    backend = InferenceBackend.__new__(InferenceBackend)
    backend.active_model_name = _NAME
    backend.models = {_NAME: model_info}
    monkeypatch.setattr(inf_mod, "get_chat_template", get_chat_template)
    monkeypatch.setattr(ds, "MODEL_TO_TEMPLATE_MAPPER", {_NAME.lower(): "gemma2"}, raising = False)
    monkeypatch.setattr(backend, "_normalize_top_k", lambda k: k, raising = False)
    rendered_with = []

    def _render(tok, *a, **k):
        rendered_with.append(tok)
        return "PROMPT"

    monkeypatch.setattr(backend, "_apply_chat_template_for_generation", _render, raising = False)
    monkeypatch.setattr(backend, "generate_stream", lambda *a, **k: iter(()), raising = False)
    return backend, rendered_with


def _send(backend, text = "hi"):
    list(backend._generate_chat_response_inner(messages = [{"role": "user", "content": text}]))


class _Counter:
    def __init__(self, make):
        self.calls = []
        self.make = make

    def __call__(
        self,
        tok,
        chat_template = None,
    ):
        self.calls.append((tok, chat_template))
        return self.make(tok)


def _remapped(tok):
    return _FakeTokenizer(1, chat_template = _CHATML, token_ids = {"<|im_end|>": 1})


def test_template_applied_once_and_reused_across_messages(monkeypatch):
    orig = _FakeTokenizer(1, token_ids = {"<|im_end|>": 107})
    model_info = {"tokenizer": orig, "is_vision": False, "chat_turn_end_eos_ids": [1]}
    gct = _Counter(_remapped)
    backend, rendered_with = _backend(monkeypatch, model_info, gct)

    for i in range(5):
        _send(backend, f"msg {i}")

    assert len(gct.calls) == 1
    assert gct.calls[0] == (orig, "gemma2")
    assert len(rendered_with) == 5
    assert all(t is rendered_with[0] for t in rendered_with)
    assert rendered_with[0] is not orig
    # The turn-end refresh still runs every message and still unions onto the load-time ids.
    assert model_info["chat_turn_end_eos_ids"] == [1, 107]


def test_eos_union_matches_uncached_repeat(monkeypatch):
    orig = _FakeTokenizer(1, token_ids = {"<|im_end|>": 107})
    cached_info = {"tokenizer": orig, "is_vision": False, "chat_turn_end_eos_ids": [5]}
    backend, _ = _backend(monkeypatch, cached_info, _Counter(_remapped))
    for _ in range(3):
        _send(backend)

    # The uncached behaviour: a fresh model_info per message forces get_chat_template each time.
    fresh_ids = [5]
    for _ in range(3):
        info = {"tokenizer": orig, "is_vision": False, "chat_turn_end_eos_ids": fresh_ids}
        backend.models = {_NAME: info}
        _send(backend)
        fresh_ids = info["chat_turn_end_eos_ids"]
    assert cached_info["chat_turn_end_eos_ids"] == fresh_ids == [1, 5, 107]


def test_reload_and_tokenizer_swap_reapply(monkeypatch):
    orig = _FakeTokenizer(1)
    model_info = {"tokenizer": orig, "is_vision": False}
    gct = _Counter(_remapped)
    backend, rendered_with = _backend(monkeypatch, model_info, gct)
    _send(backend)
    _send(backend)
    assert len(gct.calls) == 1

    # Same model_info, new tokenizer object: the old remap must not be reused.
    new_tok = _FakeTokenizer(1)
    model_info["tokenizer"] = new_tok
    _send(backend)
    assert len(gct.calls) == 2 and gct.calls[-1][0] is new_tok

    # Reload: load_model builds a new model_info dict.
    backend.models = {_NAME: {"tokenizer": orig, "is_vision": False}}
    _send(backend)
    assert len(gct.calls) == 3
    assert rendered_with[-1] is not rendered_with[0]


def test_template_name_change_reapplies(monkeypatch):
    import utils.datasets as ds

    orig = _FakeTokenizer(1)
    model_info = {"tokenizer": orig, "is_vision": False}
    gct = _Counter(_remapped)
    backend, _ = _backend(monkeypatch, model_info, gct)
    _send(backend)
    monkeypatch.setattr(ds, "MODEL_TO_TEMPLATE_MAPPER", {_NAME.lower(): "chatml"}, raising = False)
    _send(backend)
    _send(backend)
    assert [c[1] for c in gct.calls] == ["gemma2", "chatml"]


def test_failure_is_not_cached_and_falls_back(monkeypatch):
    orig = _FakeTokenizer(1, chat_template = _CHATML)
    model_info = {"tokenizer": orig, "is_vision": False}
    state = {"fail": True}

    def make(tok):
        if state["fail"]:
            raise KeyError("qwen3.5")
        return _remapped(tok)

    gct = _Counter(make)
    backend, rendered_with = _backend(monkeypatch, model_info, gct)
    _send(backend)
    _send(backend)
    # Failed calls render with the tokenizer's own template, and every message retries.
    assert rendered_with == [orig, orig]
    assert len(gct.calls) == 2
    assert "_mapped_template_tokenizer" not in model_info
    state["fail"] = False
    _send(backend)
    _send(backend)
    assert len(gct.calls) == 3
    assert rendered_with[2] is rendered_with[3] is not orig


def test_concurrent_first_messages_apply_once(monkeypatch):
    orig = _FakeTokenizer(1)
    model_info = {"tokenizer": orig, "is_vision": False}
    gate = threading.Event()

    def slow(tok):
        gate.wait(5)
        return _remapped(tok)

    gct = _Counter(slow)
    backend, rendered_with = _backend(monkeypatch, model_info, gct)
    threads = [threading.Thread(target = _send, args = (backend,)) for _ in range(4)]
    for t in threads:
        t.start()
    gate.set()
    for t in threads:
        t.join(10)
    assert len(gct.calls) == 1
    assert len(rendered_with) == 4 and all(t is rendered_with[0] for t in rendered_with)


def test_unmapped_model_never_calls_get_chat_template(monkeypatch):
    import utils.datasets as ds

    orig = _FakeTokenizer(1, chat_template = _CHATML)
    model_info = {"tokenizer": orig, "is_vision": False}
    gct = _Counter(_remapped)
    backend, rendered_with = _backend(monkeypatch, model_info, gct)
    monkeypatch.setattr(ds, "MODEL_TO_TEMPLATE_MAPPER", {}, raising = False)
    _send(backend)
    assert gct.calls == [] and rendered_with == [orig]
    assert "_mapped_template_tokenizer" not in model_info
