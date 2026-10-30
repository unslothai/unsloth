# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A text-only LoRA on a VLM exports with the base's processor and without training's use_cache=False."""

from __future__ import annotations

import sys
import types
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


def _export_mod(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    return _load_module("test_core_export_vlm_processor", "core/export/export.py", monkeypatch)


class _Tokenizer:
    chat_template = "{{ trained }}"


class _Processor:
    def __init__(self):
        self.tokenizer = object()
        self.chat_template = "{{ base }}"


def _stub_auto_processor(monkeypatch, load):
    calls = []

    class AutoProcessor:
        @staticmethod
        def from_pretrained(name, **kw):
            calls.append((name, kw))
            return load()

    monkeypatch.setitem(
        sys.modules, "transformers", types.SimpleNamespace(AutoProcessor = AutoProcessor)
    )
    return calls


def test_bare_tokenizer_is_wrapped_in_the_base_processor(monkeypatch):
    mod = _export_mod(monkeypatch)
    calls = _stub_auto_processor(monkeypatch, _Processor)
    tok = _Tokenizer()
    out = mod._with_base_processor(tok, "org/vlm", trust_remote_code = True, token = None)
    assert isinstance(out, _Processor)
    assert out.tokenizer is tok
    assert out.chat_template == "{{ trained }}"
    assert calls == [("org/vlm", {"trust_remote_code": True, "token": None})]


def test_a_loaded_processor_is_kept(monkeypatch):
    mod = _export_mod(monkeypatch)
    calls = _stub_auto_processor(monkeypatch, _Processor)
    proc = _Processor()
    assert mod._with_base_processor(proc, "org/vlm") is proc
    assert calls == []


def test_base_without_processor_keeps_the_tokenizer(monkeypatch):
    mod = _export_mod(monkeypatch)
    _stub_auto_processor(monkeypatch, _Tokenizer)
    tok = _Tokenizer()
    assert mod._with_base_processor(tok, "org/text") is tok


def test_processor_load_failure_keeps_the_tokenizer(monkeypatch):
    mod = _export_mod(monkeypatch)

    def _boom():
        raise OSError("no processor_config.json")

    _stub_auto_processor(monkeypatch, _boom)
    tok = _Tokenizer()
    assert mod._with_base_processor(tok, "org/vlm") is tok


def test_export_load_restores_use_cache(monkeypatch):
    mod = _export_mod(monkeypatch)
    restored = []
    zoo = types.ModuleType("unsloth_zoo")
    training_utils = types.ModuleType("unsloth_zoo.training_utils")
    training_utils.restore_use_cache = restored.append
    zoo.training_utils = training_utils
    monkeypatch.setitem(sys.modules, "unsloth_zoo", zoo)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.training_utils", training_utils)
    model = object()
    mod._restore_use_cache(model)
    assert restored == [model]


def test_restore_use_cache_tolerates_an_old_zoo(monkeypatch):
    mod = _export_mod(monkeypatch)
    monkeypatch.setitem(sys.modules, "unsloth_zoo.training_utils", types.ModuleType("x"))
    mod._restore_use_cache(object())
