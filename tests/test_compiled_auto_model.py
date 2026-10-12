# SPDX-License-Identifier: AGPL-3.0-only
"""A concrete `auto_model` class resolves to the compiler's replacement in its modeling module."""

import sys
import types

import pytest

from unsloth.models.loader import _compiled_auto_model


class _Stock:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()


def _compiled(monkeypatch, cls):
    module = types.ModuleType("unsloth_compiled_module_fake")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    setattr(module, cls.__name__, cls)
    return cls


def _fake_module(monkeypatch, cls):
    module = types.ModuleType("_unsloth_test_modeling_fake")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(cls, "__module__", module.__name__)
    setattr(module, cls.__name__, cls)
    return module


def test_swapped_concrete_class_resolves_to_replacement(monkeypatch):
    stock = type("FakeForConditionalGeneration", (_Stock,), {})
    module = _fake_module(monkeypatch, stock)
    compiled = _compiled(monkeypatch, type("FakeForConditionalGeneration", (_Stock,), {}))
    setattr(module, "FakeForConditionalGeneration", compiled)
    assert _compiled_auto_model(stock) is compiled


def test_unswapped_class_and_none_are_unchanged(monkeypatch):
    stock = type("FakeForConditionalGeneration", (_Stock,), {})
    _fake_module(monkeypatch, stock)
    assert _compiled_auto_model(stock) is stock
    assert _compiled_auto_model(None) is None


def test_auto_class_is_unchanged(monkeypatch):
    auto = type("FakeAutoModel", (_Stock,), {"_model_mapping": {}})
    module = _fake_module(monkeypatch, auto)
    setattr(module, "FakeAutoModel", type("FakeAutoModel", (_Stock,), {}))
    assert _compiled_auto_model(auto) is auto


@pytest.mark.parametrize(
    "replacement", ["renamed", "no_from_pretrained", "not_a_class", "not_compiled"]
)
def test_unsuitable_replacement_is_ignored(monkeypatch, replacement):
    stock = type("FakeForConditionalGeneration", (_Stock,), {})
    module = _fake_module(monkeypatch, stock)
    other = {
        "renamed": type("Other", (_Stock,), {}),
        "no_from_pretrained": type("FakeForConditionalGeneration", (), {}),
        "not_a_class": object(),
        "not_compiled": type("FakeForConditionalGeneration", (_Stock,), {}),
    }[replacement]
    setattr(module, "FakeForConditionalGeneration", other)
    assert _compiled_auto_model(stock) is stock


def test_real_whisper_class_follows_module_swap(monkeypatch):
    modeling_whisper = pytest.importorskip("transformers.models.whisper.modeling_whisper")
    stock = modeling_whisper.WhisperForConditionalGeneration
    compiled = _compiled(monkeypatch, type("WhisperForConditionalGeneration", (stock,), {}))
    monkeypatch.setattr(modeling_whisper, "WhisperForConditionalGeneration", compiled)
    assert _compiled_auto_model(stock) is compiled
    from transformers import AutoModelForSpeechSeq2Seq

    assert _compiled_auto_model(AutoModelForSpeechSeq2Seq) is AutoModelForSpeechSeq2Seq
