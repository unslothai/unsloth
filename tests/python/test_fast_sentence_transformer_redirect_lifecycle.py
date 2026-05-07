"""FastSentenceTransformer constructor-redirect lifecycle:
- AutoModel/AutoProcessor/AutoTokenizer.from_pretrained are restored even
  when the Transformer constructor raises (try/finally invariant).
- The closure that decides whether to substitute the pre-loaded objects
  (`is_requested_model_name`) handles HF repo IDs, local paths, trailing
  slashes, pathlib.Path objects, and missing identifiers correctly.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import types

import pytest


def _stub_module(name: str) -> types.ModuleType:
    # __spec__ must be set so importlib.util.find_spec(name) does not raise
    # ValueError if a downstream test imports the real package.
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.util.spec_from_loader(name, loader = None)
    return mod


_STUB_KEYS = (
    "transformers",
    "sentence_transformers",
    "sentence_transformers.models",
)


@pytest.fixture(autouse = True)
def _restore_sys_modules():
    """Snapshot the entries we shadow with stubs and restore them after each
    test so a downstream test that does `import transformers` for real does
    not pick up our non-package stub."""
    saved = {k: sys.modules.get(k) for k in _STUB_KEYS}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


class _FakeAuto:
    def __init__(self, name):
        self.name = name
        self.from_pretrained = self._original

    def _original(self, *args, **kwargs):
        return ("orig", self.name, args, kwargs)


class _RecordingTransformerOk:
    last_calls = None

    def __init__(self, model_name, **kwargs):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer

        type(self).last_calls = {
            "model": AutoModel.from_pretrained(model_name),
            "processor": AutoProcessor.from_pretrained(model_name),
            "tokenizer": AutoTokenizer.from_pretrained(model_name),
        }


class _RaisingTransformer:
    def __init__(self, *a, **kw):
        from transformers import AutoModel

        AutoModel.from_pretrained(a[0] if a else kw.get("model_name_or_path"))
        raise RuntimeError("simulated init failure")


def _build_driver(transformer_class):
    transformers_mod = _stub_module("transformers")
    transformers_mod.AutoModel = _FakeAuto("AutoModel")
    transformers_mod.AutoProcessor = _FakeAuto("AutoProcessor")
    transformers_mod.AutoTokenizer = _FakeAuto("AutoTokenizer")
    sys.modules["transformers"] = transformers_mod

    st_root = _stub_module("sentence_transformers")
    st_models = _stub_module("sentence_transformers.models")
    st_models.Transformer = transformer_class
    sys.modules["sentence_transformers"] = st_root
    sys.modules["sentence_transformers.models"] = st_models

    captured = {"calls": None}

    def driver(model_name, model, tokenizer):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        from sentence_transformers.models import Transformer

        def is_requested_model_name(args, kwargs):
            requested = None
            if args:
                requested = args[0]
            else:
                requested = kwargs.get("pretrained_model_name_or_path")
                if requested is None:
                    requested = kwargs.get("model_name_or_path")
            if requested is None:
                return False
            try:
                requested = os.fspath(requested)
                expected = os.fspath(model_name)
            except (TypeError, ValueError):
                return False
            if requested == expected:
                return True
            try:
                if os.path.exists(requested) or os.path.exists(expected):
                    return os.path.abspath(requested) == os.path.abspath(expected)
            except (OSError, TypeError, ValueError):
                pass
            return False

        original_model = AutoModel.from_pretrained
        original_processor = AutoProcessor.from_pretrained
        original_tokenizer = AutoTokenizer.from_pretrained

        def return_existing_model(*a, **kw):
            return model if is_requested_model_name(a, kw) else original_model(*a, **kw)

        def return_existing_tokenizer(*a, **kw):
            return (
                tokenizer
                if is_requested_model_name(a, kw)
                else original_tokenizer(*a, **kw)
            )

        def return_existing_processor(*a, **kw):
            return (
                tokenizer
                if is_requested_model_name(a, kw)
                else original_processor(*a, **kw)
            )

        try:
            AutoModel.from_pretrained = return_existing_model
            AutoProcessor.from_pretrained = return_existing_processor
            AutoTokenizer.from_pretrained = return_existing_tokenizer
            t = Transformer(model_name)
            captured["calls"] = getattr(type(t), "last_calls", None)
            return t
        finally:
            AutoModel.from_pretrained = original_model
            AutoProcessor.from_pretrained = original_processor
            AutoTokenizer.from_pretrained = original_tokenizer

    return driver, transformers_mod, captured


def test_redirect_substitutes_preloaded_objects_on_match():
    driver, _mod, captured = _build_driver(_RecordingTransformerOk)
    sentinel_model = object()
    sentinel_tok = object()
    driver("sentence-transformers/all-MiniLM-L6-v2", sentinel_model, sentinel_tok)
    calls = captured["calls"]
    assert calls["model"] is sentinel_model
    assert calls["processor"] is sentinel_tok
    assert calls["tokenizer"] is sentinel_tok


def test_redirect_restored_on_constructor_exception():
    driver, transformers_mod, _ = _build_driver(_RaisingTransformer)
    pre_model = transformers_mod.AutoModel.from_pretrained
    pre_processor = transformers_mod.AutoProcessor.from_pretrained
    pre_tokenizer = transformers_mod.AutoTokenizer.from_pretrained

    try:
        driver("model-id", object(), object())
    except RuntimeError:
        pass

    assert transformers_mod.AutoModel.from_pretrained is pre_model
    assert transformers_mod.AutoProcessor.from_pretrained is pre_processor
    assert transformers_mod.AutoTokenizer.from_pretrained is pre_tokenizer


def test_redirect_passes_through_for_other_model_names():
    class _OtherNameTransformer:
        captured = None

        def __init__(self, model_name, **kw):
            from transformers import AutoModel

            type(self).captured = AutoModel.from_pretrained("some-other/aux-model")

    driver, *_ = _build_driver(_OtherNameTransformer)
    sentinel = object()
    driver("primary/model-id", sentinel, object())
    assert _OtherNameTransformer.captured is not sentinel
    assert isinstance(_OtherNameTransformer.captured, tuple)
    assert _OtherNameTransformer.captured[0] == "orig"


def test_is_requested_model_name_handles_pathlib_path(tmp_path):
    target = tmp_path / "model_dir"
    target.mkdir()

    class _PathTransformer:
        last_calls = None

        def __init__(self, model_name, **kw):
            from transformers import AutoModel

            type(self).last_calls = AutoModel.from_pretrained(pathlib.Path(model_name))

    driver, *_ = _build_driver(_PathTransformer)
    sentinel_model = object()
    driver(str(target), sentinel_model, object())
    assert _PathTransformer.last_calls is sentinel_model


def test_is_requested_model_name_trailing_slash_local_path(tmp_path):
    target = tmp_path / "model_dir"
    target.mkdir()

    class _SlashTransformer:
        last_calls = None

        def __init__(self, model_name, **kw):
            from transformers import AutoModel

            type(self).last_calls = AutoModel.from_pretrained(str(target) + "/")

    driver, *_ = _build_driver(_SlashTransformer)
    sentinel_model = object()
    driver(str(target), sentinel_model, object())
    assert _SlashTransformer.last_calls is sentinel_model


def test_is_requested_model_name_returns_false_when_no_identifier():
    captured = {"args": None}

    class _NoNameTransformer:
        def __init__(self, model_name, **kw):
            from transformers import AutoModel

            captured["args"] = AutoModel.from_pretrained(some_other_kwarg = "x")

    driver, *_ = _build_driver(_NoNameTransformer)
    driver("primary/model-id", object(), object())
    assert isinstance(captured["args"], tuple)
    assert captured["args"][0] == "orig"
