# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Below transformers 5.6.0 a tokenizer loaded by repo id asks the Hub for its tags; a 429 there must
not fail the load, and any other error must still surface."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path

import pytest

transformers = pytest.importorskip("transformers")
huggingface_hub = pytest.importorskip("huggingface_hub")

from packaging.version import Version  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TF_VERSION = Version(transformers.__version__.split("+")[0].split("rc")[0].split(".dev")[0])
AFFECTED = TF_VERSION < Version("5.6.0")
FLAG = "_unsloth_mistral_regex_hub_guard"


def _owner():
    for module_name, class_name in (
        ("transformers.tokenization_utils_tokenizers", "TokenizersBackend"),
        ("transformers.tokenization_utils_base", "PreTrainedTokenizerBase"),
    ):
        try:
            owner = getattr(importlib.import_module(module_name), class_name)
        except Exception:
            continue
        if isinstance(owner.__dict__.get("_patch_mistral_regex"), classmethod):
            return owner
    pytest.skip("this transformers has no _patch_mistral_regex")


def _load_import_fixes():
    path = os.environ.get("UNSLOTH_IMPORT_FIXES", str(REPO / "unsloth" / "import_fixes.py"))
    spec = importlib.util.spec_from_file_location("unsloth_import_fixes_mistral_hub", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _RateLimited(Exception):
    pass


@pytest.fixture
def owner(monkeypatch):
    owner = _owner()
    # conftest imports unsloth, which already wrapped it: start each test from upstream.
    function = owner.__dict__["_patch_mistral_regex"].__func__
    while getattr(function, FLAG, False):
        function = function.__wrapped__
    monkeypatch.setattr(owner, "_patch_mistral_regex", classmethod(function))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising = False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising = False)
    try:
        from transformers.utils import hub as tf_hub
        monkeypatch.setattr(tf_hub, "_is_offline_mode", False, raising = False)
    except Exception:
        pass
    return owner


def _guarded(owner):
    func = owner.__dict__["_patch_mistral_regex"].__func__
    return getattr(func, FLAG, False)


def _rate_limited(*args, **kwargs):
    raise _RateLimited("429 Client Error: Too Many Requests")


def test_a_rate_limited_tag_lookup_keeps_the_tokenizer(owner, monkeypatch):
    _load_import_fixes().fix_transformers_mistral_regex_hub_error()
    if not AFFECTED:
        pytest.skip("transformers 5.6.0+ guards the lookup itself")
    monkeypatch.setattr(huggingface_hub, "model_info", _rate_limited)
    tokenizer = object()
    assert owner._patch_mistral_regex(tokenizer, "unsloth/Qwen3-4B") is tokenizer


def test_unguarded_transformers_fails_the_load(owner, monkeypatch):
    fixes = _load_import_fixes()
    fixes.fix_transformers_mistral_regex_hub_error()
    if not _guarded(owner):
        pytest.skip("this transformers already guards the lookup")
    original = owner.__dict__["_patch_mistral_regex"].__func__.__wrapped__
    monkeypatch.setattr(owner, "_patch_mistral_regex", classmethod(original))
    monkeypatch.setattr(huggingface_hub, "model_info", _rate_limited)
    with pytest.raises(_RateLimited):
        owner._patch_mistral_regex(object(), "unsloth/Qwen3-4B")


def test_errors_outside_the_tag_lookup_still_surface(owner, monkeypatch):
    _load_import_fixes().fix_transformers_mistral_regex_hub_error()
    if not _guarded(owner):
        pytest.skip("this transformers already guards the lookup")

    class _Info:
        tags = ["base_model:mistralai/Mistral-7B"]

    monkeypatch.setattr(huggingface_hub, "model_info", lambda *a, **k: _Info())
    from transformers.utils import hub as tf_hub

    def _broken(*args, **kwargs):
        raise RuntimeError("config read failed")

    monkeypatch.setattr(tf_hub, "cached_file", _broken)
    monkeypatch.setattr(
        importlib.import_module(owner.__module__), "cached_file", _broken, raising = False
    )
    with pytest.raises(RuntimeError, match = "config read failed"):
        owner._patch_mistral_regex(object(), "mistralai/Mistral-7B")


def test_applying_twice_wraps_once(owner):
    fixes = _load_import_fixes()
    fixes.fix_transformers_mistral_regex_hub_error()
    first = owner.__dict__["_patch_mistral_regex"].__func__
    fixes.fix_transformers_mistral_regex_hub_error()
    assert owner.__dict__["_patch_mistral_regex"].__func__ is first
