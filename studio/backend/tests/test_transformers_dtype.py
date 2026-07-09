# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for the version-safe torch_dtype/dtype kwarg helper."""

import sys
import types

import pytest

from utils.transformers_dtype import _has_torch_dtype_kwarg, dtype_kwargs


@pytest.fixture(autouse = True)
def _clear_cache():
    _has_torch_dtype_kwarg.cache_clear()
    yield
    _has_torch_dtype_kwarg.cache_clear()


def _stub_transformers(monkeypatch, version):
    stub = types.ModuleType("transformers")
    stub.__version__ = version
    monkeypatch.setitem(sys.modules, "transformers", stub)


def test_old_transformers_uses_torch_dtype(monkeypatch):
    _stub_transformers(monkeypatch, "4.51.3")
    assert _has_torch_dtype_kwarg() is True
    assert dtype_kwargs("float16") == {"torch_dtype": "float16"}


def test_new_transformers_uses_dtype(monkeypatch):
    _stub_transformers(monkeypatch, "4.57.6")
    assert _has_torch_dtype_kwarg() is False
    assert dtype_kwargs("float16") == {"dtype": "float16"}


def test_rename_boundary_uses_dtype(monkeypatch):
    _stub_transformers(monkeypatch, "4.56.0")
    assert _has_torch_dtype_kwarg() is False


def test_survives_docstring_stripping(monkeypatch):
    """Regression: python -OO / PYTHONOPTIMIZE=2 sets __doc__ to None on every
    class, which broke the old __doc__-sniffing implementation into always
    reporting the modern ``dtype`` kwarg, TypeError-ing on the transformers
    floor. The version check must not depend on __doc__ at all."""
    _stub_transformers(monkeypatch, "4.51.3")
    monkeypatch.setattr(sys.modules["transformers"], "__doc__", None, raising = False)
    assert _has_torch_dtype_kwarg() is True


def test_missing_transformers_prefers_modern_name(monkeypatch):
    monkeypatch.delitem(sys.modules, "transformers", raising = False)
    real_import = __import__

    def _raise(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("no transformers")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise)
    assert _has_torch_dtype_kwarg() is False
    assert dtype_kwargs("float16") == {"dtype": "float16"}
