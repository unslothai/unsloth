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


def test_just_below_boundary_uses_torch_dtype(monkeypatch):
    _stub_transformers(monkeypatch, "4.55.4")
    assert _has_torch_dtype_kwarg() is True


@pytest.mark.parametrize("version", ["4.56.0.dev0", "4.56.0rc1"])
def test_rename_prerelease_uses_dtype(monkeypatch, version):
    """A pre-release of the rename version sorts *below* ``4.56.0`` but already
    accepts (and prefers) ``dtype``; the release-tuple check must not fall back to
    the legacy name there, or it re-emits the deprecation warning it suppresses."""
    _stub_transformers(monkeypatch, version)
    assert _has_torch_dtype_kwarg() is False


def test_malformed_version_prefers_modern_name(monkeypatch):
    """A non-PEP440 __version__ raises InvalidVersion; the except branch must
    swallow it and default to the modern name rather than crash the embedder warm-up."""
    _stub_transformers(monkeypatch, "not-a-version")
    assert _has_torch_dtype_kwarg() is False
    assert dtype_kwargs("float16") == {"dtype": "float16"}


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
