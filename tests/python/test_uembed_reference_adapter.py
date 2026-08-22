# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Deterministic checks for the tracked pinned UEmbed reference adapter."""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest


_FIXTURE_DIR = Path(__file__).with_name("fixtures") / "uembed_reference"
_ADAPTER_PATH = _FIXTURE_DIR / "reference_module.py"
_SNAPSHOT_PATH = _FIXTURE_DIR / "qwen35_embedding.py"
_PARITY_PATH = Path(__file__).with_name("test_uembed_parity.py")
_PINNED_MODULE_NAME = "uembed_pinned_upstream"
_SNAPSHOT_SHA256 = "689e1968d526fe8750882b2a50045aa980d1328e7b3c65068e52954178d35b85"


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_parity_defaults_to_tracked_reference_fixture(monkeypatch):
    """A clean checkout resolves parity to the immutable fixture without an env path."""
    monkeypatch.delenv("UNSLOTH_UEMBED_REFERENCE_MODULE", raising = False)
    parity = _load_file("uembed_parity_default_reference_test", _PARITY_PATH)

    assert Path(parity._reference_module_path()) == _ADAPTER_PATH
    assert _ADAPTER_PATH.parent.relative_to(_PARITY_PATH.parent) == Path(
        "fixtures/uembed_reference"
    )
    assert hashlib.sha256(_SNAPSHOT_PATH.read_bytes()).hexdigest() == _SNAPSHOT_SHA256


def test_reference_adapter_registers_dynamic_module_before_execution(monkeypatch):
    """The loader contract used by Transformers 5.4 requires pre-exec registration."""
    adapter_spec = importlib.util.spec_from_file_location("uembed_reference_adapter_test", _ADAPTER_PATH)
    assert adapter_spec is not None and adapter_spec.loader is not None
    real_spec_from_file_location = importlib.util.spec_from_file_location

    class RegistrationCheckingLoader:
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            assert sys.modules.get(module.__name__) is module

            class Qwen35Embedder:
                def process(self, inputs):
                    return inputs

            module.Qwen35Embedder = Qwen35Embedder

    def checked_spec(name, path, *args, **kwargs):
        if name == _PINNED_MODULE_NAME:
            return importlib.machinery.ModuleSpec(name, RegistrationCheckingLoader(), origin = str(path))
        return real_spec_from_file_location(name, path, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "spec_from_file_location", checked_spec)
    module = importlib.util.module_from_spec(adapter_spec)
    try:
        adapter_spec.loader.exec_module(module)
        assert module.Qwen35Embedder.__mro__[1] is sys.modules[_PINNED_MODULE_NAME].Qwen35Embedder
        embedder = module.Qwen35Embedder.__new__(module.Qwen35Embedder)
        assert embedder.encode(["query", {"image": "page.png"}]) == [
            {"text": "query"},
            {"image": "page.png"},
        ]
    finally:
        sys.modules.pop(_PINNED_MODULE_NAME, None)


def test_reference_adapter_resolves_qwen35_embedder_with_transformers_5_4():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("qwen_vl_utils")
    if tuple(map(int, transformers.__version__.split(".")[:2])) < (5, 4):
        pytest.skip("the pinned upstream source requires Transformers 5.4 or newer")

    try:
        module = _load_file("uembed_reference_transformers_5_4_test", _ADAPTER_PATH)
        upstream = sys.modules[_PINNED_MODULE_NAME]
        assert module.Qwen35Embedder.__mro__[1] is upstream.Qwen35Embedder
        assert upstream.Qwen35Embedder.__module__ == _PINNED_MODULE_NAME
        assert callable(module.Qwen35Embedder.encode)
    finally:
        sys.modules.pop("uembed_reference_transformers_5_4_test", None)
        sys.modules.pop(_PINNED_MODULE_NAME, None)


def test_reference_comparison_detaches_without_changing_values():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    parity = _load_file("uembed_parity_comparison_test", _PARITY_PATH)
    tensor = torch.tensor([[1.25, -2.5]], dtype = torch.bfloat16, requires_grad = True)

    actual = parity._comparison_array(tensor)

    np.testing.assert_array_equal(actual, np.array([[1.25, -2.5]], dtype = np.float32))
    assert tensor.dtype == torch.bfloat16
    assert tensor.requires_grad
