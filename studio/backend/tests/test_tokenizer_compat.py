# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for utils.tokenizer_compat.

Covers the crash reported on unsloth/Qwen3.6-35B-A3B-MLX-8bit: a model whose
tokenizer_config.json ships extra_special_tokens as a JSON array instead of an
object makes transformers raise "'list' object has no attribute 'keys'" during
tokenizer init. install_extra_special_tokens_compat() coerces a non-dict value
to {} so the load succeeds.
"""

import logging
import sys
import types
from pathlib import Path

import pytest

# Self-contained: put backend root on sys.path and stub the structlog-backed
# ``loggers`` module so the unit under test imports without structlog installed
# (mirrors tests/test_transformers_version.py).
_backend_root = Path(__file__).resolve().parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

import utils.tokenizer_compat as tc
from utils.tokenizer_compat import install_extra_special_tokens_compat

tub = pytest.importorskip("transformers.tokenization_utils_base")


def _make_mixin():
    """A real SpecialTokensMixin instance (its __init__ sets _special_tokens_map)."""

    class _Dummy(tub.SpecialTokensMixin):
        pass

    return _Dummy()


class _RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, msg, *args, **kwargs):
        self.warnings.append(msg % args if args else msg)


@pytest.fixture
def recording_logger(monkeypatch):
    rec = _RecordingLogger()
    monkeypatch.setattr(tc, "logger", rec)
    return rec


def test_method_exists_and_install_is_idempotent():
    # The method we patch must exist in the installed transformers; otherwise the
    # bug shape (and this shim) would not apply.
    assert hasattr(tub.SpecialTokensMixin, "_set_model_specific_special_tokens")
    assert install_extra_special_tokens_compat() is True
    first = tub.SpecialTokensMixin._set_model_specific_special_tokens
    # A second install must not re-wrap.
    assert install_extra_special_tokens_compat() is True
    assert tub.SpecialTokensMixin._set_model_specific_special_tokens is first
    assert hasattr(first, "__wrapped__")  # original kept reachable


def test_original_method_crashes_on_list():
    install_extra_special_tokens_compat()
    original = tub.SpecialTokensMixin._set_model_specific_special_tokens.__wrapped__
    d = _make_mixin()
    with pytest.raises(AttributeError):
        original(d, [])  # the exact "'list' object has no attribute 'keys'" shape


def test_patched_coerces_empty_list(recording_logger):
    install_extra_special_tokens_compat()
    d = _make_mixin()
    d._set_model_specific_special_tokens([])  # must not raise
    assert d.extra_special_tokens == {}
    assert len(recording_logger.warnings) == 1


def test_patched_coerces_non_empty_list(recording_logger):
    install_extra_special_tokens_compat()
    d = _make_mixin()
    d._set_model_specific_special_tokens(["<extra>"])  # must not raise
    assert d.extra_special_tokens == {}
    assert len(recording_logger.warnings) == 1


def test_patched_preserves_valid_dict(recording_logger):
    install_extra_special_tokens_compat()
    d = _make_mixin()
    d._set_model_specific_special_tokens({"img_token": "<img>"})
    assert "img_token" in d.SPECIAL_TOKENS_ATTRIBUTES
    assert d._special_tokens_map.get("img_token") == "<img>"
    assert recording_logger.warnings == []  # no coercion for a well-formed dict
