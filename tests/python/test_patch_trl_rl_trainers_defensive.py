# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests: _patch_trl_rl_trainers must never raise.

The wrapper in unsloth/models/rl.py ring-fences the impl so direct
callers (CI shims, downstream tools) don't have to. Lock that
contract here.
"""

from __future__ import annotations

import pytest


pytest.importorskip("trl")


def _import_helpers():
    try:
        from unsloth.models.rl import (
            _patch_trl_rl_trainers,
            _patch_trl_rl_trainers_impl,
        )
    except ImportError as e:
        pytest.skip(f"unsloth.models.rl helpers not importable: {e}")
    return _patch_trl_rl_trainers, _patch_trl_rl_trainers_impl


def test_patch_trl_rl_trainers_swallows_unknown_trainer_name():
    wrapper, _impl = _import_helpers()
    assert wrapper("definitely_not_a_real_trainer_xyz") is None


def test_patch_trl_rl_trainers_swallows_garbage_input():
    wrapper, _impl = _import_helpers()
    for bad in ("", "..", "trainer with space", "sft_trainer; rm -rf /"):
        assert wrapper(bad) is None, f"raised on input: {bad!r}"


def test_impl_is_separately_exposed():
    # Power users can still call the impl directly for the raising path.
    _wrapper, impl = _import_helpers()
    assert callable(impl)


def test_wrapper_delegates_to_impl(monkeypatch):
    from unsloth.models import rl as _rl

    sentinel = object()
    calls = []

    def _fake_impl(trainer_file):
        calls.append(trainer_file)
        return sentinel

    monkeypatch.setattr(_rl, "_patch_trl_rl_trainers_impl", _fake_impl)
    assert _rl._patch_trl_rl_trainers("sft_trainer") is sentinel
    assert calls == ["sft_trainer"]


def test_wrapper_swallows_impl_exception(monkeypatch):
    from unsloth.models import rl as _rl

    def _boom(_trainer_file):
        raise RuntimeError("simulated TRL 1.x rename failure")

    monkeypatch.setattr(_rl, "_patch_trl_rl_trainers_impl", _boom)
    assert _rl._patch_trl_rl_trainers("sft_trainer") is None
