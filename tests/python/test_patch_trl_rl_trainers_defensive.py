# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for unsloth.models.rl._patch_trl_rl_trainers.

History: TRL 1.x moved several trainers into trl.experimental and left
thin wrappers behind under trl.trainer. Calling _patch_trl_rl_trainers
directly on one of these wrappers used to raise (e.g. inspect.getsource
fails on dynamically-generated wrappers, or a renamed *Config attribute
ends up missing), which forced the CI shim in
.github/workflows/consolidated-tests-ci.yml to ring-fence the call with
its own `pytest.skip(f"_patch_trl_rl_trainers raised: ...")` block.

The umbrella patch_trl_rl_trainers() always wrapped each call in
try/except + warning_once, but direct callers did not have that
luxury. The wrapper added in this file's sibling rl.py mirrors that
behaviour so:

  - direct callers (CI shim, downstream tools, end-user scripts) never
    see a raw exception from this helper;
  - the underlying logic still lives in _patch_trl_rl_trainers_impl
    so future refactors can target it directly when needed.

These tests guard the contract: the public name must never raise on
any string we hand it.
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
    """A nonsense trainer file name must not raise."""
    wrapper, _impl = _import_helpers()
    # The function logs + returns None; never raises.
    assert wrapper("definitely_not_a_real_trainer_xyz") is None


def test_patch_trl_rl_trainers_swallows_garbage_input():
    """Even a clearly malformed input (special chars, empty) must not
    raise. The wrapper catches every Exception and routes it through
    logger.info."""
    wrapper, _impl = _import_helpers()
    for bad in ("", "..", "trainer with space", "sft_trainer; rm -rf /"):
        assert wrapper(bad) is None, f"raised on input: {bad!r}"


def test_impl_is_separately_exposed():
    """Power users who want the raising behaviour for their own
    diagnostics can still call _patch_trl_rl_trainers_impl directly."""
    _wrapper, impl = _import_helpers()
    assert callable(impl)


def test_wrapper_delegates_to_impl(monkeypatch):
    """When impl returns cleanly, wrapper returns the same value."""
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
    """When impl raises, wrapper logs and returns None (does NOT
    propagate). This is the contract the CI shim relies on."""
    from unsloth.models import rl as _rl

    def _boom(_trainer_file):
        raise RuntimeError("simulated TRL 1.x rename failure")

    monkeypatch.setattr(_rl, "_patch_trl_rl_trainers_impl", _boom)
    # No exception should escape:
    assert _rl._patch_trl_rl_trainers("sft_trainer") is None
