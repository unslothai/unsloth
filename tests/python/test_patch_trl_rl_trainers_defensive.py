# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression: _patch_trl_rl_trainers (the ring-fencing wrapper in
unsloth/models/rl.py) must never raise."""

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
    # The impl stays directly callable for the raising path.
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


def test_grpo_config_sibling_module_import_is_patched(tmp_path):
    import unsloth  # noqa: F401
    from trl import GRPOConfig as top_config
    from trl.trainer import GRPOConfig as trainer_config
    from trl.trainer.grpo_config import GRPOConfig as config_module_config
    from trl.trainer.grpo_trainer import GRPOConfig as trainer_module_config

    assert top_config is trainer_config
    assert top_config is trainer_module_config
    assert top_config is config_module_config

    args = config_module_config(output_dir = str(tmp_path))
    assert hasattr(args, "unsloth_grpo_mini_batch")
    assert args.unsloth_grpo_mini_batch is None
