# Unsloth - 2x faster, 60% less VRAM LLM training and finetuning
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.

"""A grad-enabled forward before train() poisons the AOTAutograd backward cache; train() resets it."""

from __future__ import annotations

import warnings

import unsloth  # noqa: F401  (installs the unsloth patches the functions live behind)

import torch

from unsloth.models._utils import (
    _unsloth_install_pretrain_detector,
    _unsloth_reset_stray_compile_cache,
)


class _Trainer:
    """Minimal ``self`` stand-in: the reset only reads ``self.model``."""


def test_reset_helper_is_importable_and_exported():
    from unsloth.models import _utils
    assert callable(_utils._unsloth_reset_stray_compile_cache)
    assert "_unsloth_reset_stray_compile_cache" in _utils.__all__


def test_fresh_install_starts_unseen():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    marker = m._unsloth_pretrain_marker
    assert marker["seen"] is False
    assert "hook" in marker


def test_reinstall_with_live_hook_preserves_seen():
    # Re-entering get_peft_model must keep the recorded poisoning, or train() skips the reset.
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    hook = m._unsloth_pretrain_marker["hook"]
    m._unsloth_pretrain_marker["seen"] = True

    _unsloth_install_pretrain_detector(m)
    marker = m._unsloth_pretrain_marker
    assert marker["seen"] is True
    assert marker["hook"] is hook


def test_reinstall_after_teardown_resets_and_reregisters():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    marker = m._unsloth_pretrain_marker
    marker["seen"] = True
    marker.pop("hook").remove()

    _unsloth_install_pretrain_detector(m)
    assert marker["seen"] is False
    assert "hook" in marker


def test_grad_enabled_forward_marks_seen_no_grad_does_not():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    with torch.no_grad():
        m(torch.zeros(1, 2))
    assert m._unsloth_pretrain_marker["seen"] is False
    m(torch.zeros(1, 2))
    assert m._unsloth_pretrain_marker["seen"] is True


def test_reset_clears_seen_and_warns_when_a_stray_forward_was_seen(monkeypatch):
    # Pin compile on: the reset is skipped when UNSLOTH_COMPILE_DISABLE=1, which CI may set.
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    m._unsloth_pretrain_marker["seen"] = True
    trainer = _Trainer()
    trainer.model = m

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _unsloth_reset_stray_compile_cache(trainer)

    assert any("manual forward/backward" in str(w.message) for w in caught)
    assert "hook" not in m._unsloth_pretrain_marker
    assert m._unsloth_pretrain_marker["seen"] is False


def test_reset_tears_down_hook_even_when_not_seen(monkeypatch):
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    trainer = _Trainer()
    trainer.model = m

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _unsloth_reset_stray_compile_cache(trainer)

    assert not any("manual forward/backward" in str(w.message) for w in caught)
    assert "hook" not in m._unsloth_pretrain_marker
    assert m._unsloth_pretrain_marker["seen"] is False


def test_reset_walks_wrapper_chain_to_reach_a_nested_marker():
    inner = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(inner)
    inner._unsloth_pretrain_marker["seen"] = True

    class _Wrapper:
        pass

    outer = _Wrapper()
    outer.base_model = inner
    trainer = _Trainer()
    trainer.model = outer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _unsloth_reset_stray_compile_cache(trainer)

    assert "hook" not in inner._unsloth_pretrain_marker
    assert inner._unsloth_pretrain_marker["seen"] is False
