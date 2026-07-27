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

"""The stray-pre-train-forward detector and its torch.compile cache reset.

A grad-enabled forward/backward run before ``trainer.train()`` poisons the
AOTAutograd backward-graph cache; the detector records it so train() can drop
that cache. These cover the idempotent-reinstall evidence guard, the reset's
chain-walk/teardown behaviour, and that the helper is importable at module
scope (every non-RL training entry point imports it). Runs under the GPU-free
``tests/conftest.py`` harness.
"""

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
    # Regression: the helper used to live only inside rl.py's RLTrainer_replacement template
    # string (exec'd into a generated trainer module), so importing it from a real module raised
    # ImportError and every non-RL consumer (SFT trainer.py, the plain-Trainer loop, the RL
    # template's own delegation) silently no-op'd. Pin it as an exported module-level symbol.
    from unsloth.models import _utils
    assert callable(_utils._unsloth_reset_stray_compile_cache)
    assert "_unsloth_reset_stray_compile_cache" in _utils.__all__


def test_fresh_install_starts_unseen():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    marker = m._unsloth_pretrain_marker
    assert marker["seen"] is False
    assert "hook" in marker  # a live hook is registered


def test_reinstall_with_live_hook_preserves_seen():
    # Re-entering get_peft_model/patch_peft_model after a grad-enabled probe must NOT wipe the
    # recorded poisoning, or train() skips the reset and the NaN/flat-loss bug returns.
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    hook = m._unsloth_pretrain_marker["hook"]
    m._unsloth_pretrain_marker["seen"] = True  # a probe the live hook recorded

    _unsloth_install_pretrain_detector(m)  # idempotent re-install
    marker = m._unsloth_pretrain_marker
    assert marker["seen"] is True  # evidence kept
    assert marker["hook"] is hook  # same hook, not double-registered


def test_reinstall_after_teardown_resets_and_reregisters():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    marker = m._unsloth_pretrain_marker
    marker["seen"] = True
    marker.pop("hook").remove()  # simulate teardown (what the reset does)

    _unsloth_install_pretrain_detector(m)  # no live hook -> fresh registration
    assert marker["seen"] is False  # reset for the new session
    assert "hook" in marker


def test_grad_enabled_forward_marks_seen_no_grad_does_not():
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    with torch.no_grad():
        m(torch.zeros(1, 2))
    assert m._unsloth_pretrain_marker["seen"] is False  # no backward graph -> clean
    m(torch.zeros(1, 2))  # grad-enabled forward poisons the cache
    assert m._unsloth_pretrain_marker["seen"] is True


def test_reset_clears_seen_and_warns_when_a_stray_forward_was_seen(monkeypatch):
    # Pin compile on: the reset only warns/resets when UNSLOTH_COMPILE_DISABLE != "1", which a
    # GPU-free CI env may set, so force it here to make the warn assertion deterministic.
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)
    m._unsloth_pretrain_marker["seen"] = True  # a stray pre-train forward
    trainer = _Trainer()
    trainer.model = m

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _unsloth_reset_stray_compile_cache(trainer)

    assert any("manual forward/backward" in str(w.message) for w in caught)
    assert "hook" not in m._unsloth_pretrain_marker  # hook torn down
    assert m._unsloth_pretrain_marker["seen"] is False  # evidence consumed


def test_reset_tears_down_hook_even_when_not_seen(monkeypatch):
    # The clean path still removes the one-shot hook so it adds no per-step cost, but must not
    # warn or reset Dynamo (nothing was poisoned). Pin compile on so the absent warning proves
    # seen==False is the reason, not a disabled-compile short circuit.
    monkeypatch.setenv("UNSLOTH_COMPILE_DISABLE", "0")
    m = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(m)  # seen stays False
    trainer = _Trainer()
    trainer.model = m

    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _unsloth_reset_stray_compile_cache(trainer)

    assert not any("manual forward/backward" in str(w.message) for w in caught)
    assert "hook" not in m._unsloth_pretrain_marker
    assert m._unsloth_pretrain_marker["seen"] is False


def test_reset_walks_wrapper_chain_to_reach_a_nested_marker():
    # The probe may have run on an inner wrapper (.model/.base_model/.module), not self.model.
    inner = torch.nn.Linear(2, 2)
    _unsloth_install_pretrain_detector(inner)
    inner._unsloth_pretrain_marker["seen"] = True

    class _Wrapper:  # e.g. a PEFT base_model wrapping the real module
        pass

    outer = _Wrapper()
    outer.base_model = inner
    trainer = _Trainer()
    trainer.model = outer

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _unsloth_reset_stray_compile_cache(trainer)

    assert "hook" not in inner._unsloth_pretrain_marker  # found and torn down through the chain
    assert inner._unsloth_pretrain_marker["seen"] is False
