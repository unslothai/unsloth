"""PR #11238, commit ee46e1305: `_unsloth_train_if_needed` replaces the per-micro-step
`model.train()` inside the rewritten `Trainer.training_step`.

These tests do not measure time. They assert the *behavioural* difference between calling
`model.train()` on every micro-step and asserting train mode once, which is the only thing
that can make the optimisation unsafe.

Run with the arm on PYTHONPATH:
    PYTHONPATH=<tree> pytest -q test_train_mode_asserted_once.py
On the base tree the helper does not exist and every test that needs it is skipped; the
`_reference_*` tests describe what the stock loop does and pass on both arms.
"""

import copy

import pytest
import torch.nn as nn

from unsloth.models import _utils as U

HELPER = getattr(U, "_unsloth_train_if_needed", None)
needs_helper = pytest.mark.skipif(HELPER is None, reason = "base tree: no _unsloth_train_if_needed")


class _Tree(nn.Module):
    """A root with a deep child, so a submodule can be flipped independently."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))

    def forward(self, x):
        return self.block(x)


class _Wrapper(nn.Module):
    """The shape of DDP / accelerate: `training_step` is handed this, but other code
    (a TrainerCallback, TRL, an eval loop) holds `.module` instead."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)


def _modes(m):
    return {n or "<root>": mod.training for n, mod in m.named_modules()}


# --------------------------------------------------------------------------- A1/A2
@needs_helper
def test_first_call_asserts_train_mode_on_every_module():
    m = _Tree()
    m.eval()
    assert not any(_modes(m).values())
    HELPER(m)
    assert all(_modes(m).values())
    assert getattr(m, "_unsloth_train_mode_asserted", False) is True


@needs_helper
def test_root_eval_rearms_the_walk():
    m = _Tree()
    HELPER(m)
    m.eval()  # evaluation loop / for_inference
    assert m.training is False
    HELPER(m)
    assert all(_modes(m).values()), "a root .eval() must be repaired by the next micro-step"


@needs_helper
def test_marker_is_not_in_state_dict_and_does_not_survive_a_fresh_module():
    m = _Tree()
    HELPER(m)
    assert "_unsloth_train_mode_asserted" not in m.state_dict()
    fresh = _Tree()
    assert getattr(fresh, "_unsloth_train_mode_asserted", None) is None
    clone = copy.deepcopy(m)
    # a deepcopy carries the marker; it also carries .training, so the pair stays consistent
    assert clone.training == m.training


# --------------------------------------------------------------------------- A3
def test_reference_submodule_only_eval_is_repaired_by_stock_train():
    """What the stock `model.train()` every micro-step does. Passes on both arms."""
    m = _Tree()
    m.train()
    m.block[1].eval()
    assert m.block[1].training is False
    m.train()  # the stock loop, next micro-step
    assert m.block[1].training is True


@needs_helper
def test_submodule_only_eval_is_NOT_repaired_by_the_helper():
    """Documented semantic change: a submodule flipped to eval on its own keeps that mode."""
    m = _Tree()
    HELPER(m)
    m.block[1].eval()
    HELPER(m)  # next micro-step
    assert m.block[1].training is False, "if this repairs, the PR body's claim is wrong"


# --------------------------------------------------------------------------- A4
def test_reference_wrapper_inner_eval_is_repaired_by_stock_train():
    inner = _Tree()
    w = _Wrapper(inner)
    w.train()
    inner.eval()  # a callback / eval loop holding `self.model`, not `self.model_wrapped`
    assert w.training is True and inner.training is False
    w.train()  # the stock loop repairs the whole tree
    assert inner.training is True
    assert all(_modes(w).values())


@needs_helper
@pytest.mark.skipif(
    getattr(U, "_unsloth_wrappees_are_in_train_mode", None) is not None,
    reason = "the wrappee check repairs this; kept to document what it repairs",
)
def test_wrapper_inner_eval_is_permanently_stale_under_the_helper():
    """THE defect candidate. `training_step` is handed `self.model_wrapped`; anything that
    calls `.eval()` on the inner `self.model` leaves the wrapper's `.training` True and the
    marker set, so the walk never runs again and training silently continues with dropout
    disabled and every module in eval mode."""
    inner = _Tree()
    w = _Wrapper(inner)
    HELPER(w)
    inner.eval()
    for _ in range(100):  # a hundred micro-steps later
        HELPER(w)
    assert w.training is True
    assert inner.training is False
    assert w.module.block[1].training is False


# --------------------------------------------------------------------------- misc
@needs_helper
def test_helper_survives_a_model_that_rejects_attribute_assignment():
    class _Frozen(_Tree):
        def __setattr__(self, k, v):
            if k == "_unsloth_train_mode_asserted":
                raise AttributeError("read-only")
            super().__setattr__(k, v)

    m = _Frozen()
    HELPER(m)  # must not raise
    assert m.training is True
    HELPER(m)  # walks every time, but never errors
    assert m.training is True


# --------------------------------------------------------------------------- the fix
WRAPPEE_CHECK = getattr(U, "_unsloth_wrappees_are_in_train_mode", None)
needs_fix = pytest.mark.skipif(WRAPPEE_CHECK is None, reason = "tree without the wrappee check")


@needs_fix
def test_wrapper_inner_eval_is_repaired_once_wrappees_are_checked():
    """Regression for the DDP + eval_strategy case: `Trainer.evaluation_loop` puts
    `self.model` in eval, `training_step` is handed `self.model_wrapped`, and the whole model
    would otherwise stay in eval mode for the rest of the run."""
    inner = _Tree()
    w = _Wrapper(inner)
    HELPER(w)
    inner.eval()
    HELPER(w)
    assert inner.training is True
    assert all(_modes(w).values())


@needs_fix
def test_wrappee_check_is_cheap_and_stops_at_the_first_non_wrapper():
    m = _Tree()
    HELPER(m)
    # `_Tree` has no .module / ._orig_mod / ._fsdp_wrapped_module, so the check answers
    # immediately and the skip is still taken.
    assert WRAPPEE_CHECK(m) is True
    calls = []
    orig = nn.Module.train
    try:
        nn.Module.train = lambda self, mode = True: calls.append(1) or orig(self, mode)
        HELPER(m)
    finally:
        nn.Module.train = orig
    assert calls == [], "an unwrapped model must still skip the walk"


@needs_fix
def test_wrappee_check_follows_a_nested_wrapper_chain():
    inner = _Tree()
    w = _Wrapper(_Wrapper(inner))
    HELPER(w)
    assert WRAPPEE_CHECK(w) is True
    inner.eval()
    assert WRAPPEE_CHECK(w) is False
