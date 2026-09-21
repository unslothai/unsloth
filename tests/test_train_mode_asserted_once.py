"""Behavioural difference between calling `model.train()` every micro-step and asserting
train mode once, which is the only thing that can make that optimisation unsafe.
"""

import copy

import pytest

# These runners do not all ship torch; skip the module rather than erroring at collection.
pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")
U = pytest.importorskip("unsloth.models._utils")

HELPER = getattr(U, "_unsloth_train_if_needed", None)
needs_helper = pytest.mark.skipif(HELPER is None, reason = "base tree: no _unsloth_train_if_needed")


class _Tree(nn.Module):
    """Root with a deep child, so a submodule can be flipped on its own."""

    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))

    def forward(self, x):
        return self.block(x)


class _Wrapper(nn.Module):
    """DDP shape: `training_step` gets this, other code holds `.module`."""

    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)


def _modes(m):
    return {n or "<root>": mod.training for n, mod in m.named_modules()}


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
    m.eval()
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
    assert clone.training == m.training


def test_reference_submodule_only_eval_is_repaired_by_stock_train():
    """What the stock per-micro-step `model.train()` does. Passes on both arms."""
    m = _Tree()
    m.train()
    m.block[1].eval()
    assert m.block[1].training is False
    m.train()
    assert m.block[1].training is True


@needs_helper
def test_submodule_only_eval_is_NOT_repaired_by_the_helper():
    """Documented semantic change: a submodule flipped to eval on its own keeps that mode."""
    m = _Tree()
    HELPER(m)
    m.block[1].eval()
    HELPER(m)
    assert m.block[1].training is False, "if this repairs, the PR body's claim is wrong"


def test_reference_wrapper_inner_eval_is_repaired_by_stock_train():
    inner = _Tree()
    w = _Wrapper(inner)
    w.train()
    inner.eval()
    assert w.training is True and inner.training is False
    w.train()
    assert inner.training is True
    assert all(_modes(w).values())


@needs_helper
@pytest.mark.skipif(
    getattr(U, "_unsloth_wrappees_are_in_train_mode", None) is not None,
    reason = "the wrappee check repairs this; kept to document what it repairs",
)
def test_wrapper_inner_eval_is_permanently_stale_under_the_helper():
    """A wrapper's own `.training` never flips when the module inside it is eval'd, so a
    root-flag check strands the whole model in eval for the rest of the run."""
    inner = _Tree()
    w = _Wrapper(inner)
    HELPER(w)
    inner.eval()
    for _ in range(100):
        HELPER(w)
    assert w.training is True
    assert inner.training is False
    assert w.module.block[1].training is False


@needs_helper
def test_helper_survives_a_model_that_rejects_attribute_assignment():
    class _Frozen(_Tree):
        def __setattr__(self, k, v):
            if k == "_unsloth_train_mode_asserted":
                raise AttributeError("read-only")
            super().__setattr__(k, v)

    m = _Frozen()
    HELPER(m)
    assert m.training is True
    HELPER(m)
    assert m.training is True


WRAPPEE_CHECK = getattr(U, "_unsloth_wrappees_are_in_train_mode", None)
needs_fix = pytest.mark.skipif(WRAPPEE_CHECK is None, reason = "tree without the wrappee check")


@needs_fix
def test_wrapper_inner_eval_is_repaired_once_wrappees_are_checked():
    """DDP + eval_strategy: evaluation_loop evals `self.model`, training_step gets
    `self.model_wrapped`."""
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
