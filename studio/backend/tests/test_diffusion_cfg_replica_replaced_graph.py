# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A resident whole-step graph the speed layer armed before placement, then replaced after it, leaves no trace in the
offload hook chain, so a CFG-parallel replica still builds. CPU only (the replica shares the CPU device).

Automatic placement previews a layer split while the speed layer runs, so the denoiser is set up resident and its
whole-step graph armed on the forward slot. When the final placement keeps one card plus a CFG replica, the block
offload hooks wrap that slot (diffusers wraps from the original forward, so its ``__wrapped__`` names the graph) and
``arm_after_placement`` swaps the graph for a block-streamed one. The replica builder peels offload wrappers by
``__wrapped__``: a stale link to the retired graph made it refuse the replica on the first guided render (Kaggle 2x T4,
FLUX.2-klein-base-4B: one card at 2.9 s/step instead of 1.9 with the replica)."""

from __future__ import annotations

import pytest
import torch

hooks = pytest.importorskip("diffusers.hooks", reason = "diffusers is not importable here")
HookRegistry, ModelHook = hooks.HookRegistry, hooks.ModelHook

from core.inference import diffusion_cuda_graph as cg
from core.inference import diffusion_multigpu as mg


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    monkeypatch.delenv(mg.MULTIGPU_ENV, raising = False)
    monkeypatch.delenv(mg.CFG_PARALLEL_ENV, raising = False)
    yield


class _Dit(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, hidden_states):
        return self.proj(hidden_states)


class _NewForwardHook(ModelHook):
    def new_forward(self, module, *args, **kwargs):
        return self.fn_ref.original_forward(*args, **kwargs)


def _wrapped_chain(fn):
    out, seen = [], 0
    while fn is not None and seen < 16:
        seen += 1
        out.append(fn)
        fn = getattr(fn, "__wrapped__", None)
    return out


def _armed_then_hooked(*hooks):
    """The Automatic load order: resident graph armed, offload hooks registered over it, graph retired."""
    dit = _Dit()
    early = cg.GraphedForward(dit).enable()
    assert dit.__dict__["forward"] is early
    registry = HookRegistry.check_if_exists_or_initialize(dit)
    for name, hook in hooks:
        registry.register_hook(hook, name)
    # arm_graphs_after_placement -> arm_after_placement -> uninstall_all
    cg.uninstall_all((early,))
    return dit, early


@pytest.mark.parametrize(
    "hooks",
    [
        (("group_offloading", ModelHook()),),
        (("group_offloading", ModelHook()), ("layer_execution_tracker", ModelHook())),
        (("group_offloading", _NewForwardHook()),),
    ],
    ids = ["one_hook", "two_hooks", "new_forward_hook"],
)
def test_retired_graph_leaves_no_link_in_the_hook_chain(hooks):
    dit, early = _armed_then_hooked(*hooks)
    slot = dit.__dict__["forward"]
    assert all(link is not early for link in _wrapped_chain(slot))
    for ref in dit._diffusers_hook._fn_refs:
        assert getattr(ref, "forward", None) is not early
        assert getattr(ref, "original_forward", None) is not early
    # the attributes update_wrapper copied from the graph do not keep it (or its cache) reachable
    for link in _wrapped_chain(slot):
        state = getattr(link, "__dict__", None) or {}
        assert "cache" not in state and "module" not in state
    x = torch.randn(2, 4)
    torch.testing.assert_close(dit(x), dit.proj(x))
    assert early.stats["eager_calls"] == 0


def test_replica_builds_after_the_resident_graph_was_replaced():
    dit, _early = _armed_then_hooked(("group_offloading", ModelHook()))
    router = mg.CfgParallelRouter(dit, primary = "cpu", secondary = "cpu")
    assert router.had_override
    dit.forward = router
    replica = mg.build_replica(dit, torch.device("cpu"))
    assert replica is not dit
    x = torch.randn(2, 4)
    torch.testing.assert_close(replica(x), dit.proj(x))


def test_a_live_unknown_root_override_is_still_refused():
    """Only the retired graph is peeled: an override nobody retired still declines the replica."""
    dit = _Dit()
    dit.__dict__["forward"] = lambda hidden_states: hidden_states
    HookRegistry.check_if_exists_or_initialize(dit).register_hook(ModelHook(), "group_offloading")
    router = mg.CfgParallelRouter(dit, primary = "cpu", secondary = "cpu")
    dit.forward = router
    with pytest.raises(RuntimeError, match = "cannot be reproduced on a replica"):
        mg.build_replica(dit, torch.device("cpu"))
