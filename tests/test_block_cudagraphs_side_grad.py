# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.

"""Block CUDA graphs skip calls whose side input takes a gradient (Mllama cross-attention): auto mode reports the
reason (CPU); forced graphs on a GPU train like eager, side gradient included."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_block_cudagraphs_auto as auto  # noqa: E402

bc = auto.bc


@pytest.fixture(autouse = True)
def _block_env(monkeypatch):
    for key in (
        "UNSLOTH_BLOCK_COMPILE",
        "UNSLOTH_COMPILE_DISABLE",
        "UNSLOTH_FORCE_FLOAT32",
        "UNSLOTH_BLOCK_CUDAGRAPHS",
    ):
        monkeypatch.delenv(key, raising = False)
    monkeypatch.setattr(bc, "_CUDAGRAPH_STATE", bc._new_cudagraph_state())
    monkeypatch.setattr(bc, "_CUDAGRAPH_MODE", "auto")


def test_auto_keeps_graphs_off_for_a_grad_side_input(monkeypatch):
    layers, calls, _, timed = auto._harness(monkeypatch)
    cross = torch.randn(2, 4, 8, requires_grad = True)
    for _ in range(6):
        assert auto._step(layers, calls, cross_attention_states = cross) == ["plain", "plain"]
    state = bc._CUDAGRAPH_STATE["auto"]
    assert state["phase"] == "off" and "side input" in state["why"]
    assert not timed


def test_a_side_input_without_grad_does_not_count():
    assert not bc._grad_side_inputs(
        (torch.ones(2),), {"attention_mask": torch.ones(2, 2), "use_cache": False}
    )
    assert bc._grad_side_inputs((), {"cross_attention_states": torch.ones(2, requires_grad = True)})


H = 64


def _forward(
    self,
    hidden_states,
    cross_attention_states = None,
    **kwargs,
):
    attn = torch.softmax(
        self.q(hidden_states) @ self.k(cross_attention_states).transpose(-1, -2), -1
    )
    return hidden_states + attn @ self.v(cross_attention_states)


def _init(self):
    torch.nn.Module.__init__(self)
    self.q, self.k, self.v = (torch.nn.Linear(H, H) for _ in range(3))


class EagerCross(torch.nn.Module):
    __init__ = _init
    forward = _forward


class SideGradCrossAttentionDecoderLayer(torch.nn.Module):
    __init__ = _init
    forward = _forward


@pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a CUDA GPU")
@pytest.mark.parametrize("checkpoint", [False, True], ids = ["plain", "reentrant_checkpoint"])
def test_forced_graphs_train_like_eager_with_a_grad_side_input(monkeypatch, checkpoint):
    from torch.utils.checkpoint import checkpoint as ckpt

    monkeypatch.setattr(bc, "_CUDAGRAPH_MODE", "on")
    torch._dynamo.reset()
    bc._wrap(SideGradCrossAttentionDecoderLayer, {**bc._COMPILE_OPTIONS, "triton.cudagraphs": True})
    torch.manual_seed(0)
    graphed = (
        torch.nn.ModuleList(SideGradCrossAttentionDecoderLayer() for _ in range(3)).cuda().train()
    )
    eager = torch.nn.ModuleList(EagerCross() for _ in range(3)).cuda().train()
    eager.load_state_dict(graphed.state_dict())
    proj = {name: torch.nn.Linear(H, H).cuda() for name in ("graphed", "eager")}
    proj["eager"].load_state_dict(proj["graphed"].state_dict())
    side = {}
    for step in range(4):
        g = torch.Generator(device = "cuda").manual_seed(step)
        x = torch.randn(2, 8, H, device = "cuda", generator = g)
        img = torch.randn(2, 5, H, device = "cuda", generator = g)
        for name, layers in (("graphed", graphed), ("eager", eager)):
            cross = proj[name](img)  # the trained projector: its output takes a gradient
            y = x.clone().requires_grad_(checkpoint)
            for layer in layers:
                if checkpoint:
                    y = ckpt(
                        lambda t, c, layer = layer: layer(t, cross_attention_states = c),
                        y,
                        cross,
                        use_reentrant = True,
                    )
                else:
                    y = layer(y, cross_attention_states = cross)
            y.float().square().mean().backward()
            side[name] = proj[name].weight.grad.clone()
            for p in list(layers.parameters()) + list(proj[name].parameters()):
                p.data.add_(p.grad, alpha = -0.1)
                p.grad = None
        torch.testing.assert_close(side["graphed"], side["eager"], rtol = 2e-3, atol = 2e-4)
    for p, q in zip(graphed.parameters(), eager.parameters()):
        torch.testing.assert_close(p, q, rtol = 2e-3, atol = 2e-4)
    torch._dynamo.reset()
