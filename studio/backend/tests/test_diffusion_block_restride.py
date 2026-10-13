# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""FLUX.1's first single block gets the slice layout of the other 37, so the regional compile traces ONE graph."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch._dynamo  # noqa: E402

from core.inference import diffusion_block_restride as restride  # noqa: E402


class FluxSingleTransformerBlock(torch.nn.Module):
    """Same call contract as diffusers': concatenates text + image itself, returns two slices of one tensor."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.proj = torch.nn.Linear(dim, dim)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb = None,
        image_rotary_emb = None,
        joint_attention_kwargs = None,
    ):
        text = encoder_hidden_states.shape[1]
        x = torch.cat([encoder_hidden_states, hidden_states], dim = 1)
        x = x + self.proj(x)
        return x[:, :text], x[:, text:]


class FluxTransformer2DModel(torch.nn.Module):
    def __init__(
        self,
        n: int = 4,
        dim: int = 8,
    ):
        super().__init__()
        self.single_transformer_blocks = torch.nn.ModuleList(
            FluxSingleTransformerBlock(dim) for _ in range(n)
        )

    def forward(self, hidden_states, encoder_hidden_states):
        for block in self.single_transformer_blocks:
            encoder_hidden_states, hidden_states = block(
                hidden_states = hidden_states,
                encoder_hidden_states = encoder_hidden_states,
                temb = None,
                image_rotary_emb = None,
                joint_attention_kwargs = None,
            )
        return hidden_states


def _inputs(
    batch = 1,
    text = 3,
    image = 5,
    dim = 8,
):
    g = torch.Generator().manual_seed(0)
    return torch.randn(batch, image, dim, generator = g), torch.randn(batch, text, dim, generator = g)


def _frames_compiled(model, hidden, encoder) -> int:
    """Dynamo frames compiled for one forward of a regional-compiled model (eager backend: dynamo only)."""
    torch._dynamo.reset()
    graphs = []

    def backend(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    for block in model.single_transformer_blocks:
        block.compile(backend = backend, fullgraph = True, dynamic = None)
    restride.install(model)
    with torch.no_grad():
        model(hidden, encoder)
    torch._dynamo.reset()
    return len(graphs)


def test_restrided_pair_is_the_later_blocks_layout_with_the_same_values():
    hidden, encoder = _inputs(batch = 2)
    pair = restride._restrided(hidden, encoder)
    assert pair is not None
    h2, e2 = pair
    assert torch.equal(h2, hidden) and torch.equal(e2, encoder)
    # What diffusers' single block returns: two slices of one [B, text + image, D] tensor.
    later_e, later_h = FluxSingleTransformerBlock()(hidden, encoder)
    assert h2.stride() == later_h.stride() and e2.stride() == later_e.stride()
    assert h2.untyped_storage().data_ptr() == e2.untyped_storage().data_ptr()


def test_already_slice_layout_and_mismatched_pairs_pass_through():
    hidden, encoder = _inputs()
    e, h = FluxSingleTransformerBlock()(hidden, encoder)
    assert restride._restrided(h, e) is None
    assert restride._restrided(hidden, encoder[..., :4]) is None
    assert restride._restrided(hidden, encoder.double()) is None
    assert restride._restrided(hidden, None) is None


def test_first_single_block_compiles_one_graph_instead_of_two(monkeypatch):
    model = FluxTransformer2DModel()
    hidden, encoder = _inputs()
    monkeypatch.setenv("UNSLOTH_DIFFUSION_BLOCK_RESTRIDE", "0")
    assert _frames_compiled(model, hidden, encoder) == 2
    model = FluxTransformer2DModel()
    monkeypatch.delenv("UNSLOTH_DIFFUSION_BLOCK_RESTRIDE")
    assert _frames_compiled(model, hidden, encoder) == 1


def test_outputs_identical_with_and_without_restride(monkeypatch):
    torch.manual_seed(0)
    model = FluxTransformer2DModel()
    hidden, encoder = _inputs(batch = 2)
    with torch.no_grad():
        ref = model(hidden, encoder)
    assert restride.install(model) is False  # nothing compiled: nothing to wrap
    for block in model.single_transformer_blocks:
        block.compile(backend = "eager", fullgraph = True)
    assert restride.install(model) is True
    assert restride.install(model) is True  # idempotent
    with torch.no_grad():
        out = model(hidden, encoder)
    assert torch.equal(ref, out)
    torch._dynamo.reset()


def test_other_transformers_and_kill_switch_are_left_alone(monkeypatch):
    class Other(FluxTransformer2DModel):
        pass

    model = Other()
    for block in model.single_transformer_blocks:
        block.compile(backend = "eager")
    assert restride.install(model) is False
    model = FluxTransformer2DModel()
    for block in model.single_transformer_blocks:
        block.compile(backend = "eager")
    monkeypatch.setenv("UNSLOTH_DIFFUSION_BLOCK_RESTRIDE", "0")
    before = model.single_transformer_blocks[0]._compiled_call_impl
    assert restride.install(model) is False
    assert model.single_transformer_blocks[0]._compiled_call_impl is before


def test_positional_calls_pass_through_unchanged():
    model = FluxTransformer2DModel()
    seen = []
    block = model.single_transformer_blocks[0]
    block._compiled_call_impl = lambda *a, **k: seen.append((a, k)) or (a[1], a[0])
    assert restride.install(model) is True
    hidden, encoder = _inputs()
    block(hidden, encoder)
    assert seen[0][0][0] is hidden and seen[0][0][1] is encoder
