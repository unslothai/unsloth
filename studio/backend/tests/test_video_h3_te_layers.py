# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""MiniMax-H3: the streamed conditioner uploads one decoder layer per transfer. CPU only (the CPU device runs the
same plan, slots and swaps with synchronous copies)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

import core.inference.video_minimax_h3_te as te
import core.inference.video_minimax_h3_te_layers as tl

transformers = pytest.importorskip("transformers")


class _Group:
    """The parts of a diffusers stream ModuleGroup the streamer reads: host copies keyed by the module tensor."""

    def __init__(self, params, buffers):
        self.modules = []
        self.parameters = list(params)
        self.buffers = list(buffers)
        self.cpu_param_dict = {t: t.data.cpu() for t in self.parameters + self.buffers}
        self.onloads = 0

    def onload_(self):
        self.onloads += 1

    def offload_(self):
        pass


def _conditioner(layers = 3, seed = 0):
    """A tiny Qwen3-VL language model in the hosted H3 layout: ConvRot INT8 projections, bf16 norms."""
    torch.manual_seed(seed)
    config = transformers.Qwen3VLTextConfig(
        hidden_size = 256,
        intermediate_size = 512,
        num_hidden_layers = layers,
        num_attention_heads = 4,
        num_key_value_heads = 2,
        head_dim = 64,
        vocab_size = 64,
        rope_scaling = {"rope_type": "default", "mrope_section": [8, 12, 12]},
    )
    model = transformers.Qwen3VLTextModel(config).to(torch.bfloat16).eval()
    cls = te._int8_convrot_linear_class()
    for layer in model.layers:
        for parent in (layer.self_attn, layer.mlp):
            for name, child in list(parent.named_children()):
                if isinstance(child, torch.nn.Linear):
                    q = cls(child.in_features, child.out_features, bias = False, group_size = 256)
                    q.weight = torch.randint(
                        -127, 128, (child.out_features, child.in_features), dtype = torch.int8
                    )
                    q.weight_scale = torch.rand(child.out_features, 1, dtype = torch.float32) * 1e-3
                    setattr(parent, name, q)
    model.requires_grad_(False)
    return model


def _groups(model):
    """Leaf-level grouping as diffusers builds it: one group per ConvRot projection (buffers), one per norm."""
    groups = []
    for layer in model.layers:
        for module in layer.modules():
            own_p = list(module.parameters(recurse = False))
            own_b = list(module.buffers(recurse = False))
            if own_p or own_b:
                groups.append(_Group(own_p, own_b))
    return groups


def _streamer(model, groups, **kw):
    layers = list(model.layers)
    plans = tl._layer_plans(layers, groups)
    assert plans is not None
    streamer = tl.H3TELayerStream(layers, plans, "cpu", **kw)
    tl.attach_h3_te_layer_stream(model, layers, plans, streamer)
    return streamer, plans


def _run(model, ids):
    with torch.no_grad():
        return model(input_ids = ids, output_hidden_states = True).hidden_states


IDS = torch.tensor([[1, 5, 9, 33, 2, 7, 60, 11]])


def test_plan_packs_every_layer_tensor_aligned():
    model = _conditioner()
    plans = tl._layer_plans(list(model.layers), _groups(model))
    for (entries, groups, size), layer in zip(plans, model.layers):
        assert {id(e[0]) for e in entries} == {
            id(t) for t in list(layer.parameters()) + list(layer.buffers())
        }
        assert all(offset % tl._ALIGN == 0 for _t, _g, offset, _n in entries)
        assert size == tl._align(entries[-1][2] + entries[-1][3])
        assert len(groups) == 11  # 7 projections + input / post / q / k norms


def test_plan_refuses_a_tensor_without_a_host_copy_or_a_shared_group():
    model = _conditioner()
    groups = _groups(model)
    assert tl._layer_plans(list(model.layers), groups[1:]) is None
    spanning = _Group(
        list(model.layers[0].input_layernorm.parameters())
        + list(model.layers[1].input_layernorm.parameters()),
        [],
    )
    assert tl._layer_plans(list(model.layers), [spanning] + groups) is None


def test_hidden_states_bit_identical_and_tensors_back_on_host():
    model = _conditioner()
    reference = _run(model, IDS)
    groups = _groups(model)
    streamer, plans = _streamer(model, groups)
    hosts = {id(t): t.data_ptr() for e, _g, _s in plans for t, *_ in e}
    seen = []

    def probe(
        _m,
        _args,
        _index = 0,
    ):
        slots = streamer.slots
        layer_tensors = plans[len(seen) % len(plans)][0]
        lo = min(s.data_ptr() for s in slots)
        hi = max(s.data_ptr() + s.numel() for s in slots)
        seen.append(all(lo <= t.data_ptr() < hi for t, *_ in layer_tensors))

    for layer in model.layers:
        layer.mlp.register_forward_pre_hook(probe)
    for _ in range(2):
        got = _run(model, IDS)
        assert len(got) == len(reference)
        for a, b in zip(got, reference):
            assert torch.equal(a, b)
    assert seen == [True] * (2 * len(model.layers))  # every layer computed from its slot
    assert all(t.data_ptr() == hosts[id(t)] for e, _g, _s in plans for t, *_ in e)
    assert streamer.sessions == 2 and streamer.layers_streamed == 2 * len(model.layers)
    assert not streamer.active and streamer.slots == []
    assert all(g.onloads == 0 for g in groups)  # the diffusers groups move nothing
    assert streamer.bytes_direct == 2 * sum(n for e, _g, _s in plans for *_x, n in e)


def test_reads_the_current_host_copy():
    """keep_h3_te_resident / yield re-point cpu_param_dict after install: the streamer must follow."""
    model = _conditioner()
    groups = _groups(model)
    _streamer(model, groups)
    _run(model, IDS)
    target = model.layers[1].mlp.down_proj
    group = next(g for g in groups if any(t is target.weight_scale for t in g.buffers))
    fresh = group.cpu_param_dict[target.weight_scale].clone() * 2
    group.cpu_param_dict[target.weight_scale] = fresh
    target.weight_scale.data = fresh
    got = _run(model, IDS)

    other = _conditioner()
    other.layers[1].mlp.down_proj.weight_scale.data = (
        other.layers[1].mlp.down_proj.weight_scale.data * 2
    )
    want = _run(other, IDS)
    for a, b in zip(got, want):
        assert torch.equal(a, b)
    assert target.weight_scale.data_ptr() == fresh.data_ptr()


def test_an_exception_in_a_layer_restores_every_layer():
    model = _conditioner()
    groups = _groups(model)
    streamer, plans = _streamer(model, groups)
    hosts = {id(t): t.data_ptr() for e, _g, _s in plans for t, *_ in e}
    boom = {"on": True}

    def fail(_m, _args):
        if boom["on"]:
            raise RuntimeError("boom")

    handle = model.layers[1].mlp.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match = "boom"):
        _run(model, IDS)
    assert not streamer.active and streamer.swapped == {}
    assert all(t.data_ptr() == hosts[id(t)] for e, _g, _s in plans for t, *_ in e)
    handle.remove()
    reference = _run(_conditioner(), IDS)
    for a, b in zip(_run(model, IDS), reference):
        assert torch.equal(a, b)


def test_kill_switch_and_non_cuda_devices_install_nothing(monkeypatch):
    model = _conditioner()
    holder = type("TE", (), {})()
    holder.model = type("M", (torch.nn.Module,), {})()
    holder.model.language_model = model
    monkeypatch.setenv(tl.H3_TE_LAYER_STREAM_ENV, "0")
    assert tl.install_h3_te_layer_stream(holder, "cuda", pinned = True) is None
    monkeypatch.delenv(tl.H3_TE_LAYER_STREAM_ENV)
    assert tl.install_h3_te_layer_stream(holder, "cpu", pinned = True) is None
    assert not any(layer._forward_pre_hooks for layer in model.layers)


def test_attach_failure_leaves_nothing_behind():
    model = _conditioner()
    groups = _groups(model)
    layers = list(model.layers)
    plans = tl._layer_plans(layers, groups)
    streamer = tl.H3TELayerStream(layers, plans, "cpu")

    class Frozen(torch.nn.Module):
        def __setattr__(self, name, value):
            if name == tl._ATTR:
                raise AttributeError(name)
            super().__setattr__(name, value)

    target = Frozen()
    with pytest.raises(AttributeError):
        tl.attach_h3_te_layer_stream(target, layers, plans, streamer)
    assert not any(layer._forward_pre_hooks or layer._forward_hooks for layer in layers)
    assert all("onload_" not in g.__dict__ for g in groups)


def test_the_stream_path_installs_after_the_rotation_unhook():
    src = Path(te.__file__).read_text(encoding = "utf-8")
    fn = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == "stream_h3_text_encoder"
    )
    body = ast.get_source_segment(src, fn)
    assert body.index("_unhook_from_manager(") < body.index("install_h3_te_layer_stream(")
    assert "if stream:" in body[body.index("_evict_rotation_hook(manager") :]


def test_no_room_for_the_slots_uploads_each_layer_on_its_own(monkeypatch):
    model = _conditioner()
    reference = _run(model, IDS)
    groups = _groups(model)
    streamer, plans = _streamer(model, groups)
    hosts = {id(t): t.data_ptr() for e, _g, _s in plans for t, *_ in e}

    def refuse():
        raise torch.OutOfMemoryError("no room")

    monkeypatch.setattr(streamer, "_start", refuse)
    for a, b in zip(_run(model, IDS), reference):
        assert torch.equal(a, b)
    assert streamer.fallbacks == len(model.layers)
    assert not streamer.active and streamer.swapped == {}
    assert all(t.data_ptr() == hosts[id(t)] for e, _g, _s in plans for t, *_ in e)
