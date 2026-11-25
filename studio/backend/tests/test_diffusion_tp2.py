# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-process TP-2 (diffusion_tp2): a tiny real Qwen-Image-2.1 transformer split over two CPU "cards" matches
the unsplit model (prefill, KV-cache extract and cached steps); native int8 ConvRot Linears shard exactly."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
q21 = pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
from torch import nn  # noqa: E402

from core.inference import diffusion_tp2 as tp2  # noqa: E402

CPU2 = [torch.device("cpu"), torch.device("cpu")]


def _tiny(layers = 3):
    torch.manual_seed(0)
    m = q21.QwenImage21Transformer2DModel(
        num_layers = layers,
        attention_head_dim = 32,
        num_attention_heads = 4,
        context_in_dim = 48,
        in_channels = 16,
        out_channels = 16,
        axes_dims_rope = (8, 12, 12),
    ).eval()
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.05)
    return m


def _inputs(
    h = 4,
    w = 6,
    text = 5,
):
    g = torch.Generator().manual_seed(1)
    tokens = h * w
    img_mask = torch.zeros(1, text + tokens // 4, dtype = torch.bool)
    img_mask[:, text:] = True
    return dict(
        hidden_states = torch.randn(1, tokens, 16, generator = g),
        encoder_hidden_states = torch.randn(1, text, 48, generator = g),
        timestep = torch.tensor([0.7]),
        img_shapes = [[(1, h, w)]],
        img_mask = img_mask,
        encoder_hidden_states_mask = torch.ones(1, text, dtype = torch.bool),
        return_dict = False,
    )


def _run(
    m,
    cache = None,
    mode = None,
    **kw,
):
    with torch.no_grad():
        return m(**_inputs(**kw), kv_cache = cache, kv_cache_mode = mode)[0]


def test_tp2_matches_one_card_prefill_and_kv_cache():
    ref = _tiny()
    tp = copy.deepcopy(ref)
    rec = tp2.install(tp, CPU2)
    assert rec["blocks"] == 3 and rec["comm"] == "cpu"
    assert len(tp.transformer_blocks) == 1
    # shards are outside the module tree: the transformer reports only its prologue / epilogue
    assert sum(p.numel() for p in tp.parameters()) < sum(p.numel() for p in ref.parameters()) / 2
    shard = tp.transformer_blocks[0].shards()[1][0]
    assert (
        shard.attn.heads == 2
        and shard.attn.to_q.out_features == 64
        and shard.attn.to_out[0].in_features == 64
    )
    assert shard.img_mlp.out.in_features == 3 * 128 // 2

    torch.testing.assert_close(_run(tp), _run(ref), rtol = 1e-5, atol = 1e-5)

    c_ref, c_tp = q21.QwenImage21KVCache(3), q21.QwenImage21KVCache(len(tp.transformer_blocks))
    torch.testing.assert_close(
        _run(tp, c_tp, "extract"), _run(ref, c_ref, "extract"), rtol = 1e-5, atol = 1e-5
    )
    torch.testing.assert_close(
        _run(tp, c_tp, "cached"), _run(ref, c_ref, "cached"), rtol = 1e-5, atol = 1e-5
    )
    # each card cached only its own heads
    per = tp.transformer_blocks[0]._kv[c_tp.get_layer(0)]
    assert per[0][0].k.shape[2] == 2 and per[1][0].k.shape[2] == 2
    assert tp2.status(tp)["calls"] == 3


def test_refusals():
    m = _tiny(layers = 1)
    assert "not verified" in tp2.refusal(nn.Linear(2, 2), CPU2)
    assert "exactly two" in tp2.refusal(m, CPU2[:1])
    m.transformer_blocks[0].attn.to_q = nn.Conv1d(2, 2, 1)
    assert "Conv1d" in tp2.refusal(m, CPU2)


def _native(lin, rot_group):
    from core.inference.diffusion_native_quant import native_linear_class
    return native_linear_class()(lin, "int8", act_int8 = True, rot_group = rot_group)


@pytest.mark.parametrize("rot_group", [0, 256])
def test_native_int8_shards(rot_group):
    torch.manual_seed(0)
    lin = nn.Linear(1024, 512, bias = False)
    full = _native(lin, rot_group)
    x = torch.randn(3, 1024)
    with torch.no_grad():
        y = full(x)
        cols = [tp2.shard_linear(full, 0, i, 2, "cpu") for i in range(2)]
        assert [c.out_features for c in cols] == [256, 256]
        assert torch.equal(torch.cat([c(x) for c in cols], -1), y)  # column split: bit-equal
        rows = [tp2.shard_linear(full, 1, i, 2, "cpu") for i in range(2)]
        assert [r.in_features for r in rows] == [512, 512]
        assert torch.equal(rows[0].weight_scale, full.weight_scale)
        torch.testing.assert_close(
            rows[0](x[:, :512]) + rows[1](x[:, 512:]), y, rtol = 1e-4, atol = 1e-4
        )
    # the full layer is untouched
    assert full.weight_q.shape == (512, 1024) and full.in_features == 1024


def test_row_split_refuses_cut_convrot_group():
    lin = _native(nn.Linear(768, 64, bias = False), 256)  # 384 per card is not whole 256 groups
    with pytest.raises(ValueError, match = "ConvRot group"):
        tp2.shard_linear(lin, 1, 0, 2, "cpu")


def test_requested_env(monkeypatch):
    monkeypatch.delenv(tp2.TP_ENV, raising = False)
    assert not tp2.requested()
    monkeypatch.setenv(tp2.TP_ENV, "1")
    assert tp2.requested()


def test_engage_for_plan(monkeypatch):
    import dataclasses
    from types import SimpleNamespace

    from core.inference import diffusion_multigpu as mg

    @dataclasses.dataclass(frozen = True)
    class Plan:
        stream_transformer: bool = True
        resident_transformer_mib: int = 100
        secondary_ordinal: int = 1
        secondary_components: tuple = ("vae",)
        layer_split: object = "split"
        fallback_plan: object = None
        reasons: tuple = ("planned",)
        offload_policy: str = "group"
        stream_text_encoders: bool = False

    plan, pipe = Plan(), SimpleNamespace(transformer = _tiny(layers = 1))
    monkeypatch.delenv(tp2.TP_ENV, raising = False)
    assert tp2.engage_for_plan(pipe, plan) is plan  # off by default
    monkeypatch.setenv(tp2.TP_ENV, "1")
    monkeypatch.setattr(mg, "current_selection", lambda: None)
    assert tp2.engage_for_plan(pipe, plan) is plan  # one card
    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    )
    seen = {}
    monkeypatch.setattr(tp2, "refusal", lambda tr, devs: None)
    monkeypatch.setattr(
        tp2,
        "install",
        lambda tr, devs, comm = None, logger = None: seen.update(devs = devs)
        or {"devices": [str(d) for d in devs]},
    )
    out = tp2.engage_for_plan(pipe, plan)
    assert [str(d) for d in seen["devs"]] == ["cuda:0", "cuda:1"]
    assert (
        out.stream_transformer is False
        and out.layer_split is None
        and out.secondary_components is None
    )
    assert out.resident_transformer_mib is None and "tensor parallel" in out.reasons[-1]
    assert out.stream_text_encoders is True  # the plan's offload now covers only the encoders
    monkeypatch.setattr(tp2, "refusal", lambda tr, devs: "nope")
    assert tp2.engage_for_plan(pipe, plan) is plan


def test_compiled_block_shard_runs_eager_on_its_own_weights():
    ref = _tiny(layers = 1)
    blk = ref.transformer_blocks[0]
    blk._compiled_call_impl = (
        blk._call_impl
    )  # what Module.compile() leaves behind (bound to the original)
    sb = tp2.shard_block(blk, 1, 2, "cpu")
    assert sb._compiled_call_impl is None
    assert sb.attn.to_q.weight.data_ptr() != blk.attn.to_q.weight.data_ptr()


def test_hook_is_a_no_op_when_off_or_on_one_card(monkeypatch):
    from types import SimpleNamespace

    from core.inference import diffusion_multigpu as mg

    calls = []
    monkeypatch.setattr(tp2, "install", lambda *a, **k: calls.append(a))
    plan, pipe = object(), SimpleNamespace(transformer = _tiny(layers = 1))
    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    )
    for off in (None, "0", "off", ""):
        if off is None:
            monkeypatch.delenv(tp2.TP_ENV, raising = False)
        else:
            monkeypatch.setenv(tp2.TP_ENV, off)
        assert tp2.engage_for_plan(pipe, plan) is plan
    monkeypatch.setenv(tp2.TP_ENV, "1")
    for one_card in (None, mg.MultiGpuSelection(primary = 0, secondaries = ())):
        monkeypatch.setattr(mg, "current_selection", lambda sel = one_card: sel)
        assert tp2.engage_for_plan(pipe, plan) is plan
    assert calls == [] and not hasattr(pipe, "_unsloth_tp2")
    assert not tp2.is_installed(pipe.transformer)


def test_install_marks_graph_unsafe_and_keeps_regional_compile(monkeypatch):
    compiled = []

    def fake_compile(fn, *a, **kw):
        compiled.append((fn.__name__, kw))
        return fn

    monkeypatch.setattr(torch, "compile", fake_compile)
    ref = _tiny(layers = 2)
    m = copy.deepcopy(ref)
    m._unsloth_regional_compile_kwargs = {"fullgraph": False, "dynamic": False}
    rec = tp2.install(m, CPU2)
    assert rec["stage_compile"] is True
    assert sorted(n for n, _ in compiled) == ["_attn_half", "_mlp_half"]
    assert all(kw == {"fullgraph": False, "dynamic": False} for _, kw in compiled)
    assert m.__dict__[tp2.GRAPH_UNSAFE_ATTR]
    assert m._repeated_blocks == ("TPStack",)
    # both cards' shards of a block share its processor object (configuration only, never copied)
    shards = m.transformer_blocks[0].shards()
    assert all(a.attn.processor is b.attn.processor for a, b in zip(*shards))
    torch.testing.assert_close(_run(m), _run(ref), rtol = 1e-5, atol = 1e-5)
    # a later compile_repeated_blocks reaches the stages; the env keeps them eager
    compiled.clear()
    monkeypatch.setenv(tp2.TP_COMPILE_ENV, "0")
    e = copy.deepcopy(ref)
    e.transformer_blocks[0]._compiled_call_impl = e.transformer_blocks[0]._call_impl
    assert tp2.install(e, CPU2)["stage_compile"] is False and compiled == []
    monkeypatch.delenv(tp2.TP_COMPILE_ENV)
    e.compile_repeated_blocks(fullgraph = True)
    assert [n for n, _ in compiled] == ["_attn_half", "_mlp_half"]


def test_int8_attention_census_sees_the_shards():
    from core.inference import diffusion_int8_attention as ia

    class Proc:
        _unsloth_int8 = False

    m = _tiny(layers = 2)
    for blk in m.transformer_blocks:
        blk.attn.processor = Proc()
    before = len(ia.int8_processors(m))
    tp2.install(m, CPU2)
    assert before == 2
    # both cards' shards share each block's processor: the census still finds the attention modules
    assert len(ia.int8_processors(m)) == 4
    assert tp2.shard_modules(_tiny(layers = 1)) == []
