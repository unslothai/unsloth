# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Single-process TP-2 (diffusion_tp2): a tiny real Qwen-Image-2.1 transformer split over two CPU "cards" matches
the unsplit model (prefill, KV-cache extract and cached steps); native int8 ConvRot Linears shard exactly."""

from __future__ import annotations

import copy
import os

import pytest

if os.environ.get("UNSLOTH_TP2_REQUIRE_REAL_MODEL") == "1":
    # the CI step that installs the Studio diffusers pin: a missing import fails collection instead of skipping
    import torch
    from diffusers.models.transformers import transformer_qwenimage21 as q21
else:
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
    monkeypatch.setattr(tp2, "_card_free_total_mib", lambda d: (15000, 15360))
    monkeypatch.setattr(
        tp2,
        "install",
        lambda tr, devs, comm = None, logger = None, retire = True: seen.update(devs = devs)
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
    monkeypatch.setenv(tp2.TP_COMPILE_ENV, "1")  # opt-in; the default is eager stages

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
    # default (unset) and explicit off keep the stages eager, even under a regional compile
    for off in (None, "0"):
        compiled.clear()
        if off is None:
            monkeypatch.delenv(tp2.TP_COMPILE_ENV)
        else:
            monkeypatch.setenv(tp2.TP_COMPILE_ENV, off)
        e = copy.deepcopy(ref)
        e.transformer_blocks[0]._compiled_call_impl = e.transformer_blocks[0]._call_impl
        assert tp2.install(e, CPU2)["stage_compile"] is False and compiled == []
        e.compile_repeated_blocks(fullgraph = True)
        assert compiled == [] and not tp2.status(e)["stage_compiled"]
    # opted in: a later compile_repeated_blocks reaches the stages
    monkeypatch.setenv(tp2.TP_COMPILE_ENV, "1")
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


def test_convrot_linear_and_linear_subclasses_are_refused():
    from core.inference.diffusion_convrot import _install_rotation

    m = _tiny(layers = 1)
    _install_rotation(
        m.transformer_blocks[0].attn.to_q, 64
    )  # what Studio's torchao INT8 route does first
    why = tp2.refusal(m, CPU2)
    assert why is not None and "ConvRotLinear" in why

    class MyLinear(nn.Linear):
        pass

    m = _tiny(layers = 1)
    old = m.transformer_blocks[0].img_mlp.out
    sub = MyLinear(old.in_features, old.out_features, bias = False)
    m.transformer_blocks[0].img_mlp.out = sub
    assert "MyLinear" in tp2.refusal(m, CPU2)
    m = _tiny(layers = 1)
    k = m.transformer_blocks[0].attn.to_k
    k.weight = nn.Parameter(k.weight.data.to(torch.int8), requires_grad = False)
    assert "weight" in tp2.refusal(m, CPU2)


@pytest.mark.parametrize("where", ["comm", "shard", "place"])
def test_failed_install_keeps_the_original_blocks(monkeypatch, where):
    ref = _tiny(layers = 3)
    m = copy.deepcopy(ref)
    if where == "comm":
        monkeypatch.setattr(
            tp2, "make_all_reduce", lambda *a, **k: (_ for _ in ()).throw(MemoryError("probe OOM"))
        )
    elif where == "shard":
        calls = []
        real = tp2.shard_block

        def flaky(*a, **k):
            calls.append(1)
            if len(calls) == 4:
                raise MemoryError("shard OOM")
            return real(*a, **k)

        monkeypatch.setattr(tp2, "shard_block", flaky)
    else:
        real_to = type(m).to

        def failing_to(self, *a, **k):
            if self is m and any(isinstance(x, torch.device) for x in a):
                raise MemoryError("placement OOM")
            return real_to(self, *a, **k)

        monkeypatch.setattr(type(m), "to", failing_to)
    with pytest.raises(MemoryError):
        tp2.install(m, CPU2)
    monkeypatch.undo()
    assert not tp2.is_installed(m)
    assert len(m.transformer_blocks) == 3
    assert all(p.device.type == "cpu" for p in m.parameters())  # nothing left on meta
    torch.testing.assert_close(_run(m), _run(ref))


def _plan_cls():
    import dataclasses

    @dataclasses.dataclass(frozen = True)
    class Plan:
        stream_transformer: bool = True
        resident_transformer_mib: int = 100
        secondary_ordinal: object = None
        secondary_components: object = None
        layer_split: object = None
        fallback_plan: object = None
        reasons: tuple = ("planned",)
        offload_policy: str = "group"
        stream_text_encoders: bool = False

    return Plan


def _two_cards(monkeypatch):
    from core.inference import diffusion_multigpu as mg

    monkeypatch.setenv(tp2.TP_ENV, "1")
    monkeypatch.setattr(
        mg, "current_selection", lambda: mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    )
    monkeypatch.setattr(tp2, "refusal", lambda tr, devs: None)
    monkeypatch.setattr(tp2, "_card_free_total_mib", lambda d: (15000, 15360))


def test_engage_falls_back_to_the_plan_when_setup_fails(monkeypatch):
    from types import SimpleNamespace

    _two_cards(monkeypatch)
    monkeypatch.setattr(
        tp2, "install", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("NCCL init failed"))
    )
    plan = _plan_cls()()
    pipe = SimpleNamespace(transformer = _tiny(layers = 1))
    assert tp2.engage_for_plan(pipe, plan) is plan
    assert not hasattr(pipe, "_unsloth_tp2")


def test_engage_declines_a_companion_split_and_load_time_loras(monkeypatch):
    from types import SimpleNamespace

    _two_cards(monkeypatch)
    calls = []
    monkeypatch.setattr(
        tp2, "install", lambda *a, **k: calls.append(1) or {"devices": ["cuda:0", "cuda:1"]}
    )
    pipe = SimpleNamespace(transformer = _tiny(layers = 1))
    split = _plan_cls()(
        offload_policy = "none", secondary_components = ("text_encoder",), secondary_ordinal = 1
    )
    assert tp2.engage_for_plan(pipe, split) is split
    plan = _plan_cls()()
    assert tp2.engage_for_plan(pipe, plan, loras = [("style", 0.8)]) is plan
    assert calls == []
    assert (
        tp2.engage_for_plan(pipe, plan, loras = [("style", 0.0)]) is not plan
    )  # a zero weight names no adapter
    assert calls == [1]


def test_cfg_replica_declines_a_tensor_parallel_denoiser():
    from types import SimpleNamespace

    from core.inference import diffusion_multigpu as mg

    m = _tiny(layers = 1)
    tp2.install(m, CPU2)
    pipe = SimpleNamespace(transformer = m)
    sel = mg.MultiGpuSelection(primary = 0, secondaries = (1,))
    why = mg.cfg_parallel_decline(pipe, offload_policy = "none", selection = sel)
    assert why is not None and "tensor parallel" in why
    with pytest.raises(RuntimeError, match = "tensor-parallel"):
        mg.build_replica(m, "cpu")


def test_final_step_release_frees_the_per_card_prefix_kv():
    from core.inference import diffusion_prefix_kv as pkv

    m = _tiny(layers = 2)
    tp2.install(m, CPU2)
    pkv.track_prefix_kv(m)
    cache = q21.QwenImage21KVCache(len(m.transformer_blocks))
    _run(m, cache, "extract")
    _run(m, cache, "cached")
    per = m.transformer_blocks[0]._kv[cache.get_layer(0)]
    held = sum(
        t.untyped_storage().nbytes() for card in per for layer in card for t in (layer.k, layer.v)
    )
    assert held > 0
    assert pkv.release_prefix_kv(m) == held
    assert all(layer.k is None and layer.v is None for card in per for layer in card)


def test_fp16_row_partials_do_not_overflow_into_nan():
    # unsplit: 3e4 * 4 - 3e4 * 4 = 0; per card: +1.2e5 -> +Inf and -1.2e5 -> -Inf, NaN after the sum
    lin = nn.Linear(4, 1, bias = False).half()
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[4.0, 4.0, -4.0, -4.0]]))
    x = torch.tensor([[1.5e4, 1.5e4, 1.5e4, 1.5e4]], dtype = torch.float16)
    rows = [tp2._row_parallel_class()(tp2.shard_linear(lin, 1, i, 2, "cpu")) for i in range(2)]
    parts = tp2.make_all_reduce("cpu", CPU2)([rows[0](x[:, :2]), rows[1](x[:, 2:])])
    out = parts[0].to(torch.float16)
    assert torch.isfinite(out).all() and float(out) == 0.0


def test_fp16_tp_forward_stays_fp16_and_matches(monkeypatch):
    ref = _tiny(layers = 2).half()
    m = copy.deepcopy(ref)
    tp2.install(m, CPU2)

    def run_half(model):
        kw = _inputs()
        kw["hidden_states"] = kw["hidden_states"].half()
        kw["encoder_hidden_states"] = kw["encoder_hidden_states"].half()
        with torch.no_grad():
            return model(**kw)[0]

    a, b = run_half(m), run_half(ref)
    assert a.dtype == torch.float16
    torch.testing.assert_close(a.float(), b.float(), rtol = 2e-2, atol = 2e-2)


def test_lora_is_blocked_under_tp_and_peft_cannot_see_the_shards():
    if os.environ.get("UNSLOTH_TP2_REQUIRE_REAL_MODEL") == "1":
        import peft
    else:
        peft = pytest.importorskip("peft")
    from types import SimpleNamespace

    cfg = peft.LoraConfig(r = 4, target_modules = ["to_q"])
    stock = _tiny(layers = 1)
    peft.inject_adapter_in_model(cfg, stock)  # stock blocks: found
    m = _tiny(layers = 1)
    tp2.install(m, CPU2)
    with pytest.raises(ValueError):
        peft.inject_adapter_in_model(cfg, m)  # the reason render-time adapters are refused under TP
    reason = tp2.lora_block_reason(SimpleNamespace(transformer = m))
    assert reason and tp2.TP_ENV in reason
    assert tp2.lora_block_reason(SimpleNamespace(transformer = _tiny(layers = 1))) is None


def test_comm_choice_prefers_nccl_within_the_margin():
    assert tp2.choose_comm({"nccl": 4.4, "swap": 4.3, "peer": 7.0}) == "nccl"  # 2 % faster: noise
    assert tp2.choose_comm({"nccl": 5.7, "swap": 4.4, "peer": 7.5}) == "swap"  # 23 % faster
    assert tp2.choose_comm({"nccl": "failed: x", "swap": 4.4, "peer": 4.3}) == "peer"
    assert tp2.choose_comm({"nccl": "failed: x"}) is None


def test_hostswap_buffers_are_bounded():
    from collections import OrderedDict

    host, tmp = OrderedDict(), OrderedDict()
    for n in (32, 64, 128):
        tp2._evict(host, tmp, tp2.HOSTSWAP_SIZES - 1)
        host[n], tmp[n] = [n], [n]
    assert list(host) == [64, 128] and list(tmp) == [64, 128]


def test_render_time_lora_gate_names_tp():
    from types import SimpleNamespace

    from core.inference import diffusion

    m = _tiny(layers = 1)
    tp2.install(m, CPU2)
    reason = diffusion._lora_load_block_reason(SimpleNamespace(transformer = m))
    assert reason and "tensor parallel" in reason


def test_row_parallel_wrapper_is_one_class():
    # one class for every shard: a compiled stage guards on the module type and would retrace per block otherwise
    m = _tiny(layers = 3)
    tp2.install(m, CPU2)
    kinds = {type(blk.attn.to_out[0]) for card in m.transformer_blocks[0].shards() for blk in card}
    kinds |= {type(blk.img_mlp.out) for card in m.transformer_blocks[0].shards() for blk in card}
    assert len(kinds) == 1


def test_int8_dispatch_route_census_and_remove_see_the_shards(monkeypatch):
    from types import SimpleNamespace

    from core.inference import diffusion_int8_attention_dispatch as disp

    m = _tiny(layers = 2)
    for blk in m.transformer_blocks:
        blk.attn.processor._attention_backend = disp.Int8Route("native", 0, torch.float16, 0, 32)
    pipe = SimpleNamespace(transformer = m)
    monkeypatch.setattr(disp, "_dits", lambda p: [p.transformer])
    assert disp.routed_count(pipe) == 2
    tp2.install(m, CPU2)
    # both cards' shards share each block's processor: the census still finds the routes, and remove reaches them
    assert disp.routed_count(pipe) == 4
    assert disp.status(pipe)["modules"] == 4
    disp.remove(pipe)
    assert disp.routed_count(pipe) == 0


def test_compiled_style_view_caches_are_compacted_per_card(monkeypatch):
    # Inductor lowers the extract step's prefix clone to a view of the full K/V; Studio compacts the unsplit cache,
    # and the per-card TP caches must get the same copy or they pin the full prefill K/V through decode
    def view_store(self, k, v):
        self.k = torch.cat([k, k], 1)[:, : k.shape[1]]
        self.v = torch.cat([v, v], 1)[:, : v.shape[1]]

    monkeypatch.setattr(q21.QwenImage21KVLayerCache, "store", view_store)
    monkeypatch.delenv("UNSLOTH_DIFFUSION_Q21_COMPACT_KV", raising = False)
    ref = _tiny(layers = 2)
    tp = copy.deepcopy(ref)
    tp2.install(tp, CPU2)
    c_ref, c_tp = q21.QwenImage21KVCache(2), q21.QwenImage21KVCache(1)
    _run(ref, c_ref, "extract")
    _run(tp, c_tp, "extract")
    per = tp.transformer_blocks[0]._kv[c_tp.get_layer(0)]
    for layer in (layer for card in per for layer in card):
        for t in (layer.k, layer.v):
            assert t.untyped_storage().nbytes() == t.numel() * t.element_size()
    torch.testing.assert_close(
        _run(tp, c_tp, "cached"), _run(ref, c_ref, "cached"), rtol = 1e-5, atol = 1e-5
    )


def test_speed_profile_backend_change_is_refused_under_tp():
    from types import SimpleNamespace

    from core.inference import diffusion_attention as da

    m = _tiny(layers = 2)
    tp2.install(m, CPU2)
    procs = [card[i].attn.processor for card in m.transformer_blocks[0].shards() for i in range(2)]
    before = [getattr(p, "_attention_backend", None) for p in procs]
    pipe = SimpleNamespace(transformer = m)
    kept = getattr(m, da.ATTENTION_BACKEND_ATTR, None)
    engaged = da.apply_attention_backend(pipe, "_native_math")
    # the shards cannot be reached by diffusers' setter, so nothing changes and nothing new is reported
    assert engaged == kept and getattr(m, da.ATTENTION_BACKEND_ATTR, None) == kept
    assert [getattr(p, "_attention_backend", None) for p in procs] == before


def test_small_host_dense_denoiser_declines_tp():
    # Studio's small-host dense load: bf16 storage, fp16 compute casts on each Linear (diffusers layerwise casting)
    from core.inference.diffusion_small_host import prepare_streamed_encoder_

    m = _tiny(layers = 2).to(torch.bfloat16)
    prepare_streamed_encoder_(m, torch.float16)
    blocks = list(m.transformer_blocks)
    why = tp2.refusal(m, CPU2)
    assert why is not None and "hooks" in why
    with pytest.raises(RuntimeError, match = "declined"):
        tp2.install(m, CPU2)
    assert list(m.transformer_blocks) == blocks
