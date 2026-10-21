# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Speed paths on a layer split denoiser, CPU only: per-card block graphs, the step cache across cards, and the int8
GEMM swap per card. The split is placed with the weights left on the CPU (``_move_weights`` stubbed); where a card
must be read off a weight, the test names it."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn

from core.inference import diffusion_block_graph as bg
from core.inference import diffusion_cache as dc
from core.inference import diffusion_cuda_graph as cg
from core.inference import diffusion_int8_gemm as g8
from core.inference import diffusion_layer_split as ls
from core.inference import diffusion_multigpu as mg

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_diffusion_layer_split import FakeDiT, Pipe, _two_card_budget  # noqa: E402

_needs_diffusers = pytest.mark.skipif(
    importlib.util.find_spec("diffusers") is None, reason = "needs diffusers"
)


@pytest.fixture(autouse = True)
def _env(monkeypatch):
    for name in (ls.LAYER_SPLIT_ENV, mg.MULTIGPU_ENV, mg.CFG_PARALLEL_ENV, bg.BLOCK_GRAPHS_ENV):
        monkeypatch.delenv(name, raising = False)
    yield


def _split(monkeypatch, dit):
    pipe = Pipe(dit)
    monkeypatch.setattr(ls, "_move_weights", lambda *a, **k: None)
    info = ls.place_layer_split(pipe, primary = 0, cards = [1], budgets_mib = _two_card_budget(dit))
    assert {r["device"] for r in info["ranges"]} == {"cuda:0", "cuda:1"}
    split = ls.layer_split_of(dit)
    card = {qual: int(dev.split(":")[1]) for qual, dev in _assignment(dit).items()}
    return pipe, split, card


def _assignment(dit) -> dict:
    out = {}
    for qual, _, block in ls.find_blocks(dit):
        out[qual] = str(block.__dict__[ls._SPLIT_ATTR].device)
    return out


def test_block_graphs_record_each_block_on_its_own_card(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, card = _split(monkeypatch, dit)
    by_block = {id(b): card[q] for q, _, b in ls.find_blocks(dit)}
    monkeypatch.setattr(bg, "_block_card", lambda block: by_block.get(id(block)))
    handle, reason = bg.install_block_graphs(dit, device = "cuda:0")
    assert handle is not None, reason
    assert handle.cards() == {0, 1}
    for graph in handle.graphs:
        assert graph.shared.device_index == by_block[id(graph.block)]
        assert graph.shared.strict_device  # records only while its own card is current
    assert handle.shared.device_index == 0 and set(handle.card_shared) == {1}
    assert handle.describe()["cards"] == [0, 1]
    handle.free()
    assert not handle.graphs
    ls.undo_layer_split(pipe)


def test_one_card_block_graphs_keep_one_shared_state(monkeypatch):
    dit = FakeDiT(dim = 64, double = 2, single = 2)
    monkeypatch.setattr(
        bg, "_block_card", lambda block: pytest.fail("read a block card on one card")
    )
    handle, reason = bg.install_block_graphs(dit, device = "cuda:0")
    assert handle is not None, reason
    assert not handle.card_shared and handle.cards() == {0}
    assert all(g.shared is handle.shared and not g.shared.strict_device for g in handle.graphs)
    assert "cards" not in handle.describe()
    handle.free()


def test_a_split_block_off_its_card_runs_its_compute(monkeypatch):
    dit = FakeDiT(dim = 64, double = 2, single = 2)
    shared = bg._Shared(1, None)
    shared.strict_device = True
    calls = []
    graph = bg.BlockGraph(
        dit.transformer_blocks[0], lambda *a, **k: calls.append(1) or "eager", shared
    )
    monkeypatch.setattr(bg._WeightView, "placement", lambda self, index: (1,))
    monkeypatch.setattr(
        torch.cuda, "current_device", lambda: 0
    )  # the block's card (1) is not current
    monkeypatch.setattr(bg, "_walk", lambda obj, live: live.append(_FakeCuda(1)) or ("k",))
    monkeypatch.setattr(bg, "_refusal", lambda key, kwargs: None)
    with torch.no_grad():
        assert graph(torch.zeros(2, 64)) == "eager" and calls
    assert graph.stats["refused_host_input"] == 1 and not graph.cache and not graph.seen


class _FakeCuda:
    def __init__(self, index):
        self.device = torch.device("cuda", index)


def test_arm_block_graphs_records_a_split_denoiser_per_block(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    armed = []
    monkeypatch.setattr(cg, "_block_basics", lambda *a, **k: None)
    monkeypatch.setattr(
        bg,
        "install_block_graphs",
        lambda t, **k: armed.append(t) or (types.SimpleNamespace(slots_mib = 0), "armed"),
    )
    monkeypatch.setattr("core.inference.diffusion_speed._denoiser_dits", lambda p: [p.transformer])
    applied = {"cuda_graph": False}
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:0")
    handles = cg.arm_block_graphs(pipe, applied, target = target, family = None, hooked = False)
    assert armed == [dit] and len(handles) == 1 and applied["cuda_graph"] is True
    assert pipe._unsloth_cuda_graph_mode == "blocks"
    assert "split denoiser" in pipe._unsloth_cuda_graph_reason
    ls.undo_layer_split(pipe)


def test_split_block_graphs_honour_the_kill_switch(monkeypatch):
    monkeypatch.setenv(bg.BLOCK_GRAPHS_ENV, "0")
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    monkeypatch.setattr(g8, "after_layer_split", lambda t, logger = None: {})
    monkeypatch.setattr(cg, "_block_basics", lambda *a, **k: None)
    monkeypatch.setattr(
        bg, "install_block_graphs", lambda t, **k: pytest.fail("armed with the kill switch set")
    )
    monkeypatch.setattr("core.inference.diffusion_speed._denoiser_dits", lambda p: [p.transformer])
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:0")
    assert (
        cg.arm_block_graphs(pipe, {"cuda_graph": True}, target = target, family = None, hooked = False)
        == ()
    )
    assert "UNSLOTH_DIFFUSION_BLOCK_GRAPHS=0" in pipe._unsloth_cuda_graph_reason
    ls.undo_layer_split(pipe)


def test_arm_block_graphs_leaves_a_resident_one_card_denoiser_alone(monkeypatch):
    dit = FakeDiT(dim = 64, double = 2, single = 2)
    pipe = Pipe(dit)
    monkeypatch.setattr(cg, "_block_basics", lambda *a, **k: None)
    monkeypatch.setattr(
        bg, "install_block_graphs", lambda t, **k: pytest.fail("armed per block on one card")
    )
    monkeypatch.setattr("core.inference.diffusion_speed._denoiser_dits", lambda p: [p.transformer])
    applied = {"cuda_graph": False}
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:0")
    assert cg.arm_block_graphs(pipe, applied, target = target, family = None, hooked = False) == ()


@_needs_diffusers
def test_split_block_classes_register_with_a_copy_of_their_base_metadata(monkeypatch):
    from diffusers.hooks._helpers import TransformerBlockMetadata, TransformerBlockRegistry

    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    base_meta = {}
    for base in (
        type(dit.transformer_blocks[0])._unsloth_layer_split_base,
        type(dit.single_transformer_blocks[0])._unsloth_layer_split_base,
    ):
        meta = TransformerBlockMetadata(return_hidden_states_index = 0)
        TransformerBlockRegistry.register(model_class = base, metadata = meta)
        base_meta[base] = meta
    try:
        assert dc._register_layer_split_blocks(dit)
        for block in (dit.transformer_blocks[0], dit.single_transformer_blocks[0]):
            sub = type(block)
            base = sub._unsloth_layer_split_base
            got = TransformerBlockRegistry.get(sub)
            assert (
                got is not base_meta[base]
            )  # never the base's own object: register writes _cls into it
            assert got.return_hidden_states_index == 0
            assert base_meta[base]._cls is base
    finally:
        for cls in list(TransformerBlockRegistry._registry):
            if getattr(cls, "__name__", "") in ("FakeBlock", "FakeSingle"):
                TransformerBlockRegistry._registry.pop(cls, None)
        ls.undo_layer_split(pipe)


def _tail_hook(head_output):
    from diffusers.hooks.first_block_cache import FBCBlockHook, FBCSharedBlockState
    from diffusers.hooks._helpers import TransformerBlockMetadata
    from diffusers.hooks.hooks import StateManager

    manager = StateManager(FBCSharedBlockState, (), {})
    try:
        from diffusers.hooks.hooks import CacheContext
        manager.set_context(CacheContext(name = "cond"))
    except ImportError:  # an older diffusers keys the state by a plain name
        manager.set_context("cond")
    hook = FBCBlockHook(manager, is_tail = True)
    hook._metadata = TransformerBlockMetadata(return_hidden_states_index = 0)
    hook._metadata._cached_parameter_indices = {"hidden_states": 0}
    state = manager.get_state()
    state.head_block_output = head_output
    return hook, state


@_needs_diffusers
def test_split_guard_leaves_a_one_card_tail_call_unchanged():
    assert dc.install_fbcache_split_guard()
    assert dc.install_fbcache_split_guard()  # idempotent
    from diffusers.hooks.first_block_cache import FBCBlockHook

    assert getattr(FBCBlockHook.new_forward, "_unsloth_split_guard", False)
    head = torch.randn(2, 8)
    hook, state = _tail_hook(head)
    x = torch.randn(2, 8)
    fn = types.SimpleNamespace(original_forward = lambda h: h * 2)
    hook.fn_ref = fn
    out = hook.new_forward(None, x)
    assert torch.equal(out, x * 2)
    assert state.head_block_output is head
    assert torch.equal(state.tail_block_residuals[0], x * 2 - head)


@_needs_diffusers
def test_split_guard_computes_the_tail_residual_on_the_tail_card_and_returns_it_home(monkeypatch):
    dc.install_fbcache_split_guard()
    moves = []

    head = torch.randn(2, 8)
    hook, state = _tail_hook(head)
    x = torch.randn(2, 8)
    seen = {}

    def original_forward(h):
        seen["head"] = state.head_block_output
        return h * 2

    hook.fn_ref = types.SimpleNamespace(original_forward = original_forward)
    home, here = torch.device("cuda", 0), torch.device("cuda", 1)
    monkeypatch.setattr(dc, "_first_device", lambda obj: home if obj is head else None)
    monkeypatch.setattr(
        hook._metadata,
        "_get_parameter_from_args_kwargs",
        lambda name, args, kwargs: types.SimpleNamespace(device = here),
    )
    copied = torch.randn(2, 8)
    monkeypatch.setattr(
        dc, "_tensors_to", lambda obj, device: moves.append(("head", device)) or copied
    )
    real_to = torch.Tensor.to

    def fake_to(self, *args, **kwargs):
        moves.append(("residual", args[0] if args else kwargs.get("device")))
        return self

    monkeypatch.setattr(torch.Tensor, "to", fake_to)
    out = hook.new_forward(None, x)
    monkeypatch.setattr(torch.Tensor, "to", real_to)
    assert seen["head"] is copied  # the residual subtracts the copy that sits on the tail's card
    assert state.head_block_output is head  # the shared state keeps the head card's tensor
    assert ("head", here) in moves and ("residual", home) in moves
    assert torch.equal(out, x * 2)


@_needs_diffusers
def test_apply_step_cache_engages_on_a_split_through_the_guard(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    guarded = []
    monkeypatch.setattr(dc, "install_fbcache_split_guard", lambda: guarded.append(1) or True)
    monkeypatch.setattr(dc, "_register_layer_split_blocks", lambda t: True)
    monkeypatch.setattr(dc, "_pipeline_opens_cache_context", lambda p: True)
    monkeypatch.setattr(dc, "_reuses_prefix_kv", lambda p, t: False)
    enabled = []
    dit.enable_cache = lambda config: enabled.append(config)
    assert dc.apply_step_cache(pipe, mode = "fbcache") == "fbcache"
    assert guarded and enabled
    ls.undo_layer_split(pipe)


def test_apply_step_cache_stays_off_on_a_split_without_the_guard(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    monkeypatch.setattr(dc, "install_fbcache_split_guard", lambda: False)
    dit.enable_cache = lambda config: pytest.fail("cache engaged without the split guard")
    assert dc.apply_step_cache(pipe, mode = "fbcache") is None
    ls.undo_layer_split(pipe)


def _int8_dit(monkeypatch, cards):
    """Every Linear eligible; its weight's card read from ``cards`` (module -> ordinal)."""
    dit = FakeDiT(dim = 64, double = 2, single = 2)
    recs = {}
    for m in dit.modules():
        if type(m) is nn.Linear:
            recs[id(m)] = ("v2", None, torch.ones(1), m.weight)
    monkeypatch.setattr(g8, "_eligible", lambda m: recs.get(id(m)))
    monkeypatch.setattr(g8, "_weight_card", lambda m: cards.get(id(m), 0))
    monkeypatch.setattr(g8, "_op", lambda: "op")
    monkeypatch.setattr(
        "core.inference.diffusion_convrot_quant.rotquant_device_config", lambda index: None
    )
    monkeypatch.setattr(g8, "_v1_act_quant_matches", lambda index: True)
    return dit, recs


def _card1_singles(dit) -> dict:
    return {id(m): 1 for block in dit.single_transformer_blocks for m in block.modules()}


def test_split_int8_swap_probes_and_swaps_each_card(monkeypatch):
    dit, recs = _int8_dit(monkeypatch, {})
    cards = _card1_singles(dit)
    monkeypatch.setattr(g8, "_weight_card", lambda m: cards.get(id(m), 0))
    monkeypatch.setattr(g8, "_split_cards", lambda t: (0, 1))
    monkeypatch.setattr("core.inference.diffusion_int8_fused.resident_cuda_device", lambda t: None)
    probed = []
    monkeypatch.setattr(
        g8, "device_config", lambda index: probed.append(index) or ((1,) if index == 0 else None)
    )
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (10, 0))
    try:
        count = g8._swap(dit)
        assert set(probed) == {0, 1}
        linears = [m for m in dit.modules() if id(m) in recs]
        on0 = [m for m in linears if cards.get(id(m), 0) == 0]
        assert on0 and count == len(on0)
        for m in linears:
            if cards.get(id(m), 0) == 0:
                assert m.__dict__[g8._REC][4] is True and g8.is_installed(m)
            else:
                assert not g8.is_installed(m)
    finally:
        g8.uninstall(dit)


def test_split_int8_swap_refreshes_a_pre_split_onload_install(monkeypatch):
    dit, recs = _int8_dit(monkeypatch, {})
    cards = _card1_singles(dit)
    monkeypatch.setattr(g8, "_weight_card", lambda m: cards.get(id(m), 0))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (10, 0))
    monkeypatch.setattr(g8, "device_config", lambda index: (1,) if index == 0 else None)
    monkeypatch.setattr(g8, "_split_cards", lambda t: ())
    try:
        assert g8._swap_on(dit, None, 0) == len(recs)
        moved = [m for m in dit.modules() if id(m) in recs and cards.get(id(m), 0) == 1]
        assert moved and all(m.__dict__[g8._REC][4] is True for m in moved)
        monkeypatch.setattr(g8, "_split_cards", lambda t: (0, 1))
        monkeypatch.setattr(g8, "device_config", lambda index: (2,) if index == 1 else (1,))
        assert g8._swap_on(dit, None, 1) == len(moved)
        assert all(m.__dict__[g8._REC][4] is True for m in moved)
        monkeypatch.setattr(g8, "device_config", lambda index: None)
        assert (
            g8._swap_on(dit, None, 1) == 0
        )  # nothing to swap there and no rotation: count 0, records untouched
    finally:
        g8.uninstall(dit)


def test_one_card_int8_swap_never_reads_a_weight_card(monkeypatch):
    dit, recs = _int8_dit(monkeypatch, {})
    monkeypatch.setattr(g8, "_weight_card", lambda m: pytest.fail("read a weight card on one card"))
    monkeypatch.setattr(g8, "_split_cards", lambda t: ())
    monkeypatch.setattr(g8, "device_config", lambda index: (1,))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index = None: (10, 0))
    try:
        assert g8._swap_on(dit, None, 0) == len(recs)
    finally:
        g8.uninstall(dit)


def _swiglu_module(stale_device = "meta"):
    from core.inference import diffusion_int8_fused as f8

    m = nn.Module()
    m.w1, m.w3, m.w2 = (
        nn.Linear(16, 32, bias = False),
        nn.Linear(16, 32, bias = False),
        nn.Linear(32, 16, bias = False),
    )
    stale = nn.Linear(16, 64, bias = False, device = stale_device)
    parts = (("w1", m.w1), ("w3", m.w3), ("w2", m.w2))
    m.__dict__[f8._SWIGLU_ATTR] = (stale, parts, "w2", 32, 0, 32, 0, 0)
    return m, stale


def test_a_stale_fused_swiglu_is_rebuilt_on_its_parts_card_and_shares_their_storage():
    from core.inference import diffusion_int8_fused as f8

    m, stale = _swiglu_module()
    gate = m.w1.weight.detach().clone()
    out = f8.relocate_after_layer_split(m)
    rec = m.__dict__[f8._SWIGLU_ATTR]
    assert out["rebuilt"] == 1 and rec[0] is not stale
    assert rec[0].weight.device == m.w1.weight.device
    assert rec[0].weight.data_ptr() == m.w1.weight.data_ptr()
    assert torch.equal(m.w1.weight, gate) and rec[1:] == m.__dict__[f8._SWIGLU_ATTR][1:]
    assert f8.relocate_after_layer_split(m) == {
        "rebuilt": 0,
        "dropped": 0,
        "restored": 0,
    }  # idempotent


def test_a_fused_qkv_that_cannot_be_rebuilt_is_dropped():
    from core.inference import diffusion_int8_fused as f8

    attn = nn.Module()
    attn.to_q, attn.to_k = nn.Linear(16, 16), nn.Linear(16, 16)
    attn.to_v = nn.Linear(16, 16).to(torch.float16)  # mixed dtypes: no fused Linear over the three
    attn.__dict__["_unsloth_fused_qkv"] = (
        nn.Linear(16, 48, device = "meta"),
        attn.to_q,
        attn.to_k,
        attn.to_v,
    )
    out = f8.relocate_after_layer_split(attn)
    assert out["dropped"] == 1 and "_unsloth_fused_qkv" not in attn.__dict__


def test_fused_qkv_readers_ignore_a_fused_linear_on_another_card():
    from core.inference import diffusion_qwenimage21_fused as q21
    from core.inference import diffusion_zimage_fused as zf
    for mod in (zf, q21):
        attn = nn.Module()
        attn.to_q, attn.to_k, attn.to_v = nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8)
        attn.__dict__[mod._QKV_ATTR] = (
            nn.Linear(8, 24, device = "meta"),
            attn.to_q,
            attn.to_k,
            attn.to_v,
        )
        assert mod._qkv_intact(attn) is None
        attn.__dict__[mod._QKV_ATTR] = (nn.Linear(8, 24), attn.to_q, attn.to_k, attn.to_v)
        assert mod._qkv_intact(attn) is not None


def test_after_layer_split_is_a_no_op_on_one_card(monkeypatch):
    called = []
    monkeypatch.setattr(g8, "_split_cards", lambda t: ())
    monkeypatch.setattr(
        "core.inference.diffusion_int8_fused.relocate_after_layer_split",
        lambda t, l = None: called.append(t),
    )
    assert g8.after_layer_split(nn.Linear(4, 4)) == {"cards": (), "gemm": 0}
    assert not called


def test_arm_block_graphs_re_homes_the_split_records_before_arming(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, _ = _split(monkeypatch, dit)
    order = []
    monkeypatch.setattr(
        g8, "after_layer_split", lambda t, logger = None: order.append(("rehome", t)) or {}
    )
    monkeypatch.setattr(cg, "_block_basics", lambda *a, **k: None)
    monkeypatch.setattr(
        bg,
        "install_block_graphs",
        lambda t, **k: order.append(("arm", t)) or (types.SimpleNamespace(slots_mib = 0), "armed"),
    )
    monkeypatch.setattr("core.inference.diffusion_speed._denoiser_dits", lambda p: [p.transformer])
    target = types.SimpleNamespace(device = "cuda", backend = "cuda", torch_device = "cuda:0")
    cg.arm_block_graphs(pipe, {"cuda_graph": False}, target = target, family = None, hooked = False)
    assert order == [("rehome", dit), ("arm", dit)]
    ls.undo_layer_split(pipe)


def test_segments_are_per_card_runs_within_one_block_list(monkeypatch):
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, card = _split(monkeypatch, dit)
    by_block = {id(b): card[q] for q, _, b in ls.find_blocks(dit)}
    monkeypatch.setattr(bg, "_block_card", lambda block: by_block.get(id(block)))
    handle, reason = bg.install_block_graphs(dit, device = "cuda:0")
    assert handle is not None, reason
    runs = [[ls.find_blocks(dit)[i][0] for i in range(len(ls.find_blocks(dit)))]]
    quals = {id(b): q for q, _, b in ls.find_blocks(dit)}
    got = [[quals[id(g.block)] for g in seg.graphs] for seg in handle.segments]
    for run in got:
        assert len(run) >= 2
        assert len({card[q] for q in run}) == 1
        assert len({q.split(".")[0] for q in run}) == 1
    assert handle.mode == "segments" and handle.describe()["segments"] == len(handle.segments)
    assert runs  # forward order list exists
    handle.free()
    assert not handle.segments and all(g.segment is None for g in handle.graphs)
    ls.undo_layer_split(pipe)


def test_segment_kill_switch_keeps_per_block_graphs(monkeypatch):
    monkeypatch.setenv(bg.SEGMENT_GRAPHS_ENV, "0")
    dit = FakeDiT(dim = 1024, double = 4, single = 3)
    pipe, _, card = _split(monkeypatch, dit)
    by_block = {id(b): card[q] for q, _, b in ls.find_blocks(dit)}
    monkeypatch.setattr(bg, "_block_card", lambda block: by_block.get(id(block)))
    handle, _ = bg.install_block_graphs(dit, device = "cuda:0")
    assert handle is not None and not handle.segments and handle.mode == "blocks"
    handle.free()
    ls.undo_layer_split(pipe)


def _replaying_segment(n = 3):
    shared = bg._Shared(1, None)
    blocks = [nn.Linear(4, 4) for _ in range(n)]
    graphs = [bg.BlockGraph(b, b.forward, shared) for b in blocks]
    seg = bg.SegmentGraph(graphs, shared)
    rec = bg._SegRecord()
    x, temb = torch.zeros(2, 4), torch.ones(2, 4)
    outs = [[torch.full((2, 4), float(i))] for i in range(n)]
    for i in range(n):
        live: list = []
        rec.keys.append(bg._walk(((outs[i - 1][0] if i else x, temb), {}), live))
        rec.prov.append((("e", 0), ("e", 1)) if i == 0 else (("o", i - 1, 0), ("e", 1)))
        rec.out_specs.append(("t", 0))
    entry = bg._SegEntry()
    entry.record, entry.outputs, entry.replays = rec, outs, 1
    seg.mode, seg.cur, seg.ext = "replay", entry, [x, temb]
    seg.returned = [outs[0]]
    seg.versions = [[bg._version(t) for t in outs[0]]]
    seg.pos = 1
    return seg, outs, temb


def test_a_replayed_segment_hands_on_each_blocks_recorded_outputs():
    seg, outs, temb = _replaying_segment()
    mid = seg.call(1, (outs[0][0], temb), {})
    assert mid is outs[1][0]  # the next block must receive exactly this tensor
    last = seg.call(2, (mid, temb), {})
    assert torch.equal(last, outs[2][0]) and last is not outs[2][0]  # what leaves the run is a copy
    assert seg.pos is None and seg.stats["segment_breaks"] == 0


def test_a_segment_breaks_when_a_block_gets_anything_else():
    seg, outs, temb = _replaying_segment()
    rewritten = outs[0][0] + 0  # an op between blocks: same values, another tensor
    assert seg.call(1, (rewritten, temb), {}) is bg._NOT_HANDLED
    assert seg.stats["segment_breaks"] == 1 and seg.pos is None
    assert seg.call(2, (rewritten, temb), {}) is bg._NOT_HANDLED


def test_a_segment_breaks_on_a_fresh_copy_of_a_shared_input():
    seg, outs, temb = _replaying_segment()
    assert seg.call(1, (outs[0][0], temb.clone()), {}) is bg._NOT_HANDLED
    assert seg.stats["segment_breaks"] == 1


def test_a_segment_breaks_when_an_op_between_blocks_rewrites_an_output_in_place():
    seg, outs, temb = _replaying_segment()
    outs[0][0].add_(100)
    assert seg.call(1, (outs[0][0], temb), {}) is bg._NOT_HANDLED
    assert seg.stats["segment_breaks"] == 1 and seg.pos is None


def test_a_segment_starts_only_on_a_lazily_routed_chain(monkeypatch):
    walked = []
    real_walk = bg._walk
    monkeypatch.setattr(bg, "_walk", lambda tree, live: walked.append(1) or real_walk(tree, live))
    shared = bg._Shared(1, None)
    root = nn.Linear(4, 4)
    shared.root = root
    blocks = [nn.Linear(4, 4) for _ in range(2)]
    seg = bg.SegmentGraph([bg.BlockGraph(b, b.forward, shared) for b in blocks], shared)
    state = types.SimpleNamespace(lazy_now = False)
    root.__dict__[ls._SPLIT_ATTR + "_root"] = state
    x = torch.zeros(2, 4)
    with torch.inference_mode():
        assert seg.call(0, (x,), {}) is bg._NOT_HANDLED
        assert walked == []  # declined before it looked at the inputs
        state.lazy_now = True
        assert seg.call(0, (x,), {}) is bg._NOT_HANDLED  # host tensors: declined by the next check
        assert walked  # it got past the gate


def test_a_segment_never_starts_while_a_step_cache_is_on(monkeypatch):
    walked = []
    real_walk = bg._walk
    monkeypatch.setattr(bg, "_walk", lambda tree, live: walked.append(1) or real_walk(tree, live))
    shared = bg._Shared(1, None)
    root = nn.Linear(4, 4)
    shared.root = root
    root.__dict__[ls._SPLIT_ATTR + "_root"] = types.SimpleNamespace(lazy_now = True)
    root._unsloth_step_cache = "fbcache@0.12"
    seg = bg.SegmentGraph(
        [bg.BlockGraph(nn.Linear(4, 4), lambda x: x, shared) for _ in range(2)], shared
    )
    with torch.inference_mode():
        assert seg.call(0, (torch.zeros(2, 4),), {}) is bg._NOT_HANDLED
    assert walked == []


def test_video_generate_bypasses_graphs_after_the_auto_cache_toggle():
    import inspect

    from core.inference import video as V

    src = inspect.getsource(V.VideoBackend)
    toggle = src.index("toggled = maybe_toggle_step_cache(")
    bypass = src.index("diffusion_cuda_graph.set_bypass(", toggle)
    assert "cache_breaks_graph(state.transformer_cache)" in src[bypass : bypass + 200]
    assert bypass < src.index("if static_skip:", toggle)


@_needs_diffusers
def test_split_guard_wraps_the_mixed_stream_tail_once():
    from diffusers.hooks.first_block_cache import FBCBlockHook

    assert dc.install_fbcache_split_guard()
    outer = FBCBlockHook.new_forward
    assert outer._unsloth_split_guard and outer._unsloth_mixed_stream_tail
    inner = outer.__closure__ and [
        c.cell_contents for c in outer.__closure__ if callable(c.cell_contents)
    ]
    assert any(
        getattr(fn, "_unsloth_mixed_stream_tail", False)
        and not getattr(fn, "_unsloth_split_guard", False)
        for fn in inner
    )
    assert dc.install_fbcache_mixed_stream_tail() and dc.install_fbcache_split_guard()
    assert FBCBlockHook.new_forward is outer  # neither re-wraps the other


def test_resident_cuda_cards_lists_every_card_of_a_split(monkeypatch):
    from core.inference import diffusion_int8_fused as f8

    monkeypatch.setattr(f8, "resident_cuda_device", lambda m: None)
    monkeypatch.setattr(g8, "_split_cards", lambda m: (0, 1))
    assert f8.resident_cuda_cards(nn.Linear(2, 2)) == (0, 1)
    monkeypatch.setattr(g8, "_split_cards", lambda m: ())
    assert f8.resident_cuda_cards(nn.Linear(2, 2)) == ()
    monkeypatch.setattr(f8, "resident_cuda_device", lambda m: torch.device("cuda", 3))
    assert f8.resident_cuda_cards(nn.Linear(2, 2)) == (3,)


def test_qwen_real_rope_installs_on_a_split_after_probing_each_card(monkeypatch):
    from core.inference import diffusion_int8_fused as f8
    from core.inference import diffusion_qwenimage_rope as qr

    QwenImageTransformer2DModel = type("QwenImageTransformer2DModel", (nn.Module,), {})
    monkeypatch.setattr(qr, "disabled", lambda: False)
    monkeypatch.setattr(f8, "resident_cuda_cards", lambda m: (0, 1))
    probed, patched = [], []
    monkeypatch.setattr(qr, "_probe", lambda index: probed.append(index) or True)
    monkeypatch.setattr(
        qr, "_patch_table", lambda index, logger = None: patched.append(index) or True
    )
    assert qr.install(QwenImageTransformer2DModel()) is True
    assert probed == [0, 1] and patched == [0]
    monkeypatch.setattr(qr, "_probe", lambda index: False)
    assert qr.install(QwenImageTransformer2DModel()) is False


def test_zimage_fused_probes_each_card_of_a_split(monkeypatch):
    from core.inference import diffusion_int8_fused as f8
    from core.inference import diffusion_qwenimage21_rope as q21r
    from core.inference import diffusion_zimage_fused as zf

    monkeypatch.setattr(zf, "zimage_fused_disabled", lambda: False)
    monkeypatch.setattr(zf, "_patch_class", lambda logger = None: True)
    monkeypatch.setitem(
        sys.modules, zf._MODULE, types.SimpleNamespace(ZSingleStreamAttnProcessor = object)
    )
    monkeypatch.setattr(f8, "resident_cuda_cards", lambda m: (5, 6))
    monkeypatch.setattr(q21r, "inductor_addcmul_is_fma", lambda: True)
    monkeypatch.setattr(q21r, "probe_fusion", lambda dev: ("fma", dev.index))
    monkeypatch.setattr(zf, "_FUSION", {})
    out = zf.install_modules(nn.Linear(2, 2))
    assert out["real_rope"] is True and set(zf._FUSION) == {5, 6}
    monkeypatch.setattr(zf, "_FUSION", {5: ("fma", 5), 6: None})
    assert zf.install_modules(nn.Linear(2, 2))["real_rope"] is False


def test_q21_fused_checks_the_arch_of_every_split_card(monkeypatch):
    from core.inference import diffusion_int8_fused as f8
    from core.inference import diffusion_qwenimage21_fused as q21f

    monkeypatch.setattr(q21f, "q21_convrot_fused_disabled", lambda: False)
    monkeypatch.setattr(f8, "resident_cuda_cards", lambda m: (0, 1))
    seen = []
    monkeypatch.setattr(q21f, "arch_ok", lambda dev: seen.append(dev.index) or dev.index == 0)
    assert q21f.install_modules(nn.Linear(2, 2)) == {"fused_qkv": 0, "out": 0}
    assert seen == [0, 1]


def test_hunyuan_image_and_hidream_route_lazily():
    assert "HunyuanImageTransformer2DModel" in ls.LAZY_CLASSES
    assert "HiDreamImageTransformer2DModel" in ls.LAZY_CLASSES


def test_real_rope_probes_every_card_of_a_split(monkeypatch):
    from core.inference import diffusion_qwenimage21_rope as q21r

    monkeypatch.setattr(q21r, "real_rope_disabled", lambda: False)
    monkeypatch.setattr(q21r, "_FUSION", {})
    assert q21r.probe_split_cards((0, 1)) == ()  # no install probed a card: the patch is not in use
    probed = []
    monkeypatch.setattr(
        q21r,
        "probe_fusion",
        lambda dev: probed.append(dev.index) or (None if dev.index == 2 else "fma"),
    )
    monkeypatch.setattr(q21r, "_FUSION", {0: "fma"})
    assert q21r.probe_split_cards((0, 1, 2)) == (0, 1)
    assert probed == [1, 2]
    monkeypatch.setattr(g8, "_split_cards", lambda m: (0, 1))
    seen = []
    monkeypatch.setattr(q21r, "probe_split_cards", lambda cards: seen.append(cards) or cards)
    assert g8.after_layer_split(nn.Linear(2, 2))["rope_cards"] == (0, 1) and seen == [(0, 1)]


def test_split_graphs_seal_new_prompt_lengths_per_image_scope():
    den = nn.Linear(4, 4)
    handle = bg.BlockGraphSet(den, bg._Shared(0, None))
    handle.track_scopes()
    assert handle.length_varies and not handle.scope_sealed()
    den(torch.randn(2, 4))  # the pre-hook reads this call's image scope
    first = handle.scope
    handle.note_capture()
    cg.seal_lengths((handle,))
    assert handle.scope_sealed()
    den(torch.randn(3, 4))  # another batch: a new image scope still records
    assert handle.scope != first and not handle.scope_sealed()
    cg.seal_lengths((handle,))  # nothing recorded in that scope: its window stays open
    assert not handle.scope_sealed()
    den(torch.randn(2, 4))
    assert handle.scope_sealed()
    handle.reset()
    assert not handle.scope_sealed()
    for restore in handle.restores:
        restore()
    assert not den._forward_pre_hooks


def test_a_split_records_block_graphs_for_a_prompt_length_family(monkeypatch):
    import inspect

    src = inspect.getsource(cg.arm_block_graphs)
    assert "None if split else prompt_length_reason(pipe, family)" in src
    assert "handle.track_scopes()" in inspect.getsource(bg.install_block_graphs)


def test_zimage_real_rope_probes_every_card_of_a_split(monkeypatch):
    from core.inference import diffusion_qwenimage21_rope as q21r
    from core.inference import diffusion_zimage_fused as zf

    monkeypatch.setattr(zf, "zimage_fused_disabled", lambda: False)
    probed = []
    monkeypatch.setattr(q21r, "inductor_addcmul_is_fma", lambda: True)
    monkeypatch.setattr(
        q21r,
        "probe_fusion",
        lambda dev: probed.append(dev.index) or (None if dev.index == 2 else "x"),
    )
    monkeypatch.setattr(zf, "_FUSION", {0: "x"})
    monkeypatch.setattr(q21r, "_FUSION", {})
    monkeypatch.setattr(g8, "_split_cards", lambda m: (0, 1, 2))
    out = g8.after_layer_split(nn.Linear(2, 2))
    assert zf._FUSION == {0: "x", 1: "x", 2: None} and probed == [1, 2]
    assert out["zimage_rope_cards"] == (0, 1)
    monkeypatch.setattr(zf, "_FUSION", {})
    assert zf.probe_split_cards((0, 1)) == ()  # no install probed a card: the patch is not in use
