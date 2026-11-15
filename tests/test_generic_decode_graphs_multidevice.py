# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel decode CUDA graphs for a model split over cards (device_map / Unsloth's layer split):
one graph holds every card's kernels. On the host through the stand-in backend of
test_generic_decode_graphs with the parameter devices faked as two cards; the real two-card capture
is the gpu-marked test."""

import contextlib
import os
import sys

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_generic_decode_graphs as T  # noqa: E402
import unsloth.models._generic_decode_graphs as GG
from real_accelerator import has_real_cuda

from transformers import AutoModelForCausalLM


class _Stream(T._FakeStream):
    def __init__(self, log, name):
        self.log, self.name = log, name

    def wait_stream(self, other):
        self.log.append((self.name, "waits", getattr(other, "name", "?")))


class _TwoCards(T._Host):
    """Host stand-in for a model split over cards 0 and 1 (this thread on card 0)."""

    others_recorded = []
    supported = True
    log = []
    current_device = staticmethod(lambda: 0)
    mempool = staticmethod(lambda: object())

    @classmethod
    def multidevice_supported(cls):
        return cls.supported

    @classmethod
    def current_stream(cls):
        return _Stream(cls.log, "card0")

    @classmethod
    def device_stream(cls, index):
        return _Stream(cls.log, f"card{index}")

    @classmethod
    def side_stream(cls, index):
        return _Stream(cls.log, f"side{index}")

    @classmethod
    @contextlib.contextmanager
    def on_streams(cls, streams):
        cls.log.append(("on_streams", tuple(sorted(streams))))
        yield

    @classmethod
    def record(cls, pool, fn, cache, *side):
        others, mempools = side or ({}, {})
        cls.others_recorded.append((tuple(sorted(others)), sorted(mempools)))
        return super().record(pool, fn, cache)


@pytest.fixture
def two_cards(monkeypatch):
    backend = type(
        "TwoCards",
        (_TwoCards,),
        {"graphs": [], "others_recorded": [], "log": [], "supported": True},
    )
    monkeypatch.setattr(GG, "_BACKEND", backend)
    # Parameters on "card 0" and "card 1" (host tensors: the stand-in backend runs on the CPU).
    monkeypatch.setattr(
        GG, "_param_devices", lambda model: {torch.device("cpu", 0), torch.device("cpu", 1)}
    )
    monkeypatch.delenv("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", raising = False)
    return backend


@pytest.fixture
def llama():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(T.LLAMA, dtype = torch.float32).eval()


def test_split_model_is_graphed_as_one_multi_card_graph(two_cards, llama):
    hooks = pytest.importorskip("accelerate.hooks")
    # As dispatch_model leaves a split model: an AlignDevicesHook on every decoder layer.
    for layer in llama.model.layers:
        hooks.add_hook_to_module(layer, hooks.AlignDevicesHook(execution_device = "cpu"))
    s = T._both(llama, *T._ids(llama, 2, pad = 1))
    assert s["captures"] == 1 and s["replays"] == 8 and s["refusals"] == {}, s
    # Card 1 joined with its own MemPool, and on its side stream for the warm-up.
    assert two_cards.others_recorded == [((1,), [1])]
    assert ("on_streams", (1,)) in two_cards.log
    # Every replay waits on card 1's stream first, and card 1 waits on the replay after.
    assert ("card0", "waits", "card1") in two_cards.log
    assert ("card1", "waits", "card0") in two_cards.log


@pytest.mark.parametrize(
    "setup, reason",
    [
        ("kill", "multi-device (UNSLOTH_DECODE_MULTIDEVICE_GRAPHS=0)"),
        ("old_torch", "multi-device capture needs a newer torch"),
        ("offload", "multi-device with offload or non-align hooks"),
        ("other_hook", "multi-device with offload or non-align hooks"),
    ],
)
def test_split_model_refusals_decode_eagerly(two_cards, llama, monkeypatch, setup, reason):
    accelerate_hooks = pytest.importorskip("accelerate.hooks")
    if setup == "kill":
        monkeypatch.setenv("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", "0")
    elif setup == "old_torch":
        two_cards.supported = False
    elif setup == "offload":
        hook = accelerate_hooks.AlignDevicesHook(execution_device = "cpu")
        hook.offload = True
        llama.model.layers[0].__dict__["_hf_hook"] = hook
    else:
        llama.model.layers[0].__dict__["_hf_hook"] = accelerate_hooks.ModelHook()
    s = T._both(llama, *T._ids(llama, 1))
    assert s["captures"] == 0 and s["replays"] == 0, s
    assert set(s["refusals"]) == {reason}, s
    assert two_cards.others_recorded == []


def test_single_card_model_records_without_other_cards(two_cards, llama, monkeypatch):
    monkeypatch.setattr(GG, "_param_devices", lambda model: {torch.device("cpu", 0)})
    s = T._both(llama, *T._ids(llama, 1))
    assert s["captures"] == 1 and s["refusals"] == {}, s
    # The single-card record call is unchanged (no other cards, no MemPools).
    assert two_cards.others_recorded == [((), [])]
    assert not any("card1" in entry for entry in two_cards.log), two_cards.log


@pytest.mark.gpu
@pytest.mark.skipif(
    not has_real_cuda() or torch.cuda.device_count() < 2, reason = "needs two CUDA GPUs"
)
def test_real_two_card_capture_matches_eager():
    """A tiny Llama split over cuda:0 / cuda:1 with accelerate's dispatch: one multi-card graph,
    tokens and logits equal to the eager step."""
    from accelerate import dispatch_model

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(T.LLAMA, dtype = torch.float16).eval()
    n = len(model.model.layers)
    device_map = {"model.embed_tokens": 0, "model.rotary_emb": 0, "model.norm": 1, "lm_head": 1}
    for i in range(n):
        device_map[f"model.layers.{i}"] = 0 if i < n // 2 else 1
    dispatch_model(model, device_map = device_map, main_device = 0, force_hooks = True)
    ids, mask = T._ids(model, 2, pad = 1)
    ids, mask = ids.cuda(0), mask.cuda(0)
    eager = T._generate(model, ids, mask, False, new = 24)
    graphed = T._generate(model, ids, mask, True, new = 24)
    assert torch.equal(eager, graphed)
    s = GG.stats(model)
    assert s["captures"] >= 1 and s["replays"] > 0, s


class _Seg:
    def __init__(self, log, n):
        self.log, self.n = log, n

    def replay(self):
        self.log.append(("replay", self.n))


class _HostCards:
    """Two host "cards" for the segment recorder: card 0 = cpu, card 1 = meta (copies cpu -> meta work)."""

    device_type = "cpu"

    def __init__(self):
        self.log = []
        self.n = 0

    @staticmethod
    def card_of(value):
        return {"cpu": 0, "meta": 1}.get(value.device.type)

    @staticmethod
    def index_of(device):
        if isinstance(device, int):
            return device
        return {"cpu": 0, "meta": 1}.get(torch.device(device).type)

    @staticmethod
    def card_device(index):
        return torch.device(("cpu", "meta")[index])

    current_device = staticmethod(lambda: 0)
    device = staticmethod(lambda index: contextlib.nullcontext())

    def new_graph(self):
        self.n += 1
        return _Seg(self.log, self.n)

    def begin_on(self, graph, pool, card):
        self.log.append(("begin", graph.n, card, pool))

    def end_on(self, graph):
        self.log.append(("end", graph.n))


def _send_to_device(value, device, *args, **kwargs):
    return value.to(device)


class _Block(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.lin = torch.nn.Linear(4, 4, device = device)

    def forward(self, x, extra):
        return self.lin(x) + extra


def test_segment_recorder_cuts_at_the_card_boundary_and_moves_shared_inputs_once(monkeypatch):
    hooks = pytest.importorskip("accelerate.hooks")
    cards = _HostCards()
    monkeypatch.setattr(GG, "_BACKEND", cards)
    torch.manual_seed(0)
    first = _Block("cpu")
    second, third = _Block("meta"), _Block("meta")
    for block in (second, third):
        hooks.add_hook_to_module(block, hooks.AlignDevicesHook(execution_device = "meta"))
    x = torch.randn(2, 4)
    extra = torch.randn(2, 4)  # a layer-invariant input on card 0 (rotary cos / sin, a mask)

    def step():
        h = first(x, extra = extra)
        h = second(h, extra = extra)
        return third(h, extra = extra)

    originals = (
        hooks.send_to_device,
        hooks.AlignDevicesHook.pre_forward,
        hooks.AlignDevicesHook.post_forward,
    )
    segments, out = GG._SegmentRecorder(cards, {0: "pool0", 1: "pool1"}, 0).run(step)
    assert out.device.type == "meta"
    kinds = [
        (p[0], p[2] if p[0] == "graph" else (p[2].device.type, p[1].device.type))
        for p in segments.parts
    ]
    # Card 0's segment, the hidden state and the shared input cross once, then one card-1 segment
    # for both card-1 blocks (the second block's inputs are already there: no new cut).
    assert kinds == [
        ("graph", 0),
        ("copy", ("cpu", "meta")),
        ("copy", ("cpu", "meta")),
        ("graph", 1),
    ], kinds
    assert cards.log == [
        ("begin", 1, 0, "pool0"),
        ("end", 1),
        ("begin", 2, 1, "pool1"),
        ("end", 2),
    ], cards.log
    assert (
        hooks.send_to_device,
        hooks.AlignDevicesHook.pre_forward,
        hooks.AlignDevicesHook.post_forward,
    ) == originals
    assert second(x.to("meta"), extra = extra).device.type == "meta"


def test_segments_replay_graphs_and_hand_offs_in_capture_order(monkeypatch):
    monkeypatch.setattr(GG, "_BACKEND", _HostCards())
    src, dst = torch.zeros(3), torch.zeros(3)
    log = []

    class Writes:
        def replay(self):
            log.append("card0")
            src.fill_(3)

    class Reads:
        def replay(self):
            log.append(("card1 sees", dst.tolist()))

    segments = GG._Segments()
    segments.parts = [("graph", Writes(), 0), ("copy", dst, src), ("graph", Reads(), 1)]
    segments.replay()
    assert log == ["card0", ("card1 sees", [3.0, 3.0, 3.0])]


def test_segment_recorder_leaves_movable_objects_to_accelerate(monkeypatch):
    # A non-tensor with .to (BatchEncoding, ...) goes through accelerate's own send_to_device, as eager.
    cards = _HostCards()
    monkeypatch.setattr(GG, "_BACKEND", cards)
    seen = []

    class Movable:
        def to(self, device, *args, **kwargs):
            seen.append(device)
            return self

    recorder = GG._SegmentRecorder(cards, {0: "pool0", 1: "pool1"}, 0)
    recorder.original = _send_to_device
    item = Movable()
    assert recorder.send(_send_to_device, {"a": item}, "meta")["a"] is item
    assert (
        seen == [torch.device("meta")]
        and recorder.segments.parts == []
        and recorder.pending is None
    )


def test_split_model_root_hook_on_the_first_blocks_card_is_plain(monkeypatch):
    # 16-bit models with tied embeddings: Unsloth's split puts the tied embedding / lm_head (the first
    # parameter, so model.device) on cuda:1 while the root hook moves inputs to cuda:0.
    from accelerate.hooks import AlignDevicesHook

    class Root:
        device = torch.device("cuda", 1)

    root = Root()
    root.__dict__["_hf_hook"] = AlignDevicesHook(
        execution_device = torch.device("cuda", 0), io_same_device = True
    )
    cards = {torch.device("cuda", 0), torch.device("cuda", 1)}
    monkeypatch.setattr(GG, "_param_devices", lambda model: cards)
    monkeypatch.setenv("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", "1")
    assert GG._single_device_hook(root)
    monkeypatch.setenv("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", "0")
    assert not GG._single_device_hook(root)
    monkeypatch.setenv("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", "1")
    # A card the model is not on stays refused; a model on one card keeps the exact-device rule.
    root.__dict__["_hf_hook"] = AlignDevicesHook(execution_device = torch.device("cuda", 2))
    assert not GG._single_device_hook(root)
    monkeypatch.setattr(GG, "_param_devices", lambda model: {torch.device("cuda", 1)})
    root.__dict__["_hf_hook"] = AlignDevicesHook(execution_device = torch.device("cuda", 0))
    assert not GG._single_device_hook(root)


def _graph_block_mask(device = "cpu"):
    # A BlockMask publishing the graph contract as zoo's gpt-oss flex mask does: mask_mod reads an offset
    # and padding rows through `rebuild`, and the builder's own per-device rebuild sits on the mask.
    from torch.nn.attention.flex_attention import BlockMask

    offset = torch.full((), 3, dtype = torch.int64, device = device)
    pad = torch.ones(1, 128, dtype = torch.bool, device = device)

    def rebuild(ts):
        off, rows = ts

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx + off <= q_idx + off) & rows[b, kv_idx]

        mask_mod.reads = ts
        return mask_mod

    bm = BlockMask.from_kv_blocks(
        torch.ones(1, 1, 1, dtype = torch.int32, device = device),
        torch.zeros(1, 1, 1, 1, dtype = torch.int32, device = device),
        BLOCK_SIZE = 128,
        mask_mod = rebuild((offset, pad)),
        seq_lengths = (1, 128),
    )
    bm._unsloth_graph_closure = ((offset, pad), rebuild)
    bm._unsloth_on_device = lambda target: pytest.fail(
        "the builder's rebuild reads the unrefilled tensors"
    )
    return bm


def test_graph_block_mask_moves_its_closure_tensors_to_the_layer_card(monkeypatch):
    # Split model + flex decode: the static BlockMask is built on card 0 and accelerate moves it to card 1's
    # layers. BlockMask.to leaves mask_mod's tensors on card 0 (the gpt-oss "Tensor device mismatch ...
    # cuda:1 and cuda:0" at kv_idx + kv_o), so both the warm-up and the recording move them as well.
    pytest.importorskip("torch.nn.attention.flex_attention")
    cards = _HostCards()
    monkeypatch.setattr(GG, "_BACKEND", cards)
    value = _graph_block_mask()
    leaves, cache = [], object()
    GG._spec(value, leaves, cache)
    static = [leaf.clone() for leaf in leaves]
    built = GG._build(value, iter(static), cache)
    closure = built._unsloth_graph_closure[0]
    assert built.mask_mod.reads == closure and all(any(t is s for s in static) for t in closure)

    # Eager warm-up: accelerate's .to (zoo's patched BlockMask.to calls _unsloth_on_device).
    on_device = built.__dict__.get("_unsloth_on_device")
    assert on_device is not None
    moved = on_device("meta")
    assert all(t.device.type == "meta" for t in moved.mask_mod.reads)
    assert moved.kv_num_blocks.device.type == "meta"
    # Same card (the root hook, the first card's layers): the mask itself, which keeps its per-device move,
    # so the next card's hook still moves mask_mod's tensors (Kaggle T4x2: the root hook's same-card move
    # dropped it and the cuda:1 layers read the cuda:0 offsets).
    here = on_device("cpu")
    assert here is built
    later = here.__dict__["_unsloth_on_device"]("meta")
    assert all(t.device.type == "meta" for t in later.mask_mod.reads)
    assert moved.__dict__["_unsloth_on_device"]("meta") is moved

    # Recording: the fields and closure tensors cross as hand-offs (refilled every replay), once per card.
    recorder = GG._SegmentRecorder(cards, {0: "pool0", 1: "pool1"}, 0)
    recorder.original = _send_to_device
    out = recorder.send(recorder.original, {"attention_mask": built}, "meta")["attention_mask"]
    again = recorder.send(recorder.original, {"attention_mask": built}, "meta")["attention_mask"]
    assert again is out
    assert all(t.device.type == "meta" for t in out.mask_mod.reads)
    assert out._unsloth_graph_closure[0] == out.mask_mod.reads
    copies = [p for p in recorder.segments.parts if p[0] == "copy"]
    sources = [p[2] for p in copies]
    assert all(any(t is s for s in sources) for t in closure), "closure tensors not handed off"
    assert any(s is built.kv_num_blocks for s in sources)
    assert recorder.pending == 1


def test_overlapping_recordings_from_two_threads_restore_accelerates_hooks():
    # accelerate's hooks are process-global: overlapping recordings used to restore each other's replacements and
    # leave a finished recorder's send_to_device installed.
    import threading
    import time

    import accelerate.hooks as hooks

    originals = (
        hooks.send_to_device,
        hooks.AlignDevicesHook.pre_forward,
        hooks.AlignDevicesHook.post_forward,
    )
    active, peak, errors = [0], [0], []

    def body():
        active[0] += 1
        peak[0] = max(peak[0], active[0])
        time.sleep(0.05)
        active[0] -= 1

    def record():
        try:
            GG._SegmentRecorder(_HostCards(), {0: "pool0"}, 0).run(body)
        except BaseException as error:
            errors.append(repr(error))

    threads = [threading.Thread(target = record) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert not errors and peak[0] == 1, (errors, peak)
    assert (
        hooks.send_to_device,
        hooks.AlignDevicesHook.pre_forward,
        hooks.AlignDevicesHook.post_forward,
    ) == originals
