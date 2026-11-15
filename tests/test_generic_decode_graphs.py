# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastModel decode CUDA graphs on the host, through a stand-in backend whose "graph" re-runs the
captured step into the same static outputs. The real capture is the gpu-marked test."""

import contextlib
import functools
import inspect
import os
import sys
import types

import pytest
import torch

import unsloth  # noqa: F401  (must precede transformers)

from real_accelerator import has_real_cuda
import unsloth.models._decode_graphs as G
import unsloth.models._generic_decode_graphs as GG
from unsloth.models import vision

from torch._dynamo.utils import counters
from transformers import AutoModelForCausalLM, DynamicCache, StaticCache
from transformers import cache_utils

LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
GEMMA3 = "hf-internal-testing/tiny-random-Gemma3ForCausalLM"
QWEN3_VL = "trl-internal-testing/tiny-Qwen3VLForConditionalGeneration"
NO_ROLL = "no static sliding layer with the host-built roll index in this transformers"


def test_imports_this_tree():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    assert os.path.abspath(GG.__file__).startswith(here)


class _FakeStream:
    class device:
        index = None

    wait_stream = lambda self, other: None
    synchronize = lambda self: None


def _copy_into(dst, src):
    xs, ys = GG._tensors(dst, []), GG._tensors(src, [])
    assert len(xs) == len(ys)
    for x, y in zip(xs, ys):
        x.copy_(y)


class _FakeGraph:
    """Replay = the captured step run again into the captured outputs; host counters are left
    as a real replay leaves them (unchanged). perturb: True = every replay, an int n = from the
    n-th replay on (drift after validation)."""

    def __init__(self, fn, outputs, cache, perturb):
        self.fn, self.outputs, self.cache, self.perturb = fn, outputs, cache, perturb
        self.replays = 0

    def replay(self):
        self.replays += 1
        host = GG._snapshot(self.cache)[0]
        new = self.fn()
        GG._restore((host, []))
        _copy_into(self.outputs, new)
        if self.perturb is True or (type(self.perturb) is int and self.replays >= self.perturb):
            GG._tensors(self.outputs, [])[0][..., 0].add_(1)


class _Host(GG._Cuda):
    device_type = "cpu"
    perturb = False
    fail = False
    graphs = []
    # As a real capture: no host-to-device copies (torch.tensor(..., device=...)) while recording.
    recording = False
    current_device = staticmethod(lambda: None)
    device = staticmethod(lambda index: contextlib.nullcontext())
    pool = staticmethod(lambda: object())
    new_stream = staticmethod(lambda: _FakeStream())
    stream = staticmethod(lambda stream: contextlib.nullcontext())
    current_stream = staticmethod(lambda: _FakeStream())

    @classmethod
    def capturing(cls):
        return cls.recording

    @classmethod
    def record(cls, pool, fn, cache):
        if cls.fail:
            error = cls.fail if isinstance(cls.fail, BaseException) else None
            raise error or RuntimeError("operation not permitted when stream is capturing")
        # A capture launches nothing: put back the length counters and every cache row the step wrote.
        snapshot = GG._snapshot(cache)
        rows = [
            (t, t.clone())
            for layer in cache.layers
            for t in (getattr(layer, "keys", None), getattr(layer, "values", None))
            if isinstance(t, torch.Tensor)
        ]
        real_tensor = torch.tensor

        def tensor(*args, **kwargs):
            if kwargs.get("device") is not None:
                raise RuntimeError(
                    "Cannot copy between CPU and CUDA tensors during CUDA graph capture unless the CPU tensor is pinned."
                )
            return real_tensor(*args, **kwargs)

        cls.recording = True
        torch.tensor = tensor
        try:
            outputs = fn()
        finally:
            torch.tensor = real_tensor
            cls.recording = False
        GG._restore(snapshot)
        for live, saved in rows:
            live.copy_(saved)
        graph = _FakeGraph(fn, outputs, cache, cls.perturb)
        cls.graphs.append(graph)
        return graph, outputs


@pytest.fixture
def host(monkeypatch):
    backend = type("Host", (_Host,), {"graphs": []})
    monkeypatch.setattr(GG, "_BACKEND", backend)
    return backend


@pytest.fixture(scope = "module")
def llama():
    torch.manual_seed(0)
    return AutoModelForCausalLM.from_pretrained(LLAMA, dtype = torch.float32).eval()


@pytest.fixture(scope = "module")
def gemma3():
    torch.manual_seed(0)
    # A short window, so a short generation runs past it.
    return AutoModelForCausalLM.from_pretrained(
        GEMMA3, dtype = torch.float32, sliding_window = 8
    ).eval()


def _fresh(model):
    model.__dict__.pop(GG._ATTR, None)


def _ids(
    model,
    bsz = 1,
    length = 6,
    pad = 0,
):
    g = torch.Generator().manual_seed(1)
    vocab = model.config.get_text_config().vocab_size
    # Below the multimodal special tokens (image / video pads sit at the top of the vocabulary).
    ids = torch.randint(3, min(vocab, 1000), (bsz, length), generator = g)
    mask = torch.ones_like(ids)
    if pad:
        ids[0, :pad] = 0
        mask[0, :pad] = 0
    return ids, mask


def _generate(
    model,
    ids,
    mask,
    graphs,
    new = 10,
):
    scope = vision._GraphDecodeSteps(model) if graphs else contextlib.nullcontext()
    with torch.inference_mode(), scope:
        return model.generate(
            input_ids = ids,
            attention_mask = mask,
            max_new_tokens = new,
            min_new_tokens = new,
            do_sample = False,
            cache_implementation = "static",
            pad_token_id = 0,
            # transformers 4.57 compiles static-cache generation on CUDA, which the graph path steps aside for.
            disable_compile = True,
        )


def _both(
    model,
    ids,
    mask,
    new = 10,
):
    """Eager then graphed tokens for a fresh model state; asserts they match, returns stats."""
    _fresh(model)
    eager = _generate(model, ids, mask, False, new)
    assert torch.equal(eager, _generate(model, ids, mask, True, new))
    return GG.stats(model)


def _scope(model, forward, eager):
    scope = vision._GraphDecodeSteps.__new__(vision._GraphDecodeSteps)
    scope.model, scope.forward = model, forward
    scope.eager_decodes = scope.eager_regions = eager
    return scope


def _sliding_layer(cls):
    try:
        return cls(max_cache_len = 32, sliding_window = 4)
    except TypeError:
        return cls(sliding_window = 4)


@pytest.mark.parametrize("bsz,pad,length", [(1, 0, 6), (3, 2, 6), (1, 0, 1)])
def test_greedy_tokens_match_eager_llama(host, llama, bsz, pad, length):
    s = _both(llama, *_ids(llama, bsz, length, pad))
    # Step 1 = prefill (eager, also for a one-token prompt), step 2 eager warm-up + capture, step 3
    # validated replay, then replays: 9 decode steps take the graph path.
    assert GG._state(llama).step == 9
    assert s["captures"] == 1 and s["validations"] == 1 and s["replays"] == 8, s
    assert s["refusals"] == {}, s
    assert s["graphs"] == 0  # freed when generate() returns
    assert "forward" not in llama.__dict__


def test_batches_over_the_knob_decode_eagerly(host, llama, monkeypatch):
    monkeypatch.setattr(GG, "MAX_BSZ", 2)
    ids, mask = _ids(llama, 3, pad = 1)
    s = _both(llama, ids, mask)
    assert s["captures"] == 0 and s["refusals"] == {"batch over max": 9}, s
    monkeypatch.setattr(GG, "MAX_BSZ", 3)
    _generate(llama, ids, mask, True)
    assert GG.stats(llama)["replays"] == 8


def test_counts_captures_refusals_and_recompiles_separately(host, llama):
    def compiles():
        return counters["stats"]["unique_graphs"], sum(counters["recompiles"].values())

    layer = torch.nn.Linear(4, 4)
    compiled = torch.compile(lambda x: layer(x).relu(), backend = "eager")
    compiled(torch.ones(2, 4))
    ids, mask = _ids(llama, 2, pad = 1)
    # Warm the compiled mask builder for these shapes eagerly first, so the count does not depend on test order.
    _generate(llama, ids, mask, False)
    before = compiles()
    _fresh(llama)
    _generate(llama, ids, mask, True)
    compiled(torch.ones(2, 4))
    # Decode graphs add no Dynamo graph and no recompile of an existing compiled region.
    assert compiles() == before
    s = GG.stats(llama)
    assert s["captures"] == 1 and s["max_captures_per_key"] == 1 and s["refusals"] == {}, s
    # A second call captures again but does not revalidate the key.
    _generate(llama, ids, mask, True)
    s = GG.stats(llama)
    assert s["captures"] == 2 and s["max_captures_per_key"] == 1 and s["validations"] == 1, s
    assert sum(g.replays for g in host.graphs) == 2 * 8


def test_graph_path_runs_the_compiled_variant_eager_runs(host, llama):
    """A compiled region on the step's arguments (zoo compiles create_causal_mask) guards on their
    inference-ness and strides: the graph path must hit the variant the eager step compiled, never
    a new one, and give the eager logits. (Dynamo's global-state guard also covers the autocast cache
    flag, so the step keeps the ambient autocast; CUDA autocast is not reachable on the host.)"""
    seen, calls = [], []

    @torch.compile(backend = "eager", dynamic = False)
    def region(mask, position):
        # A different variant would add to the logits: the tokens would move.
        flip = 0.0 if mask.is_inference() else 1000.0
        return mask.float().mean() * 0 + position.float().mean() * 0 + flip

    def pre_hook(module, args, kwargs):
        seen.append(kwargs.get("attention_mask"))
        # transformers 5.x passes no cache_position: fall back to the positions.
        position = kwargs.get("cache_position")
        seen.append(position if position is not None else kwargs.get("position_ids"))

    def head_hook(module, args, out):
        mask, position = seen[-2], seen[-1]
        if mask is None or position is None:
            return out
        if isinstance(mask, dict):
            mask = next(iter(mask.values()))
        calls.append(1)
        return out + region(mask, position)

    handles = [
        llama.model.register_forward_pre_hook(pre_hook, with_kwargs = True),
        llama.lm_head.register_forward_hook(head_hook),
    ]
    try:
        _fresh(llama)
        ids, mask = _ids(llama, 3, pad = 2)
        eager = _generate(llama, ids, mask, False)
        assert calls, "the compiled region never ran: the check would be vacuous"
        before = counters["stats"]["unique_graphs"]
        graphed = _generate(llama, ids, mask, True)
        assert counters["stats"]["unique_graphs"] == before
        assert torch.equal(eager, graphed)
        s = GG.stats(llama)
        assert s["replays"] == 8 and not s["refusals"], s
    finally:
        for h in handles:
            h.remove()


@pytest.mark.parametrize("make", ["dense", "sliced", "broadcast", "sliced_broadcast"])
def test_static_buffers_keep_eager_strides(host, make):
    base = torch.arange(64, dtype = torch.float32).view(2, 32)
    leaf = {
        "dense": base,
        "sliced": base[:, ::2],  # shape (2, 16), strides (32, 2)
        "broadcast": base[:1].expand(4, 32),  # as Qwen3-VL positions
        "sliced_broadcast": base[:1, ::2].expand(3, 16),
    }[make]
    compact = GG._compact(leaf)
    if "broadcast" in make:
        assert compact.shape[0] == 1
    buffer = GG._buffer_like(compact)
    buffer.copy_(compact)
    replayed = buffer.expand(leaf.shape)
    assert replayed.stride() == leaf.stride() and torch.equal(replayed, leaf)
    with torch.inference_mode():
        x = torch.arange(6.0).view(2, 3)
    assert GG._buffer_like(x).is_inference() and not GG._buffer_like(base).is_inference()
    t = torch.zeros(4, 6).t()  # non-contiguous
    assert GG._buffer_like(t).stride() == t.stride()
    spec = GG._spec(t, [], None)
    assert t.stride() in spec and False in spec


@pytest.mark.parametrize("past", [False, True])
def test_sliding_window_roll_is_captured_without_host_copies(host, gemma3, monkeypatch, past):
    # Inside the window every sliding layer updates by index: one key, every step after the first two replays.
    # Past the window the sliding layers roll by one slot per step: a second key (window full) is captured and
    # validated (its rolled rows restored between the replay and the eager step), then replays. transformers
    # rolls a full window with an index made by torch.tensor(..., device=...), a host-to-device copy a real
    # capture refuses; the warm-up and recording run that branch on an index made outside the capture. The
    # recording runs no device work, so the capture holds no copy of the full windows' rows while it records.
    taken = []
    snapshot = GG._snapshot

    def recorder(cache, *args, **kwargs):
        out = snapshot(cache, *args, **kwargs)
        if sys._getframe(1).f_code.co_name == "_capture":
            taken.append(max((e[1].numel() for e in out[1]), default = 0))
        return out

    monkeypatch.setattr(GG, "_snapshot", recorder)
    assert "sliding_attention" in gemma3.config.layer_types
    new = gemma3.config.sliding_window + (12 if past else -3)
    keys = 1 + past
    s = _both(gemma3, *_ids(gemma3, 2, length = 3, pad = 1), new)
    assert not s["refusals"], s
    assert s["captures"] == keys and s["validations"] == keys, s
    assert s["replays"] == new - 1 - keys, s
    assert len(taken) == keys and max(taken) <= 1, taken
    for cls, _ in GG._rolling_update_classes():
        assert not getattr(cls.__dict__["update"], "_unsloth_graphable_roll", False)


def test_beam_search_reordering_the_cache_stays_eager(host, llama):
    _fresh(llama)
    ids, mask = _ids(llama)
    kw = dict(max_new_tokens = 8, min_new_tokens = 8, num_beams = 2, do_sample = False)
    kw.update(cache_implementation = "static", pad_token_id = 0, attention_mask = mask)
    with torch.inference_mode():
        eager = llama.generate(ids, **kw)
        with vision._GraphDecodeSteps(llama):
            graphed = llama.generate(ids, **kw)
    assert torch.equal(eager, graphed)
    s = GG.stats(llama)
    moved = s["refusals"].get("cache moved", 0)
    # Either the reorder kept the cache in place (all replays) or every step after it ran eagerly.
    assert (moved == 0 and s["replays"] > 0) or s["replays"] + moved + s["captures"] + s[
        "validations"
    ] == 7, s


def test_qwen3_vl_text_decode_positions_match_the_model(host, monkeypatch):
    import transformers
    from transformers import AutoModelForImageTextToText

    torch.manual_seed(0)
    model = AutoModelForImageTextToText.from_pretrained(QWEN3_VL, dtype = torch.float32).eval()
    seen = []
    real = GG._mrope_positions

    def spy(m, kwargs):
        out = real(m, kwargs)
        seen.append(out is not kwargs)
        return out

    monkeypatch.setattr(GG, "_mrope_positions", spy)
    s = _both(model, *_ids(model, 2, length = 5, pad = 1))
    assert s["replays"] == 8 and not s["refusals"], s
    if transformers.__version__.startswith("4."):
        # 4.x leaves decode positions to the forward, which reads cache_position[0] on the host.
        assert seen and all(seen)


def test_validation_mismatch_refuses_the_key(host, llama):
    host.perturb = True
    ids, mask = _ids(llama)
    s = _both(llama, ids, mask)
    assert s["refusals"].get("validation mismatch", 0) >= 1 and s["replays"] == 0, s
    # Refused for the process: the next call does not capture again.
    _generate(llama, ids, mask, True)
    assert GG.stats(llama)["captures"] == 1


def test_validate_all_pinpoints_a_later_drift(host, llama, monkeypatch):
    _fresh(llama)
    host.perturb = 4  # bitwise at the validated step, then the replay drifts
    ids, mask = _ids(llama)
    eager = _generate(llama, ids, mask, False)
    monkeypatch.setenv("UNSLOTH_DECODE_GENERIC_GRAPHS_VALIDATE_ALL", "1")
    # Every replay is compared and the eager result returned: tokens stay eager's.
    assert torch.equal(eager, _generate(llama, ids, mask, True))
    check = GG.stats(llama)["check"]
    # Decode step 1 captures, step 2 validates (replay 1), drift from replay 4 = decode step 5.
    assert check["steps"] == 8 and check["mismatches"] == 5, check
    assert check["first_mismatch_step"] == 5 and check["max_abs_diff"] == 1.0, check
    assert check["max_kl"] >= 0.0 and check["top1_disagree"] >= 0
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS_VALIDATE_ALL")
    # Without it, the drift goes unseen once the key is trusted: why the debug mode exists.
    _fresh(llama)
    _generate(llama, ids, mask, True)
    assert GG.stats(llama)["check"]["mismatches"] == 0


def test_graph_path_runs_the_eager_path_regions(monkeypatch):
    seen = []
    monkeypatch.setattr(
        vision._generic_decode_graphs,
        "decode_step",
        lambda model, forward, kwargs, regions: seen.append(regions),
    )
    model = torch.nn.Linear(1, 1)
    model.forward = lambda **kw: 0
    for compiled_decode in (False, True):
        _scope(model, model.forward, compiled_decode)._forward(input_ids = torch.zeros(1, 1))
    # Compiled regions stay compiled unless this model's eager decode already leaves them.
    assert seen == [None, vision.unsloth_eager_decode]


def test_capture_failure_falls_back_for_the_key(host, llama):
    host.fail = True
    s = _both(llama, *_ids(llama))
    assert s["captures"] == 0 and s["replays"] == 0, s
    # Refused once for the process; every later step of the key counts as a refused eager step.
    reason = "capture failed: RuntimeError: operation not permitted when stream is capturing"
    assert s["refused_keys"] == 1 and s["refusals"] == {reason: 9}, s
    # What broke it and where (innermost frame outside torch).
    assert s["capture_errors"] == [(reason, s["capture_errors"][0][1])], s
    assert "test_generic_decode_graphs.py" in s["capture_errors"][0][1], s
    assert s["eager_steps"] == 8, s


def test_out_of_memory_capture_stops_the_call_not_the_key(host, llama):
    host.fail = torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 MiB")
    ids, mask = _ids(llama)
    s = _both(llama, ids, mask)
    assert s["captures"] == 0 and s["refused_keys"] == 0, s
    assert s["refusals"] == {"capture failed: out of memory": 9}, s
    host.fail = False
    assert torch.equal(_generate(llama, ids, mask, False), _generate(llama, ids, mask, True))
    s = GG.stats(llama)
    assert s["captures"] == 1 and s["replays"] == 8, s


def test_capture_setup_failure_runs_eagerly(host, llama, monkeypatch):
    def no_pool():
        raise torch.OutOfMemoryError("pool")

    monkeypatch.setattr(host, "pool", staticmethod(no_pool))
    s = _both(llama, *_ids(llama))
    assert s["replays"] == 0 and any("capture setup failed" in r for r in s["refusals"]), s


def test_fingerprint_invalidation(host, llama, monkeypatch):
    from unsloth.kernels import utils as K

    _fresh(llama)
    first = GG._fingerprint(llama)
    assert first is not None and first == GG._fingerprint(llama)
    with torch.no_grad():
        next(llama.parameters()).add_(0)  # in-place edit (merge, optimizer step): bumps the version
    assert GG._fingerprint(llama) != first
    before = GG._fingerprint(llama)
    monkeypatch.setattr(K, "_OPTIMIZER_STEPS", K._OPTIMIZER_STEPS + 1)
    assert GG._fingerprint(llama) != before
    # A step after the optimizer moved re-checks, and a changed fingerprint drops the graphs.
    state = GG._state(llama)
    state.fingerprint = ("stale",)
    state.graphs["k"] = object()
    state.recheck = False
    _generate(llama, *_ids(llama), True)
    assert state.fingerprint != ("stale",)


def test_lora_scale_changes_the_fingerprint(host, llama):
    # PEFT's set_scale / scale_layer move a LoRA layer's Python scale without touching a parameter; the captured
    # kernels baked it in, so a kept graph must not replay across it.
    _fresh(llama)
    proj = llama.model.layers[0].self_attn.o_proj
    proj.scaling = {"default": 1.0}
    try:
        first = GG._fingerprint(llama)
        assert first == GG._fingerprint(llama)
        proj.scaling["default"] = 0.0
        assert GG._fingerprint(llama) != first
        proj.scaling["default"] = 1.0
        assert GG._fingerprint(llama) == first
    finally:
        del proj.scaling
    peft = pytest.importorskip("peft")
    model = peft.get_peft_model(
        AutoModelForCausalLM.from_pretrained(LLAMA, dtype = torch.float32).eval(),
        peft.LoraConfig(r = 4, lora_alpha = 8, target_modules = ["o_proj"]),
    )
    first = GG._fingerprint(model)
    model.base_model.model.model.layers[0].self_attn.o_proj.set_scale("default", 0.0)
    assert GG._fingerprint(model) != first


def test_model_on_two_devices_stays_eager_for_the_call(host, llama, monkeypatch):
    monkeypatch.setattr(GG, "_fingerprint", lambda model, state = None: None)
    s = _both(llama, *_ids(llama))
    assert s["captures"] == 0 and s["refusals"] == {"not one CUDA device / adapters": 9}, s


@pytest.mark.parametrize("drop", [GG.release, G.invalidate, G.release])
def test_weight_changes_revoke_graphs_and_key_decisions(host, llama, drop):
    _fresh(llama)
    state = GG._state(llama)
    state.graphs["k"] = object()
    state.validated.add("key")
    state.refused["other"] = "validation mismatch"
    state.fingerprint = ("old",)
    drop(llama)
    assert not state.graphs and not state.validated and not state.refused
    assert state.fingerprint is None


def test_cache_layer_checks(host, llama):
    assert GG._cache_reason(None) == "no cache"
    assert GG._cache_reason(DynamicCache()) == "cache DynamicCache"
    cache = StaticCache(config = llama.config, max_cache_len = 16)
    assert GG._cache_reason(cache) == "uninitialized layer"
    with torch.inference_mode():
        llama(input_ids = torch.tensor([[5, 6, 7]]), past_key_values = cache, use_cache = True)
    assert GG._cache_reason(cache) is None

    class Recurrent(cache_utils.StaticLayer):  # 5.x linear-attention layers subclass StaticLayer
        pass

    layer = cache.layers[0]
    layer.__class__ = Recurrent
    assert GG._cache_reason(cache) == "layer Recurrent"
    layer.__class__ = cache_utils.StaticLayer
    cache.offloading = True
    assert GG._cache_reason(cache) == "offloaded cache"
    cache.offloading = False
    # Real backend: a host cache is not replayable.
    host.device_type = "cuda"
    assert GG._cache_reason(cache) == "cache not on CUDA"


def test_window_and_host_state_checks():
    layer = cache_utils.StaticSlidingWindowLayer(max_cache_len = 32, sliding_window = 4)
    layer.keys = torch.zeros(1)
    cache = types.SimpleNamespace(layers = [layer])
    name = (
        "cumulative_length_int"
        if "cumulative_length_int" in layer.__dict__
        else "cumulative_length"
    )
    layer.__dict__[name] = 3
    assert GG._window_reason(cache, 1) is None
    # Full (rolls every step): replayable, keyed apart from the filling state.
    layer.__dict__[name] = 4
    assert GG._window_reason(cache, 1) is None
    assert GG._window_state(cache) == (True,)
    # A multi-token step that fills the window switches branches mid-step.
    layer.__dict__[name] = 3
    assert GG._window_reason(cache, 2) == "sliding window filling"
    assert GG._window_state(cache) == (False,)
    layer.__dict__[name] = 4
    before = GG._host_state(cache)
    layer.__dict__[name] = 5
    assert GG._host_state_reason(before, GG._host_state(cache), 1) is None
    layer.__dict__[name] = 7
    assert GG._host_state_reason(before, GG._host_state(cache), 1) == f"cache host state {name}"
    layer.__dict__[name] = 5
    layer.other = "changed"
    assert GG._host_state_reason(before, GG._host_state(cache), 1) == "cache host state other"
    snap = GG._snapshot(cache)
    GG._advance(cache, 1)
    assert layer.__dict__[name] == 6
    GG._restore(snap)
    assert layer.__dict__[name] == 5


def test_argument_keys(host):
    cache = object()
    a, b = [], []
    x = torch.zeros(2, 1, dtype = torch.long)
    k1 = GG._spec({"ids": x, "mask": {"full": torch.zeros(2, 1, 1, 8)}, "n": 1}, a, cache)
    k2 = GG._spec({"ids": x, "mask": {"full": torch.zeros(2, 1, 1, 9)}, "n": 1}, b, cache)
    assert k1 != k2 and len(a) == len(b) == 2
    assert GG._spec({"ids": x.float()}, [], cache) != GG._spec({"ids": x}, [], cache)
    assert GG._spec({"n": 2}, [], cache) != GG._spec({"n": 1}, [], cache)
    assert GG._spec(cache, [], cache) == ("C",)
    for bad, why in (
        (object(), "argument object"),
        (torch.zeros(3, requires_grad = True), "tensor argument requires grad"),
        (torch.zeros(3, device = "meta"), "tensor argument on meta"),
        (torch.zeros(5 * 1024 * 1024), "large tensor argument"),
    ):
        with pytest.raises(GG._Refuse, match = why):
            GG._spec(bad, [], cache)
    leaves = []
    tree = {"ids": x, "mask": {"full": torch.ones(2)}, "n": 1}
    GG._spec(tree, leaves, cache)
    rebuilt = GG._build(tree, iter([t + 1 for t in leaves]), cache)
    assert torch.equal(rebuilt["ids"], x + 1) and rebuilt["n"] == 1


def test_wanted_gating(monkeypatch):
    model = types.SimpleNamespace(device = torch.device("cuda", 0))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS", raising = False)
    monkeypatch.setattr(G, "ENABLED", True)
    assert GG.wanted(model)
    monkeypatch.setenv("UNSLOTH_DECODE_GENERIC_GRAPHS", "0")
    assert not GG.wanted(model)
    monkeypatch.delenv("UNSLOTH_DECODE_GENERIC_GRAPHS")
    monkeypatch.setattr(G, "ENABLED", False)
    assert not GG.wanted(model)
    monkeypatch.setattr(G, "ENABLED", None)
    monkeypatch.setenv("UNSLOTH_DECODE_CUDA_GRAPHS", "0")
    with G.rl_generation():
        assert not GG.wanted(model)
    monkeypatch.setenv("UNSLOTH_DECODE_CUDA_GRAPHS", "1")
    assert GG.wanted(model)
    monkeypatch.setattr(torch.version, "hip", "6.4", raising = False)
    assert not GG.wanted(model)  # =1 does not force it on ROCm
    monkeypatch.setattr(torch.version, "hip", None, raising = False)
    model.forward = lambda **kw: None  # an instance forward (accelerate hook) is left alone
    assert not GG.wanted(model)
    del model.forward
    model.device = torch.device("cpu")
    assert not GG.wanted(model)


def test_inert_when_compiling(host):
    def forward(**kwargs):
        return kwargs["input_ids"] * 2

    model = torch.nn.Linear(1, 1)
    model.forward = forward
    scope = _scope(model, forward, False)

    def step(x):
        assert GG.decode_step(model, forward, {"input_ids": x}) is None
        assert not GG.wanted(model)
        return scope._forward(input_ids = x) + 1

    counters.clear()
    x = torch.ones(2, 1)
    assert torch.equal(torch.compile(step, backend = "eager", fullgraph = True)(x), x * 2 + 1)
    assert not counters["graph_break"]
    assert GG._ATTR not in model.__dict__


def test_generate_routes_only_static_cache_calls():
    source = inspect.getsource(vision.unsloth_base_fast_generate)
    # plan() checks wanted() itself; only static-cache, non-compiled-decode calls reach it.
    call = "_generic_decode_graphs.plan(self, args, kwargs, input_ids)"
    assert call in source
    i = source.index(call)
    guard = source[source.rindex("\n", 0, source.rindex(" if ", 0, i)) : i]
    assert guard.count(" if ") == 1, guard
    for clause in (
        'cache_implementation == "static"',
        "not force_dynamic_cache",
        "not compile_decode",
    ):
        assert clause in guard


@pytest.mark.gpu
@pytest.mark.skipif(not has_real_cuda() or torch.version.hip is not None, reason = "NVIDIA CUDA")
@pytest.mark.parametrize("name", [LLAMA, GEMMA3])
def test_real_capture_matches_eager(name, monkeypatch):
    monkeypatch.setattr(G, "ENABLED", True)
    torch.manual_seed(0)
    # bf16 only where the GPU computes it natively: below sm80 Inductor skips a bf16 frame, which torch 2.14 turns
    # into a hard error inside zoo's fullgraph-compiled Gemma3 RMSNorm (and the skip sticks to that code object for
    # the rest of the process). Unsloth itself never runs Gemma3 in bf16 there (it forces float32).
    bf16 = torch.cuda.is_bf16_supported(including_emulation = False)
    model = (
        AutoModelForCausalLM.from_pretrained(name, dtype = torch.bfloat16 if bf16 else torch.float16)
        .cuda()
        .eval()
    )
    ids, mask = _ids(model, 2, pad = 1)
    ids, mask = ids.cuda(), mask.cuda()
    eager = _generate(model, ids, mask, False, 24)
    graphed = _generate(model, ids, mask, True, 24)
    s = GG.stats(model)
    assert torch.equal(eager, graphed), s
    assert s["captures"] == 1 and s["replays"] == 22 and not s["refusals"], s


def test_grpo_hidden_states_wrapper_keeps_graphs_eligible():
    class M(torch.nn.Module):
        def forward(self, x):
            return x

    m = M()
    assert GG._plain_forward(m)
    m.forward = types.MethodType(M.forward, m)  # the class forward re-bound on the instance
    assert GG._plain_forward(m)
    m.forward = types.MethodType(M.forward, M())  # bound to another module: not plain
    assert not GG._plain_forward(m)

    @functools.wraps(M.forward)
    def autocast_forward(self, *args, **kwargs):  # accelerate's mixed-precision wrapper shape
        return M.forward(self, *args, **kwargs)

    m.forward = types.MethodType(autocast_forward, m)
    assert GG._plain_forward(m)
    m._hf_hook = object()  # accelerate device hook on the module: stays eager
    assert not GG._plain_forward(m)
    del m._hf_hook

    def wrapped_forward(*args, **kwargs):
        return M.forward(m, *args, **kwargs)

    wrapped_forward._unsloth_grpo_hidden_states_forward_wrapped = True
    m.forward = wrapped_forward
    assert GG._plain_forward(m)
    m.forward = types.MethodType(wrapped_forward, m)  # accelerate rebinds it as a bound method
    assert GG._plain_forward(m)
    m.forward = lambda *a, **k: None  # e.g. an accelerate device hook: stays eager
    assert not GG._plain_forward(m)

    # The decode scope puts back whatever instance forward was there before (none: none).
    m = M()
    with vision._EagerDecodeSteps(m):
        assert "forward" in m.__dict__
    assert "forward" not in m.__dict__
    m.forward = types.MethodType(autocast_forward, m)
    previous = m.__dict__["forward"]
    with vision._EagerDecodeSteps(m):
        assert m.__dict__["forward"] is not previous
    assert m.__dict__["forward"] is previous


def test_rolling_update_matches_transformers_rows(host):
    classes = GG._rolling_update_classes()
    assert classes, NO_ROLL
    for cls, counter in classes:
        torch.manual_seed(0)
        layers = [_sliding_layer(cls), _sliding_layer(cls)]
        prefill = torch.randn(2, 3, 6, 8), torch.randn(2, 3, 6, 8)
        steps = [(torch.randn(2, 3, 1, 8), torch.randn(2, 3, 1, 8)) for _ in range(5)]
        for layer in layers:
            layer.update(*prefill, {"cache_position": torch.arange(6)})
        for i, (k, v) in enumerate(steps):
            step_kwargs = {"cache_position": torch.tensor([6 + i])}
            a = layers[0].update(k, v, step_kwargs)
            with GG._graphable_rolls():
                b = layers[1].update(k, v, step_kwargs)
            assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
            assert layers[0].__dict__[counter] == layers[1].__dict__[counter]


def test_full_window_snapshot_keeps_slot_zero_and_undoes_the_roll(host):
    # Validation replays a step that rolls every full window, then puts the rows back for the eager step: keeping
    # the slot the roll drops is enough, not a clone of every window (bsz x 116 MiB on Gemma-3-4B).
    classes = GG._rolling_update_classes()
    assert classes, NO_ROLL
    for cls, counter in classes:
        for graphable in (False, True):
            torch.manual_seed(0)
            layer = _sliding_layer(cls)
            layer.update(
                torch.randn(2, 3, 6, 8),
                torch.randn(2, 3, 6, 8),
                {"cache_position": torch.arange(6)},
            )
            cache = types.SimpleNamespace(layers = [layer])
            for i in range(3):
                keys, values = layer.keys.clone(), layer.values.clone()
                count = layer.__dict__[counter]
                snapshot = GG._snapshot(cache)
                rows = [e[1] for e in snapshot[1] if e[0] is layer.keys or e[0] is layer.values]
                assert len(rows) == 2 and all(r.shape[-2] == 1 for r in rows), [
                    r.shape for r in rows
                ]
                step = (
                    torch.randn(2, 3, 1, 8),
                    torch.randn(2, 3, 1, 8),
                    {"cache_position": torch.tensor([6 + i])},
                )
                with GG._graphable_rolls() if graphable else contextlib.nullcontext():
                    layer.update(*step)
                assert not torch.equal(layer.keys, keys)
                GG._restore(snapshot)
                assert torch.equal(layer.keys, keys) and torch.equal(layer.values, values)
                assert layer.__dict__[counter] == count
                layer.update(*step)
            # Counters only: what a recording, which runs no device work, needs back.
            assert all(e[1].numel() == 1 for e in GG._snapshot(cache, rows = False)[1])


def test_keys_changing_every_step_stop_capturing_for_the_call(host, llama, monkeypatch):
    # An argument whose shape or value moves every step gives every step a new key: each capture is an eager
    # warm-up plus a recording that is never replayed. After a few in a row the call decodes eagerly.
    ids, mask = _ids(llama)
    step = iter(range(10**6))
    real = GG._window_state
    monkeypatch.setattr(GG, "_window_state", lambda cache: real(cache) + (next(step),))
    s = _both(llama, ids, mask, 16)
    assert s["captures"] == GG._MAX_UNREPLAYED_CAPTURES and s["replays"] == 0, s
    assert s["graphs"] == 0, s
    assert s["refusals"] == {"graph key changes every step": 15 - GG._MAX_UNREPLAYED_CAPTURES}, s
    _generate(llama, ids, mask, True, 16)
    assert GG.stats(llama)["captures"] == 2 * GG._MAX_UNREPLAYED_CAPTURES


def test_full_length_type_ids_from_a_trainer_still_replay(host, gemma3, monkeypatch):
    # TRL's GRPO passes the processor's mm_token_type_ids to generate(); it reached every decode step at full
    # length (one column longer per step), so every step had a new key and nothing replayed. With the step's
    # masks prebuilt (a dict) the forward never reads them: the graphed step gets the step's columns.
    real = type(gemma3).prepare_inputs_for_generation

    def prepare(self, input_ids, *args, **kwargs):
        out = real(self, input_ids, *args, **kwargs)
        out["mm_token_type_ids"] = torch.zeros(input_ids.shape, dtype = torch.long)
        return out

    monkeypatch.setattr(type(gemma3), "prepare_inputs_for_generation", prepare)
    seen = []
    real_spec = GG._spec

    def spec(value, leaves, cache):
        out = real_spec(value, leaves, cache)
        if isinstance(value, torch.Tensor) and value.dtype == torch.long and value.dim() == 2:
            seen.append(tuple(value.shape))
        return out

    monkeypatch.setattr(GG, "_spec", spec)
    new = 6
    s = _both(gemma3, *_ids(gemma3, 2, length = 3), new)
    assert s["captures"] == 1 and s["replays"] == new - 2 and not s["refusals"], s
    assert seen and all(shape[-1] == 1 for shape in seen), seen


def test_step_type_ids_only_with_prebuilt_masks():
    t = torch.zeros(2, 9, dtype = torch.long)
    kwargs = {"mm_token_type_ids": t, "attention_mask": torch.ones(2, 9)}
    assert GG._step_type_ids(kwargs, 1) is kwargs
    kwargs = {
        "mm_token_type_ids": t,
        "token_type_ids": t,
        "attention_mask": {"full_attention": None},
    }
    out = GG._step_type_ids(kwargs, 1)
    assert out is not kwargs and kwargs["mm_token_type_ids"] is t
    for name in ("mm_token_type_ids", "token_type_ids"):
        assert out[name].shape == (2, 1) and out[name].stride() == (1, 1)


def test_step_that_fills_the_window_is_never_captured(host, gemma3):
    # A first decode step at count == window - 1 fills the window. Its key is the filling one, but the capture
    # records after the warm-up advanced the counters (the rolling branch): a kept graph of that key would then
    # roll a window that is still filling. That step runs eagerly; the window-full key is captured after it.
    window = gemma3.config.sliding_window
    # A call past the window first: the roll index exists, so a capture of the rolling branch would succeed.
    _fresh(gemma3)
    _generate(gemma3, *_ids(gemma3, 2, length = 3), True, window + 4)
    assert GG._ROLL_INDEX
    new = 8
    s = _both(gemma3, *_ids(gemma3, 2, length = window - 1), new)
    assert s["refusals"] == {"sliding window fills in this step": 1}, s
    assert s["captures"] == 1 and s["validations"] == 1 and s["replays"] == new - 3, s
    state = gemma3.__dict__[GG._ATTR]
    assert all(all(key[2]) for key in state.validated), state.validated
