# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CUDA-graph decode for any FastModel model generating over a transformers StaticCache.

Each one-token decode step of `unsloth_base_fast_generate` replays a CUDAGraph of the model's own step,
with tensor arguments copied into static buffers. Only where `_decode_graphs.enabled()`, on NVIDIA CUDA,
over static attention layers that index the cache on the device. Per graph key: the first step runs
eagerly, then is captured; the first replay must match the eager step bitwise; a host sync, host-state
change or mismatch refuses the key. The step runs in the eager step's exact state (argument inference-ness
and strides, ambient autocast + cache flag, compiled regions as eager runs them): Dynamo guards on all of
these and another compiled variant rounds differently. Plain generate() runs inside generate_scope(), so
graphs are on by default there (plan() decides per call: short calls, low free memory and unsupported
calls decode eagerly; an out-of-memory graphed call is redone once eagerly). After an inference-mode call
the graph and its StaticCache are kept for the next call of the same batch (keep()); training, a returned
or caller-owned cache, and parameter / adapter / quant changes drop them. UNSLOTH_DECODE_GENERIC_GRAPHS=0
disables; _VALIDATE_ALL=1 checks every replay (debug); batches over UNSLOTH_DECODE_GENERIC_GRAPH_MAX_BSZ
(16) decode eagerly; UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP=0 drops graphs at the end of every call."""

import collections
import collections.abc
import contextlib
import warnings
import copy
import functools
import os
import traceback
import weakref

import torch

from . import _decode_graphs as G

__all__ = [
    "wanted",
    "plan",
    "ineligible",
    "begin",
    "end",
    "decode_step",
    "release",
    "stats",
    "keepable",
    "kept_cache",
    "redo_eagerly",
]

_ATTR = "_unsloth_generic_decode_graphs"
# Host length counters of static cache layers (4.57 sliding: cumulative_length; 5.x: cumulative_length_int).
_COUNTERS = ("cumulative_length", "cumulative_length_int")
_MAX_ARG_BYTES = 16 * 1024 * 1024
_CONSTANTS = (bool, int, float, str, torch.dtype, torch.device)
_MAX_KEYS = 4096
# Graphs only remove launch overhead (T4 1B 4-bit: 4-7x at bsz 1, 2.7-4.7x at 16).
MAX_BSZ = int(os.environ.get("UNSLOTH_DECODE_GENERIC_GRAPH_MAX_BSZ", "16"))
# Free memory a graphed call needs beyond the eager one: the graph pool (8-16 MiB measured on 1B models),
# static input buffers and the side stream's workspace. Below it the call decodes eagerly.
_FREE_MARGIN_BYTES = 256 * 1024 * 1024
# A kept cache longer than this many times the call's own length is not reused (attention reads every slot).
_KEEP_MAX_RATIO = 2
# Captures in a row without a replay in between: the step's key changes every step (an argument whose shape or
# value moves), so every capture costs an eager warm-up plus a recording and is never replayed. The rest of the
# call decodes eagerly. A healthy call captures at most twice in a row (window filling, then full).
_MAX_UNREPLAYED_CAPTURES = 4
# Per-call plan decisions by reason (tests, benchmarks).
PLANS = collections.Counter()
EAGER_REDOS = 0
_OOM_WARNED = False

try:
    from torch.nn.attention.flex_attention import BlockMask as _BlockMask
except Exception:
    _BlockMask = None


class _Refuse(Exception):
    pass


class _Cuda:
    """The CUDA calls this path makes (tests swap in a host stand-in)."""

    device_type = "cuda"

    @staticmethod
    def available():
        # NVIDIA CUDA only: HIP replay is not verified against the eager step.
        return torch.version.hip is None and torch.cuda.is_available()

    @staticmethod
    def current_device():
        return torch.cuda.current_device()

    @staticmethod
    def free_bytes(device):
        return G._free_bytes(device)

    @staticmethod
    def device(index):
        return torch.cuda.device(index)

    @staticmethod
    def capturing():
        return torch.cuda.is_current_stream_capturing()

    @staticmethod
    def pool():
        return torch.cuda.graph_pool_handle()

    @staticmethod
    def new_stream():
        return torch.cuda.Stream()

    @staticmethod
    def stream(stream):
        return torch.cuda.stream(stream)

    @staticmethod
    def current_stream():
        return torch.cuda.current_stream()

    @staticmethod
    def device_stream(index):
        return torch.cuda.current_stream(index)

    @staticmethod
    def multidevice_supported():
        return True

    @staticmethod
    def mempool():
        # One graph pool per other card, shared by that card's segments (captured and replayed in order).
        return torch.cuda.graph_pool_handle()

    @staticmethod
    def side_stream(index):
        return torch.cuda.Stream(device = index)

    @staticmethod
    @contextlib.contextmanager
    def on_streams(streams):
        """The other cards' current streams set to `streams` ({index: stream}), restored after."""
        before = {d: torch.cuda.current_stream(d) for d in streams}
        here = torch.cuda.current_device()
        try:
            for d, stream in streams.items():
                stream.wait_stream(before[d])
                with torch.cuda.device(d):
                    torch.cuda.set_stream(stream)
            yield
        finally:
            for d, stream in before.items():
                stream.wait_stream(streams[d])
                with torch.cuda.device(d):
                    torch.cuda.set_stream(stream)
            torch.cuda.set_device(here)

    @staticmethod
    def card_of(value):
        """The card index a tensor lives on, or None (host tensors)."""
        if not value.is_cuda:
            return None
        return value.device.index if value.device.index is not None else torch.cuda.current_device()

    @staticmethod
    def index_of(device):
        """The card index a send_to_device target names (int, "cuda:1", torch.device), or None."""
        if isinstance(device, int):
            return device
        try:
            device = torch.device(device)
        except Exception:
            return None
        if device.type != "cuda":
            return None
        return device.index if device.index is not None else torch.cuda.current_device()

    @staticmethod
    def card_device(index):
        return torch.device("cuda", index)

    @staticmethod
    def new_graph():
        return torch.cuda.CUDAGraph()

    @staticmethod
    def begin_on(graph, pool, index):
        """Capture on card `index`'s current stream (its side stream, made current by on_streams)."""
        torch.cuda.set_device(index)
        G._begin(graph, pool)

    @staticmethod
    def end_on(graph):
        with warnings.catch_warnings():
            # A segment with no kernels (the tail after the last hand-off) is an empty, valid graph.
            warnings.simplefilter("ignore")
            G._end(graph)

    @staticmethod
    def record(
        pool,
        fn,
        cache,
        others = None,
        mempools = None,
    ):
        """(graph, fn()) with fn's launches captured into graph (`cache`: for host stand-ins).
        `others` ({index: stream}): the other cards of a model split over cards, their side streams
        current there (on_streams). The step is then captured as one graph per run of work on one
        card, cut where accelerate's AlignDevicesHook moves tensors to another card; those hand-offs
        run eagerly between the segments' replays (_Segments). Capturing a cross-card copy is not
        possible without peer access (cudaErrorStreamCaptureUnsupported on PCIe T4s)."""
        if others:
            here = torch.cuda.current_device()
            pools = dict(mempools)
            pools[here] = pool
            try:
                return _SegmentRecorder(_BACKEND, pools, here).run(fn)
            finally:
                torch.cuda.set_device(here)
        graph = torch.cuda.CUDAGraph()
        G._begin(graph, pool)
        try:
            out = fn()
        except BaseException:
            # End the invalidated capture, but report what broke it, not the capture_end error.
            if torch.cuda.is_current_stream_capturing():
                try:
                    G._end(graph)
                except Exception:
                    pass
            raise
        G._end(graph)
        return graph, out


class _Segments:
    """One decode step of a model split over cards: ("graph", graph, card) parts replayed on their card's
    current stream, and ("copy", dst, src) hand-offs (accelerate's moves at a block boundary) run eagerly
    between them into the static buffers the next segment reads. Same order as the capture."""

    def __init__(self):
        self.parts = []

    def replay(self):
        for kind, a, b in self.parts:
            if kind == "graph":
                with _BACKEND.device(b):
                    a.replay()
            else:
                # Eager cross-card copy: PyTorch orders it after src's stream and before dst's.
                a.copy_(b)


class _SegmentRecorder:
    """Captures fn() as per-card segments. accelerate's AlignDevicesHook pre_forward / post_forward move
    tensors through accelerate.hooks.send_to_device; while recording, that name points at self.send:
    the first move of a tensor to another card ends the open segment, copies eagerly into a buffer kept
    for replay (recorded as a hand-off), and the next segment starts on the target card when the hook
    returns. A tensor already moved to that card in this step reuses its buffer (no new cut), so a
    layer-invariant input (rotary cos / sin, masks) crosses once."""

    def __init__(self, backend, pools, first):
        self.backend = backend
        self.pools = pools
        self.segments = _Segments()
        self.graph = None
        self.card = first
        self.pending = None
        self.moved = {}
        self.original = None

    def begin(self, card):
        self.graph = self.backend.new_graph()
        self.card = card
        self.backend.begin_on(self.graph, self.pools[card], card)

    def end(self):
        if self.graph is not None:
            graph, self.graph = self.graph, None
            self.backend.end_on(graph)
            self.segments.parts.append(("graph", graph, self.card))

    def resume(self):
        if self.pending is not None and self.graph is None:
            card, self.pending = self.pending, None
            self.begin(card)

    def _move(self, value, card, target, skip_keys):
        if isinstance(value, torch.Tensor):
            if self.backend.card_of(value) in (None, card):
                return value
            key = (id(value), card)
            hit = self.moved.get(key)
            if hit is not None:
                return hit[1]
            # First crossing of this tensor to `card`: the segment ends here, the copy runs eagerly.
            self.end()
            dst = value.to(target)
            self.moved[key] = (value, dst)
            self.segments.parts.append(("copy", dst, value))
            self.pending = card
            return dst
        if hasattr(value, "to"):
            # Not a tensor but movable (BatchEncoding, ...): accelerate's own move, as eager. A real
            # cross-card copy in it fails the capture, which refuses the key.
            return self.original(value, target, skip_keys = skip_keys)
        if isinstance(value, (tuple, list)):
            from accelerate.utils.operations import honor_type
            return honor_type(value, (self._move(v, card, target, skip_keys) for v in value))
        if isinstance(value, collections.abc.Mapping):
            skip = [skip_keys] if isinstance(skip_keys, str) else (skip_keys or [])
            return type(value)(
                {
                    k: v if k in skip else self._move(v, card, target, skip_keys)
                    for k, v in value.items()
                }
            )
        return value

    def send(
        self,
        original,
        tensor,
        device,
        non_blocking = False,
        skip_keys = None,
    ):
        card = self.backend.index_of(device)
        if card is None:
            return original(tensor, device, non_blocking = non_blocking, skip_keys = skip_keys)
        return self._move(tensor, card, self.backend.card_device(card), skip_keys)

    def run(self, fn):
        import accelerate.hooks as hooks

        original_send = hooks.send_to_device
        cls = hooks.AlignDevicesHook
        original_pre, original_post = cls.pre_forward, cls.post_forward
        recorder = self

        def pre_forward(hook, module, *args, **kwargs):
            out = original_pre(hook, module, *args, **kwargs)
            recorder.resume()
            return out

        def post_forward(hook, module, output):
            out = original_post(hook, module, output)
            recorder.resume()
            return out

        self.original = original_send
        hooks.send_to_device = functools.partial(self.send, original_send)
        cls.pre_forward, cls.post_forward = pre_forward, post_forward
        try:
            self.begin(self.card)
            out = fn()
            self.resume()
            self.end()
        except BaseException:
            if self.graph is not None:
                try:
                    self.backend.end_on(self.graph)
                except Exception:
                    pass
                self.graph = None
            raise
        finally:
            hooks.send_to_device = original_send
            cls.pre_forward, cls.post_forward = original_pre, original_post
        return self.segments, out


_BACKEND = _Cuda


def _static_layer_types():
    try:
        from transformers import cache_utils
    except Exception:
        return (), ()
    full = tuple(t for t in (getattr(cache_utils, "StaticLayer", None),) if isinstance(t, type))
    sliding = tuple(
        t
        for t in (
            getattr(cache_utils, name, None)
            for name in ("StaticSlidingWindowLayer", "SlidingWindowLayer", "ChunkedSlidingLayer")
        )
        if isinstance(t, type)
    )
    return full, sliding


_FULL_LAYERS, _SLIDING_LAYERS = _static_layer_types()


def wanted(model):
    """Whether this generate() call routes its decode steps through decode_step()."""
    if torch.compiler.is_compiling():
        return False
    if os.environ.get("UNSLOTH_DECODE_GENERIC_GRAPHS", "1") == "0" or not G.enabled():
        return False
    if torch.version.hip is not None or not _BACKEND.available():
        return False
    if not _FULL_LAYERS:
        return False
    device = getattr(model, "device", None)
    return getattr(device, "type", None) == _BACKEND.device_type and _plain_forward(model)


def _plan_len(model, args, kwargs):
    """prompt + new tokens this call can reach (0 if unknown), as the llama path sizes its plan."""
    try:
        from .llama import _generation_plan_len
        return int(_generation_plan_len(model, args, kwargs) or 0)
    except Exception:
        return 0


def plan(model, args, kwargs, input_ids):
    """ "graph" or "eager" for this generate() call, decided once before it decodes (inside
    generate_scope(), so graphs are on by default). Eager: graphs not wanted (wanted()), batch over
    MAX_BSZ, too few decode steps to win back a capture (unless a kept graph of this batch replays from
    the first step), or too little free memory for the graph pool. A kept graph / cache that this call
    cannot reuse is dropped here."""
    state = model.__dict__.get(_ATTR)
    if not wanted(model):
        if state is not None:
            state.clear()
        PLANS["not wanted"] += 1
        return "eager"
    try:
        prompt = int(input_ids.shape[1]) if input_ids.dim() >= 2 else 0
        bsz = G.decode_batch(
            int(input_ids.shape[0]), kwargs, getattr(model, "generation_config", None)
        )
    except Exception:
        PLANS["unknown shape"] += 1
        return "eager"
    max_len = _plan_len(model, args, kwargs)
    kept = (
        state is not None
        and state.kept is not None
        and state.kept_bsz == bsz
        and bool(state.graphs)
    )
    if kept and max_len:
        # transformers sizes the cache prompt + new tokens - 1 (the last token is never written back).
        length = _kept_len(state)
        kept = length is not None and max_len - 1 <= length <= _KEEP_MAX_RATIO * max_len
    if state is not None and state.kept is not None and not kept:
        state.clear()
    if bsz > MAX_BSZ:
        PLANS["batch over max"] += 1
        return "eager"
    if max_len:
        left = max_len - prompt - 1
        # Step 1 is eager + capture, step 2 a validated replay: fewer steps cannot win back a capture.
        if left < 2 or (left < G.CAPTURE_MIN_STEPS and not kept):
            PLANS["short call"] += 1
            return "eager"
    if not kept:
        try:
            # A model split over cards needs room for its graph pool on every card it runs on.
            devices = [input_ids.device] + [
                torch.device(_BACKEND.device_type, d)
                for d in sorted(set((getattr(model, "hf_device_map", None) or {}).values()))
                if type(d) is int and d != input_ids.device.index
            ]
            free = min(_BACKEND.free_bytes(d) for d in devices)
        except Exception:
            free = None
        if free is not None and free < _FREE_MARGIN_BYTES:
            PLANS["low free memory"] += 1
            return "eager"
    PLANS["graph kept" if kept else "graph"] += 1
    return "graph"


def ineligible(model):
    """A call that cannot decode through graphs (dynamic cache, compiled decode): eager, and a kept
    graph / cache goes (it would only hold memory)."""
    state = model.__dict__.get(_ATTR)
    if state is not None and state.kept is not None:
        state.clear()
    PLANS["ineligible"] += 1
    return "eager"


def replayable(kwargs):
    """A failed graphed call can be redone eagerly only when no caller object holds its partial state."""
    return not (
        kwargs.get("streamer", None) is not None
        or kwargs.get("past_key_values", None) is not None
        or kwargs.get("stopping_criteria", None)
        or kwargs.get("logits_processor", None)
    )


def rng_state(device):
    """Host + this CUDA device's RNG state, so an eager redo samples as the failed attempt started."""
    try:
        cuda = torch.cuda.get_rng_state(device) if getattr(device, "type", None) == "cuda" else None
    except Exception:
        cuda = None
    return torch.get_rng_state(), cuda, device


def set_rng_state(saved):
    host, cuda, device = saved
    torch.set_rng_state(host)
    if cuda is not None:
        torch.cuda.set_rng_state(cuda, device)


def redo_eagerly(model):
    """After an out-of-memory graphed call: free graphs, kept cache and allocator blocks before the redo."""
    global EAGER_REDOS, _OOM_WARNED
    release(model)
    model.__dict__.pop("_unsloth_generic_call_cache", None)
    try:
        G._empty_cache()
    except Exception:
        pass
    EAGER_REDOS += 1
    if not _OOM_WARNED:
        _OOM_WARNED = True
        import warnings
        warnings.warn(
            "Unsloth: generate() ran out of GPU memory with CUDA-graph decode; freed the decode graphs "
            "and decoded this call eagerly (UNSLOTH_DECODE_CUDA_GRAPHS=0 always decodes eagerly).",
            stacklevel = 3,
        )


def _plain_forward(model):
    """No instance forward; the class forward bound on the instance, bare or under functools.wraps
    wrappers (accelerate's mixed-precision autocast / fp32-output wrappers, which eager decode runs too);
    Unsloth's GRPO hidden-states wrapper, which passes straight through while generating; or accelerate's
    device hook of a model on one device (a bitsandbytes model placed on a card other than cuda:0 gets
    one): its input / output moves are no-ops there, and the eager step runs the same hook. Offloading
    hooks and device maps over several devices stay eager."""
    forward = model.__dict__.get("forward")
    if forward is None:
        return True
    if isinstance(forward, functools.partial) and "_hf_hook" in model.__dict__:
        # accelerate's add_hook_to_module: forward = partial(new_forward, model), original in _old_forward.
        if not (forward.args and forward.args[0] is model and not forward.keywords):
            return False
        if not _single_device_hook(model):
            return False
        old = model.__dict__.get("_old_forward")
        return old is not None and _plain_bound(model, old)
    if hasattr(model, "_hf_hook"):
        return False
    return _plain_bound(model, forward)


def _plain_bound(model, forward):
    func = getattr(forward, "__func__", forward)
    if getattr(func, "_unsloth_grpo_hidden_states_forward_wrapped", False):
        return True
    if getattr(forward, "__self__", None) is not model:
        return False
    target = getattr(type(model), "forward", None)
    for _ in range(4):
        if func is target:
            return True
        func = getattr(func, "__wrapped__", None)
        if func is None:
            return False
    return False


def _single_device_hook(model):
    """accelerate AlignDevicesHook(s) that only move tensors to the model's own (single) device: no
    offload, no weights map. Another hook type, or an execution device elsewhere, is not ours to graph."""
    try:
        from accelerate.hooks import AlignDevicesHook, SequentialHook
    except Exception:
        return False
    hook = model.__dict__.get("_hf_hook")
    hooks = list(hook.hooks) if isinstance(hook, SequentialHook) else [hook]
    device = getattr(model, "device", None)
    for h in hooks:
        if type(h) is not AlignDevicesHook:
            return False
        if h.offload or getattr(h, "offload_buffers", False) or h.weights_map is not None:
            return False
        target = h.execution_device
        if target is None:
            continue
        try:
            target = torch.device(target)
        except Exception:
            return False
        if not isinstance(device, torch.device) or target.type != device.type:
            return False
        same = (target.index if target.index is not None else 0) == (
            device.index if device.index is not None else 0
        )
        if not same and not _split_entry_card(model, target):
            return False
    return True


def _split_entry_card(model, target):
    """A model split over cards (multi-device graphs on): its root hook moves the inputs to the card of the
    first block, which need not be `model.device` (the first parameter: a tied embedding / lm_head can sit
    on the last card). Any card the model's parameters are on is its own."""
    if not _multidevice_enabled():
        return False
    devices = _param_devices(model)
    return len(devices) > 1 and _device_index(target) in {
        _device_index(d) for d in devices if d.type == target.type
    }


class _Entry:
    __slots__ = ("graph", "leaves", "outputs", "validated")

    def __init__(self, graph, leaves, outputs, validated):
        self.graph = graph
        self.leaves = leaves
        self.outputs = outputs
        self.validated = validated


class _State:
    def __init__(self):
        self.graphs = collections.OrderedDict()
        self.pool = None
        # A model split over cards: the other cards' indices and their MemPools for the graphs.
        self.others = ()
        self.mempools = {}
        self.side = {}
        self.fp_reason = None
        self.stream = None
        self.fingerprint = None
        self.optimizer_steps = -1
        self.recheck = True
        self.cache_sig = None
        # Keys refused for the process (host sync in capture, host state, validation mismatch).
        self.refused = {}
        self.validated = set()
        self.moved = None
        self.key_captures = collections.Counter()
        self.step = 0
        self.forwards = 0
        self.capture_errors = []
        self.check = _new_check()
        self.captures = 0
        self.replays = 0
        self.validations = 0
        self.eager_steps = 0
        self.refusals = collections.Counter()
        # keep(): the StaticCache the kept graphs address, its batch, and the training-forward hook.
        self.kept = None
        self.kept_bsz = None
        # The max_cache_len the kept cache was built for (5.x), else None (its full layers' length is used).
        self.kept_len = None
        self.hook = None
        self.owner = None
        self.kept_calls = 0
        self.active = False
        # This call: captures since the last replay, and the reason it stopped capturing (None: it did not).
        self.unreplayed = 0
        self.call_stop = None

    def clear(self):
        self.graphs.clear()
        self.cache_sig = None
        # A pool whose graphs are all freed is released by the allocator: never reuse its handle.
        self.pool = None
        self.mempools = {}
        self.drop_kept()

    def invalidate(self):
        """Weights / adapters / quant state changed: graphs and every per-key decision go."""
        self.clear()
        self.fingerprint = None
        self.validated.clear()
        self.refused.clear()

    def drop_kept(self):
        """Forget the kept cache (and free it when the model still holds it for the next call)."""
        kept, self.kept, self.kept_bsz, self.kept_len = self.kept, None, None, None
        if self.hook is not None:
            self.hook.remove()
            self.hook = None
        owner = self.owner() if self.owner is not None else None
        # Inside a call the cache is that call's (transformers holds it): only forget it.
        if (
            not self.active
            and kept is not None
            and owner is not None
            and owner.__dict__.get("_cache") is kept
        ):
            del owner.__dict__["_cache"]


def _state(model):
    state = model.__dict__.get(_ATTR)
    if state is None:
        state = model.__dict__[_ATTR] = _State()
        state.owner = weakref.ref(model)
    return state


def begin(model):
    """Start of a generate() call: the first graph step re-checks the fingerprint."""
    state = _state(model)
    state.recheck = True
    state.forwards = 0
    state.moved = None
    state.key_captures.clear()
    state.step = 0
    state.check = _new_check()
    state.active = True
    state.unreplayed = 0
    state.call_stop = None
    if state.hook is not None:
        # Only needed between calls: no per-step hook while this call decodes.
        state.hook.remove()
        state.hook = None
    if len(state.refused) + len(state.validated) > _MAX_KEYS:
        state.refused.clear()
        state.validated.clear()


def end(model, keep = False):
    """End of a generate() call. Its graphs read this call's cache: freed, unless `keep` (keepable()), when
    the graphs and that cache stay for the next call of the same batch (released by training)."""
    state = model.__dict__.get(_ATTR)
    if state is None:
        return
    state.active = False
    cache = model.__dict__.pop("_unsloth_generic_call_cache", None)
    length = model.__dict__.pop("_unsloth_generic_call_len", None)
    if (
        not keep
        or not state.graphs
        or cache is None
        or state.cache_sig is None
        or state.cache_sig[0] != id(cache)
    ):
        state.clear()
        return
    state.kept, state.kept_bsz, state.kept_len = cache, _cache_batch(cache), length
    state.kept_calls += 1
    if state.hook is None:
        state.hook = model.register_forward_pre_hook(_release_if_training)


def _release_if_training(module, args):
    # Training can resume without for_training() (sampling in a trainer callback): free the kept graph and cache.
    if module.training and torch.is_grad_enabled():
        release(module)


def _cache_batch(cache):
    """Rows of a StaticCache (4.x: max_batch_size; 5.x layers only carry their tensors)."""
    try:
        keys = cache.layers[0].keys
        if isinstance(keys, torch.Tensor) and keys.dim() >= 1:
            return int(keys.shape[0])
    except Exception:
        pass
    try:
        return int(cache.max_batch_size)
    except Exception:
        return None


def _kept_len(state):
    # Sliding layers are capped at their window, so only the length the cache was built for says what it holds.
    return state.kept_len if state.kept_len is not None else _cache_len(state.kept)


def _cache_len(cache):
    """Slots of a StaticCache's full layers (sliding layers are shorter; 5.x has no cache-level length)."""
    try:
        return max(int(layer.max_cache_len) for layer in cache.layers)
    except Exception:
        return None


def kept_cache(model):
    state = model.__dict__.get(_ATTR)
    return state.kept if state is not None else None


def keepable(model, kwargs, output):
    """Whether this inference-mode call's graphs and StaticCache may stay for the next call: not when the
    caller passed or got back the cache (a later call would overwrite it), nor in training mode."""
    if os.environ.get("UNSLOTH_DECODE_GENERIC_GRAPHS_KEEP", "1") == "0":
        return False
    if getattr(model, "training", False) or kwargs.get("past_key_values", None) is not None:
        return False
    if getattr(output, "past_key_values", None) is not None:
        return False
    return True


def release(model):
    for module in model.modules():
        state = module.__dict__.get(_ATTR)
        if state is not None:
            state.invalidate()


def _kept_compatible(state, batch_size, max_cache_len, offloading):
    kept = state.kept
    if kept is None:
        return False
    length = _kept_len(state)
    if length is None:
        return False
    return (
        state.kept_bsz == batch_size
        and bool(getattr(kept, "offloading", False)) == bool(offloading)
        and max_cache_len <= length <= _KEEP_MAX_RATIO * max(int(max_cache_len), 1)
    )


class reuse_kept_cache:
    """One graphed generate() call: transformers builds this call's StaticCache, unless a compatible kept
    one exists, which is reset and handed back (same tensors, so the kept graphs replay). 5.x builds a new
    cache per call in _prepare_static_cache; 4.x reuses model._cache, which _clear_generation_caches keeps
    while it is the kept cache. The cache the call ends up with is recorded for end(keep=True)."""

    def __init__(self, model):
        self.model = model
        self.prepare = None

    def _prepare(self, *args, **kwargs):
        model = self.model
        state = _state(model)
        names = ("cache_implementation", "batch_size", "max_cache_len")
        values = dict(zip(names, args))
        values.update({k: v for k, v in kwargs.items() if k in names})
        cache_implementation = values.get("cache_implementation", "static")
        batch_size, max_cache_len = values.get("batch_size"), values.get("max_cache_len")
        prefill_chunk = kwargs.get("prefill_chunk_size", args[3] if len(args) > 3 else None)
        effective = max_cache_len
        previous = getattr(model, "_previous_max_cache_length", -1)
        if type(effective) is int and type(previous) is int:
            effective = max(effective, previous)
        if (
            type(batch_size) is int
            and type(effective) is int
            and prefill_chunk is None
            and not getattr(model.config, "is_encoder_decoder", False)
            and _kept_compatible(
                state, batch_size, effective, "offloaded" in str(cache_implementation)
            )
        ):
            cache = state.kept
            cache.reset()
            PLANS["reused kept cache"] += 1
        else:
            if state.kept is not None:
                # A new cache: the kept graphs address the old one.
                state.clear()
            cache = self.prepare(*args, **kwargs)
        model.__dict__["_unsloth_generic_call_cache"] = cache
        if type(effective) is int:
            model.__dict__["_unsloth_generic_call_len"] = (
                effective if cache is not state.kept else _kept_len(state)
            )
        return cache

    def __enter__(self):
        model = self.model
        if "_prepare_static_cache" in model.__dict__ or not hasattr(model, "_prepare_static_cache"):
            # 4.x (or a wrapper owns the hook): the call's cache is model._cache, read in end().
            self.prepare = None
            return self
        self.prepare = model._prepare_static_cache
        model._prepare_static_cache = self._prepare
        return self

    def __exit__(self, *exc):
        model = self.model
        if self.prepare is not None:
            del model._prepare_static_cache
        elif "_unsloth_generic_call_cache" not in model.__dict__:
            cache = model.__dict__.get("_cache")
            if cache is not None:
                model.__dict__["_unsloth_generic_call_cache"] = cache
        return False


def stats(model):
    state = getattr(model, "__dict__", {}).get(_ATTR)
    if state is None:
        return {}
    return {
        "captures": state.captures,
        "replays": state.replays,
        "validations": state.validations,
        "eager_steps": state.eager_steps,
        "graphs": len(state.graphs),
        "refusals": dict(state.refusals),
        "refused_keys": len(state.refused),
        "max_captures_per_key": max(state.key_captures.values(), default = 0),
        "capture_errors": list(state.capture_errors),
        "check": dict(state.check),
        # Other cards of a model split over cards that the graphs run on.
        "other_devices": list(state.others),
    }


def _new_check():
    return {
        "steps": 0,
        "mismatches": 0,
        "first_mismatch_step": None,
        "max_abs_diff": 0.0,
        "max_kl": 0.0,
        "top1_disagree": 0,
    }


def _validate_all():
    return os.environ.get("UNSLOTH_DECODE_GENERIC_GRAPHS_VALIDATE_ALL", "0") == "1"


def _record_check(state, eager, replay, equal):
    """Accumulates how far the replay's first output (logits) is from the eager step's."""
    check = state.check
    check["steps"] += 1
    if not equal:
        check["mismatches"] += 1
        if check["first_mismatch_step"] is None:
            check["first_mismatch_step"] = state.step
    xs, ys = _tensors(eager, []), _tensors(replay, [])
    if not xs or not ys or not xs[0].is_floating_point() or xs[0].shape != ys[0].shape:
        return
    x, y = xs[0].float(), ys[0].float()
    check["max_abs_diff"] = max(check["max_abs_diff"], float((x - y).abs().max()))
    if x.dim() >= 2:
        p, q = torch.log_softmax(x, -1), torch.log_softmax(y, -1)
        kl = (p.exp() * (p - q)).sum(-1).max()
        check["max_kl"] = max(check["max_kl"], float(kl))
        check["top1_disagree"] += int((x.argmax(-1) != y.argmax(-1)).sum())


def _where(error):
    """file:line of the innermost frame outside torch that raised `error`."""
    for frame in reversed(traceback.extract_tb(error.__traceback__)):
        if f"{os.sep}torch{os.sep}" not in frame.filename:
            return f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}"
    return None


def _refusal(state, reason):
    state.refusals[reason] += 1
    state.eager_steps += 1
    return None


def _cache_reason(cache):
    """None when every layer of `cache` is a static attention layer this path can replay."""
    if cache is None:
        return "no cache"
    try:
        from transformers import StaticCache
    except Exception:
        return "no StaticCache"
    if not isinstance(cache, StaticCache):
        return f"cache {type(cache).__name__}"
    if getattr(cache, "offloading", False):
        return "offloaded cache"
    layers = getattr(cache, "layers", None)
    if not layers:
        return "legacy cache"
    for layer in layers:
        kind = type(layer)
        # Exact types: 5.x linear-attention / indexed layers subclass StaticLayer.
        if kind not in _FULL_LAYERS and kind not in _SLIDING_LAYERS:
            return f"layer {kind.__name__}"
        keys = getattr(layer, "keys", None)
        if not getattr(layer, "is_initialized", False) or not isinstance(keys, torch.Tensor):
            return "uninitialized layer"
        if keys.device.type != _BACKEND.device_type:
            return "cache not on CUDA"
    return None


def _host_counter(layer):
    for name in ("cumulative_length_int", "cumulative_length"):
        value = layer.__dict__.get(name)
        if type(value) is int:
            return value
    return None


def _window_reason(cache, q_len):
    # A step that fills a sliding window switches the update's branch mid-call: eager. Before and after it, every
    # one-token step runs the same device work (write at the device counter, or roll by one slot), and the key
    # carries which of the two each layer is in (_window_state), so a replay repeats the right one.
    for layer in cache.layers:
        if type(layer) in _SLIDING_LAYERS:
            count = _host_counter(layer)
            if count is None:
                return "sliding layer without host counter"
            if count < layer.max_cache_len < count + q_len:
                return "sliding window filling"
    return None


def _window_fills(cache, q_len):
    """Whether this step fills a sliding window (count + q_len reaches it). Its key is the filling one, but a
    capture records after the warm-up advanced the counters, i.e. the rolling branch: never capture it."""
    for layer in cache.layers:
        if type(layer) in _SLIDING_LAYERS:
            count = _host_counter(layer)
            if count is not None and count < layer.max_cache_len <= count + q_len:
                return True
    return False


def _window_state(cache):
    """Per sliding layer: whether its window is full (it rolls) or still filling (writes at the counter)."""
    return tuple(
        _host_counter(layer) >= layer.max_cache_len
        for layer in cache.layers
        if type(layer) in _SLIDING_LAYERS
    )


# transformers 5.x rolls a full static sliding window by one slot per one-token step and indexes the last slot
# with `torch.tensor([-1], device=...)`: a host-to-device copy, which a CUDA graph capture refuses ("Cannot copy
# between CPU and CUDA tensors during CUDA graph capture"), so every step past the window stayed eager. While a
# step is warmed up and recorded, that branch runs with the same ops on an index tensor made outside the
# capture (identical rows; the first replay is still compared bitwise with the eager step).
_ROLL_INDEX = {}


def _roll_index(device):
    index = _ROLL_INDEX.get(device)
    if index is None:
        if _BACKEND.capturing():
            return None
        with G._no_inference_tensors():
            index = _ROLL_INDEX[device] = torch.tensor([-1], dtype = int, device = device)
    return index


def _rolling_update_classes():
    """(class, host counter) of the static sliding layers (5.x StaticSlidingWindowLayer, 4.57 SlidingWindowLayer)
    whose one-token full-window update builds its roll index on the host."""
    try:
        import inspect
        from transformers import cache_utils
    except Exception:
        return ()
    out = []
    for cls in _SLIDING_LAYERS:
        try:
            source = inspect.getsource(cls.__dict__["update"])
        except Exception:
            continue
        counter = next(
            (c for c in ("cumulative_length_int", "cumulative_length") if f"self.{c} +=" in source),
            None,
        )
        needed = ("roll(-1, dims=-2)", "torch.tensor([-1], dtype=int", "self.keys.copy_(new_keys)")
        if counter is not None and all(n in source for n in needed):
            out.append((cls, counter))
    return tuple(out)


_ROLLING_CLASSES = []


def _graphable_update(original, counter):
    def update(self, key_states, value_states, *args, **kwargs):
        count = self.__dict__.get(counter)
        if (
            getattr(self, "is_initialized", False)
            and key_states.shape[-2] == 1
            and type(count) is int
            and count >= self.max_cache_len
        ):
            index = _roll_index(self.keys.device)
            if index is not None:
                # transformers' one-token full-window branch, op for op.
                self.__dict__[counter] = count + 1
                new_keys = self.keys.roll(-1, dims = -2)
                new_values = self.values.roll(-1, dims = -2)
                new_keys[:, :, index] = key_states
                new_values[:, :, index] = value_states
                self.keys.copy_(new_keys)
                self.values.copy_(new_values)
                return self.keys, self.values
        return original(self, key_states, value_states, *args, **kwargs)

    update._unsloth_graphable_roll = True
    return update


@contextlib.contextmanager
def _graphable_rolls():
    if not _ROLLING_CLASSES:
        _ROLLING_CLASSES.append(_rolling_update_classes())
    patched = []
    try:
        for cls, counter in _ROLLING_CLASSES[0]:
            original = cls.__dict__.get("update")
            if original is None or getattr(original, "_unsloth_graphable_roll", False):
                continue
            cls.update = _graphable_update(original, counter)
            patched.append((cls, original))
        yield
    finally:
        for cls, original in reversed(patched):
            cls.update = original


def _host_state(cache):
    """Non-tensor scalar attributes of the cache and its layers."""
    out = []
    for owner in (cache, *cache.layers):
        out.append({k: v for k, v in owner.__dict__.items() if v is None or type(v) in _CONSTANTS})
    return out


def _host_state_reason(before, after, q_len):
    for b, a in zip(before, after):
        for name in set(b) | set(a):
            if b.get(name) == a.get(name):
                continue
            if name in _COUNTERS and type(b.get(name)) is int and a.get(name) == b[name] + q_len:
                continue
            return f"cache host state {name}"
    return None


def _snapshot(cache):
    """Length counters (host ints, device scalars) a step advances, plus the rows of every full sliding window
    (a step rolls them, so unlike a slot write it is not idempotent); restored by _restore."""
    host, device = [], []
    for layer in cache.layers:
        for name in _COUNTERS:
            value = layer.__dict__.get(name)
            if type(value) is int:
                host.append((layer, name, value))
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                device.append((value, value.clone()))
        if type(layer) in _SLIDING_LAYERS:
            count = _host_counter(layer)
            if count is not None and count >= layer.max_cache_len:
                for rows in (getattr(layer, "keys", None), getattr(layer, "values", None)):
                    if isinstance(rows, torch.Tensor):
                        device.append((rows, rows.clone()))
    return host, device


def _restore(snapshot):
    host, device = snapshot
    for layer, name, value in host:
        layer.__dict__[name] = value
    for live, saved in device:
        live.copy_(saved)


def _advance(cache, q_len):
    # The replay advanced the device counters; the host ones are advanced here, as the eager update.
    for layer in cache.layers:
        for name in _COUNTERS:
            value = layer.__dict__.get(name)
            if type(value) is int:
                layer.__dict__[name] = value + q_len


def _spec(value, leaves, cache):
    """Hashable structure of a forward argument; its tensors appended to `leaves` in order."""
    if isinstance(value, torch.Tensor):
        if value.requires_grad:
            raise _Refuse("tensor argument requires grad")
        if value.device.type != _BACKEND.device_type:
            raise _Refuse(f"tensor argument on {value.device}")
        if value.numel() * value.element_size() > _MAX_ARG_BYTES:
            raise _Refuse("large tensor argument")
        leaves.append(value)
        # Inference-ness: compiled regions (create_causal_mask) guard on the dispatch key set.
        return (
            "T",
            tuple(value.shape),
            value.stride(),
            value.dtype,
            value.device,
            value.is_inference(),
        )
    if value is cache:
        return ("C",)
    if value is None or type(value) in _CONSTANTS:
        return ("V", value)
    if type(value) in (tuple, list):
        return (type(value).__name__, tuple(_spec(v, leaves, cache) for v in value))
    if type(value) is dict and all(type(k) is str for k in value):
        return ("D", tuple((k, _spec(v, leaves, cache)) for k, v in value.items()))
    if _BlockMask is not None and isinstance(value, _BlockMask):
        return _block_mask_spec(value, leaves, cache)
    raise _Refuse(f"argument {type(value).__name__}")


# BlockMask graph contract. A flex BlockMask argument replays only when its builder publishes
#     block_mask._unsloth_graph_closure = (tensors, rebuild)
# `tensors`: a tuple of every tensor its mask_mod reads (offsets, padding rows, ...), fixed shapes from
# step to step; `rebuild(tensors)`: a mask_mod of the same code reading the given tensors instead. The
# graph keeps static copies of the BlockMask's tensor fields and of `tensors`, refills them each step and
# captures flex over a BlockMask built from them (mask_mod = rebuild(static copies)). Key: BLOCK_SIZE,
# seq_lengths, the field layout, mask_mod's code and the closure tensors' layout. Any other BlockMask
# (a mask_mod closing over tensors the graph cannot see) is refused: "argument BlockMask".
_BLOCK_MASK_SKIP = ("mask_mod", "_unsloth_graph_closure", "_unsloth_on_device")


def _block_mask_fields(value):
    return sorted((k, v) for k, v in vars(value).items() if k not in _BLOCK_MASK_SKIP)


def _block_mask_spec(value, leaves, cache):
    contract = value.__dict__.get("_unsloth_graph_closure")
    if (
        type(contract) is not tuple
        or len(contract) != 2
        or type(contract[0]) is not tuple
        or not all(isinstance(t, torch.Tensor) for t in contract[0])
        or not callable(contract[1])
    ):
        raise _Refuse("argument BlockMask")
    fields = []
    for name, field in _block_mask_fields(value):
        if field is None or isinstance(field, torch.Tensor) or type(field) in _CONSTANTS:
            fields.append((name, _spec(field, leaves, cache)))
        elif type(field) is tuple and all(type(x) in _CONSTANTS for x in field):
            fields.append((name, ("V", field)))
        else:
            raise _Refuse("argument BlockMask")
    code = getattr(value.mask_mod, "__code__", None)
    if code is None:
        raise _Refuse("argument BlockMask")
    closure = tuple(_spec(t, leaves, cache) for t in contract[0])
    return ("BM", tuple(fields), code, closure)


def _build(value, buffers, cache):
    """`value` with its tensors replaced by the next static buffers (same order as _spec)."""
    if isinstance(value, torch.Tensor):
        return next(buffers)
    if type(value) in (tuple, list):
        return type(value)(_build(v, buffers, cache) for v in value)
    if type(value) is dict:
        return {k: _build(v, buffers, cache) for k, v in value.items()}
    if _BlockMask is not None and isinstance(value, _BlockMask):
        built = copy.copy(value)
        for name, field in _block_mask_fields(value):
            if isinstance(field, torch.Tensor):
                setattr(built, name, next(buffers))
        tensors, rebuild = value.__dict__["_unsloth_graph_closure"]
        static = tuple(next(buffers) for _ in tensors)
        built.mask_mod = rebuild(static)
        built.__dict__.pop("_unsloth_on_device", None)
        built._unsloth_graph_closure = (static, rebuild)
        return built
    return value


def _map_outputs(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, collections.OrderedDict) and hasattr(value, "to_tuple"):
        return type(value)(**{k: _map_outputs(v, fn) for k, v in value.items()})
    if type(value) in (tuple, list):
        return type(value)(_map_outputs(v, fn) for v in value)
    if type(value) is dict:
        return {k: _map_outputs(v, fn) for k, v in value.items()}
    return value


def _tensors(value, out):
    _map_outputs(value, lambda t: out.append(t) or t)
    return out


def _outputs_equal(a, b):
    xs, ys = _tensors(a, []), _tensors(b, [])
    if len(xs) != len(ys):
        return False
    for x, y in zip(xs, ys):
        if x.shape != y.shape or x.dtype != y.dtype:
            return False
        if not torch.equal(x, y):
            same_nan = torch.isnan(x) & torch.isnan(y) if x.is_floating_point() else None
            if same_nan is None or not torch.equal(
                torch.where(same_nan, 0, x), torch.where(same_nan, 0, y)
            ):
                return False
    return True


def _multidevice_enabled():
    return os.environ.get("UNSLOTH_DECODE_MULTIDEVICE_GRAPHS", "1") != "0"


def _hooks_align_only(model):
    """Every accelerate hook in `model` only moves tensors to its block's card (AlignDevicesHook, no
    offload, no weights map): a model split over cards by device_map / Unsloth's layer split. Offload,
    other hook types or an unreadable hook keep the model eager."""
    hooked = [m.__dict__["_hf_hook"] for m in model.modules() if "_hf_hook" in m.__dict__]
    if not hooked:
        return True
    try:
        from accelerate.hooks import AlignDevicesHook, SequentialHook
    except Exception:
        return False
    for hook in hooked:
        hooks = list(hook.hooks) if isinstance(hook, SequentialHook) else [hook]
        for h in hooks:
            if type(h) is not AlignDevicesHook:
                return False
            if h.offload or getattr(h, "offload_buffers", False) or h.weights_map is not None:
                return False
            target = h.execution_device
            if target is not None:
                try:
                    target = torch.device(target)
                except Exception:
                    return False
                if target.type != _BACKEND.device_type:
                    return False
    return True


def _param_devices(model):
    # An input embedding offloaded to the host is looked up outside the graph: its device does not count.
    host = _offloaded_input_embedding(model)
    exempt = {id(p) for p in host.parameters()} if host is not None else ()
    return {p.device for p in model.parameters() if id(p) not in exempt}


def _device_index(device):
    return device.index if device.index is not None else _BACKEND.current_device()


def _multidevice_reason(model, devices):
    """None when a model split over several cards may be graphed as one multi-card graph."""
    if not _multidevice_enabled():
        return "multi-device (UNSLOTH_DECODE_MULTIDEVICE_GRAPHS=0)"
    if not _BACKEND.multidevice_supported():
        return "multi-device capture needs a newer torch"
    if not _hooks_align_only(model):
        return "multi-device with offload or non-align hooks"
    return None


def _fingerprint(model, state = None):
    """Parameters (address, dtype, shape, in-place version), quant state and adapter state; None
    when the model is not on CUDA cards only, or is split over cards in a way this path does not
    graph (offload, other hooks). `state` gets the other cards of a split model and the refusal."""
    from ..kernels import utils as K

    items = []
    host = _offloaded_input_embedding(model)
    # Its weights stay fingerprinted, by identity too (a swapped table re-captures); only their device is exempt.
    exempt = {id(p) for p in host.parameters()} if host is not None else ()
    for p in model.parameters():
        if id(p) in exempt:
            items.append(id(p))
        items.append((p.data_ptr(), p.dtype, tuple(p.shape), G._version(p)))
    devices = _param_devices(model)
    if not devices or any(d.type != _BACKEND.device_type for d in devices):
        if state is not None:
            state.fp_reason = "not one CUDA device / adapters"
        return None
    others = ()
    if len(devices) > 1:
        reason = _multidevice_reason(model, devices)
        if reason is not None:
            if state is not None:
                state.fp_reason = reason
            return None
        here = _BACKEND.current_device()
        others = tuple(sorted({_device_index(d) for d in devices} - {here}))
    if state is not None:
        state.others = others
        state.fp_reason = None
    items.append(("devices", tuple(sorted(str(d) for d in devices))))
    for b in model.buffers():
        items.append((b.data_ptr(), b.dtype, tuple(b.shape)))
    for module in model.modules():
        weight = getattr(module, "weight", None)
        quant_state = getattr(weight, "quant_state", None)
        if quant_state is not None:
            absmax = getattr(quant_state, "absmax", None)
            items.append(
                (id(quant_state), absmax.data_ptr() if isinstance(absmax, torch.Tensor) else None)
            )
    adapters = G._adapter_state(model)
    if adapters is None:
        if state is not None:
            state.fp_reason = "not one CUDA device / adapters"
        return None
    items.extend(adapters)
    items.append(K._OPTIMIZER_STEPS)
    items.append(G.WEIGHTS_EPOCH)
    return tuple(items)


# Forward derives positions from `cache_position[0] == 0` (host sync) without position_ids (4.x Qwen3-VL).
_MROPE_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe")

# Per-token type ids a trainer passes to generate() (TRL's GRPO forwards the processor's mm_token_type_ids /
# token_type_ids) can reach every decode step at full length, one column longer per step, so every step had a
# new graph key and nothing replayed. The multimodal forwards that take them (Gemma 3 / 4) read them only to
# build the attention masks, which they skip when the step's masks come prebuilt (a dict per layer type, as
# prepare_inputs_for_generation builds over a StaticCache); transformers itself slices them to the step's
# tokens. With prebuilt masks the graphed step gets the last q_len columns.
_TYPE_ID_ARGS = ("token_type_ids", "mm_token_type_ids")


def _step_type_ids(kwargs, q_len):
    if not isinstance(kwargs.get("attention_mask"), dict):
        return kwargs
    out = None
    for name in _TYPE_ID_ARGS:
        value = kwargs.get(name)
        if isinstance(value, torch.Tensor) and value.dim() >= 2 and value.shape[-1] > q_len:
            if out is None:
                out = dict(kwargs)
            # As transformers slices them: a contiguous copy (a view's row stride would still grow every step).
            out[name] = value[..., -q_len:].clone(memory_format = torch.contiguous_format)
    return kwargs if out is None else out


def _mrope_positions(model, kwargs):
    """kwargs with the decode position_ids the model's forward would derive, computed outside the
    graph (same ops, so the same values); unchanged when they do not apply."""
    if kwargs.get("position_ids") is not None:
        return kwargs
    if getattr(getattr(model, "config", None), "model_type", None) not in _MROPE_MODEL_TYPES:
        return kwargs
    inner = getattr(model, "model", None)
    deltas = getattr(inner, "rope_deltas", None)
    cache_position = kwargs.get("cache_position")
    ids = kwargs.get("input_ids")
    if ids is None:
        ids = kwargs.get("inputs_embeds")
    if not isinstance(deltas, torch.Tensor) or not isinstance(cache_position, torch.Tensor):
        return kwargs
    batch_size, seq_length = ids.shape[0], ids.shape[1]
    if deltas.shape[0] == 0 or batch_size % deltas.shape[0] != 0:
        return kwargs
    delta = (cache_position[0] + deltas).to(ids.device)
    position_ids = torch.arange(seq_length, device = ids.device)
    position_ids = position_ids.view(1, -1).expand(batch_size, -1)
    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim = 0)
    position_ids = position_ids.add(delta)
    position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
    return {**kwargs, "position_ids": position_ids}


# Multimodal decoders whose forward zeroes placeholder ids with a boolean index (`ids[mask] = 0`, a host sync a
# capture cannot hold): the decode step's embeddings are looked up outside the graph instead, with the same ops on
# the same ids, so the graph and the eager step see identical inputs_embeds. Model type -> placeholder token ids.
_PLACEHOLDER_EMBED_MODEL_TYPES = {
    "muse_glimmer": ("image_token_id", "video_token_id"),
    "muse_spark": ("image_token_id", "video_token_id", "audio_token_id"),
    "kimi_k25": ("image_token_id", "video_token_id"),
}


def _placeholder_embeds(model, kwargs):
    """kwargs with input_ids replaced by the inputs_embeds the model's forward would compute from them (a
    one-token text decode step: no media); unchanged when this does not apply."""
    config = getattr(model, "config", None)
    names = _PLACEHOLDER_EMBED_MODEL_TYPES.get(getattr(config, "model_type", None))
    ids = kwargs.get("input_ids")
    if (
        names is None
        or not isinstance(ids, torch.Tensor)
        or kwargs.get("inputs_embeds") is not None
    ):
        return kwargs
    if any(
        kwargs.get(k) is not None
        for k in kwargs
        if k.startswith(("pixel_values", "mm_encoder", "input_features"))
    ):
        return kwargs
    tokens = [getattr(config, name, None) for name in names]
    embed = getattr(model, "get_input_embeddings", None)
    if any(type(t) is not int for t in tokens) or embed is None:
        return kwargs
    mask = ids == tokens[0]
    for token in tokens[1:]:
        mask = mask | (ids == token)
    # As the forward's own lookup runs (no autograd under generate), on the decoder's card: an offloaded or
    # hooked embedding may hand back a host tensor or one that tracks its trainable weight.
    with torch.no_grad():
        embeds = embed()(ids.masked_fill(mask, 0))
    embeds = embeds.detach()
    if embeds.device != ids.device:
        embeds = embeds.to(ids.device)
    return {**kwargs, "input_ids": None, "inputs_embeds": embeds}


def _offloaded_input_embedding(model):
    """The input embedding Unsloth's `offload_embedding` keeps in host RAM (its hooks move the ids there and the
    looked-up rows back to the decoder's card), or None. Not when the output head shares a weight with it."""
    get = getattr(model, "get_input_embeddings", None)
    try:
        module = get() if get is not None else None
    except Exception:
        return None
    if not isinstance(module, torch.nn.Module) or not getattr(
        module, "_unsloth_offload_hooks_installed", False
    ):
        return None
    params = list(module.parameters())
    if not params or all(p.device.type == _BACKEND.device_type for p in params):
        return None
    get_out = getattr(model, "get_output_embeddings", None)
    try:
        out = get_out() if get_out is not None else None
    except Exception:
        return None
    if isinstance(out, torch.nn.Module):
        shared = {id(p) for p in params}
        if any(id(p) in shared for p in out.parameters()):
            return None
    return module


def _host_embeds(model, kwargs):
    """kwargs with input_ids replaced by the offloaded input embedding's rows, looked up outside the graph (a host
    lookup and copy cannot be captured) exactly as the forward would look them up; unchanged otherwise."""
    ids = kwargs.get("input_ids")
    if not isinstance(ids, torch.Tensor) or kwargs.get("inputs_embeds") is not None:
        return kwargs
    module = _offloaded_input_embedding(model)
    if module is None:
        return kwargs
    with torch.no_grad():
        embeds = module(ids)
    embeds = embeds.detach()
    if embeds.device != ids.device:
        embeds = embeds.to(ids.device)
    return {**kwargs, "input_ids": None, "inputs_embeds": embeds}


def decode_step(
    model,
    forward,
    kwargs,
    eager_regions = None,
):
    """The outputs of one decode step run through a CUDA graph (captured on first use), or None for
    the caller's eager step. `forward` is the model's real forward; `eager_regions` an optional
    context manager (zoo `unsloth_eager_decode`) every graph-path step runs inside."""
    if torch.compiler.is_compiling() or torch.is_grad_enabled():
        return None
    if _BACKEND.capturing():
        return None
    state = _state(model)
    state.forwards += 1
    if state.forwards == 1:
        # The first forward is the prefill, even for a one-token prompt (Qwen3-VL sets rope_deltas there).
        return None
    ids = kwargs.get("input_ids")
    if ids is None:
        ids = kwargs.get("inputs_embeds")
    if (
        not isinstance(ids, torch.Tensor)
        or ids.dim() < 2
        or ids.shape[1] != 1
        or ids.device.type != _BACKEND.device_type
    ):
        return None
    state.step += 1
    if ids.shape[0] > MAX_BSZ:
        return _refusal(state, "batch over max")
    cache = kwargs.get("past_key_values")
    reason = _cache_reason(cache)
    if reason is None:
        reason = _window_reason(cache, 1)
    if reason is not None:
        return _refusal(state, reason)
    device = getattr(model, "device", None)
    if (
        isinstance(device, torch.device)
        and device.type == ids.device.type
        and device.index is not None
        and ids.device.index is not None
        and device.index != ids.device.index
    ):
        # Inputs on another card than the weights (an accelerate hook moves them every step): not one device.
        return _refusal(state, "inputs on another device")
    index = ids.device.index
    if index is not None and index != _BACKEND.current_device():
        with _BACKEND.device(index):
            return decode_step(model, forward, kwargs, eager_regions)

    from ..kernels import utils as K

    if state.recheck or state.optimizer_steps != K._OPTIMIZER_STEPS:
        fingerprint = _fingerprint(model, state)
        if fingerprint is None:
            # Ineligible until the next generate() call re-checks.
            state.invalidate()
            state.recheck = False
            state.optimizer_steps = K._OPTIMIZER_STEPS
            return _refusal(state, state.fp_reason or "not one CUDA device / adapters")
        if fingerprint != state.fingerprint:
            state.invalidate()
            state.fingerprint = fingerprint
        state.recheck = False
        state.optimizer_steps = K._OPTIMIZER_STEPS
    elif state.fingerprint is None:
        return _refusal(state, state.fp_reason or "not one CUDA device / adapters")

    kwargs = _host_embeds(
        model,
        _placeholder_embeds(model, _mrope_positions(model, _step_type_ids(kwargs, ids.shape[1]))),
    )
    leaves = []
    spec = []
    for name, value in kwargs.items():
        try:
            spec.append((name, _spec(value, leaves, cache)))
        except _Refuse as error:
            # The argument's name makes a refusal in a GPU log self-explanatory.
            return _refusal(state, f"{error} ({name})")
    spec = tuple(spec)
    if not torch.is_inference_mode_enabled() and any(leaf.is_inference() for leaf in leaves):
        # Their static buffers could not be refilled outside inference mode.
        return _refusal(state, "inference tensor outside inference mode")
    layout = tuple(
        (type(layer).__name__, layer.max_cache_len, tuple(layer.keys.shape), layer.keys.dtype)
        for layer in cache.layers
    )
    # Ambient autocast incl. cache flag (Dynamo guards on it); cached casts outlive end() in generate().
    autocast = (G._autocast_state(), torch.is_autocast_cache_enabled())
    key = (
        spec,
        layout,
        _window_state(cache),
        autocast,
        torch.is_inference_mode_enabled(),
        eager_regions is not None,
    )
    refused = state.refused.get(key)
    if refused is not None:
        return _refusal(state, refused)
    # Graphs address this cache's tensors: a new cache re-captures; one reordered in place stays eager.
    cache_sig = (id(cache),) + tuple(
        (layer.keys.data_ptr(), layer.values.data_ptr()) for layer in cache.layers
    )
    if cache_sig != state.cache_sig:
        moved = state.cache_sig is not None and state.cache_sig[0] == id(cache)
        state.clear()
        if moved or state.moved == id(cache):
            state.moved = id(cache)
            return _refusal(state, "cache moved")
        state.cache_sig = cache_sig
    regions = eager_regions() if eager_regions is not None else contextlib.nullcontext()
    with regions:
        entry = state.graphs.get(key)
        if entry is None:
            if state.call_stop is None and state.unreplayed >= _MAX_UNREPLAYED_CAPTURES:
                state.call_stop = "graph key changes every step"
                # Its graphs were never replayed: free them (and their pool) for the eager rest of the call.
                sig = state.cache_sig
                state.clear()
                state.cache_sig = sig
            if state.call_stop is not None:
                return _refusal(state, state.call_stop)
            if _window_fills(cache, 1):
                # A filling-key graph already captured replays this step correctly; a new one would not.
                return _refusal(state, "sliding window fills in this step")
            return _capture(state, key, forward, kwargs, leaves, cache)
        for buffer, leaf in zip(entry.leaves, leaves):
            buffer.copy_(_compact(leaf))
        if not entry.validated or _validate_all():
            return _validate(state, key, entry, forward, kwargs, cache)
        _replay(state, entry)
        _advance(cache, 1)
        state.replays += 1
        state.unreplayed = 0
        return _map_outputs(entry.outputs, torch.Tensor.clone)


def _replay(state, entry):
    """Replays from this card's current stream. A split model's segments run on their own cards'
    current streams: every card's stream waits on the others before (the eager work of the step
    before) and after (outputs and cache rows are read on any of them)."""
    if not state.others:
        entry.graph.replay()
        return
    streams = [_BACKEND.current_stream()] + [_BACKEND.device_stream(d) for d in state.others]
    for a in streams:
        for b in streams:
            if a is not b:
                a.wait_stream(b)
    entry.graph.replay()
    for a in streams:
        for b in streams:
            if a is not b:
                a.wait_stream(b)


def _compact(leaf):
    """`leaf` without its broadcast (stride 0) dimensions: one row of each."""
    if 0 not in leaf.stride():
        return leaf
    return leaf[tuple(slice(0, 1) if st == 0 else slice(None) for st in leaf.stride())]


def _empty_strided_like(leaf):
    # empty_like makes a non-dense view (a sliced mask) dense: keep eager's strides exactly.
    return torch.empty_strided(leaf.shape, leaf.stride(), dtype = leaf.dtype, device = leaf.device)


def _buffer_like(leaf):
    if leaf.is_inference():
        with torch.inference_mode():
            return _empty_strided_like(leaf)
    with G._no_inference_tensors():
        return _empty_strided_like(leaf)


def _capture(state, key, forward, kwargs, leaves, cache):
    """Runs this step eagerly (warm-up, its outputs are returned), then captures it."""
    if len(state.graphs) >= G.MAX_GRAPHS:
        sig = state.cache_sig
        state.clear()
        state.cache_sig = sig
    try:
        if state.stream is not None and state.stream.device.index != _BACKEND.current_device():
            state.stream = None
            state.pool = None
        if state.pool is None:
            state.pool = _BACKEND.pool()
            state.mempools = {}
        for d in state.others:
            if d not in state.mempools:
                state.mempools[d] = _BACKEND.mempool()
            if d not in state.side:
                state.side[d] = _BACKEND.side_stream(d)
        side = {d: state.side[d] for d in state.others}
        if state.stream is None:
            state.stream = _BACKEND.new_stream()
        stream = state.stream
        # Static buffers match their argument (dtype, strides, inference-ness): same compiled variants as eager.
        buffers = [_buffer_like(_compact(leaf)) for leaf in leaves]
        for buffer, leaf in zip(buffers, leaves):
            buffer.copy_(_compact(leaf))
        it = iter([buffer.expand(leaf.shape) for buffer, leaf in zip(buffers, leaves)])
        static_kwargs = {k: _build(v, it, cache) for k, v in kwargs.items()}
    except Exception as error:
        reason = f"capture setup failed: {type(error).__name__}"
        state.pool = None
        state.refused[key] = reason
        state.refusals[reason] += 1
        return forward(**kwargs)
    before = _host_state(cache)
    stream.wait_stream(_BACKEND.current_stream())
    graph = None
    try:
        # A split model's other cards run on their side streams in the warm-up too (their cuBLAS
        # workspaces exist before the capture).
        with (
            _BACKEND.stream(stream),
            _BACKEND.on_streams(side) if side else contextlib.nullcontext(),
            _graphable_rolls(),
        ):
            # Warm-up: JIT, cuBLAS workspaces and lazy caches built outside the capture.
            outputs = forward(**static_kwargs)
            stream.synchronize()
            reason = _host_state_reason(before, _host_state(cache), 1)
            if reason is not None:
                state.refused[key] = reason
                state.refusals[reason] += 1
                return outputs
            after = _snapshot(cache)
            graph = False
            try:
                if side:
                    graph, static_outputs = _BACKEND.record(
                        state.pool, lambda: forward(**static_kwargs), cache, side, state.mempools
                    )
                else:
                    graph, static_outputs = _BACKEND.record(
                        state.pool, lambda: forward(**static_kwargs), cache
                    )
            finally:
                # Capturing ran the Python update again: put the host counters back.
                _restore((after[0], []))
    except Exception as error:
        if graph is None:
            # The eager step itself failed: as without graphs.
            raise
        if isinstance(error, torch.cuda.OutOfMemoryError):
            # Out of memory while recording: free every graph of this call and decode the rest of it eagerly.
            # Not refused for the process (the next call may have the room).
            reason = "capture failed: out of memory"
            if len(state.capture_errors) < 4:
                state.capture_errors.append((reason, _where(error)))
            sig = state.cache_sig
            state.clear()
            state.cache_sig = sig
            state.call_stop = reason
            state.refusals[reason] += 1
            return outputs
        message = " ".join(str(error).split())[:200]
        reason = f"capture failed: {type(error).__name__}: {message}"
        if len(state.capture_errors) < 4:
            state.capture_errors.append((reason, _where(error)))
        state.pool = None
        state.refused[key] = reason
        state.refusals[reason] += 1
        if os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS_STRICT", "0") == "1":
            raise
        return outputs
    finally:
        # Also after a failed capture: the eager steps must not race the side stream.
        _BACKEND.current_stream().wait_stream(stream)
    state.graphs[key] = _Entry(graph, buffers, static_outputs, key in state.validated)
    state.captures += 1
    state.unreplayed += 1
    state.key_captures[hash(key)] += 1
    return outputs


def _validate(state, key, entry, forward, kwargs, cache):
    """First replay of a key must match the eager step bitwise, else the key is refused. Eager runs
    last, so generation continues from its result and cache rows."""
    state.validations += 1
    start = _snapshot(cache)
    _replay(state, entry)
    _restore(start)
    eager = forward(**kwargs)
    equal = _outputs_equal(eager, entry.outputs)
    _record_check(state, eager, entry.outputs, equal)
    if equal or entry.validated:
        # VALIDATE_ALL only records: generation stays on the eager tokens, so steps share a prefix.
        entry.validated = True
        state.validated.add(key)
        state.replays += 1
        state.unreplayed = 0
        return eager
    reason = "validation mismatch"
    state.refused[key] = reason
    state.refusals[reason] += 1
    state.graphs.pop(key, None)
    if os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS_STRICT", "0") == "1":
        raise RuntimeError("Unsloth: decode CUDA graph replay differs from the eager step")
    return eager
