# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CUDA-graph decode for any FastModel model generating over a transformers StaticCache.

Each one-token decode step of `unsloth_base_fast_generate` replays a CUDAGraph of the model's own step,
with tensor arguments copied into static buffers. Only where `_decode_graphs.enabled()`, on NVIDIA CUDA,
over static attention layers that index the cache on the device. Per graph key: the first step runs
eagerly, then is captured; the first replay must match the eager step bitwise; a host sync, host-state
change or mismatch refuses the key. The step runs in the eager step's exact state (argument inference-ness
and strides, ambient autocast + cache flag, compiled regions as eager runs them): Dynamo guards on all of
these and another compiled variant rounds differently. Graphs are dropped at the end of generate() and on
parameter / adapter / quant changes. UNSLOTH_DECODE_GENERIC_GRAPHS=0 disables; _VALIDATE_ALL=1 checks every
replay (debug); batches over UNSLOTH_DECODE_GENERIC_GRAPH_MAX_BSZ (16) decode eagerly."""

import collections
import contextlib
import os
import traceback

import torch

from . import _decode_graphs as G

__all__ = [
    "wanted",
    "begin",
    "end",
    "decode_step",
    "release",
    "stats",
]

_ATTR = "_unsloth_generic_decode_graphs"
# Host length counters of static cache layers (4.57 sliding: cumulative_length; 5.x: cumulative_length_int).
_COUNTERS = ("cumulative_length", "cumulative_length_int")
_MAX_ARG_BYTES = 16 * 1024 * 1024
_CONSTANTS = (bool, int, float, str, torch.dtype, torch.device)
_MAX_KEYS = 4096
# Graphs only remove launch overhead (T4 1B 4-bit: 4-7x at bsz 1, 2.7-4.7x at 16).
MAX_BSZ = int(os.environ.get("UNSLOTH_DECODE_GENERIC_GRAPH_MAX_BSZ", "16"))


class _Refuse(Exception):
    pass


class _Cuda:
    """The CUDA calls this path makes (tests swap in a host stand-in)."""

    device_type = "cuda"

    @staticmethod
    def current_device():
        return torch.cuda.current_device()

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
    def record(pool, fn, cache):
        """(graph, fn()) with fn's launches captured into graph (`cache`: for host stand-ins)."""
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
    # NVIDIA CUDA only: HIP replay is not verified against the eager step.
    if torch.version.hip is not None or not torch.cuda.is_available():
        return False
    if not _FULL_LAYERS:
        return False
    device = getattr(model, "device", None)
    return getattr(device, "type", None) == "cuda" and _plain_forward(model)


def _plain_forward(model):
    """No instance forward; the class forward bound on the instance, bare or under functools.wraps
    wrappers (accelerate's mixed-precision autocast / fp32-output wrappers, which eager decode runs too);
    or Unsloth's GRPO hidden-states wrapper, which passes straight through while generating. Accelerate
    device hooks (offload, device maps) stay eager."""
    forward = model.__dict__.get("forward")
    if forward is None:
        return True
    func = getattr(forward, "__func__", forward)
    if getattr(func, "_unsloth_grpo_hidden_states_forward_wrapped", False):
        return True
    if getattr(forward, "__self__", None) is not model or hasattr(model, "_hf_hook"):
        return False
    target = getattr(type(model), "forward", None)
    for _ in range(4):
        if func is target:
            return True
        func = getattr(func, "__wrapped__", None)
        if func is None:
            return False
    return False


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

    def clear(self):
        self.graphs.clear()
        self.cache_sig = None
        # A pool whose graphs are all freed is released by the allocator: never reuse its handle.
        self.pool = None

    def invalidate(self):
        """Weights / adapters / quant state changed: graphs and every per-key decision go."""
        self.clear()
        self.fingerprint = None
        self.validated.clear()
        self.refused.clear()


def _state(model):
    state = model.__dict__.get(_ATTR)
    if state is None:
        state = model.__dict__[_ATTR] = _State()
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
    if len(state.refused) + len(state.validated) > _MAX_KEYS:
        state.refused.clear()
        state.validated.clear()


def end(model):
    """End of a generate() call: its graphs read this call's cache, so free them."""
    state = model.__dict__.get(_ATTR)
    if state is not None:
        state.clear()


def release(model):
    for module in model.modules():
        state = module.__dict__.get(_ATTR)
        if state is not None:
            state.invalidate()


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
    # A full sliding window rolls, which a replay cannot repeat.
    for layer in cache.layers:
        if type(layer) in _SLIDING_LAYERS:
            count = _host_counter(layer)
            if count is None:
                return "sliding layer without host counter"
            if count + q_len > layer.max_cache_len:
                return "sliding window full"
    return None


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
    """Length counters (host ints, device scalars) a step advances; restored by _restore."""
    host, device = [], []
    for layer in cache.layers:
        for name in _COUNTERS:
            value = layer.__dict__.get(name)
            if type(value) is int:
                host.append((layer, name, value))
            elif isinstance(value, torch.Tensor) and value.numel() == 1:
                device.append((value, value.clone()))
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
        if value.requires_grad or value.device.type != _BACKEND.device_type:
            raise _Refuse("tensor argument")
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
    raise _Refuse(f"argument {type(value).__name__}")


def _build(value, buffers, cache):
    """`value` with its tensors replaced by the next static buffers (same order as _spec)."""
    if isinstance(value, torch.Tensor):
        return next(buffers)
    if type(value) in (tuple, list):
        return type(value)(_build(v, buffers, cache) for v in value)
    if type(value) is dict:
        return {k: _build(v, buffers, cache) for k, v in value.items()}
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


def _fingerprint(model):
    """Parameters (address, dtype, shape, in-place version), quant state and adapter state; None
    when the model is not on exactly one CUDA device."""
    from ..kernels import utils as K

    items = []
    devices = set()
    for p in model.parameters():
        devices.add(p.device)
        items.append((p.data_ptr(), p.dtype, tuple(p.shape), G._version(p)))
    if len(devices) != 1 or next(iter(devices)).type != _BACKEND.device_type:
        return None
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
        return None
    items.extend(adapters)
    items.append(K._OPTIMIZER_STEPS)
    return tuple(items)


# Forward derives positions from `cache_position[0] == 0` (host sync) without position_ids (4.x Qwen3-VL).
_MROPE_MODEL_TYPES = ("qwen3_vl", "qwen3_vl_moe")


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
    index = ids.device.index
    if index is not None and index != _BACKEND.current_device():
        with _BACKEND.device(index):
            return decode_step(model, forward, kwargs, eager_regions)

    from ..kernels import utils as K

    if state.recheck or state.optimizer_steps != K._OPTIMIZER_STEPS:
        fingerprint = _fingerprint(model)
        if fingerprint is None:
            # Ineligible until the next generate() call re-checks.
            state.invalidate()
            state.recheck = False
            state.optimizer_steps = K._OPTIMIZER_STEPS
            return _refusal(state, "not one CUDA device / adapters")
        if fingerprint != state.fingerprint:
            state.invalidate()
            state.fingerprint = fingerprint
        state.recheck = False
        state.optimizer_steps = K._OPTIMIZER_STEPS
    elif state.fingerprint is None:
        return _refusal(state, "not one CUDA device / adapters")

    kwargs = _mrope_positions(model, kwargs)
    leaves = []
    try:
        spec = tuple((k, _spec(v, leaves, cache)) for k, v in kwargs.items())
    except _Refuse as error:
        return _refusal(state, str(error))
    if not torch.is_inference_mode_enabled() and any(leaf.is_inference() for leaf in leaves):
        # Their static buffers could not be refilled outside inference mode.
        return _refusal(state, "inference tensor outside inference mode")
    layout = tuple(
        (type(layer).__name__, layer.max_cache_len, tuple(layer.keys.shape), layer.keys.dtype)
        for layer in cache.layers
    )
    # Ambient autocast incl. cache flag (Dynamo guards on it); cached casts outlive end() in generate().
    autocast = (G._autocast_state(), torch.is_autocast_cache_enabled())
    key = (spec, layout, autocast, torch.is_inference_mode_enabled(), eager_regions is not None)
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
            return _capture(state, key, forward, kwargs, leaves, cache)
        for buffer, leaf in zip(entry.leaves, leaves):
            buffer.copy_(_compact(leaf))
        if not entry.validated or _validate_all():
            return _validate(state, key, entry, forward, kwargs, cache)
        entry.graph.replay()
        _advance(cache, 1)
        state.replays += 1
        return _map_outputs(entry.outputs, torch.Tensor.clone)


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
        with _BACKEND.stream(stream):
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
    state.key_captures[hash(key)] += 1
    return outputs


def _validate(state, key, entry, forward, kwargs, cache):
    """First replay of a key must match the eager step bitwise, else the key is refused. Eager runs
    last, so generation continues from its result and cache rows."""
    state.validations += 1
    start = _snapshot(cache)
    entry.graph.replay()
    _restore(start)
    eager = forward(**kwargs)
    equal = _outputs_equal(eager, entry.outputs)
    _record_check(state, eager, entry.outputs, equal)
    if equal or entry.validated:
        # VALIDATE_ALL only records: generation stays on the eager tokens, so steps share a prefix.
        entry.validated = True
        state.validated.add(key)
        state.replays += 1
        return eager
    reason = "validation mismatch"
    state.refused[key] = reason
    state.refusals[reason] += 1
    state.graphs.pop(key, None)
    if os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS_STRICT", "0") == "1":
        raise RuntimeError("Unsloth: decode CUDA graph replay differs from the eager step")
    return eager
