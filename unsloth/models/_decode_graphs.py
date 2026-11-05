# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CUDA-graph decode; UNSLOTH_DECODE_CUDA_GRAPHS=1 forces, =0 disables; unset, on for every
Unsloth generate() on NVIDIA CUDA outside native Windows for the model types below.

UNSLOTH_DECODE_CUDA_GRAPHS_FULL=1 also captures the attention cores: each step is then ONE graph that
attends over a fixed span of cache rows (the next power of two >= the KV length, at least BUCKET, at
most the cache) with a device-side keep mask, so no eager launch is left between the layers. Rows past
the true length are masked to -inf, so the result matches the eager step up to the summation order of
the softmax / PV reductions (not bit for bit); one graph is captured per (batch size, span). Only
batch 1 by default (UNSLOTH_DECODE_FULL_GRAPH_MAX_BSZ): larger batches keep the segmented capture."""

import math
import os
import sys
import threading

import torch

# Programmatic override (tests, harnesses): True / False force graphs on / off, None (default) follows
# the environment variable read at call time and the RL / Studio generation scopes (enabled()).
ENABLED = None
# Whole-step graphs (attention captured too, over a power-of-two span of cache rows).
FULL = os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS_FULL", "0") == "1"
# Largest batch the whole-step capture takes; larger batches keep the segmented (bit-exact) capture.
# Batched decode measured slower whole-step on a T4 (+9-15%: GPU bound, nothing to hide).
FULL_MAX_BSZ = int(os.environ.get("UNSLOTH_DECODE_FULL_GRAPH_MAX_BSZ", "1"))
# KV-cache rows are allocated in multiples of this with graphs on.
BUCKET = 256
# T4 Qwen3-0.6B: 3-6 new tokens ran 5-58% slower graphed, 8 tokens 1.25x faster.
CAPTURE_MIN_STEPS = int(os.environ.get("UNSLOTH_DECODE_GRAPH_CAPTURE_MIN_STEPS", "6"))
# Captured steps (one per batch size / autocast state) kept per model.
MAX_GRAPHS = int(os.environ.get("UNSLOTH_DECODE_GRAPH_MAX", "8"))
# Whole-step graphs keep one graph per span too (log2(cache / BUCKET) + 1 per batch size).
MAX_FULL_GRAPHS = int(os.environ.get("UNSLOTH_DECODE_FULL_GRAPH_MAX", "16"))

__all__ = [
    "ENABLED",
    "enabled",
    "rl_generation",
    "studio_generation",
    "generate_scope",
    "eager_scope",
    "graphable",
    "can_replay",
    "short_call",
    "kept_graph",
    "plan_call",
    "record_need",
    "decode_batch",
    "decode_batch_used",
    "EAGER_UNFIT",
    "generate_with_fallback",
    "keep_only",
    "release_if_training",
    "EAGER_REDOS",
    "static",
    "alloc_decode_buffers",
    "decode_step",
    "invalidate",
    "release",
    "release_for_training",
    "register_attention",
    "start_generation",
    "end_generation",
]

# The graph-mode statics of the step this thread is warming up / capturing (static()); per thread,
# so another thread's (or model's) eager decode never reads them.
_TLS = threading.local()
# Bumped whenever a persistent decode buffer is (re)allocated or resized.
BUFFER_EPOCH = 0
# Bumped per generate() call and per prefill: the next graph step revalidates the fingerprint.
GENERATION_EPOCH = 0
# prompt + max_new_tokens of the running generate() call, 0 when unknown.
MAX_LEN_HINT = 0
# 1: size the graph KV cache for prompt + max_new_tokens up front (early stops still hold it all).
PREALLOC = os.environ.get("UNSLOTH_DECODE_GRAPH_PREALLOC", "0") == "1"
_SUPPORTED_ATTENTION = set()
# Tested families only (Qwen2 shares the Llama attention but is untested here).
_SUPPORTED_MODEL_TYPES = {"llama", "qwen3"}
_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def enabled():
    """Whether a decode step / prefill on this thread runs (and sizes its buffers) for graph replay."""
    if getattr(_TLS, "eager", 0):
        return False
    if ENABLED is not None:
        return bool(ENABLED)
    if torch.compiler.is_compiling():
        # Traced code never replays graphs (decode_step bails too); keep the trace free of env / TLS reads.
        return False
    flag = os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS", "")
    if flag == "1":
        return True
    if flag == "0":
        return False
    return _DEFAULT_OK and getattr(_TLS, "rl_depth", 0) > 0


def _default_ok():
    # The RL default is NVIDIA CUDA only: HIP graph replay is not verified bitwise against the eager
    # step on ROCm (=1 still forces it), and XPU / MPS / CPU never take the graphed step.
    try:
        return torch.version.hip is None and torch.cuda.is_available()
    except Exception:
        return False


_DEFAULT_OK = _default_ok()


class rl_generation:
    """Scope of an RL trainer's generate() call (per thread): graphs default on inside it."""

    __slots__ = ()

    def __enter__(self):
        _TLS.rl_depth = getattr(_TLS, "rl_depth", 0) + 1
        return self

    def __exit__(self, *exc):
        _TLS.rl_depth -= 1
        return False


# Native Windows CUDA capture is unverified: Studio default off there (=1 still forces).
_STUDIO_DEFAULT_OK = sys.platform != "win32"


class studio_generation(rl_generation):
    """Per-thread scope for Studio's chat generate(): graphs default on, freed on unload (release())."""

    __slots__ = ("_entered",)

    def __enter__(self):
        self._entered = _STUDIO_DEFAULT_OK
        return super().__enter__() if self._entered else self

    def __exit__(self, *exc):
        return super().__exit__(*exc) if self._entered else False


class generate_scope(studio_generation):
    """Per-thread scope of plain model.generate(): graphs default on, as in Studio."""

    __slots__ = ()


class eager_scope:
    """Per-thread: force eager decode (the OOM redo in generate())."""

    __slots__ = ()

    def __enter__(self):
        _TLS.eager = getattr(_TLS, "eager", 0) + 1
        return self

    def __exit__(self, *exc):
        _TLS.eager -= 1
        return False


# Need = CALIB_WORK x (KV + prefill work) + CALIB_GRAPH_KV x KV. Calibrated on Qwen3-4B QLoRA bs16 x 2048,
# T4: eager peaked 1.14x the estimate; graph capture OOMed at ~12 GiB free (~0.26 x KV extra).
CALIB_WORK = float(os.environ.get("UNSLOTH_DECODE_GRAPH_CALIB_WORK", "1.15"))
CALIB_GRAPH_KV = float(os.environ.get("UNSLOTH_DECODE_GRAPH_CALIB_KV", "0.30"))
# A shape's largest measured footprint x this raises (never lowers) the estimate.
MEASURED_MARGIN = 1.05
EAGER_UNFIT = 0
NEEDS = "_unsloth_decode_needs"


def _call_bytes(
    config,
    bsz,
    prompt,
    dtype,
    increment,
    max_len = 0,
):
    """(cache rows, KV bytes, prefill work bytes), sized like alloc_decode_buffers."""
    try:
        layers = int(config.num_hidden_layers)
        hidden = int(config.hidden_size)
        heads = int(config.num_attention_heads)
        kv_heads = int(getattr(config, "num_key_value_heads", None) or heads)
        head_dim = int(getattr(config, "head_dim", None) or hidden // heads)
        inter = int(getattr(config, "intermediate_size", None) or 4 * hidden)
        vocab = int(getattr(config, "vocab_size", 0) or 0)
    except (AttributeError, TypeError, ValueError):
        return None, None, None
    size = torch.empty((), dtype = dtype).element_size()
    rows = _round_up(max(prompt + increment + 1, max_len + 1), BUCKET)
    kv = layers * rows * 2 * bsz * kv_heads * head_dim * size
    tokens = bsz * prompt
    work = tokens * (4 * hidden + 3 * inter + 2 * heads * head_dim) * size + bsz * vocab * 4
    return rows, kv, work


def _allocator(device):
    """(allocated, reserved, peak allocated) from one stats read."""
    try:
        stats = torch.cuda.memory_stats_as_nested_dict(device)
        return (
            stats["allocated_bytes"]["all"]["current"],
            stats["reserved_bytes"]["all"]["current"],
            stats["allocated_bytes"]["all"]["peak"],
        )
    except (KeyError, TypeError):
        return (
            torch.cuda.memory_allocated(device),
            torch.cuda.memory_reserved(device),
            torch.cuda.max_memory_allocated(device),
        )


def _free_bytes(device):
    free, _ = torch.cuda.mem_get_info(device)
    allocated, reserved, _ = _allocator(device)
    return free + reserved - allocated


def _empty_cache():
    torch.cuda.empty_cache()


def _alloc_stats(device):
    # Peak is never reset: other code may rely on max_memory_allocated.
    allocated, _, peak = _allocator(device)
    return allocated, peak


def _shape_key(bsz, prompt, max_len):
    # Includes the prompt bucket: a longer prompt has a larger prefill.
    return (bsz, _round_up(prompt + 1, BUCKET), _round_up(max(prompt, max_len) + 1, BUCKET))


def _kept_buffers(model, key, rows):
    """(any buffers kept, bytes of a kept cache for `key` with >= rows); no buffer reference outlives release()."""
    kept, reused = False, 0
    for m in model.modules():
        store = m.__dict__.get("_unsloth_graph_buffers")
        if not store:
            continue
        kept = True
        bufs = store.get(key)
        if bufs is not None and bufs[0].shape[0] >= rows:
            reused += bufs[0].numel() * bufs[0].element_size()
    return kept, reused


def can_replay(max_len, prompt):
    """The first decode step is always eager, so <= 2 new tokens never replay; 0 (unknown) may."""
    return not max_len or max_len - prompt > 2


def kept_graph(model, bsz):
    return any(key[0] == bsz for state in _states(model) for key in state.graphs)


def short_call(model, max_len, prompt, bsz):
    """Too short to win back a capture unless a graph of its batch is kept: decode eagerly."""
    if not max_len:
        return False
    left = max_len - prompt - 1
    return left < 2 or (left < CAPTURE_MIN_STEPS and not kept_graph(model, bsz))


def graphable(model):
    """Other model types always decode eagerly, so planning them only costs."""
    return getattr(getattr(model, "config", None), "model_type", None) in _SUPPORTED_MODEL_TYPES


def plan_call(
    model,
    input_ids,
    dtype,
    increment,
    max_len = 0,
    bsz = None,
    eager = False,
):
    """("graph", token) or ("eager", None). Releases kept graphs / buffers first if the call would not
    fit beside them; still too big runs eagerly up front rather than relying on the OOM redo. `bsz` is
    the expanded decode batch (decode_batch()). eager: only makes room, uncounted."""
    global EAGER_UNFIT
    device = getattr(input_ids, "device", None)
    if (
        not graphable(model)
        or not hasattr(input_ids, "shape")
        or len(input_ids.shape) < 2
        or getattr(device, "type", None) != "cuda"
    ):
        return "graph", None
    bsz, prompt = int(bsz or input_ids.shape[0]), int(input_ids.shape[1])
    rows, kv, work = _call_bytes(model.config, bsz, prompt, dtype, increment, max_len)
    if rows is None:
        return "graph", None
    key = _shape_key(bsz, prompt, max_len)
    measured = model.__dict__.get(NEEDS, {}).get(key)
    # A measurement can only raise the estimate; it includes a reused cache, so reuse is subtracted once.
    estimate = CALIB_WORK * (kv + work) + CALIB_GRAPH_KV * kv
    full = estimate if measured is None else max(estimate, MEASURED_MARGIN * measured)
    kept, reused = _kept_buffers(model, (bsz, dtype, device), rows)
    if eager:
        # Eager decode uses fresh buffers: no graph extra, no kept-cache credit.
        full, reused = CALIB_WORK * (kv + work), 0
    # Cap credit at this call's KV: a larger kept cache cannot also cover prefill.
    reused = min(reused, kv)
    need = full - reused
    free = _free_bytes(device)
    if free < need and (kept or any(state.graphs for state in _states(model))):
        release(model)
        _empty_cache()
        free = _free_bytes(device)
        reused = 0
        need = full
    if eager:
        return "eager", None
    if free < need:
        EAGER_UNFIT += 1
        return "eager", None
    return "graph", (key, device, *_alloc_stats(device), reused)


def record_need(model, token):
    """Store the largest graphed footprint per shape (peak delta + reused kept cache). The peak is never
    reset, so a call that did not raise it is skipped."""
    if token is None:
        return
    key, device, start, peak_before, reused = token
    peak = _alloc_stats(device)[1]
    if peak > peak_before:
        needs = model.__dict__.setdefault(NEEDS, {})
        needs[key] = max(needs.get(key, 0), peak - start + reused)


def decode_batch(
    bsz,
    kwargs,
    generation_config = None,
):
    """Batch size generate() decodes at: prompts x num_beams, else x num_return_sequences; kwargs win."""
    config = kwargs.get("generation_config", None) or generation_config

    def get(name):
        value = kwargs.get(name, None)
        if value is None:
            value = getattr(config, name, None)
        try:
            return max(1, int(value or 1))
        except (TypeError, ValueError):
            return 1

    beams = get("num_beams")
    return bsz * (beams if beams > 1 else get("num_return_sequences"))


def decode_batch_used(model):
    """Batch size of the cache the last graph-mode decode ran on, or None."""
    for module in model.modules():
        d = module.__dict__
        store = d.get("_unsloth_graph_buffers")
        paged = d.get("paged_attention")
        if store and paged is not None:
            for key, bufs in store.items():
                if bufs[0] is paged:
                    return key[0]
    return None


_OOM_WARNED = False
EAGER_REDOS = 0
HELD = "_unsloth_decode_held"


def generate_with_fallback(model, generate, graphs, args, kwargs):
    """On a graph-mode OOM, free graphs and redo once eagerly; caller streamer / past_key_values /
    stopping_criteria / logits_processor may hold failed-attempt state, so those re-raise."""
    global _OOM_WARNED, EAGER_REDOS
    try:
        return generate(*args, **kwargs)
    except torch.cuda.OutOfMemoryError:
        if (
            not graphs
            or not graphable(model)
            or kwargs.get("streamer", None) is not None
            or kwargs.get("past_key_values", None) is not None
            or kwargs.get("stopping_criteria", None)
            or kwargs.get("logits_processor", None)
        ):
            raise
    if not _OOM_WARNED:
        _OOM_WARNED = True
        import warnings
        warnings.warn(
            "Unsloth: generate() ran out of GPU memory with CUDA-graph decode; freed the decode graphs "
            "and buffers and decoded this call eagerly (UNSLOTH_DECODE_CUDA_GRAPHS=0 always decodes eagerly).",
            stacklevel = 3,
        )
    release_for_training(model)
    _empty_cache()
    EAGER_REDOS += 1
    with eager_scope():
        return generate(*args, **kwargs)


def keep_only(model, bsz = None):
    """Drop decode buffers and graphs of every other batch size; this one's graphs stay valid. The batch
    size decode actually ran at wins over `bsz`."""
    used = decode_batch_used(model)
    if used is not None:
        bsz = used
    if bsz is None:
        return False
    dropped = False
    for module in model.modules():
        store = module.__dict__.get("_unsloth_graph_buffers")
        if store:
            for key in [k for k in store if k[0] != bsz]:
                bufs = store.pop(key)
                dropped = True
                if module.__dict__.get("paged_attention") is bufs[0]:
                    for name in _BUFFER_NAMES:
                        module.__dict__.pop(name, None)
        state = module.__dict__.get("_unsloth_decode_graphs")
        if state is not None:
            for key in [k for k in state.graphs if k[0] != bsz]:
                del state.graphs[key]
                dropped = True
            if not state.graphs:
                state.pool = None
    return dropped


def release_if_training(model):
    """Run from the training forward: training can resume without for_training() (e.g. sampling in a
    trainer callback), so this is the only place guaranteed to free kept graphs."""
    if model.__dict__.pop(HELD, False):
        release_for_training(model)
        return True
    return False


def static():
    """The _StepStatics of a graph-mode step running on this thread, else None (eager)."""
    if torch.compiler.is_compiling():
        return None
    return getattr(_TLS, "statics", None)


def register_attention(fn):
    """Marks a decode attention function as reading static() (device slot, eager core split)."""
    _SUPPORTED_ATTENTION.add(fn)
    return fn


def _round_up(n, m):
    return -(-n // m) * m


class _StepStatics:
    """What a graph-mode step reads instead of host state: the device cache slot, the RoPE table
    length, and (per replayed step) the true KV length + mask for the eager attention cores.
    attention(fn) is the split point: eager it runs fn, capturing it closes the current segment."""

    __slots__ = ("rope_len", "slot", "kv_len", "mask", "attention", "span", "keep", "drop")

    def __init__(
        self,
        rope_len,
        slot,
        span = None,
        keep = None,
        drop = None,
    ):
        self.rope_len = rope_len
        self.slot = slot
        self.kv_len = None
        self.mask = None
        self.attention = _run_now
        # Whole-step graphs: cache rows the captured attention reads, and their bool keep mask
        # [bsz, 1, 1, rows] (refilled outside the graph before every replay).
        self.span = span
        self.keep = keep
        self.drop = drop  # ~keep, so the graph does not invert it once per layer


def _run_now(fn):
    return fn()


def _no_inference_tensors():
    # Persistent buffers outlive the generate() inference_mode; inference tensors could not be
    # written by a later decode run under no_grad only.
    import contextlib

    if not torch.is_inference_mode_enabled():
        return contextlib.nullcontext()
    stack = contextlib.ExitStack()
    stack.enter_context(torch.inference_mode(False))
    stack.enter_context(torch.no_grad())
    return stack


def alloc_decode_buffers(
    attn, K1, V1, bsz, n_heads, n_kv_heads, head_dim, hidden_size, dtype, device, increment
):
    """The decode buffers the first decode step allocates. Eager (graphs off) is unchanged; with
    graphs on they persist per (bsz, dtype, device) and are sized for the whole generate() call."""
    global BUFFER_EPOCH, GENERATION_EPOCH
    seq_len = K1.shape[-2]
    attention_size = n_heads * head_dim
    config = getattr(attn, "config", None)
    if not enabled() or getattr(config, "model_type", None) not in _SUPPORTED_MODEL_TYPES:
        attn.paged_attention = torch.empty(
            (increment + seq_len + 1, 2, bsz, n_kv_heads, head_dim),
            dtype = dtype,
            device = device,
        )
        attn.paged_attention_K = attn.paged_attention[:, 0]
        attn.paged_attention_V = attn.paged_attention[:, 1]
        attn.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        attn.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        attn.temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
        attn.temp_KV = torch.empty((2, bsz, 1, n_kv_heads * head_dim), dtype = dtype, device = device)
        attn.RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)
        # Mistral Nemo 12b has weird dimensions.
        if attention_size != hidden_size:
            attn.temp_O = torch.empty((bsz, 1, hidden_size), dtype = dtype, device = device)
        else:
            attn.temp_O = attn.temp_QA[1][:, :, :hidden_size]
        # Row stride kept a multiple of 8 so the grouped decode bmm stays on the fast cuBLAS path.
        attn.attention = torch.empty(
            (bsz, n_heads, 1, increment + seq_len + (-seq_len) % 8),
            dtype = dtype,
            device = device,
        )
    else:
        rows = _round_up(max(increment + seq_len + 1, MAX_LEN_HINT + 1), BUCKET)
        if not PREALLOC:
            # Start at the eager size; resize_decode_buffers grows it.
            rows = min(rows, _round_up(increment + seq_len + 1, BUCKET))
        store = attn.__dict__.get("_unsloth_graph_buffers")
        if store is None:
            store = attn._unsloth_graph_buffers = {}
        key = (bsz, dtype, device)
        bufs = store.get(key)
        if bufs is None or bufs[0].shape[0] < rows:
            # Evict before allocating so old and new caches are never both held (peak memory).
            if bufs is not None or len(store) >= 2:
                store.clear()
                for name in _BUFFER_NAMES:
                    attn.__dict__.pop(name, None)
                bufs = None
                BUFFER_EPOCH += 1
            with _no_inference_tensors():
                # Zeroed: a captured step reads every row of its bucket, and a masked row still
                # multiplies its V by 0, which is NaN for uninitialised NaN / inf bit patterns.
                paged = torch.zeros(
                    (rows, 2, bsz, n_kv_heads, head_dim), dtype = dtype, device = device
                )
                temp_QA = torch.empty((2, bsz, 1, attention_size), dtype = dtype, device = device)
                temp_KV = torch.empty(
                    (2, bsz, 1, n_kv_heads * head_dim), dtype = dtype, device = device
                )
                RH_Q = torch.empty((bsz, n_heads, 1, head_dim), dtype = dtype, device = device)
                if attention_size != hidden_size:
                    temp_O = torch.empty((bsz, 1, hidden_size), dtype = dtype, device = device)
                else:
                    temp_O = temp_QA[1][:, :, :hidden_size]
                scores = torch.empty((bsz, n_heads, 1, rows), dtype = dtype, device = device)
            bufs = store[key] = (paged, temp_QA, temp_KV, RH_Q, temp_O, scores)
        elif FULL:
            # A reused buffer still holds the last generation's rows: a whole-step graph reads them
            # past this one's length, and a masked inf / NaN there still turns 0 * V into NaN.
            bufs[0][seq_len:].zero_()
        paged, temp_QA, temp_KV, RH_Q, temp_O, scores = bufs
        # A new sequence: its first graph step re-checks the fingerprint (adapters toggled or
        # merged since the last one, also in decode loops that do not go through generate()).
        GENERATION_EPOCH += 1
        attn.paged_attention = paged
        attn.paged_attention_K = paged[:, 0]
        attn.paged_attention_V = paged[:, 1]
        attn.paged_attention_K[:seq_len] = K1.permute(2, 0, 1, 3)
        attn.paged_attention_V[:seq_len] = V1.permute(2, 0, 1, 3)
        attn.temp_QA, attn.temp_KV, attn.RH_Q, attn.temp_O, attn.attention = (
            temp_QA,
            temp_KV,
            RH_Q,
            temp_O,
            scores,
        )
    attn.scalar = 1.0 / math.sqrt(head_dim)  # as the eager code always computed it
    attn.half_head_dim = head_dim // 2


def resize_decode_buffers(attn, increment, bsz, n_heads, n_kv_heads, head_dim):
    """The eager growth of the KV cache by `increment` rows (new rows zeroed with graphs on)."""
    global BUFFER_EPOCH
    rows = attn.paged_attention.shape[0]
    store = attn.__dict__.get("_unsloth_graph_buffers") or {}
    graph_backed = any(bufs[0] is attn.paged_attention for bufs in store.values())
    if graph_backed:
        # Each growth re-captures: double, capped at the hint, so O(log n) re-captures.
        grown = max(rows + increment, 2 * rows)
        if MAX_LEN_HINT + 1 > rows:
            grown = min(grown, MAX_LEN_HINT + 1)
        increment = _round_up(grown, BUCKET) - rows
    attn.paged_attention.resize_((rows + increment, 2, bsz, n_kv_heads, head_dim))
    attn.paged_attention_K = attn.paged_attention[:, 0]
    attn.paged_attention_V = attn.paged_attention[:, 1]
    attn.attention.resize_((bsz, n_heads, 1, attn.attention.shape[-1] + increment))
    if graph_backed:
        attn.paged_attention[rows:].zero_()
        BUFFER_EPOCH += 1


def start_generation(max_len):
    global GENERATION_EPOCH, MAX_LEN_HINT
    GENERATION_EPOCH += 1
    MAX_LEN_HINT = max(0, int(max_len or 0))


def worth_capturing(kv_len):
    """CAPTURE_MIN_STEPS steps left at kv_len; unknown length (0): yes."""
    if MAX_LEN_HINT <= 0:
        return True
    return MAX_LEN_HINT - kv_len + 1 >= CAPTURE_MIN_STEPS


def end_generation():
    global MAX_LEN_HINT
    MAX_LEN_HINT = 0


class _ModelGraphs:
    def __init__(self):
        self.graphs = {}
        self.statics = {}
        self.pool = None
        self.stream = None
        self.fingerprint = None
        self.generation_epoch = -1
        self.optimizer_steps = -1
        self.buffer_epoch = -1
        self.disabled = None
        self.ineligible_epoch = None
        self.rope_len = None
        self.captures = 0
        self.replays = 0
        self.eager_steps = 0
        self.short_skips = 0

    def clear(self):
        self.graphs.clear()
        self.fingerprint = None
        # A pool whose graphs are all freed is released by the allocator: never reuse its handle.
        self.pool = None


def _state(model):
    state = model.__dict__.get("_unsloth_decode_graphs")
    if state is None:
        state = model.__dict__["_unsloth_decode_graphs"] = _ModelGraphs()
    return state


_BUFFER_NAMES = (
    "paged_attention",
    "paged_attention_K",
    "paged_attention_V",
    "temp_QA",
    "temp_KV",
    "RH_Q",
    "temp_O",
    "attention",
)


def _states(model):
    # The state lives on the CausalLM the decode forward runs on, under any PEFT wrapper.
    for module in model.modules():
        state = module.__dict__.get("_unsloth_decode_graphs")
        if state is not None:
            yield state


def invalidate(model):
    """Drops captured steps (they re-capture on the next decode); buffers stay."""
    global BUFFER_EPOCH
    for state in _states(model):
        state.clear()
    BUFFER_EPOCH += 1


def release(model):
    """Drops captured steps and the persistent decode buffers (frees their memory)."""
    global BUFFER_EPOCH
    found = False
    for module in model.modules():
        d = module.__dict__
        d.pop(HELD, None)
        state = d.get("_unsloth_decode_graphs")
        if state is not None:
            found = True
            state.clear()
            state.statics.clear()
        if "_unsloth_graph_buffers" in d:
            found = True
            del d["_unsloth_graph_buffers"]
            for name in _BUFFER_NAMES:
                d.pop(name, None)
    if found:
        BUFFER_EPOCH += 1


def release_for_training(model):
    """release(), plus the eager decode buffers the last generate() left on the attention modules:
    the next prefill allocates fresh ones, so training gets the KV cache memory back."""
    release(model)
    for module in model.modules():
        d = module.__dict__
        if "paged_attention" in d:
            for name in _BUFFER_NAMES:
                d.pop(name, None)


def _version(t):
    if t is None or not isinstance(t, torch.Tensor):
        return None
    if t.is_inference():
        return (t.data_ptr(), None)
    return (t.data_ptr(), t._version)


def _fingerprint(model, layers, rope_len):
    """Everything a captured step baked in; None when the model cannot be graphed."""
    from ..kernels import utils as K

    items = []
    devices = set()
    for p in model.parameters():
        devices.add(p.device)
        items.append((p.data_ptr(), p.dtype, p.shape))
    if len(devices) != 1 or next(iter(devices)).type != "cuda":
        return None
    compute_dtype = None
    seen_rope = set()
    for layer in layers:
        attn = layer.self_attn
        for owner in (attn, getattr(layer, "mlp", None)):
            for name in _PROJ_NAMES:
                proj = getattr(owner, name, None)
                if proj is None:
                    continue
                if (
                    K._has_non_lora_adapter(proj)
                    or K._has_multiple_active_adapters(proj)
                    or K._has_active_dora_adapter(proj)
                    or K._has_active_lora_bias(proj)
                    or K._has_packed_qweight(proj)
                ):
                    return None
                W, W_quant, A, B, S, bias = K.get_lora_parameters_bias(proj)
                absmax = getattr(W_quant, "absmax", None)
                cast = getattr(A, "_fast_lora", None) if A is not None else None
                items.append(
                    (
                        W.data_ptr(),
                        absmax.data_ptr() if isinstance(absmax, torch.Tensor) else id(W_quant),
                        _version(A),
                        _version(B),
                        S,
                        bias.data_ptr() if bias is not None else None,
                        None if cast is None else (cast[1].data_ptr(), cast[2].data_ptr()),
                    )
                )
        rope = getattr(attn, "rotary_emb", None)
        if rope is not None and id(rope) not in seen_rope:
            seen_rope.add(id(rope))
            index = next(iter(devices)).index or 0
            cos, sin = rope.get_cached(rope_len, index)
            items.append((cos.data_ptr(), sin.data_ptr(), cos.shape[0]))
    # Every adapter-aware module outside the projections too (embeddings, lm_head, modules_to_save):
    # their enabled / active / merged state picks the branch a captured step baked in.
    for module in model.modules():
        if hasattr(module, "disable_adapters") or hasattr(module, "active_adapter"):
            try:
                active = (
                    module.active_adapters
                    if hasattr(module, "active_adapters")
                    else module.active_adapter
                )
                items.append(
                    (
                        id(module),
                        getattr(module, "disable_adapters", None),
                        tuple(active) if isinstance(active, (list, tuple)) else active,
                        getattr(module, "merged", None),
                    )
                )
            except Exception:
                return None
    items.append(K._OPTIMIZER_STEPS)
    return tuple(items)


def _eligible(model, layers):
    config = getattr(model, "config", None)
    if getattr(config, "model_type", None) not in _SUPPORTED_MODEL_TYPES:
        return "model type"
    if getattr(layers, "_unsloth_block_swap", None) is not None:
        return "block swap"
    return None


def _full_span(
    model,
    kv_len,
    cap,
    bsz = 1,
):
    """The cache rows a whole-step graph attends over, or None for the segmented capture."""
    if not FULL or bsz > FULL_MAX_BSZ:
        return None
    sliding_window = getattr(model.config, "sliding_window", None)
    if sliding_window is not None and cap > sliding_window:
        # The segmented path slices the window off eagerly; a span would read rows outside it.
        return None
    return min(cap, max(BUCKET, 1 << (kv_len - 1).bit_length()))


def _fill_keep(state, s, mask, bsz, kv_len, cap, device):
    """Writes this step's keep mask (true KV rows, padding applied) into the static buffer."""
    keep = s.get("keep")
    if keep is None or keep.shape[0] != bsz or keep.shape[-1] != cap:
        with _no_inference_tensors():
            keep = torch.zeros((bsz, 1, 1, cap), dtype = torch.bool, device = device)
            drop = torch.ones((bsz, 1, 1, cap), dtype = torch.bool, device = device)
        if s.get("keep") is not None:
            # Graphs captured against the old buffers would read freed memory.
            state.clear()
        s["keep"], s["drop"] = keep, drop
    drop = s["drop"]
    keep[..., kv_len:] = False
    if mask is None:
        keep[..., :kv_len] = True
    else:
        keep[..., :kv_len] = mask[..., -kv_len:]
    torch.logical_not(keep, out = drop)
    return keep


def _static_inputs(state, bsz, device, dtype, hidden_size):
    key = (bsz, device)
    s = state.statics.get(key)
    if s is None:
        with _no_inference_tensors():
            s = {
                "ids": torch.zeros((bsz, 1), dtype = torch.long, device = device),
                "pos": torch.zeros((bsz, 1), dtype = torch.long, device = device),
                "slot": torch.zeros((1,), dtype = torch.long, device = device),
            }
        state.statics[key] = s
    like = s.get("like")
    if like is None or like.dtype != dtype:
        # dtype / device stand-in for the hidden states the mask builder reads.
        s["like"] = torch.empty((bsz, 1, hidden_size), dtype = dtype, device = device)
    return s


class _Step:
    """One captured decode step: graph segments, and the eager attention cores between them."""

    __slots__ = ("segments", "cores", "logits", "statics")

    def __init__(self, statics):
        self.segments = []
        self.cores = []
        self.logits = None
        self.statics = statics

    def replay(self):
        segments, cores = self.segments, self.cores
        for i in range(len(cores)):
            segments[i].replay()
            cores[i]()
        segments[-1].replay()


def _run_step(model, fast_forward_inference, logits_fn, s, st, past_key_values):
    _TLS.statics = st
    try:
        outputs = fast_forward_inference(
            model,
            s["ids"],
            past_key_values,
            position_ids = s["pos"],
            attention_mask = None,
        )
        return logits_fn(model, outputs[0])
    finally:
        _TLS.statics = None


def decode_step(
    model,
    fast_forward_inference,
    logits_fn,
    mask_fn,
    input_ids,
    past_key_values,
    position_ids,
    attention_mask,
):
    """Replays (capturing on first use) the decode step; returns (logits, past_key_values) or None
    for the eager path."""
    if torch.is_grad_enabled() or not input_ids.is_cuda:
        return None
    if (
        getattr(fast_forward_inference, "_unsloth_graph_attention", None)
        not in _SUPPORTED_ATTENTION
    ):
        return None
    if position_ids is None or os.environ.get("UNSLOTH_RETURN_HIDDEN_STATES", "0") == "1":
        return None
    if position_ids.numel() != input_ids.shape[0] and position_ids.shape[-1] == 1:
        # Shared / broadcast positions take a different eager RoPE path.
        return None
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return None
    index = input_ids.device.index
    if index is not None and index != torch.cuda.current_device():
        # Capture on the model's GPU, else the capture fails and decode stays eager.
        with torch.cuda.device(index):
            return decode_step(
                model, fast_forward_inference, logits_fn, mask_fn, input_ids, past_key_values,
                position_ids, attention_mask,
            )  # fmt: skip
    state = _state(model)
    if state.disabled is not None:
        return None
    layers = model.model.layers
    attn0 = layers[0].self_attn
    paged = attn0.__dict__.get("paged_attention")
    bsz = input_ids.shape[0]
    store = attn0.__dict__.get("_unsloth_graph_buffers") or {}
    if (
        paged is None
        or not any(bufs[0] is paged for bufs in store.values())
        or paged.shape[2] != bsz
        or input_ids.shape[1] != 1
    ):
        # First decode step after a prefill (allocates / fills the cache), or a cache this path
        # did not size.
        state.eager_steps += 1
        return None
    seq_len = past_key_values[0][0].shape[-2]
    kv_len = seq_len + 1
    cap = paged.shape[0]
    if kv_len >= cap:
        # The eager step grows the cache (new addresses: the next step re-captures).
        state.eager_steps += 1
        return None

    from ..kernels import utils as K

    # RoPE rows the captured steps may read: the table is sized (and its address fingerprinted)
    # for this many, so a position at or past it runs eagerly (the eager step grows the table).
    rope_len = max(state.rope_len or 0, cap + 1)
    bound = getattr(position_ids, "_unsloth_position_bound", None)
    if bound is None:
        # Caller-supplied positions can exceed the KV length: one host read, as the eager step does.
        bound = int(position_ids.max()) + 1
    if bound > rope_len:
        state.eager_steps += 1
        state.generation_epoch = -1
        return None
    if state.ineligible_epoch == GENERATION_EPOCH:
        state.eager_steps += 1
        return None
    if (
        state.generation_epoch != GENERATION_EPOCH
        or state.optimizer_steps != K._OPTIMIZER_STEPS
        or state.buffer_epoch != BUFFER_EPOCH
        or state.rope_len != rope_len
    ):
        reason = _eligible(model, layers)
        if reason is not None:
            state.disabled = reason
            return None
        fingerprint = _fingerprint(model, layers, rope_len)
        if fingerprint is None:
            # Ineligible for this generation only (several / non-CUDA devices, or a DoRA / non-LoRA
            # adapter, e.g. one side of a compare): free the captured steps and re-check next time.
            state.clear()
            state.ineligible_epoch = GENERATION_EPOCH
            state.eager_steps += 1
            return None
        if (
            fingerprint != state.fingerprint
            or state.buffer_epoch != BUFFER_EPOCH
            or state.rope_len != rope_len
        ):
            state.clear()
            state.fingerprint = fingerprint
        state.generation_epoch = GENERATION_EPOCH
        state.optimizer_steps = K._OPTIMIZER_STEPS
        state.buffer_epoch = BUFFER_EPOCH
        state.rope_len = rope_len

    device = input_ids.device
    config = model.config
    s = _static_inputs(state, bsz, device, paged.dtype, config.hidden_size)
    s["ids"].copy_(input_ids)
    last = position_ids[:, -1:] if position_ids.dim() == 2 else position_ids
    if last.numel() != bsz:
        return None
    s["pos"].copy_(last.view(bsz, 1))
    s["slot"].fill_(seq_len)

    span = _full_span(model, kv_len, cap, bsz)
    if span is not None:
        mask = mask_fn(model, attention_mask, bsz, 1, s["like"], seq_len)
        if mask is not None and mask.shape[-2] != 1:
            return None
        _fill_keep(state, s, mask, bsz, kv_len, cap, device)

    autocast = _autocast_state()
    key = (bsz, device, autocast, torch.is_inference_mode_enabled(), span)
    step = state.graphs.get(key)
    if step is None and not worth_capturing(kv_len):
        state.eager_steps += 1
        state.short_skips += 1
        return None
    if step is None:
        if len(state.graphs) >= (MAX_GRAPHS if span is None else MAX_FULL_GRAPHS):
            state.graphs.clear()
            state.pool = None
        try:
            step = _capture(
                state, model, fast_forward_inference, logits_fn, mask_fn, s, rope_len,
                past_key_values, autocast, attention_mask, seq_len, span,
            )  # fmt: skip
            # The warmup may have built LoRA cast caches or RoPE rows the graph now reads.
            state.fingerprint = _fingerprint(model, layers, rope_len)
        except Exception as error:
            state.disabled = f"capture failed: {type(error).__name__}: {error}"
            state.clear()
            if os.environ.get("UNSLOTH_DECODE_CUDA_GRAPHS_STRICT", "0") == "1":
                raise
            return None
        state.graphs[key] = step
        state.captures += 1
    st = step.statics
    st.kv_len = kv_len
    st.mask = (
        None if span is not None else mask_fn(model, attention_mask, bsz, 1, s["like"], seq_len)
    )
    try:
        step.replay()
    finally:
        st.mask = None
    state.replays += 1
    next_cache = [
        (
            layer.self_attn.paged_attention_K[:kv_len].permute(1, 2, 0, 3),
            layer.self_attn.paged_attention_V[:kv_len].permute(1, 2, 0, 3),
        )
        for layer in layers
    ]
    return step.logits.clone(), next_cache


def _autocast_state():
    try:
        enabled = torch.is_autocast_enabled("cuda")
        dtype = torch.get_autocast_dtype("cuda") if enabled else None
    except TypeError:
        enabled = torch.is_autocast_enabled()
        dtype = torch.get_autocast_gpu_dtype() if enabled else None
    return (enabled, dtype)


def _capture(
    state,
    model,
    fast_forward_inference,
    logits_fn,
    mask_fn,
    s,
    rope_len,
    past_key_values,
    autocast,
    attention_mask,
    seq_len,
    span = None,
):
    if state.stream is not None and state.stream.device.index != torch.cuda.current_device():
        state.stream = None
        state.pool = None
    if state.pool is None:
        state.pool = torch.cuda.graph_pool_handle()
    if state.stream is None:
        state.stream = torch.cuda.Stream()
    stream = state.stream
    st = _StepStatics(
        rope_len, s["slot"], span,
        s.get("keep") if span is not None else None, s.get("drop") if span is not None else None,
    )  # fmt: skip
    st.kv_len = seq_len + 1
    if span is None:
        st.mask = mask_fn(model, attention_mask, s["ids"].shape[0], 1, s["like"], seq_len)
    enabled, dtype = autocast
    # No autocast weight cache: a cast cached during capture would be freed into the graph's pool.
    no_cache = lambda: torch.autocast("cuda", dtype = dtype, enabled = enabled, cache_enabled = False)
    stream.wait_stream(torch.cuda.current_stream())
    step = _Step(st)
    try:
        with torch.cuda.stream(stream), no_cache():
            # Warmup (JIT, workspaces, lazy RoPE / LoRA casts); its K / V write equals the replay's.
            _run_step(model, fast_forward_inference, logits_fn, s, st, past_key_values)
            stream.synchronize()

            def split(fn):
                # The next segment reads the attention core's output address.
                _end(step.segments[-1])
                # Same address on two calls = persistent output buffer; else (SDPA) copy to a fixed one.
                out = fn()
                fixed = out if fn().data_ptr() == out.data_ptr() else torch.empty_like(out)
                step.cores.append(fn if fixed is out else (lambda: fixed.copy_(fn())))
                graph = torch.cuda.CUDAGraph()
                step.segments.append(graph)
                _begin(graph, state.pool)
                return fixed

            graph = torch.cuda.CUDAGraph()
            step.segments.append(graph)
            if span is None:
                st.attention = split
            _begin(graph, state.pool)
            try:
                step.logits = _run_step(
                    model, fast_forward_inference, logits_fn, s, st, past_key_values
                )
            finally:
                st.attention = _run_now
                if torch.cuda.is_current_stream_capturing():
                    _end(step.segments[-1])
        return step
    finally:
        # Also after a failed warmup / capture: the eager fallback must not race the side stream.
        torch.cuda.current_stream().wait_stream(stream)
        st.mask = None


def _begin(graph, pool):
    # Outside inference mode: the first capture creates the CUDA RNG's graph state tensors, and
    # inference tensors there make every later capture under no_grad fail (torch 2.10).
    with torch.inference_mode(False):
        graph.capture_begin(pool = pool, capture_error_mode = "thread_local")


def _end(graph):
    with torch.inference_mode(False):
        graph.capture_end()


def stats(model):
    state = getattr(model, "__dict__", {}).get("_unsloth_decode_graphs")
    if state is None:
        return {}
    return {
        "captures": state.captures,
        "replays": state.replays,
        "graphs": len(state.graphs),
        "eager_steps": state.eager_steps,
        "short_skips": state.short_skips,
        "disabled": state.disabled,
    }
