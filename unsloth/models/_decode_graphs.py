# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""CUDA-graph decode; UNSLOTH_DECODE_CUDA_GRAPHS=1 forces, =0 disables; unset, only RL-trainer
generate and Studio's chat worker on NVIDIA CUDA; graphs dropped when params, LoRA, quant or RoPE
change."""

import math
import os
import sys
import threading

import torch

# Programmatic override (tests, harnesses): True / False force graphs on / off, None (default) follows
# the environment variable read at call time and the RL / Studio generation scopes (enabled()).
ENABLED = None
# KV-cache rows are allocated in multiples of this with graphs on.
BUCKET = 256
# Captured steps (one per batch size / autocast state) kept per model.
MAX_GRAPHS = int(os.environ.get("UNSLOTH_DECODE_GRAPH_MAX", "8"))

__all__ = [
    "ENABLED",
    "enabled",
    "rl_generation",
    "studio_generation",
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

    __slots__ = ("rope_len", "slot", "kv_len", "mask", "attention")

    def __init__(self, rope_len, slot):
        self.rope_len = rope_len
        self.slot = slot
        self.kv_len = None
        self.mask = None
        self.attention = _run_now


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
            # Graphs reading a replaced / evicted buffer are dropped (BUFFER_EPOCH); a new batch size
            # beside the ones kept leaves their graphs valid.
            if bufs is not None or len(store) >= 2:
                store.clear()
                BUFFER_EPOCH += 1
            bufs = store[key] = (paged, temp_QA, temp_KV, RH_Q, temp_O, scores)
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
    if "_unsloth_graph_buffers" in attn.__dict__:
        # Each growth re-captures: double, capped at the hint, so O(log n) re-captures.
        grown = max(rows + increment, 2 * rows)
        if MAX_LEN_HINT + 1 > rows:
            grown = min(grown, MAX_LEN_HINT + 1)
        increment = _round_up(grown, BUCKET) - rows
    attn.paged_attention.resize_((rows + increment, 2, bsz, n_kv_heads, head_dim))
    attn.paged_attention_K = attn.paged_attention[:, 0]
    attn.paged_attention_V = attn.paged_attention[:, 1]
    attn.attention.resize_((bsz, n_heads, 1, attn.attention.shape[-1] + increment))
    if "_unsloth_graph_buffers" in attn.__dict__:
        attn.paged_attention[rows:].zero_()
        BUFFER_EPOCH += 1


def start_generation(max_len):
    global GENERATION_EPOCH, MAX_LEN_HINT
    GENERATION_EPOCH += 1
    MAX_LEN_HINT = max(0, int(max_len or 0))


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
    if (
        paged is None
        or "_unsloth_graph_buffers" not in attn0.__dict__
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

    autocast = _autocast_state()
    key = (bsz, device, autocast, torch.is_inference_mode_enabled())
    step = state.graphs.get(key)
    if step is None:
        if len(state.graphs) >= MAX_GRAPHS:
            state.graphs.clear()
            state.pool = None
        try:
            step = _capture(
                state, model, fast_forward_inference, logits_fn, mask_fn, s, rope_len,
                past_key_values, autocast, attention_mask, seq_len,
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
    st.mask = mask_fn(model, attention_mask, bsz, 1, s["like"], seq_len)
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
):
    if state.stream is not None and state.stream.device.index != torch.cuda.current_device():
        state.stream = None
        state.pool = None
    if state.pool is None:
        state.pool = torch.cuda.graph_pool_handle()
    if state.stream is None:
        state.stream = torch.cuda.Stream()
    stream = state.stream
    st = _StepStatics(rope_len, s["slot"])
    st.kv_len = seq_len + 1
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
        "disabled": state.disabled,
    }
