# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""One torch.compile region per decoder layer for training over the stock forward (attention,
MLP/MoE, norms, LoRA, NF4); UNSLOTH_BLOCK_COMPILE=0 or UNSLOTH_COMPILE_DISABLE turns it off."""

import contextlib
import functools
import inspect
import os
import sys
import time
import types
import weakref

import torch

__all__ = ["patch_block_compile"]

_MARK = "_unsloth_block_compiled"
# No max_autotune: it multiplies first-step compile time for every block.
_COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "shape_padding": True,
    "triton.cudagraphs": False,
}


def _strict_dynamo():
    """Dynamo with suppress_errors off; some torch builds (AMD's ROCm torch 2.11) have no such flag."""
    config = torch._dynamo.config
    return (
        config.patch(suppress_errors = False)
        if hasattr(config, "suppress_errors")
        else contextlib.nullcontext()
    )


def _cudagraph_mode():
    """UNSLOTH_BLOCK_CUDAGRAPHS: "1" always on, unset or "auto" decided per run (see _auto_observe),
    anything else off (before auto, every value but "1" meant off)."""
    value = os.environ.get("UNSLOTH_BLOCK_CUDAGRAPHS", "auto").strip().lower()
    if value == "1":
        return "on"
    return "auto" if value in ("", "auto") else "off"


# Per-block CUDA graphs only pay off for static shapes on a launch-bound step.
_CUDAGRAPH_MODE = _cudagraph_mode()
if _CUDAGRAPH_MODE == "on":
    _COMPILE_OPTIONS["triton.cudagraphs"] = True
# Shape cap: variable lengths recorded a graph set per length (3.5x slower, pool grew every step).
# Two shapes = one fixed length plus a short last batch. Past the cap graphs go off for every block
# and the graph pool is released (T4: 8 shapes cost 15-65 s over 40 steps and kept 3-4.75 GB).
try:
    _CUDAGRAPH_MAX_SHAPES = max(0, int(os.environ.get("UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES", "2")))
except ValueError:
    _CUDAGRAPH_MAX_SHAPES = 2


# Process-wide cudagraph state (one tree, one pool per device): off = why graphs went off, None =
# on; release = None, done or kept; ours = compile ids of tree functions block calls made; built =
# compile ids the graphs-on backend compiled; managers = CUDA index to tree manager
def _new_cudagraph_state():
    return {
        "off": None,
        "release": None,
        "ours": set(),
        "built": set(),
        "managers": {},
        "notes": set(),
        "auto": _new_auto_state(),
    }


# Auto mode: phase "probe" (graphs off, watching), "on" or "off" (final). first: the layer whose
# call starts a forward pass; passes: forward passes seen, all with batch shape `shape` on `device`.
def _new_auto_state():
    return {
        "phase": "probe",
        "first": None,
        "shape": None,
        "device": None,
        "passes": 0,
        "host": [],
        "gpu": [],
        "decision": None,
        "why": None,
    }


_CUDAGRAPH_STATE = _new_cudagraph_state()


_PACKED_NOTE = (
    "Unsloth: UNSLOTH_BLOCK_CUDAGRAPHS=1 does not apply to packed / padding-free rows (their total "
    "length changes every step, so a graph would be recorded per step); blocks run without CUDA graphs."
)
_SHAPES_NOTE = (
    "Unsloth: block CUDA graphs turned off: more than {} distinct batch shapes (variable padded "
    "lengths). Pad every batch to one length (padding='max_length') to keep them, or raise "
    "UNSLOTH_BLOCK_CUDAGRAPHS_MAX_SHAPES."
)


_PACKED_STOCK_NOTE = (
    "Unsloth: packed / padding-free rows run each transformer block uncompiled; only unpacked "
    "batches use the compiled blocks."
)


def _rank0_note(key, message):
    state = _CUDAGRAPH_STATE
    if key in state["notes"]:
        return
    state["notes"].add(key)
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(message, flush = True)


def _graphs_off(reason, message):
    """Every block runs its graphs-off copy from now on; the pool is released at a later call."""
    if _CUDAGRAPH_STATE["off"] is None:
        _CUDAGRAPH_STATE["off"] = reason
        _rank0_note("off", message)


def _tree_manager(device):
    if device is None:
        return None
    try:
        from torch._inductor import cudagraph_trees
        return cudagraph_trees.get_manager(device, create_if_none_exists = False)
    except Exception:
        return None


def _block_backend(options):
    """Inductor noting each compile id it builds, so foreign regions behind a graph break are not
    counted as ours. None if unsupported (then the pool is kept)."""
    try:
        from torch._guards import CompileContext
        class _BlockInductor(torch._TorchCompileInductorWrapper):
            def __call__(
                self,
                *args,
                options = None,
                **kwargs,
            ):
                # options were already applied by the constructor.
                _CUDAGRAPH_STATE["built"].add(CompileContext.current_compile_id())
                return super().__call__(*args, **kwargs)

        return _BlockInductor(None, options, None)
    except Exception:
        return None


def _graphed_call(compiled, device, *args, **kwargs):
    """Records the compile ids of tree functions this block built, so release knows the pool is ours."""
    managers = _CUDAGRAPH_STATE["managers"]
    manager = managers.get(device)
    if manager is None:
        # The manager may predate the first block call (another region made it).
        manager = _tree_manager(device)
        if manager is not None:
            managers[device] = manager
    before = len(manager.ids_to_funcs) if manager is not None else 0
    out = _mark_block_outputs(compiled(*args, **kwargs))
    if manager is None:
        manager = _tree_manager(device)
        if manager is None:
            return out
        managers[device] = manager
    if len(manager.ids_to_funcs) > before:
        built = _CUDAGRAPH_STATE["built"]
        cids = (manager.id_to_compile_id.get(i) for i in list(manager.ids_to_funcs)[before:])
        _CUDAGRAPH_STATE["ours"].update(cid for cid in cids if cid is not None and cid in built)
    return out


_BLOCK_NODE = "_unsloth_graphed_block"


def _mark_block_outputs(out):
    """Tag the autograd nodes of a graphed block call, so a later block's input check can stop there:
    that call decided its own input gradient. Other compiled functions may pass gradients through."""
    import torch.utils._pytree as pytree

    for x in pytree.tree_leaves(out):
        if torch.is_tensor(x) and x.grad_fn is not None:
            x.grad_fn.metadata[_BLOCK_NODE] = True
    return out


def _pool_release_verdict(managers, ours):
    """ "ok" when every tree function belongs to a block compile and nothing in the pool is alive
    (no pending backward), "pending" to retry at a later call, "foreign" when another torch.compile
    region also records into this tree: resetting would break its next call."""
    for manager in managers:
        if any(cid is None or cid not in ours for cid in manager.id_to_compile_id.values()):
            return "foreign"
    for manager in managers:
        if manager.running_forwards_with_pending_backwards:
            return "pending"
        if manager.live_cudagraph_pool_storages_in_curr_execution():
            return "pending"
    return "ok"


def _release_graph_pool():
    """Free the cudagraph tree pool via reset_cudagraph_trees(), only when every tree function is ours:
    a foreign one called afterwards fails ("Running CUDAGraph after shutdown")."""
    state = _CUDAGRAPH_STATE
    if state["release"] is not None:
        return
    try:
        from torch._inductor import cudagraph_trees
        containers = cudagraph_trees.get_obj(cudagraph_trees.local, "tree_manager_containers")
        managers = [c.tree_manager for c in list(containers.values()) if c.tree_manager is not None]
    except Exception:
        state["release"] = "kept"
        return
    if not managers:
        state["release"] = "done"  # nothing was recorded
        return
    verdict = _pool_release_verdict(managers, state["ours"])
    if verdict == "pending":
        return
    if verdict == "foreign":
        state["release"] = "kept"
        _rank0_note(
            "kept",
            "Unsloth: block CUDA graphs are off; their memory pool stays allocated because another "
            "compiled region records into the same CUDA graph tree.",
        )
        return
    devices = sorted(state["managers"])
    before = sum(torch.cuda.memory_reserved(d) for d in devices)
    cudagraph_trees.reset_cudagraph_trees()
    torch.cuda.empty_cache()
    freed = before - sum(torch.cuda.memory_reserved(d) for d in devices)
    state["release"] = "done"
    state["managers"].clear()
    _rank0_note(
        "released",
        f"Unsloth: released the block CUDA graph pool ({max(freed, 0) / 2**30:.2f} GiB reserved freed).",
    )


def _packed_kwargs(kwargs):
    get = kwargs.get
    return (
        get("packed_seq_lengths") is not None
        or get(_PACKED_BLOCK_KWARG) is not None
        or get("cu_seq_lens_q") is not None
        or get("cu_seq_lens_k") is not None
        or get("cu_seqlens") is not None
    )


def _grad_side_inputs(args, kwargs):
    """A non-hidden_states input needing grad (Mllama cross-attention states): its replayed-backward grad lives in
    graph pool memory the next replay overwrites before it is summed, and only hidden_states is guarded (_CloneGrad)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor) and value.requires_grad:
            return True
    return False


# Bumped on every module / parameter registration (PEFT add_adapter), so cached block parameter
# lists are rebuilt; None until the global hooks are installed, False if this torch has none.
_PARAMS_GEN = [None]


def _bump_params_gen(*args):
    _PARAMS_GEN[0] += 1


def _watch_registrations():
    if _PARAMS_GEN[0] is not None:
        return
    from torch.nn.modules import module as nn_module

    names = (
        "register_module_module_registration_hook",
        "register_module_parameter_registration_hook",
    )
    registers = [getattr(nn_module, name, None) for name in names]
    if any(register is None for register in registers):
        _PARAMS_GEN[0] = False
        return
    _PARAMS_GEN[0] = 0
    for register in registers:
        register(_bump_params_gen)


def _block_params(layer):
    d = layer.__dict__
    params = d.get("_unsloth_block_params")
    _watch_registrations()
    gen = _PARAMS_GEN[0]
    if params is None or (gen is not False and d.get("_unsloth_block_params_gen") != gen):
        params = d["_unsloth_block_params"] = list(layer.parameters())
        d["_unsloth_block_params_gen"] = gen
    return params


def _trainable(layer):
    return any(p.requires_grad for p in _block_params(layer))


# The grad-mode trial before a checkpointed block's first no-grad call stands in for the recompute.
_IN_TRIAL = [False]


def _same(tensor):
    return tensor


@contextlib.contextmanager
def _no_output_capture(layer):
    """Hide the trial forward from transformers output recording (an extra router_logits entry skews the aux loss)."""
    swapped = []
    for module in layer.modules():
        fn = module.__dict__.get("forward")
        # functools.wraps copies __qualname__, not the code object's.
        code = getattr(fn, "__code__", None)
        qual = getattr(code, "co_qualname", None)  # Python >= 3.11
        if qual is None:
            qual = (
                "make_capture_wrapper" if getattr(code, "co_name", "") == "wrapped_forward" else ""
            )
        if hasattr(fn, "__wrapped__") and "make_capture_wrapper" in qual:
            swapped.append((module, fn))
            module.forward = fn.__wrapped__
    # Live collector, trial entries trimmed after: a hidden (None) one compiled a graph no real call reuses.
    try:
        from transformers.utils.output_capturing import _active_collector
        collected = _active_collector.get()
    except Exception:
        collected = None
    lists = []
    if isinstance(collected, dict):
        lists = [(v, len(v)) for v in collected.values() if type(v) is list]
    try:
        yield
    finally:
        for values, length in lists:
            del values[length:]
        for module, fn in swapped:
            module.forward = fn


# output_hidden_states hands block outputs to the caller, who may keep them past the generation.
_HIDDEN_STATES_EXPOSED = [False]


def _note_hidden_states_exposed(module, args, kwargs):
    want = kwargs.get("output_hidden_states")
    if want is None:
        want = getattr(getattr(module, "config", None), "output_hidden_states", False)
    _HIDDEN_STATES_EXPOSED[0] = bool(want)


def _watch_hidden_states(model):
    """Hook the model and each decoder list's owner to record whether hidden states are returned."""
    modules = dict(model.named_modules())
    lists = {id(layer_list) for layer_list in _layer_lists(model)}
    owners = [model]
    for name, module in modules.items():
        if id(module) in lists and "." in name:
            owners.append(modules[name.rsplit(".", 1)[0]])
    for owner in owners:
        if not owner.__dict__.get("_unsloth_watch_hidden_states", False):
            owner.__dict__["_unsloth_watch_hidden_states"] = True
            owner.register_forward_pre_hook(_note_hidden_states_exposed, with_kwargs = True)


def _pool_outputs_stay_valid(layer, hidden_states):
    """Grad-mode calls outside backward leave a forward pending, so their outputs stay valid in the pool."""
    return (
        not _IN_TRIAL[0]
        and not _HIDDEN_STATES_EXPOSED[0]
        and torch.is_grad_enabled()
        and torch._C._current_graph_task_id() == -1
        and (hidden_states.requires_grad or _trainable(layer))
    )


def _input_grad_must_leave_pool(hidden_states):
    """True when the input gradient outlives the generation: a leaf's .grad (kept live, so the pool is
    never released) or a checkpoint's backward, which recomputes (a new generation) before reading it."""
    if not hidden_states.requires_grad:
        return False
    node = hidden_states.grad_fn
    if node is None:
        return True  # leaf: its .grad would be pool memory
    # (node, fresh): fresh once the path passed an op whose backward allocates its input gradient.
    stack, seen = [(node, False)], 0
    while stack:
        node, fresh = stack.pop()
        seen += 1
        if seen > 32:
            return True
        name = type(node).__name__
        if name.startswith("CompiledFunction") and node.metadata.get(_BLOCK_NODE, False):
            continue  # the previous block's region: its own input is its own call's decision
        if isinstance(node, torch.autograd.function.BackwardCFunction):
            forward_cls = getattr(node, "_forward_cls", None)
            if not (isinstance(forward_cls, type) and issubclass(forward_cls, _CloneGrad)):
                return True  # a checkpoint (or other Python autograd Function) may recompute first
        if name == "AccumulateGrad" and not fresh:
            return True  # a view / add of a leaf (inputs_embeds = param.view(...)) keeps the pool gradient
        allocates = name.startswith(_ALLOCATING_BACKWARD)
        for i, (n, _) in enumerate(node.next_functions):
            if n is None:
                continue
            edge_fresh = fresh or allocates
            if allocates and i == 0 and name.startswith(_ADDITIVE_BACKWARD):
                # addmm / baddbmm hand their additive input `grad` itself (beta = 1), unless it was
                # broadcast (a bias), which the engine sums to its shape in new storage
                full = 3 if name.startswith("Baddbmm") else 2
                variable = getattr(n, "variable", None)
                edge_fresh = fresh or (variable is not None and variable.dim() < full)
            stack.append((n, edge_fresh))
    return False


_ADDITIVE_BACKWARD = ("AddmmBackward", "BaddbmmBackward")


# Backward ops that allocate fresh storage for the input gradient (not a view, add or cat).
_ALLOCATING_BACKWARD = (
    "MmBackward",
    "AddmmBackward",
    "BmmBackward",
    "BaddbmmBackward",
    "MulBackward",
    "DivBackward",
    "EmbeddingBackward",
    "EmbeddingDenseBackward",
    "NativeLayerNormBackward",
    "FusedRmsNormBackward",
    "SoftmaxBackward",
    "SiluBackward",
    "GeluBackward",
    "IndexBackward",
    "SliceBackward",
    "SelectBackward",
    "ConvolutionBackward",
)


class _CloneGrad(torch.autograd.Function):
    """Identity; backward copies the input gradient out of the CUDA graph pool."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return grad.clone()


def _clone_outputs(out):
    import torch.utils._pytree as pytree
    return pytree.tree_map(lambda x: x.clone() if torch.is_tensor(x) else x, out)


def _pool_grad_hook(param):
    ref = weakref.ref(param)

    def hook(grad):
        p = ref()
        if p is None or p.grad is not None:
            return None
        return grad.clone()

    return hook


def _param_grads_out_of_pool(layer):
    """A first .grad from a replayed backward is graph pool memory the next replay overwrites: copy it.
    Hooks follow requires_grad at each grad-mode call, not the first call: PEFT's set_adapter /
    disable_adapter flip it, and the first graphed call can be a reference forward (TRL DPO's "ref"
    adapter, disable_adapter) with the trained adapter frozen."""
    if not torch.is_grad_enabled():
        return  # this call makes no parameter gradient
    hooked = layer.__dict__.get("_unsloth_block_grad_hooks")
    if not isinstance(hooked, dict):
        hooked = layer.__dict__["_unsloth_block_grad_hooks"] = {}
    for param in _block_params(layer):
        if param.requires_grad:
            seen = hooked.get(id(param))
            if seen is None or seen() is not param:
                param.register_hook(_pool_grad_hook(param))
                hooked[id(param)] = weakref.ref(param)


def _in_nonreentrant_checkpoint():
    """Non-reentrant checkpoint frees recomputed inputs mid graph recording, which cudagraph trees refuse."""
    if _IN_TRIAL[0] == "nonreentrant":
        return True
    try:
        hooks = torch._C._autograd._top_saved_tensors_default_hooks(False)
    except Exception:
        return False
    return bool(hooks) and getattr(hooks[0], "__module__", None) == "torch.utils.checkpoint"


def _graphs_unsafe_moe(layer, note = True):
    """MoE routing / grouped GEMMs read token counts the capture cannot hold (Qwen3-MoE: host copy
    during capture, routing outputs overwritten by the next replay)."""
    moe = layer.__dict__.get("_unsloth_block_graphs_moe")
    if moe is None:
        # "experts" child: transformers 4.57 DeepSeek-V2 / V3, GLM4-MoE have no num_experts.
        moe = any(
            _experts_of(m) is not None
            or m._modules.get("experts") is not None
            or (hasattr(m, "num_experts") and hasattr(m, "experts"))
            for m in layer.modules()
        )
        layer.__dict__["_unsloth_block_graphs_moe"] = moe
    if moe and note:
        _rank0_note(
            "moe",
            "Unsloth: block CUDA graphs skip MoE layers; they run compiled without graphs.",
        )
    return moe


def _flex_block_mask_call(
    args,
    kwargs,
    note = True,
):
    """BlockMask index tensors are static graph inputs rebuilt every step, so the graph re-records
    each call (StaticInputIdxMismatch is outside the re-record limit); Gemma3 was slower with graphs."""
    for value in (*args, *kwargs.values()):
        if type(value).__name__ == "BlockMask":
            if note:
                _rank0_note(
                    "flex",
                    "Unsloth: block CUDA graphs skip flex attention blocks; they run compiled without graphs.",
                )
            return True
    return False


# Auto: graphs start when launch time exceeds GPU time by _AUTO_RATIO and all calls had one shape.
_AUTO_WARMUP_PASSES = 2
_AUTO_MEASURE_CALLS = 4
_AUTO_SLEEP_CYCLES = 20_000_000  # ~10-15 ms: the block's launches queue behind it
try:
    _AUTO_RATIO = float(os.environ.get("UNSLOTH_BLOCK_CUDAGRAPHS_AUTO_RATIO", "1.2"))
except ValueError:
    _AUTO_RATIO = 1.2
# 2.13+: the graphed region gets its own recompile bucket (isolate_recompiles) instead of sharing
# the stock region's cache limit. Before 2.11 FastLanguageModel LoRA blocks are not compiled at all.
_AUTO_MIN_TORCH = "2.13"
# The graph pool adds 0.5-0.8 GB reserved (Qwen3-0.6B LoRA): keep graphs off without this much free.
_AUTO_FREE_GB = 2.0
_AUTO_FREE_FRACTION = 0.10


def _auto_note(message):
    if os.environ.get("UNSLOTH_ENABLE_LOGGING", "0") == "1":
        _rank0_note("auto", message)


def _auto_off(auto, why):
    auto["phase"] = "off"
    auto["why"] = why
    auto["first"] = None
    _auto_note(f"Unsloth: block CUDA graphs stay off ({why}).")


def _auto_platform_reason():
    """Why auto mode can never turn graphs on in this process (None: it may)."""
    if getattr(torch.version, "hip", None):
        return "ROCm"
    from packaging.version import Version

    if Version(torch.__version__.split("+")[0]) < Version(_AUTO_MIN_TORCH):
        return f"torch < {_AUTO_MIN_TORCH}"
    return None


def _auto_static_reason(layer, hidden_states):
    """Why auto mode never turns graphs on for this run (None: it may)."""
    if not hidden_states.is_cuda:
        return "not a CUDA tensor"
    why = _auto_platform_reason()
    if why is not None:
        return why
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            return "distributed"
    except Exception:
        return "distributed"
    total = trainable = 0
    for p in layer.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    if total and trainable * 2 > total:
        # Measured only for LoRA / QLoRA: full finetuning copies every block weight grad per step.
        return "full finetuning"
    return None


def _auto_call_reason(layer, hidden_states, args, kwargs):
    """Why this probed call rules graphs out (None: it does not)."""
    if _packed_kwargs(kwargs):
        return "packed / padding-free rows"
    if _grad_side_inputs(args, kwargs):
        return "a side input takes a gradient"
    if _in_nonreentrant_checkpoint():
        return "non-reentrant checkpointing"
    if _graphs_unsafe_moe(layer, note = False):
        return "MoE layer"
    if _flex_block_mask_call(args, kwargs, note = False):
        return "flex attention"
    return None


def _auto_block_device(hidden_states):
    return hidden_states.device


def _auto_observe(layer, hidden_states, args, kwargs):
    """Auto mode, per block call while undecided: "graphs", "plain", or "measure" (plain, timed)."""
    auto = _CUDAGRAPH_STATE["auto"]
    if auto["phase"] == "off":
        return "plain"
    if auto["phase"] == "on":
        return "graphs"
    outside = _IN_TRIAL[0] or torch._C._current_graph_task_id() != -1
    if auto["first"] is None:
        if outside:
            return "plain"  # the trial / a recompute before any forward pass was seen
        why = _auto_static_reason(layer, hidden_states)
        if why is not None:
            _auto_off(auto, why)
            return "plain"
        auto["first"] = weakref.ref(layer)
        auto["shape"] = tuple(hidden_states.shape)
        auto["device"] = _auto_block_device(hidden_states)
    why = _auto_call_reason(layer, hidden_states, args, kwargs)
    if why is None and tuple(hidden_states.shape) != auto["shape"]:
        why = "batch shape changed"
    if why is None and _auto_block_device(hidden_states) != auto["device"]:
        # A device_map split: timing and free memory are only read on one device.
        why = "blocks on several devices"
    if why is not None:
        _auto_off(auto, why)
        return "plain"
    if outside:
        return "plain"
    if auto["first"]() is layer:  # a new forward pass
        if auto["decision"] == "on":
            auto["phase"] = "on"
            auto["first"] = None
            return "graphs"
        auto["passes"] += 1
    if auto["passes"] > _AUTO_WARMUP_PASSES and len(auto["host"]) < _AUTO_MEASURE_CALLS:
        return "measure"
    return "plain"


def _auto_decide(auto, device):
    host, gpu = sum(auto["host"]), sum(auto["gpu"])
    ratio = host / max(gpu, 1e-9)
    detail = f"block launch {host * 1e3 / len(auto['host']):.2f} ms vs GPU {gpu * 1e3 / len(auto['gpu']):.2f} ms"
    auto["ratio"] = ratio
    if ratio < _AUTO_RATIO:
        _auto_off(auto, f"GPU-bound: {detail}")
        return
    try:
        free, total = torch.cuda.mem_get_info(device)
    except Exception:
        _auto_off(auto, "free memory unknown")
        return
    need = max(_AUTO_FREE_GB * 2**30, _AUTO_FREE_FRACTION * total)
    if free < need:
        _auto_off(auto, f"{free / 2**30:.1f} GiB free")
        return
    auto["decision"] = "on"
    _rank0_note(
        "auto",
        f"Unsloth: block CUDA graphs on (one batch shape, {detail}). "
        "UNSLOTH_BLOCK_CUDAGRAPHS=0 keeps them off.",
    )


def _auto_time(call, device):
    """Sleep and events run on the block device (cuda:1 or a device_map shard need not be current);
    the sleep lets launches queue behind it; host and GPU are None when timing is unavailable"""
    try:
        guard = torch.cuda.device(device)
    except Exception:
        return call(), None, None
    with guard:
        try:
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing = True)
            end = torch.cuda.Event(enable_timing = True)
            torch.cuda._sleep(_AUTO_SLEEP_CYCLES)
            start.record()
        except Exception:
            return call(), None, None
        t0 = time.perf_counter()
        out = call()
        host = time.perf_counter() - t0
        try:
            end.record()
            end.synchronize()
            return out, host, start.elapsed_time(end) / 1e3
        except Exception:
            return out, None, None


def _auto_measure(call, hidden_states):
    auto = _CUDAGRAPH_STATE["auto"]
    out, host, gpu = _auto_time(call, hidden_states.device)
    if auto["phase"] != "probe" or auto["decision"] is not None:
        return out
    if host is None or gpu is None:
        _auto_off(auto, "timing unavailable")
        return out
    auto["host"].append(host)
    auto["gpu"].append(gpu)
    if len(auto["host"]) >= _AUTO_MEASURE_CALLS:
        _auto_decide(auto, hidden_states.device)
    return out


def _note_block_mask_call(layer, args, kwargs):
    """_flex_block_mask_call, also recording `_unsloth_block_flex_only`; the observer runs on each change."""
    flex = _flex_block_mask_call(args, kwargs)
    seen = layer.__dict__.get("_unsloth_block_flex_only")
    if seen is False or seen is flex:
        return flex
    # None -> True on a first BlockMask call; True -> False (sticky) on a call without one.
    layer.__dict__["_unsloth_block_flex_only"] = flex
    observer = layer.__dict__.get("_unsloth_block_flex_observer")
    if observer is not None and (flex or seen is True):
        observer(layer)
    return flex


def _ddp_optimizer_mode():
    # optimize_ddp also takes "python_reducer" / "no_optimization" strings: only "ddp_optimizer"
    # (or True) makes dynamo split graphs (convert_frame.py checks == "ddp_optimizer").
    getter = getattr(torch._dynamo.utils, "get_optimize_ddp_mode", None) or getattr(
        torch._dynamo.config, "_get_optimize_ddp_mode", None
    )
    if getter is not None:
        return getter() == "ddp_optimizer"
    return getattr(torch._dynamo.config, "optimize_ddp", False) is True


try:
    from torch.nn.parallel import DistributedDataParallel as _DDP
except Exception:
    _DDP = None


def _no_ddp_graph_split():
    """Stop DDPOptimizer splitting block graphs so checkpoint recomputes save the same tensors as
    the forward (CheckpointError under DDP non-reentrant checkpointing); gradients still reach
    DDP buckets block by block"""
    try:
        if _DDP is None or _DDP._get_active_ddp_module() is None or not _ddp_optimizer_mode():
            return contextlib.nullcontext()
    except Exception:
        return contextlib.nullcontext()
    return torch._dynamo.config.patch(optimize_ddp = False)


def _isolate_supported():
    # torch 2.13+: per-call recompile_limit and isolate_recompiles (own cache bucket per region).
    try:
        params = inspect.signature(torch.compile).parameters
    except (TypeError, ValueError):
        return False
    return "isolate_recompiles" in params and "recompile_limit" in params


_ISOLATE_RECOMPILES = _isolate_supported()


def _region_recompile_limit(kinds):
    """Graphs per block region before eager. Measured at most 4 per kind (Gemma3 GRPO: 20 for 2)."""
    env = os.environ.get("UNSLOTH_BLOCK_COMPILE_RECOMPILE_LIMIT", "")
    try:
        if int(env) > 0:
            return int(env)
    except ValueError:
        pass
    return max(16 * kinds + 16, 32)


# gpt-oss builds its flex mask inside attention, so the block broke and recompiled per length;
# DeepSeek-V4.1 top-k gather [B, S, min(index_topk, T), D] fails Inductor backward on torch 2.14
_SKIP = {"GptOssDecoderLayer", "DeepseekV41DecoderLayer"}


def _enabled():
    return os.environ.get("UNSLOTH_BLOCK_COMPILE", "1") != "0" and os.environ.get(
        "UNSLOTH_COMPILE_DISABLE", "0"
    ) not in ("1", "partial")


def _zero3_partitioned(layer = None):
    # Compiled regions see ZeRO-3's empty placeholder, not the gathered weight (zoo's check).
    module = sys.modules.get("transformers.integrations.deepspeed")
    ref = getattr(module, "_hf_deepspeed_config_weak_ref", None)
    config = ref() if ref is not None else None
    try:
        if config is not None and config.is_zero3():
            return True
    except Exception:
        pass
    if layer is not None and "deepspeed" in sys.modules:
        # Any parameter: adapters added outside zero.Init sit next to partitioned base weights.
        return any(hasattr(param, "ds_id") for param in layer.parameters())
    return False


def _forced_float32(module):
    # Zoo's float32-forced patches round to fp16 on purpose; Inductor drops those roundings.
    # Stamped per layer at load: the env var only describes the latest load.
    forced = module.__dict__.get("_unsloth_forced_float32")
    if forced is None:
        forced = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    return forced


def _writes_globals(code):
    import dis
    if any(i.opname in ("STORE_GLOBAL", "DELETE_GLOBAL") for i in dis.get_instructions(code)):
        return True
    return any(_writes_globals(c) for c in code.co_consts if isinstance(c, types.CodeType))


def _rebind_globals(fn):
    """Dynamo resolves globals via sys.modules; exec-generated code is rebuilt on a registered
    module holding a copy of its dict; functions that assign globals are left alone since a copy
    cannot follow later writes"""
    g = getattr(fn, "__globals__", None)
    if g is None or _writes_globals(fn.__code__):
        return fn
    mod = sys.modules.get(g.get("__name__"))
    if mod is not None and mod.__dict__ is g:
        return fn
    name = f"unsloth_block_globals_{id(g)}"
    holder = sys.modules.get(name)
    if holder is None:
        holder = types.ModuleType(name)
        holder.__dict__.update(g)
        holder.__dict__["__name__"] = name
        sys.modules[name] = holder
    new = types.FunctionType(
        fn.__code__, holder.__dict__, fn.__name__, fn.__defaults__, fn.__closure__
    )
    new.__kwdefaults__ = fn.__kwdefaults__
    new.__dict__.update(fn.__dict__)
    new.__qualname__ = fn.__qualname__
    new.__module__ = name
    return new


def _inline_compiled(fn):
    """Inside a block, call the original of a compiled method so dynamo inlines it: inlining the
    wrapper guards on its closure and hits Unexpected type CatchErrorsWrapper on torch 2.13 and
    2.14 (Gemma-4 fp16 MLP); outside a block the wrapper runs as before"""
    original = getattr(fn, "_torchdynamo_orig_callable", None)
    if (
        not isinstance(original, types.FunctionType)
        or getattr(fn, "_torchdynamo_disable", False)
        or getattr(fn, "_unsloth_block_inline", False)
    ):
        return fn
    compiled = fn
    original = _rebind_globals(original)

    @functools.wraps(original, updated = ())
    def dispatch(*args, **kwargs):
        if torch.compiler.is_compiling():
            return original(*args, **kwargs)
        return compiled(*args, **kwargs)

    dispatch._unsloth_block_inline = True
    return dispatch


def _rebind_class(cls):
    for key, value in list(cls.__dict__.items()):
        new = _inline_compiled(value) if callable(value) else value
        if isinstance(new, types.FunctionType):
            new = _rebind_globals(new)
        if new is not value:
            setattr(cls, key, new)


def _relax_strict_dynamic(obj):
    """Views drop the strict mark, so a block's specialized length cannot raise ConstraintViolationError."""
    if isinstance(obj, torch.Tensor):
        return obj.view(obj.shape) if getattr(obj, "_dynamo_dynamic_indices", None) else obj
    if type(obj) in (tuple, list):
        new = [_relax_strict_dynamic(o) for o in obj]
        return obj if all(a is b for a, b in zip(new, obj)) else type(obj)(new)
    if type(obj) is dict:
        new = {k: _relax_strict_dynamic(v) for k, v in obj.items()}
        return obj if all(new[k] is obj[k] for k in obj) else new
    return obj


def _mark_seq_dynamic(seq_len, obj):
    if isinstance(obj, torch.Tensor):
        # A float tensor's last dim is a width even when it equals seq_len (mrope ids are int).
        last = obj.dim() - 1 if obj.dim() == 3 and obj.is_floating_point() else None
        for dim, size in enumerate(obj.shape):
            if size == seq_len and dim != last:
                torch._dynamo.maybe_mark_dynamic(obj, dim)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _mark_seq_dynamic(seq_len, o)
    elif isinstance(obj, dict):
        for o in obj.values():
            _mark_seq_dynamic(seq_len, o)


def _mark_batch_dynamic(batch, obj):
    # Batch dim too: an uneven last training batch (size >= 2) otherwise recompiles every graph.
    if isinstance(obj, torch.Tensor):
        if obj.dim() >= 2 and obj.shape[0] == batch:
            torch._dynamo.maybe_mark_dynamic(obj, 0)
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            _mark_batch_dynamic(batch, o)
    elif isinstance(obj, dict):
        for o in obj.values():
            _mark_batch_dynamic(batch, o)


_DYNAMIC_BATCH = os.environ.get("UNSLOTH_BLOCK_COMPILE_DYNAMIC_BATCH", "1") != "0"
# Per-document metadata of a packed row: one entry per document (+1 for cu_seqlens).
_PACKED_LENGTH_KWARGS = ("packed_seq_lengths", "cu_seq_lens_q", "cu_seq_lens_k", "cu_seqlens")


def _mark_packed_dynamic(kwargs):
    # The document count changes from one packed batch to the next: without the mark every block (and
    # the attention frame after its graph break) recompiled once on the second packed batch.
    for key in _PACKED_LENGTH_KWARGS:
        value = kwargs.get(key)
        if isinstance(value, torch.Tensor) and value.dim() == 1:
            torch._dynamo.maybe_mark_dynamic(value, 0)


# fla's opaque op state: None until a GatedDeltaNet layer ran its self-check, then True / False.
_GDN_OP = [None]
# Packed batches (cu_seqlens) through the op: set eagerly with _GDN_OP from the op's self-check.
_GDN_VARLEN = [False]


def _hub_forwarded_params(fn):
    """The keywords transformers' hub-kernel wrapper passes on to fla (it drops the rest, e.g. the
    decoder's num_items_in_batch), unwrapping Unsloth's disable wrappers; None if not a hub call."""
    for _ in range(8):
        code = getattr(fn, "__code__", None)
        if code is not None and fn.__closure__:
            cells = dict(zip(code.co_freevars, fn.__closure__))
            if "implementation" in cells and "applicable_params" in cells:
                try:
                    return frozenset(cells["applicable_params"].cell_contents)
                except (ValueError, TypeError):
                    return None
        fn = getattr(fn, "_unsloth_undisabled", None) or getattr(fn, "__wrapped__", None)
        if fn is None:
            break
    return None


def _gdn_chunk_adapter(original):
    forwarded = _hub_forwarded_params(original)

    # updated = (): the original's __dict__ carries torch.compiler.disable's marker.
    @functools.wraps(original, updated = ())
    def chunk_gated_delta_rule(
        q,
        k,
        v,
        g = None,
        beta = None,
        **kwargs,
    ):
        cu_seqlens = (
            kwargs.get("cu_seqlens") if forwarded is None or "cu_seqlens" in forwarded else None
        )
        if (
            _GDN_OP[0]
            and torch.compiler.is_compiling()
            and kwargs.get("use_qk_l2norm_in_kernel") is True
            # Only the self-checked default; packed batches only if their self-check passed.
            and (cu_seqlens is None or (_GDN_VARLEN[0] and q.shape[0] == 1))
            and all(
                value is None or value is False
                for key, value in kwargs.items()
                if key not in ("use_qk_l2norm_in_kernel", "cu_seqlens")
                and (forwarded is None or key in forwarded)
            )
        ):
            # Inductor pins a custom op's input strides; a sliced input fails at the next length.
            q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
            scale = q.shape[-1] ** -0.5
            if cu_seqlens is None:
                out = torch.ops.unsloth.qwen3_5_gdn_chunk_v2(q, k, v, g, beta, scale)
            else:
                out = torch.ops.unsloth.qwen3_5_gdn_chunk_v2(
                    q, k, v, g, beta, scale, cu_seqlens.contiguous()
                )
            return out[0], None
        return original(q, k, v, g, beta, **kwargs)

    chunk_gated_delta_rule._unsloth_block_adapter = True
    return chunk_gated_delta_rule


def _route_fla_kernel():
    """Repoint namespace refs to fla chunk_gated_delta_rule at the adapter; callers are found by gc
    referrers, not by name (they reach it via globals, closures and compiled copies); closures
    are left alone"""
    import gc

    # gc.get_referrers skips frozen objects, so thaw before the walk or the op never engages.
    frozen = gc.get_freeze_count() > 0
    if frozen:
        gc.unfreeze()
    try:
        return _route_fla_callers(gc)
    finally:
        if frozen:
            gc.freeze()


def _route_fla_callers(gc):
    kernels = []
    for name, module in list(sys.modules.items()):
        if module is not None and name.endswith("fla.ops.gated_delta_rule.chunk"):
            fn = getattr(module, "chunk_gated_delta_rule", None)
            if callable(fn) and fn not in kernels:
                kernels.append(fn)
    callers, frontier = list(kernels), list(kernels)
    for _ in range(4):
        found = []
        for inner in frontier:
            for cell in gc.get_referrers(inner):
                if type(cell).__name__ != "cell":
                    continue
                for closure in gc.get_referrers(cell):
                    if not isinstance(closure, tuple):
                        continue
                    for fn in gc.get_referrers(closure):
                        if (
                            isinstance(fn, types.FunctionType)
                            and fn.__closure__ is closure
                            and fn not in callers
                            and not getattr(fn, "_unsloth_block_adapter", False)
                        ):
                            callers.append(fn)
                            found.append(fn)
        frontier = found
        if not frontier:
            break
    routed = 0
    for caller in callers:
        if getattr(caller, "_unsloth_block_adapter", False):
            continue
        adapter = _gdn_chunk_adapter(caller)
        for space in gc.get_referrers(caller):
            if not isinstance(space, dict) or space is adapter.__dict__:
                continue
            for key, value in list(space.items()):
                if value is not caller or not isinstance(key, str):
                    continue
                # A packing varlen probe keeps the kernel as its `_fn` keyword default; route under it.
                if key.endswith("chunk_gated_delta_rule") or (
                    key == "_fn" and str(space.get("_name", "")).endswith("chunk_gated_delta_rule")
                ):
                    space[key] = adapter
                    routed += 1
    return routed


def _causal_conv1d_adapter(original, depthwise_conv):
    # Inductor's conv backward guards on the exact sequence length: recompile per length.
    # transformers < 5.15 calls with causal-conv1d kwargs (x = ..., seq_idx = None): pass args through unchanged.
    @functools.wraps(original, updated = ())
    def causal_conv1d_fn(*args, **kwargs):
        named = dict(zip(("hidden_states", "weight", "bias", "activation"), args))
        x_key = "x" if "x" in kwargs else "hidden_states"
        hidden_states = named.get("hidden_states", kwargs.get(x_key))
        weight = named.get("weight", kwargs.get("weight"))
        bias = named.get("bias", kwargs.get("bias"))
        activation = named.get("activation", kwargs.get("activation"))
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and len(args) <= 4
            and hidden_states is not None
            and weight is not None
            and weight.dim() == 2
            and activation in (None, "silu", "swish")
            and all(kwargs.get(key) is None for key in ("seq_idx", "cu_seqlens", "initial_states"))
            and not kwargs.get("return_final_states")
        ):
            out = depthwise_conv(hidden_states.to(weight.dtype), weight)
            if bias is not None:
                out = out + bias[:, None]
            if activation is not None:
                out = torch.nn.functional.silu(out)
            return out.to(hidden_states.dtype)
        return original(*args, **kwargs)

    causal_conv1d_fn._unsloth_block_adapter = True
    return causal_conv1d_fn


def _namespaces(fn, depth = 3):
    """Global namespaces a forward reaches through wrappers (closures) and module-level helpers."""
    seen, out, todo = set(), [], [(fn, 0)]
    while todo:
        f, d = todo.pop()
        if not isinstance(f, types.FunctionType) or id(f) in seen:
            continue
        seen.add(id(f))
        if all(f.__globals__ is not g for g in out):
            out.append(f.__globals__)
        if d >= depth:
            continue
        for cell in f.__closure__ or ():
            try:
                todo.append((cell.cell_contents, d + 1))
            except ValueError:
                pass
        for name in f.__code__.co_names:
            todo.append((f.__globals__.get(name), d + 1))
    return out


def _alias_namespace(space):
    """The namespace Dynamo resolves `space`'s globals through: an inlined function's globals are
    read as sys.modules[__name__], and Unsloth's compiled modules import transformers' __name__."""
    module = (
        sys.modules.get(space.get("__name__")) if isinstance(space.get("__name__"), str) else None
    )
    alias = getattr(module, "__dict__", None)
    return alias if isinstance(alias, dict) and alias is not space else None


def _route_causal_conv(gdn):
    from ._gdn_op import _causal_depthwise_conv
    for space in _namespaces(type(gdn).forward):
        fn = space.get("causal_conv1d_fn")
        if not callable(fn) or getattr(fn, "_unsloth_block_adapter", False):
            continue
        probe_defaults = (
            getattr(fn, "__kwdefaults__", None)
            if getattr(fn, "_unsloth_varlen_probe_of", None) is not None
            else None
        )
        if probe_defaults is not None:
            # Packing's varlen probe stays outermost so its boundary handshake still sees the call.
            inner = probe_defaults.get("_fn")
            if callable(inner) and not getattr(inner, "_unsloth_block_adapter", False):
                probe_defaults["_fn"] = _causal_conv1d_adapter(inner, _causal_depthwise_conv)
        else:
            space["causal_conv1d_fn"] = _causal_conv1d_adapter(fn, _causal_depthwise_conv)
        alias = _alias_namespace(space)
        if alias is not None and callable(alias.get("causal_conv1d_fn")):
            # Dynamo guards the alias binding: it must be the object the forward actually calls.
            alias["causal_conv1d_fn"] = space["causal_conv1d_fn"]


def _route_sm_count(modules = None):
    # ROCm: Triton's hip_utils.get_device_properties is untraceable; the SM count is per device.
    for name, module in list((sys.modules if modules is None else modules).items()):
        if module is None or "fla." not in name:
            continue
        fn = getattr(module, "get_multiprocessor_count", None)
        if not callable(fn) or getattr(fn, "_unsloth_block_adapter", False):
            continue
        try:
            counts = {i: int(fn(i)) for i in range(torch.cuda.device_count())}
        except Exception:
            continue

        @functools.wraps(fn, updated = ())
        def get_multiprocessor_count(
            tensor_idx = 0,
            _original = fn,
            _counts = counts,
        ):
            count = _counts.get(tensor_idx)
            return _original(tensor_idx) if count is None else count

        get_multiprocessor_count._unsloth_block_adapter = True
        module.get_multiprocessor_count = get_multiprocessor_count


def _prepare_gdn(layer, hidden_states):
    """Self-check fla's opaque op on this layer's shapes, then point the GatedDeltaNet forward's
    chunk kernel at the adapter. Runs eagerly, before the first trace."""
    gdn = getattr(layer, "linear_attn", None)
    if gdn is None or _GDN_OP[0] is not None:
        return
    _GDN_OP[0] = False
    try:
        _route_sm_count()
    except Exception:
        pass
    try:
        from ._gdn_op import _fla_ops, varlen_ok

        heads, dk, dv = gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim
        # transformers >= 5.23 passes q / k unrepeated (huggingface/transformers#49141).
        qk_heads = getattr(gdn, "num_k_heads", heads)
        if not qk_heads or heads % qk_heads:
            qk_heads = heads
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        dev, dt = hidden_states.device, hidden_states.dtype

        def make_inputs():
            gen = torch.Generator(device = dev).manual_seed(0)
            r = lambda *s, dtype = dt: torch.randn(*s, device = dev, dtype = dtype, generator = gen)
            g = -torch.rand(B, T, heads, device = dev, generator = gen) * 0.1
            return (
                r(B, T, qk_heads, dk),
                r(B, T, qk_heads, dk),
                r(B, T, heads, dv),
                g,
                r(B, T, heads).sigmoid(),
            )

        if not _fla_ops(make_inputs):
            return
        _GDN_VARLEN[0] = varlen_ok()
        _GDN_OP[0] = _route_fla_kernel() > 0
        if _GDN_OP[0]:
            _route_causal_conv(gdn)
    except Exception:
        _GDN_OP[0] = False


# A fresh class per PEFT forward would change the type id Dynamo guards, so each module keeps one.
_PARAMETRIZED_CLASSES = weakref.WeakKeyDictionary()


def _stable_parametrized_classes():
    try:
        from torch.nn.utils import parametrize
    except Exception:
        return
    original = getattr(parametrize, "_inject_new_class", None)
    if original is None or getattr(original, "_unsloth_stable", False):
        return

    # Eager: the swap is the graph break PEFT already takes; Dynamo must not fold the lookup.
    @torch.compiler.disable
    @functools.wraps(original)
    def _inject_new_class(module):
        cls = module.__class__
        try:
            cached = _PARAMETRIZED_CLASSES.get(module)
        except TypeError:  # not weak-referenceable / hashable
            return original(module)
        if cached is not None and cached.__bases__ == (cls,):
            module.__class__ = cached
            return
        original(module)
        try:
            _PARAMETRIZED_CLASSES[module] = module.__class__
        except TypeError:
            pass

    _inject_new_class._unsloth_stable = True
    parametrize._inject_new_class = _inject_new_class


# zoo's Gemma-4 sliding router scans the mask (data dependent) and declines while compiling, so
# the block runs the scan eagerly and a router wrapper reads the verdict stored on the mask.
_BAND_MODULE = "unsloth_zoo.temporary_patches.gemma4_banded_attention"
_FLASH_MODULE = "unsloth_zoo.temporary_patches.gemma4_flash_sliding"
# Read by unsloth/kernels/large_head_attention.py (PLAIN_CAUSAL_ATTR): a compiled head_dim 512 layer
# takes the kernel over a causal mask the block verified here, instead of masked SDPA.
_PLAIN_CAUSAL_ATTR = "_unsloth_plain_causal"
_BAND_CHECK = [None]


def _gemma4_router_hook():
    """Sliding layers: a plain causal band within the window runs is_causal SDPA, longer bands run
    masked SDPA. Router stays untraced: its counters drive a branch dynamo guards on; traced
    banded kernel breaks checkpoints"""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    router = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if not getattr(router, "_unsloth_gemma4_flash", False) or getattr(
        router, "_unsloth_block_band", False
    ):
        return
    # Other routers wrap zoo's (unsloth's head_dim 512 kernel, installed at load before this runs, has no
    # _ORIG_SDPA), so fall back to zoo's module: without it the hook never installed for Gemma-4.
    plain = None
    for name in (router.__module__, _FLASH_MODULE):
        plain = getattr(sys.modules.get(name), "_ORIG_SDPA", [None])[0]
        if plain is not None:
            break
    check = getattr(sys.modules.get(_BAND_MODULE), "_mask_is_plain_band", None)
    if plain is None or not callable(check):
        return
    _BAND_CHECK[0] = check

    # wraps() keeps zoo's marker, else a later load wraps this as its own fallback (endless loop).
    @functools.wraps(router)
    def sdpa_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        dropout = 0.0,
        scaling = None,
        is_causal = None,
        **kwargs,
    ):
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and getattr(module, "is_sliding", False)
            and type(module).__name__.startswith("Gemma4")
            and query.dim() == 4
            and query.shape[-1] <= 256
        ):
            window = kwargs.get("sliding_window", None) or getattr(module, "sliding_window", None)
            causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
            if (
                window
                and query.shape[2] == key.shape[2]
                and query.shape[2] <= window
                and (
                    causal
                    if attention_mask is None
                    else getattr(attention_mask, "_unsloth_block_band", False) is True
                )
            ):
                out = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p = dropout if module.training else 0.0,
                    is_causal = True,
                    scale = scaling,
                    enable_gqa = True,
                )
                return out.transpose(1, 2).contiguous(), None
            if (
                window
                and attention_mask is not None
                and query.shape[2] == key.shape[2]
                and query.shape[2] > window
                and not (dropout and module.training)
                and kwargs.get("softcap", None) is None
                and getattr(attention_mask, _LONG_BAND_ATTR, False) is True
                and query.dtype == key.dtype == value.dtype
                and query.dtype in (torch.float16, torch.bfloat16)
                and query.shape[-1] == key.shape[-1] == value.shape[-1]
                and query.shape[1] % key.shape[1] == 0
                and _band_kernel_checked(query)
                and _band_offsets_fit(query, key, value)
            ):
                # Longer than the window: masked SDPA computes all S x S scores (and repeats the kv
                # heads); the interval kernel reads only each row's window (O(S * w)).
                return _band_kernel_attention(query, key, value, window, scaling), None
            return plain(
                module,
                query,
                key,
                value,
                attention_mask,
                dropout = dropout,
                scaling = scaling,
                is_causal = is_causal,
                **kwargs,
            )
        return router(
            module,
            query,
            key,
            value,
            attention_mask,
            dropout = dropout,
            scaling = scaling,
            is_causal = is_causal,
            **kwargs,
        )

    sdpa_attention_forward._unsloth_block_band = True
    ALL_ATTENTION_FUNCTIONS["sdpa"] = sdpa_attention_forward


# Set on a sliding layer's 4D mask when it is the plain band, the sequence is longer than the window and
# the interval kernel passed its self check at this head_dim / dtype; read by the router above.
_LONG_BAND_ATTR = "_unsloth_block_long_band"
_BAND_KERNEL_ENV = "UNSLOTH_BLOCK_SLIDING_KERNEL"
_LH_MODULE = "unsloth.kernels.large_head_attention"


def _band_offsets_fit(query, key, value):
    lh = sys.modules.get(_LH_MODULE)
    if lh is None:  # loaded by _band_kernel_checked whenever the kernel can run
        return True
    limit = lh._INT32_MAX
    if max(query.numel(), key.numel(), value.numel()) > limit:
        return False
    return max(lh._row_span(query), lh._row_span(key), lh._row_span(value)) <= limit


def _band_kernel_attention(query, key, value, window, scaling):
    lh = sys.modules[_LH_MODULE]
    B, S = query.shape[0], query.shape[2]
    rows = torch.arange(S, device = query.device, dtype = torch.int32)
    lo = (rows - (window - 1)).clamp_min(0).expand(B, S).contiguous()
    hi = rows.expand(B, S).contiguous()
    scale = scaling if scaling is not None else query.shape[-1] ** -0.5
    return lh.large_head_attention(query, key, value, lo, hi, scale).contiguous()


def _band_kernel_ready(layer, hidden_states):
    """Eager: the interval kernel may serve this sliding layer (CUDA sm80+, passed its self check at the
    layer's head_dim). Checked per dtype; the router also reads the verdict for the call's dtype."""
    if os.environ.get(_BAND_KERNEL_ENV, "1") == "0" or hidden_states is None:
        return False
    if not hidden_states.is_cuda or torch.version.hip is not None:
        return False
    key = str(hidden_states.device)
    cached = layer.__dict__.get("_unsloth_band_kernel_ready")
    if cached is not None and cached[0] == key:
        return cached[1]
    ready = _band_kernel_check(layer, hidden_states.device)
    layer.__dict__["_unsloth_band_kernel_ready"] = (key, ready)
    return ready


def _band_kernel_check(layer, device):
    head_dim = None
    for module in layer.modules():
        if getattr(module, "is_sliding", False) and getattr(module, "sliding_window", None):
            head_dim = getattr(module, "head_dim", None)
            break
    if head_dim not in (64, 128, 256):
        return False
    try:
        from unsloth.kernels import large_head_attention as lh

        if not lh._enabled() or not lh._measured_faster_on(device):
            return False
        dtypes = [torch.float16] + ([torch.bfloat16] if torch.cuda.is_bf16_supported() else [])
        return any([bool(lh._self_check(device, dtype, head_dim)) for dtype in dtypes])
    except Exception:
        return False


def _band_kernel_checked(query):
    lh = sys.modules.get(_LH_MODULE)
    if lh is None:
        return False
    return lh._SELF_CHECK.get((str(query.device), query.dtype, query.shape[-1]), False) is True


def _sliding_window(layer):
    window = layer.__dict__.get("_unsloth_block_window", 0)
    if window == 0:
        window = None
        for module in layer.modules():
            if getattr(module, "is_sliding", False) and getattr(module, "sliding_window", None):
                window = module.sliding_window
                break
        layer.__dict__["_unsloth_block_window"] = window
    return window


def _precheck_band(
    layer,
    seq_len,
    args,
    kwargs,
    hidden_states = None,
):
    # Every 4D mask gets a verdict: a missing attribute would recompile.
    check = _BAND_CHECK[0]
    if check is None:
        return
    window = _sliding_window(layer)
    for obj in (*args, *kwargs.values()):
        if isinstance(obj, torch.Tensor) and obj.dim() == 4 and not obj.requires_grad:
            if window is not None and seq_len <= window:
                try:
                    obj._unsloth_block_band = bool(check(obj, seq_len, window))
                except Exception:
                    obj._unsloth_block_band = False
            elif not hasattr(obj, "_unsloth_block_band"):
                obj._unsloth_block_band = False
            # Full-attention layers: a band as wide as the sequence is the plain causal mask
            # (scan cached on the mask by version, one per forward for every global layer).
            if window is None:
                try:
                    causal = bool(check(obj, seq_len, seq_len))
                except Exception:
                    causal = False
                setattr(obj, _PLAIN_CAUSAL_ATTR, causal)
            elif not hasattr(obj, _PLAIN_CAUSAL_ATTR):
                setattr(obj, _PLAIN_CAUSAL_ATTR, False)
            if window is not None:
                long_band = False
                if seq_len > window and _band_kernel_ready(layer, hidden_states):
                    try:
                        long_band = bool(check(obj, seq_len, window))
                    except Exception:
                        long_band = False
                setattr(obj, _LONG_BAND_ATTR, long_band)
            elif not hasattr(obj, _LONG_BAND_ATTR):
                setattr(obj, _LONG_BAND_ATTR, False)


# transformers drops the mask (None) for unpadded micro batches and dynamo guards on mask vs None, so
# each flip recompiles. Once a layer has seen a mask, rebuild None in that layout (same SDPA result).
_MASK_KWARG = "attention_mask"
_MASK_SPEC = "_unsloth_block_mask_spec"
_MASK_FLAGS = "_unsloth_block_mask_flags"
_MATERIALISED_ATTR = "_unsloth_block_materialised"
_MASK_FLAG_ATTRS = ("_unsloth_block_band", _PLAIN_CAUSAL_ATTR, _LONG_BAND_ATTR)
# Shared rebuilt mask; held strongly only up to this size.
_MATERIALISED = [None]
_MATERIALISED_KEEP_BYTES = 1 << 28


def _mask_layout(mask, seq_len):
    if not isinstance(mask, torch.Tensor) or mask.requires_grad or mask.is_floating_point():
        return None
    if mask.is_inference() or getattr(mask, "is_nested", False) or mask.shape[-1] != seq_len:
        return None
    if mask.dim() == 2:
        heads = None
    elif mask.dim() == 4 and mask.dtype == torch.bool and mask.shape[-2] == seq_len:
        heads = mask.shape[1]
    else:
        return None
    shape, stride = mask.shape, mask.stride()
    broadcast = tuple(i for i in range(mask.dim()) if stride[i] == 0 and shape[i] > 1)
    expected, step = [0] * mask.dim(), 1
    for i in reversed(range(mask.dim())):
        if i in broadcast:
            continue
        expected[i] = step
        step *= shape[i]
    if any(shape[i] > 1 and stride[i] != expected[i] for i in range(mask.dim())):
        return None
    # Dynamo also guards a view's base (transformers expands/slices its masks).
    view = mask._base is not None
    if view:
        base = mask._base
        stored = [1 if i in broadcast else shape[i] for i in range(mask.dim())]
        if list(base.shape) != stored or not base.is_contiguous():
            return None
    elif broadcast:
        return None
    return (mask.dim(), mask.dtype, heads, broadcast, view)


def _layer_is_causal(layer):
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None)
    causal = getattr(attn, "is_causal", None)
    return causal if isinstance(causal, bool) else None


def _materialise_mask(layout, causal, batch, seq_len, device):
    dim, dtype, heads, broadcast, view = layout
    key = (layout, causal, batch, seq_len, device)
    cached = _MATERIALISED[0]
    if cached is not None and cached[0] == key:
        mask = cached[1] if cached[2] else cached[1]()
        if mask is not None:
            return mask
    full = (batch, seq_len) if dim == 2 else (batch, heads, seq_len, seq_len)
    stored = tuple(1 if i in broadcast else size for i, size in enumerate(full))
    if dim == 4 and causal:
        # A causal mask needs both token dims stored; a broadcast one (padding-only layout) keeps None.
        if 2 in broadcast or 3 in broadcast:
            return None
        # A fresh tensor in the stored shape (contiguous() keeps a (1, 1, L, L) expand as a view of the 2D base).
        mask = torch.empty(stored, dtype = torch.bool, device = device)
        mask.copy_(torch.ones((seq_len, seq_len), dtype = torch.bool, device = device).tril_())
    else:
        mask = torch.ones(stored, dtype = dtype if dim == 2 else torch.bool, device = device)
    if view:
        mask = mask.expand(full)
    setattr(mask, _MATERIALISED_ATTR, True)
    keep = mask.untyped_storage().nbytes() <= _MATERIALISED_KEEP_BYTES
    _MATERIALISED[0] = (key, mask if keep else weakref.ref(mask), keep)
    return mask


def _normalise_mask(layer, hidden_states, kwargs):
    if _MASK_KWARG not in kwargs or hidden_states.dim() != 3:
        return kwargs
    mask = kwargs[_MASK_KWARG]
    if mask is not None:
        if not getattr(mask, _MATERIALISED_ATTR, False):
            layout = _mask_layout(mask, hidden_states.shape[1])
            if layout is not None:
                layer.__dict__[_MASK_SPEC] = layout
            else:
                layer.__dict__.pop(_MASK_SPEC, None)
        return kwargs
    layout = layer.__dict__.get(_MASK_SPEC)
    if layout is None or kwargs.get("is_causal") is not None:
        return kwargs
    causal = _layer_is_causal(layer) if layout[0] == 4 else None
    if layout[0] == 4 and causal is None:
        return kwargs
    mask = _materialise_mask(
        layout, causal, hidden_states.shape[0], hidden_states.shape[1], hidden_states.device
    )
    return kwargs if mask is None else {**kwargs, _MASK_KWARG: mask}


def _settle_mask_flags(layer, kwargs):
    # Dynamo guards on fast-path flags: keep one only where the last real mask had it.
    mask = kwargs.get(_MASK_KWARG)
    if not isinstance(mask, torch.Tensor):
        return
    if not getattr(mask, _MATERIALISED_ATTR, False):
        layer.__dict__[_MASK_FLAGS] = tuple(getattr(mask, name, None) for name in _MASK_FLAG_ATTRS)
        return
    flags = layer.__dict__.get(_MASK_FLAGS)
    for i, name in enumerate(_MASK_FLAG_ATTRS):
        if hasattr(mask, name):
            prior = flags[i] if flags is not None else None
            setattr(mask, name, bool(getattr(mask, name)) and prior is True)


# Mode of the block being traced (_SPLIT: flex_attention outside); dynamo guards on it.
_IN_BLOCK = [0]
# True only inside Unsloth's own compiled block (fullgraph=False, a graph break is legal), not when the
# block is inlined into a user's torch.compile, which may be fullgraph=True. Read by unsloth_zoo.
_IN_COMPILED_REGION = [False]
_PACKED_MODULE = "unsloth.utils.hf_packed_attention"
_DISPATCH_MODULE = "unsloth.utils.attention_dispatch"
_PACKED_BLOCK_KWARG = "_unsloth_packed_varlen"
_NO_PACKED = [False]
_OFF, _FULL, _SPLIT = 0, 1, 2
_DEMOTION = (_FULL, _SPLIT, _OFF)
_FIRST_MODE = _SPLIT if os.environ.get("UNSLOTH_BLOCK_FLEX", "inside") == "outside" else _FULL


def _training_flex_options(options):
    """Force the main flex kernel: Inductor's decoding kernel for queries < 128 fails to lower
    (GQA, head_dim 256), demoting every block. A caller's explicit choice is kept."""
    if options is None:
        return {"FORCE_USE_FLEX_ATTENTION": True}
    # torch 2.10+ rejects BACKEND together with the legacy FORCE_USE_FLEX_ATTENTION.
    if isinstance(options, dict) and not ({"FORCE_USE_FLEX_ATTENTION", "BACKEND"} & options.keys()):
        return {**options, "FORCE_USE_FLEX_ATTENTION": True}
    return options


# Memory-efficient SDPA backward ops (masked training calls).
_EFFICIENT_SDPA_BACKWARD = (
    "_scaled_dot_product_efficient_attention_backward",
    "_efficient_attention_backward",
)


def _efficient_backward_keeps_guard():
    """CUDA torch < 2.14: without torch's (always true) guard the efficient backward copies the bias densely
    and faults on padded lengths not a multiple of 8 (`attn_bias is not correctly aligned`)."""
    from packaging.version import Version
    return torch.version.hip is None and Version(torch.__version__.split("+")[0]) < Version("2.14")


# Cache keys do not see the lowering, so graphs cached without the guard must be invalidated.
_EFFICIENT_BACKWARD_CACHE_TAG = "unsloth/efficient-backward-guard"


def _tag_efficient_backward_cache():
    config = getattr(torch.compiler, "config", None)
    if config is None or not isinstance(getattr(config, "cache_key_tag", None), str):
        return False
    tags = [t for t in config.cache_key_tag.split(",") if t]
    if _EFFICIENT_BACKWARD_CACHE_TAG in tags:
        return True
    value = ",".join(tags + [_EFFICIENT_BACKWARD_CACHE_TAG])
    try:
        from ..kernels.triton_launch import set_torch_config, sync_compile_cache_tag_to_workers
    except Exception:
        setattr(config, "cache_key_tag", value)
        return True
    # Every thread: a reentrant checkpoint recompiles its grad graph on autograd's device thread.
    set_torch_config(config, "cache_key_tag", value)
    sync_compile_cache_tag_to_workers()
    return True


def _sdpa_alignment_without_guard():
    """Stop Inductor recompiling a block per `seq_len % 8` class: inside torch 2.14's SDPA lowering only,
    `guard_or_false` becomes `statically_known_true` (False is its documented safe answer), so a dynamic length
    takes the general unaligned path instead of installing a `Mod(s, 8)` guard (20-44 s recompile per layer kind).
    The memory-efficient backward keeps torch's guard where it needs it (_efficient_backward_keeps_guard).
    UNSLOTH_BLOCK_SDPA_ALIGN_GUARD=1 keeps torch's guards."""
    if os.environ.get("UNSLOTH_BLOCK_SDPA_ALIGN_GUARD", "0") == "1":
        if _efficient_backward_keeps_guard():
            # Stock lowering, but the same cache key as a run without the guard before the tag existed:
            # tag it too, or that run's faulting efficient backward is served again.
            _tag_efficient_backward_cache()
        return False
    try:
        import torch._inductor.lowering as lowering
        from torch._inductor.sizevars import SizeVarAllocator
        from torch._inductor.virtualized import V
    except Exception:
        return False
    original = getattr(lowering, "sdpa_constraint", None)
    table = getattr(lowering, "_maybe_layout_constraints", None)
    if original is None or not isinstance(table, dict):
        return False
    if getattr(original, "_unsloth_no_align_guard", False):
        return True
    if not hasattr(SizeVarAllocator, "statically_known_true") or not hasattr(
        SizeVarAllocator, "guard_or_false"
    ):
        return False
    guarded = set()
    if _efficient_backward_keeps_guard():
        for name in _EFFICIENT_SDPA_BACKWARD:
            packet = getattr(torch.ops.aten, name, None)
            if packet is not None and hasattr(packet, "default"):
                guarded.add(packet.default)
        _tag_efficient_backward_cache()

    @functools.wraps(original)
    def sdpa_constraint(*args, **kwargs):
        if args and getattr(args[0], "target", None) in guarded:
            return original(*args, **kwargs)
        sizevars = V.graph.sizevars
        shadowed = "guard_or_false" in sizevars.__dict__
        previous = sizevars.__dict__.get("guard_or_false")
        sizevars.guard_or_false = sizevars.statically_known_true
        try:
            return original(*args, **kwargs)
        finally:
            if shadowed:
                sizevars.guard_or_false = previous
            else:
                del sizevars.guard_or_false

    sdpa_constraint._unsloth_no_align_guard = True
    replaced = [op for op, fn in table.items() if fn is original]
    for op in replaced:
        table[op] = sdpa_constraint
    lowering.sdpa_constraint = sdpa_constraint
    return bool(replaced)


def _flex_split_hook():
    """Lets a block keep flex_attention out of its graph. Some masks do not lower inside a larger
    graph (torch 2.13, Gemma-4 12B blockwise mask: `unsupported operand type(s) for &:
    'FloorDiv' and 'Add'`); flex then runs as its own compiled region, one break per layer."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    flex = ALL_ATTENTION_FUNCTIONS.get("flex_attention")
    if flex is None or getattr(flex, "_unsloth_block_split", False):
        return
    outside = torch.compiler.disable(flex)

    @functools.wraps(flex)
    def flex_attention_forward(*args, **kwargs):
        if _IN_BLOCK[0] != _OFF:
            kwargs["kernel_options"] = _training_flex_options(kwargs.get("kernel_options"))
        if _IN_BLOCK[0] == _SPLIT and torch.compiler.is_compiling():
            return outside(*args, **kwargs)
        return flex(*args, **kwargs)

    flex_attention_forward._unsloth_block_split = True
    ALL_ATTENTION_FUNCTIONS["flex_attention"] = flex_attention_forward


def _nf4_in_blocks():
    """bitsandbytes NF4 Linear4bit inside a block: Unsloth's Triton dequantize + F.linear (bit-exact
    to bitsandbytes' dequantize; kernels/bnb_override.py), on every GPU. Outside blocks the
    override keeps its own per-GPU default."""
    try:
        import bitsandbytes as bnb

        from ..kernels import bnb_override as O
    except Exception:
        return
    if not O._TRACE or O._LINEAR == "0":
        return  # torch < 2.11 miscompiles the traced backward; "0" = the user's opt-out
    Linear4bit = bnb.nn.Linear4bit
    current = Linear4bit.forward
    if getattr(current, "_unsloth_block_nf4", False):
        return

    @functools.wraps(current)
    def forward(self, x, *args, **kwargs):
        # Both checkpoint halves must take one kernel: a trainable bias fails _eligible only under
        # grad, and one row picks fast_gemv only without grad (O._linear).
        bias = self.bias
        rows = x.numel() // max(x.shape[-1], 1)
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
            and not args
            and not kwargs
            and (bias is None or not bias.requires_grad)
            and rows > 1
        ):
            weight = self.weight
            quant_state = getattr(weight, "quant_state", None)
            if weight.dtype is not torch.uint8:
                # FSDP2 QLoRA packs the NF4 bytes as floats; the kernels read the uint8 view.
                weight = O._float_packed_as_uint8(weight)
            if (
                O._eligible(self, x, weight, quant_state)
                and (quant_state.dtype is not torch.bfloat16 or O._TRACE_BF16)
                # bitsandbytes' fp32 small-row kernel is more accurate (O._forward keeps it too).
                and not (quant_state.dtype is torch.float32 and rows < 8)
            ):
                dtype = quant_state.dtype
                bias = None if self.bias is None else self.bias.to(dtype)
                return O._linear(x.to(dtype), weight, quant_state, bias).to(x.dtype)
        return current(self, x, *args, **kwargs)

    forward._unsloth_block_nf4 = True
    Linear4bit.forward = forward


_CACHE_KWARGS = ("past_key_values", "past_key_value", "cache_params", "layer_past")


def _has_cache(args, kwargs):
    # transformers 4.x layers take the singular `past_key_value`, sometimes positionally.
    if any(kwargs.get(key) is not None for key in _CACHE_KWARGS):
        return True
    try:
        from transformers.cache_utils import Cache
    except Exception:
        return False
    return any(isinstance(arg, Cache) for arg in args)


def _shared_kv_scratch(layer, kwargs):
    """KV-storing layers get an empty `shared_kv_states` (copied back): dynamo guards on it being empty and a recompute sees it filled."""
    shared = kwargs.get("shared_kv_states")
    if shared is None or not hasattr(shared, "update"):
        return kwargs, None
    attn = getattr(layer, "self_attn", None)
    # Readers keep the real dict.
    if attn is None or getattr(attn, "is_kv_shared_layer", False):
        return kwargs, None
    try:
        scratch = type(shared)()
    except Exception:
        return kwargs, None
    return {**kwargs, "shared_kv_states": scratch}, (shared, scratch)


def _layer_kind(layer):
    attn = (
        getattr(layer, "self_attn", None)
        or getattr(layer, "linear_attn", None)
        or getattr(layer, "cross_attn", None)
        or getattr(layer, "attention", None)
        or getattr(layer, "mixer", None)
    )
    return (
        type(attn).__name__,
        getattr(layer, "attention_type", None),
        getattr(attn, "is_sliding", None),
        getattr(attn, "sliding_window", None),
        getattr(attn, "head_dim", None),
    )


def _is_hf_checkpoint_wrapper(fn):
    """transformers >= 5.23 checkpoints in a wrapper GradientCheckpointingLayer.__init_subclass__ puts on
    each layer class's `forward` (it was `__call__`); it checkpoints only while it is `type(self).forward`."""
    code = getattr(fn, "__code__", None)
    return (
        code is not None
        and code.co_name == "forward_with_checkpointing"
        and callable(getattr(fn, "__wrapped__", None))
    )


# The cache arguments transformers' wrapper clears before it checkpoints (unless _can_checkpoint_with_cache).
_CHECKPOINT_CACHE_KWARGS = ("past_key_values", "layer_past")


def _checkpointing(forward):
    """The block's forward hides transformers' checkpoint wrapper, so layers ran without checkpointing."""

    @functools.wraps(forward)
    def forward_with_checkpointing(self, *args, **kwargs):
        outermost = type(self).forward is forward_with_checkpointing or (
            getattr(self.__dict__.get("forward"), "__func__", None) is forward_with_checkpointing
        )
        if self.gradient_checkpointing and self.training and outermost:
            if kwargs.get("use_cache"):
                kwargs["use_cache"] = False
            if not getattr(self, "_can_checkpoint_with_cache", False):
                for name in _CHECKPOINT_CACHE_KWARGS:
                    if kwargs.get(name) is not None:
                        kwargs[name] = None
            return self._gradient_checkpointing_func(
                functools.partial(forward, self, **kwargs), *args
            )
        return forward(self, *args, **kwargs)

    return forward_with_checkpointing


def _wrap(
    cls,
    compile_options,
    kinds = 1,
    *,
    install = True,
):
    original = getattr(cls.forward, "_unsloth_block_original", cls.forward)
    checkpoint_outside = getattr(cls.forward, "_unsloth_block_checkpoint_outside", False)
    if _is_hf_checkpoint_wrapper(original):
        # Compile the layer's own forward; checkpointing goes back outside the block (below).
        original, checkpoint_outside = original.__wrapped__, True
    # Host-read packed lengths / PrefixGrouper segments stay stock.
    unsloth_forward = getattr(original, "__module__", "").startswith("unsloth.models.")
    try:
        takes_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD
            for p in inspect.signature(original).parameters.values()
        )
    except (TypeError, ValueError):
        takes_kwargs = False
    isolate = {}
    if _ISOLATE_RECOMPILES:
        isolate = dict(isolate_recompiles = True, recompile_limit = _region_recompile_limit(kinds))
    graphs = bool(compile_options.get("triton.cudagraphs"))
    backend = _block_backend(compile_options) if graphs else None
    owned = {} if backend is None else {"backend": backend}
    compiled = torch.compile(
        original, fullgraph = False, dynamic = None, options = compile_options, **owned, **isolate
    )
    try:
        from unsloth_zoo.temporary_patches.utils import UNSLOTH_DECODE_COMPILE as inline_flag
    except Exception:
        inline_flag = [False]
    try:
        from unsloth_zoo.temporary_patches.misc import output_capture_scratch as capture_scratch
    except Exception:
        capture_scratch = contextlib.nullcontext
    try:
        scratch_takes_owner = "owner" in inspect.signature(capture_scratch).parameters
    except (TypeError, ValueError):
        scratch_takes_owner = False
    # First call per grad mode compiles with errors raised: on failure retry with flex outside,
    # then the stock forward. The mode is per layer (a checkpoint recompute must not switch graph);
    # a failure is also the starting mode for later layers of the same kind.
    state = {"reported": set(), "start": {}}
    # Under Unsloth checkpointing each block call is a new cudagraph generation: the next replay
    # overwrites this block's outputs and gradients, so they are cloned out of the pool.
    # Auto: `compiled` stays the graphs-off region; the graphed one is compiled once graphs turn on.
    auto = not graphs and _CUDAGRAPH_MODE == "auto"
    if graphs:
        plain = torch.compile(
            original,
            fullgraph = False,
            dynamic = None,
            options = {**compile_options, "triton.cudagraphs": False},
            **isolate,
        )
    else:
        plain = compiled
    graphed_box = [compiled if graphs else None]

    def graphed():
        if graphed_box[0] is None:
            options = {**compile_options, "triton.cudagraphs": True}
            backend = _block_backend(options)
            owned = {} if backend is None else {"backend": backend}
            graphed_box[0] = torch.compile(
                original,
                fullgraph = False,
                dynamic = None,
                options = options,
                **owned,
                **isolate,
            )
        return graphed_box[0]

    graph_shapes = set()

    def run(self, hidden_states, args, kwargs, mode):
        kwargs, shared = _shared_kv_scratch(self, kwargs)
        out = _run(self, hidden_states, args, kwargs, mode)
        if shared is not None:
            shared[0].update(shared[1])
        return out

    def _run(self, hidden_states, args, kwargs, mode):
        previous = inline_flag[0]
        inline_flag[0] = True
        _IN_BLOCK[0] = mode
        _IN_COMPILED_REGION[0] = True
        packed_module = sys.modules.get(_PACKED_MODULE)
        packed = getattr(packed_module, "IN_COMPILED_BLOCK", _NO_PACKED)
        packed_previous = packed[0]
        packed[0] = True
        try:
            if (
                takes_kwargs
                and packed_module is not None
                and kwargs.get("packed_seq_lengths") is not None
            ):
                varlen = packed_module.block_varlen_kwargs(self, hidden_states, kwargs)
                if varlen is not None:
                    kwargs = {**kwargs, packed_module.BLOCK_VARLEN_KWARG: varlen}
            # Capture hooks (router_logits) append to empty scratch lists: no guard on the layer index.
            scratch = capture_scratch(owner = self) if scratch_takes_owner else capture_scratch()
            with scratch, _no_ddp_graph_split():
                use_graphs = graphs
                if auto and _CUDAGRAPH_STATE["auto"]["phase"] != "off":
                    verdict = _auto_observe(self, hidden_states, args, kwargs)
                    if verdict == "measure":
                        return _auto_measure(
                            lambda: compiled(self, hidden_states, *args, **kwargs), hidden_states
                        )
                    use_graphs = verdict == "graphs"
                if use_graphs and _CUDAGRAPH_STATE["off"] is None:
                    shape = tuple(hidden_states.shape)
                    if _packed_kwargs(kwargs):
                        _graphs_off("packed", _PACKED_NOTE)
                    elif shape not in graph_shapes:
                        if len(graph_shapes) >= _CUDAGRAPH_MAX_SHAPES:
                            _graphs_off("shapes", _SHAPES_NOTE.format(_CUDAGRAPH_MAX_SHAPES))
                        else:
                            graph_shapes.add(shape)
                # Before the graphs_off read: the observer may set or clear it for this block.
                flex = use_graphs and _note_block_mask_call(self, args, kwargs)
                if use_graphs and (
                    _CUDAGRAPH_STATE["off"] is not None
                    or self.__dict__.get("_unsloth_block_graphs_off", False)
                    or _in_nonreentrant_checkpoint()
                    or _graphs_unsafe_moe(self)
                    or flex
                    or _grad_side_inputs(args, kwargs)
                ):
                    if _CUDAGRAPH_STATE["off"] is not None and _CUDAGRAPH_STATE["release"] is None:
                        _release_graph_pool()
                    return plain(self, hidden_states, *args, **kwargs)
                if use_graphs:
                    _param_grads_out_of_pool(self)
                    device = hidden_states.device.index
                    if _pool_outputs_stay_valid(self, hidden_states):
                        if _input_grad_must_leave_pool(hidden_states):
                            # A leaf's .grad outlives the generation (trainable inputs_embeds); a
                            # checkpoint's backward recomputes (a new generation) before reading it.
                            hidden_states = _CloneGrad.apply(hidden_states)
                        return _graphed_call(
                            graphed(), device, self, hidden_states, *args, **kwargs
                        )
                    if hidden_states.requires_grad and torch.is_grad_enabled():
                        hidden_states = _CloneGrad.apply(hidden_states)
                    return _clone_outputs(
                        _graphed_call(graphed(), device, self, hidden_states, *args, **kwargs)
                    )
                return compiled(self, hidden_states, *args, **kwargs)
        finally:
            inline_flag[0] = previous
            _IN_BLOCK[0] = _OFF
            _IN_COMPILED_REGION[0] = False
            packed[0] = packed_previous

    def _inline(self, hidden_states, args, kwargs, mode):
        previous = inline_flag[0]
        inline_flag[0] = True
        _IN_BLOCK[0] = mode
        try:
            return original(self, hidden_states, *args, **kwargs)
        finally:
            inline_flag[0] = previous
            _IN_BLOCK[0] = _OFF

    @functools.wraps(original)
    def forward(self, hidden_states, *args, **kwargs):
        tracing = torch.compiler.is_compiling()
        mode = self.__dict__.get("_unsloth_block_mode")
        if mode is None:
            mode = state["start"].get(_layer_kind(self), _FIRST_MODE)
            if not tracing:
                self.__dict__["_unsloth_block_mode"] = mode
        # Training only: eval logits stay bit-equal to transformers.
        stock = (
            mode == _OFF
            or not self.training
            or _has_cache(args, kwargs)
            or not _enabled()
            or _forced_float32(self)
            or _zero3_partitioned(self)
            # PrefixGrouper's int segment table would recompile at every new GRPO length.
            or kwargs.get("prefix_seg_info") is not None
        )
        if not stock and unsloth_forward and kwargs.get("packed_seq_lengths") is not None:
            # Packed lengths -> device cu_seqlens; no traceable varlen kernel keeps the stock forward.
            prepare = getattr(sys.modules.get(_DISPATCH_MODULE), "prepare_packed_block", None)
            prepared = None if tracing or prepare is None else prepare(self, hidden_states, kwargs)
            if prepared is None:
                stock = True
            else:
                kwargs = prepared
        if stock:
            phase = _CUDAGRAPH_STATE["auto"]["phase"] if auto else None
            if (
                (graphs or phase == "on")
                and self.training
                and _CUDAGRAPH_STATE["off"] is None
                and _packed_kwargs(kwargs)
            ):
                _graphs_off("packed", _PACKED_NOTE)
            elif phase == "probe" and self.training and _packed_kwargs(kwargs):
                _auto_off(_CUDAGRAPH_STATE["auto"], "packed / padding-free rows")
            if (
                unsloth_forward
                and self.training
                and not tracing
                and kwargs.get("packed_seq_lengths") is not None
            ):
                _rank0_note("packed_stock", _PACKED_STOCK_NOTE)
            return original(self, hidden_states, *args, **kwargs)
        if tracing:
            # Inside a user's torch.compile the enclosing graph owns the layer: no dynamic marking
            # (forbidden while tracing) and no nested compiled call, but the same in-block state.
            if _eager_setup_pending(self):
                _eager_setup(self, hidden_states)
            return _inline(self, hidden_states, args, kwargs, mode)
        if not self.__dict__.get("_unsloth_block_moe_done", False):
            _experts_outside(self)
        if _GDN_OP[0] is None and hasattr(self, "linear_attn"):
            _prepare_gdn(self, hidden_states)
        if hidden_states.dim() == 3 and hidden_states.shape[1] > 1:
            kwargs = _normalise_mask(self, hidden_states, kwargs)
            hidden_states, args, kwargs = _relax_strict_dynamic((hidden_states, args, kwargs))
            _mark_seq_dynamic(hidden_states.shape[1], (hidden_states, args, kwargs))
            if _DYNAMIC_BATCH and hidden_states.shape[0] > 1:
                _mark_batch_dynamic(hidden_states.shape[0], (hidden_states, args, kwargs))
            _mark_packed_dynamic(kwargs)
            _precheck_band(self, hidden_states.shape[1], args, kwargs, hidden_states)
            _settle_mask_flags(self, kwargs)
        bit = 2 if torch.is_grad_enabled() else 1
        checked = self.__dict__.get("_unsloth_block_checked", 0)
        nonreentrant = (
            bit == 2 and not checked & 2 and not _IN_TRIAL[0] and _in_nonreentrant_checkpoint()
        )
        if (bit == 1 or nonreentrant) and not checked & 2:
            # Grad-mode trial runs first (RNG forked); a checkpoint recompute must not demote below
            # this call. Non-reentrant: identity saved-tensor hooks keep a failed trial graph out
            devices = [hidden_states.device.index] if hidden_states.is_cuda else []
            hooks = (
                torch.autograd.graph.saved_tensors_hooks(_same, _same)
                if nonreentrant
                else contextlib.nullcontext()
            )
            with (
                torch.random.fork_rng(devices = devices),
                torch.enable_grad(),
                hooks,
                _no_output_capture(self),
            ):
                _IN_TRIAL[0] = "nonreentrant" if nonreentrant else True
                try:
                    forward(self, hidden_states.detach().requires_grad_(), *args, **kwargs)
                finally:
                    _IN_TRIAL[0] = False
            mode = self.__dict__["_unsloth_block_mode"]
            if mode == _OFF:
                return original(self, hidden_states, *args, **kwargs)
            checked = self.__dict__.get("_unsloth_block_checked", 0)
        if checked & bit:
            return run(self, hidden_states, args, kwargs, mode)
        while True:
            try:
                with _strict_dynamo():
                    out = run(self, hidden_states, args, kwargs, mode)
                self.__dict__["_unsloth_block_checked"] = checked | bit
                return out
            except torch.OutOfMemoryError:
                raise
            except Exception as e:
                # Non-reentrant checkpoint ends a recompute by raising through the forward: control
                # flow, not a compile failure (demoting on it left the stock forward in every block).
                if type(e).__name__ == "_StopRecomputationError":
                    raise
                if os.environ.get("UNSLOTH_BLOCK_COMPILE_DEBUG") == "1":
                    import traceback
                    traceback.print_exc()
                mode = _SPLIT if mode == _FULL else _OFF
                self.__dict__["_unsloth_block_mode"] = mode
                kind = _layer_kind(self)
                state["start"][kind] = max(
                    state["start"].get(kind, _FIRST_MODE), mode, key = _DEMOTION.index
                )
                first = str(e).strip().splitlines()[0][:200] if str(e).strip() else type(e).__name__
                if (first, mode) not in state["reported"]:
                    state["reported"].add((first, mode))
                    print(
                        f"Unsloth: {cls.__name__} block compile failed ({first}); "
                        + (
                            "retrying with flex_attention outside the block."
                            if mode == _SPLIT
                            else "keeping the stock forward for this layer."
                        )
                    )
                if mode == _OFF:
                    return original(self, hidden_states, *args, **kwargs)

    forward._unsloth_block_original = original
    forward._unsloth_block_state = state
    forward._unsloth_block_checkpoint_outside = checkpoint_outside
    setattr(forward, _MARK, True)
    # A new name: `forward` is also the closure the grad-mode trial above calls (the shim would recurse).
    installed = _checkpointing(forward) if checkpoint_outside else forward
    if install:
        cls.forward = installed
    return installed


_TOWER_SEGMENTS = (
    "vision",
    "visual",
    "image",
    "audio",
    "speech",
    "video",
    "encoder",
    "tower",
    "projector",
    "vqmodel",
    "vq_model",
)
_LAYER_SUFFIXES = ("DecoderLayer", "Block", "Layer")
# The module holding a text decoder's layer list also holds its token embedding.
_TEXT_EMBEDDINGS = (
    "embed_tokens",
    "wte",
    "embed_in",
    "word_embeddings",
    "tok_embeddings",
    "embeddings",
)


def _gc_layer_class():
    try:
        from transformers.modeling_layers import GradientCheckpointingLayer
    except Exception:
        return None
    return GradientCheckpointingLayer


def _in_tower(name):
    return any(segment in part for part in name.lower().split(".") for segment in _TOWER_SEGMENTS)


def _is_ssm_layer(layer):
    # Traced, transformers swaps Mamba CUDA scan kernels for a torch scan; UNSLOTH_BLOCK_COMPILE_SSM=1.
    for module in layer.modules():
        name = type(module).__name__
        if ("Mamba" in name and name.endswith("Mixer")) or hasattr(module, "ssm_state_size"):
            return True
    return False


def _find_layer_lists(model):
    """(ModuleList, legacy) per text decoder list; legacy = the old one-*DecoderLayer-class rule."""
    gc_layer = _gc_layer_class()
    found, prefixes = [], []
    modules = dict(model.named_modules())
    for name, module in modules.items():
        if not isinstance(module, torch.nn.ModuleList) or len(module) == 0:
            continue
        if any(name.startswith(prefix) for prefix in prefixes):
            continue
        classes = {type(m) for m in module}
        legacy = len(classes) == 1 and type(module[0]).__name__.endswith("DecoderLayer")
        if not legacy:
            parent = modules.get(name.rsplit(".", 1)[0] if "." in name else "")
            if _in_tower(name) or not any(
                getattr(parent, key, None) is not None for key in _TEXT_EMBEDDINGS
            ):
                continue
            if gc_layer is not None and all(issubclass(c, gc_layer) for c in classes):
                pass
            elif len(classes) == 1 and type(module[0]).__name__.endswith(_LAYER_SUFFIXES):
                pass  # remote code without GradientCheckpointingLayer
            else:
                continue
        found.append((module, legacy))
        prefixes.append(name + "." if name else "")
    return found


def _layer_lists(model):
    return [layer_list for layer_list, _ in _find_layer_lists(model)]


def _layers(model):
    """(layer, legacy) per decoder layer, shared modules once."""
    seen, layers = set(), []
    for layer_list, legacy in _find_layer_lists(model):
        for layer in layer_list:
            if id(layer) not in seen:
                seen.add(id(layer))
                layers.append((layer, legacy))
    return layers


def _decoder_layer_classes(model):
    classes = []
    for layer, _ in _layers(model):
        if type(layer) not in classes:
            classes.append(type(layer))
    return classes


def _torch_supported():
    # torch 2.6: blocks fragmented and QLoRA grads moved vs the stock forward.
    from packaging.version import Version
    return Version(torch.__version__.split("+")[0]) >= Version("2.7")


def _experts_of(module):
    while module is not None:
        if type(module).__name__.endswith("Experts"):
            return module
        module = getattr(module, "base_layer", None)
    return None


def _lora_would_fold(experts):
    # Traced, zoo folds the LoRA delta into every expert's weights when the forward ignores the stash.
    try:
        from unsloth_zoo.temporary_patches import moe_utils as zoo_moe
        reads = zoo_moe._forward_statically_reads_stash(experts)
        if not reads and zoo_moe._interface_route_reads_stash(experts):
            reads = True
    except Exception:
        return False
    return reads is False


def _selective_dequant_requested(experts):
    # zoo's selective / chunked bnb 4-bit expert dequant plans from host-read token counts, so it
    # declines inside a trace: experts it should run on must stay out of the compiled block.
    # Never getattr: on packed experts (MXFP4) gate_up_proj is a property that dequantizes the stack.
    try:
        chunked = int(os.environ.get("UNSLOTH_MOE_DEQUANT_CHUNK_EXPERTS", "0")) > 0
    except ValueError:
        chunked = False
    if os.environ.get("UNSLOTH_MOE_SELECTIVE_DEQUANT", "0") != "1" and not chunked:
        return False
    return type(experts._parameters.get("gate_up_proj")).__name__ == "Params4bit"


def _experts_outside(layer):
    # At the layer's first call, after get_peft_model wrapped the experts.
    layer.__dict__["_unsloth_block_moe_done"] = True
    mode = os.environ.get("UNSLOTH_BLOCK_MOE", "auto")
    if mode == "inside":
        return
    for _, module in layer.named_modules():
        if getattr(module, "_unsloth_block_outside", False):
            continue
        experts = _experts_of(module)
        if experts is None:  # named_modules() is pre-order: the outermost wrapper comes first
            continue
        if (
            mode != "outside"
            and not _lora_would_fold(experts)
            and not _selective_dequant_requested(experts)
        ):
            continue
        module.forward = torch.compiler.disable(module.forward)
        for inner in module.modules():
            inner._unsloth_block_outside = True


def _experts_may_move(layer):
    """Could _experts_outside move these experts? Precomputed: its bytecode scan cannot run while tracing."""
    if os.environ.get("UNSLOTH_BLOCK_MOE", "auto") == "inside":
        return False
    for module in layer.modules():
        experts = _experts_of(module)
        if experts is not None and (
            os.environ.get("UNSLOTH_BLOCK_MOE", "auto") == "outside"
            or _lora_would_fold(experts)
            or _selective_dequant_requested(experts)
        ):
            return True
    return False


def _eager_setup_pending(layer):
    return (
        not layer.__dict__.get("_unsloth_block_moe_done", False)
        and layer.__dict__.get("_unsloth_block_moe_may_move", False)
    ) or (_GDN_OP[0] is None and hasattr(layer, "linear_attn"))


@torch.compiler.disable
def _eager_setup(layer, hidden_states):
    """Eager first-call setup under an enclosing compile (expert move, GDN self-check): one break, once."""
    if not layer.__dict__.get("_unsloth_block_moe_done", False):
        _experts_outside(layer)
    if _GDN_OP[0] is None and hasattr(layer, "linear_attn"):
        _prepare_gdn(layer, hidden_states)


def _raise_cache_limit(kinds):
    # 4 graphs per kind (grad x flex retry): mixed-kind classes would hit dynamo's cache limit.
    config = torch._dynamo.config
    name = "recompile_limit" if hasattr(config, "recompile_limit") else "cache_size_limit"
    need = 4 * kinds
    if getattr(config, name, need) < need:
        setattr(config, name, need)


def _install_router_capture_hooks(model):
    """Install transformers' output capture hooks before any block compiles, for models that record
    router_logits.

    transformers installs them on the first forward that asks for an output (output_router_logits),
    and Dynamo does not guard on module hooks (skip_nnmodule_hook_guards). A block compiled during an
    earlier forward without them kept running without the router hook: no router logits, and the
    aux loss raised IndexError. Installed up front, the hook body runs on every call and returns
    early unless the collector asks for its key (a guarded dict lookup). Needs the zoo collector
    mirror: the stock ContextVar.get graph-breaks every compiled block."""
    try:
        from transformers.utils import output_capturing
    except Exception:
        return 0
    registry = getattr(output_capturing, "_CAN_RECORD_REGISTRY", None)
    install = getattr(output_capturing, "maybe_install_capturing_hooks", None)
    collector = getattr(output_capturing, "CompileableContextVar", None)
    if (
        not isinstance(registry, dict)
        or install is None
        or not getattr(collector, "_unsloth_eager_mirror", False)
    ):
        return 0
    installed = 0
    # Outermost first, as transformers does; skip modules with a hooked descendant (no double
    # hooks).
    for module in model.modules():
        # dict.get: 5.23's _CanRecordRegistry.get expects the model and returned {} for a string (transformers#49594).
        flags = dict.get(registry, str(type(module))) or {}
        if "router_logits" not in flags or any(
            getattr(m, "_output_capturing_hooks_installed", False) for m in module.modules()
        ):
            continue
        try:
            install(module)
            installed += 1
        except Exception:
            pass
    return installed


def patch_block_compile(model):
    """Compile every decoder-layer class of `model` as one region. Returns the classes wrapped."""
    # getattr: PeftModel delegates. The env fills only unstamped layers, so a re-patch keeps the stamp.
    forced = getattr(model, "_unsloth_forced_float32", None)
    env = os.environ.get("UNSLOTH_FORCE_FLOAT32", "0") == "1"
    layers = _layers(model)
    for layer, _ in layers:
        if forced is None:
            layer.__dict__.setdefault("_unsloth_forced_float32", env)
        else:
            layer.__dict__["_unsloth_forced_float32"] = bool(forced)
    forced = env if forced is None else forced
    if (
        forced
        or not _enabled()
        or not torch.cuda.is_available()
        or not _torch_supported()
        or _zero3_partitioned()
    ):
        return []
    ssm = os.environ.get("UNSLOTH_BLOCK_COMPILE_SSM", "0") == "1"
    compiled_layers = []
    for layer, legacy in layers:
        if type(layer).__name__ in _SKIP:
            continue
        # Per layer, not class: NemotronHBlock holds both kinds. Legacy lists compile as before.
        if not legacy and not ssm and _is_ssm_layer(layer):
            layer.__dict__["_unsloth_block_mode"] = _OFF
            continue
        compiled_layers.append(layer)
    classes = []
    for layer in compiled_layers:
        if type(layer) not in classes:
            classes.append(type(layer))
        try:
            layer.__dict__["_unsloth_block_moe_may_move"] = _experts_may_move(layer)
        except Exception:
            layer.__dict__["_unsloth_block_moe_may_move"] = True
    if not classes:
        return []
    _install_router_capture_hooks(model)
    inner = {type(m) for layer in compiled_layers for m in layer.modules()}
    for cls in inner:
        _rebind_class(cls)
    _flex_split_hook()
    _sdpa_alignment_without_guard()
    _gemma4_router_hook()
    _stable_parametrized_classes()
    if any(type(m).__name__ == "Linear4bit" for m in model.modules()):
        _nf4_in_blocks()
    kinds = {}
    for layer in compiled_layers:
        kinds.setdefault(type(layer), set()).add(_layer_kind(layer))
    if not _ISOLATE_RECOMPILES:
        # Older torch: one global per-code-object limit. 2.13+ sets a limit per region in _wrap.
        _raise_cache_limit(max(len(k) for k in kinds.values()))
    # The wrap is per class: instances outside the decoder lists (T5 encoder) stay stock.
    found = {id(layer) for layer, _ in layers}
    for module in model.modules():
        if type(module) in classes and id(module) not in found:
            module.__dict__["_unsloth_block_mode"] = _OFF
    wrapped = []
    for cls in classes:
        if getattr(cls.forward, _MARK, False):
            continue
        _wrap(cls, _COMPILE_OPTIONS, len(kinds.get(cls, ())) or 1)
        wrapped.append(cls.__name__)
    for layer in compiled_layers:
        if not any(m.__dict__.get("_unsloth_fp8_lora") is not None for m in layer.modules()):
            continue
        default = type(layer).forward
        precise = getattr(default, "_unsloth_fp8_block_forward", None)
        if precise is None:
            # FP8 dgrad amplifies elided BF16 rounding; other instances of this class keep their options.
            precise = _wrap(
                type(layer),
                {**_COMPILE_OPTIONS, "emulate_precision_casts": True},
                len(kinds.get(type(layer), ())) or 1,
                install = False,
            )
            default._unsloth_fp8_block_forward = precise
        if getattr(layer.forward, "__func__", None) is not precise:
            layer.forward = types.MethodType(precise, layer)
            layer.__dict__.pop("_unsloth_block_checked", None)
    if _COMPILE_OPTIONS.get("triton.cudagraphs") or (
        _CUDAGRAPH_MODE == "auto" and _auto_platform_reason() is None
    ):
        _watch_hidden_states(model)
    model.__dict__["_unsloth_block_compile_installed"] = True
    if _CUDAGRAPH_MODE == "auto":
        # A new model decides again: a launch-bound model loaded first must not turn graphs on for
        # a GPU-bound one. Graphs that went off for shape churn stay off.
        _CUDAGRAPH_STATE["auto"] = _new_auto_state()
    if wrapped:
        print(f"Unsloth: compiling each transformer block as one region ({', '.join(wrapped)}).")
    return wrapped
