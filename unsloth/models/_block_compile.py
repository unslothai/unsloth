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


# Process wide: every block class shares one cudagraph tree (one pool per device).
# off: why graphs went off (None = on); release: None, "done" or "kept"; ours: compile ids of the
# tree functions block calls created; built: compile ids the graphs-on block backend compiled;
# managers: CUDA index -> the tree manager block graphs ran on.
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
# call starts a forward pass; passes: forward passes seen, all with batch shape `shape`.
def _new_auto_state():
    return {
        "phase": "probe",
        "first": None,
        "shape": None,
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
    """Call the graphs-on region; note the compile ids of tree functions it created (forward here,
    the backward of the same compile later) that the block backend built, so the release can tell
    its pool is only ours."""
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
        or get("cu_seq_lens_q") is not None
        or get("cu_seq_lens_k") is not None
        or get("cu_seqlens") is not None
    )


def _trainable(layer):
    params = layer.__dict__.get("_unsloth_block_params")
    if params is None:
        params = layer.__dict__["_unsloth_block_params"] = list(layer.parameters())
    return any(p.requires_grad for p in params)


# The grad-mode trial before a checkpointed block's first no-grad call stands in for the recompute.
_IN_TRIAL = [False]


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
    """A grad-mode call outside backward leaves a forward pending, so cudagraph trees start no new
    generation (can_start_new_generation) and its outputs need no copy out of the pool. Checkpointed
    calls (no-grad forward, recompute in backward, and the trial standing in for it) may start one."""
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


# Backward nodes that write their input gradient to new storage (a view, add, clone or cat passes
# on the incoming gradient or a narrowed view of it).
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


def _param_grads_out_of_pool(layer):
    """A first .grad from a replayed backward is graph pool memory the next replay overwrites: copy it."""
    if layer.__dict__.get("_unsloth_block_grad_hooks", False):
        return
    layer.__dict__["_unsloth_block_grad_hooks"] = True

    def hook_for(param):
        ref = weakref.ref(param)

        def hook(grad):
            p = ref()
            if p is None or p.grad is not None:
                return None
            return grad.clone()

        return hook

    for param in layer.parameters():
        if param.requires_grad:
            param.register_hook(hook_for(param))


def _in_nonreentrant_checkpoint():
    """Non-reentrant checkpoint frees recomputed inputs mid graph recording, which cudagraph trees refuse."""
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


# Auto mode. Graphs remove launch overhead only: they pay off when the host takes longer to launch a
# block than the GPU takes to run it, and cost a recording per batch shape. Forward passes 1-2 run
# without graphs (compile); pass 3 times a few block calls; graphs start at the next pass when every
# call so far had one batch shape and launching was slower than running by _AUTO_RATIO.
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
    if _in_nonreentrant_checkpoint():
        return "non-reentrant checkpointing"
    if _graphs_unsafe_moe(layer, note = False):
        return "MoE layer"
    if _flex_block_mask_call(args, kwargs, note = False):
        return "flex attention"
    return None


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
    why = _auto_call_reason(layer, hidden_states, args, kwargs)
    if why is None and tuple(hidden_states.shape) != auto["shape"]:
        why = "batch shape changed"
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
    """(output, host seconds to launch one block call, GPU seconds to run it). The launches queue
    behind a GPU sleep, so the event pair after the sleep sees the block's kernels back to back; if
    launching outlasts the sleep the GPU time includes gaps, which can only understate the ratio.
    Sleep and events run on the block's device (cuda:1 or a device_map shard need not be current).
    Host / GPU are None when timing is unavailable."""
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
    """Inside DDP.forward dynamo hands a block's graph to DDPOptimizer, which splits it at gradient
    bucket boundaries. A checkpoint recompute runs from backward, outside DDP.forward, so a block
    graph compiled there is not split and saves a different number of tensors than the forward
    (CheckpointError under DDP's non-reentrant checkpointing, DeepSeek-OCR). Each block is its own
    region already, so its gradients still reach DDP's buckets block by block."""
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


# gpt-oss builds its flex mask inside attention: the block broke and recompiled per length.
_SKIP = {"GptOssDecoderLayer"}


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
    """dynamo resolves a function's globals through sys.modules[__globals__["__name__"]]; Unsloth's
    generated modules exec code into a dict that module does not own (Gemma-4: `module
    'transformers...modeling_gemma4' has no attribute 'Gemma4RMSNorm_forward'`). Rebuild such a
    function on a registered module holding a copy of that dict. A copy cannot follow later
    writes, so functions that assign globals are left alone."""
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
    """A method that is itself torch.compile'd (zoo's patch_function): inside a block, call the
    original so dynamo inlines it. Inlining through the compiled wrapper guards on the wrapper's
    closure as if it were the original's (torch 2.13 / 2.14, Gemma-4 fp16 MLP: `Unexpected type
    CatchErrorsWrapper`). Outside a block the compiled wrapper runs as before."""
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
                out = torch.ops.unsloth.qwen3_5_gdn_chunk(q, k, v, g, beta, scale)
            else:
                out = torch.ops.unsloth.qwen3_5_gdn_chunk(
                    q, k, v, g, beta, scale, cu_seqlens.contiguous()
                )
            return out[0], None
        return original(q, k, v, g, beta, **kwargs)

    chunk_gated_delta_rule._unsloth_block_adapter = True
    return chunk_gated_delta_rule


def _route_fla_kernel():
    """Point modeling code's references to fla's chunk_gated_delta_rule at the adapter. Callers
    reach it through module globals, transformers' hub-kernel wrapper (a closure over it) and
    Unsloth's compiled-module copies, so callers are found from the function object (gc
    referrers), not by name. Closures are left alone: only namespace entries are swapped."""
    import gc

    # gc.get_referrers skips the permanent generation, and Trainer.train() freezes the setup-time
    # heap (_gc_train_scope) before the first GatedDeltaNet forward gets here: the walk found no
    # caller and the op never engaged. Thaw for the walk; refreezing also freezes what train()
    # allocated so far, which train()'s exit unfreezes with the rest.
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
    @functools.wraps(original, updated = ())
    def causal_conv1d_fn(
        hidden_states,
        weight,
        bias = None,
        activation = None,
        **kwargs,
    ):
        if (
            _IN_BLOCK[0]
            and torch.compiler.is_compiling()
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
        return original(hidden_states, weight, bias, activation = activation, **kwargs)

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
        B, T = hidden_states.shape[0], hidden_states.shape[1]
        dev, dt = hidden_states.device, hidden_states.dtype

        def make_inputs():
            gen = torch.Generator(device = dev).manual_seed(0)
            r = lambda *s, dtype = dt: torch.randn(*s, device = dev, dtype = dtype, generator = gen)
            g = -torch.rand(B, T, heads, device = dev, generator = gen) * 0.1
            return (
                r(B, T, heads, dk),
                r(B, T, heads, dk),
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


# zoo's Gemma-4 sliding router scans the mask (data dependent) and declines while compiling, so
# the block runs the scan eagerly and a router wrapper reads the verdict stored on the mask.
_BAND_MODULE = "unsloth_zoo.temporary_patches.gemma4_banded_attention"
_BAND_CHECK = [None]


def _gemma4_router_hook():
    """Inside a block, a sliding layer whose mask is a plain causal band no longer than the
    window runs is_causal SDPA, as the stock router does; longer bands run masked SDPA. The
    router itself is not traced: its engagement counters drive a log-once branch dynamo guards
    on, and its banded kernel traced in a checkpointed block recomputes different saved tensors
    than its forward (CheckpointError)."""
    try:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    except Exception:
        return
    router = ALL_ATTENTION_FUNCTIONS.get("sdpa")
    if not getattr(router, "_unsloth_gemma4_flash", False) or getattr(
        router, "_unsloth_block_band", False
    ):
        return
    zoo = sys.modules.get(router.__module__)
    plain = getattr(zoo, "_ORIG_SDPA", [None])[0]
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


def _precheck_band(layer, seq_len, args, kwargs):
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


# Mode of the block being traced (_SPLIT: flex_attention outside); dynamo guards on it.
_IN_BLOCK = [0]
# True only inside Unsloth's own compiled block (fullgraph=False, a graph break is legal), not when the
# block is inlined into a user's torch.compile, which may be fullgraph=True. Read by unsloth_zoo.
_IN_COMPILED_REGION = [False]
_PACKED_MODULE = "unsloth.utils.hf_packed_attention"
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


def _wrap(
    cls,
    compile_options,
    kinds = 1,
):
    original = cls.forward
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
            with capture_scratch(), _no_ddp_graph_split():
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
        if (
            mode == _OFF
            or not self.training
            or _has_cache(args, kwargs)
            or not _enabled()
            or _forced_float32(self)
            or _zero3_partitioned(self)
            or (unsloth_forward and kwargs.get("packed_seq_lengths") is not None)
            # PrefixGrouper's int segment table would recompile at every new GRPO length.
            or (unsloth_forward and kwargs.get("prefix_seg_info") is not None)
        ):
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
            _mark_seq_dynamic(hidden_states.shape[1], (hidden_states, args, kwargs))
            if _DYNAMIC_BATCH and hidden_states.shape[0] > 1:
                _mark_batch_dynamic(hidden_states.shape[0], (hidden_states, args, kwargs))
            _precheck_band(self, hidden_states.shape[1], args, kwargs)
        bit = 2 if torch.is_grad_enabled() else 1
        checked = self.__dict__.get("_unsloth_block_checked", 0)
        if bit == 1 and not checked & 2:
            # Grad-mode trial first (RNG forked): a checkpoint recompute must not demote below this call.
            devices = [hidden_states.device.index] if hidden_states.is_cuda else []
            with torch.random.fork_rng(devices = devices), torch.enable_grad():
                _IN_TRIAL[0] = True
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
                with torch._dynamo.config.patch(suppress_errors = False):
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
    setattr(forward, _MARK, True)
    cls.forward = forward


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
    inner = {type(m) for layer in compiled_layers for m in layer.modules()}
    for cls in inner:
        _rebind_class(cls)
    _flex_split_hook()
    _gemma4_router_hook()
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
    if _COMPILE_OPTIONS.get("triton.cudagraphs") or (
        _CUDAGRAPH_MODE == "auto" and _auto_platform_reason() is None
    ):
        _watch_hidden_states(model)
    if _CUDAGRAPH_MODE == "auto":
        # A new model decides again: a launch-bound model loaded first must not turn graphs on for
        # a GPU-bound one. Graphs that went off for shape churn stay off.
        _CUDAGRAPH_STATE["auto"] = _new_auto_state()
    if wrapped:
        print(f"Unsloth: compiling each transformer block as one region ({', '.join(wrapped)}).")
    return wrapped
