# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline schedules for apply_pipeline_parallel: one stage per rank (GPipe, 1F1B) or several (looped / V).

    pp = apply_pipeline_parallel(model, mesh["pp"], n_microbatches = 8, schedule = "interleaved1f1b")

Multi-stage schedules split the decoder into pp x stages_per_rank stages. Rank r keeps stages
r, r + pp, r + 2 pp, ... (looped: Interleaved1F1B, LoopedBFS, InterleavedZeroBubble) or r and 2 pp - 1 - r
(V: ZBVZeroBubble, DualPipeV, two stages per rank, the first and the last stage on rank 0). Each local stage runs
the same model object with only its own layers active, so parameter names, adapter keys and the 1-GPU
initialization stay as they are.

Each choice is checked before the model is cut by building the torch schedule on placeholder stages: its
micro-batch constraints, whether a rank runs a forward after a backward inside one step (smart gradient
checkpointing offload slots must then be leased per checkpoint, unsloth-zoo PR #2071) and whether it splits
the backward into input and weight passes (zero bubble: torch.autograd.grad, which reentrant checkpointing
cannot run).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import Optional

import torch
import torch.nn as nn

__all__ = [
    "ScheduleSpec",
    "schedule_spec",
    "stage_indices",
    "validate_schedule",
    "SCHEDULE_NAMES",
]


@dataclasses.dataclass(frozen = True)
class ScheduleSpec:
    name: str  # normalized, e.g. "interleaved1f1b"
    torch_name: str  # torch.distributed.pipelining.schedules class
    multi: bool  # PipelineScheduleMulti: a list of local stages
    style: str  # "single", "loop" or "v": which stages a rank keeps
    default_stages_per_rank: int


_SPECS = {
    "gpipe": ScheduleSpec("gpipe", "ScheduleGPipe", False, "single", 1),
    "1f1b": ScheduleSpec("1f1b", "Schedule1F1B", False, "single", 1),
    "interleaved1f1b": ScheduleSpec("interleaved1f1b", "ScheduleInterleaved1F1B", True, "loop", 2),
    "loopedbfs": ScheduleSpec("loopedbfs", "ScheduleLoopedBFS", True, "loop", 2),
    "interleavedzerobubble": ScheduleSpec(
        "interleavedzerobubble", "ScheduleInterleavedZeroBubble", True, "loop", 2
    ),
    "zbvzerobubble": ScheduleSpec("zbvzerobubble", "ScheduleZBVZeroBubble", True, "v", 2),
    "dualpipev": ScheduleSpec("dualpipev", "ScheduleDualPipeV", True, "v", 2),
}
_ALIASES = {"zbv": "zbvzerobubble", "interleavedzb": "interleavedzerobubble"}
SCHEDULE_NAMES = tuple(_SPECS)


def _normalize(name):
    key = str(name).lower().replace("-", "").replace("_", "").replace(" ", "")
    if key.startswith("schedule"):
        key = key[len("schedule") :]
    return _ALIASES.get(key, key)


def schedule_spec(name) -> ScheduleSpec:
    spec = _SPECS.get(_normalize(name))
    if spec is None:
        raise ValueError(
            f"Unsloth: pipeline schedule must be one of {SCHEDULE_NAMES}, got {name!r}."
        )
    return spec


def torch_schedule_class(spec):
    from torch.distributed.pipelining import schedules

    cls = getattr(schedules, spec.torch_name, None)
    if cls is None:
        raise RuntimeError(
            f"Unsloth: torch {torch.__version__} has no {spec.torch_name} "
            f"(pipeline schedule {spec.name!r})."
        )
    return cls


def resolve_stages_per_rank(spec, stages_per_rank):
    """stages_per_rank for `spec` (None = its default); ValueError when the schedule cannot run it."""
    v = spec.default_stages_per_rank if stages_per_rank is None else int(stages_per_rank)
    if v < 1:
        raise ValueError("Unsloth: PPConfig.stages_per_rank must be >= 1.")
    if spec.style == "single" and v != 1:
        raise ValueError(
            f"Unsloth: the {spec.name} schedule runs one stage per rank (PPConfig.stages_per_rank = {v}); "
            "use 'interleaved1f1b' or 'loopedbfs' for several."
        )
    if spec.style == "v" and v != 2:
        raise ValueError(
            f"Unsloth: the {spec.name} schedule runs exactly 2 stages per rank (V placement), got "
            f"PPConfig.stages_per_rank = {v}."
        )
    return v


def stage_indices(spec, rank, pp, stages_per_rank):
    """Global stage indices pp rank `rank` keeps, in execution order of its forward."""
    if spec.style == "single":
        return (rank,)
    if spec.style == "loop":
        return tuple(rank + k * pp for k in range(stages_per_rank))
    return (rank, 2 * pp - 1 - rank)  # "v"


class _NoParams(nn.Module):
    def forward(self, x):
        return x


def v_schedules_supported():
    """torch's PipelineStage handles two adjacent stages on one rank (V placement) from torch 2.12
    (PipelineStage._is_same_rank); the stage setup of torch 2.10 / 2.11 fails there (ZBVZeroBubble / DualPipeV,
    run on torch 2.10: gloo pair errors in the schedule's first exchange)."""
    try:
        from torch.distributed.pipelining import PipelineStage
    except ImportError:
        return False
    return hasattr(PipelineStage, "_is_same_rank")


def validate_schedule(spec, group, pp, stages_per_rank, n_microbatches, rank):
    """Build the torch schedule on parameter-free placeholder stages (no communication) so its own checks run
    before the model is cut. Returns {"interleaved": bool, "split_backward": bool} for the compute order of
    every rank."""
    from torch.distributed.pipelining import PipelineStage

    cls = torch_schedule_class(spec)
    if spec.style == "v" and not v_schedules_supported():
        raise ValueError(
            f"Unsloth: the {spec.name} pipeline schedule (V placement: two adjacent stages on one rank) needs "
            f"a torch whose PipelineStage handles adjacent stages on one rank (torch >= 2.12); on torch "
            f"{torch.__version__} torch's stage setup between them fails. Use 'interleaved1f1b' or "
            "'interleavedzerobubble'."
        )
    num_stages = pp * stages_per_rank
    local = stage_indices(spec, rank, pp, stages_per_rank)
    example = torch.empty((1, 1), device = "meta")
    stages = [
        PipelineStage(
            _NoParams(),
            k,
            num_stages,
            torch.device("cpu"),
            input_args = example,
            output_args = example,
            group = group,
        )
        for k in local
    ]
    loss_fn = lambda output, target: output.sum()  # noqa: E731
    try:
        if spec.multi:
            schedule = cls(stages, n_microbatches, loss_fn = loss_fn, scale_grads = False)
        else:
            schedule = cls(stages[0], n_microbatches, loss_fn = loss_fn, scale_grads = False)
    except (ValueError, AssertionError, RuntimeError) as exc:
        raise ValueError(
            f"Unsloth: the {spec.name} pipeline schedule cannot run {n_microbatches} micro-batches over "
            f"{num_stages} stages ({pp} ranks x {stages_per_rank}): {exc}"
        ) from None
    order = getattr(schedule, "pipeline_order", None)
    if not order:
        # Single-stage schedules: GPipe runs every forward first, 1F1B interleaves (when there are stages).
        return {"interleaved": spec.name == "1f1b" and pp > 1, "split_backward": False}
    interleaved = split = False
    for actions in order.values():
        seen_backward = False
        for action in actions:
            if action is None:
                continue
            # By name: the enum's values are ints on torch 2.10, letters later.
            kind = getattr(action.computation_type, "name", str(action.computation_type))
            if kind == "FORWARD":
                interleaved = interleaved or seen_backward
            elif kind in ("FULL_BACKWARD", "BACKWARD_INPUT", "BACKWARD_WEIGHT"):
                seen_backward = True
                split = split or kind != "FULL_BACKWARD"
    return {"interleaved": interleaved, "split_backward": split}


def non_reentrant_checkpointing_only(model) -> Optional[bool]:
    """None: no gradient checkpointing; True: every checkpointed module uses torch's non-reentrant
    checkpoint (torch.autograd.grad works through it); False otherwise (reentrant, Unsloth smart
    checkpointing or unknown)."""
    import torch.utils.checkpoint as checkpointing

    # unsloth_zoo smart checkpointing replaces torch.utils.checkpoint.checkpoint (keeping the original as
    # _old_checkpoint): a model checkpointed through it is not on torch's non-reentrant path.
    original = getattr(checkpointing, "_old_checkpoint", None) or checkpointing.checkpoint
    patched = checkpointing.checkpoint is not original
    found = False
    for module in model.modules():
        if getattr(module, "gradient_checkpointing", False) is not True:
            continue
        found = True
        func = getattr(module, "_gradient_checkpointing_func", None)
        if patched or not isinstance(func, functools.partial) or func.func is not original:
            return False
        if func.keywords.get("use_reentrant", True) is not False:
            return False
    if not found and bool(getattr(model, "is_gradient_checkpointing", False)):
        return False
    return True if found else None


def allow_split_backward():
    """Zero-bubble schedules run each micro-batch's backward as torch.autograd.grad(..., retain_graph = True)
    for the inputs, then again for the weights. A compiled backward (Unsloth compiles every decoder block)
    that AOTAutograd built with donated buffers refuses retain_graph ("compiled with non-empty donated
    buffers", seen on T4 with torch 2.14), so compile without them for this process. Returns True if the setting
    was changed. Blocks compiled before this call keep their buffers: apply the split before the first forward."""
    try:
        import torch._functorch.config as functorch_config
    except ImportError:
        return False
    if getattr(functorch_config, "donated_buffer", False):
        functorch_config.donated_buffer = False
        return True
    return False


@functools.lru_cache(maxsize = None)
def split_backward_keeps_function_nodes() -> bool:
    """True if a Python autograd node (an autograd.Function's grad_fn, which is what an AOTAutograd compiled
    region becomes: CompiledFunctionBackward) still answers `_input_metadata` once its graph root is gone.
    torch's split backward needs that: stage_backward_input keeps the intermediate nodes, detaches the stage
    outputs, and stage_backward_weight then builds GradientEdges on them, which autograd.grad checks through
    node._input_metadata. Up to torch 2.12 THPFunction holds its node through a std::weak_ptr (expired by then:
    "Attribute '_input_metadata' is invalid for this instance of _C._FunctionBase"); torch 2.13 made it an
    owning c10::intrusive_ptr. Probed directly (CPU, no compile) rather than read from the version."""

    class _Probe(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x * 2

        @staticmethod
        def backward(ctx, grad):
            return grad * 2

    try:
        with torch.enable_grad():
            out = _Probe.apply(torch.ones(1, requires_grad = True))
            node = out.grad_fn
            del out
            node._input_metadata
    except RuntimeError:
        return False
    except Exception:
        # Unknown failure shape: do not refuse on a probe we cannot interpret.
        return True
    return True


def compiled_blocks(model):
    """Names of submodules of `model` that run through torch.compile: Unsloth's block compile (class forward
    marked _unsloth_block_compiled), nn.Module.compile(), torch.compile(module) wrappers, torch.compile'd
    forwards and forwards calling a compiled function (unsloth_zoo's compiled modules). Their backward is an
    AOTAutograd CompiledFunctionBackward node."""
    found = []
    calls = {}  # per class forward: does it call a compiled function
    for name, module in model.named_modules():
        cls_forward = getattr(type(module), "forward", None)
        inst_forward = module.__dict__.get("forward")
        if id(cls_forward) not in calls:
            calls[id(cls_forward)] = _calls_compiled(cls_forward)
        if (
            getattr(cls_forward, "_unsloth_block_compiled", False)
            or module.__dict__.get("_compiled_call_impl") is not None
            or type(module).__name__ == "OptimizedModule"
            or hasattr(_unwrap_callable(inst_forward), "_torchdynamo_orig_callable")
            or hasattr(inst_forward, "_torchdynamo_orig_callable")
            or calls[id(cls_forward)]
            or (inst_forward is not None and _calls_compiled(inst_forward))
        ):
            found.append(name or type(module).__name__)
    return found


def _is_compiled(value):
    try:
        return (
            hasattr(value, "_torchdynamo_orig_callable")
            or hasattr(value, "_unsloth_compiled_func")
            or type(value).__name__ == "OptimizedModule"
        )
    except Exception:
        return False


def _calls_compiled(forward):
    """True if `forward` calls, by global name, a torch.compile'd function: unsloth_zoo's generated
    unsloth_compiled_module_* files keep each module's class forward eager and have it call a module-level
    compiled function (`Gemma3MLP.forward` -> `Gemma3MLP_forward`, torch_compile_with_fallback, whose wrapper
    keeps the compiled markers). Its backward is a CompiledFunctionBackward node like a compiled block's."""
    fn = _unwrap_callable(forward)
    code = getattr(fn, "__code__", None)
    scope = getattr(fn, "__globals__", None)
    if code is None or not isinstance(scope, dict):
        return False
    if any(_is_compiled(scope.get(name)) for name in code.co_names if name in scope):
        return True
    for cell in (
        getattr(fn, "__closure__", None) or ()
    ):  # a compiled helper captured from an enclosing scope
        try:
            value = cell.cell_contents
        except ValueError:  # empty cell
            continue
        if _is_compiled(value):
            return True
    return False


_SINGLE_BACKWARD_MODULES = ("unsloth.kernels.fast_lora",)


def _unwrap_callable(value):
    """The function under bound methods and functools.partial layers, in any nesting: patch_peft_model installs
    the MLP kernel as MethodType(partial(apply_lora_mlp, inplace=False), mlp) for Cohere (see _fused_lora_modules)."""
    for _ in range(16):
        if isinstance(value, functools.partial):
            value = value.func
        elif hasattr(value, "__func__"):
            value = value.__func__
        else:
            break
    return value


def single_backward_kernels(model):
    """Unsloth fused LoRA kernels attached to `model` (apply_qkv / apply_o / the MLP forward). Their backward
    reuses saved activations in place, so it can run once per forward only; a zero-bubble schedule runs the
    graph twice (input pass, then weight pass) and fails on a modified saved tensor (seen on T4, torch 2.14.1)."""
    found = set()
    for module in model.modules():
        candidates = list(module.__dict__.values()) + [getattr(type(module), "forward", None)]
        for value in candidates:
            fn = _unwrap_callable(value)
            if callable(fn) and getattr(fn, "__module__", None) in _SINGLE_BACKWARD_MODULES:
                found.add(getattr(fn, "__name__", repr(fn)))
    return sorted(found)
