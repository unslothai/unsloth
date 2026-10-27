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
    "interleaved1f1b": ScheduleSpec(
        "interleaved1f1b", "ScheduleInterleaved1F1B", True, "loop", 2
    ),
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
