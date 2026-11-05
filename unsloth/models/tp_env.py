# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor / pipeline parallelism from the environment, for unmodified training scripts.

    UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 2 train.py                    # tp 2
    UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 4 train.py                    # dp 2 (DDP-style) x tp 2
    UNSLOTH_TP_SIZE=2 UNSLOTH_FSDP2=1 torchrun --nproc-per-node 4 train.py    # dp 2 (native FSDP2) x tp 2
    UNSLOTH_PP_SIZE=2 torchrun --nproc-per-node 2 train.py                    # pp 2
    UNSLOTH_PP_SIZE=2 UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 4 train.py  # pp 2 x tp 2
    UNSLOTH_PP_SIZE=2 [UNSLOTH_FSDP2=1] torchrun --nproc-per-node 4 train.py  # pp 2 x dp 2 (DDP-style / FSDP2)

UNSLOTH_PARALLEL="dp=..,tp=..,pp=..,cp=.." (unsloth/parallel_env.py) sets these variables in one flag, and adds
UNSLOTH_CP_SIZE: context parallel on the same mesh. The layout (parallel_env.ParallelLayout, TPEnvLayout here) is
read once. The world splits into pp x dp x cp x tp ranks (pp outermost, tp the inner dimension; one root
DeviceMesh, size-1 axes left out). Tensor parallel alone:
- FastLanguageModel / FastModel.from_pretrained: 16-bit checkpoints load through sharded_loading (each rank
  reads its slices of the attention / MLP projections); 4-bit loads in full and is sharded at get_peft_model.
- get_peft_model: apply_tensor_parallel over mesh["tp"] once LoRA is attached (full-size LoRA, 1-GPU init).
- Trainer.__init__: tensor parallel if get_peft_model did not apply it; native FSDP2 shards over mesh["dp"]
  (UNSLOTH_FSDP2=1), otherwise gradients are averaged over mesh["dp"] before clipping and the optimizer step.
So the order is always split -> tp -> FSDP2.

With UNSLOTH_PP_SIZE (models/pp_env.py) the model loads in full on every rank and stays whole until the
Trainer: Trainer.__init__ splits it (apply_pipeline_parallel over mesh["pp"], the other stages' weights kept on
the CPU for the post-training cells), then applies tensor parallel inside the stage, then FSDP2 / DDP-style
averaging over mesh["dp"], and runs every training step through the pipeline schedule.
UNSLOTH_PP_MICROBATCHES (default 2 x pp, lowered to a divisor of the batch) and UNSLOTH_PP_SCHEDULE (gpipe,
1f1b) tune it.

UNSLOTH_TP_SHARDED_LOAD=0 loads 16-bit checkpoints in full on every rank instead (then shards them).
"""

from __future__ import annotations

import functools
import inspect
import os

import torch

__all__ = [
    "TPEnvLayout",
    "env_layout",
    "env_hint",
    "install_tensor_parallel_env_trainer_init",
    "tensor_parallel_env_load",
    "tensor_parallel_env_peft",
    "tensor_parallel_env_trainer_pre",
]

_UNSET = object()
_LAYOUT = _UNSET
_DEPTH = {"load": 0, "peft": 0}
_DDP_MESH = "_unsloth_tp_ddp_mesh"


from ..parallel_env import PP_SCHEDULES as _PP_SCHEDULES  # noqa: F401  (old import path)
from ..parallel_env import ParallelLayout as TPEnvLayout
from ..parallel_env import _int_env  # noqa: F401
from ..parallel_env import read_layout as _read_layout


def env_layout():
    """The layout from UNSLOTH_TP_SIZE (read once), or None when tensor parallelism is not requested."""
    global _LAYOUT
    if _LAYOUT is _UNSET:
        _LAYOUT = _read_layout()
    return _LAYOUT


def _reset_env_layout():
    """Tests: read the environment again."""
    global _LAYOUT
    _LAYOUT = _UNSET


def env_hint():
    """A suffix for refusals under an env-driven launch (empty otherwise)."""
    try:
        layout = _LAYOUT if _LAYOUT is not _UNSET else None
    except Exception:
        layout = None
    if layout is None:
        return ""
    if layout.pp > 1:
        what = "Pipeline parallelism" + (" x tensor parallelism" if layout.tp > 1 else "")
        return (
            f" ({what} came from {layout.env_vars}: unset {'them' if layout.tp > 1 else 'it'} to run this "
            "script with DDP or UNSLOTH_FSDP2=1.)"
        )
    return (
        f" (Tensor parallelism came from UNSLOTH_TP_SIZE={layout.tp}: unset it to run this script with "
        "DDP or UNSLOTH_FSDP2=1.)"
    )


def _reraise_with_hint(exc):
    hint = env_hint()
    if hint and hint not in str(exc):
        try:
            new = type(exc)(f"{exc}{hint}")
        except Exception:
            raise exc
        raise new from exc
    raise exc


def _rank0():
    import torch.distributed as dist
    return not dist.is_initialized() or dist.get_rank() == 0


_REFUSED_LOAD_FLAGS = (
    ("fast_inference", "vLLM (fast_inference = True)"),
    ("load_in_8bit", "8-bit weights (load_in_8bit = True; load in 4-bit or 16-bit)"),
    ("load_in_fp8", "fp8 weights (load_in_fp8)"),
)
_REFUSED_PIPELINE_LOAD_FLAGS = (
    ("full_finetuning", "full finetuning (full_finetuning = True; use UNSLOTH_FSDP2=1 instead)"),
    ("qat_scheme", "QAT (qat_scheme)"),
)


def _bound(fn, args, kwargs):
    try:
        signature = inspect.signature(fn)
        bound = signature.bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)
    values = dict(bound.arguments)
    for name, param in signature.parameters.items():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            values.update(values.pop(name, None) or {})
    return values


def _four_bit(values, fn):
    if values.get("quantization_config") is not None:
        return True
    if "load_in_4bit" in values:
        return bool(values["load_in_4bit"])
    if values.get("load_in_16bit"):
        return False
    try:
        default = inspect.signature(fn).parameters["load_in_4bit"].default
    except (KeyError, TypeError, ValueError):
        return False
    return default is True


def tensor_parallel_env_load(fn):
    """Decorator for a from_pretrained (FastLanguageModel / FastModel): sharded 16-bit loads under UNSLOTH_TP_SIZE."""

    @functools.wraps(fn)
    def from_pretrained(*args, **kwargs):
        layout = env_layout()
        if layout is None or _DEPTH["load"]:
            return fn(*args, **kwargs)
        values = _bound(fn, args, kwargs)
        what = (
            "tensor parallel"
            if layout.pp == 1
            else ("pipeline parallel" + (" x tensor parallel" if layout.tp > 1 else ""))
        )
        unset = " / ".join(var.split("=")[0] for var in layout.env_vars.split())
        refused = [what_ for key, what_ in _REFUSED_LOAD_FLAGS if values.get(key)]
        if (
            layout.pp > 1
        ):  # tensor parallel alone trains these; pipeline parallel v1 trains LoRA adapters
            refused += [what_ for key, what_ in _REFUSED_PIPELINE_LOAD_FLAGS if values.get(key)]
        if refused:
            raise NotImplementedError(
                f"Unsloth: {layout.env_vars} ({what}) does not support "
                + "; ".join(refused)
                + f". Unset {unset} to run this script with DDP or UNSLOTH_FSDP2=1."
            )
        from .fsdp2_load import fsdp2_sharded_load_requested

        if fsdp2_sharded_load_requested():
            # Its per-rank shard reads know nothing of the tp / pp mesh.
            raise NotImplementedError(
                f"Unsloth: {layout.env_vars} ({what}) does not support "
                "UNSLOTH_FSDP2_SHARDED_LOAD=1. Unset UNSLOTH_FSDP2_SHARDED_LOAD"
                + (
                    "; tensor parallel already loads 16-bit weights sharded."
                    if layout.pp == 1
                    else "."
                )
            )
        if layout.tp == 1:
            # Pipeline parallel alone: a whole model on every rank (split at the Trainer), checked now.
            layout.mesh()
            _DEPTH["load"] += 1
            try:
                out = fn(*args, **kwargs)
            finally:
                _DEPTH["load"] -= 1
            return _check_pipeline_model(out, layout)
        mesh = layout.tp_mesh
        _DEPTH["load"] += 1
        try:
            # QAT swaps the loaded Linears for torchao FakeQuantizedLinear copies: load in full, shard later.
            qat = values.get("qat_scheme") is not None
            if _four_bit(values, fn) or not layout.sharded_loading or _is_adapter(values) or qat:
                out = fn(
                    *args, **kwargs
                )  # full load; get_peft_model shards it (Params4bit included)
                return _check_pipeline_model(out, layout) if layout.pp > 1 else out
            from .tp_native import sharded_loading

            try:
                with sharded_loading(mesh) as state:
                    out = fn(*args, **kwargs)
            except ValueError as exc:
                if not str(exc).startswith("Unsloth: tensor parallel sharded loading"):
                    raise
                if _rank0():
                    print(f"{exc} Loading in full on every rank instead; get_peft_model shards it.")
                return fn(*args, **kwargs)
            if not state["models"] and _rank0():
                print(
                    "Unsloth: UNSLOTH_TP_SIZE: this loader did not go through transformers from_pretrained; "
                    "the model loaded in full and is sharded at get_peft_model."
                )
            return out
        finally:
            _DEPTH["load"] -= 1

    from_pretrained._unsloth_tp_env = True
    return _marks_world_save(from_pretrained)


def _check_pipeline_model(out, layout):
    """UNSLOTH_PP_SIZE: refuse a model the pipeline split cannot take right after loading it (not at the
    Trainer), naming the environment variable."""
    model = _model_of(out)
    if model is None or layout.pp == 1 or _is_peft_adapter_model(model):
        return out
    from .pp_native import check_pipeline_model

    try:
        check_pipeline_model(model)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        _reraise_with_hint(exc)
    return out


def _is_peft_adapter_model(model):
    return hasattr(model, "peft_config") and bool(getattr(model, "peft_config", None))


def _marks_world_save(from_pretrained):
    @functools.wraps(from_pretrained)
    def wrapper(*args, **kwargs):
        out = from_pretrained(*args, **kwargs)
        return out if env_layout() is None else _mark_world_save(out)

    return wrapper


def _mark_world_save(out):
    """The tokenizer / processor of a tensor parallel run saves once per world (rank 0 writes, save.py)."""
    if isinstance(out, (tuple, list)) and len(out) > 1 and hasattr(out[1], "__dict__"):
        from .tp_replica import mark_world_save
        try:
            mark_world_save(out[1])
        except Exception:
            pass
    return out


def _is_adapter(values):
    """A LoRA adapter directory / repo (a reload cell): loaded in full on every rank, nothing to shard."""
    name = values.get("model_name", values.get("pretrained_model_name_or_path"))
    if not isinstance(name, str):
        return False
    if os.path.isdir(name):
        return os.path.isfile(os.path.join(name, "adapter_config.json"))
    if os.path.exists(name) or name.count("/") != 1:
        return False
    try:
        from huggingface_hub import file_exists
        return bool(file_exists(name, "adapter_config.json", token = values.get("token")))
    except Exception:
        return False


def _model_of(out):
    if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.nn.Module):
        return out[0]
    return out if isinstance(out, torch.nn.Module) else None


def _apply(model, layout):
    from .tp_native import apply_tensor_parallel, is_tensor_parallel
    if is_tensor_parallel(model):
        return model
    try:
        return apply_tensor_parallel(model, layout.tp_mesh)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        _reraise_with_hint(exc)


def tensor_parallel_env_peft(fn):
    """Decorator for get_peft_model: shard over mesh["tp"] once LoRA is attached."""

    @functools.wraps(fn)
    def get_peft_model(*args, **kwargs):
        layout = env_layout()
        if layout is None or _DEPTH["peft"] or layout.tp == 1 or layout.pp > 1:
            # Under UNSLOTH_PP_SIZE tensor parallel comes after the split, at the Trainer.
            return fn(*args, **kwargs)
        _DEPTH["peft"] += 1
        try:
            out = fn(*args, **kwargs)
        finally:
            _DEPTH["peft"] -= 1
        model = _model_of(out)
        if model is not None:
            _apply(model, layout)
        return out

    get_peft_model._unsloth_tp_env = True
    return get_peft_model


def _levels(model):
    from .tp_native import _tp_chain
    try:
        return _tp_chain(model)
    except Exception:
        return [model]


def parallel_env_requested():
    return any(
        str(os.environ.get(name, "")).strip() not in ("", "1")
        for name in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE")
    )


def install_tensor_parallel_env_trainer_init():
    """Install the Trainer.__init__ wrapper at import: loads that all run inside model_init (and non-Fast*
    models such as encoders) would otherwise train in an unwrapped Trainer, without tensor / pipeline parallel."""
    if not parallel_env_requested():
        return False
    from transformers import Trainer

    from ._utils import patch_gradient_accumulation_fix

    patch_gradient_accumulation_fix(Trainer)
    return True


def tensor_parallel_env_trainer_pre(model):
    """Trainer.__init__ before anything else touches the model: the pipeline split (UNSLOTH_PP_SIZE), tensor
    parallel (if get_peft_model did not apply it) and the DDP-style dp dimension. Returns the layout (None
    without UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE)."""
    layout = env_layout()
    if layout is None or not isinstance(model, torch.nn.Module):
        return layout
    if layout.pp > 1:
        from .pp_env import pipeline_parallel_env_apply

        pipeline_parallel_env_apply(
            model, layout
        )  # also the DDP-style dp start (broadcast, stage.ddp_mesh)
        if layout.tp > 1:
            _apply(model, layout)
            if layout.grad_size > 1 and not layout.fsdp2:
                for level in _levels(model):
                    level.__dict__[_DDP_MESH] = layout.dp_mesh
        return layout
    _apply(model, layout)
    if layout.grad_size > 1 and not layout.fsdp2:
        levels = _levels(model)
        if not any(level.__dict__.get(_DDP_MESH) is not None for level in levels):
            dp_mesh = layout.dp_mesh
            _broadcast_trainable(model, dp_mesh)
            for level in levels:
                level.__dict__[_DDP_MESH] = dp_mesh
    return layout


def _broadcast_trainable(model, dp_mesh):
    """DDP's start: every dp replica begins from dp rank 0's trainable weights (same tp index)."""
    import torch.distributed as dist

    group = dp_mesh.get_group()
    src = dist.get_global_rank(group, 0)
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                local = getattr(p, "_local_tensor", p.data)
                work = (
                    local if local.is_contiguous() else local.contiguous()
                )  # NCCL takes dense tensors
                dist.broadcast(work, src = src, group = group)
                if work is not local:
                    local.copy_(work)


def fsdp2_dp_mesh(model):
    """The dp mesh native FSDP2 shards a tensor parallel model over under UNSLOTH_TP_SIZE + UNSLOTH_FSDP2=1."""
    layout = _LAYOUT if _LAYOUT is not _UNSET else None
    if (
        layout is None
        or layout.grad_size == 1
        or not layout.fsdp2
        or not isinstance(model, torch.nn.Module)
    ):
        return None
    from .pp_native import is_pipeline_parallel
    from .tp_native import is_tensor_parallel

    return layout.dp_mesh if is_tensor_parallel(model) or is_pipeline_parallel(model) else None


def fsdp2_has_no_dp_dimension():
    """UNSLOTH_FSDP2=1 with UNSLOTH_TP_SIZE = world size: nothing for FSDP2 to shard over."""
    layout = _LAYOUT if _LAYOUT is not _UNSET else None
    return layout is not None and layout.grad_size == 1
