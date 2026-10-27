# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tensor / pipeline parallelism from the environment, for unmodified training scripts.

    UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 2 train.py                    # tp 2
    UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 4 train.py                    # dp 2 (DDP-style) x tp 2
    UNSLOTH_TP_SIZE=2 UNSLOTH_FSDP2=1 torchrun --nproc-per-node 4 train.py    # dp 2 (native FSDP2) x tp 2
    UNSLOTH_PP_SIZE=2 torchrun --nproc-per-node 2 train.py                    # pp 2
    UNSLOTH_PP_SIZE=2 UNSLOTH_TP_SIZE=2 torchrun --nproc-per-node 4 train.py  # pp 2 x tp 2
    UNSLOTH_PP_SIZE=2 [UNSLOTH_FSDP2=1] torchrun --nproc-per-node 4 train.py  # pp 2 x dp 2 (DDP-style / FSDP2)

The layout is read once. The world splits into pp x dp x tp ranks (pp outermost, tp the inner dimension, as in
init_device_mesh(..., (pp, dp, tp), ("pp", "dp", "tp")), torchtitan's order). Tensor parallel alone:
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


def _int_env(name):
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError(f"Unsloth: {name}={raw!r} must be an integer.") from None
    if value < 1:
        raise ValueError(f"Unsloth: {name}={value} must be >= 1.")
    return value


class TPEnvLayout:
    """world = pp x dp x tp; `mesh()` builds (once) the device mesh, initializing the process group if needed."""

    def __init__(self, tp, dp, world, fsdp2, pp = 1, microbatches = None, schedule = "gpipe"):
        self.tp, self.dp, self.world, self.fsdp2 = tp, dp, world, fsdp2
        self.pp, self.microbatches, self.schedule = pp, microbatches, schedule
        self._mesh = None

    def __repr__(self):
        parts = [f"pp {self.pp}"] if self.pp > 1 else []
        if self.dp > 1:
            parts.append(f"dp {self.dp} ({'native FSDP2' if self.fsdp2 else 'DDP'})")
        if self.tp > 1:
            parts.append(f"tp {self.tp}")
        return " x ".join(parts)

    @property
    def env_vars(self):
        """The environment variables that asked for this layout, e.g. "UNSLOTH_PP_SIZE=2 UNSLOTH_TP_SIZE=2"."""
        out = [f"UNSLOTH_PP_SIZE={self.pp}"] if self.pp > 1 else []
        if self.tp > 1:
            out.append(f"UNSLOTH_TP_SIZE={self.tp}")
        return " ".join(out)

    @property
    def sharded_loading(self):
        if self.pp > 1:
            return False  # the pipeline split comes first, on whole layers (models/pp_env.py)
        return os.environ.get("UNSLOTH_TP_SHARDED_LOAD", "1").strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
        )

    def mesh(self):
        if self._mesh is not None:
            return self._mesh
        import torch.distributed as dist
        from torch.distributed.device_mesh import init_device_mesh

        cuda = torch.cuda.is_available()
        if cuda:
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if not dist.is_initialized():
            # What the Trainer's accelerate state would create; it reuses an existing group.
            dist.init_process_group("nccl" if cuda else "gloo")
        if dist.get_world_size() != self.world:
            raise RuntimeError(
                f"Unsloth: {self.env_vars} read WORLD_SIZE={self.world}, but the process group has "
                f"{dist.get_world_size()} ranks."
            )
        device = "cuda" if cuda else "cpu"
        dims = [(name, size) for name, size in (("pp", self.pp), ("dp", self.dp), ("tp", self.tp))]
        dims = [(name, size) for name, size in dims if size > 1] or [("tp", self.tp)]
        mesh = init_device_mesh(
            device, tuple(size for _, size in dims), mesh_dim_names = tuple(name for name, _ in dims)
        )
        if len(dims) > 1:
            from .pp_native import check_mesh_composition
            check_mesh_composition(
                {name: mesh[name] for name, _ in dims}, f"{self.env_vars} ({self})"
            )
        self._mesh = mesh
        if dist.get_rank() == 0:
            what = "pipeline" if self.pp > 1 else "tensor"
            if self.pp > 1 and self.tp > 1:
                what = "pipeline x tensor"
            print(f"Unsloth: {what} parallel from {self.env_vars}: {self} over {self.world} ranks.")
        return mesh

    @property
    def tp_mesh(self):
        return self.mesh()["tp"] if self.tp > 1 else None

    @property
    def dp_mesh(self):
        return self.mesh()["dp"] if self.dp > 1 else None

    @property
    def pp_mesh(self):
        return self.mesh()["pp"] if self.pp > 1 else None


_PP_SCHEDULES = ("gpipe", "1f1b")


def _read_layout():
    pp = _int_env("UNSLOTH_PP_SIZE") or 1
    tp = _int_env("UNSLOTH_TP_SIZE") or 1
    if pp == 1 and tp == 1:
        return None
    try:
        world = int(os.environ.get("WORLD_SIZE", "1"))
    except ValueError:
        world = 1
    names = " x ".join(
        name for name, size in (("UNSLOTH_PP_SIZE", pp), ("UNSLOTH_TP_SIZE", tp)) if size > 1
    )
    given = " ".join(
        f"{name}={size}" for name, size in (("UNSLOTH_PP_SIZE", pp), ("UNSLOTH_TP_SIZE", tp)) if size > 1
    )
    if world % (pp * tp):
        what = (
            "data parallel x tensor parallel"
            if pp == 1
            else "data parallel x pipeline parallel" + (" x tensor parallel" if tp > 1 else "")
        )
        raise ValueError(
            f"Unsloth: {given}: {names} = {pp * tp} must divide the number of ranks (WORLD_SIZE={world}); "
            f"launch with torchrun --nproc-per-node {pp * tp} (or a multiple of {pp * tp} for {what}), or "
            f"unset {' / '.join(name for name in names.split(' x '))}."
        )
    microbatches, schedule = None, "gpipe"
    if pp > 1:
        microbatches = _int_env("UNSLOTH_PP_MICROBATCHES")
        schedule = (os.environ.get("UNSLOTH_PP_SCHEDULE") or "gpipe").strip().lower().replace("-", "")
        schedule = schedule.replace("_", "")
        if schedule not in _PP_SCHEDULES:
            raise ValueError(
                f"Unsloth: UNSLOTH_PP_SCHEDULE={os.environ.get('UNSLOTH_PP_SCHEDULE')!r} must be one of "
                f"{_PP_SCHEDULES}."
            )
        if schedule == "1f1b" and microbatches is not None and microbatches < pp:
            raise ValueError(
                f"Unsloth: UNSLOTH_PP_SCHEDULE=1f1b needs UNSLOTH_PP_MICROBATCHES >= UNSLOTH_PP_SIZE "
                f"({microbatches} < {pp})."
            )
        from .pp_native import _check_torch_pipelining

        try:
            _check_torch_pipelining()
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} (UNSLOTH_PP_SIZE={pp}: unset it to run without pipeline parallel.)") from None
    from ..fsdp2_launch import unsloth_fsdp2_requested

    fsdp2 = bool(unsloth_fsdp2_requested())
    backend = os.environ.get("UNSLOTH_FSDP2_BACKEND", "").strip().lower()
    if fsdp2 and backend in ("accelerate", "hf", "transformers"):
        what = "pipeline" if pp > 1 else "tensor"
        raise ValueError(
            f"Unsloth: {given} composes with native FSDP2 only (UNSLOTH_FSDP2=1); "
            f"UNSLOTH_FSDP2_BACKEND={backend} would hand the {what} parallel model to accelerate's FSDP. "
            "Unset UNSLOTH_FSDP2_BACKEND."
        )
    return TPEnvLayout(tp, world // (pp * tp), world, fsdp2, pp, microbatches, schedule)


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
    ("full_finetuning", "full finetuning (full_finetuning = True; use UNSLOTH_FSDP2=1 instead)"),
    ("load_in_8bit", "8-bit weights (load_in_8bit = True; load in 4-bit or 16-bit)"),
    ("load_in_fp8", "fp8 weights (load_in_fp8)"),
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
                + ("; tensor parallel already loads 16-bit weights sharded." if layout.pp == 1 else ".")
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
            if _four_bit(values, fn) or not layout.sharded_loading or _is_adapter(values):
                out = fn(*args, **kwargs)  # full load; get_peft_model shards it (Params4bit included)
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


def install_tensor_parallel_env_trainer_init():
    """Install the Trainer.__init__ wrapper at import: loads that all run inside model_init would otherwise
    train in an unwrapped Trainer, without tensor / pipeline parallel."""
    if all(str(os.environ.get(k, "")).strip() in ("", "1") for k in ("UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE")):
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

        pipeline_parallel_env_apply(model, layout)  # also the DDP-style dp start (broadcast, stage.ddp_mesh)
        if layout.tp > 1:
            _apply(model, layout)
            if layout.dp > 1 and not layout.fsdp2:
                for level in _levels(model):
                    level.__dict__[_DDP_MESH] = layout.dp_mesh
        return layout
    _apply(model, layout)
    if layout.dp > 1 and not layout.fsdp2:
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
        or layout.dp == 1
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
    return layout is not None and layout.dp == 1
