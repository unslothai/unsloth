# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallelism from the environment (UNSLOTH_PP_SIZE, models/tp_env.py) for unmodified scripts.

    UNSLOTH_PP_SIZE=2 torchrun --nproc-per-node 2 train.py

- Trainer.__init__ splits the model (apply_pipeline_parallel over mesh["pp"]); the other stages' layers,
  embedding and lm_head go to this rank's CPU memory instead of being dropped (PPConfig.keep_replica). A
  checkpoint larger than the GPU loads only each rank's stages (pipeline_sharded_loading): no replica then.
- Trainer / SFTTrainer: every training step runs through the pipeline schedule (PipelineParallel.step runs the
  backward, the Trainer does not). The Trainer batch is split into UNSLOTH_PP_MICROBATCHES micro-batches
  (default 2 x pp, lowered to a divisor of the batch) under UNSLOTH_PP_SCHEDULE (default 1f1b; a
  step with fewer micro-batches than stages, or a MoE step with the router aux loss on, runs as gpipe: the aux loss
  is formed over the whole batch, as one GPU does); every micro-batch loss is its summed token loss over the
  token count of the whole accumulation window over every dp replica, as with gradient accumulation on one
  GPU. With gradient_accumulation_steps > 1 the window's batches wait and run together, as micro-batches
  of the same row count, in as few schedule steps as padding allows (rows grouped by length), so the
  pipeline fills and drains once per optimizer step instead of once per batch; UNSLOTH_PP_FOLD_GA=0 runs
  one schedule step per batch. The pp (and tp) ranks of one dp replica read the same batches; the optimizer
  steps each stage's own parameters; clipping, the fp16 GradScaler skip and the logged loss are global.
  Evaluation runs the pipeline forward. Trainer checkpoints: the 1-GPU adapter (rank 0) plus per-rank optimizer state.
- for_inference / generate / merged and GGUF exports: each rank puts the whole model back on its GPU (the CPU
  copies of the other stages plus their trained weights, broadcast from the stage that trained them), then runs
  the one-GPU path; the next training step splits it again. Exports run on rank 0 only; the other ranks wait
  on the rendezvous store (no collective timeout). An adapter save_pretrained gathers the LoRA weights only.
- GRPO / DPO / other RL trainers, vLLM and full finetuning are refused (v1).
"""

from __future__ import annotations

import collections.abc
import functools
import gc
import os
import types

import torch
import torch.nn as nn

__all__ = [
    "pipeline_parallel_env_apply",
    "offload_kept",
    "gather_replica",
    "release_replica",
    "replica_generate",
    "pp_for_inference",
    "pp_for_training",
    "pp_collective",
    "pp_save_adapter",
    "pp_trainer_refusals",
    "pp_trainer_setup",
]

_MiB = 1 << 20
_GENERATE_DEPTH = [0]
# Trainers whose loss the pipeline computes the way they do (causal LM loss on `labels`).
_ALLOWED_TRAINERS = {
    "Trainer",
    "Seq2SeqTrainer",
    "SFTTrainer",
    "UnslothSFTTrainer",
    "_UnslothSFTTrainer",
}
# Batch keys the pipeline carries (or that only shaped `labels`, already applied by the collator).
_BATCH_KEYS = ("input_ids", "labels", "attention_mask", "position_ids")
_DROPPED_KEYS = ("completion_mask", "assistant_masks", "return_loss", "num_items_in_batch")
_TEXT_ONLY = "_unsloth_pp_text_only"
_MEDIA_COLUMNS = ("image", "images", "video", "videos", "audio", "audios") + (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "input_features",
)
_MEDIA_PART_TYPES = ("image", "image_url", "video", "audio", "input_audio")


def _dist():
    import torch.distributed as dist
    return dist if dist.is_available() and dist.is_initialized() else None


def _rank():
    dist = _dist()
    return dist.get_rank() if dist is not None else 0


def _print_rank0(message):
    if _rank() == 0:
        print(message)


def _gib(n):
    return f"{n / (1 << 30):.2f} GiB" if n >= (1 << 30) // 10 else f"{n / _MiB:.1f} MiB"


def _hint():
    from .tp_env import env_hint
    return env_hint()


def _stage_of(model):
    from .pp_native import PipelineParallel, pipeline_stage_of
    return model if isinstance(model, PipelineParallel) else pipeline_stage_of(model)


def _env_vars(stage):
    layout = getattr(stage, "env_layout", None)
    return layout.env_vars if layout is not None else f"UNSLOTH_PP_SIZE={stage.num_stages}"


def pipeline_parallel_env_apply(model, layout):
    """Trainer.__init__ under UNSLOTH_PP_SIZE: split `model` over mesh["pp"] (once) and attach the DDP-style dp
    dimension. Native FSDP2 (UNSLOTH_FSDP2=1) shards the stage over mesh["dp"] next (fsdp2_native)."""
    from .pp_native import PPConfig, apply_pipeline_parallel, pipeline_stage_of

    stage = pipeline_stage_of(model)
    if stage is not None:
        if getattr(stage, "env_layout", None) is None:
            raise RuntimeError(
                "Unsloth: this model was split by apply_pipeline_parallel already; under "
                f"{layout.env_vars} pass the Trainer the model loaded and given LoRA as usual."
            )
        return stage
    keep = os.environ.get("UNSLOTH_PP_KEEP_REPLICA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )
    from .pp_load import _preload_info

    if _preload_info(model)[1] is not None:
        keep = False  # loaded per stage (larger than the GPU): there is no whole model to keep
    default = layout.microbatches or 2 * layout.pp
    schedule = layout.schedule
    if getattr(layout, "schedule_auto", False) and schedule == "1f1b" and default < layout.pp:
        schedule = "gpipe"  # UNSLOTH_PP_MICROBATCHES below the stage count: every step would fall back anyway
    text_only = None
    for level in _model_levels(model):
        text_only = level.__dict__.get(_TEXT_ONLY, text_only)
    try:
        stage = apply_pipeline_parallel(
            model,
            layout.pp_mesh,
            n_microbatches = default,
            schedule = schedule,
            config = PPConfig(
                keep_replica = keep,
                text_only = text_only,
                stages_per_rank = getattr(layout, "stages_per_rank", None),
            ),
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        from .tp_env import _reraise_with_hint
        _reraise_with_hint(exc)
    stage.env_layout = layout
    stage.schedule_auto = bool(getattr(layout, "schedule_auto", False))
    if layout.dp > 1 and not layout.fsdp2:
        from .tp_env import _broadcast_trainable

        dp_mesh = layout.dp_mesh
        _broadcast_trainable(model, dp_mesh)  # DDP's start: dp rank 0's adapter on every replica
        stage.ddp_mesh = dp_mesh
    return stage


@functools.lru_cache(maxsize = None)
def _gpipe_needs_stage_count():
    """torch 2.10's single-stage schedules (GPipe too) refuse fewer micro-batches than stages; later ones
    only 1F1B."""
    import inspect

    from torch.distributed.pipelining import ScheduleGPipe

    for cls in ScheduleGPipe.__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        try:
            if "must be greater than" in inspect.getsource(init):
                return True
        except (OSError, TypeError):
            continue
    return False


def resolve_microbatches(
    stage,
    batch_size,
    train = True,
):
    """(n_microbatches, padded rows) for a batch of `batch_size` rows. UNSLOTH_PP_MICROBATCHES must divide it;
    the default (2 x pp) is lowered to the largest divisor of the batch. A schedule that needs at least pp
    micro-batches (1F1B; GPipe on torch 2.10) gets fully masked filler rows (labels -100: no loss, no gradient)
    when the batch has fewer rows than that."""
    layout = getattr(stage, "env_layout", None)
    explicit = layout.microbatches if layout is not None else None
    want = int(explicit or stage.n_microbatches)
    m, pad = _plan_microbatches(
        want,
        explicit,
        stage.num_stages,
        stage.schedule,
        batch_size,
        train,
        auto = getattr(stage, "schedule_auto", False),
    )
    if m == want and not pad:
        return m, 0
    if train and not stage.__dict__.get("_unsloth_mb_note"):
        stage.__dict__["_unsloth_mb_note"] = True
        rows = batch_size + pad
        floor = m if pad else 1
        note = (
            f", padded with {pad} fully masked row(s) (this schedule needs {floor} micro-batches; a "
            f"per_device_train_batch_size of {floor} or more avoids the extra compute)"
            if pad
            else ""
        )
        _print_rank0(
            f"Unsloth: pipeline parallel: {m} micro-batch(es) of {rows // m} row(s) per step (batch of "
            f"{batch_size}{note}; UNSLOTH_PP_MICROBATCHES sets the count)."
        )
    return m, pad


def _plan_microbatches(
    want,
    explicit,
    num_stages,
    schedule,
    batch_size,
    train,
    auto = False,
):
    """(n_microbatches, padded rows); shared with the Trainer init refusals so they predict the per-step count.
    `auto` (the default 1F1B, UNSLOTH_PP_SCHEDULE unset): a step with fewer micro-batches than stages runs as
    GPipe instead of padding."""
    floor = (
        num_stages
        if (schedule == "1f1b" and train and not auto) or _gpipe_needs_stage_count()
        else 1
    )
    if batch_size % want == 0:
        m = want
    elif explicit is not None and train:
        raise ValueError(
            f"Unsloth: UNSLOTH_PP_MICROBATCHES={explicit} must divide the Trainer batch size ({batch_size} rows "
            "per device): set per_device_train_batch_size to a multiple of it, or unset UNSLOTH_PP_MICROBATCHES."
        )
    else:
        m = max(d for d in range(1, min(want, batch_size) + 1) if batch_size % d == 0)
    pad = 0
    if m < floor:
        # Whole micro-batches of filler rows after the real ones (the real rows keep their micro-batch shape).
        pad = (floor - m) * (batch_size // m)
        m = floor
    return m, pad


def _pad_rows(input_ids, labels, extra, pad):
    """`pad` copies of the first row with every label ignored (attention on, so no fully masked row)."""
    if not pad:
        return input_ids, labels, extra
    rows = input_ids.shape[0]

    def take(t):
        return t[:1].expand(pad, *t.shape[1:])

    out = {}
    for key, value in extra.items():
        if isinstance(value, torch.Tensor) and value.dim() and value.shape[0] == rows:
            filler = torch.ones_like(take(value)) if key == "attention_mask" else take(value)
            value = torch.cat([value, filler])
        out[key] = value
    labels = torch.cat([labels, torch.full_like(take(labels), -100)])
    return torch.cat([input_ids, take(input_ids)]), labels, out


def _move(module, device, skip):
    """Every parameter / buffer of `module` not in `skip` (ids) to `device`, in place (Params4bit with its
    QuantState)."""
    for sub in module.modules():
        for param in sub._parameters.values():
            if param is None or id(param) in skip:
                continue
            if param.device != device:
                param.data = param.data.to(device)
            state = getattr(param, "quant_state", None)
            if state is not None and hasattr(state, "to"):
                state.to(device)
        for name, buf in list(sub._buffers.items()):
            if buf is not None and id(buf) not in skip and buf.device != device:
                sub._buffers[name] = buf.to(device)


def _put(container, key, module):
    if isinstance(container, nn.ModuleList):
        container[key] = module
    else:
        setattr(container, key, module)


def _live_ids(stage):
    out = {id(p) for p in stage.model.parameters()}
    out.update(id(b) for b in stage.model.buffers())
    return out


def offload_kept(stage):
    """The kept modules of the other stages to the CPU; tensors this stage still uses (tied weights) stay."""
    live = _live_ids(stage)
    for _, _, module, _ in stage._kept or ():
        _move(module, torch.device("cpu"), live)


def _tensor_bytes(t):
    return t.numel() * t.element_size() if isinstance(t, torch.Tensor) else 0


def _kept_bytes(stage):
    """Bytes the kept modules occupy on the CPU (what a whole replica adds to this GPU)."""
    seen, total = set(), 0
    for _, _, module, _ in stage._kept or ():
        for t in list(module.parameters()) + list(module.buffers()):
            if id(t) in seen or t.device.type != "cpu":
                continue
            seen.add(id(t))
            total += _tensor_bytes(t)
            state = getattr(t, "quant_state", None)
            if state is not None:
                for value in vars(state).values():
                    total += _tensor_bytes(value)
    return total


def _free_bytes(device):
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return free + max(torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device), 0)


def _comm_device(stage):
    from .tp_replica import _comm_device
    return _comm_device(stage.group, stage.device)


def replica_refusal(stage, what):
    """None when the whole model can be put back on each rank for `what`, else why not."""
    if stage is None:
        return f"Unsloth: `{what}` found no pipeline stage on this model."
    hint = _hint()
    if stage._kept is None:
        return (
            f"Unsloth: `{what}` needs the whole model, but this pipeline parallel model dropped the other "
            "stages' layers (apply_pipeline_parallel without PPConfig(keep_replica = True), or "
            "UNSLOTH_PP_KEEP_REPLICA=0). Save the adapter (model.save_pretrained / PipelineParallel.save_adapter) "
            f"and run `{what}` on one GPU.{hint}"
        )
    if stage.tp_mesh is not None or stage.dp_mesh is not None:
        dim = (
            "tensor parallel (UNSLOTH_TP_SIZE)"
            if stage.tp_mesh is not None
            else "native FSDP2 data parallel"
        )
        return (
            f"Unsloth: `{what}` is not supported yet on a pipeline parallel x {dim} model (each stage's weights "
            "are sharded over its other ranks). Save the adapter (model.save_pretrained, on every rank) and run "
            f"`{what}` on one GPU, or launch with UNSLOTH_PP_SIZE alone (DDP-style data parallel).{hint}"
        )
    return None


def gather_replica(
    model,
    what = "generate",
    persistent = True,
):
    """Collective: restore the whole model on every rank from kept CPU copies and every stage's
    trained weights; refused on all ranks alike if any lacks room. Returns the stage."""
    dist = _dist()
    stage = _stage_of(model)
    refusal = replica_refusal(stage, what)
    if refusal is not None:
        raise RuntimeError(refusal)
    if stage.replica is not None:
        if persistent:
            stage.replica["persistent"] = True
        return stage
    device = stage.device
    need = _kept_bytes(stage)
    free = _free_bytes(device)
    margin = max(512 * _MiB, need // 10)
    fits = free is None or need + margin <= free
    if dist is not None:
        comm = _comm_device(stage)
        row = torch.tensor(
            [float(need), float(free if free is not None else -1), 1.0 if fits else 0.0],
            dtype = torch.float64,
            device = comm,
        )
        rows = [torch.empty_like(row) for _ in range(dist.get_world_size())]
        dist.all_gather(rows, row)
        rows = [r.cpu().tolist() for r in rows]
    else:
        rows = [[float(need), float(free if free is not None else -1), 1.0 if fits else 0.0]]
    short = [(rank, r) for rank, r in enumerate(rows) if r[2] < 1]
    if short:
        detail = "; ".join(
            f"rank {rank} needs {_gib(int(r[0]))} (+ {_gib(max(512 * _MiB, int(r[0]) // 10))} margin) and has "
            f"{_gib(int(r[1]))} free"
            for rank, r in short[:4]
        )
        raise RuntimeError(
            f"Unsloth: `{what}` on a pipeline parallel model puts the whole model back on every GPU, which does "
            f"not fit: {detail}. Save the adapter (model.save_pretrained) and run `{what}` on one GPU, or free "
            f"memory first.{_hint()}"
        )
    owned = _trained_names(stage)
    adapters = _adapter_state(stage.model)
    for container, key, module, _ in stage._kept:
        _move(module, device, set())
        # Detached while the caller set the mode (model.eval() before generate): follow the root's mode.
        module.train(stage.model.training)
        # The same for PEFT's adapter switches (disable_adapter(), set_adapter) made while they were detached.
        _apply_adapter_state(module, adapters)
        _put(container, key, module)
    stage.replica = {"persistent": persistent}
    try:
        _sync_trained(stage, owned)
    except BaseException:
        release_replica(stage)
        raise
    if _rank() == 0 and not stage.__dict__.get("_unsloth_replica_note"):
        stage.__dict__["_unsloth_replica_note"] = True
        print(
            f"Unsloth: pipeline parallel `{what}`: the whole model is back on every rank ({_gib(need)} from the "
            "CPU copies of the other stages); the next training step splits it again."
        )
    return stage


def _adapter_state(model):
    """(disabled, active adapters) of the attached PEFT tuner layers, or None without any."""
    try:
        from peft.tuners.tuners_utils import BaseTunerLayer
    except ImportError:
        return None
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            return bool(module.disable_adapters), module.active_adapter
    return None


def _apply_adapter_state(root, state):
    # PEFT's private fields, not enable_adapters / set_adapter: those also flip requires_grad on the adapters.
    if state is None:
        return
    from peft.tuners.tuners_utils import BaseTunerLayer

    disabled, active = state
    for module in root.modules():
        if isinstance(module, BaseTunerLayer):
            module._disable_adapters = disabled
            module._active_adapter = active


def _note_trained(stage):
    """Remember trained params; a later requires_grad_(False) must not drop them from the sync."""
    seen = stage.__dict__.setdefault("_unsloth_trained_names", set())
    seen.update(name for name, p in stage.model.named_parameters() if p.requires_grad)


def _trained_names(stage):
    seen = stage.__dict__.get("_unsloth_trained_names", ())
    return [name for name, p in stage.model.named_parameters() if p.requires_grad or name in seen]


def _sync_trained(stage, owned):
    """Broadcast every trained parameter from the stage that trained it (same names on every rank now)."""
    dist = _dist()
    if dist is None or stage.num_stages == 1:
        return
    names = [None] * stage.num_stages
    dist.all_gather_object(names, owned, group = stage.group)
    owner = {}
    for index, part in enumerate(names):
        for name in part:
            owner.setdefault(name, index)
    with torch.no_grad():
        for name, param in stage.model.named_parameters():
            src = owner.get(name)
            if src is None:
                continue
            data = param.data
            work = data if data.is_contiguous() else data.contiguous()
            dist.broadcast(work, src = stage._global_ranks[src], group = stage.group)
            if work is not data:
                data.copy_(work)


def release_replica(model):
    """Back to this rank's stage (local, no collective): the kept modules to the CPU, the stubs in place."""
    stage = _stage_of(model)
    if stage is None or stage.replica is None:
        return False
    stage.replica = None
    for container, key, _, stub in stage._kept:
        _put(container, key, stub)
    offload_kept(stage)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return True


def _sync_generation_rng(stage):
    """Global rank 0's RNG states everywhere: every rank samples the same tokens."""
    dist = _dist()
    if dist is None:
        return
    comm = _comm_device(stage)
    cpu = torch.get_rng_state().to(comm)
    dist.broadcast(cpu, src = 0)
    torch.set_rng_state(cpu.cpu())
    device = stage.device
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device).to(comm)
        dist.broadcast(state, src = 0)
        torch.cuda.set_rng_state(state.cpu(), device)


def replica_generate(level, prior):
    """generate on a pipeline parallel level: every rank puts the whole model back first (it stays until the
    next training step), then the one-GPU generate runs locally."""

    def generate(*args, **kwargs):
        if _GENERATE_DEPTH[0] == 0:
            stage = gather_replica(level, "generate")
            _sync_generation_rng(stage)
        if prior is not None:
            fn = prior
        elif hasattr(type(level), "generate"):
            fn = types.MethodType(type(level).generate, level)
        else:  # a forwarding wrapper (PEFT LoraModel): what its __getattr__ resolves, the level below
            fn = type(level).__getattr__(level, "generate")
        _GENERATE_DEPTH[0] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _GENERATE_DEPTH[0] -= 1

    generate._unsloth_pp_generate = True
    return generate


def pp_for_inference(model):
    gather_replica(model, "for_inference")


def pp_for_training(model):
    release_replica(model)


def pp_collective(model, what, call):
    """Every rank calls: the whole model goes back on every rank, rank 0 runs `call` (the unchanged
    single-process export), the others wait on the store and return its result."""
    from . import tp_replica

    if tp_replica._ACTIVE or _dist() is None:
        return call()
    stage = _stage_of(model)
    gathered = stage is not None and stage.replica is None
    gather_replica(model, what, persistent = False)
    key = tp_replica._key("pp_save")
    try:
        return tp_replica._on_rank0(key, what, call, model)
    finally:
        if gathered and stage.replica is not None and not stage.replica["persistent"]:
            release_replica(stage)


def pp_save_adapter(
    model,
    args,
    kwargs,
    cast_lora = False,
):
    """PeftModel.save_pretrained on a pipeline parallel model: every stage's adapter weights gathered, the
    standard 1-GPU adapter written by rank 0 (collective). `cast_lora`: the LoRA tensors are written in the
    input embedding dtype, as `model.save_lora` does on one GPU."""
    import inspect

    from peft import PeftModel

    stage = _stage_of(model)
    try:
        bound = inspect.signature(PeftModel.save_pretrained).bind(model, *args, **kwargs)
    except TypeError:
        bound = None
    if bound is None:
        raise TypeError(
            f"Unsloth: save_pretrained{args} could not be read as PeftModel.save_pretrained."
        )
    values = dict(bound.arguments)
    values.pop("self", None)
    extra = values.pop("kwargs", {}) or {}
    directory = values.pop("save_directory")
    for key in (
        "push_to_hub",
        "token",
        "state_dict",
        "save_function",
        "max_shard_size",
        "tokenizer",
    ):
        extra.pop(key, None)
    if values.get("safe_serialization", True) is None:
        values["safe_serialization"] = True
    values.pop("is_main_process", None)
    if cast_lora:
        extra["_unsloth_cast_lora"] = True
    stage.save_adapter(directory, **values, **extra)
    return None


def _model_levels(model):
    from .pp_native import _model_chain
    try:
        return _model_chain(model)
    except Exception:
        return [model]


def _media_in_rows(dataset, rows = 16):
    try:
        n = min(len(dataset), rows)
    except TypeError:
        return None
    for i in range(n):
        try:
            row = dataset[i]
        except Exception:
            return None
        if not isinstance(row, collections.abc.Mapping):
            return None
        for key in ("messages", "conversations", "prompt", "completion"):
            turns = row.get(key)
            if not isinstance(turns, (list, tuple)):
                continue
            for turn in turns:
                content = turn.get("content") if isinstance(turn, collections.abc.Mapping) else None
                for part in content if isinstance(content, (list, tuple)) else ():
                    kind = part.get("type") if isinstance(part, collections.abc.Mapping) else None
                    if kind in _MEDIA_PART_TYPES:
                        return f"row {i} {key} holds a {kind!r} part"
    return None


def media_evidence(
    train_dataset = None,
    eval_dataset = None,
    data_collator = None,
):
    """Why this Trainer would feed media (columns, chat parts, collator), else None."""
    if data_collator is not None:
        name = type(data_collator).__name__
        if any(word in name for word in ("Vision", "Audio", "MultiModal", "Multimodal")):
            return f"data_collator {name}"
    for what, dataset in (("train_dataset", train_dataset), ("eval_dataset", eval_dataset)):
        datasets = dataset.items() if isinstance(dataset, dict) else [(None, dataset)]
        for split, ds in datasets:
            if ds is None:
                continue
            label = what if split is None else f"{what}[{split!r}]"
            columns = getattr(ds, "column_names", None)
            if isinstance(columns, (list, tuple)):
                media = [c for c in columns if c in _MEDIA_COLUMNS]
                if media:
                    return f"{label} column {media[0]!r}"
            found = _media_in_rows(ds)
            if found is not None:
                return f"{label} {found}"
    return None


def _resolve_text_only(layout, model, train_dataset, eval_dataset, data_collator):
    """UNSLOTH_PP_TEXT_ONLY=1 / 0 decides; unset, the dataset and collator do."""
    if not isinstance(model, nn.Module):
        return
    from .pp_native import _media_keys, _unwrap

    inner = _unwrap(model)
    media = _media_keys(inner)
    if not media:
        return
    flag = os.environ.get("UNSLOTH_PP_TEXT_ONLY", "").strip().lower()
    evidence = media_evidence(train_dataset, eval_dataset, data_collator)
    if flag in ("0", "false", "no", "off") or (
        evidence is not None and flag not in ("1", "true", "yes", "on")
    ):
        why = (
            f"the run feeds media ({evidence})"
            if evidence is not None
            else "UNSLOTH_PP_TEXT_ONLY=0"
        )
        raise NotImplementedError(
            f"Unsloth: {layout.env_vars} (pipeline parallel) trains {type(inner).__name__} on text only (its "
            f"{', '.join(media)} towers stay out of the pipeline), but {why}. Use native FSDP2 (UNSLOTH_FSDP2=1) "
            f"for multimodal training.{_hint()}"
        )
    for level in _model_levels(model):
        level.__dict__[_TEXT_ONLY] = True


def _trainer_classes(cls):
    return [c for c in cls.__mro__ if c is not object]


def pp_trainer_refusals(
    trainer_cls,
    args,
    layout,
    compute_loss_func = None,
    model = None,
    train_dataset = None,
    eval_dataset = None,
    data_collator = None,
):
    """Before Trainer.__init__ splits anything: the trainers / arguments the pipeline step cannot honour."""
    hint = _hint()
    builtin = ("transformers.", "trl.", "unsloth", "Unsloth")
    kind = next(
        (
            c
            for c in _trainer_classes(trainer_cls)
            if c.__module__.startswith(builtin) or c.__name__.startswith("Unsloth")
        ),
        None,
    )
    if kind is not None and kind.__name__ not in _ALLOWED_TRAINERS:
        raise NotImplementedError(
            f"Unsloth: {layout.env_vars} (pipeline parallel) trains with the Trainer / SFTTrainer only in v1; "
            f"{kind.__name__} (GRPO / DPO / other RL trainers need the whole model per rank for their rollouts / "
            f"reference logits) is not supported.{hint}"
        )
    for cls in _trainer_classes(trainer_cls):
        if cls.__name__ in _ALLOWED_TRAINERS:
            break
        for method in ("compute_loss", "training_step"):
            if method in cls.__dict__:
                raise NotImplementedError(
                    f"Unsloth: {layout.env_vars} (pipeline parallel): {cls.__name__} overrides {method}, but under "
                    "pipeline parallel the loss is computed by the pipeline schedule (the causal LM loss on "
                    f"`labels`), so the override would be skipped.{hint}"
                )
    if compute_loss_func is not None:
        raise NotImplementedError(
            f"Unsloth: {layout.env_vars} (pipeline parallel): Trainer(compute_loss_func = ...) would be skipped, "
            f"the pipeline schedule computes the causal LM loss on `labels`.{hint}"
        )
    _resolve_text_only(layout, model, train_dataset, eval_dataset, data_collator)
    problems = []
    if args is None:
        return
    if getattr(args, "label_smoothing_factor", 0) or 0:
        problems.append("label_smoothing_factor (the pipeline computes the plain causal LM loss)")
    loss_type = getattr(args, "loss_type", None)
    if isinstance(loss_type, str) and loss_type.lower() != "nll" and "SFT" in type(args).__name__:
        # TRL's SFTTrainer.compute_loss applies DFT (and later variants); the pipeline schedule never calls it.
        problems.append(
            f'SFTConfig(loss_type = "{loss_type}") (the pipeline computes the plain causal LM loss; use "nll")'
        )
    if getattr(args, "deepspeed", None):
        problems.append("DeepSpeed")
    if getattr(args, "fsdp", None) or getattr(args, "parallelism_config", None) is not None:
        problems.append(
            "accelerate FSDP / ParallelismConfig (UNSLOTH_FSDP2=1 shards the dp dimension)"
        )
    for flag in ("load_best_model_at_end", "push_to_hub"):
        if getattr(args, flag, False):
            problems.append(flag)
    config = getattr(args, "accelerator_config", None)
    if getattr(config, "dispatch_batches", None):
        problems.append("accelerator_config.dispatch_batches")
    if getattr(args, "packing", False) or getattr(args, "padding_free", False):
        problems.append("packing / padding_free (packed rows are not carried between stages yet)")
    if problems:
        raise NotImplementedError(
            f"Unsloth: the Trainer under {layout.env_vars} (pipeline parallel) does not support "
            + "; ".join(problems)
            + "."
            + hint
        )
    _refuse_split_router_aux_at_init(trainer_cls, args, layout, model, train_dataset, eval_dataset)
    _refuse_uncarried_router_aux_at_init(trainer_cls, args, model)


def _refuse_uncarried_router_aux_at_init(trainer_cls, args, model):
    """Refuse a Trainer that turns on an uncarried MoE router aux loss. TRL >= 1.x sets it later in __init__
    (coef None = architecture's), so the stage forward re-checks."""
    if not isinstance(model, nn.Module):
        return
    from ..parallel_capabilities import _trainer_kind
    from .pp_native import (
        _router_aux_coef_of,
        _text_config,
        _uncarried_moe_owner,
        _uncarried_router_aux_message,
    )

    try:
        inner = _uncarried_moe_owner(model)
    except Exception:
        return  # the split reports what it cannot find
    if inner is None:
        return
    text = _text_config(model)
    on = bool(getattr(text, "output_router_logits", False))
    trl = (
        args is not None
        and _trainer_kind(trainer_cls) in ("sft", "dpo", "grpo", "other_rl")
        and hasattr(args, "router_aux_loss_coef")
        and hasattr(text, "router_aux_loss_coef")
    )
    if trl:
        coef = args.router_aux_loss_coef
        coef = (getattr(text, "router_aux_loss_coef", 0) or 0) if coef is None else coef
        on = hasattr(text, "output_router_logits") and bool(coef)
    if not on or _router_aux_coef_of(inner, text) == 0:
        return
    raise NotImplementedError(
        _uncarried_router_aux_message(
            inner,
            "this Trainer turns it on"
            + (
                f" (router_aux_loss_coef = {args.router_aux_loss_coef!r} in {type(args).__name__})"
                if trl
                else " (config.output_router_logits = True)"
            ),
        )
        + _hint()
    )


def _router_aux_coef_at_init(trainer_cls, args, model):
    """Router aux coefficient the last stage will apply (0 = none). TRL trainers rewrite output_router_logits and
    router_aux_loss_coef inside their own __init__ (after this check) from args.router_aux_loss_coef (None = the
    architecture's)."""
    from ..parallel_capabilities import _trainer_kind
    from .pp_native import _PP_MOE_AUX_FAMILIES

    if not isinstance(model, nn.Module):
        return 0
    owner = next((m for m in model.modules() if type(m).__name__ in _PP_MOE_AUX_FAMILIES), None)
    config = getattr(owner, "config", None)
    if config is None:
        return 0  # the split refuses other MoE models
    text = config.get_text_config() if hasattr(config, "get_text_config") else config
    coef = getattr(owner, "router_aux_loss_coef", getattr(text, "router_aux_loss_coef", 0)) or 0
    on = bool(getattr(text, "output_router_logits", False))
    if (
        _trainer_kind(trainer_cls) in ("sft", "dpo", "grpo", "other_rl")
        and args is not None
        and hasattr(args, "router_aux_loss_coef")
        and hasattr(text, "router_aux_loss_coef")
    ):
        trl = args.router_aux_loss_coef
        coef = (getattr(text, "router_aux_loss_coef", 0) or 0) if trl is None else trl
        on = hasattr(text, "output_router_logits") and bool(coef)
    return coef if on else 0


def _largest_batch(dataset, batch_size, dp, packing):
    """`batch_size` when every rank surely gets a full batch from `dataset`, else None (the step decides): no
    dataset (train() / evaluate() cannot run on it), no len(), packing (TRL rebuilds the rows inside __init__),
    or fewer rows than one batch per dp rank (the dataloader's sharding / padding sets the size then)."""
    batch_size = int(batch_size or 0)
    if dataset is None or batch_size < 1 or packing:
        return None
    if isinstance(dataset, dict):
        sizes = [_largest_batch(d, batch_size, dp, packing) for d in dataset.values()]
        return max((size for size in sizes if size), default = None)
    try:
        rows = len(dataset)
    except TypeError:
        return None
    return batch_size if rows >= batch_size * max(int(dp or 1), 1) else None


def _refuse_split_router_aux_at_init(
    trainer_cls,
    args,
    layout,
    model,
    train_dataset = None,
    eval_dataset = None,
):
    """_refuse_split_router_aux before the stage split; the per-step check covers what only the step knows."""
    if args is None or not _router_aux_coef_at_init(trainer_cls, args, model):
        return
    try:
        from .pp_schedules import schedule_spec
        schedule = schedule_spec(layout.schedule).name
    except Exception:
        schedule = layout.schedule
    want = int(layout.microbatches or 2 * layout.pp)
    dp = getattr(layout, "dp", 1)
    packing = bool(getattr(args, "packing", False))
    eval_packing = getattr(args, "eval_packing", None)
    batches = [
        (
            _largest_batch(
                train_dataset, getattr(args, "per_device_train_batch_size", 0), dp, packing
            ),
            True,
        )
    ]
    strategy = getattr(args, "eval_strategy", getattr(args, "evaluation_strategy", "no"))
    if str(getattr(strategy, "value", strategy) or "no").lower() != "no":
        packed = packing if eval_packing is None else bool(eval_packing)
        batches.append(
            (
                _largest_batch(
                    eval_dataset, getattr(args, "per_device_eval_batch_size", 0), dp, packed
                ),
                False,
            )
        )
    for batch_size, train in batches:
        if not batch_size:
            continue
        try:
            m, pad = _plan_microbatches(
                want,
                layout.microbatches,
                layout.pp,
                schedule,
                batch_size,
                train,
                auto = getattr(layout, "schedule_auto", False),
            )
        except ValueError:
            continue  # the step raises its own error
        if pad and layout.microbatches:
            continue  # this schedule cannot run the UNSLOTH_PP_MICROBATCHES asked for; the split says so
        # A split batch carries the aux loss over the whole batch under GPipe (the default schedule runs it);
        # evaluation is forward only.
        whole = schedule == "gpipe" or (
            getattr(layout, "schedule_auto", False) and schedule == "1f1b"
        )
        if m > 1 and train and not whole:
            raise NotImplementedError(_split_router_aux_message(layout.env_vars, m))


def _dp_info(stage):
    mesh = getattr(stage, "dp_mesh", None)
    mesh = mesh if mesh is not None else getattr(stage, "ddp_mesh", None)
    return (mesh.size(), mesh.get_local_rank(), mesh) if mesh is not None else (1, 0, None)


def _batch(stage, inputs):
    if not isinstance(inputs, collections.abc.Mapping):  # dict, BatchEncoding
        raise RuntimeError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel) needs dict batches from the Trainer's collator, "
            f"got {type(inputs).__name__}.{_hint()}"
        )
    other = sorted(k for k in inputs if k not in _BATCH_KEYS and k not in _DROPPED_KEYS)
    from .pp_native import _MEDIA_INPUT_KEYS

    media = [k for k in other if k in _MEDIA_INPUT_KEYS and inputs[k] is not None]
    if media:
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel) trains multimodal checkpoints on text only, but this "
            f"Trainer batch carries {media}: it is not dropped, training stops. Use native FSDP2 "
            f"(UNSLOTH_FSDP2=1) for multimodal training.{_hint()}"
        )
    if other:
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel): the Trainer batch has {other}, which the pipeline "
            "does not carry between stages yet (packed / padding-free rows, multimodal inputs). Use text rows "
            f"with packing = False.{_hint()}"
        )
    input_ids, labels = inputs.get("input_ids"), inputs.get("labels")
    if input_ids is None or labels is None:
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel) needs input_ids and labels in every batch (the last "
            f"stage computes the causal LM loss).{_hint()}"
        )
    extra = {k: inputs[k] for k in ("attention_mask", "position_ids") if inputs.get(k) is not None}
    return input_ids, labels, extra


def _refuse_split_router_aux(stage, n_microbatches):
    """One GPU step forms the router load-balancing loss once over the batch (a product of means), not per
    micro-batch. A split training batch carries it over the whole batch (pp_native._CarriedRouterAux), which
    needs every forward before the first backward: GPipe, or the default schedule (which then runs GPipe).
    Refused under an explicit 1F1B / multi-stage schedule."""
    moe = getattr(getattr(stage, "ctx", None), "moe", None)
    if n_microbatches <= 1 or moe is None:
        return
    from .pp_native import _router_aux_coef, _router_aux_requested

    if not _router_aux_requested(stage.ctx, moe.config) or not _router_aux_coef(moe):
        return
    if _whole_batch_aux_schedule(stage, n_microbatches):
        return
    raise NotImplementedError(_split_router_aux_message(_env_vars(stage), n_microbatches))


def _whole_batch_aux_schedule(stage, n_microbatches):
    """The schedule a split batch with the router aux loss runs runs every forward before any backward."""
    if getattr(getattr(stage, "spec", None), "multi", False):
        return False
    step_schedule = getattr(stage, "step_schedule", None)
    if callable(step_schedule) and getattr(stage.ctx, "aux_batch_rows", None) is not None:
        return step_schedule(n_microbatches) == "gpipe"
    schedule = getattr(stage, "schedule", None)
    return schedule == "gpipe" or (getattr(stage, "schedule_auto", False) and schedule == "1f1b")


def _split_router_aux_message(env_vars, n_microbatches):
    return (
        f"Unsloth: {env_vars} (pipeline parallel) splits each Trainer batch into {n_microbatches} "
        "micro-batches. The router load-balancing loss is formed over the whole batch, as the Trainer does on one "
        "GPU (not as a sum of per micro-batch losses), which needs every micro-batch forward before the first "
        "backward: only the GPipe schedule does that, and this run asked for another one. Leave "
        "UNSLOTH_PP_SCHEDULE unset or set it to gpipe, set UNSLOTH_PP_MICROBATCHES=1, or turn the auxiliary loss "
        "off (router_aux_loss_coef = 0.0 in SFTConfig, or model.config.output_router_logits = False with a plain "
        "Trainer)."
        f"{_hint()}"
    )


def _first_stage_batch(stage, input_ids, labels, extra, count):
    """Every stage takes the first stage's batch and token count: a stochastic collator or dataset
    transform draws from each rank's own RNG, so stages could see different batches; tensor
    parallel peers in a stage sync first, since their reductions combine one example's
    activations."""
    import torch.distributed as dist

    if getattr(stage, "num_stages", 1) <= 1 or not dist.is_available() or not dist.is_initialized():
        return input_ids, labels, extra, count
    tp_mesh = getattr(stage, "tp_mesh", None)
    if tp_mesh is not None and tp_mesh.size() > 1:
        group = tp_mesh.get_group()
        src = dist.get_process_group_ranks(group)[0]
        input_ids, labels, extra, count = _broadcast_batch(
            stage, src, group, input_ids, labels, extra, count
        )
    return _broadcast_batch(
        stage, stage._global_ranks[0], stage.group, input_ids, labels, extra, count
    )


def _broadcast_batch(stage, src, group, input_ids, labels, extra, count):
    import torch.distributed as dist

    payload = [None]
    if dist.get_rank() == src:
        tensors = {"input_ids": input_ids, "labels": labels, **extra}
        payload = [({k: v.detach().cpu() for k, v in tensors.items()}, count)]
    dist.broadcast_object_list(payload, src = src, group = group, device = _comm_device(stage))
    tensors, count = payload[0]
    device = input_ids.device
    tensors = {k: v.to(device) for k, v in tensors.items()}
    input_ids, labels = tensors.pop("input_ids"), tensors.pop("labels")
    return input_ids, labels, tensors, count


def _global_items(trainer, stage, num_items_in_batch):
    """The token count of the whole accumulation window over every dp replica. The Trainer's own count is
    gathered over accelerator.gather, which counts the dp group only here (see _dp_gather)."""
    if num_items_in_batch is None:
        return None
    if torch.is_tensor(num_items_in_batch):
        value = int(num_items_in_batch.detach().reshape(-1)[0])
    else:
        value = int(num_items_in_batch)
    if (
        getattr(trainer.args, "average_tokens_across_devices", False)
        and trainer.args.world_size > 1
    ):
        return value  # already summed over the dp replicas
    dp, _, mesh = _dp_info(stage)
    if dp == 1:
        return value
    import torch.distributed as dist

    count = torch.tensor([value], dtype = torch.int64, device = _comm_device(stage))
    dist.all_reduce(count, group = mesh.get_group())
    return int(count[0])


def _cpu_chunked_ce_model(trainer, model):
    """Model whose zoo CPU chunked CE compute_loss would arm, or None; the pipeline schedule never calls compute_loss."""
    if not getattr(getattr(type(trainer), "compute_loss", None), "_unsloth_cpu", False):
        return None
    loss_type = getattr(trainer.args, "loss_type", None)
    if str(getattr(loss_type, "value", loss_type) or "nll") != "nll":
        return None
    accelerator = getattr(trainer, "accelerator", None)
    unwrapped = accelerator.unwrap_model(model) if accelerator is not None else model
    base = getattr(unwrapped, "base_model", None)
    for candidate in (unwrapped, base, getattr(base, "model", None)):
        # vars(): PEFT forwards attribute reads to the inner model.
        if candidate is not None and vars(candidate).get("_unsloth_cpu_chunked_ce", False):
            return candidate
    return None


def _compute_loss_adds_router_aux(compute_loss):
    """TRL >= 1.15 SFTTrainer.compute_loss adds router_aux_loss_coef x aux / accumulation steps itself; older
    ones (1.14 already has the attribute) leave it to the model forward. Read from the code, not the version."""
    for _ in range(8):
        names = getattr(getattr(compute_loss, "__code__", None), "co_names", ())
        if "router_aux_loss_coef" in names and "current_gradient_accumulation_steps" in names:
            return True
        compute_loss = getattr(compute_loss, "__wrapped__", None)
        if compute_loss is None:
            return False
    return False


def _trainer_router_aux(
    trainer,
    stage,
    num_items_in_batch,
    chunked,
    train = True,
):
    """(coef, divisor) when stock TRL's compute_loss adds the router aux loss in one process (CPU backend, zoo's
    chunked CE not armed), else None: the model forward adds it. Its coef, not the CausalLM's stale cached one."""
    if getattr(getattr(stage, "ctx", None), "moe", None) is None or (train and chunked is not None):
        return None
    compute_loss = getattr(type(trainer), "compute_loss", None)
    if not getattr(compute_loss, "_unsloth_cpu", False):
        return None
    if not _compute_loss_adds_router_aux(compute_loss):
        return None
    if not getattr(trainer, "aux_loss_enabled", False):
        return None
    coef = getattr(trainer, "router_aux_loss_coef", None)
    if coef is None:
        return None
    steps = 1
    if train and num_items_in_batch is not None:
        steps = int(
            getattr(trainer, "current_gradient_accumulation_steps", 0)
            or trainer.args.gradient_accumulation_steps
        )
    return float(coef), float(max(steps, 1))


def _set_trainer_aux(stage, value):
    ctx = getattr(stage, "ctx", None)
    if ctx is not None:
        ctx.trainer_aux = value


def _router_aux_request(trainer):
    """TRL >= 1.18 passes output_router_logits per forward, not via the config; the schedule skips compute_loss."""
    return True if getattr(trainer, "aux_loss_enabled", False) else None


def _set_router_aux_request(stage, value):
    ctx = getattr(stage, "ctx", None)
    if ctx is not None:
        ctx.output_router_logits = value


def _apply_trainer_aux_coef(trainer, stage):
    """The trainer's router_aux_loss_coef on the text config the stage reads for MoE VLM wrappers, for one step.

    TRL >= 1.18 no longer writes it into the config, so the wrappers' coef (config.text_config) would stay the
    checkpoint's. Returns what to hand to _restore_trainer_aux_coef, so the value never reaches a saved config."""
    config = getattr(getattr(getattr(stage, "ctx", None), "moe", None), "config", None)
    coef = getattr(trainer, "router_aux_loss_coef", None)
    if coef is None or config is None or not hasattr(config, "router_aux_loss_coef"):
        return None
    old = config.router_aux_loss_coef
    if old == coef:
        return None
    config.router_aux_loss_coef = coef
    return (config, old)


def _restore_trainer_aux_coef(applied):
    if applied is not None:
        config, old = applied
        config.router_aux_loss_coef = old


_FOLD = "_unsloth_pp_fold"


def _fold_requested():
    return os.environ.get("UNSLOTH_PP_FOLD_GA", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


def _router_aux_on(trainer, stage):
    moe = getattr(getattr(stage, "ctx", None), "moe", None)
    if moe is None:
        return False
    from .pp_native import _router_aux_coef

    on = _router_aux_request(trainer)
    if on is None:
        on = getattr(stage.ctx, "output_router_logits", None)
    if on is None:
        on = getattr(getattr(moe, "config", None), "output_router_logits", False)
    if not on:
        return False
    try:
        return bool(_router_aux_coef(moe))
    except AttributeError:
        return True  # unknown coefficient: keep one schedule step per batch


# HF rotary embeddings whose frequencies follow the batch's max(position_ids) + 1 (dynamic_rope_update):
# regrouping and re-padding rows would change them.
_LENGTH_DEPENDENT_ROPE = ("dynamic", "longrope")


def _rope_types(config, depth = 0):
    """rope_type / type of every rope_scaling / rope_parameters dict (transformers 5 may hold one per layer
    type), here and in the nested text configs."""
    if config is None or depth > 3:
        return set()
    get = config.get if isinstance(config, dict) else lambda key: getattr(config, key, None)
    found = set()
    for name in ("rope_scaling", "rope_parameters"):
        value = get(name)
        if not isinstance(value, dict):
            continue
        for entry in (value, *value.values()):
            kind = entry.get("rope_type", entry.get("type")) if isinstance(entry, dict) else None
            if isinstance(kind, str):
                found.add(kind.lower())
    for sub in ("text_config", "language_config", "llm_config"):
        found |= _rope_types(get(sub), depth + 1)
    return found


def _length_dependent_rope(stage):
    """The rotary frequencies depend on the batch's sequence length (dynamic NTK, LongRoPE). Read from the
    config, which every stage holds alike, so all pp ranks decide the same."""
    cached = stage.__dict__.get("_unsloth_length_rope")
    if cached is None:
        kinds = _rope_types(getattr(getattr(stage, "model", None), "config", None))
        cached = any(kind in _LENGTH_DEPENDENT_ROPE for kind in kinds)
        stage.__dict__["_unsloth_length_rope"] = cached
    return cached


def _foldable(trainer, stage, input_ids, extra, n):
    """This Trainer batch can join one schedule step over the whole accumulation window: the window token
    count is known (each micro-batch loss is its token sum / that count, however the rows are grouped), a
    single-stage schedule (multi-stage ones are validated for one micro-batch count), no router aux loss
    (summed per micro-batch), no sequence-length dependent RoPE, and only [rows, seq] side inputs."""
    from .pp_native import PipelineParallel

    if not _fold_requested() or n is None or not isinstance(stage, PipelineParallel):
        return False
    if int(getattr(trainer.args, "gradient_accumulation_steps", 1) or 1) <= 1:
        return False
    if getattr(getattr(stage, "spec", None), "multi", False) or _router_aux_on(trainer, stage):
        return False
    if _length_dependent_rope(stage):
        return False  # regrouped rows change max(position_ids), so the rotary frequencies
    from ..utils.token_budget import _exactness_blocker

    # The token budget's exactness guards (NEFTune, LOMO / layerwise optimizers, router aux, RoPE); the
    # loss always divides by the window's num_items_in_batch here (n above), so that check is skipped.
    if _exactness_blocker(trainer, step_loss = False) is not None:
        return False
    rows, seq = input_ids.shape
    for value in extra.values():
        if not torch.is_tensor(value) or value.dim() != 2 or value.shape[1] != seq:
            return False
        if value.shape[0] not in (1, rows):
            return False
    return True


def _row_lengths(labels, extra):
    """Per row, the positions up to its last attended or labelled token: later ones are right padding (masked,
    no label), which reaches no earlier token under causal attention. Without a mask every position counts."""
    seq = labels.shape[1]
    mask = extra.get("attention_mask")
    if mask is None:
        return [seq] * labels.shape[0]
    keep = (labels != -100) | (mask.expand_as(labels) != 0)
    index = torch.arange(1, seq + 1, device = labels.device).expand_as(labels)
    return (index * keep).amax(dim = 1).clamp_(min = 1).tolist()


def _fold_plan(lengths, rows_per_mb, num_stages, floor, cap):
    """Contiguous groups of `lengths` (descending), each one schedule step of whole micro-batches of
    rows_per_mb rows padded to its longest row. A step of m micro-batches over p stages costs about
    (m + p - 1) micro-batch times (the bubble), each proportional to its padded length; the plan minimises the
    window's sum (ties: fewer steps). floor / cap bound m per step (1F1B's stage count, GPipe's in-flight
    micro-batches). Returns [(first row, end row, n_microbatches)]."""
    rows = len(lengths)
    units = -(-rows // rows_per_mb)
    best = [(0.0, 0)] + [(float("inf"), 0)] * units
    cut = [0] * (units + 1)
    for end in range(1, units + 1):
        for start in range(end - 1, -1, -1):
            m = end - start
            if cap is not None and m > max(cap, floor):
                break
            cost = best[start][0] + (max(m, floor) + num_stages - 1) * lengths[start * rows_per_mb]
            key = (cost, best[start][1] + 1)
            if key < best[end]:
                best[end], cut[end] = key, start
    plan, end = [], units
    while end:
        start = cut[end]
        plan.append((start * rows_per_mb, min(end * rows_per_mb, rows), max(end - start, floor)))
        end = start
    return plan[::-1]


def _fit(t, length, fill):
    """[rows, seq] -> [rows, length]: cut, or right-pad with `fill` (a callable gets the pad width)."""
    have = t.shape[1]
    if have >= length:
        return t[:, :length]
    if callable(fill):
        tail = fill(t, length - have)
    else:
        tail = t.new_full((t.shape[0], length - have), fill)
    return torch.cat([t, tail], dim = 1)


def _next_positions(t, width):
    return t[:, -1:] + torch.arange(1, width + 1, device = t.device, dtype = t.dtype)


_FILL = {"attention_mask": 0, "position_ids": _next_positions}


def _window_rows(stage, window):
    """The window's rows by descending length (each cut to its own length), the rows per micro-batch the
    unfolded steps used, and the plan (shared by every dp replica, which must run the same schedule steps)."""
    import torch.distributed as dist

    rows = []
    for input_ids, labels, extra in window:
        count = input_ids.shape[0]
        extra = {k: v.expand(count, -1) if v.shape[0] != count else v for k, v in extra.items()}
        for i, length in enumerate(_row_lengths(labels, extra)):
            rows.append(
                (
                    length,
                    input_ids[i : i + 1, :length],
                    labels[i : i + 1, :length],
                    {k: v[i : i + 1, :length] for k, v in extra.items()},
                )
            )
    rows.sort(key = lambda r: -r[0])  # stable: equal lengths keep the window's order
    lengths = [r[0] for r in rows]
    first = window[0][0].shape[0]
    m0, pad0 = resolve_microbatches(stage, first)
    per_mb = (first + pad0) // m0
    p = stage.num_stages
    auto = getattr(stage, "schedule_auto", False)
    floor = p if (stage.schedule == "1f1b" and not auto) or _gpipe_needs_stage_count() else 1
    # An explicit GPipe holds every micro-batch's activations until its backward: at most as many as
    # the unfolded step (or 1F1B's first stage) held.
    cap = max(m0, p) if stage.schedule == "gpipe" else None
    dp, _, mesh = _dp_info(stage)
    if dp > 1:
        device = _comm_device(stage)
        count = torch.tensor([len(lengths)], dtype = torch.int64, device = device)
        dist.all_reduce(count, op = dist.ReduceOp.MAX, group = mesh.get_group())
        shared = torch.zeros(int(count[0]), dtype = torch.int64, device = device)
        shared[: len(lengths)] = torch.tensor(lengths, dtype = torch.int64)
        dist.all_reduce(shared, op = dist.ReduceOp.MAX, group = mesh.get_group())
        plan_lengths = [max(int(x), 1) for x in shared.tolist()]
    else:
        plan_lengths = lengths
    plan = _fold_plan(plan_lengths, per_mb, p, floor, cap)
    packed = None
    if _pack_requested() and _packs_documents(stage) and _dp_agree(stage, _plain_rows(rows), 0):
        # The documents packed back to back into equal-length rows when that is cheaper than the length
        # groups. Planned from the dp-shared lengths: every replica's i-th longest row is at most the shared
        # i-th, so its rows fit the same bins.
        best = _pack_plan(plan_lengths, per_mb, p, floor, cap)
        if best is not None and best[0] < _fold_cost(plan, plan_lengths, per_mb, p):
            packed = (best[1], best[2], best[3][: len(rows)], floor)
    return rows, per_mb, plan, packed


def _pack_requested():
    return os.environ.get("UNSLOTH_PP_PACK", "1").strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )


# RoPE kinds whose rotation is a function of each token's own position only (so restarting the positions at
# every document gives each document the angles it has alone).
_POSITION_ONLY_ROPE = ("default", "linear", "llama3", "yarn")
_HF_MASK_BUILDERS = ("create_causal_mask", "create_sliding_window_causal_mask")


def _rope_dicts(config, depth = 0):
    if config is None or depth > 3:
        return []
    get = config.get if isinstance(config, dict) else lambda key: getattr(config, key, None)
    found = []
    for name in ("rope_scaling", "rope_parameters"):
        value = get(name)
        if isinstance(value, dict):
            found.append(value)
            found.extend(v for v in value.values() if isinstance(v, dict))
    for sub in ("text_config", "language_config", "llm_config"):
        found.extend(_rope_dicts(get(sub), depth + 1))
    return found


def _code_names(fn):
    """Global names the function (through its decorators' __wrapped__ chain) reads."""
    names = set()
    for _ in range(8):
        fn = getattr(fn, "__func__", fn)
        if fn is None:
            break
        names |= set(getattr(getattr(fn, "__code__", None), "co_names", ()))
        fn = getattr(fn, "__wrapped__", None)
    return names


def _attention_keeps_documents(stage):
    """Rows of several documents (position_ids restarting at 0 per document, no attention_mask) attend only
    within each document on this stage: Unsloth's fast decoder (varlen attention from the reset positions)
    or a transformers decoder whose mask builder makes the packed block-causal mask. Anything else (carried
    state between layers, a model stage forward, stateful mixers, multimodal / M-RoPE positions, other RoPE
    kinds, older mask code) keeps the rows apart."""
    from .pp_native import _MEDIA_CONFIG_KEYS

    ctx, parts = getattr(stage, "ctx", None), getattr(stage, "parts", None)
    if ctx is None or parts is None:
        return False
    if ctx.stage_forward is not None or ctx.carry is not None or ctx.ple is not None:
        return False
    if getattr(parts, "ple", None) is not None or getattr(parts, "adapter", None) is not None:
        return False
    if getattr(parts, "towers", None):
        return False
    inner, decoder = parts.inner, parts.decoder
    config = getattr(inner, "config", None)
    if config is None or any(getattr(config, k, None) is not None for k in _MEDIA_CONFIG_KEYS):
        return False
    for rope in _rope_dicts(config):
        kind = str(rope.get("rope_type", rope.get("type", "default"))).lower()
        if kind not in _POSITION_ONLY_ROPE or "mrope_section" in rope:
            return False
    try:
        from ..utils.packing import _iter_stateful_mixers, packing_boundary_blocker
        if packing_boundary_blocker(inner) is not None or _iter_stateful_mixers(decoder):
            return False
    except Exception:
        return False
    from .llama import LlamaModel_fast_forward, running_forward, runs_fast_forward

    if runs_fast_forward(decoder, LlamaModel_fast_forward):
        # Every owned attention reads the packed lengths LlamaModel_fast_forward infers.
        for i in range(stage.start, stage.end):
            attn = getattr(parts.layers[i], "self_attn", None)
            if attn is None:
                return False
            if "get_packed_info_from_kwargs" not in _code_names(running_forward(attn)):
                return False
        return True
    implementation = getattr(config, "_attn_implementation", None)
    if implementation not in ("eager", "sdpa", "flex_attention"):
        return False
    return bool(_code_names(running_forward(decoder)) & set(_HF_MASK_BUILDERS))


def _packs_documents(stage):
    """_attention_keeps_documents on every pp rank (each sees only its own layers; all must run the same
    schedule steps). One collective over the pp group, the first time."""
    cached = stage.__dict__.get("_unsloth_pp_packs")
    if cached is None:
        try:
            cached = bool(_attention_keeps_documents(stage))
        except Exception:
            cached = False
        import torch.distributed as dist

        if stage.num_stages > 1 and dist.is_initialized():
            flag = torch.tensor([int(cached)], dtype = torch.int64, device = _comm_device(stage))
            dist.all_reduce(flag, op = dist.ReduceOp.MIN, group = stage.group)
            cached = bool(flag[0])
        stage.__dict__["_unsloth_pp_packs"] = cached
    return cached


def _plain_rows(rows):
    """Every row is one document: attended up to its length (right padding only) with positions 0, 1, ..."""
    for length, _, _, extra in rows:
        mask = extra.get("attention_mask")
        if mask is not None and not bool((mask != 0).all()):
            return False
        positions = extra.get("position_ids")
        if positions is not None:
            want = torch.arange(length, device = positions.device, dtype = positions.dtype)
            if positions.dim() != 2 or not torch.equal(positions[0], want):
                return False
    return True


_PACK_BUCKET = 32


def _bins_fit(lengths, m, capacity):
    """Best-fit decreasing of `lengths` (descending) into m bins of `capacity` tokens: the bin of each length,
    or None."""
    import bisect

    loads = [(0, b) for b in range(m)]  # (load, bin), ascending
    where = []
    for length in lengths:
        k = bisect.bisect_right(loads, (capacity - length, m)) - 1  # the fullest bin it still fits
        if k < 0:
            return None
        load, b = loads.pop(k)
        bisect.insort(loads, (load + length, b))
        where.append(b)
    return where


def _pack_plan(lengths, per_mb, num_stages, floor, cap):
    """One schedule step of m single-row micro-batches, each whole documents (descending `lengths`) packed
    into `width` tokens: (cost, m, width, bin of each length), or None. Cost is _fold_plan's,
    (max(m, floor) + p - 1) x width. A micro-batch holds at most what the unfolded step's largest one did
    (per_mb rows of the longest); GPipe keeps at most `cap` in flight. Counts whose lower bound cannot beat
    the best so far are skipped, and the smallest fitting width is bisected, so large windows plan quickly."""
    if not lengths:
        return None
    total, longest = sum(lengths), lengths[0]
    limit = max(per_mb * longest, longest)
    best = None
    for m in range(1, len(lengths) + 1):
        if cap is not None and m > max(cap, floor):
            break
        steps = max(m, floor) + num_stages - 1
        if best is not None and steps * longest >= best[0]:
            break  # the width is at least the longest document: no larger m can be cheaper either
        width = max(longest, -(-total // m))
        width = min(-(-width // _PACK_BUCKET) * _PACK_BUCKET, limit)
        if width * m < total or (best is not None and steps * width >= best[0]):
            continue
        where = _bins_fit(lengths, m, width)
        if where is None:
            top = limit
            if best is not None:  # widths stay whole buckets (or the limit): fewer step shapes
                top = min(top, (best[0] - 1) // steps // _PACK_BUCKET * _PACK_BUCKET)
            if top <= width or _bins_fit(lengths, m, top) is None:
                continue
            low, high = width, top  # fits at high, not at low
            while high - low > _PACK_BUCKET:
                mid = low + (high - low) // (2 * _PACK_BUCKET) * _PACK_BUCKET
                mid = mid if mid > low else low + _PACK_BUCKET
                if _bins_fit(lengths, m, mid) is None:
                    low = mid
                else:
                    high = mid
            width, where = high, _bins_fit(lengths, m, high)
        cost = steps * width
        if best is None or cost < best[0]:
            best = (cost, m, width, where)
    return best


def _fold_cost(plan, lengths, per_mb, num_stages):
    """_fold_plan's cost in tokens of work: its micro-batches hold per_mb rows each (packed ones one)."""
    return per_mb * sum((m + num_stages - 1) * lengths[start] for start, _, m in plan)


def _stack_packed(rows, m, width, where, floor):
    """m rows of `width` tokens: each the documents of its bin back to back, positions restarting at 0 per
    document and for the padding tail (a segment of its own), the first label of every document ignored
    (the previous document's last token must not predict it; one process never targets it either)."""
    device = rows[0][1].device if rows else None
    ids_dtype, label_dtype = rows[0][1].dtype, rows[0][2].dtype
    bins = [[] for _ in range(m)]
    for row, b in zip(rows, where):
        bins[b].append(row)
    ids, labels, positions = [], [], []
    for docs in bins:
        pieces_i, pieces_l, pieces_p = [], [], []
        for length, input_ids, row_labels, _ in docs:
            pieces_i.append(input_ids[0])
            first = row_labels[0].clone()
            first[0] = -100
            pieces_l.append(first)
            pieces_p.append(torch.arange(length, device = device))
        used = sum(d[0] for d in docs)
        tail = width - used
        if tail:
            pieces_i.append(torch.zeros(tail, dtype = ids_dtype, device = device))
            pieces_l.append(torch.full((tail,), -100, dtype = label_dtype, device = device))
            pieces_p.append(torch.arange(tail, device = device))
        ids.append(torch.cat(pieces_i))
        labels.append(torch.cat(pieces_l))
        positions.append(torch.cat(pieces_p))
    input_ids, labels = torch.stack(ids), torch.stack(labels)
    extra = {"position_ids": torch.stack(positions)}
    pad = max(m, floor) - m
    return _pad_rows(input_ids, labels, extra, pad) if pad > 0 else (input_ids, labels, extra)


def _stack_group(rows, start, end, m, per_mb):
    """Rows [start, end) padded to their longest, plus fully masked filler rows to m whole micro-batches."""
    group = rows[start:end]
    if group:
        length = max(r[0] for r in group)
        input_ids = torch.cat([_fit(r[1], length, 0) for r in group])
        labels = torch.cat([_fit(r[2], length, -100) for r in group])
        extra = {
            k: torch.cat([_fit(r[3][k], length, _FILL.get(k, 0)) for r in group])
            for k in group[0][3]
        }
    else:
        # This replica ran out of rows (a shorter last batch than another dp replica's): one filler row,
        # no labels and always attended (a left padded first column would leave it fully masked: NaN).
        input_ids, labels, extra = (
            rows[0][1][:, :1],
            torch.full_like(rows[0][2][:, :1], -100),
            {
                k: torch.ones_like(v[:, :1]) if k == "attention_mask" else v[:, :1]
                for k, v in rows[0][3].items()
            },
        )
    pad = m * per_mb - input_ids.shape[0]
    return _pad_rows(input_ids, labels, extra, pad) if pad > 0 else (input_ids, labels, extra)


def _pp_window_step(self, model, stage, window, n, scaler):
    """One accumulation window as few schedule steps (usually one) over all of its micro-batches."""
    rows, per_mb, plan, packed = _window_rows(stage, window)
    stats = stage.__dict__.setdefault(
        "_unsloth_fold_stats",
        {"windows": 0, "schedule_steps": 0, "batches": 0, "plans": [], "packed": 0},
    )
    stats["windows"] += 1
    stats["batches"] += len(window)
    if packed is not None:
        m, width, where, floor = packed
        stats["packed"] = stats.get("packed", 0) + 1
        stats["schedule_steps"] += 1
        stats["plans"] = (stats["plans"] + [[max(m, floor), width]])[-8:]
        if not stage.__dict__.get("_unsloth_pack_note"):
            stage.__dict__["_unsloth_pack_note"] = True
            _print_rank0(
                f"Unsloth: pipeline parallel: a gradient accumulation window of {len(window)} batch(es) runs as one "
                f"schedule step of {max(m, floor)} micro-batch(es), its documents packed into rows of {width} "
                "tokens with positions restarting per document (UNSLOTH_PP_PACK=0 groups rows by length)."
            )
        with _pp_step_context(self, model, stage, n) as run:
            input_ids, labels, extra = _stack_packed(rows, m, width, where, floor)
            return run(input_ids, labels, extra, max(m, floor), scaler, 1.0)
    stats["schedule_steps"] += len(plan)
    stats["plans"] = (stats["plans"] + [[m for _, _, m in plan]])[-8:]
    if not stage.__dict__.get("_unsloth_fold_note"):
        stage.__dict__["_unsloth_fold_note"] = True
        _print_rank0(
            f"Unsloth: pipeline parallel: each gradient accumulation window of {len(window)} batch(es) runs as "
            f"{len(plan)} schedule step(s) of {[m for _, _, m in plan]} micro-batch(es) of {per_mb} row(s), "
            "rows grouped by length (UNSLOTH_PP_FOLD_GA=0 runs one schedule step per batch)."
        )
    total = None
    with _pp_step_context(self, model, stage, n) as run:
        for start, end, m in plan:
            input_ids, labels, extra = _stack_group(rows, start, end, m, per_mb)
            loss = run(input_ids, labels, extra, m, scaler, 1.0)
            total = loss if total is None else total + loss
    return total


class _pp_step_context:
    """Router aux / CPU chunked CE switches around the schedule steps of one Trainer step."""

    def __init__(self, trainer, model, stage, n):
        self.trainer, self.model, self.stage, self.n = trainer, model, stage, n

    def __enter__(self):
        trainer, stage = self.trainer, self.stage
        self.chunked = None
        _set_router_aux_request(stage, _router_aux_request(trainer))
        self.applied = _apply_trainer_aux_coef(trainer, stage)
        try:
            self.chunked = _cpu_chunked_ce_model(trainer, self.model)
            if self.chunked is not None:
                self.chunked._unsloth_cpu_chunked_ce_active = True
            _set_trainer_aux(stage, _trainer_router_aux(trainer, stage, self.n, self.chunked))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self.run

    def run(self, input_ids, labels, extra, m, scaler, scale):
        return self.stage.step(
            input_ids,
            labels,
            num_items_in_batch = self.n,
            scaler = scaler,
            n_microbatches = m,
            loss_scale = scale,
            **extra,
        )

    def __exit__(self, *exc):
        _restore_trainer_aux_coef(self.applied)
        _set_trainer_aux(self.stage, None)
        _set_router_aux_request(self.stage, None)
        if self.chunked is not None:
            self.chunked._unsloth_cpu_chunked_ce_active = False
        return False


def _dp_agree(stage, flag, shape):
    """`flag` on every dp replica and the same `shape` (rows of the window's first batch, which sets the rows
    per micro-batch and GPipe's cap) on all of them: they must run the same schedule steps. Else False."""
    dp, _, mesh = _dp_info(stage)
    if dp == 1:
        return bool(flag)
    import torch.distributed as dist

    value = torch.tensor(
        [int(bool(flag)), shape, -shape], dtype = torch.int64, device = _comm_device(stage)
    )
    dist.all_reduce(value, op = dist.ReduceOp.MIN, group = mesh.get_group())
    return bool(value[0]) and int(value[1]) == -int(value[2])


def _add_fold_reset(trainer):
    """A new train() starts a new window: drop rows an interrupted one left buffered."""
    from transformers import TrainerCallback

    class _UnslothPPFoldReset(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            trainer.__dict__.pop(_FOLD, None)

    if not any(
        type(c).__name__ == "_UnslothPPFoldReset" for c in trainer.callback_handler.callbacks
    ):
        trainer.add_callback(_UnslothPPFoldReset())


def _pp_training_step(
    self,
    model,
    inputs,
    num_items_in_batch = None,
    *args,
    **kwargs,
):
    stage = self.__dict__["_unsloth_pipeline_parallel"]
    _note_trained(stage)
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()
    inputs = self._prepare_inputs(inputs)
    input_ids, labels, extra = _batch(stage, inputs)
    n = _global_items(self, stage, num_items_in_batch)
    input_ids, labels, extra, n = _first_stage_batch(stage, input_ids, labels, extra, n)
    del inputs
    scaler = getattr(self.accelerator, "scaler", None)
    scaler = scaler if scaler is not None and scaler.is_enabled() else None
    window = self.__dict__.get(_FOLD)
    # A window starts on every dp replica or on none: a buffering replica runs no collective until the
    # window's last batch. (The run's settings alone rule it out without a collective.)
    ga = int(getattr(self.args, "gradient_accumulation_steps", 1) or 1)
    start = window is None and ga > 1 and _fold_requested()
    if window is not None or (
        start and _dp_agree(stage, _foldable(self, stage, input_ids, extra, n), 0)
    ):
        # The Trainer runs no backward of its own here, so the window's batches wait (loss 0) and the
        # last one runs them all as micro-batches: the pipeline fills and drains once per window.
        window = self.__dict__.setdefault(_FOLD, [])
        window.append((input_ids, labels, extra, n))
        if not self.accelerator.sync_gradients:
            return torch.zeros((), device = self.args.device)
        self.__dict__.pop(_FOLD, None)
        fold = len(window) > 1 and len({entry[3] for entry in window}) == 1
        # Rows of different batches are stacked, so every batch must carry the same side inputs.
        fold = fold and len({tuple(sorted(entry[2])) for entry in window}) == 1
        fold = fold and all(_foldable(self, stage, e[0], e[2], e[3]) for e in window)
        if _dp_agree(stage, fold, window[0][0].shape[0]):
            loss = _pp_window_step(self, model, stage, [e[:3] for e in window], n, scaler)
        else:  # mixed batches: one schedule step per batch, as without folding
            loss = sum(_pp_batch_step(self, model, stage, *entry, scaler) for entry in window)
        return loss.detach().to(self.args.device)
    return (
        _pp_batch_step(self, model, stage, input_ids, labels, extra, n, scaler)
        .detach()
        .to(self.args.device)
    )


def _pp_batch_step(self, model, stage, input_ids, labels, extra, n, scaler):
    scale = 1.0
    if n is None:
        # No window token count: each step averages its own tokens, the window averages the steps.
        scale = 1.0 / max(
            int(
                getattr(self, "current_gradient_accumulation_steps", 0)
                or self.args.gradient_accumulation_steps
            ),
            1,
        )
    m, pad = resolve_microbatches(stage, input_ids.shape[0])
    with (
        _pp_step_context(self, model, stage, n) as run,
        _whole_batch_router_aux(stage, input_ids.shape[0]),
    ):
        _refuse_split_router_aux(stage, m)
        input_ids, labels, extra = _pad_rows(input_ids, labels, extra, pad)
        return run(input_ids, labels, extra, m, scaler, scale)


class _whole_batch_router_aux:
    """For one Trainer batch of `rows` real rows: a split batch's router aux loss is the one-GPU loss over the
    whole batch (pp_native._CarriedRouterAux), not one per micro-batch."""

    def __init__(self, stage, rows):
        self.ctx, self.rows = getattr(stage, "ctx", None), int(rows)

    def __enter__(self):
        if self.ctx is not None:
            self.ctx.aux_batch_rows = self.rows
        return self

    def __exit__(self, *exc):
        if self.ctx is not None:
            self.ctx.aux_batch_rows = None
        return False


def _eval_items(trainer, inputs, labels):
    """This trainer's one-process evaluation token count, local to the data parallel replica."""
    if getattr(getattr(type(trainer), "compute_loss", None), "_unsloth_cpu", False):
        import trl
        from packaging.version import Version

        # TRL 1.15 evaluates its fused head without num_items_in_batch, even with CPU chunked CE installed.
        if Version(trl.__version__) >= Version("1.15.0"):
            return int((labels[:, 1:] != -100).sum())
    getter = getattr(trainer, "_get_num_items_in_batch", None)
    if callable(getter):
        args = trainer.args
        flag = getattr(args, "average_tokens_across_devices", None)
        try:
            if flag:
                args.average_tokens_across_devices = False
            value = getter([inputs], args.device)
        except Exception:
            value = None
        finally:
            if flag:
                args.average_tokens_across_devices = flag
        if value is not None:
            return int(value.reshape(-1)[0]) if torch.is_tensor(value) else int(value)
    return int((labels[:, 1:] != -100).sum())


def _pp_prediction_step(
    self,
    model,
    inputs,
    prediction_loss_only,
    ignore_keys = None,
):
    stage = self.__dict__["_unsloth_pipeline_parallel"]
    if not prediction_loss_only:
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel): evaluation returns the loss only (the last stage "
            "does not send logits); remove compute_metrics / set prediction_loss_only = True, or evaluate on one "
            f"GPU.{_hint()}"
        )
    inputs = self._prepare_inputs(inputs)
    input_ids, labels, extra = _batch(stage, inputs)
    count = _eval_items(self, inputs, labels)
    input_ids, labels, extra, count = _first_stage_batch(stage, input_ids, labels, extra, count)
    m, pad = resolve_microbatches(stage, input_ids.shape[0], train = False)
    _set_router_aux_request(stage, _router_aux_request(self))
    applied_coef = _apply_trainer_aux_coef(self, stage)
    try:
        # Forward only: a split batch's router aux loss is formed over the whole batch on any schedule.
        rows = input_ids.shape[0]
        input_ids, labels, extra = _pad_rows(input_ids, labels, extra, pad)
        # TRL's prediction step passes no num_items_in_batch: the aux term is added once per batch.
        _set_trainer_aux(stage, _trainer_router_aux(self, stage, None, None, train = False))
        with _whole_batch_router_aux(stage, rows):
            loss = stage.eval_loss(
                input_ids, labels, num_items_in_batch = count, n_microbatches = m, **extra
            )
    finally:
        _restore_trainer_aux_coef(applied_coef)
        _set_trainer_aux(stage, None)
        _set_router_aux_request(stage, None)
    return (loss.detach().to(self.args.device), None, None)


def _pp_save_model(
    self,
    output_dir = None,
    _internal_call = False,
):
    stage = self.__dict__["_unsloth_pipeline_parallel"]
    output_dir = output_dir if output_dir is not None else self.args.output_dir
    model = stage.model
    if not hasattr(model, "peft_config"):
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel) saves LoRA adapters only.{_hint()}"
        )
    stage.save_adapter(output_dir)
    processing = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
    if processing is not None and self.args.should_save:
        processing.save_pretrained(output_dir)


def _dp_gather(accelerator, stage):
    """accelerator.gather over the dp group only: the pp / tp ranks of a replica hold the same batch, so the
    world gather would count it pp x tp times (token counts, eval losses, logged losses)."""
    from accelerate.utils import recursively_apply

    dp, _, mesh = _dp_info(stage)

    def one(tensor):
        if tensor.dim() == 0:
            tensor = tensor.clone()[None]
        if dp == 1:
            return tensor
        import torch.distributed as dist

        work = tensor.contiguous()
        out = [torch.empty_like(work) for _ in range(dp)]
        dist.all_gather(out, work, group = mesh.get_group())
        return torch.cat(out, dim = 0)

    def gather(tensor):
        return recursively_apply(one, tensor, error_on_other_type = True)

    return gather


def _unwrap_optimizer(opt):
    try:
        from accelerate.optimizer import AcceleratedOptimizer
    except Exception:
        AcceleratedOptimizer = ()
    seen = 0
    while isinstance(opt, AcceleratedOptimizer) and seen < 8:
        opt = opt.optimizer
        seen += 1
    return opt


def _pp_clip(accelerator, stage):
    from .tp_native import _DP_SYNC, _dp_average
    def clip_grad_norm_(
        parameters,
        max_norm,
        norm_type = 2,
    ):
        if float(norm_type) != 2.0:
            raise RuntimeError(
                "Unsloth: pipeline parallel gradient clipping supports the 2-norm only."
            )
        params = [p for p in parameters if p.requires_grad]
        sync = accelerator.__dict__.get(_DP_SYNC)
        if sync is not None:
            _dp_average(params, sync["mesh"])
            sync["synced"] = True
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            from torch.amp.grad_scaler import OptState
            for opt in getattr(accelerator, "_optimizers", []):
                # The torch optimizer AcceleratedOptimizer.step hands the scaler: the scaler keys its
                # unscaled state by optimizer, so unscaling the wrapper would divide the gradients again there.
                opt = _unwrap_optimizer(opt)
                state = scaler._per_optimizer_states[id(opt)]
                if state["stage"] is OptState.READY:
                    stage.unscale_(scaler, opt)
        return stage.clip_grad_norm_(max_norm, params)

    return clip_grad_norm_


def _pp_save_optimizer_and_scheduler(self, output_dir):
    """A checkpoint records this rank's stage only: drop a replica left by generate / for_inference first."""
    from .tp_native import _tp_save_optimizer_and_scheduler

    release_replica(self.model)
    return _tp_save_optimizer_and_scheduler(self, output_dir)


def _pp_load_from_checkpoint(self, *args, **kwargs):
    """Checkpoint holds this rank's stage only: drop a generate / for_inference replica first."""
    from .tp_native import _tp_load_from_checkpoint

    release_replica(self.model)
    return _tp_load_from_checkpoint(self, *args, **kwargs)


def pp_trainer_setup(trainer):
    """After Trainer.__init__ (models/_utils.py) for a model split under UNSLOTH_PP_SIZE."""
    from .tp_native import (
        _DP_SYNC,
        _TP_DATA,
        _dp_sync_before_step,
        _install_tp_accelerate_hooks,
        _tp_load_optimizer_and_scheduler,
        _tp_save_optimizer_and_scheduler,
        _tp_train_sampler,
    )

    stage = _stage_of(trainer.model)
    dp, dp_rank, _ = _dp_info(stage)
    accelerator = trainer.accelerator
    if not trainer.__dict__.get("_unsloth_tensor_parallel"):
        # Every stage must draw the same rows: seed the train sampler per dp replica, not from the global RNG.
        _tp_train_sampler(trainer, dp, dp_rank)
    _install_tp_accelerate_hooks()  # map-style data sharded over dp only, no DDP / device move of the model
    accelerator.__dict__[_TP_DATA] = (dp, dp_rank)
    accelerator.__dict__["_unsloth_pp_stage"] = stage
    accelerator.__dict__["gather"] = _dp_gather(accelerator, stage)
    accelerator.__dict__["clip_grad_norm_"] = _pp_clip(accelerator, stage)
    if stage.ddp_mesh is not None and accelerator.__dict__.get(_DP_SYNC) is None:
        accelerator.__dict__[_DP_SYNC] = {"mesh": stage.ddp_mesh, "synced": False}
    sync = accelerator.__dict__.get(_DP_SYNC)
    # The generated TRL trainers skip their post-init for_inference() (it would gather the whole model).
    trainer.__dict__["_unsloth_tensor_parallel"] = True
    trainer.__dict__["_unsloth_pipeline_parallel"] = stage
    for name, method in (
        ("training_step", _pp_training_step),
        ("prediction_step", _pp_prediction_step),
        ("save_model", _pp_save_model),
        ("_save_optimizer_and_scheduler", _pp_save_optimizer_and_scheduler),
        ("_load_optimizer_and_scheduler", _tp_load_optimizer_and_scheduler),
        ("_load_from_checkpoint", _pp_load_from_checkpoint),
    ):
        trainer.__dict__[name] = types.MethodType(method, trainer)
    trainer.__dict__["_has_mixed_mesh_grads"] = lambda model: False
    _add_fold_reset(trainer)
    from .fsdp2_native import _scaler_for_native
    from .tp_native import _SCALE_DIVISOR

    # Read by _utils._parallel_forced_scaler.
    accelerator.__dict__[_SCALE_DIVISOR] = dp
    try:
        scaled = _scaler_for_native(trainer)
    except Exception:
        scaled = False
    if scaled is not False and getattr(accelerator, "scaler", None) is not None:
        handler = getattr(accelerator, "scaler_handler", None)
        user = handler.to_kwargs() if handler is not None and hasattr(handler, "to_kwargs") else {}
        if "init_scale" not in user:
            accelerator.scaler._init_scale = (
                2.0**16 / dp
            )  # the loss scale grows with dp, never with pp
    if sync is not None and sync.get("_unsloth_pp_hooked") is None:
        create_optimizer = trainer.create_optimizer

        def create_optimizer_pp(*args, **kwargs):
            # DDP-style dp without clipping: average the gradients at the optimizer step instead.
            optimizer = create_optimizer(*args, **kwargs)
            if optimizer is None:
                # An override may set self.optimizer and return nothing (the Trainer reads self.optimizer).
                optimizer = trainer.optimizer
            from .fsdp2_native import _inner_optimizer

            inner = _inner_optimizer(optimizer)
            if not getattr(inner, "_unsloth_tp_dp_hooked", False):
                inner.register_step_pre_hook(functools.partial(_dp_sync_before_step, sync))
                inner._unsloth_tp_dp_hooked = True
            return optimizer

        trainer.__dict__["create_optimizer"] = create_optimizer_pp
        sync["_unsloth_pp_hooked"] = True
    _print_rank0(
        f"Unsloth: Trainer under {_env_vars(stage)}: {stage}; every step runs through the pipeline schedule."
    )
    return True
