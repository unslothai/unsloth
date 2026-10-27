# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Pipeline parallelism from the environment (UNSLOTH_PP_SIZE, models/tp_env.py) for unmodified scripts.

    UNSLOTH_PP_SIZE=2 torchrun --nproc-per-node 2 train.py

- Trainer.__init__ splits the model (apply_pipeline_parallel over mesh["pp"]); the other stages' layers,
  embedding and lm_head go to this rank's CPU memory instead of being dropped (PPConfig.keep_replica).
- Trainer / SFTTrainer: every training step runs through the pipeline schedule (PipelineParallel.step runs the
  backward, the Trainer does not). The Trainer batch is split into UNSLOTH_PP_MICROBATCHES micro-batches
  (default 2 x pp, lowered to a divisor of the batch); every micro-batch loss is its summed token loss over the
  token count of the whole accumulation window over every dp replica, as with gradient accumulation on one
  GPU. The pp (and tp) ranks of one dp replica read the same batches; the optimizer steps each stage's own
  parameters; clipping, the fp16 GradScaler skip and the logged loss are global. Evaluation runs the pipeline
  forward. Trainer checkpoints: the 1-GPU adapter (rank 0) plus per-rank optimizer state.
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
_ALLOWED_TRAINERS = {"Trainer", "Seq2SeqTrainer", "SFTTrainer", "UnslothSFTTrainer", "_UnslothSFTTrainer"}
# Batch keys the pipeline carries (or that only shaped `labels`, already applied by the collator).
_BATCH_KEYS = ("input_ids", "labels", "attention_mask", "position_ids")
_DROPPED_KEYS = ("completion_mask", "assistant_masks", "return_loss", "num_items_in_batch")


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


# ---------------------------------------------------------------------------------------------- the split


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
    keep = os.environ.get("UNSLOTH_PP_KEEP_REPLICA", "1").strip().lower() not in ("0", "false", "no", "off")
    default = layout.microbatches or 2 * layout.pp
    try:
        stage = apply_pipeline_parallel(
            model,
            layout.pp_mesh,
            n_microbatches = default,
            schedule = layout.schedule,
            config = PPConfig(keep_replica = keep),
        )
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        from .tp_env import _reraise_with_hint
        _reraise_with_hint(exc)
    stage.env_layout = layout
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


def resolve_microbatches(stage, batch_size, train = True):
    """(n_microbatches, padded rows) for a batch of `batch_size` rows. UNSLOTH_PP_MICROBATCHES must divide it;
    the default (2 x pp) is lowered to the largest divisor of the batch. A schedule that needs at least pp
    micro-batches (1F1B; GPipe on torch 2.10) gets fully masked filler rows (labels -100: no loss, no gradient)
    when the batch has fewer rows than that."""
    layout = getattr(stage, "env_layout", None)
    explicit = layout.microbatches if layout is not None else None
    want = int(explicit or stage.n_microbatches)
    floor = stage.num_stages if (stage.schedule == "1f1b" and train) or _gpipe_needs_stage_count() else 1
    if batch_size % want == 0:
        return want, 0
    if explicit is not None and train:
        raise ValueError(
            f"Unsloth: UNSLOTH_PP_MICROBATCHES={explicit} must divide the Trainer batch size ({batch_size} rows "
            "per device): set per_device_train_batch_size to a multiple of it, or unset UNSLOTH_PP_MICROBATCHES."
        )
    m = max(d for d in range(1, min(want, batch_size) + 1) if batch_size % d == 0)
    pad = 0
    if m < floor:
        m = floor
        pad = -batch_size % floor
    if train and not stage.__dict__.get("_unsloth_mb_note"):
        stage.__dict__["_unsloth_mb_note"] = True
        rows = batch_size + pad
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


# ------------------------------------------------------------------------- the other stages, kept on the CPU


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
        dim = "tensor parallel (UNSLOTH_TP_SIZE)" if stage.tp_mesh is not None else "native FSDP2 data parallel"
        return (
            f"Unsloth: `{what}` is not supported yet on a pipeline parallel x {dim} model (each stage's weights "
            "are sharded over its other ranks). Save the adapter (model.save_pretrained, on every rank) and run "
            f"`{what}` on one GPU, or launch with UNSLOTH_PP_SIZE alone (DDP-style data parallel).{hint}"
        )
    return None


def gather_replica(model, what = "generate", persistent = True):
    """Collective over the world: put the whole model back on every rank (the kept CPU copies moved to this
    GPU, the trained weights of every stage broadcast from it). Refused on every rank alike, with the memory
    numbers, when some rank lacks the room. Returns the stage."""
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
    owned = [name for name, p in stage.model.named_parameters() if p.requires_grad]
    for container, key, module, _ in stage._kept:
        _move(module, device, set())
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


# ----------------------------------------------------------------------------------------------------- saves


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


def pp_save_adapter(model, args, kwargs):
    """PeftModel.save_pretrained on a pipeline parallel model: every stage's adapter weights gathered, the
    standard 1-GPU adapter written by rank 0 (collective)."""
    import inspect

    from peft import PeftModel

    stage = _stage_of(model)
    try:
        bound = inspect.signature(PeftModel.save_pretrained).bind(model, *args, **kwargs)
    except TypeError:
        bound = None
    if bound is None:
        raise TypeError(f"Unsloth: save_pretrained{args} could not be read as PeftModel.save_pretrained.")
    values = dict(bound.arguments)
    values.pop("self", None)
    extra = values.pop("kwargs", {}) or {}
    directory = values.pop("save_directory")
    for key in ("push_to_hub", "token", "state_dict", "save_function", "max_shard_size", "tokenizer"):
        extra.pop(key, None)
    if values.get("safe_serialization", True) is None:
        values["safe_serialization"] = True
    values.pop("is_main_process", None)
    stage.save_adapter(directory, **values, **extra)
    return None


# --------------------------------------------------------------------------------------------- the Trainer


def _trainer_classes(cls):
    return [c for c in cls.__mro__ if c is not object]


def pp_trainer_refusals(trainer_cls, args, layout):
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
    problems = []
    if args is None:
        return
    if getattr(args, "deepspeed", None):
        problems.append("DeepSpeed")
    if getattr(args, "fsdp", None) or getattr(args, "parallelism_config", None) is not None:
        problems.append("accelerate FSDP / ParallelismConfig (UNSLOTH_FSDP2=1 shards the dp dimension)")
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


def _dp_info(stage):
    mesh = stage.dp_mesh if stage.dp_mesh is not None else stage.ddp_mesh
    return (mesh.size(), mesh.get_local_rank(), mesh) if mesh is not None else (1, 0, None)


def _batch(stage, inputs):
    if not isinstance(inputs, collections.abc.Mapping):  # dict, BatchEncoding
        raise RuntimeError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel) needs dict batches from the Trainer's collator, "
            f"got {type(inputs).__name__}.{_hint()}"
        )
    other = sorted(k for k in inputs if k not in _BATCH_KEYS and k not in _DROPPED_KEYS)
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


def _global_items(trainer, stage, num_items_in_batch):
    """The token count of the whole accumulation window over every dp replica. The Trainer's own count is
    gathered over accelerator.gather, which counts the dp group only here (see _dp_gather)."""
    if num_items_in_batch is None:
        return None
    if torch.is_tensor(num_items_in_batch):
        value = int(num_items_in_batch.detach().reshape(-1)[0])
    else:
        value = int(num_items_in_batch)
    if getattr(trainer.args, "average_tokens_across_devices", False) and trainer.args.world_size > 1:
        return value  # already summed over the dp replicas
    dp, _, mesh = _dp_info(stage)
    if dp == 1:
        return value
    import torch.distributed as dist

    count = torch.tensor([value], dtype = torch.int64, device = _comm_device(stage))
    dist.all_reduce(count, group = mesh.get_group())
    return int(count[0])


def _pp_training_step(self, model, inputs, num_items_in_batch = None, *args, **kwargs):
    stage = self.__dict__["_unsloth_pipeline_parallel"]
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()
    inputs = self._prepare_inputs(inputs)
    input_ids, labels, extra = _batch(stage, inputs)
    n = _global_items(self, stage, num_items_in_batch)
    scale = 1.0
    if n is None:
        # No window token count: each step averages its own tokens, the window averages the steps.
        scale = 1.0 / max(int(getattr(self, "current_gradient_accumulation_steps", 0) or self.args.gradient_accumulation_steps), 1)
    scaler = getattr(self.accelerator, "scaler", None)
    scaler = scaler if scaler is not None and scaler.is_enabled() else None
    m, pad = resolve_microbatches(stage, input_ids.shape[0])
    input_ids, labels, extra = _pad_rows(input_ids, labels, extra, pad)
    loss = stage.step(
        input_ids,
        labels,
        num_items_in_batch = n,
        scaler = scaler,
        n_microbatches = m,
        loss_scale = scale,
        **extra,
    )
    del inputs
    return loss.detach().to(self.args.device)


def _pp_prediction_step(self, model, inputs, prediction_loss_only, ignore_keys = None):
    stage = self.__dict__["_unsloth_pipeline_parallel"]
    if not prediction_loss_only:
        raise NotImplementedError(
            f"Unsloth: {_env_vars(stage)} (pipeline parallel): evaluation returns the loss only (the last stage "
            "does not send logits); remove compute_metrics / set prediction_loss_only = True, or evaluate on one "
            f"GPU.{_hint()}"
        )
    inputs = self._prepare_inputs(inputs)
    input_ids, labels, extra = _batch(stage, inputs)
    m, pad = resolve_microbatches(stage, input_ids.shape[0], train = False)
    count = int((labels[:, 1:] != -100).sum())
    input_ids, labels, extra = _pad_rows(input_ids, labels, extra, pad)
    loss = stage.eval_loss(input_ids, labels, num_items_in_batch = count, n_microbatches = m, **extra)
    return (loss.detach().to(self.args.device), None, None)


def _pp_save_model(self, output_dir = None, _internal_call = False):
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


def _pp_clip(accelerator, stage):
    from .tp_native import _DP_SYNC, _dp_average

    def clip_grad_norm_(parameters, max_norm, norm_type = 2):
        if float(norm_type) != 2.0:
            raise RuntimeError("Unsloth: pipeline parallel gradient clipping supports the 2-norm only.")
        params = [p for p in parameters if p.requires_grad]
        sync = accelerator.__dict__.get(_DP_SYNC)
        if sync is not None:
            _dp_average(params, sync["mesh"])
            sync["synced"] = True
        scaler = getattr(accelerator, "scaler", None)
        if scaler is not None and scaler.is_enabled():
            from torch.amp.grad_scaler import OptState

            for opt in getattr(accelerator, "_optimizers", []):
                while hasattr(opt, "optimizer") and not isinstance(opt, torch.optim.Optimizer):
                    opt = opt.optimizer
                state = scaler._per_optimizer_states[id(opt)]
                if state["stage"] is OptState.READY:
                    stage.unscale_(scaler, opt)
        return stage.clip_grad_norm_(max_norm, params)

    return clip_grad_norm_


def pp_trainer_setup(trainer):
    """After Trainer.__init__ (models/_utils.py) for a model split under UNSLOTH_PP_SIZE."""
    from .tp_native import (
        _DP_SYNC,
        _TP_DATA,
        _dp_sync_before_step,
        _install_tp_accelerate_hooks,
        _tp_load_from_checkpoint,
        _tp_load_optimizer_and_scheduler,
        _tp_save_optimizer_and_scheduler,
    )

    stage = _stage_of(trainer.model)
    dp, dp_rank, _ = _dp_info(stage)
    accelerator = trainer.accelerator
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
        ("_save_optimizer_and_scheduler", _tp_save_optimizer_and_scheduler),
        ("_load_optimizer_and_scheduler", _tp_load_optimizer_and_scheduler),
        ("_load_from_checkpoint", _tp_load_from_checkpoint),
    ):
        trainer.__dict__[name] = types.MethodType(method, trainer)
    trainer.__dict__["_has_mixed_mesh_grads"] = lambda model: False
    from .fsdp2_native import _scaler_for_native

    try:
        scaled = _scaler_for_native(trainer)
    except Exception:
        scaled = False
    if scaled is not False and getattr(accelerator, "scaler", None) is not None:
        handler = getattr(accelerator, "scaler_handler", None)
        user = handler.to_kwargs() if handler is not None and hasattr(handler, "to_kwargs") else {}
        if "init_scale" not in user:
            accelerator.scaler._init_scale = 2.0**16 / dp  # the loss scale grows with dp, never with pp
    if sync is not None and sync.get("_unsloth_pp_hooked") is None:
        create_optimizer = trainer.create_optimizer

        def create_optimizer_pp(*args, **kwargs):
            # DDP-style dp without clipping: average the gradients at the optimizer step instead.
            optimizer = create_optimizer(*args, **kwargs)
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
