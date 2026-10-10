# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import contextlib
import functools
from typing import Iterator, Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.device_mesh import DeviceMesh
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None

from .device_type import DEVICE_TYPE_TORCH

# A module global, not a ContextVar: on CUDA autograd runs backward (and so the reentrant
# checkpoint recompute) on its own device thread, where a ContextVar set here reads as unset.
_ACTIVE_MANAGER: Optional["ContextParallelManager"] = None

# shift_labels is built before sharding so each shard's last token keeps its next-token target.
_BUFFER_NAMES = ("input_ids", "attention_mask", "labels", "position_ids", "shift_labels")
_PAD_VALUES = {"labels": -100, "shift_labels": -100, "attention_mask": 0, "input_ids": 0}


# AcceleratorState is process-wide; a later non-CP trainer must not inherit our mesh.
_INSTALLED_MESH = []


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER


def _supports_context_parallel(model) -> bool:
    # Llama forward (Llama, Qwen2, Gemma): RoPE from global position_ids, SDPA via the patched F.
    from .models.llama import LlamaAttention_fast_forward

    # Every attention layer: one without ring attention would attend over its local shard only.
    forwards = [
        getattr(type(module).forward, "__func__", type(module).forward)
        for name, module in model.named_modules()
        if name.endswith("self_attn")
    ]
    return bool(forwards) and all(f is LlamaAttention_fast_forward for f in forwards)


def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
    # Ring attention only supports is_causal with no mask; right padding is already -100 in labels.
    if get_cp_manager() is not None and "attention_mask" in module_kwargs:
        module_kwargs["attention_mask"] = None
    return module_args, module_kwargs


class ContextParallelManager:
    def __init__(self, size: int):
        self.size = size
        world_size = dist.get_world_size()
        # DeviceMesh is SPMD (per-group meshes hang); accelerate reads "cp" to feed CP peers one batch.
        self.device_mesh = DeviceMesh(
            DEVICE_TYPE_TORCH,
            torch.arange(world_size).reshape(world_size // size, size),
            mesh_dim_names = ("dp_replicate", "cp"),
        )
        self.mesh = self.device_mesh["cp"]

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        # Marked on the module: a second CP trainer on the same model must not stack another hook.
        for name, module in model.named_modules():
            if name.endswith("self_attn") and not getattr(module, "_unsloth_cp_hooked", False):
                module.register_forward_pre_hook(
                    _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
                )
                module._unsloth_cp_hooked = True

    def _prepare_inputs(self, inputs: dict) -> None:
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or "inputs_embeds" in inputs:
            raise ValueError(
                "Unsloth: context parallelism needs input_ids batches (not inputs_embeds)."
            )
        bsz, seq_len = input_ids.shape
        # .get: a collator may carry the key with None, which would leave positions shard-local.
        if inputs.get("position_ids") is None:
            inputs["position_ids"] = (
                torch.arange(seq_len, device = input_ids.device).expand(bsz, -1).contiguous()
            )
        mask = inputs.get("attention_mask")
        # The hook drops the mask: only 2D right padding is safe (causal) and shards on dim 1.
        if isinstance(mask, torch.Tensor) and (
            mask.ndim != 2 or (mask[:, 1:] > mask[:, :-1]).any()
        ):
            raise ValueError(
                "Unsloth: context parallelism needs 2D right-padded attention masks without holes."
            )
        labels = inputs.get("labels")
        if inputs.get("shift_labels") is None and labels is not None:
            inputs["shift_labels"] = F.pad(labels, (0, 1), value = -100)[:, 1:].contiguous()
        # The load balancer splits the sequence into 2 * size chunks.
        pad = (-seq_len) % (2 * self.size)
        if pad:
            for name in _BUFFER_NAMES:
                tensor = inputs.get(name)
                if not isinstance(tensor, torch.Tensor) or tensor.ndim < 2:
                    continue
                if name == "position_ids":
                    extra = tensor[:, -1:] + torch.arange(1, pad + 1, device = tensor.device)
                    inputs[name] = torch.cat([tensor, extra.to(tensor.dtype)], dim = 1)
                else:
                    inputs[name] = F.pad(tensor, (0, pad), value = _PAD_VALUES.get(name, 0))

    @contextlib.contextmanager
    def apply(self, inputs: dict) -> Iterator[None]:
        self._prepare_inputs(inputs)
        buffers = [
            inputs[name]
            for name in _BUFFER_NAMES
            if isinstance(inputs.get(name), torch.Tensor) and inputs[name].ndim > 1
        ]
        global _ACTIVE_MANAGER
        previous, _ACTIVE_MANAGER = _ACTIVE_MANAGER, self
        # torch < 2.13 does not restore SDPA when the step raises (OOM, KeyboardInterrupt).
        sdpa = F.scaled_dot_product_attention
        try:
            with context_parallel(
                self.mesh,
                buffers = buffers,
                buffer_seq_dims = [1] * len(buffers),
                no_restore_buffers = set(buffers),
            ):
                yield
        finally:
            _ACTIVE_MANAGER = previous
            F.scaled_dot_product_attention = sdpa


def _refuse_iterable_datasets(*datasets) -> None:
    # accelerate's batch dispatcher (dispatch_batches, the default for iterable datasets) ignores cp.
    for dataset in datasets:
        for d in dataset.values() if isinstance(dataset, dict) else [dataset]:
            if isinstance(d, torch.utils.data.IterableDataset) or "IterableDataset" in type(d).__name__:
                raise NotImplementedError(
                    "Unsloth: context parallelism does not support iterable datasets."
                )


def patch_sft_trainer() -> None:
    import trl

    trainer_cls = trl.SFTTrainer
    if trainer_cls.__dict__.get("__unsloth_context_parallel__"):
        return

    original_init = trainer_cls.__init__
    original_prediction_step = trainer_cls.prediction_step
    original_training_step = trainer_cls.training_step
    original_train = trainer_cls.train
    original_evaluate = trainer_cls.evaluate

    def _install_accelerator_state(self):
        manager = getattr(self, "_context_parallel_manager", None)
        accelerator = getattr(self, "accelerator", None)
        state = getattr(accelerator, "state", None)
        if manager is None:
            # AcceleratorState is process-wide: a non-CP trainer must not inherit a CP mesh.
            if _INSTALLED_MESH and getattr(state, "device_mesh", None) is _INSTALLED_MESH[0]:
                state.device_mesh = None
            return
        if accelerator is None:
            return
        accelerator.state.device_mesh = manager.device_mesh
        _INSTALLED_MESH[:] = [manager.device_mesh]
        # Ring attention's backward collectives must not straddle DDP's no_sync accumulation.
        if hasattr(accelerator, "gradient_state"):
            accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
        # A PEFT model is not a PreTrainedModel, so the Trainer asks DDP for find_unused_parameters,
        # which fails with reentrant checkpointing ("mark a variable ready only once"). transformers
        # 5.x builds the DDP handler at init, 4.x rebuilds it from the argument when it wraps the model.
        if getattr(self.args, "ddp_find_unused_parameters", None) in (None, False):
            self.args.ddp_find_unused_parameters = False
            handler = getattr(accelerator, "ddp_handler", None)
            if handler is not None:
                handler.find_unused_parameters = False

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = None
        # Unsloth's attention ignores transformers' own CP: it would stay local to each shard.
        parallelism_config = getattr(getattr(self, "args", None), "parallelism_config", None)
        if (getattr(parallelism_config, "cp_size", 1) or 1) > 1:
            raise NotImplementedError(
                "Unsloth: use SFTConfig(context_parallel_size = N) instead of "
                "parallelism_config with cp_size > 1."
            )
        size = int(getattr(getattr(self, "args", None), "context_parallel_size", 1) or 1)
        if size <= 1:
            _install_accelerator_state(self)
            return
        if context_parallel is None:
            raise RuntimeError("Unsloth: context_parallel_size > 1 needs PyTorch >= 2.7.")
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Unsloth: context_parallel_size > 1 needs torch.distributed. "
                f"Launch with torchrun or accelerate launch using a multiple of {size} processes."
            )
        world_size = dist.get_world_size()
        if world_size % size != 0:
            raise RuntimeError(
                f"Unsloth: world size {world_size} is not a multiple of context_parallel_size {size}."
            )
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
            # Pure torch ring attention fails in backward without flash (2x T4, torch 2.10).
            raise NotImplementedError(
                "Unsloth: context parallelism needs flash attention (compute capability >= 8.0)."
            )
        # These recompute loss outside our forward from sharded labels, losing cross-shard targets.
        if (
            getattr(self.args, "label_smoothing_factor", 0)
            or getattr(self, "compute_loss_func", None)
            or getattr(self.args, "loss_type", None) not in (None, "nll")
        ):
            raise NotImplementedError(
                "Unsloth: context parallelism needs loss_type = 'nll' without "
                "label_smoothing_factor or compute_loss_func."
            )
        # Eval counts tokens per shard; only a cross-rank token count weights shards correctly (empty shard = NaN).
        if getattr(self.args, "average_tokens_across_devices", True) is False:
            raise NotImplementedError(
                "Unsloth: context parallelism needs average_tokens_across_devices = True."
            )
        import accelerate
        from packaging.version import Version

        # Older accelerate ignores the "cp" mesh dim, so CP peers would get different batches.
        if Version(accelerate.__version__) < Version("1.10.0"):
            raise NotImplementedError("Unsloth: context parallelism needs accelerate >= 1.10.0.")
        # The ring runs plain causal SDPA: the windowed mask Unsloth builds otherwise is dropped.
        sliding_window = getattr(getattr(self.model, "config", None), "sliding_window", None)
        if isinstance(sliding_window, int) and sliding_window > 0:
            raise NotImplementedError(
                "Unsloth: context parallelism does not support sliding window attention."
            )
        if not _supports_context_parallel(self.model):
            raise NotImplementedError(
                "Unsloth: context parallelism currently supports Llama-style attention only "
                "(Llama, Qwen2, Gemma)."
            )
        accelerator = getattr(self, "accelerator", None)
        # DeepSpeed's loader ignores cp; FSDP2 needs a parallelism_config with the mesh.
        distributed_type = getattr(getattr(accelerator, "distributed_type", None), "name", "NO")
        if distributed_type != "NO" and not distributed_type.startswith("MULTI_"):
            raise NotImplementedError(
                f"Unsloth: context parallelism supports DDP only, not {distributed_type}."
            )
        if getattr(accelerator, "dispatch_batches", None):
            raise NotImplementedError(
                "Unsloth: context parallelism does not support dispatch_batches."
            )
        _refuse_iterable_datasets(getattr(self, "train_dataset", None), getattr(self, "eval_dataset", None))
        manager = ContextParallelManager(size)
        self._context_parallel_manager = manager
        manager.attach_attention_hooks(self.model)
        _install_accelerator_state(self)
        print(f"Unsloth: Context parallelism enabled with size = {size}.")

    @functools.wraps(original_prediction_step)
    def patched_prediction_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        prediction_loss_only = args[0] if args else kwargs.get("prediction_loss_only", True)
        if manager is not None and not prediction_loss_only:
            raise NotImplementedError(
                "Unsloth: context parallelism supports loss-only evaluation; "
                "compute_metrics / predict() need context_parallel_size = 1."
            )
        if manager is None:
            return original_prediction_step(self, model, inputs, *args, **kwargs)
        inputs = self._prepare_inputs(inputs)
        manager._prepare_inputs(inputs)
        shift_labels = inputs.get("shift_labels")
        if not isinstance(shift_labels, torch.Tensor):
            with manager.apply(inputs):
                return original_prediction_step(self, model, inputs, *args, **kwargs)
        # Not the Trainer's count: transformers 4.x counts nothing in eval (each shard takes its own
        # mean, an empty shard is NaN) and 5.x scales by world size. Counted before sharding; CP
        # peers hold the same batch, so the world sum counts every token size times.
        total = shift_labels.ne(-100).sum()
        dist.all_reduce(total)
        inputs["num_items_in_batch"] = (total // manager.size).clamp_min(1)
        with manager.apply(inputs), torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs).detach()
        # Sum of the shards = this replica's tokens over all tokens; x replicas so the eval loop's
        # mean over ranks is the token mean over the whole batch.
        dist.all_reduce(loss, group = manager.mesh.get_group())
        return loss * (dist.get_world_size() // manager.size), None, None

    @functools.wraps(original_train)
    def patched_train(self, *args, **kwargs):
        _install_accelerator_state(self)
        return original_train(self, *args, **kwargs)

    @functools.wraps(original_evaluate)
    def patched_evaluate(self, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            _refuse_iterable_datasets(args[0] if args else kwargs.get("eval_dataset"))
        _install_accelerator_state(self)
        return original_evaluate(self, *args, **kwargs)

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is not None:
            # Counted before sharding on every CP peer; HF divides the same for parallelism_config.
            if args and args[0] is not None:
                args = (args[0] / manager.size, *args[1:])
            elif kwargs.get("num_items_in_batch") is not None:
                kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"] / manager.size
        # Forward and backward both inside, so gradient checkpointing recomputes with ring attention.
        with manager.apply(inputs) if manager else contextlib.nullcontext():
            return original_training_step(self, model, inputs, *args, **kwargs)

    trainer_cls.__init__ = patched_init
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.train = patched_train
    trainer_cls.evaluate = patched_evaluate
    trainer_cls.training_step = patched_training_step
    trainer_cls.__unsloth_context_parallel__ = True
