# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
from dataclasses import dataclass, field
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

_ACTIVE_MANAGER: contextvars.ContextVar[Optional["ContextParallelManager"]] = (
    contextvars.ContextVar("unsloth_active_cp_manager", default = None)
)

# Sharded along dim 1 by torch's load balancer. shift_labels is built before sharding so the
# next-token target of each shard's last token survives the split.
_BUFFER_NAMES = ("input_ids", "attention_mask", "labels", "position_ids", "shift_labels")
_PAD_VALUES = {"labels": -100, "shift_labels": -100, "attention_mask": 0, "input_ids": 0}


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER.get()


def _supports_context_parallel(model) -> bool:
    # Only the Llama attention forward (Llama, Qwen2, Gemma) takes global position ids for RoPE and
    # reaches SDPA through F.scaled_dot_product_attention, which is what context_parallel patches.
    from .models.llama import LlamaAttention_fast_forward
    for name, module in model.named_modules():
        if name.endswith("self_attn"):
            return getattr(type(module).forward, "__func__", type(module).forward) is (
                LlamaAttention_fast_forward
            )
    return False


def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
    # Ring attention only supports is_causal with no mask; right padding is already -100 in labels.
    if get_cp_manager() is not None and "attention_mask" in module_kwargs:
        module_kwargs["attention_mask"] = None
    return module_args, module_kwargs


class ContextParallelManager:
    def __init__(self, size: int):
        self.size = size
        world_size = dist.get_world_size()
        # DeviceMesh is SPMD: every rank must build the identical global mesh (a per-group mesh
        # hangs once world_size > size), then take its own "cp" row. accelerate reads the "cp"
        # dim to give every CP rank the same batch.
        self.device_mesh = DeviceMesh(
            DEVICE_TYPE_TORCH,
            torch.arange(world_size).reshape(world_size // size, size),
            mesh_dim_names = ("dp_replicate", "cp"),
        )
        self.mesh = self.device_mesh["cp"]
        self._hooked = False

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        if self._hooked:
            return
        for name, module in model.named_modules():
            if name.endswith("self_attn"):
                module.register_forward_pre_hook(
                    _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
                )
        self._hooked = True

    def _prepare_inputs(self, inputs: dict) -> None:
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return
        bsz, seq_len = input_ids.shape
        if "position_ids" not in inputs:
            inputs["position_ids"] = (
                torch.arange(seq_len, device = input_ids.device).expand(bsz, -1).contiguous()
            )
        labels = inputs.get("labels")
        if "shift_labels" not in inputs and labels is not None:
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
        token = _ACTIVE_MANAGER.set(self)
        try:
            with context_parallel(
                self.mesh,
                buffers = buffers,
                buffer_seq_dims = [1] * len(buffers),
                no_restore_buffers = set(buffers),
            ):
                yield
        finally:
            _ACTIVE_MANAGER.reset(token)


def patch_sft_config():
    import trl

    base_cls = trl.SFTConfig
    if hasattr(base_cls, "context_parallel_size"):
        return

    @dataclass
    class PatchedSFTConfig(base_cls):  # type: ignore[misc, valid-type]
        context_parallel_size: int = field(
            default = 1,
            metadata = {
                "help": "Ranks per context parallel group (SDPA ring attention). 1 disables it."
            },
        )

    PatchedSFTConfig.__name__ = base_cls.__name__
    PatchedSFTConfig.__qualname__ = base_cls.__qualname__
    PatchedSFTConfig.__module__ = base_cls.__module__
    module = sys.modules.get(base_cls.__module__)
    if module is not None:
        setattr(module, base_cls.__name__, PatchedSFTConfig)
    trl.SFTConfig = PatchedSFTConfig
    if hasattr(trl, "trainer") and hasattr(trl.trainer, "sft_trainer"):
        trl.trainer.sft_trainer.SFTConfig = PatchedSFTConfig


def patch_sft_trainer() -> None:
    import trl

    trainer_cls = trl.SFTTrainer
    if hasattr(trainer_cls, "__unsloth_context_parallel__"):
        return

    original_init = trainer_cls.__init__
    original_prediction_step = trainer_cls.prediction_step
    original_training_step = trainer_cls.training_step

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = None
        size = int(getattr(getattr(self, "args", None), "context_parallel_size", 1) or 1)
        if size <= 1:
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
            # Measured on 2x T4 (torch 2.10): torch's own ring attention fails in backward
            # (mixed Tensor / DTensor add) without a flash kernel, before Unsloth code runs.
            raise NotImplementedError(
                "Unsloth: context parallelism needs flash attention (compute capability >= 8.0)."
            )
        # These recompute loss outside our forward (TRL chunked_nll calls the backbone directly)
        # from the sharded, load-balanced labels, losing every cross-shard target.
        if (
            getattr(self.args, "label_smoothing_factor", 0)
            or getattr(self, "compute_loss_func", None)
            or getattr(self.args, "loss_type", None) not in (None, "nll")
        ):
            raise NotImplementedError(
                "Unsloth: context parallelism needs loss_type = 'nll' without "
                "label_smoothing_factor or compute_loss_func."
            )
        if not _supports_context_parallel(self.model):
            raise NotImplementedError(
                "Unsloth: context parallelism currently supports Llama-style attention only "
                "(Llama, Qwen2, Gemma)."
            )
        manager = ContextParallelManager(size)
        self._context_parallel_manager = manager
        manager.attach_attention_hooks(self.model)
        accelerator = getattr(self, "accelerator", None)
        if accelerator is not None:
            accelerator.state.device_mesh = manager.device_mesh
            # Ring attention's backward collectives must not straddle DDP's no_sync accumulation.
            if hasattr(accelerator, "gradient_state"):
                accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True
        print(f"Unsloth: Context parallelism enabled with size = {size}.")

    @functools.wraps(original_prediction_step)
    def patched_prediction_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        prediction_loss_only = args[0] if args else kwargs.get("prediction_loss_only", True)
        if manager is not None and not prediction_loss_only:
            # Logits come back as per-rank sequence shards, duplicated across the CP group.
            raise NotImplementedError(
                "Unsloth: context parallelism supports loss-only evaluation; "
                "compute_metrics / predict() need context_parallel_size = 1."
            )
        with manager.apply(inputs) if manager else contextlib.nullcontext():
            return original_prediction_step(self, model, inputs, *args, **kwargs)

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is not None:
            # The trainer counted tokens before sharding, so every CP rank holds its group's whole
            # count (x size once gathered); HF divides the same way for its parallelism_config.
            # Eval counts after sharding, so prediction_step needs no division.
            if args and args[0] is not None:
                args = (args[0] / manager.size, *args[1:])
            elif kwargs.get("num_items_in_batch") is not None:
                kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"] / manager.size
        # Forward and backward both inside, so gradient checkpointing recomputes with ring attention.
        with manager.apply(inputs) if manager else contextlib.nullcontext():
            return original_training_step(self, model, inputs, *args, **kwargs)

    trainer_cls.__init__ = patched_init
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.training_step = patched_training_step
    trainer_cls.__unsloth_context_parallel__ = True
