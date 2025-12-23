from __future__ import annotations

import contextlib
import contextvars
import functools
import sys
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.distributed as dist
import trl

try:
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor import DeviceMesh
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None

from .device_type import DEVICE_TYPE_TORCH
from .utils.packing import mask_packed_sequence_boundaries

_ACTIVE_MANAGER: contextvars.ContextVar[Optional["ContextParallelManager"]] = (
    contextvars.ContextVar("unsloth_active_cp_manager", default = None)
)

_BUFFER_NAMES = (
    "input_ids",
    "attention_mask",
    "labels",
    "position_ids",
    "shift_labels",
)


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER.get()


@dataclass
class ContextParallelSettings:
    size: int = field(
        default = 1,
        metadata = {
            "help": (
                "Number of ranks that should participate in context parallelism. "
                "Set to >1 only when running under torch.distributed / accelerate."
            )
        },
    )

    @classmethod
    def from_args(cls, args: Optional[object]) -> "ContextParallelSettings":
        if args is None:
            return cls()
        size = int(getattr(args, "context_parallel_size", 1))
        return cls(size = size)


def _attach_context_parallel_attention_hooks(model: torch.nn.Module) -> list:
    """
    Attach forward_pre_hooks to self_attn modules to ensure correct attention behavior
    during context parallelism with load balancing.

    Args:
        model: The model to attach hooks to

    Returns:
        List of hook handles that can be used to remove the hooks later
    """
    handles = []

    def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
        # Remove attention_mask and set is_causal=True
        # This ensures ring attention uses causal masking correctly
        if "attention_mask" in module_kwargs:
            module_kwargs["attention_mask"] = None
        if "is_causal" in module_kwargs or hasattr(_module, "is_causal"):
            module_kwargs["is_causal"] = True
        return module_args, module_kwargs

    # Find all self_attn modules - they may be nested in PEFT wrappers
    attn_modules = []
    for name, module in model.named_modules():
        # Attach to modules ending with self_attn (transformers convention)
        if name.endswith("self_attn"):
            attn_modules.append((name, module))

    for _, module in attn_modules:
        handle = module.register_forward_pre_hook(
            _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
        )
        handles.append(handle)

    return handles


class ContextParallelManager:
    """Toggles PyTorch context parallelism."""

    def __init__(self, settings: ContextParallelSettings):
        self.settings = settings
        self._mesh: Optional[DeviceMesh] = None
        self._device_mesh: Optional[DeviceMesh] = None
        self._cp_group: Optional[dist.ProcessGroup] = None
        self._cp_rank_index: int = 0
        self._dp_world_size: int = 1
        self._world_size: int = dist.get_world_size()
        self._report_loss: Optional[torch.Tensor] = None
        self._attention_hook_handles: list = []
        self._mesh = self._build_mesh()
        self._device_mesh = self._build_device_mesh()

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        """
        Attach hooks to self_attn modules to ensure correct attention behavior during
        context parallelism with load balancing.
        """
        if self._attention_hook_handles:
            return
        self._attention_hook_handles = _attach_context_parallel_attention_hooks(model)

    def _build_mesh(self) -> DeviceMesh:
        rank = torch.distributed.get_rank()
        group_index = rank // self.settings.size
        start = group_index * self.settings.size
        cp_ranks = torch.arange(start, start + self.settings.size, dtype = torch.int64)
        mesh = DeviceMesh(DEVICE_TYPE_TORCH, cp_ranks)
        self._cp_group = mesh.get_group()
        self._cp_rank_index = int(rank - start)
        return mesh

    def _build_device_mesh(self) -> DeviceMesh:
        self._dp_world_size = self._world_size // self.settings.size
        mesh = torch.arange(self._world_size, dtype = torch.int64).reshape(
            self._dp_world_size, self.settings.size
        )
        return DeviceMesh(
            DEVICE_TYPE_TORCH, mesh, mesh_dim_names = ("dp_replicate", "cp")
        )

    @property
    def device_mesh(self) -> Optional[DeviceMesh]:
        return self._device_mesh

    @property
    def data_parallel_world_size(self) -> int:
        return self._dp_world_size

    @property
    def cp_rank_index(self) -> int:
        return self._cp_rank_index

    def data_parallel_rank(self) -> int:
        return dist.get_rank() // self.settings.size

    def _collect_buffers(
        self, inputs: dict[str, torch.Tensor]
    ) -> Tuple[list[torch.Tensor], list[int], set[torch.Tensor]]:
        buffers: list[torch.Tensor] = []
        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if tensor is None or not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim <= 1:
                continue
            buffers.append(tensor)
        return buffers, [1] * len(buffers), set(buffers)

    def _ensure_position_ids(self, inputs: dict[str, torch.Tensor]) -> None:
        if "position_ids" in inputs:
            return
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, dtype = torch.long, device = input_ids.device)
        inputs["position_ids"] = positions.unsqueeze(0).expand(input_ids.size(0), -1)

    def _ensure_shift_labels(self, inputs: dict[str, torch.Tensor]) -> None:
        """Pre-shift labels globally before sharding for correct next-token prediction."""
        if "shift_labels" in inputs:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        # Pad with -100, then take [1:] to get shifted labels
        shift_labels = F.pad(labels, (0, 1), value = -100)[:, 1:].contiguous()
        packed_seq_lengths = inputs.get("packed_seq_lengths")
        if packed_seq_lengths is not None:
            mask_packed_sequence_boundaries(shift_labels, packed_seq_lengths)
        inputs["shift_labels"] = shift_labels

    @contextlib.contextmanager
    def apply(self, inputs: dict[str, torch.Tensor]) -> Iterator[None]:
        """Wrap training step to shard buffers and patch SDPA for ring attention."""
        token = _ACTIVE_MANAGER.set(self)
        self._ensure_position_ids(inputs)
        self._ensure_shift_labels(inputs)
        buffers, seq_dims, no_restore = self._collect_buffers(inputs)
        with context_parallel(
            self._mesh,
            buffers = buffers,
            buffer_seq_dims = seq_dims,
            no_restore_buffers = no_restore,
        ):
            yield
        _ACTIVE_MANAGER.reset(token)

    def _set_report_loss(self, value: torch.Tensor) -> None:
        self._report_loss = value.detach() if torch.is_tensor(value) else None

    def consume_report_loss(self) -> Optional[torch.Tensor]:
        value = self._report_loss
        self._report_loss = None
        return value

    def reduce_loss(self, loss, inputs):
        if self._cp_group is None:
            return loss

        # Handle (loss, outputs) tuple from return_outputs=True
        is_tuple = isinstance(loss, tuple)
        if is_tuple:
            tensor, rest = loss[0], loss[1:]
        else:
            tensor = loss

        # Count local valid tokens
        shift_labels = inputs["shift_labels"]
        local_tokens = (
            shift_labels.ne(-100).sum().to(dtype = tensor.dtype, device = tensor.device)
        )

        # Get global token count
        global_tokens = local_tokens.clone()
        dist.all_reduce(global_tokens, op = dist.ReduceOp.SUM, group = self._cp_group)

        # Weight loss by local fraction
        weight = local_tokens.detach() / global_tokens.detach()
        weighted_loss = tensor * weight

        # Reduce for reporting
        global_loss = weighted_loss.detach().clone()
        dist.all_reduce(global_loss, op = dist.ReduceOp.SUM, group = self._cp_group)
        self._set_report_loss(global_loss)

        return (weighted_loss, *rest) if is_tuple else weighted_loss

    def reduce_grad_norm(self, grad_norm: float) -> float:
        """
        Reduce gradient norm across CP group.

        Each rank computes a local gradient norm (L2). The global norm is:
            sqrt(sum(local_norm_i^2 for all ranks))

        This is needed because the Trainer computes grad_norm locally, but with
        CP each rank only has partial gradients for its sequence shard.
        """
        if self._cp_group is None:
            return grad_norm

        # Square the local norm, sum across ranks, then sqrt
        local_norm_sq = torch.tensor(
            grad_norm**2,
            dtype = torch.float32,
            device = torch.device(DEVICE_TYPE_TORCH),
        )
        dist.all_reduce(local_norm_sq, op = dist.ReduceOp.SUM, group = self._cp_group)
        return float(local_norm_sq.sqrt().item())


def patch_sft_config():
    """Patch SFTConfig to add context_parallel_size and shuffle_dataset fields."""
    base_cls = trl.SFTConfig
    if hasattr(base_cls, "context_parallel_size"):
        return

    @dataclass
    class PatchedSFTConfig(base_cls):  # type: ignore[misc, valid-type]
        context_parallel_size: int = field(
            default = 1,
            metadata = {
                "help": (
                    "Number of ranks participating in context parallelism. "
                    "Set to 1 to disable context parallelism."
                )
            },
        )
        shuffle_dataset: bool = field(
            default = True,
            metadata = {
                "help": (
                    "Whether to shuffle the training dataset before each epoch. "
                    "Exposed for CP = 1 vs. CP > 1 debugging purposes."
                )
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
    """Patch SFTTrainer to add context parallelism support."""
    trainer_cls = trl.SFTTrainer
    if hasattr(trainer_cls, "__unsloth_context_parallel__"):
        return

    original_init = trainer_cls.__init__
    original_compute_loss = trainer_cls.compute_loss
    original_prediction_step = trainer_cls.prediction_step
    original_training_step = trainer_cls.training_step
    original_log = trainer_cls.log
    original_get_train_sampler = getattr(trainer_cls, "_get_train_sampler", None)

    def _patch_train_sampler(original_fn):
        @functools.wraps(original_fn)
        def wrapper(self, *args, **kwargs):
            sampler = original_fn(self, *args, **kwargs)
            manager = getattr(self, "_context_parallel_manager", None)
            dataset = args[0] if args else None
            if dataset is None:
                dataset = getattr(self, "train_dataset", None)
            shuffle_dataset = getattr(self.args, "shuffle_dataset", True)
            if (
                manager
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
                and dataset is not None
            ):
                dp_world = manager.data_parallel_world_size
                world_size = torch.distributed.get_world_size()
                if dp_world != world_size:
                    try:
                        from torch.utils.data.distributed import DistributedSampler
                    except ImportError:
                        return sampler
                    dp_rank = manager.data_parallel_rank()
                    shuffle = shuffle_dataset and not getattr(
                        self.args, "group_by_length", False
                    )
                    return DistributedSampler(
                        dataset,
                        num_replicas = dp_world,
                        rank = dp_rank,
                        shuffle = shuffle,
                        drop_last = getattr(self.args, "dataloader_drop_last", False),
                    )
            if not shuffle_dataset and dataset is not None:
                try:
                    from torch.utils.data import SequentialSampler
                except ImportError:
                    return sampler
                return SequentialSampler(dataset)
            return sampler

        return wrapper

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        settings = ContextParallelSettings.from_args(getattr(self, "args", None))
        if settings.size > 1:
            if context_parallel is None or DeviceMesh is None:
                warnings.warn(
                    "Context parallelism requested but PyTorch >= 2.7 is required.",
                    stacklevel = 2,
                )
                self._context_parallel_manager = None
            else:
                self._context_parallel_manager = ContextParallelManager(settings)
        else:
            self._context_parallel_manager = None
        accelerator = getattr(self, "accelerator", None)
        manager = self._context_parallel_manager
        if manager:
            print(
                f"Unsloth: Context parallelism enabled with size={manager.settings.size}"
            )
        mesh = getattr(manager, "device_mesh", None) if manager else None
        existing_mesh = (
            getattr(accelerator, "torch_device_mesh", None)
            if accelerator is not None
            else None
        )
        if (
            accelerator is not None
            and mesh is not None
            and (
                existing_mesh is None
                or "cp" not in getattr(existing_mesh, "mesh_dim_names", ())
            )
        ):
            setattr(accelerator.state, "device_mesh", mesh)

        # When using pure context parallelism (dp_world_size=1), disable DDP
        # to avoid gradient checkpointing compatibility issues
        if manager and manager.data_parallel_world_size == 1:
            try:
                from accelerate.utils import DistributedType

                args = getattr(self, "args", None)
                distributed_state = getattr(args, "distributed_state", None)
                if (
                    distributed_state is not None
                    and distributed_state.distributed_type == DistributedType.MULTI_GPU
                ):
                    distributed_state.distributed_type = DistributedType.NO
            except ImportError:
                pass

        # Enable sync_each_batch when using CP with gradient accumulation and DDP.
        # This keeps the computation graph constant for DDP + static_graph mode.
        if (
            manager
            and manager.data_parallel_world_size > 1  # Only needed with actual DP
            and accelerator is not None
            and hasattr(accelerator, "gradient_state")
        ):
            grad_accum_steps = getattr(
                getattr(self, "args", None), "gradient_accumulation_steps", 1
            )
            if grad_accum_steps > 1:
                accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

        # Attach attention hooks for proper ring attention behavior with load balancing.
        # This ensures attention_mask is removed and is_causal=True for all self_attn calls.
        if manager:
            model = getattr(self, "model", None)
            if model is not None:
                manager.attach_attention_hooks(model)

    @functools.wraps(original_compute_loss)
    def patched_compute_loss(self, model, inputs, return_outputs = False, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        kwargs.pop("num_items_in_batch", None)

        # For context parallelism with shift_labels, prefer letting the model
        # handle the pre-shifted targets when it advertises support. Otherwise
        # fall back to an external loss that consumes the sharded tensors.
        shift_labels = inputs.get("shift_labels")
        use_cp_shift_labels = manager and isinstance(shift_labels, torch.Tensor)
        model_supports_shift_labels = bool(
            getattr(
                model,
                "_unsloth_supports_context_parallel_shift_labels",
                False,
            )
        )

        if use_cp_shift_labels and not model_supports_shift_labels:
            # Remove labels so model doesn't compute loss internally
            saved_labels = inputs.pop("labels", None)
            # Also remove shift_labels from inputs (model doesn't expect it)
            local_shift_labels = inputs.pop("shift_labels", None)

            # Get model outputs (logits only, no loss)
            outputs = model(**inputs)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]

            # Compute loss using pre-shifted labels
            from unsloth.kernels.cross_entropy_loss import fast_cross_entropy_loss

            loss = fast_cross_entropy_loss(
                logits = logits,
                labels = local_shift_labels,
            )

            # Restore labels for reduce_loss token counting
            if saved_labels is not None:
                inputs["labels"] = saved_labels
            if local_shift_labels is not None:
                inputs["shift_labels"] = local_shift_labels

            if return_outputs:
                loss = (loss, outputs)
        else:
            loss = original_compute_loss(
                self,
                model,
                inputs,
                return_outputs = return_outputs,
                **kwargs,
            )

        if manager:
            loss = manager.reduce_loss(loss, inputs)
        return loss

    @functools.wraps(original_prediction_step)
    def patched_prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys = None,
        **kwargs,
    ):
        manager = getattr(self, "_context_parallel_manager", None)
        context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with context:
            return original_prediction_step(
                self,
                model,
                inputs,
                prediction_loss_only,
                ignore_keys,
                **kwargs,
            )

    def _maybe_enable_sync_each_batch(trainer):
        """Enable sync_each_batch at runtime if gradient checkpointing is detected."""
        if getattr(trainer, "_sync_each_batch_checked", False):
            return
        setattr(trainer, "_sync_each_batch_checked", True)

        accelerator = getattr(trainer, "accelerator", None)
        if accelerator is None or not hasattr(accelerator, "gradient_state"):
            return

        # Check if already enabled
        if accelerator.gradient_state.plugin_kwargs.get("sync_each_batch", False):
            return

        model = getattr(trainer, "model", None)
        is_checkpointing = getattr(model, "is_gradient_checkpointing", False)
        grad_accum_steps = getattr(trainer.args, "gradient_accumulation_steps", 1)

        if is_checkpointing and grad_accum_steps > 1:
            accelerator.gradient_state.plugin_kwargs["sync_each_batch"] = True

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        original_n_gpu = getattr(self.args, "n_gpu", 1)
        if manager:
            setattr(self.args, "_n_gpu", manager.data_parallel_world_size)
            _maybe_enable_sync_each_batch(self)
            # Attach attention hooks if not already done (model may not be ready at init)
            if not manager._attention_hook_handles:
                m = getattr(self, "model", None)
                if m is not None:
                    manager.attach_attention_hooks(m)

        # Wrap entire training step (forward + backward) in context_parallel
        # This keeps SDPA patched and buffers sharded throughout, including
        # during gradient checkpoint recomputation in backward pass.
        cp_context = manager.apply(inputs) if manager else contextlib.nullcontext()
        with cp_context:
            loss = original_training_step(self, model, inputs, *args, **kwargs)

        if manager:
            setattr(self.args, "_n_gpu", original_n_gpu)

        report_loss = manager.consume_report_loss() if manager else None
        if report_loss is not None:
            return report_loss
        return loss

    @functools.wraps(original_log)
    def patched_log(self, logs, start_time = None):
        manager = getattr(self, "_context_parallel_manager", None)
        # Reduce grad_norm across CP group if present
        if manager and "grad_norm" in logs:
            grad_norm = logs["grad_norm"]
            if isinstance(grad_norm, (int, float)):
                logs["grad_norm"] = manager.reduce_grad_norm(grad_norm)
        return original_log(self, logs, start_time)

    trainer_cls.__init__ = patched_init
    trainer_cls.compute_loss = patched_compute_loss
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.training_step = patched_training_step
    trainer_cls.log = patched_log
    trainer_cls.__unsloth_context_parallel__ = True
    if original_get_train_sampler is not None:
        trainer_cls._get_train_sampler = _patch_train_sampler(
            original_get_train_sampler
        )
