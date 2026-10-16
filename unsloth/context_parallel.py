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
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    context_parallel = None
    DeviceMesh = None
    DTensor = None

try:
    # Re-exported from _context_parallel._attention on torch >= 2.10; defined here on 2.7 - 2.9.
    from torch.distributed.tensor.experimental._attention import _cp_options, set_rotate_method
except (ImportError, AttributeError):
    _cp_options = None
    set_rotate_method = None

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:
    SDPBackend = None
    sdpa_kernel = None

_CP_BACKENDS = ("ring", "ulysses")
_ROTATE_METHODS = ("allgather", "alltoall")


def _torch_supports_context_parallel() -> bool:
    from packaging.version import Version
    return context_parallel is not None and Version(
        Version(torch.__version__).base_version
    ) >= Version("2.7.0")


def _ring_sdpa_backends():
    # The math kernel is not ringed: it would attend over the local shard only.
    if SDPBackend is None:
        return None
    backends = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    cudnn = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    op = getattr(torch.ops.aten, "_scaled_dot_product_cudnn_attention", None)
    if cudnn is not None and op is not None:
        try:
            from torch.distributed.tensor.experimental._context_parallel import _attention
        except ImportError:
            try:
                from torch.distributed.tensor.experimental import _attention
            except ImportError:
                _attention = None
        handled = getattr(_attention, "custom_ops", None)
        if isinstance(handled, dict) and op.default in handled:
            backends.append(cudnn)
    return backends


def context_parallel_options(config) -> tuple:
    """(size, backend, rotate_method) from a training config, validated before any dataset work."""
    size = getattr(config, "context_parallel_size", 1)
    size = 1 if size is None else size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(
            f"Unsloth: context_parallel_size must be a positive integer, got {size!r}."
        )
    backend = getattr(config, "context_parallel_backend", None) or "ring"
    if backend not in _CP_BACKENDS:
        raise ValueError(
            f"Unsloth: context_parallel_backend must be one of {_CP_BACKENDS}, got {backend!r}."
        )
    rotate_method = getattr(config, "context_parallel_rotate_method", None) or "allgather"
    if rotate_method not in _ROTATE_METHODS:
        raise ValueError(
            "Unsloth: context_parallel_rotate_method must be one of "
            f"{_ROTATE_METHODS}, got {rotate_method!r}."
        )
    return size, backend, rotate_method


def _unsupported_attention(config) -> Optional[str]:
    # Ring and Llama's forward ignore mask, window and softcap, so both backends refuse them.
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)
    for c in configs:
        if c is None:
            continue
        layer_types = getattr(c, "layer_types", None)
        if layer_types and any(t != "full_attention" for t in layer_types):
            kinds = sorted({t for t in layer_types if t != "full_attention"})
            return f"{', '.join(kinds)} layers"
        # Any positive window, even with use_sliding_window off: Unsloth's masks read it directly.
        sliding_window = getattr(c, "sliding_window", None)
        if (isinstance(sliding_window, int) and sliding_window > 0) or getattr(
            c, "use_sliding_window", None
        ) is True:
            return "sliding window attention"
        if getattr(c, "attention_chunk_size", None):
            return "chunked attention"
        if getattr(c, "attn_logit_softcapping", None):
            return "attention logit softcapping"
    return None


def _unregister_cp_sharding_rules() -> None:
    try:
        from torch.distributed.tensor.experimental._context_parallel._sharding_rules import (
            unregister_cp_sharding_rules,
        )
    except ImportError:
        return
    unregister_cp_sharding_rules(clear_the_cache = False)


from .device_type import DEVICE_TYPE_TORCH
from .utils.ulysses import _document_positions, ulysses_head_plan

# A module global, not a ContextVar: CUDA runs backward (and checkpoint recompute) on another thread.
_ACTIVE_MANAGER: Optional["ContextParallelManager"] = None

# shift_labels is built before sharding so each shard's last token keeps its next-token target.
_BUFFER_NAMES = ("input_ids", "attention_mask", "labels", "position_ids", "shift_labels")
_PAD_VALUES = {"labels": -100, "shift_labels": -100, "attention_mask": 0, "input_ids": 0}


_INSTALLED_MESH = []
_INSTALLED_PARALLELISM_CONFIG = []


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER


def _supports_context_parallel(model) -> bool:
    from .models.llama import LlamaAttention_fast_forward
    forwards = [
        getattr(type(module).forward, "__func__", type(module).forward)
        for name, module in model.named_modules()
        if name.endswith("self_attn")
    ]
    return bool(forwards) and all(f is LlamaAttention_fast_forward for f in forwards)


def _self_attn_pre_forward_hook(_module, module_args, module_kwargs):
    if get_cp_manager() is not None and "attention_mask" in module_kwargs:
        module_kwargs["attention_mask"] = None
    return module_args, module_kwargs


class ContextParallelManager:
    backend = "ring"
    rotate_method = "allgather"

    def __init__(
        self,
        size: int,
        backend: str = "ring",
        rotate_method: str = "allgather",
        fsdp2: bool = False,
    ):
        self.size = size
        self.backend = backend
        self.rotate_method = rotate_method
        self.fsdp2 = fsdp2
        world_size = dist.get_world_size()
        # FSDP2 shards parameters over dp_shard x cp flattened (torchtitan, accelerate's dp_shard_cp):
        # CP peers hold different tokens, so their gradients must be reduced like data parallel ones.
        self.device_mesh = DeviceMesh(
            DEVICE_TYPE_TORCH,
            torch.arange(world_size).reshape(world_size // size, size),
            mesh_dim_names = ("dp_shard" if fsdp2 else "dp_replicate", "cp"),
        )
        if fsdp2:
            self.device_mesh[("dp_shard", "cp")]._flatten("dp_shard_cp")
        self.mesh = self.device_mesh["cp"]

    @property
    def group(self):
        return self.mesh.get_group()

    def attach_attention_hooks(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if name.endswith("self_attn") and not getattr(module, "_unsloth_cp_hooked", False):
                module.register_forward_pre_hook(
                    _self_attn_pre_forward_hook, with_kwargs = True, prepend = True
                )
                module._unsloth_cp_hooked = True

    def _prepare_inputs(self, inputs: dict) -> None:
        input_ids = inputs.get("input_ids")
        if not isinstance(input_ids, torch.Tensor) or inputs.get("inputs_embeds") is not None:
            raise ValueError(
                "Unsloth: context parallelism needs input_ids batches (not inputs_embeds)."
            )
        bsz, seq_len = input_ids.shape
        packed_seq_lengths = inputs.get("packed_seq_lengths")
        ulysses = self.backend == "ulysses"
        if packed_seq_lengths is not None and not ulysses:
            raise ValueError("Unsloth: ring context parallelism does not support packing.")
        if packed_seq_lengths is not None:
            if bsz != 1:
                raise ValueError(f"Unsloth: packed batches must be a single row, got {bsz} rows.")
            used = int(torch.as_tensor(packed_seq_lengths).sum())
            if used > seq_len:
                raise ValueError(
                    f"Unsloth: packed_seq_lengths sum to {used} but the packed row has {seq_len} tokens."
                )
        # .get: a collator may carry the key with None, which would leave positions shard-local.
        if inputs.get("position_ids") is None and packed_seq_lengths is not None:
            inputs["position_ids"] = _document_positions(
                packed_seq_lengths, seq_len, input_ids.device
            )
        elif inputs.get("position_ids") is None:
            inputs["position_ids"] = (
                torch.arange(seq_len, device = input_ids.device).expand(bsz, -1).contiguous()
            )
        mask = inputs.get("attention_mask")
        if isinstance(mask, torch.Tensor) and (
            mask.ndim != 2 or (mask[:, 1:] > mask[:, :-1]).any()
        ):
            raise ValueError(
                "Unsloth: context parallelism needs 2D right-padded attention masks without holes."
            )
        labels = inputs.get("labels")
        if inputs.get("shift_labels") is None and labels is not None:
            # Shard-local labels cannot see document starts on another shard: mask them here.
            from .utils.packing import mask_packed_boundary_labels
            labels = mask_packed_boundary_labels(labels, packed_seq_lengths)
            inputs["shift_labels"] = F.pad(labels, (0, 1), value = -100)[:, 1:].contiguous()
        pad = (-seq_len) % (self.size if ulysses else 2 * self.size)
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
    def _rotate_method_scope(self) -> Iterator[None]:
        # Process-wide torch state, read in forward and backward: set per step, restored after.
        rotate_method = self.rotate_method
        if _cp_options is None or set_rotate_method is None:
            if rotate_method != "allgather":
                raise RuntimeError(
                    f"Unsloth: this PyTorch cannot set the ring rotate method to {rotate_method!r}."
                )
            yield
            return
        previous = _cp_options.rotate_method
        set_rotate_method(rotate_method)
        try:
            yield
        finally:
            _cp_options.rotate_method = previous

    @contextlib.contextmanager
    def _ulysses_scope(self, inputs: dict) -> Iterator[None]:
        # Contiguous shards; packed_seq_lengths stays global.
        rank = self.mesh.get_local_rank()
        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 1:
                local = tensor.shape[1] // self.size
                inputs[name] = tensor[:, rank * local : (rank + 1) * local].contiguous()
        global _ACTIVE_MANAGER
        previous, _ACTIVE_MANAGER = _ACTIVE_MANAGER, self
        try:
            yield
        finally:
            _ACTIVE_MANAGER = previous

    @contextlib.contextmanager
    def apply(self, inputs: dict) -> Iterator[None]:
        self._prepare_inputs(inputs)
        if self.backend == "ulysses":
            with self._ulysses_scope(inputs):
                yield
            return
        buffers = [
            inputs[name]
            for name in _BUFFER_NAMES
            if isinstance(inputs.get(name), torch.Tensor) and inputs[name].ndim > 1
        ]
        global _ACTIVE_MANAGER
        previous, _ACTIVE_MANAGER = _ACTIVE_MANAGER, self
        # torch < 2.13 leaves its CP dispatcher installed when the step raises.
        sdpa = F.scaled_dot_product_attention
        dispatcher = getattr(DTensor, "_op_dispatcher", None)
        handlers = getattr(dispatcher, "_custom_op_handlers", None)
        ring_backends = _ring_sdpa_backends()
        raised = False
        try:
            with (
                (
                    sdpa_kernel(ring_backends)
                    if ring_backends is not None
                    else contextlib.nullcontext()
                ),
                self._rotate_method_scope(),
                context_parallel(
                    self.mesh,
                    buffers = buffers,
                    buffer_seq_dims = [1] * len(buffers),
                    no_restore_buffers = set(buffers),
                ),
            ):
                yield
        except BaseException:
            raised = True
            raise
        finally:
            _ACTIVE_MANAGER = previous
            F.scaled_dot_product_attention = sdpa
            if raised:
                if handlers is not None:
                    dispatcher._custom_op_handlers = handlers
                _unregister_cp_sharding_rules()


def _refuse_iterable_datasets(*datasets) -> None:
    # accelerate's batch dispatcher (dispatch_batches, the default for iterable datasets) ignores cp.
    for dataset in datasets:
        for d in dataset.values() if isinstance(dataset, dict) else [dataset]:
            if (
                isinstance(d, torch.utils.data.IterableDataset)
                or "IterableDataset" in type(d).__name__
            ):
                raise NotImplementedError(
                    "Unsloth: context parallelism does not support iterable datasets."
                )


def _cp_gather_for_metrics(accelerator, size: int):
    # Keep one loss per CP group, else last-batch truncation keeps duplicates and drops replicas.
    def gather(tensor, *args, **kwargs):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            return accelerator.gather_for_metrics(tensor, *args, **kwargs)
        gathered = accelerator.gather(tensor)
        gathered = gathered.reshape(-1, size, *tensor.shape)[:, 0].reshape(-1, *tensor.shape[1:])
        state = accelerator.gradient_state
        if state.end_of_dataloader and state.remainder > 0:
            gathered = gathered[: state.remainder]
        return gathered

    return gather


def _is_fsdp2(accelerator, distributed_type: str) -> bool:
    plugin = getattr(getattr(accelerator, "state", None), "fsdp_plugin", None)
    return distributed_type == "FSDP" and getattr(plugin, "fsdp_version", 1) == 2


def _check_fsdp2_parallelism_config(accelerator) -> None:
    pc = getattr(getattr(accelerator, "state", None), "parallelism_config", None)
    if pc is None:
        return
    others = {name: getattr(pc, f"{name}_size", 1) or 1 for name in ("dp_replicate", "tp", "sp")}
    others = {name: n for name, n in others.items() if n > 1}
    if others:
        raise NotImplementedError(
            "Unsloth: context parallelism with FSDP2 supports a (dp_shard, cp) mesh only, "
            f"not parallelism_config {others}."
        )


def _install_fsdp2_parallelism_config(state, manager) -> None:
    # accelerate's fsdp2_prepare_model shards over mesh[parallelism_config.fsdp_dim_names], i.e.
    # "dp_shard_cp" (accelerate >= 1.10). cp_size stays 1 here: cp_size > 1 would make accelerate attach
    # its own ring hooks and transformers divide num_items_in_batch by cp, both done by Unsloth already.
    from accelerate.parallelism_config import ParallelismConfig

    current = getattr(state, "parallelism_config", None)
    if _INSTALLED_PARALLELISM_CONFIG and current is _INSTALLED_PARALLELISM_CONFIG[1]:
        previous = _INSTALLED_PARALLELISM_CONFIG[0]
    else:
        previous = current
    pc = ParallelismConfig(
        dp_replicate_size = 1,
        dp_shard_size = manager.device_mesh["dp_shard_cp"].size(),
        tp_size = 1,
        cp_size = 1,
    )
    state.parallelism_config = pc
    _INSTALLED_PARALLELISM_CONFIG[:] = [previous, pc]


def _restore_parallelism_config(state) -> None:
    if _INSTALLED_PARALLELISM_CONFIG and (
        getattr(state, "parallelism_config", None) is _INSTALLED_PARALLELISM_CONFIG[1]
    ):
        state.parallelism_config = _INSTALLED_PARALLELISM_CONFIG[0]
        _INSTALLED_PARALLELISM_CONFIG.clear()


def patch_sft_trainer() -> None:
    import trl

    trainer_cls = trl.SFTTrainer
    if trainer_cls.__dict__.get("__unsloth_context_parallel__"):
        return

    def _original(name):
        # Inherited methods are looked up per call: Trainer.train etc. may be patched after import.
        if name in trainer_cls.__dict__:
            return trainer_cls.__dict__[name]

        @functools.wraps(getattr(trainer_cls, name, None))
        def inherited(self, *args, **kwargs):
            return getattr(super(trainer_cls, self), name)(*args, **kwargs)

        return inherited

    original_init = trainer_cls.__init__
    original_prediction_step = _original("prediction_step")
    original_training_step = _original("training_step")
    original_train = _original("train")
    original_evaluate = _original("evaluate")
    original_predict = _original("predict")
    original_evaluation_loop = _original("evaluation_loop")
    has_total_train_batch_size = hasattr(trainer_cls, "get_total_train_batch_size")
    if has_total_train_batch_size:
        original_get_total_train_batch_size = _original("get_total_train_batch_size")

    def _install_accelerator_state(self):
        manager = getattr(self, "_context_parallel_manager", None)
        accelerator = getattr(self, "accelerator", None)
        state = getattr(accelerator, "state", None)
        if manager is None:
            if _INSTALLED_MESH and getattr(state, "device_mesh", None) is _INSTALLED_MESH[0]:
                # The mesh an accepted FSDP2 ParallelismConfig indexes, restored with that config.
                state.device_mesh = _INSTALLED_MESH[1] if len(_INSTALLED_MESH) > 1 else None
                _INSTALLED_MESH.clear()
            _restore_parallelism_config(state)
            return
        if accelerator is None:
            return
        current = getattr(accelerator.state, "device_mesh", None)
        if _INSTALLED_MESH and current is _INSTALLED_MESH[0]:
            current = _INSTALLED_MESH[1] if len(_INSTALLED_MESH) > 1 else None
        accelerator.state.device_mesh = manager.device_mesh
        _INSTALLED_MESH[:] = [manager.device_mesh, current]
        if getattr(manager, "fsdp2", False):
            _install_fsdp2_parallelism_config(accelerator.state, manager)
        else:
            _restore_parallelism_config(accelerator.state)
        # PEFT makes the Trainer ask DDP for find_unused_parameters, which breaks reentrant checkpointing.
        if getattr(self.args, "ddp_find_unused_parameters", None) in (None, False):
            self.args.ddp_find_unused_parameters = False
            handler = getattr(accelerator, "ddp_handler", None)
            if handler is not None:
                handler.find_unused_parameters = False

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        config = kwargs["args"] if "args" in kwargs else (args[1] if len(args) > 1 else None)
        context_parallel_options(config)
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = None
        # Unsloth's attention ignores transformers' own CP: it would stay local to each shard.
        parallelism_config = getattr(getattr(self, "args", None), "parallelism_config", None)
        if (getattr(parallelism_config, "cp_size", 1) or 1) > 1:
            raise NotImplementedError(
                "Unsloth: use SFTConfig(context_parallel_size = N) instead of "
                "parallelism_config with cp_size > 1."
            )
        size, backend, rotate_method = context_parallel_options(getattr(self, "args", None))
        if size <= 1:
            _install_accelerator_state(self)
            return
        # train() would swap in a fresh model without the attention hooks or the support check.
        if getattr(self, "model_init", None) is not None:
            raise NotImplementedError("Unsloth: context parallelism does not support model_init.")
        if not _torch_supports_context_parallel():
            raise RuntimeError(
                f"Unsloth: context_parallel_size > 1 needs PyTorch >= 2.7, found {torch.__version__}."
            )
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
        if (
            backend == "ring"
            and torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] < 8
        ):
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
        if getattr(self, "model_accepts_loss_kwargs", True) is False:
            raise NotImplementedError(
                "Unsloth: context parallelism needs a model whose forward accepts loss kwargs "
                "(num_items_in_batch)."
            )
        if getattr(self.args, "use_legacy_prediction_loop", False):
            raise NotImplementedError(
                "Unsloth: context parallelism needs use_legacy_prediction_loop = False."
            )
        if getattr(self.args, "average_tokens_across_devices", True) is False:
            raise NotImplementedError(
                "Unsloth: context parallelism needs average_tokens_across_devices = True."
            )
        import accelerate
        from packaging.version import Version

        if Version(accelerate.__version__) < Version("1.10.0"):
            raise NotImplementedError("Unsloth: context parallelism needs accelerate >= 1.10.0.")
        model_config = getattr(self.model, "config", None)
        unsupported = _unsupported_attention(model_config)
        if unsupported is not None:
            raise NotImplementedError(
                f"Unsloth: {backend} context parallelism does not support {unsupported}."
            )
        if backend == "ulysses":
            # Every rank raises here, before any collective.
            text_config = getattr(model_config, "text_config", None) or model_config
            ulysses_head_plan(
                text_config.num_attention_heads,
                getattr(text_config, "num_key_value_heads", None)
                or text_config.num_attention_heads,
                size,
            )
        if not _supports_context_parallel(self.model):
            raise NotImplementedError(
                "Unsloth: context parallelism currently supports Llama-style attention only "
                "(Llama, Qwen2, Gemma)."
            )
        accelerator = getattr(self, "accelerator", None)
        distributed_type = getattr(getattr(accelerator, "distributed_type", None), "name", "NO")
        fsdp2 = _is_fsdp2(accelerator, distributed_type)
        if distributed_type != "NO" and not distributed_type.startswith("MULTI_") and not fsdp2:
            raise NotImplementedError(
                f"Unsloth: context parallelism supports DDP and FSDP2 only, not {distributed_type}"
                + (" version 1." if distributed_type == "FSDP" else ".")
            )
        if fsdp2:
            _check_fsdp2_parallelism_config(accelerator)
        if getattr(accelerator, "dispatch_batches", None):
            raise NotImplementedError(
                "Unsloth: context parallelism does not support dispatch_batches."
            )
        _refuse_iterable_datasets(
            getattr(self, "train_dataset", None), getattr(self, "eval_dataset", None)
        )
        manager = ContextParallelManager(size, backend, rotate_method, fsdp2 = fsdp2)
        self._context_parallel_manager = manager
        manager.attach_attention_hooks(self.model)
        _install_accelerator_state(self)
        # The ring calls the flash ops FA3 / FA4 override, so scope them to train() here too.
        from .utils.packing import _scope_flash_attention_impl

        _scope_flash_attention_impl(self.model, self)
        print(
            f"Unsloth: Context parallelism enabled with size = {size} ({backend}"
            + (f", rotate_method = {rotate_method})" if backend == "ring" else ")")
            + (
                f", FSDP2 sharding over dp_shard = {world_size // size} x cp = {size}."
                if fsdp2
                else "."
            )
        )

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
        inputs["num_items_in_batch"] = shift_labels.ne(-100).sum().clamp_min(1)
        inputs.setdefault("use_cache", False)
        with manager.apply(inputs), torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs).detach()
        dist.all_reduce(loss, group = manager.mesh.get_group())
        return loss, None, None

    @functools.wraps(original_train)
    def patched_train(self, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            _refuse_iterable_datasets(getattr(self, "train_dataset", None))
        _install_accelerator_state(self)
        return original_train(self, *args, **kwargs)

    @functools.wraps(original_evaluate)
    def patched_evaluate(self, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            eval_dataset = args[0] if args else kwargs.get("eval_dataset")
            if eval_dataset is None or isinstance(eval_dataset, str):
                current = self.eval_dataset
                eval_dataset = (
                    current.get(eval_dataset)
                    if isinstance(eval_dataset, str) and isinstance(current, dict)
                    else current
                )
            _refuse_iterable_datasets(eval_dataset)
        _install_accelerator_state(self)
        return original_evaluate(self, *args, **kwargs)

    @functools.wraps(original_predict)
    def patched_predict(self, test_dataset, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            _refuse_iterable_datasets(test_dataset)
        _install_accelerator_state(self)
        return original_predict(self, test_dataset, *args, **kwargs)

    @functools.wraps(original_evaluation_loop)
    def patched_evaluation_loop(self, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is not None:
            self.gather_function = _cp_gather_for_metrics(self.accelerator, manager.size)
        return original_evaluation_loop(self, *args, **kwargs)

    @functools.wraps(original_training_step)
    def patched_training_step(self, model, inputs, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is not None:
            num_items = args[0] if args else kwargs.get("num_items_in_batch")
            if (
                num_items is None
                and hasattr(inputs, "get")
                and (inputs.get("labels") is not None or inputs.get("shift_labels") is not None)
            ):
                raise RuntimeError(
                    "Unsloth: context parallelism needs num_items_in_batch in training_step; "
                    "this transformers / trainer path did not pass one."
                )
            # HF summed the count over all D x N ranks while CP peers share a batch: divide by N (exact).
            if args and args[0] is not None:
                args = (args[0] // manager.size, *args[1:])
            elif kwargs.get("num_items_in_batch") is not None:
                kwargs["num_items_in_batch"] = kwargs["num_items_in_batch"] // manager.size
        # Forward and backward both inside, so gradient checkpointing recomputes with ring attention.
        with manager.apply(inputs) if manager else contextlib.nullcontext():
            return original_training_step(self, model, inputs, *args, **kwargs)

    if has_total_train_batch_size:

        @functools.wraps(original_get_total_train_batch_size)
        def patched_get_total_train_batch_size(self, *args, **kwargs):
            total = original_get_total_train_batch_size(self, *args, **kwargs)
            manager = getattr(self, "_context_parallel_manager", None)
            # CP peers share one batch, but the parallelism_config Unsloth exposes (DDP: none,
            # FSDP2: cp_size = 1) makes transformers count all D x N ranks as data parallel.
            if manager is not None and manager.size > 1:
                total = total // manager.size
            return total

        trainer_cls.get_total_train_batch_size = patched_get_total_train_batch_size

    trainer_cls.__init__ = patched_init
    trainer_cls.prediction_step = patched_prediction_step
    trainer_cls.train = patched_train
    trainer_cls.evaluate = patched_evaluate
    trainer_cls.predict = patched_predict
    trainer_cls.evaluation_loop = patched_evaluation_loop
    trainer_cls.training_step = patched_training_step
    trainer_cls.__unsloth_context_parallel__ = True
