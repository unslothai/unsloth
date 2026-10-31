# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

from __future__ import annotations

import contextlib
import functools
import weakref
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


def _cp_replaced_functions():
    # torch's CP registry of SDPA wrappers; restore never drops entries, so each step adds one.
    # torch >= 2.10 keeps it in _context_parallel._attention, 2.7 - 2.9 in experimental._attention
    # (a stub without it on 2.10+).
    modules = []
    try:
        from torch.distributed.tensor.experimental._context_parallel import _attention as new_layout
        modules.append(new_layout)
    except ImportError:
        pass
    try:
        from torch.distributed.tensor.experimental import _attention as old_layout
        modules.append(old_layout)
    except ImportError:
        pass
    for module in modules:
        registry = getattr(module, "_replaced_functions", None)
        if isinstance(registry, dict):
            return registry
    return None


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


def _unsupported_attention(config, backend = "ring") -> Optional[str]:
    # Ring and Llama's forward ignore mask, window and softcap, so both backends refuse them.
    allowed = {"full_attention", "linear_attention"} if backend == "ulysses" else {"full_attention"}
    configs = [config]
    text_config = getattr(config, "text_config", None)
    if text_config is not None and text_config is not config:
        configs.append(text_config)
    for c in configs:
        if c is None:
            continue
        layer_types = getattr(c, "layer_types", None)
        if layer_types and any(t not in allowed for t in layer_types):
            kinds = sorted({t for t in layer_types if t not in allowed})
            hint = (
                " (use context_parallel_backend = 'ulysses')"
                if kinds == ["linear_attention"]
                else ""
            )
            return f"{', '.join(kinds)} layers{hint}"
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
from .utils.ulysses_hybrid import (
    GATED_DELTA_ATTENTION_CLASSES,
    check_gated_delta_heads,
    install_hf_attention_wrapper,
    is_ulysses_gated_delta,
    wrap_gated_delta_forward,
)

# A module global, not a ContextVar: CUDA runs backward (and checkpoint recompute) on another thread.
_ACTIVE_MANAGER: Optional["ContextParallelManager"] = None

# shift_labels is built before sharding so each shard's last token keeps its next-token target.
_BUFFER_NAMES = ("input_ids", "attention_mask", "labels", "position_ids", "shift_labels")
_PAD_VALUES = {"labels": -100, "shift_labels": -100, "attention_mask": 0, "input_ids": 0}
# Only the text buffers above are sharded; image / video / audio features would stay whole on every rank.
_MULTIMODAL_INPUTS = (
    "pixel_values",
    "pixel_values_videos",
    "image_grid_thw",
    "video_grid_thw",
    "input_features",
)


def _multimodal_inputs(inputs: dict) -> list:
    return sorted(
        name
        for name, value in inputs.items()
        if value is not None and (name in _MULTIMODAL_INPUTS or name.startswith("pixel_values"))
    )


_INSTALLED_MESH = []
_INSTALLED_PARALLELISM_CONFIG = []


def get_cp_manager() -> Optional["ContextParallelManager"]:
    return _ACTIVE_MANAGER


def _stateful_mixers(model) -> list:
    from .utils.packing import _iter_stateful_mixers
    return _iter_stateful_mixers(model) if isinstance(model, torch.nn.Module) else []


def _positions_restart(position_ids, attention_mask) -> bool:
    restart = position_ids[:, 1:] <= position_ids[:, :-1]
    if isinstance(attention_mask, torch.Tensor) and attention_mask.shape == position_ids.shape:
        # Right padding may carry any positions; only steps between real tokens count.
        valid = attention_mask.bool()
        restart = restart & valid[:, 1:] & valid[:, :-1]
    return bool(restart.any())


def _hybrid_config_blocker(model) -> Optional[str]:
    config = getattr(model, "config", None)
    # vision_config is allowed (Qwen3.5 ships one); image / video batches are refused in _prepare_inputs.
    text_config = getattr(config, "text_config", None) or config
    if getattr(text_config, "output_router_logits", False):
        # The load balancing loss is a product of per-sequence means, not a sum over shards.
        return "the MoE router auxiliary loss (output_router_logits)"
    if any(isinstance(getattr(m, "experts", None), torch.nn.ModuleList) for m in model.modules()):
        # Per-expert modules run only for experts the local shard routes to, so with DDP's
        # find_unused_parameters off an expert unused on one rank stalls the gradient reduction.
        return "sparse MoE layers with per-expert modules (transformers 4.x Qwen3-Next)"
    return None


def _supports_context_parallel(model, backend = "ring") -> bool:
    from .models.llama import LlamaAttention_fast_forward, runs_fast_forward

    modules = [module for name, module in model.named_modules() if name.endswith("self_attn")]
    if bool(modules) and all(runs_fast_forward(m, LlamaAttention_fast_forward) for m in modules):
        return not _stateful_mixers(model)
    mixers = _stateful_mixers(model)
    return (
        backend == "ulysses"
        and bool(mixers)
        and all(is_ulysses_gated_delta(m) for m, _ in mixers)
        and all(type(m).__name__ in GATED_DELTA_ATTENTION_CLASSES for m in modules)
        and _hybrid_config_blocker(model) is None
    )


def _check_stateful_mixers(model, backend, size) -> None:
    mixers = _stateful_mixers(model)
    if not mixers:
        return
    if backend != "ulysses":
        kinds = sorted({type(m).__name__ for m, _ in mixers})
        raise NotImplementedError(
            f"Unsloth: ring context parallelism cannot split the sequential scans of {kinds}; "
            "use context_parallel_backend = 'ulysses'."
        )
    unsupported = sorted({type(m).__name__ for m, _ in mixers if not is_ulysses_gated_delta(m)})
    if unsupported:
        raise NotImplementedError(
            "Unsloth: Ulysses context parallelism supports Gated DeltaNet linear attention only, "
            f"not {unsupported}."
        )
    for module, _ in mixers:
        check_gated_delta_heads(module, size)


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
        if self.backend == "ulysses":
            for module, _ in _stateful_mixers(model):
                if is_ulysses_gated_delta(module):
                    wrap_gated_delta_forward(module, get_cp_manager)
            for name, module in model.named_modules():
                if (
                    name.endswith("self_attn")
                    and type(module).__name__ in GATED_DELTA_ATTENTION_CLASSES
                ):
                    install_hf_attention_wrapper(
                        module, module.config._attn_implementation, get_cp_manager
                    )
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
        multimodal = _multimodal_inputs(inputs)
        if multimodal:
            raise ValueError(
                "Unsloth: context parallelism shards text tokens only; batches with image / video "
                f"inputs ({', '.join(multimodal)}) are not supported. Train on text-only data, or "
                "use context_parallel_size = 1."
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
        if inputs.get("output_router_logits"):
            raise ValueError(
                "Unsloth: context parallelism does not support the MoE router auxiliary loss "
                "(output_router_logits): it is a product of per-sequence means, not a sum over shards."
            )
        position_ids = inputs.get("position_ids")
        if isinstance(position_ids, torch.Tensor) and position_ids.ndim != 2:
            # Sharding slices dim 1, which is the batch axis of e.g. Qwen3.5's [4, batch, seq] ids.
            raise ValueError(
                f"Unsloth: context parallelism needs 2D [batch, seq] position_ids, got {position_ids.ndim}D."
            )
        if (
            ulysses
            and packed_seq_lengths is None
            and isinstance(position_ids, torch.Tensor)
            and position_ids.ndim == 2
            and _positions_restart(position_ids, inputs.get("attention_mask"))
        ):
            # Restarting positions mark packed documents that attention and scans could not see.
            raise ValueError(
                "Unsloth: Ulysses context parallelism needs packed_seq_lengths for packed batches; "
                "position_ids that restart are not enough."
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

    def _ulysses_shard(self, inputs: dict) -> None:
        # Contiguous shards; packed_seq_lengths stays global.
        rank = self.mesh.get_local_rank()
        for name in _BUFFER_NAMES:
            tensor = inputs.get(name)
            if isinstance(tensor, torch.Tensor) and tensor.ndim > 1:
                local = tensor.shape[1] // self.size
                inputs[name] = tensor[:, rank * local : (rank + 1) * local].contiguous()

    @contextlib.contextmanager
    def _ulysses_scope(self, inputs: dict) -> Iterator[None]:
        self._ulysses_shard(inputs)
        global _ACTIVE_MANAGER
        previous, _ACTIVE_MANAGER = _ACTIVE_MANAGER, self
        # Some decoders (transformers 5.3 / 5.5) do not forward packed_seq_lengths to linear_attn.
        self.packed_seq_lengths = inputs.get("packed_seq_lengths")
        try:
            yield
        finally:
            _ACTIVE_MANAGER = previous
            self.packed_seq_lengths = None

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
        registry = _cp_replaced_functions()
        wrappers = set(registry) if registry is not None else None
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
            if wrappers is not None:
                # Drop only the wrappers this step's context added, never one still installed.
                for wrapper in [w for w in registry if w not in wrappers]:
                    if wrapper is not F.scaled_dot_product_attention:
                        registry.pop(wrapper, None)


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


def _save_model_checks_parallelism_config_first() -> bool:
    # transformers < 5 Trainer.save_model takes any non-None parallelism_config before FSDP.
    try:
        import transformers
        from packaging.version import Version
        return Version(transformers.__version__).major < 5
    except Exception:
        return False


def _fsdp2_units_of(trainer):
    from .models.fsdp2_save import fsdp2_units
    return fsdp2_units(getattr(trainer, "model", None))


def _gathered_save_model(trainer, original_save_model, args, kwargs):
    """save_model under Unsloth's CP config on transformers < 5, which hands every rank to `_save`
    without a state dict. Gather like Unsloth's FSDP2 save: an adapter save gathers only trainable and
    still-saved frozen parameters, never the frozen base model (Accelerator.get_state_dict would), and a
    SHARDED_STATE_DICT plugin still exports (clearing the config made the FSDP branch skip it)."""
    from .models.fsdp2_save import fsdp2_gathered
    from .save import _fsdp2_peft_saved_frozen

    try:
        from peft import PeftModel
    except ImportError:
        PeftModel = None
    model = trainer.model
    adapter = PeftModel is not None and isinstance(getattr(model, "_orig_mod", model), PeftModel)
    include = _fsdp2_peft_saved_frozen(model, {}) if adapter else None
    with fsdp2_gathered(model, trainable_only = adapter, include = include) as writer:
        if writer is not False:
            return original_save_model(trainer, *args, **kwargs)
        # Rank 0 writes the gathered weights; the others still run save_model (a push_to_hub save is
        # collective again) but must not write their shards.
        missing = object()
        previous = trainer.__dict__.get("_save", missing)
        trainer._save = lambda *a, **k: None
        try:
            return original_save_model(trainer, *args, **kwargs)
        finally:
            if previous is missing:
                trainer.__dict__.pop("_save", None)
            else:
                trainer._save = previous


def _restore_parallelism_config(state) -> None:
    if _INSTALLED_PARALLELISM_CONFIG and (
        getattr(state, "parallelism_config", None) is _INSTALLED_PARALLELISM_CONFIG[1]
    ):
        state.parallelism_config = _INSTALLED_PARALLELISM_CONFIG[0]
        _INSTALLED_PARALLELISM_CONFIG.clear()


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
    _cp_scaler_start(self, accelerator)
    # PEFT makes the Trainer ask DDP for find_unused_parameters, which breaks reentrant checkpointing.
    if getattr(self.args, "ddp_find_unused_parameters", None) in (None, False):
        self.args.ddp_find_unused_parameters = False
        handler = getattr(accelerator, "ddp_handler", None)
        if handler is not None:
            handler.find_unused_parameters = False


def _cp_scaler_start(trainer, accelerator) -> None:
    """float16: start the GradScaler at 2**16 / world, as native FSDP2 and tensor parallel do.
    CP rank grads are world x one GPU's (loss is multiplied by world). Only touches a scaler still at the default."""
    scaler = getattr(accelerator, "scaler", None)
    if scaler is None or getattr(scaler, "_unsloth_cp_init_scale", False):
        return
    try:
        if not scaler.is_enabled() or getattr(scaler, "_scale", None) is not None:
            return
        if float(getattr(scaler, "_init_scale", 0.0)) != 2.0**16:
            return
        handler = getattr(accelerator, "scaler_handler", None)
        user = handler.to_kwargs() if handler is not None and hasattr(handler, "to_kwargs") else {}
        if "init_scale" in user:
            return
        from .models.fsdp2_native import _loss_scale_ranks

        world = _loss_scale_ranks(trainer)
    except Exception:
        return
    if world > 1:
        scaler._init_scale = 2.0**16 / world
    scaler._unsloth_cp_init_scale = True


def _model_parallel_axes(model) -> list:
    """Tensor / pipeline parallel axes in use: UNSLOTH_TP_SIZE or a model split by hand."""
    import os

    axes = []
    raw = str(os.environ.get("UNSLOTH_TP_SIZE", "")).strip()
    try:
        tp = int(raw) if raw else 1  # as tp_env._int_env, which raises its own error on a bad value
    except ValueError:
        tp = 1
    if tp > 1:
        axes.append(f"tensor parallel (UNSLOTH_TP_SIZE={raw})")
    if isinstance(model, torch.nn.Module):
        try:
            from .models.pp_native import is_pipeline_parallel
            from .models.tp_native import is_tensor_parallel
        except Exception:
            return axes
        if is_pipeline_parallel(model):
            axes.append("pipeline parallel (apply_pipeline_parallel)")
        if is_tensor_parallel(model) and not axes:
            axes.append("tensor parallel (apply_tensor_parallel)")
    return axes


def _refuse_model_parallel_cp(size: int, args, kwargs) -> None:
    """The CP mesh (world // cp, cp) would reuse a tp launch's groups and silently train on mixed-up shards."""
    model = kwargs["model"] if "model" in kwargs else (args[0] if args else None)
    axes = _model_parallel_axes(model)
    if not axes:
        return
    raise NotImplementedError(
        f"Unsloth: context parallelism (context_parallel_size = {size}) does not compose with "
        f"{' and '.join(axes)} yet: its context parallel groups would be the other axis' groups, and training "
        "would run with wrong gradients. Use context parallelism alone (with DDP or UNSLOTH_FSDP2=1), or "
        "tensor / pipeline parallelism with context_parallel_size = 1."
    )


def _check_cp_runtime(self, size: int, backend: str) -> None:
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


def _check_cp_model(
    model,
    size: int,
    backend: str,
    allow_hybrids: bool = True,
) -> None:
    model_config = getattr(model, "config", None)
    unsupported = _unsupported_attention(model_config, backend)
    if unsupported is not None:
        raise NotImplementedError(
            f"Unsloth: {backend} context parallelism does not support {unsupported}."
        )
    if backend == "ulysses":
        # Every rank raises here, before any collective.
        text_config = getattr(model_config, "text_config", None) or model_config
        ulysses_head_plan(
            text_config.num_attention_heads,
            getattr(text_config, "num_key_value_heads", None) or text_config.num_attention_heads,
            size,
        )
    if _stateful_mixers(model) and not allow_hybrids:
        raise NotImplementedError(
            "Unsloth: DPO / GRPO context parallelism does not support Gated DeltaNet hybrids yet."
        )
    _check_stateful_mixers(model, backend, size)
    blocker = _hybrid_config_blocker(model) if _stateful_mixers(model) else None
    if blocker is not None:
        raise NotImplementedError(
            f"Unsloth: context parallelism on Gated DeltaNet hybrids does not support {blocker}."
        )
    if not _supports_context_parallel(model, backend):
        raise NotImplementedError(
            "Unsloth: context parallelism currently supports Llama-style attention "
            "(Llama, Qwen2, Gemma), and Qwen3-Next / Qwen3.5 Gated DeltaNet hybrids with Ulysses."
        )


# Encoders and projectors a text-only batch never runs; context parallelism only takes text batches.
_MEDIA_TOWER_NAMES = frozenset(
    (
        "visual",
        "vision_tower",
        "vision_model",
        "vision_encoder",
        "multi_modal_projector",
        "mm_projector",
        "embed_vision",
        "audio_tower",
        "audio_model",
        "embed_audio",
    )
)


def _trainable_media_tower_parameters(model) -> list:
    if not isinstance(model, torch.nn.Module):
        return []
    return [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not _MEDIA_TOWER_NAMES.isdisjoint(name.split("."))
    ]


def _check_cp_trainer(
    self,
    size: int,
    backend: str,
    models,
    allow_fsdp2: bool = False,
) -> bool:
    """Shared trainer checks; returns whether the trainer runs under FSDP2."""
    import accelerate
    from packaging.version import Version

    if Version(accelerate.__version__) < Version("1.10.0"):
        raise NotImplementedError("Unsloth: context parallelism needs accelerate >= 1.10.0.")
    for model in models:
        if model is not None:
            # SFT shards the inputs of the whole model; the RL hooks only shard Llama-style decoders.
            _check_cp_model(model, size, backend, allow_hybrids = allow_fsdp2)
    accelerator = getattr(self, "accelerator", None)
    distributed_type = getattr(getattr(accelerator, "distributed_type", None), "name", "NO")
    fsdp2 = allow_fsdp2 and _is_fsdp2(accelerator, distributed_type)
    if distributed_type != "NO" and not distributed_type.startswith("MULTI_") and not fsdp2:
        if allow_fsdp2:
            raise NotImplementedError(
                f"Unsloth: context parallelism supports DDP and FSDP2 only, not {distributed_type}"
                + (" version 1." if distributed_type == "FSDP" else ".")
            )
        raise NotImplementedError(
            f"Unsloth: DPO / GRPO context parallelism supports DDP only, not {distributed_type}."
        )
    if fsdp2:
        _check_fsdp2_parallelism_config(accelerator)
    elif (
        distributed_type.startswith("MULTI_")
        and (getattr(accelerator, "num_processes", 2) or 2) > 1
    ):
        # Unused trainable towers break DDP (find_unused_parameters = False for reentrant checkpointing).
        for model in models:
            unused = _trainable_media_tower_parameters(model)
            if unused:
                raise NotImplementedError(
                    "Unsloth: context parallelism trains on text-only batches, so the trainable "
                    f"vision / audio tower ({len(unused)} parameters, e.g. {unused[0]}) never gets a "
                    "gradient and DDP fails at the second step. Freeze it (for LoRA, "
                    "finetune_vision_layers = False) or train with FSDP2."
                )
    if getattr(accelerator, "dispatch_batches", None):
        raise NotImplementedError("Unsloth: context parallelism does not support dispatch_batches.")
    _refuse_iterable_datasets(
        getattr(self, "train_dataset", None), getattr(self, "eval_dataset", None)
    )
    return fsdp2


def patch_sft_trainer(install_cp_methods: bool = False) -> None:
    # Only __init__ is wrapped up front; the other wrappers go on the class once a trainer enables
    # CP. With CP off they still slowed torch 2.10 LoRA steps by ~35% (gemma-4-E2B, L4 / A100).
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
    original_save_model = _original("save_model")
    has_total_train_batch_size = hasattr(trainer_cls, "get_total_train_batch_size")
    if has_total_train_batch_size:
        original_get_total_train_batch_size = _original("get_total_train_batch_size")

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        config = kwargs["args"] if "args" in kwargs else (args[1] if len(args) > 1 else None)
        # SFTTrainer(..., args = base_args, context_parallel_size = N) is moved into args only
        # inside the original __init__ (_backwards_compatible_trainer), so read it here too.
        # The forwarded value wins there, so it decides when given.
        requested = context_parallel_options(config)[0]
        if "context_parallel_size" in kwargs:
            # An explicit None opts out like 1; a non-integer fails validation later anyway.
            forwarded = kwargs["context_parallel_size"]
            valid = isinstance(forwarded, int) and not isinstance(forwarded, bool)
            requested = forwarded if valid else 1
        if requested > 1:
            _refuse_model_parallel_cp(requested, args, kwargs)
            # Before the original __init__: Unsloth wraps instance methods there (auto packing's
            # train reset, the token budget's training_step), and they must capture the CP ones.
            install_methods()
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = None
        # Unsloth's attention ignores transformers' own CP: it would stay local to each shard.
        # `accelerate launch` with parallelism_config_cp_size builds it from the environment
        # (ACCELERATE_USE_PARALLELISM_CONFIG), leaving args.parallelism_config None.
        for source, parallelism_config in (
            (
                "parallelism_config",
                getattr(getattr(self, "args", None), "parallelism_config", None),
            ),
            (
                "the accelerate config (parallelism_config_cp_size)",
                getattr(getattr(self, "accelerator", None), "parallelism_config", None),
            ),
        ):
            cp_size = getattr(parallelism_config, "cp_size", 1) or 1
            if cp_size > 1:
                raise NotImplementedError(
                    f"Unsloth: use SFTConfig(context_parallel_size = {cp_size}) instead of "
                    f"{source} with cp_size > 1."
                )
        size, backend, rotate_method = context_parallel_options(getattr(self, "args", None))
        if size <= 1:
            _install_accelerator_state(self)
            if not trainer_cls.__dict__.get("__unsloth_context_parallel_methods__"):
                # By identity: a subclass may define __eq__ without __hash__.
                pre_cp_trainers[:] = [ref for ref in pre_cp_trainers if ref() is not None]
                pre_cp_trainers.append(weakref.ref(self))
            return
        _check_cp_runtime(self, size, backend)
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
        fsdp2 = _check_cp_trainer(
            self, size, backend, [getattr(self, "model", None)], allow_fsdp2 = True
        )
        manager = ContextParallelManager(size, backend, rotate_method, fsdp2 = fsdp2)
        self._context_parallel_manager = manager
        install_methods()
        manager.attach_attention_hooks(self.model)
        _install_accelerator_state(self)
        # The ring calls the flash ops FA3 / FA4 override, so scope them to train() here too.
        from .utils.packing import _scope_flash_attention_impl

        _scope_flash_attention_impl(self.model, self)
        print(
            f"Unsloth: Context parallelism enabled with size = {size} ({backend}"
            + (f", rotate_method = {rotate_method})" if backend == "ring" else ")")
            + (
                f", FSDP2 sharding over dp_shard = {dist.get_world_size() // size} x cp = {size}."
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

    @functools.wraps(original_save_model)
    def patched_save_model(self, *args, **kwargs):
        state = getattr(getattr(self, "accelerator", None), "state", None)
        installed = _INSTALLED_PARALLELISM_CONFIG[1] if _INSTALLED_PARALLELISM_CONFIG else None
        if installed is None or getattr(state, "parallelism_config", None) is not installed:
            return original_save_model(self, *args, **kwargs)
        if _save_model_checks_parallelism_config_first() and _fsdp2_units_of(self):
            return _gathered_save_model(self, original_save_model, args, kwargs)
        # Otherwise (transformers >= 5 checks FSDP first, or nothing FSDP2-sharded) save without the
        # config; on transformers < 5 even a user dp_shard config would skip the FSDP branch.
        state.parallelism_config = None
        try:
            return original_save_model(self, *args, **kwargs)
        finally:
            state.parallelism_config = installed

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
            # Without gradient checkpointing the model would build a cache for the sharded scans.
            if hasattr(inputs, "setdefault"):
                inputs.setdefault("use_cache", False)
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

    # Non-CP trainers built before any trainer enabled CP: their __init__ may have wrapped train /
    # evaluate / predict on the instance around the plain methods (auto packing's train reset).
    pre_cp_trainers = []  # weakref.ref per trainer

    def _restoring(trainer, bound):
        @functools.wraps(bound)
        def call(*args, **kwargs):
            # What patched_train / evaluate / predict do for a non-CP trainer: drop the mesh and
            # parallelism config a CP trainer installed on the shared accelerator state.
            _install_accelerator_state(trainer)
            return bound(*args, **kwargs)

        return call

    def install_methods():
        if trainer_cls.__dict__.get("__unsloth_context_parallel_methods__"):
            return
        for trainer in [ref() for ref in pre_cp_trainers]:
            if trainer is None:
                continue
            for name in ("train", "evaluate", "predict"):
                bound = trainer.__dict__.get(name)
                if bound is not None:
                    trainer.__dict__[name] = _restoring(trainer, bound)
        pre_cp_trainers.clear()
        trainer_cls.prediction_step = patched_prediction_step
        trainer_cls.train = patched_train
        trainer_cls.evaluate = patched_evaluate
        trainer_cls.predict = patched_predict
        trainer_cls.evaluation_loop = patched_evaluation_loop
        trainer_cls.save_model = patched_save_model
        trainer_cls.training_step = patched_training_step
        if has_total_train_batch_size:
            trainer_cls.get_total_train_batch_size = patched_get_total_train_batch_size
        trainer_cls.__unsloth_context_parallel_methods__ = True

    trainer_cls.__init__ = patched_init
    trainer_cls.__unsloth_context_parallel__ = True
    if install_cp_methods:
        install_methods()


# Backends with single-tensor collectives; others (MPI) keep the list path.
_FUSED_GATHER_BACKENDS = ("nccl", "gloo", "xccl", "hccl", "ucc")


def _fused_gather_ok(group) -> bool:
    try:
        backend = str(dist.get_backend(group)).lower()
    except Exception:
        return False
    return "mpi" not in backend and any(name in backend for name in _FUSED_GATHER_BACKENDS)


# Newer torch deprecates *_tensor for *_single (same signatures).
_reduce_scatter = getattr(dist, "reduce_scatter_single", None) or getattr(
    dist, "reduce_scatter_tensor", None
)
_all_gather_into = getattr(dist, "all_gather_single", None) or getattr(
    dist, "all_gather_into_tensor", None
)


# DPO / GRPO: the loss reads per-token logps of whole sequences (sums over completions, group masks),
# so only the decoder is sharded; its last hidden states are gathered back to full length before lm_head.
class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, rank, size):
        ctx.group, ctx.rank, ctx.local = group, rank, x.shape[1]
        # Peers hold equal shards: the manager pads the sequence to a multiple of the group size.
        ctx.fused = (
            _reduce_scatter is not None and _all_gather_into is not None and _fused_gather_ok(group)
        )
        if not ctx.fused:
            x = x.contiguous()
            parts = [torch.empty_like(x) for _ in range(size)]
            dist.all_gather(parts, x, group = group)
            return torch.cat(parts, dim = 1)
        batch, local, rest = x.shape[0], x.shape[1], x.shape[2:]
        # Gather concatenates on dim 0, so sequence goes first; fresh output like torch.cat.
        if batch == 1:
            out = x.new_empty((batch, size * local, *rest))
            _all_gather_into(
                out.view(size * local, *rest), x.reshape(local, *rest).contiguous(), group = group
            )
            return out
        buffer = x.new_empty((size * local, batch, *rest))
        _all_gather_into(buffer, x.movedim(1, 0).contiguous(), group = group)
        # Out of place: an in-place copy into a returned fresh tensor zeroes compiled grads (torch 2.11).
        return buffer.movedim(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad):
        # Every peer computes the same loss on the gathered sequence: summing their gradients makes each
        # shard's parameter grads N x its share, and DDP's average over the group then yields one copy.
        if not ctx.fused:
            grad = grad.clone(memory_format = torch.contiguous_format)
            dist.all_reduce(grad, group = ctx.group)
            return grad.narrow(1, ctx.rank * ctx.local, ctx.local), None, None, None
        # Only this rank's slice of the sum is needed: a reduce-scatter sends half an all-reduce's bytes.
        batch, rest = grad.shape[0], grad.shape[2:]
        source = grad.movedim(1, 0).clone(memory_format = torch.contiguous_format)
        shard = grad.new_empty((ctx.local, batch, *rest))
        _reduce_scatter(shard, source, group = ctx.group)
        if batch == 1:
            return shard.view(1, ctx.local, *rest), None, None, None
        return shard.movedim(0, 1).contiguous(), None, None, None


def gather_sequence(x, manager):
    return _GatherSequence.apply(x, manager.group, manager.mesh.get_local_rank(), manager.size)


def _rl_decoder_pre_hook(module, args, kwargs):
    manager = get_cp_manager()
    stack = module._unsloth_cp_rl_stack
    if manager is None or manager is not module._unsloth_cp_rl_manager:
        stack.append(None)
        return None
    if args or kwargs.get("past_key_values") is not None or kwargs.get("input_ids") is None:
        raise NotImplementedError(
            "Unsloth: DPO / GRPO context parallelism shards keyword input_ids training forwards "
            "only (no positional inputs, inputs_embeds or KV cache)."
        )
    for key in ("prefix_seg_info", "output_attentions"):
        if kwargs.get(key):
            raise NotImplementedError(f"Unsloth: context parallelism does not support {key}.")
    batch = {
        name: kwargs.get(name)
        for name in ("input_ids", "attention_mask", "position_ids", "packed_seq_lengths")
    }
    seq_len = batch["input_ids"].shape[1]
    manager._prepare_inputs(batch)
    manager._ulysses_shard(batch)
    for name in ("input_ids", "position_ids"):
        if batch.get(name) is not None:
            # Fresh in the ambient grad / inference mode: a shard view keeps its base's inference flag.
            kwargs[name] = batch[name].clone()
    # Validated right padding needs no mask under Ulysses (attention drops it anyway). Passing the shard
    # would differ per peer (only the last shard holds padding) and recompile the compiled decoder
    # blocks on that rank alone.
    if "attention_mask" in kwargs:
        kwargs["attention_mask"] = None
    stack.append(seq_len)
    return args, kwargs


def _rl_decoder_post_hook(module, args, kwargs, output):
    seq_len = module._unsloth_cp_rl_stack.pop() if module._unsloth_cp_rl_stack else None
    if seq_len is None:
        return None
    manager = module._unsloth_cp_rl_manager
    last = output[0]

    def _full(x):
        # A fresh tensor, not a view of the padded gather: the compiled log-softmax would otherwise
        # guard on the view's base and recompile under CP only.
        x = gather_sequence(x, manager)
        return x if x.shape[1] == seq_len else x[:, :seq_len].clone()

    hidden = _full(last)
    layers = output.get("hidden_states") if not isinstance(output, tuple) else None
    if isinstance(output, tuple):
        if kwargs.get("output_hidden_states"):
            raise NotImplementedError(
                "Unsloth: context parallelism needs return_dict = True with output_hidden_states."
            )
        return (hidden, *output[1:])
    output[next(iter(output.keys()))] = hidden
    if layers is not None:
        # GRPO's hidden-state forward asks for these and reads the last one (the same tensor as above).
        output["hidden_states"] = tuple(
            hidden if layer is last else _full(layer) for layer in layers
        )
    return output


def _decoders(model):
    # The text decoder of each CausalLM (its embed_tokens + layers), under any PEFT / DDP wrapping.
    found = []
    for name, module in model.named_modules():
        if hasattr(module, "embed_tokens") and hasattr(module, "layers"):
            if not any(name.startswith(prefix + ".") for prefix, _ in found):
                found.append((name, module))
    return [module for _, module in found]


def attach_rl_hooks(manager, model) -> None:
    decoders = _decoders(model)
    if not decoders:
        raise NotImplementedError("Unsloth: context parallelism could not find the text decoder.")
    for decoder in decoders:
        decoder._unsloth_cp_rl_manager = manager
        decoder._unsloth_cp_rl_stack = []
        if not getattr(decoder, "_unsloth_cp_rl_hooked", False):
            decoder.register_forward_pre_hook(_rl_decoder_pre_hook, with_kwargs = True)
            decoder.register_forward_hook(_rl_decoder_post_hook, with_kwargs = True)
            decoder._unsloth_cp_rl_hooked = True
    manager.attach_attention_hooks(model)


def _activate(manager) -> None:
    global _ACTIVE_MANAGER
    _ACTIVE_MANAGER = manager


@contextlib.contextmanager
def _manager_scope(manager) -> Iterator[None]:
    previous = get_cp_manager()
    _activate(manager)
    try:
        yield
    finally:
        _activate(previous)


def _map_leaves(
    obj,
    fn,
    leaf = torch.Tensor,
):
    if isinstance(obj, leaf):
        return fn(obj)
    if isinstance(obj, (list, tuple)):
        mapped = [_map_leaves(x, fn, leaf) for x in obj]
        return type(obj)(*mapped) if hasattr(obj, "_fields") else type(obj)(mapped)
    if isinstance(obj, dict):
        return type(obj)((k, _map_leaves(v, fn, leaf)) for k, v in obj.items())
    return obj


class _OnDevice:
    def __init__(self, tensor):
        self.tensor, self.on_accelerator = tensor.cpu(), tensor.device.type != "cpu"
        self.inference = tensor.is_inference()

    def restore(self, device):
        # Peers get the root's inference-mode flag too: compiled forwards guard on it, so a mismatch
        # recompiles on every peer but the root.
        with torch.inference_mode(self.inference):
            tensor = self.tensor.to(device) if self.on_accelerator else self.tensor
            return tensor.clone() if self.inference and not tensor.is_inference() else tensor


def _broadcast_from_group_root(obj, manager, device):
    """CP rank 0 sends its object (tensors through CPU) so every peer scores the same samples."""
    payload = [_map_leaves(obj, _OnDevice)]
    group = manager.group
    dist.broadcast_object_list(payload, src = dist.get_global_rank(group, 0), group = group)
    return _map_leaves(
        payload[0],
        lambda x: x.restore(device),
        leaf = _OnDevice,
    )


def _check_same_prompts(prompts, manager) -> None:
    import hashlib
    import pickle

    digest = hashlib.sha256(pickle.dumps(prompts)).hexdigest()
    digests = [None] * manager.size
    dist.all_gather_object(digests, digest, group = manager.group)
    if len(set(digests)) != 1:
        raise RuntimeError(
            "Unsloth: context parallel peers received different prompts; the dataloader must shard "
            "by data parallel rank only."
        )


def _grpo_eval_groups_fit(args) -> bool:
    generations = (
        getattr(args, "num_generations_eval", None) or getattr(args, "num_generations", None) or 1
    )
    return args.per_device_eval_batch_size % generations == 0


def _check_rl_eval_entry(self, kind: str, dataset) -> None:
    # evaluate() / predict() may bring a dataset __init__ never saw.
    _refuse_iterable_datasets(dataset)
    if kind == "grpo_trainer" and dataset is not None and not _grpo_eval_groups_fit(self.args):
        raise NotImplementedError(
            "Unsloth: GRPO context parallelism needs per_device_eval_batch_size to be multiples of "
            "num_generations (num_generations_eval)."
        )


def _recount_completion_tokens(trainer, result):
    """TRL 0.24 - 1.8 return the DAPO token count from _generate (index 2 of 5 in 0.24 / 0.25, index 4 of
    7 or 9 in 0.26 - 1.8), gathered from every rank's own completions before CP peers adopted rank 0's:
    recount it from the adopted ones. TRL >= 1.9 returns 8 items and counts after _generate."""
    if not isinstance(result, tuple) or len(result) not in (5, 7, 9):
        return result
    index = 2 if len(result) == 5 else 4
    count = result[index]
    if not (isinstance(count, torch.Tensor) and count.ndim == 0):
        return result
    tool_mask = result[2] if len(result) > 5 else None
    if tool_mask is not None:  # TRL counts only model-generated tokens
        local = sum(sum(mask) for mask in tool_mask)
    else:
        local = sum(len(ids) for ids in result[1])
    # As TRL: summed over every process (DAPO later divides by num_processes, which cancels CP copies).
    total = trainer.accelerator.gather(
        torch.tensor([int(local)], device = count.device, dtype = count.dtype)
    ).sum()
    return (*result[:index], total, *result[index + 1 :])


def _refuse_transformers_cp(
    self,
    config_name: str,
    backend_hint: str = "",
) -> None:
    # Unsloth's attention ignores transformers / accelerate CP: it would stay local to each shard.
    # `accelerate launch` with parallelism_config_cp_size builds it from the environment
    # (ACCELERATE_USE_PARALLELISM_CONFIG), leaving args.parallelism_config None.
    for source, parallelism_config in (
        ("parallelism_config", getattr(getattr(self, "args", None), "parallelism_config", None)),
        (
            "the accelerate config (parallelism_config_cp_size)",
            getattr(getattr(self, "accelerator", None), "parallelism_config", None),
        ),
    ):
        cp_size = getattr(parallelism_config, "cp_size", 1) or 1
        if cp_size > 1:
            raise NotImplementedError(
                f"Unsloth: use {config_name}(context_parallel_size = {cp_size}{backend_hint}) "
                f"instead of {source} with cp_size > 1."
            )


def _check_rl_config(self, kind: str) -> None:
    args = self.args
    if kind == "dpo_trainer":
        for name in ("padding_free", "precompute_ref_log_probs", "generate_during_eval"):
            if getattr(args, name, False):
                raise NotImplementedError(
                    f"Unsloth: DPO context parallelism does not support {name} = True."
                )
        return
    if getattr(self, "use_vllm", False) or getattr(args, "use_vllm", False):
        raise NotImplementedError(
            "Unsloth: GRPO context parallelism does not support vLLM generation "
            "(fast_inference / use_vllm); generate with model.generate instead."
        )
    if getattr(args, "scale_rewards", None) == "batch" or (
        getattr(args, "multi_objective_aggregation", None) == "normalize_then_sum"
    ):
        # Batch-wide reward std over TRL's gather would count each peer's copy (Bessel bias).
        raise NotImplementedError(
            "Unsloth: GRPO context parallelism needs per-group reward scaling "
            "(scale_rewards = 'group' or 'none', multi_objective_aggregation = 'sum_then_normalize')."
        )
    if not hasattr(self, "_generate"):
        raise NotImplementedError("Unsloth: GRPO context parallelism needs TRL >= 0.24.")
    if getattr(getattr(self, "accelerator", None), "split_batches", False):
        # Each data parallel slice would get per_device_train_batch_size / D rows, part of a generation
        # group, and TRL's reward gather would then group peer copies together.
        raise NotImplementedError(
            "Unsloth: GRPO context parallelism does not support split_batches."
        )
    num_generations = getattr(args, "num_generations", None) or 1
    steps = getattr(args, "steps_per_generation", None) or args.gradient_accumulation_steps
    local = args.per_device_train_batch_size * steps
    if local % num_generations or (
        getattr(self, "eval_dataset", None) is not None and not _grpo_eval_groups_fit(args)
    ):
        # Peers duplicate their rank's slice in TRL's global reward gather; a group split across
        # ranks would mix copies into its mean and std.
        raise NotImplementedError(
            "Unsloth: GRPO context parallelism needs per_device_train_batch_size x "
            "steps_per_generation (and per_device_eval_batch_size) to be multiples of num_generations."
        )


def patch_rl_trainer(trainer_cls, kind: str) -> None:
    """Context parallelism for DPOTrainer / GRPOTrainer training forwards (Ulysses backend)."""
    if trainer_cls.__dict__.get("__unsloth_context_parallel__"):
        return
    original_init = trainer_cls.__init__

    def _original(name):
        if name in trainer_cls.__dict__:
            return trainer_cls.__dict__[name]
        if getattr(trainer_cls, name, None) is None:
            return None

        @functools.wraps(getattr(trainer_cls, name))
        def inherited(self, *args, **kwargs):
            return getattr(super(trainer_cls, self), name)(*args, **kwargs)

        return inherited

    originals = {
        name: _original(name)
        for name in (
            "training_step",
            "prediction_step",
            "train",
            "evaluate",
            "predict",
            "evaluation_loop",
            "compute_loss",
            "_generate",
            "_prepare_inputs",
            "_get_per_token_logps",
            "_get_per_token_logps_and_entropies",
            "_get_train_sampler",
        )
    }
    # GRPO generates (and scores rewards) inside training_step, so it switches CP on only for the
    # logprob forwards and from compute_loss until backward ends; DPO has no unsharded forward.
    activate_at_step = kind != "grpo_trainer"

    @functools.wraps(original_init)
    def patched_init(self, *args, **kwargs):
        config = kwargs.get("args")
        if config is None:
            config = next((x for x in args if hasattr(x, "context_parallel_size")), None)
        try:
            requested = context_parallel_options(config)[0]
        except ValueError:
            requested = 1  # Raised again after the original __init__.
        if requested > 1:
            _refuse_model_parallel_cp(requested, args, kwargs)
            # Before the original __init__, so instance wrappers made there capture the CP methods.
            install_methods()
        original_init(self, *args, **kwargs)
        self._context_parallel_manager = None
        _refuse_transformers_cp(
            self,
            "DPOConfig" if kind == "dpo_trainer" else "GRPOConfig",
            ", context_parallel_backend = 'ulysses'",
        )
        size, backend, _ = context_parallel_options(getattr(self, "args", None))
        if size <= 1:
            _install_accelerator_state(self)
            if not trainer_cls.__dict__.get("__unsloth_context_parallel_methods__"):
                # The generated trainers bind self.train in __init__ (rl.py), around the plain method.
                pre_cp_trainers[:] = [ref for ref in pre_cp_trainers if ref() is not None]
                pre_cp_trainers.append(weakref.ref(self))
            return
        if backend != "ulysses":
            raise NotImplementedError(
                "Unsloth: DPO / GRPO context parallelism needs context_parallel_backend = 'ulysses'."
            )
        _check_cp_runtime(self, size, backend)
        _check_rl_config(self, kind)
        ref_model = getattr(self, "ref_model", None)
        _check_cp_trainer(self, size, backend, [self.model, ref_model])
        manager = ContextParallelManager(size, backend)
        self._context_parallel_manager = manager
        attach_rl_hooks(manager, self.model)
        if kind == "grpo_trainer":
            # GRPO's packed and shared-prefix forwards pick their path per step from self-checks;
            # under CP every peer must run the same forwards, so keep the padded path.
            unwrap = getattr(getattr(self, "accelerator", None), "unwrap_model", None)
            for module in (self.model, unwrap(self.model) if unwrap else self.model):
                module._unsloth_seq_packing_grad_ok = False
                module._unsloth_seq_packing_nograd_ok = False
                module._unsloth_prefix_grouper_nograd_disabled = True
        if isinstance(ref_model, torch.nn.Module):
            attach_rl_hooks(manager, ref_model)
        _install_accelerator_state(self)
        print(f"Unsloth: Context parallelism enabled with size = {size} (ulysses, {kind}).")

    def _step_wrapper(name):
        original = originals[name]

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            manager = getattr(self, "_context_parallel_manager", None)
            if manager is None:
                return original(self, *args, **kwargs)
            for decoder in _decoders(self.model):
                decoder._unsloth_cp_rl_stack.clear()
            # Backward, and checkpoint recompute in it, run Ulysses attention too, so the scope spans them.
            with _manager_scope(manager if activate_at_step else None):
                return original(self, *args, **kwargs)

        return wrapped

    def _scoped(name, sticky = False):
        original = originals[name]

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            manager = getattr(self, "_context_parallel_manager", None)
            if manager is None:
                return original(self, *args, **kwargs)
            if sticky:
                # Left on: training_step's scope restores it after backward.
                _activate(manager)
                return original(self, *args, **kwargs)
            with _manager_scope(manager):
                return original(self, *args, **kwargs)

        return wrapped

    def patched_generate(self, prompts, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is None:
            return originals["_generate"](self, prompts, *args, **kwargs)
        _check_same_prompts(prompts, manager)
        with _manager_scope(None):
            result = originals["_generate"](self, prompts, *args, **kwargs)
        result = _broadcast_from_group_root(result, manager, self.accelerator.device)
        return _recount_completion_tokens(self, result)

    @functools.wraps(originals["_prepare_inputs"])
    def patched_prepare_inputs(self, *args, **kwargs):
        inputs = originals["_prepare_inputs"](self, *args, **kwargs)
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is None:
            return inputs
        # TRL shuffles the scored batch with the process RNG; peers must hold the same rows in the same order.
        return _broadcast_from_group_root(inputs, manager, self.accelerator.device)

    @functools.wraps(originals["_get_train_sampler"])
    def patched_train_sampler(self, *args, **kwargs):
        sampler = originals["_get_train_sampler"](self, *args, **kwargs)
        manager = getattr(self, "_context_parallel_manager", None)
        # TRL sizes a generation batch for every process; CP peers share one data parallel slice.
        if manager is not None and hasattr(sampler, "mini_repeat_count"):
            sampler.batch_size = sampler.batch_size // manager.size
        return sampler

    @functools.wraps(originals["train"])
    def patched_train(self, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            _refuse_iterable_datasets(getattr(self, "train_dataset", None))
        _install_accelerator_state(self)
        return originals["train"](self, *args, **kwargs)

    @functools.wraps(originals["evaluate"])
    def patched_evaluate(self, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            dataset = args[0] if args else kwargs.get("eval_dataset")
            if dataset is None or isinstance(dataset, str):
                current = getattr(self, "eval_dataset", None)
                dataset = (
                    current.get(dataset)
                    if isinstance(dataset, str) and isinstance(current, dict)
                    else current
                )
            _check_rl_eval_entry(self, kind, dataset)
        _install_accelerator_state(self)
        return originals["evaluate"](self, *args, **kwargs)

    def patched_predict(self, test_dataset, *args, **kwargs):
        if getattr(self, "_context_parallel_manager", None) is not None:
            _check_rl_eval_entry(self, kind, test_dataset)
        _install_accelerator_state(self)
        return originals["predict"](self, test_dataset, *args, **kwargs)

    @functools.wraps(originals["evaluation_loop"])
    def patched_evaluation_loop(self, *args, **kwargs):
        manager = getattr(self, "_context_parallel_manager", None)
        if manager is not None:
            self.gather_function = _cp_gather_for_metrics(self.accelerator, manager.size)
        return originals["evaluation_loop"](self, *args, **kwargs)

    # Non-CP trainers built before CP was enabled: like SFT, repair what their __init__ bound on the
    # instance, so they still drop the mesh a CP trainer installs on the shared AcceleratorState.
    pre_cp_trainers = []  # weakref.ref per trainer

    def _restoring(trainer, bound):
        @functools.wraps(bound)
        def call(*args, **kwargs):
            _install_accelerator_state(trainer)
            return bound(*args, **kwargs)

        return call

    def install_methods():
        # Only __init__ is wrapped up front (as for SFT): the wrappers below go on the class once a
        # trainer enables CP, so trainers without CP keep their original methods.
        if trainer_cls.__dict__.get("__unsloth_context_parallel_methods__"):
            return
        for trainer in [ref() for ref in pre_cp_trainers]:
            if trainer is None:
                continue
            for name in ("train", "evaluate", "predict"):
                bound = trainer.__dict__.get(name)
                if bound is not None:
                    trainer.__dict__[name] = _restoring(trainer, bound)
        pre_cp_trainers.clear()
        trainer_cls.training_step = _step_wrapper("training_step")
        trainer_cls.prediction_step = _step_wrapper("prediction_step")
        trainer_cls.train = patched_train
        trainer_cls.evaluate = patched_evaluate
        if originals["predict"] is not None:
            trainer_cls.predict = functools.wraps(originals["predict"])(patched_predict)
        trainer_cls.evaluation_loop = patched_evaluation_loop
        if kind == "grpo_trainer":
            trainer_cls.compute_loss = _scoped("compute_loss", sticky = True)
            for name in ("_get_per_token_logps", "_get_per_token_logps_and_entropies"):
                if originals[name] is not None:
                    setattr(trainer_cls, name, _scoped(name))
            if originals["_generate"] is not None:
                trainer_cls._generate = functools.wraps(originals["_generate"])(patched_generate)
            trainer_cls._prepare_inputs = patched_prepare_inputs
            if originals["_get_train_sampler"] is not None:
                trainer_cls._get_train_sampler = patched_train_sampler
        trainer_cls.__unsloth_context_parallel_methods__ = True

    trainer_cls.__init__ = patched_init
    trainer_cls.__unsloth_context_parallel__ = True
