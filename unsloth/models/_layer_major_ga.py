# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Layer-major gradient accumulation for offload_layers (block swap).

Standard gradient accumulation streams every swapped decoder layer from host RAM (or disk) twice per micro-batch, so
a step with gradient_accumulation_steps = N moves 2N copies of every layer. When the model has an active block swap,
Trainer.training_step runs the N micro-batches of the step together through unsloth_zoo's layer-major window instead
(each layer fetched once for the forward and once for the backward of all N), with each micro-batch still going
through the trainer's own training_step: same shapes, loss, num_items_in_batch normalisation and accumulation order,
so losses and gradients match standard accumulation bitwise.

Opt-in, off by default (_DEFAULT_MODE). UNSLOTH_OFFLOAD_LAYER_MAJOR: "1" runs a window whenever eligible, for any
decoder; "auto" only for the standard dense decoders in _AUTO_LAYERS (each proven bitwise against standard
accumulation), after two steps whose swap copies stalled the GPU for more than 5% of the swapped layers' time (always
on with UNSLOTH_OFFLOAD_STATS = 0, which removes the measurement); "0" or unset: never. Every refusal below applies to
both. Without an active block swap nothing here runs.

Data parallel (torch DDP, 2+ ranks): the window's micro-batches run on the module inside DDP, so DDP's reducer never
fires, and their gradients are averaged once after the window (_reduce_like_ddp: DDP's pre-scale by 1 / world size,
then a bucketed sum all-reduce). The ranks agree before every window (one MIN all-reduce of each rank's decision) and
inside it (unsloth_zoo's forward / commit votes), so either every rank runs a window or every rank runs the step the
standard way.
"""

import copy
import functools
import inspect
import os
import sys
import threading
from collections.abc import Mapping

import torch

__all__ = ["patch_layer_major_gradient_accumulation", "layer_major_ineligible"]

try:
    from unsloth_zoo.block_swap_layer_major import (
        boundary_bytes,
        layer_major_mode,
        run_layer_major,
        stall_fraction,
        swap_totals,
    )
except Exception:  # unsloth_zoo without layer-major windows
    run_layer_major = None

_TLS = threading.local()
# With UNSLOTH_OFFLOAD_LAYER_MAJOR unset: "off", so windows are opt-in (UNSLOTH_OFFLOAD_LAYER_MAJOR=1, or "auto" for
# the allowlisted decoders below after measured copy stalls).
_DEFAULT_MODE = "off"
# Packages whose module forward hooks are trusted to be deterministic and order-independent across micro-batches
# (input-grad hooks, block swap, gradient checkpointing owners, transformers' own hooks). A hook from anywhere else (a
# user's) could draw random numbers or keep state the window would reorder, so it keeps the step standard.
_TRUSTED_HOOK_PACKAGES = ("unsloth", "unsloth_zoo", "transformers", "peft", "accelerate", "torch")
_AUTO_STALL = 0.05
_AUTO_PROBE_STEPS = 2
_DROPOUT_KEYS = (
    "attention_dropout",
    "hidden_dropout",
    "dropout",
    "resid_pdrop",
    "attn_pdrop",
    "embd_pdrop",
)
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# auto engages only for these decoder layers (class name: the model_type it must come with), each covered by a
# bitwise test against standard accumulation (CPU and B200 BlockSwap RAM / disk, and the Unsloth end-to-end run).
# Any other decoder runs the standard way unless UNSLOTH_OFFLOAD_LAYER_MAJOR = 1.
_AUTO_LAYERS = {
    "LlamaDecoderLayer": ("llama",),
    "Qwen2DecoderLayer": ("qwen2",),
    "Qwen3DecoderLayer": ("qwen3",),
    "MistralDecoderLayer": ("mistral",),
    "Gemma2DecoderLayer": ("gemma2",),
    "Gemma3DecoderLayer": ("gemma3_text",),
}


def _mode():
    if not os.environ.get("UNSLOTH_OFFLOAD_LAYER_MAJOR", "").strip():
        return _DEFAULT_MODE
    return layer_major_mode()


def _say_once(trainer, key, message):
    said = trainer.__dict__.setdefault("_unsloth_layer_major_said", set())
    if key not in said:
        said.add(key)
        print(message)


def _block_swap_layers(model):
    """(decoder ModuleList, BlockSwap) of the model's active block swap, or (None, None)."""
    for module in model.modules():
        swapper = module.__dict__.get("_unsloth_block_swap")
        if swapper is not None and isinstance(module, torch.nn.ModuleList):
            return module, swapper
    return None, None


def _text_config(model):
    config = getattr(model, "config", None)
    get = getattr(config, "get_text_config", None)
    if callable(get):
        try:
            return get()
        except Exception:
            pass
    return config


def _audited_trainer_classes():
    """The exact trainer classes (identity) whose per-micro-batch methods are proven against standard accumulation:
    TRL's SFTTrainer chain (Unsloth rebinds its generated SFT trainer there) and Unsloth's own UnslothTrainer."""
    audited = set()
    for module, name in (
        ("trl.trainer.sft_trainer", "SFTTrainer"),
        ("trl", "SFTTrainer"),
        ("unsloth.trainer", "UnslothTrainer"),
    ):
        cls = sys.modules.get(module).__dict__.get(name) if module in sys.modules else None
        if isinstance(cls, type):
            audited.update(cls.__mro__)
    return audited


def _trainer_ok(trainer):
    """Positive check: every class of the trainer above transformers.Trainer is an audited one (identity), or
    overrides no Trainer attribute at all except __init__ (it may only add new methods)."""
    import transformers

    base = transformers.Trainer
    audited = _audited_trainer_classes()
    for cls in type(trainer).__mro__:
        if cls is base:
            return True
        if cls in audited:
            continue
        for name in cls.__dict__:
            if name.startswith("__") and name.endswith("__") and name != "__call__":
                continue
            if hasattr(base, name):
                return False
    return True


# Instance-level method wrappers known to run outside every training step, so the window cannot reorder them, as
# Unsloth's SFT trainer installs them: train (prepare_for_training_mode), _prepare_dataset (the one-shot packing
# guard, unsloth/trainer.py), _load_optimizer_and_scheduler (portable resume, unsloth/models/portable_resume.py) and
# get_train_dataloader (UNSLOTH_TOKEN_BALANCE, on by default under DDP: it only regroups rows, once per epoch).
# Any other instance-level override keeps the step standard.
_INSTANCE_OK = (
    "train",
    "_prepare_dataset",
    "_load_optimizer_and_scheduler",
    "get_train_dataloader",
)


def _instance_overrides(trainer):
    """Instance attributes shadowing a Trainer method (assigned wrappers), other than _INSTANCE_OK and read-only
    training_step observers (UNSLOTH_BATCH_REPORT)."""
    cls = type(trainer)
    for name, value in trainer.__dict__.items():
        attr = getattr(cls, name, None)
        if (
            not (callable(value) and callable(attr))
            or isinstance(attr, type)
            or name in _INSTANCE_OK
        ):
            continue
        if (
            name == "training_step"
            and getattr(value, "_unsloth_layer_major_observer", None) is value
        ):
            continue
        return name
    return None


def _operator_modes_active():
    """TorchFunctionMode / TorchDispatchMode on the training thread (thread-local: the workers would not see them)."""
    for probe in (
        getattr(torch._C, "_len_torch_function_stack", None),
        getattr(torch._C, "_len_torch_dispatch_stack", None),
        getattr(getattr(torch, "_ops", None), "_len_torch_dispatch_stack_pre_dispatch", None),
    ):
        try:
            if probe is not None and probe():
                return True
        except Exception:
            return True
    return False


def _caller_autocast():
    """Autocast entered around training_step (the trainer enters its own inside it)."""
    for device in ("cuda", "cpu"):
        try:
            if torch.is_autocast_enabled(device):
                return device
        except TypeError:
            if device == "cuda" and torch.is_autocast_enabled():
                return device
    return None


def _dropout_modules(model):
    """Dropout of any kind with p > 0 (the _DropoutNd family, or any class named like one): its draws would be
    taken in another order across micro-batches."""
    dropout_nd = getattr(torch.nn.modules.dropout, "_DropoutNd", torch.nn.Dropout)
    for name, module in model.named_modules():
        if isinstance(module, dropout_nd) or "dropout" in type(module).__name__.lower():
            p = getattr(module, "p", None)
            if not isinstance(p, (int, float)) or p > 0:
                return f"{type(module).__name__}(p = {p}) in {name or 'the model'}"
    return None


def _compiled_in_layers(layers):
    """torch.compile on a decoder layer or inside one, other than Unsloth's block compile (proven): module.compile()
    takes the layer's checkpoint wrapper into the graph, and the window's extra no-grad specialisations can use up
    the recompile budget the standard retry needs."""
    optimized = getattr(getattr(torch, "_dynamo", None), "OptimizedModule", None)
    for module in layers.modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            return True
        if optimized is not None and isinstance(module, optimized):
            return True
        if hasattr(getattr(module, "forward", None), "_torchdynamo_orig_callable"):
            return True
    return False


def _compiled(module):
    optimized = getattr(getattr(torch, "_dynamo", None), "OptimizedModule", None)
    # Through attribute lookup: a forward compiled on the instance or on the class.
    forward = getattr(module, "forward", None)
    return (
        (optimized is not None and isinstance(module, optimized))
        or getattr(module, "_compiled_call_impl", None) is not None
        or hasattr(forward, "_torchdynamo_orig_callable")
    )


def _compiled_around(model, layers):
    """torch.compile on the model or a module holding the decoder layers (model.compile(), torch.compile(model)):
    the window's layer interceptor would run inside the compiled graph."""
    if _compiled(model):
        return True
    for name, module in model.named_modules():
        if module is layers:
            parts = name.split(".") if name else []
            for k in range(len(parts)):
                try:
                    if _compiled(model.get_submodule(".".join(parts[:k]))):
                        return True
                except AttributeError:
                    return True
            return False
    return False


def _auto_layers_ok(model, layers):
    """(ok, class name): whether every decoder layer is one auto engages for."""
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    for cls in {type(layer) for layer in layers}:
        allowed = _AUTO_LAYERS.get(cls.__name__)
        module = getattr(cls, "__module__", "") or ""
        if (
            allowed is None
            or model_type not in allowed
            or not (
                module.startswith("transformers.models.") or "unsloth_compiled_module" in module
            )
        ):
            return False, cls.__name__
    return True, None


def _hook_package(fn):
    while isinstance(fn, functools.partial):
        fn = fn.func
    fn = getattr(fn, "__func__", fn)
    module = getattr(fn, "__module__", None) or getattr(type(fn), "__module__", "") or ""
    return module.split(".")[0]


def _untrusted_forward_hooks(model):
    """A module forward (pre-)hook, anywhere in the model, whose function does not come from a trusted package."""
    from torch.nn.modules import module as nn_module

    if getattr(nn_module, "_global_forward_pre_hooks", None) or getattr(
        nn_module, "_global_forward_hooks", None
    ):
        return "global module forward hooks"
    for name, module in model.named_modules():
        for kind in ("_forward_pre_hooks", "_forward_hooks"):
            for hook in getattr(module, kind, {}).values():
                if _hook_package(hook) not in _TRUSTED_HOOK_PACKAGES:
                    return f"a forward hook on {name or 'the model'} ({_hook_package(hook) or 'unknown'})"
    return None


def _optimizer_train_hook(trainer):
    """An optimizer with a train() method: Trainer.training_step calls it before every micro-batch's forward
    (schedule-free optimizers), which under a window runs before the previous micro-batch's decoder gradients."""
    optimizer = getattr(trainer, "optimizer", None)
    for _ in range(4):
        if optimizer is None:
            return False
        if type(optimizer).__name__ == "AcceleratedOptimizer" and "train" not in vars(optimizer):
            optimizer = getattr(optimizer, "optimizer", None)
            continue
        return callable(getattr(optimizer, "train", None))
    return True


def _window_blocker(
    trainer,
    model,
    layers = None,
):
    """Per step (they can appear after the first): what makes this step's micro-batches run the standard way."""
    control = getattr(trainer, "control", None)
    if control is not None and (
        getattr(control, "should_epoch_stop", False)
        or getattr(control, "should_training_stop", False)
    ):
        # The loop leaves after this micro-batch: a window would run the step's other micro-batches too.
        return "the trainer was asked to stop the epoch or training"
    if _optimizer_train_hook(trainer):
        return "an optimizer with a per-micro-batch train() hook (schedule-free)"
    found = _untrusted_forward_hooks(model)
    if found is not None:
        return found
    if _operator_modes_active():
        return "a TorchFunctionMode / TorchDispatchMode is active"
    device = _caller_autocast()
    if device is not None:
        return f"{device} autocast entered around training_step"
    if getattr(trainer, "compute_loss_func", None) is not None:
        # Called inside each micro-batch's loss part, which under a window runs with the step's first
        # micro-batch's accelerator sync state (and so on): standard accumulation keeps it exact.
        return "a custom compute_loss_func"
    if any(
        p.requires_grad and getattr(p, "_post_accumulate_grad_hooks", None)
        for p in model.parameters()
    ):
        return "post-accumulate-grad hooks (an optimizer step inside backward)"
    if any(p.requires_grad and getattr(p, "_backward_hooks", None) for p in model.parameters()):
        # They run inside backward, where they would see the window's accumulation state, not their micro-batch's.
        return "gradient hooks on trainable parameters"
    from torch.nn.modules import module as nn_module

    if (
        getattr(nn_module, "_global_backward_hooks", None)
        or getattr(nn_module, "_global_backward_pre_hooks", None)
        or any(m._backward_hooks or m._backward_pre_hooks for m in model.modules())
    ):
        return "module backward hooks"
    if layers is not None and _compiled_around(model, layers):
        return "torch.compile over the whole model"
    if layers is not None and _compiled_in_layers(layers):
        return "torch.compile on a decoder layer (other than Unsloth's block compile)"
    # The window bypasses instance-level wrappers for the micro-batches after the first; only read-only observers
    # (UNSLOTH_BATCH_REPORT's training_step) are allowed.
    name = _instance_overrides(trainer)
    if name is not None:
        return f"an instance-level {name} wrapper"
    handler = getattr(trainer, "callback_handler", None)
    try:
        from transformers import TrainerCallback
        base = TrainerCallback.on_substep_end
    except Exception:
        base = None
    for callback in getattr(handler, "callbacks", ()):
        # The method Transformers dispatches: on the instance (an assigned handler), else the class's.
        fn = getattr(callback, "on_substep_end", base)
        if base is not None and getattr(fn, "__func__", fn) is not base:
            # Called between micro-batches: under a window it would see every micro-batch's gradients at once.
            return f"{type(callback).__name__}.on_substep_end"
    return None


def _world_size():
    dist = torch.distributed
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def _ddp_of(model):
    """The torch DDP wrapper training_step got, or None."""
    ddp = getattr(torch.nn.parallel, "DistributedDataParallel", None)
    return model if ddp is not None and isinstance(model, ddp) else None


def _engine_votes():
    try:
        return "agree" in inspect.signature(run_layer_major).parameters
    except (TypeError, ValueError):
        return False


def _ddp_ineligible(trainer, ddp, layers):
    """Why the DDP-wrapped model cannot run windows (the manual reduce must equal DDP's own), or None."""
    from ._ddp_defaults import _plain_ddp

    if not _engine_votes():
        return "unsloth_zoo is too old for layer-major windows under DDP"
    accelerator = getattr(trainer, "accelerator", None)
    if accelerator is not None and not _plain_ddp(accelerator):
        return f"accelerate distributed type {getattr(accelerator, 'distributed_type', None)}"
    if type(ddp) is not torch.nn.parallel.DistributedDataParallel or "forward" in ddp.__dict__:
        return "a DDP subclass or a replaced DDP forward"
    if (
        ddp._forward_hooks
        or ddp._forward_pre_hooks
        or ddp._backward_hooks
        or ddp._backward_pre_hooks
    ):
        # The window runs the module inside DDP, so hooks on the wrapper itself would be skipped.
        return "hooks on the DDP wrapper"
    dist = torch.distributed
    if dist.get_world_size(ddp.process_group) != dist.get_world_size():
        return "DDP over a subgroup (other parallel dimensions)"
    if getattr(ddp, "require_backward_grad_sync", True):
        # Trainer runs a step's first micro-batch under no_sync; without it the window cannot defer the reduce.
        return "the step's first micro-batch runs outside DDP no_sync"
    for attr, why in (
        ("find_unused_parameters", "DDP find_unused_parameters = True"),
        ("static_graph", "DDP static_graph"),
        ("_comm_hooks", "a DDP communication hook"),
        ("_delay_all_reduce_all_params", "DDP delayed all-reduce"),
        ("_use_python_reducer", "DDP's Python reducer (compiled autograd)"),
    ):
        if getattr(ddp, attr, None):
            return why
    if getattr(ddp, "mixed_precision", None) is not None:
        return "DDP mixed precision"
    if getattr(getattr(ddp, "_join_config", None), "enable", False):
        return "DDP join"
    if (
        hasattr(ddp, "buffer_hook")
        or not hasattr(ddp, "_sync_buffers")
        or not hasattr(ddp, "_check_sync_bufs_pre_fwd")
    ):
        return "a DDP buffer hook, or a torch DDP without the buffer sync this needs"
    if getattr(ddp, "broadcast_buffers", False) and (
        sum(b.numel() * b.element_size() for b in ddp.module.buffers()) > _BUFFER_SNAPSHOT_MAX
    ):
        return f"over {_BUFFER_SNAPSHOT_MAX >> 20} MiB of DDP-broadcast buffers to snapshot"
    if getattr(ddp.module, "_unsloth_ddp_tower_params", None):
        return "vision / audio towers averaged outside DDP"
    return None


def layer_major_ineligible(
    trainer,
    model,
    layers,
    swapper,
    ddp = None,
):
    """Why a layer-major window cannot run for this trainer and model, or None. `model` is the module inside DDP
    when `ddp` (the wrapper training_step got) is given."""
    if run_layer_major is None or not hasattr(swapper, "hold"):
        return "unsloth_zoo is too old for layer-major windows"
    if not getattr(swapper, "indices", None):
        return "no layer is swapped"
    if getattr(swapper, "total", len(layers)) != len(layers):
        return "the block swap does not cover the decoder layer list"
    if _world_size() > 1:
        if ddp is None:
            return "distributed training other than DDP"
        why = _ddp_ineligible(trainer, ddp, layers)
        if why is not None:
            return why
    else:
        accelerator = getattr(trainer, "accelerator", None)
        kind = str(getattr(accelerator, "distributed_type", "NO"))
        if not kind.endswith("NO"):
            return f"accelerate distributed type {kind}"
    args = getattr(trainer, "args", None)
    if getattr(args, "n_gpu", 1) > 1:
        return "DataParallel over several GPUs"
    if getattr(args, "activation_offloading", False):
        return "activation_offloading"
    if getattr(args, "torch_compile", False):
        return "torch_compile over the whole model"
    optim = str(getattr(args, "optim", "")).lower()
    if "lomo" in optim or "layerwise" in optim:
        return "this optimizer updates the weights inside every backward"
    if "_unsloth_token_budget_splits" in trainer.__dict__:
        # Its instance-level training_step splits micro-batches into chunks the window never sees.
        return "UNSLOTH_TOKEN_BUDGET"
    if not _trainer_ok(trainer):
        return f"{type(trainer).__name__} overrides training_step or compute_loss"
    config = _text_config(model)
    for key in _MOE_KEYS:
        value = getattr(config, key, 0)
        if isinstance(value, int) and value > 0:
            return "a mixture-of-experts model (router losses)"
    for key in _DROPOUT_KEYS:
        value = getattr(config, key, 0)
        if isinstance(value, (int, float)) and value > 0:
            return f"{key} = {value} (dropout draws differ by micro-batch order)"
    # Anywhere in the model (lora_dropout on the head too): the window reorders the draws across micro-batches.
    found = _dropout_modules(model)
    if found is not None:
        return f"dropout {found}"
    if getattr(args, "neftune_noise_alpha", None):
        return "NEFTune embedding noise (random draws)"
    return None


def _tokens(sample):
    for key in ("input_ids", "inputs_embeds", "labels", "attention_mask"):
        value = sample.get(key) if isinstance(sample, Mapping) else None
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            return int(value.shape[0]) * int(value.shape[1])
    return None


def _usable_cuda_bytes(device):
    from ._utils import usable_cuda_bytes
    return usable_cuda_bytes(device)


def _fits(model, layers, samples):
    """Whether every micro-batch's layer inputs fit on the card at once (standard checkpointing holds one)."""
    tokens = [_tokens(s) for s in samples]
    if any(t is None for t in tokens):
        return False, "unknown micro-batch shapes"
    config = _text_config(model)
    hidden = getattr(config, "hidden_size", None)
    embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    weight = getattr(embeddings, "weight", None)
    if hidden is None or weight is None:
        return False, "unknown hidden size"
    element = (
        torch.empty((), dtype = weight.dtype).element_size() if weight.dtype.is_floating_point else 2
    )
    need = boundary_bytes(tokens, hidden, len(layers), element)
    if not torch.cuda.is_available():
        return True, need
    devices = {p.device for p in layers.parameters() if p.is_cuda}
    if weight.device.type == "cuda":
        devices.add(weight.device)
    if not devices:
        devices.add(torch.device("cuda", torch.cuda.current_device()))
    # Allocator cap (set_per_process_memory_fraction) included; with layers split over cards, the tightest one.
    free = min(_usable_cuda_bytes(device) for device in devices)
    if need > 0.6 * free:
        return (
            False,
            f"{need / 2**30:.2f} GiB of layer inputs for {len(samples)} micro-batches vs {free / 2**30:.2f} GiB free",
        )
    return True, need


def _auto_ready(trainer, swapper):
    """auto: two standard steps, then on only if the second one's swap copies stalled the GPU."""
    state = trainer.__dict__.setdefault(
        "_unsloth_layer_major_auto", {"steps": 0, "since": None, "on": None}
    )
    if state["on"] is not None:
        return state["on"]
    if not getattr(sys.modules.get(type(swapper).__module__), "_TIMING", True):
        state["on"] = True  # UNSLOTH_OFFLOAD_STATS = 0: nothing to measure with
        return True
    state["steps"] += 1
    if state["steps"] == _AUTO_PROBE_STEPS:
        state["since"] = swap_totals(swapper)
    elif state["steps"] > _AUTO_PROBE_STEPS:
        frac = (
            stall_fraction(state["since"], swap_totals(swapper))
            if state["since"] is not None
            else None
        )
        state["on"] = frac is not None and frac > _AUTO_STALL
        if not state["on"]:
            _say_once(
                trainer,
                "hidden",
                "Unsloth: offload_layers copies already hide behind compute "
                f"({100 * (frac or 0):.1f}% stall), so micro-batches stay one after another.",
            )
        return state["on"]
    return False


def _micro_step(trainer, model, inputs, num_items_in_batch, args, kwargs):
    _TLS.passthrough = True
    try:
        # The class chain (SFTTrainer and the Trainer patches), not instance-level wrappers such as
        # UNSLOTH_BATCH_REPORT's: those already saw this micro-batch's outer call and see the others' too.
        return type(trainer).training_step(
            trainer, model, inputs, num_items_in_batch, *args, **kwargs
        )
    finally:
        _TLS.passthrough = False


def _all_ranks_agree(flag, group, device):
    """True only if every rank of `group` passes True (one MIN all-reduce)."""
    value = torch.tensor([int(bool(flag))], dtype = torch.int32, device = device)
    torch.distributed.all_reduce(value, op = torch.distributed.ReduceOp.MIN, group = group)
    return bool(value.item())


def _collective_device(group):
    backend = str(torch.distributed.get_backend(group)).lower()
    if "gloo" in backend and "nccl" not in backend:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


_DDP_REDUCES = [0]
_OTHER_RANK = (
    "Unsloth: offload_layers runs micro-batches one after another this step: another rank cannot run a "
    "layer-major window."
)


def _reduce_like_ddp(ddp):
    """Average the gradients DDP owns as its reducer does: each gradient times 1 / world size, then a sum
    all-reduce, in flat buckets of at most DDP's bucket size per dtype. A weight no rank gave a gradient keeps None
    (one used anywhere is reduced, with zeros from the ranks that did not use it)."""
    dist = torch.distributed
    group = ddp.process_group
    ignore = set(getattr(ddp, "parameters_to_ignore", ()) or ())
    params = [p for n, p in ddp.module.named_parameters() if p.requires_grad and n not in ignore]
    if not params:
        return
    world = dist.get_world_size(group)
    device = _collective_device(group)
    used = torch.tensor([p.grad is not None for p in params], dtype = torch.int32, device = device)
    dist.all_reduce(used, group = group)
    cap = int(getattr(ddp, "bucket_bytes_cap", 25 * 2**20) or 25 * 2**20)
    buckets, size = [[]], 0
    for p, u in zip(params, used.tolist()):
        if not u:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        g = p.grad
        if buckets[-1] and (
            size + g.numel() * g.element_size() > cap
            or buckets[-1][-1].dtype != g.dtype
            or buckets[-1][-1].device != g.device
        ):
            buckets.append([])
            size = 0
        buckets[-1].append(g)
        size += g.numel() * g.element_size()
    for grads in buckets:
        if not grads:
            continue
        flat = torch._utils._flatten_dense_tensors(grads)
        flat.mul_(
            1.0 / world
        )  # DDP multiplies by 1 / world size while copying into its bucket, then sums
        dist.all_reduce(flat, group = group)
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)
        del flat
    _DDP_REDUCES[0] += 1


# Most DDP-broadcast buffer bytes a window snapshots (rotary tables and the like are KBs).
_BUFFER_SNAPSHOT_MAX = 256 * 2**20


def _ddp_buffers(ddp, layers):
    """The buffers DDP broadcasts before a forward: (buffer, version, copy) (block swap moves weights, not buffers).
    The copy puts back what an abandoned window's forwards wrote before the standard retry."""
    if not ddp._check_sync_bufs_pre_fwd():
        return []
    return [(b, b._version, b.detach().clone()) for b in ddp.module.buffers()]


def _restore_buffers(buffers):
    with torch.no_grad():
        for b, version, saved in buffers:
            if b._version != version:
                b.copy_(saved)


def _plan(trainer, model, ddp, samples, mode):
    """(layers, swapper, need) for a window over `samples`, or (None, message key, message)."""
    if trainer.__dict__.get("_unsloth_layer_major_off"):
        return None, None, None
    cached = trainer.__dict__.get("_unsloth_layer_major_target")
    # Re-planned swaps (offload_layers = "auto") install a new BlockSwap on the same layer list.
    if (
        cached is not None
        and cached[0] == id(model)
        and (cached[1] is None or cached[1].__dict__.get("_unsloth_block_swap") is cached[2])
    ):
        _, layers, swapper = cached
    else:
        layers, swapper = _block_swap_layers(model)
        trainer._unsloth_layer_major_target = (id(model), layers, swapper)
    if swapper is None:
        return None, None, None
    why = layer_major_ineligible(trainer, model, layers, swapper, ddp = ddp)
    if why is not None:
        return None, "why", f"Unsloth: offload_layers runs micro-batches one after another: {why}."
    if mode == "auto":
        ok, name = _auto_layers_ok(model, layers)
        if not ok:
            return (
                None,
                "opt-in",
                f"Unsloth: offload_layers layer-major gradient accumulation is opt-in for {name}: set "
                "UNSLOTH_OFFLOAD_LAYER_MAJOR=1 to run its micro-batches layer by layer.",
            )
    blocker = _window_blocker(trainer, model, layers)
    if blocker is not None:
        return (
            None,
            "blocker",
            f"Unsloth: offload_layers runs micro-batches one after another: {blocker}.",
        )
    if mode == "auto" and not _auto_ready(trainer, swapper):
        return None, None, None
    ok, need = _fits(model, layers, samples)
    if not ok:
        return (
            None,
            "fit",
            f"Unsloth: offload_layers runs micro-batches one after another this step: {need}.",
        )
    return layers, swapper, need


def _layer_major_step(trainer, model, inputs, num_items_in_batch, args, kwargs):
    """The loss of micro-batch `inputs` if a layer-major window ran (or runs now) for it, else None."""
    window = trainer.__dict__.get("_unsloth_layer_major_window")
    if window is not None:
        samples, losses, k = window
        if k < len(samples) and samples[k] is inputs:
            trainer._unsloth_layer_major_window = (
                (samples, losses, k + 1) if k + 1 < len(samples) else None
            )
            return losses[k]
        # get_batch_samples clears a window, so this is still its step, but the trainer passed something other
        # than the micro-batch the window already accumulated: running it would count that data twice.
        trainer._unsloth_layer_major_window = None
        raise RuntimeError(
            "Unsloth: offload_layers layer-major gradient accumulation already ran micro-batch "
            f"{k} of this step, but training_step got different inputs for it. "
            "Set UNSLOTH_OFFLOAD_LAYER_MAJOR=0 to turn it off."
        )
    # Consumed by the step's first call: the trainer keeps no prefetched batch past it (the window holds its own).
    batch = trainer.__dict__.pop("_unsloth_layer_major_batch", None)
    if batch is None:
        return None
    samples, items = batch
    if len(samples) < 2 or samples[0] is not inputs or items is not num_items_in_batch:
        return None
    mode = _mode() if run_layer_major is not None else "off"
    # After an evaluation the model is still in eval mode here; the trainer's training_step switches it back.
    if mode == "off" or (
        not getattr(model, "training", True) and not getattr(trainer, "is_in_train", False)
    ):
        return None
    # Above: identical on every rank. Below: this rank's decision, then agreed on.
    ddp = _ddp_of(model) if _world_size() > 1 else None
    if _world_size() > 1 and ddp is None:
        _say_once(
            trainer,
            "why",
            "Unsloth: offload_layers runs micro-batches one after another: distributed training other than DDP.",
        )
        return None
    inner = ddp.module if ddp is not None else model
    plan = _plan(trainer, inner, ddp, samples, mode)
    if plan[0] is None and plan[1] is not None:
        _say_once(trainer, plan[1], plan[2])
    layers, swapper, need = plan
    agree = None
    if ddp is not None:
        group, device = ddp.process_group, _collective_device(ddp.process_group)
        if not _all_ranks_agree(layers is not None, group, device):
            if layers is not None:
                _say_once(trainer, "rank", _OTHER_RANK)
            return None
        # DDP broadcasts these before a step's first forward; the window's forwards bypass DDP, so do it here.
        buffers = _ddp_buffers(ddp, layers)
        if buffers:
            ddp._sync_buffers()
            buffers = _ddp_buffers(ddp, layers)

        def agree(ok, stage):
            if stage == "commit" and ok:
                # A changed buffer (BatchNorm, dynamic RoPE) would be re-broadcast before the next micro-batch under DDP.
                ok = all(b._version == v for b, v, _ in buffers)
            return _all_ranks_agree(ok, group, device)

    else:
        buffers = []
        if layers is None:
            return None
    fns = [
        functools.partial(_micro_step, trainer, inner, s, num_items_in_batch, args, kwargs)
        for s in samples
    ]
    # The loss parts update TRL's metrics and token count; an abort retries the micro-batches the standard way.
    counters = {
        k: copy.deepcopy(trainer.__dict__[k])
        for k in ("_metrics", "_total_train_tokens")
        if k in trainer.__dict__
    }
    try:
        vote = {"agree": agree} if agree is not None else {}
        losses, why = run_layer_major(layers, fns, swapper = swapper, model = inner, **vote)
    except BaseException:
        trainer.__dict__.update(counters)  # the engine already put back gradients and RNG
        _restore_buffers(buffers)
        raise
    if losses is None:
        trainer.__dict__.update(counters)
        _restore_buffers(buffers)
        if why.startswith("another layer-major window"):
            return None  # transient: another trainer in this process holds the window
        _say_once(
            trainer,
            "abort",
            f"Unsloth: offload_layers layer-major gradient accumulation stopped ({why}); "
            "micro-batches run one after another from here on.",
        )
        trainer._unsloth_layer_major_off = True
        return None
    if ddp is not None:
        # The step's only gradient reduction: the window ran inside no_sync and outside DDP's forward.
        _reduce_like_ddp(ddp)
    _say_once(
        trainer,
        "on",
        f"Unsloth: offload_layers runs the {len(samples)} micro-batches of each step layer by layer, so each "
        f"swapped layer is fetched once per step instead of {len(samples)} times "
        f"({need / 2**30:.2f} GiB of layer inputs held). UNSLOTH_OFFLOAD_LAYER_MAJOR = 0 turns it off.",
    )
    trainer._unsloth_layer_major_window = (samples, losses, 1)
    return losses[0]


def patch_layer_major_gradient_accumulation(Trainer):
    """Wrap Trainer.get_batch_samples (to see a step's micro-batches) and Trainer.training_step."""
    if getattr(Trainer.training_step, "_unsloth_layer_major", False):
        return
    if not hasattr(Trainer, "get_batch_samples") or run_layer_major is None:
        return
    get_batch_samples = Trainer.get_batch_samples
    training_step = Trainer.training_step
    train = Trainer.train

    def _clear_state(trainer):
        for name in [k for k in trainer.__dict__ if k.startswith("_unsloth_layer_major_")]:
            trainer.__dict__.pop(name, None)

    @functools.wraps(train)
    def _train(self, *args, **kwargs):
        # Per train() call: an abort's switch-off, the auto probe and the pending window do not outlive it.
        _clear_state(self)
        try:
            return train(self, *args, **kwargs)
        finally:
            _clear_state(self)

    @functools.wraps(get_batch_samples)
    def _get_batch_samples(self, *args, **kwargs):
        out = get_batch_samples(self, *args, **kwargs)
        self.__dict__.pop("_unsloth_layer_major_window", None)
        self.__dict__.pop("_unsloth_layer_major_batch", None)
        try:
            samples, items = out
        except (TypeError, ValueError):
            return out
        # Only for trainers whose training_step reaches the wrapper below, which consumes it each step.
        if isinstance(samples, list) and len(samples) > 1 and _mode() != "off":
            ok = _trainer_ok(self) and _instance_overrides(self) is None
            if _world_size() > 1:
                # Every rank stashes or none does: a rank-local wrapper (a rank 0 observer) must not leave the
                # other ranks waiting in the step's window agreement.
                agreed = _all_ranks_agree(ok, None, _collective_device(None))
                if ok and not agreed:
                    _say_once(self, "rank", _OTHER_RANK)
                ok = agreed
            if ok:
                self._unsloth_layer_major_batch = (samples, items)
        elif (
            isinstance(samples, list)
            and len(samples) > 1
            and _mode() == "off"
            and not os.environ.get("UNSLOTH_OFFLOAD_LAYER_MAJOR", "").strip()
            and "opt-in hint" not in self.__dict__.get("_unsloth_layer_major_said", ())
            and _block_swap_layers(getattr(self, "model", None) or torch.nn.Module())[1] is not None
        ):
            _say_once(
                self,
                "opt-in hint",
                "Unsloth: offload_layers with gradient accumulation can run each step's micro-batches layer by "
                "layer, fetching every swapped layer once per step instead of once per micro-batch. It is opt-in: "
                "set UNSLOTH_OFFLOAD_LAYER_MAJOR=1.",
            )
        return out

    @functools.wraps(training_step)
    def _training_step(
        self,
        model,
        inputs,
        num_items_in_batch = None,
        *args,
        **kwargs,
    ):
        if not getattr(_TLS, "passthrough", False):
            loss = _layer_major_step(self, model, inputs, num_items_in_batch, args, kwargs)
            if loss is not None:
                return loss
        return training_step(self, model, inputs, num_items_in_batch, *args, **kwargs)

    _get_batch_samples._unsloth_layer_major = True
    _training_step._unsloth_layer_major = True
    _train._unsloth_layer_major = True
    Trainer.get_batch_samples = _get_batch_samples
    Trainer.training_step = _training_step
    if not getattr(train, "_unsloth_layer_major", False):
        Trainer.train = _train
