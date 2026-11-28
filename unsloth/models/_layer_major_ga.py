# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Layer-major gradient accumulation for offload_layers (block swap).

Standard gradient accumulation streams every swapped decoder layer from host RAM (or disk) twice per micro-batch, so
a step with gradient_accumulation_steps = N moves 2N copies of every layer. When the model has an active block swap,
Trainer.training_step runs the N micro-batches of the step together through unsloth_zoo's layer-major window instead
(each layer fetched once for the forward and once for the backward of all N), with each micro-batch still going
through the trainer's own training_step: same shapes, loss, num_items_in_batch normalisation and accumulation order,
so losses and gradients match standard accumulation bitwise.

Random draws (dropout, lora_dropout, NEFTune) keep standard accumulation's order: each micro-batch's draws are
learned from a standard step of its shape, the window starts every micro-batch where standard accumulation would and
checks it (see the random draws section below).

"auto" by default (_DEFAULT_MODE). UNSLOTH_OFFLOAD_LAYER_MAJOR: "1" runs a window whenever eligible, for any
decoder; "auto" (or unset) for any decoder that passes the structural checks (_structural_blocker), after two steps
whose swap copies stalled the GPU for more than 5% of the swapped layers' time (on after
them with UNSLOTH_OFFLOAD_STATS = 0, which removes the stall measurement), and only while the held layer inputs fit on the
card beside the standard step's peak measured in the second step; "0": never. Every refusal below applies to
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
try:
    from unsloth_zoo.block_swap_layer_major import (
        ZERO_CONSUMPTION,
        rng_consumption,
        rng_same,
        rng_snapshot,
    )
except Exception:  # unsloth_zoo whose windows cannot follow standard accumulation's random draws
    rng_snapshot = None
try:
    from unsloth_zoo.block_swap_layer_major import COMPILED_IN_WINDOW, compile_epoch
except Exception:  # unsloth_zoo whose windows do not watch torch.compile
    compile_epoch = None

_TLS = threading.local()
# With UNSLOTH_OFFLOAD_LAYER_MAJOR unset: "auto" (after measured copy stalls, when the held layer inputs fit beside the
# standard step's measured peak); "0" turns windows off, "1" runs them whenever eligible.
_DEFAULT_MODE = "auto"
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
# Expert counts: a mixture-of-experts window also holds every layer's recorded router logits (float32 at most).
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


def _engine_rng():
    """Whether unsloth_zoo's window takes a random-draw plan (micro-batches start where standard ones would)."""
    if run_layer_major is None or rng_snapshot is None:
        return False
    key = id(run_layer_major)
    if key not in _ENGINE_RNG:
        try:
            _ENGINE_RNG[key] = "rng_plan" in inspect.signature(run_layer_major).parameters
        except (TypeError, ValueError):
            _ENGINE_RNG[key] = False
    return _ENGINE_RNG[key]


_ENGINE_RNG = {}


def _engine_module():
    return (
        sys.modules.get(getattr(run_layer_major, "__module__", None) or "")
        if run_layer_major is not None
        else None
    )


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
    # optimizers/auto_fit.py: builds the optimizer; an in-backward step it engages is refused on its own.
    "create_optimizer",
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


def _unsloth_compiled_patch(module):
    """A forward Unsloth's own temporary patches compiled onto the module's class (patch_function(..., fullgraph =
    ...), such as Gemma 4's TextMLP): the same compiled function runs in standard accumulation, and a window's
    recompile-limit error rolls back to it."""
    if "forward" in module.__dict__:
        return False
    forward = type(module).__dict__.get("forward")
    original = getattr(forward, "_torchdynamo_orig_callable", None)
    if original is None or isinstance(original, functools.partial) or hasattr(original, "__self__"):
        return False
    owner = getattr(original, "__module__", "") or ""
    return owner.split(".")[0] in ("unsloth", "unsloth_zoo")


def _compiled_in_layers(layers):
    """torch.compile on a decoder layer or inside one, other than Unsloth's block compile and Unsloth's own compiled
    patches (proven): module.compile() takes the layer's checkpoint wrapper into the graph, and the window's extra
    no-grad specialisations can use up the recompile budget the standard retry needs."""
    optimized = getattr(getattr(torch, "_dynamo", None), "OptimizedModule", None)
    for module in layers.modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            return True
        if optimized is not None and isinstance(module, optimized):
            return True
        if hasattr(
            getattr(module, "forward", None), "_torchdynamo_orig_callable"
        ) and not _unsloth_compiled_patch(module):
            return True
    return False


def _compiled_code(layers):
    """Whether the decoder layers run torch.compile'd code (Unsloth's block compile, its compiled patches)."""
    for module in layers.modules():
        if getattr(module, "_compiled_call_impl", None) is not None:
            return True
        if "unsloth_compiled" in (getattr(type(module), "__module__", "") or ""):
            return True
        for owner in (module, type(module)):
            forward = getattr(owner, "forward", None)
            if hasattr(forward, "_torchdynamo_orig_callable"):
                return True
    return False


def _compile_settled(trainer, layers):
    """A window may run only once torch.compile has settled: Dynamo picks a graph from the first shapes and grad mode
    it sees, so a window that compiles could pick other graphs (and kernels) than standard accumulation for the same
    call. Settled = the previous step (standard or window) compiled nothing; before any step, = no compiled code in
    the decoder layers (or no torch.compile in the process)."""
    state = trainer.__dict__.get("_unsloth_layer_major_compile")
    if state is not None and state.get("settled") is not None:
        return state["settled"]
    return compile_epoch is None or compile_epoch() is None or not _compiled_code(layers)


def _compile_step_begin(trainer):
    """At each step's start: settled when nothing compiled since the previous step's start."""
    if compile_epoch is None:
        return
    state = trainer.__dict__.setdefault(
        "_unsloth_layer_major_compile", {"epoch": None, "settled": None, "steps": 0}
    )
    epoch = compile_epoch()
    if state["steps"]:
        state["settled"] = epoch == state["epoch"]
    state["epoch"] = epoch
    state["steps"] += 1


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


def _hidden_first(cls):
    """Whether a decoder layer class's forward takes the hidden states first (positionally or as hidden_states): the
    window passes every other argument through unchanged and swaps only that one."""
    forward = getattr(cls, "forward", None)
    try:
        params = list(inspect.signature(forward).parameters.values())[1:]
    except (TypeError, ValueError):
        return False
    if any(p.name == "hidden_states" for p in params):
        return True
    return bool(params) and params[0].kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_POSITIONAL,
    )


def _structural_blocker(model, layers):
    """Why the decoder stack itself cannot run layer by layer, or None. Checked for every decoder (no class list):
    the window needs each layer to map the hidden states to the next layer's hidden states with no state passed
    from one layer to another. Everything else (cache, mapping, hidden-state changes between layers, non-hidden
    inputs needing gradients, output recorders) is checked on every call by unsloth_zoo's window."""
    if not isinstance(layers, torch.nn.ModuleList) or not len(layers):
        return "the block swap is not on a decoder layer list"
    config = _text_config(model)
    shared = getattr(config, "num_kv_shared_layers", 0)
    if isinstance(shared, int) and shared > 0:
        return f"decoder layers share key / value states (num_kv_shared_layers = {shared})"
    for layer in layers:
        for module in layer.modules():
            if getattr(module, "is_kv_shared_layer", False) is True:
                return "decoder layers share key / value states (KV sharing)"
    for cls in {type(layer) for layer in layers}:
        if not _hidden_first(cls):
            return f"{cls.__name__}.forward does not take the hidden states first"
    return None


_VISUAL_KEYS = ("pixel_values", "pixel_values_videos")


def _between_layers_blocker(model, samples):
    """Per step: work the model does on the hidden states between two decoder layers for these micro-batches.
    Qwen3-VL style deepstack adds visual features after the first decoder layers whenever a micro-batch has images
    or video (text-only micro-batches do not), and the window hands each layer exactly the previous layer's output."""
    if not any(
        isinstance(s, Mapping) and any(isinstance(s.get(k), torch.Tensor) for k in _VISUAL_KEYS)
        for s in samples
    ):
        return None
    for module in model.modules():
        if callable(getattr(type(module), "_deepstack_process", None)):
            return (
                f"{type(module).__name__} adds visual features to the hidden states between decoder layers "
                "(deepstack) for micro-batches with images or video"
            )
    return None


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
    if (
        "lomo" in optim
        or "layerwise" in optim
        or getattr(trainer, "_unsloth_updates_in_backward", False)
    ):
        return "this optimizer updates the weights inside every backward"
    if "_unsloth_token_budget_splits" in trainer.__dict__:
        # Its instance-level training_step splits micro-batches into chunks the window never sees.
        return "UNSLOTH_TOKEN_BUDGET"
    if not _trainer_ok(trainer):
        return f"{type(trainer).__name__} overrides training_step or compute_loss"
    why = _structural_blocker(model, layers)
    if why is not None:
        return why
    config = _text_config(model)
    if _engine_module() is not None and not (
        getattr(_engine_module(), "MOE_ROUTER_LOSSES", False)
        or hasattr(_engine_module(), "_thread_forwards")
    ):
        for key in _MOE_KEYS:
            value = getattr(config, key, 0)
            if isinstance(value, int) and value > 0:
                return "a mixture-of-experts model (unsloth_zoo is too old to carry router losses through windows)"
    if not _engine_rng():
        # An unsloth_zoo window without random-draw plans would draw in another order across micro-batches.
        why = _random_draws_expected(trainer, model)
        if why is not None:
            return f"{why} (unsloth_zoo is too old to keep random draws in standard order)"
    return None


def _random_draws_expected(trainer, model):
    """Dropout (config, any module, lora_dropout) or NEFTune: the micro-batches draw random numbers. A hint only: a
    window learns each micro-batch shape's draws from a standard step first (and checks them in any case)."""
    config = _text_config(model)
    for key in _DROPOUT_KEYS:
        value = getattr(config, key, 0)
        if isinstance(value, (int, float)) and value > 0:
            return f"{key} = {value}"
    found = _dropout_modules(model)
    if found is not None:
        return f"dropout {found}"
    if getattr(getattr(trainer, "args", None), "neftune_noise_alpha", None):
        return "NEFTune embedding noise"
    return None


# ---- random draws in standard order ----
# A window starts micro-batch m where standard accumulation would: where micro-batch m - 1's standard step ends. What
# each micro-batch draws (dropout masks, NEFTune noise) depends on its shapes, so it is learned from standard steps
# (every standard micro-batch step while a block swap is active is bracketed by generator snapshots) per micro-batch
# signature and passed to the window as its plan; the window checks every micro-batch ended where the next started.
_RNG_VOLATILE_MAX = 2


def _signature(inputs):
    if not isinstance(inputs, Mapping):
        return None
    out = []
    for key in sorted(inputs, key = str):
        value = inputs[key]
        if isinstance(value, torch.Tensor):
            out.append((str(key), tuple(value.shape), str(value.dtype), str(value.device)))
        elif value is None or isinstance(value, (bool, int, float, str)):
            out.append((str(key), value))
        else:
            out.append((str(key), type(value).__name__))
    return tuple(out)


def _rng_state(trainer):
    return trainer.__dict__.setdefault(
        "_unsloth_layer_major_rng",
        {"ledger": {}, "draws": False, "volatile": 0, "records": [], "off": None},
    )


def _rng_off(trainer, why):
    state = _rng_state(trainer)
    if state["off"] is None:
        state["off"] = why
        _say_once(
            trainer,
            "rng-off",
            f"Unsloth: offload_layers runs micro-batches one after another from here on: {why}.",
        )


def _rng_learn(trainer, sig, consumption):
    state = _rng_state(trainer)
    if consumption is None:
        _rng_off(
            trainer,
            "random draws that a draw count cannot replay (a reseed or cached normal samples)",
        )
        return
    if consumption != ZERO_CONSUMPTION:
        state["draws"] = True
    known = state["ledger"].get(sig)
    if known is not None and known != consumption:
        state["volatile"] += 1
        if state["volatile"] >= _RNG_VOLATILE_MAX:
            _rng_off(
                trainer, "the random draws of a micro-batch depend on its data, not only its shapes"
            )
    state["ledger"][sig] = consumption


def _rng_flush(trainer):
    """Learn from the standard micro-batch steps recorded since the last step began."""
    state = trainer.__dict__.get("_unsloth_layer_major_rng")
    if not state or not state["records"]:
        return
    records, state["records"] = state["records"], []
    for (sig, before, after), nxt in zip(records, records[1:] + [None]):
        if len(before[1]) != len(after[1]):
            continue  # CUDA came up during the step: its generators were not in the first snapshot
        _rng_learn(trainer, sig, rng_consumption(before, after))
        if nxt is not None and len(nxt[1][1]) == len(after[1]) and not rng_same(after, nxt[1]):
            # A window runs every micro-batch's step back to back; draws between them would move.
            _rng_off(trainer, "random numbers are drawn between the micro-batches of a step")


def _rng_recording(trainer):
    """Whether this standard micro-batch step is bracketed by generator snapshots."""
    if not _engine_rng() or _mode() == "off":
        return False
    cached = trainer.__dict__.get("_unsloth_layer_major_target")
    return cached is not None and cached[2] is not None and _rng_state(trainer)["off"] is None


def _rng_plan(trainer, model, samples):
    """(plan, None) for a window over `samples`, or (None, message) when a micro-batch shape's draws are unknown."""
    state = _rng_state(trainer)
    if state["off"] is not None:
        return None, None
    plan = []
    for sample in samples[:-1]:
        known = state["ledger"].get(_signature(sample))
        if known is None:
            if state["draws"] or _random_draws_expected(trainer, model) is not None:
                return None, (
                    "Unsloth: offload_layers runs micro-batches one after another for a micro-batch shape whose "
                    "random draws (dropout, noise) it has not seen yet; it learns them from this step."
                )
            known = ZERO_CONSUMPTION
        plan.append(known)
    return plan, None


def _tokens(sample):
    for key in ("input_ids", "inputs_embeds", "labels", "attention_mask"):
        value = sample.get(key) if isinstance(sample, Mapping) else None
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            return int(value.shape[0]) * int(value.shape[1])
    return None


def _usable_cuda_bytes(device):
    from ._utils import usable_cuda_bytes
    return usable_cuda_bytes(device)


def _cuda_devices(model):
    return sorted({p.device for p in model.parameters() if p.is_cuda}, key = str)


def _peak_probe_start(model):
    """auto's measurement step: the allocation each card starts it from. The peak counters are left alone (they are
    process-wide and other code reports them)."""
    if not torch.cuda.is_available():
        return None
    return {device: torch.cuda.memory_allocated(device) for device in _cuda_devices(model)}


def _peak_probe_end(start):
    """Upper bound on what the measured standard step allocated above its start, per card: the process-wide peak
    since then is at least the step's own (exact when the step set a new peak, larger if an earlier phase did)."""
    if not start:
        return None
    return {
        device: max(0, torch.cuda.max_memory_allocated(device) - before)
        for device, before in start.items()
    }


def _fits(
    model,
    layers,
    samples,
    step_peak = None,
):
    """Whether every micro-batch's layer inputs fit on the card at once (standard checkpointing holds one). With
    `step_peak` (auto's measured standard step, per card) they must also fit beside that step's own peak: a window
    holds them on top of what a standard step allocates."""
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
    if getattr(config, "output_router_logits", False):
        # The router logits of every layer and micro-batch, kept as leaves for the load-balancing loss's gradient.
        experts = max(
            [v for v in (getattr(config, k, 0) for k in _MOE_KEYS) if isinstance(v, int)] + [0]
        )
        need += sum(tokens) * experts * len(layers) * 4
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
    for device, peak in (step_peak or {}).items():
        if device not in devices:
            continue
        usable = _usable_cuda_bytes(device)
        if need + peak > 0.9 * usable:
            return (
                False,
                f"{need / 2**30:.2f} GiB of layer inputs for {len(samples)} micro-batches on top of the standard "
                f"step's {peak / 2**30:.2f} GiB peak vs {usable / 2**30:.2f} GiB free on {device}",
            )
    return True, need


def _auto_state(trainer):
    return trainer.__dict__.setdefault(
        "_unsloth_layer_major_auto",
        {"steps": 0, "since": None, "on": None, "peak_start": None, "peak": None},
    )


def _auto_ready(
    trainer,
    swapper,
    model = None,
):
    """auto: two standard steps, then on only if the second one's swap copies stalled the GPU (with
    UNSLOTH_OFFLOAD_STATS = 0, which removes the stall measurement: on). The second step also measures each card's
    peak above the step's start; _fits keeps the held layer inputs beside it."""
    state = _auto_state(trainer)
    if state["on"] is not None:
        return state["on"]
    timing = getattr(sys.modules.get(type(swapper).__module__), "_TIMING", True)
    state["steps"] += 1
    if state["steps"] == _AUTO_PROBE_STEPS:
        state["since"] = swap_totals(swapper) if timing else None
        state["peak_start"] = _peak_probe_start(model) if model is not None else None
    if state["steps"] <= _AUTO_PROBE_STEPS:
        return False
    state["peak"] = _peak_probe_end(state["peak_start"])
    if not timing:
        state["on"] = True
        return True
    frac = (
        stall_fraction(state["since"], swap_totals(swapper)) if state["since"] is not None else None
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
    blocker = _window_blocker(trainer, model, layers)
    if blocker is not None:
        return (
            None,
            "blocker",
            f"Unsloth: offload_layers runs micro-batches one after another: {blocker}.",
        )
    between = _between_layers_blocker(model, samples)
    if between is not None:
        return (
            None,
            "between",
            f"Unsloth: offload_layers runs micro-batches one after another: {between}.",
        )
    if mode == "auto" and not _auto_ready(trainer, swapper, model):
        return None, None, None
    peak = _auto_state(trainer)["peak"] if mode == "auto" else None
    ok, need = _fits(model, layers, samples, peak)
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
    rng_plan = None
    if layers is not None and _engine_rng():
        _rng_flush(trainer)
        rng_plan, message = _rng_plan(trainer, inner, samples)
        if rng_plan is None:
            if message is not None:
                _say_once(trainer, "rng-learn", message)
            layers = swapper = None
    if layers is not None and not _compile_settled(trainer, layers):
        _say_once(
            trainer,
            "compile",
            "Unsloth: offload_layers runs micro-batches one after another while torch.compile is still building "
            "graphs; layer-major windows start after a step that compiles nothing.",
        )
        layers = swapper = None
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
    rng_trace = []
    try:
        vote = {"agree": agree} if agree is not None else {}
        if rng_plan is not None:
            vote.update(rng_plan = rng_plan, rng_trace = rng_trace)
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
        if compile_epoch is not None and why.startswith(COMPILED_IN_WINDOW):
            # Not a protocol break: this step's standard retry compiles in standard order, later steps can window.
            _say_once(
                trainer,
                "compile-abort",
                f"Unsloth: offload_layers ran this step's micro-batches one after another: {why}. Layer-major "
                "windows resume once torch.compile has the graphs.",
            )
            trainer.__dict__.setdefault(
                "_unsloth_layer_major_compile", {"epoch": None, "settled": None, "steps": 0}
            )["settled"] = False
            return None
        if why.startswith("random draws"):
            # The plan was wrong for this step (data-dependent draws, or a shape seen with other draws): this step's
            # standard retry is recorded and corrects it; twice turns windows off.
            state = _rng_state(trainer)
            for sample in samples:
                state["ledger"].pop(
                    _signature(sample), None
                )  # relearned from the retry, not counted twice
            state["volatile"] += 1
            if state["volatile"] >= _RNG_VOLATILE_MAX:
                _rng_off(
                    trainer,
                    "the random draws of a micro-batch depend on its data, not only its shapes",
                )
            return None
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
    if rng_trace:
        # The last micro-batch had no plan entry: learn its shape's draws from the window itself.
        sig = _signature(samples[-1])
        if sig not in _rng_state(trainer)["ledger"]:
            before, after = rng_trace[-1]
            if before is not None and after is not None and len(before[1]) == len(after[1]):
                _rng_learn(trainer, sig, rng_consumption(before, after))
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
        _compile_step_begin(self)
        if "_unsloth_layer_major_rng" in self.__dict__:
            _rng_flush(
                self
            )  # one step's records at a time: draws between steps (optimizer, logging) are not checked
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
        if getattr(_TLS, "passthrough", False):
            return training_step(self, model, inputs, num_items_in_batch, *args, **kwargs)
        loss = _layer_major_step(self, model, inputs, num_items_in_batch, args, kwargs)
        if loss is not None:
            return loss
        before = rng_snapshot() if _rng_recording(self) else None
        loss = training_step(self, model, inputs, num_items_in_batch, *args, **kwargs)
        if before is not None:
            _rng_state(self)["records"].append((_signature(inputs), before, rng_snapshot()))
        return loss

    _get_batch_samples._unsloth_layer_major = True
    _training_step._unsloth_layer_major = True
    _train._unsloth_layer_major = True
    Trainer.get_batch_samples = _get_batch_samples
    Trainer.training_step = _training_step
    if not getattr(train, "_unsloth_layer_major", False):
        Trainer.train = _train
