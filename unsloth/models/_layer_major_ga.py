# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Layer-major gradient accumulation for offload_layers (block swap).

Standard gradient accumulation streams every swapped decoder layer from host RAM (or disk) twice per micro-batch, so
a step with gradient_accumulation_steps = N moves 2N copies of every layer. When the model has an active block swap,
Trainer.training_step runs the N micro-batches of the step together through unsloth_zoo's layer-major window instead
(each layer fetched once for the forward and once for the backward of all N), with each micro-batch still going
through the trainer's own training_step: same shapes, loss, num_items_in_batch normalisation and accumulation order,
so losses and gradients match standard accumulation bitwise.

UNSLOTH_OFFLOAD_LAYER_MAJOR: "auto" (default) turns it on after two steps whose swap copies stalled the GPU for more
than 5% of the swapped layers' time (always on with UNSLOTH_OFFLOAD_STATS = 0, which removes the measurement); "1"
whenever eligible; "0" never. Without an active block swap nothing here runs.
"""

import functools
import os
import sys
import threading

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
_AUTO_STALL = 0.05
_AUTO_PROBE_STEPS = 2
# Trainer classes whose training_step / compute_loss run one model forward per micro-batch.
_SAFE_TRAINERS = ("Trainer", "UnslothTrainer")
_DROPOUT_KEYS = (
    "attention_dropout",
    "hidden_dropout",
    "dropout",
    "resid_pdrop",
    "attn_pdrop",
    "embd_pdrop",
)
_MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")


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


def _trainer_ok(trainer):
    import transformers
    for cls in type(trainer).__mro__:
        if cls is transformers.Trainer:
            return True
        for name in ("training_step", "compute_loss"):
            fn = cls.__dict__.get(name)
            if fn is None or getattr(fn, "_unsloth_layer_major", False):
                continue
            if not (cls.__name__ in _SAFE_TRAINERS or cls.__name__.endswith("SFTTrainer")):
                return False
    return True


def layer_major_ineligible(trainer, model, layers, swapper):
    """Why a layer-major window cannot run for this trainer and model, or None."""
    if run_layer_major is None or not hasattr(swapper, "hold"):
        return "unsloth_zoo is too old for layer-major windows"
    if not getattr(swapper, "indices", None):
        return "no layer is swapped"
    if getattr(swapper, "total", len(layers)) != len(layers):
        return "the block swap does not cover the decoder layer list"
    if (
        torch.distributed.is_available()
        and torch.distributed.is_initialized()
        and torch.distributed.get_world_size() > 1
    ):
        return "distributed training"
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
    if trainer.__dict__.get("_unsloth_token_budget_installed"):
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
    for module in layers.modules():
        if isinstance(module, torch.nn.Dropout) and module.p > 0:
            return f"dropout p = {module.p} in a decoder layer (lora_dropout)"
    return None


def _tokens(sample):
    for key in ("input_ids", "inputs_embeds", "labels", "attention_mask"):
        value = sample.get(key) if isinstance(sample, dict) else None
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            return int(value.shape[0]) * int(value.shape[1])
    return None


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
    device = (
        weight.device
        if weight.device.type == "cuda"
        else torch.device("cuda", torch.cuda.current_device())
    )
    free = (
        torch.cuda.mem_get_info(device)[0]
        + torch.cuda.memory_reserved(device)
        - torch.cuda.memory_allocated(device)
    )
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


def _layer_major_step(trainer, model, inputs, num_items_in_batch, args, kwargs):
    """The loss of micro-batch `inputs` if a layer-major window ran (or runs now) for it, else None."""
    window = trainer.__dict__.get("_unsloth_layer_major_window")
    batch = trainer.__dict__.get("_unsloth_layer_major_batch")
    if window is not None:
        samples, losses, k = window
        if k < len(samples) and samples[k] is inputs:
            trainer._unsloth_layer_major_window = (
                (samples, losses, k + 1) if k + 1 < len(samples) else None
            )
            return losses[k]
        trainer._unsloth_layer_major_window = None
        if batch is not None and batch[0] is samples:
            # Same step, but the trainer passed something other than the micro-batch the window already
            # accumulated (a wrapper that rewrites inputs): running it would count that data twice.
            raise RuntimeError(
                "Unsloth: offload_layers layer-major gradient accumulation already ran micro-batch "
                f"{k} of this step, but training_step got different inputs for it. "
                "Set UNSLOTH_OFFLOAD_LAYER_MAJOR=0 to turn it off."
            )
    if batch is None or trainer.__dict__.get("_unsloth_layer_major_off"):
        return None
    samples, items = batch
    if len(samples) < 2 or samples[0] is not inputs or items is not num_items_in_batch:
        return None
    mode = layer_major_mode() if run_layer_major is not None else "off"
    if mode == "off" or not getattr(model, "training", True):
        return None
    cached = trainer.__dict__.get("_unsloth_layer_major_target")
    # Re-planned swaps (offload_layers = "auto") install a new BlockSwap on the same layer list.
    if (
        cached is not None
        and cached[0] == id(model)
        and (cached[1] is None or cached[1].__dict__.get("_unsloth_block_swap") is cached[2])
    ):
        _, layers, swapper, why = cached
    else:
        layers, swapper = _block_swap_layers(model)
        why = None if swapper is None else layer_major_ineligible(trainer, model, layers, swapper)
        trainer._unsloth_layer_major_target = (id(model), layers, swapper, why)
    if swapper is None:
        return None
    if why is not None:
        _say_once(
            trainer, "why", f"Unsloth: offload_layers runs micro-batches one after another: {why}."
        )
        trainer._unsloth_layer_major_off = True
        return None
    if mode == "auto" and not _auto_ready(trainer, swapper):
        return None
    ok, need = _fits(model, layers, samples)
    if not ok:
        _say_once(
            trainer,
            "fit",
            f"Unsloth: offload_layers runs micro-batches one after another this step: {need}.",
        )
        return None
    fns = [
        functools.partial(_micro_step, trainer, model, s, num_items_in_batch, args, kwargs)
        for s in samples
    ]
    losses, why = run_layer_major(layers, fns, swapper = swapper, model = model)
    if losses is None:
        _say_once(
            trainer,
            "abort",
            f"Unsloth: offload_layers layer-major gradient accumulation stopped ({why}); "
            "micro-batches run one after another from here on.",
        )
        trainer._unsloth_layer_major_off = True
        return None
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
    if not hasattr(Trainer, "get_batch_samples"):
        return
    get_batch_samples = Trainer.get_batch_samples
    training_step = Trainer.training_step

    @functools.wraps(get_batch_samples)
    def _get_batch_samples(self, *args, **kwargs):
        out = get_batch_samples(self, *args, **kwargs)
        try:
            samples, items = out
            self._unsloth_layer_major_batch = (
                (samples, items) if isinstance(samples, list) else None
            )
        except (TypeError, ValueError):
            self._unsloth_layer_major_batch = None
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
    Trainer.get_batch_samples = _get_batch_samples
    Trainer.training_step = _training_step
