# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""DDP defaults for Unsloth trainers.

transformers' Trainer passes ``find_unused_parameters=True`` to DDP whenever the model is not a
``PreTrainedModel`` and ``ddp_find_unused_parameters`` is None, which is every PEFT model. Under reentrant
gradient checkpointing that crashes on the first backward ("marked as ready twice"): the checkpointed
forward runs without grad, DDP marks every LoRA weight unused, then the recompute marks them again.
transformers already picks False for a ``PreTrainedModel`` under checkpointing; this does the same for a
plain LoRA adapter on a dense text model, where every trainable parameter gets a gradient each step.
Without checkpointing True works (and tolerates an adapter a step never runs), so it is left alone.
"""

import os

__all__ = [
    "unsloth_default_ddp_find_unused_parameters",
    "install_unsloth_ddp_comm_defaults",
]

_AUTO_FLAG = "_unsloth_auto_ddp_find_unused_parameters"
# LoRA tensors that are used whenever their base layer runs.
_LORA_PARAM_MARKERS = (".lora_A.", ".lora_B.", ".lora_magnitude_vector.")
# Modules a step can skip (unrouted experts, a vision tower on a text batch, MTP heads), or that
# Unsloth's fused loss / embedding paths may read by weight and bypass the LoRA forward.
_SKIPPABLE_NAME_MARKERS = (
    "expert",
    "vision",
    "visual",
    "audio",
    "image",
    "projector",
    "mtp",
    "lm_head",
    "embed",
)
_SUBMODEL_CONFIG_KEYS = ("vision_config", "audio_config", "vision_tower", "speech_config")
_MOE_CONFIG_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "moe_num_experts",
    "num_experts_per_tok",
)


def _is_distributed():
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > 1
    except Exception:
        pass
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _configs(config):
    seen = []
    stack = [config]
    while stack:
        c = stack.pop()
        if c is None or any(c is s for s in seen):
            continue
        seen.append(c)
        # encoder / decoder: composite configs (SpeechEncoderDecoder, VisionEncoderDecoder).
        for key in ("text_config", "language_config", "llm_config", "encoder", "decoder"):
            value = getattr(c, key, None)
            if value is not None and not isinstance(
                value, (str, bool, int, float, list, tuple, dict)
            ):
                stack.append(value)
    return seen


def _config_may_skip(config):
    for c in _configs(config):
        for key in _SUBMODEL_CONFIG_KEYS:
            if getattr(c, key, None) is not None:
                return True
        for key in _MOE_CONFIG_KEYS:
            value = getattr(c, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return True
        # Optional cross-attention (GPT-2 style) only runs when encoder states are passed.
        if getattr(c, "add_cross_attention", False) and not getattr(c, "is_encoder_decoder", False):
            return True
        # layerdrop, encoder_layerdrop, decoder_layerdrop, speech_encoder_layerdrop, ...
        for key, value in getattr(c, "__dict__", {}).items():
            if (
                "layerdrop" in key
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value > 0
            ):
                return True
    return False


def _gradient_checkpointing_on(args, model):
    """True unless checkpointing is off both in the args (Trainer turns it on in train()) and on the model."""
    if getattr(args, "gradient_checkpointing", False):
        return True
    try:
        return any(getattr(m, "gradient_checkpointing", False) for m in model.modules())
    except Exception:
        return True


def _lora_uses_every_trainable_param(model):
    """True only when we can show every trainable tensor is a LoRA weight run on each forward."""
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        # transformers already decides for these (False under gradient checkpointing).
        return False
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or len(peft_config) != 1:
        return False
    cfg = next(iter(peft_config.values()))
    if (
        str(getattr(getattr(cfg, "peft_type", None), "value", getattr(cfg, "peft_type", "")))
        != "LORA"
    ):
        return False
    for attr in ("modules_to_save", "target_parameters", "trainable_token_indices"):
        if getattr(cfg, attr, None):
            return False
    base = getattr(model, "get_base_model", lambda: None)()
    config = getattr(base, "config", None) or getattr(model, "config", None)
    # Decoder-only causal LMs only: encoder-decoder and speech models can leave adapters idle (t2u_model).
    if getattr(config, "is_encoder_decoder", False) or not type(base).__name__.endswith(
        ("ForCausalLM", "LMHeadModel")
    ):
        return False
    if _config_may_skip(config):
        return False
    # The output head under any name (output, output_layer, embed_out): fused losses read its weight.
    try:
        head = base.get_output_embeddings()
        head_params = {id(p) for p in head.parameters()} if head is not None else set()
    except Exception:
        head_params = set()
    n_trainable = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_trainable += 1
        if id(param) in head_params:
            return False
        if not any(m in name for m in _LORA_PARAM_MARKERS):
            return False
        lowered = name.lower()
        if any(m in lowered for m in _SKIPPABLE_NAME_MARKERS):
            return False
    return n_trainable > 0


def unsloth_default_ddp_find_unused_parameters(args, model):
    """Set ``args.ddp_find_unused_parameters = False`` for a plain LoRA model under DDP with gradient
    checkpointing when the user left it None.

    Kept at None (transformers' True) for anything a step may skip: MoE / expert LoRA, vision or audio
    towers, MTP heads, lm_head / embedding adapters, modules_to_save, several adapters, non-LoRA PEFT.
    Returns the value now on ``args``.
    """
    if args is None or not hasattr(args, "ddp_find_unused_parameters"):
        return None
    if getattr(args, _AUTO_FLAG, False):
        # Reused args object: our earlier False is not a user choice, re-decide; a value set since then wins.
        if args.ddp_find_unused_parameters is False:
            args.ddp_find_unused_parameters = None
        object.__setattr__(args, _AUTO_FLAG, False)
    if args.ddp_find_unused_parameters is not None:
        return args.ddp_find_unused_parameters
    if os.environ.get("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", "1") == "0":
        return None
    if not _is_distributed():
        return None
    try:
        safe = _gradient_checkpointing_on(args, model) and _lora_uses_every_trainable_param(model)
    except Exception:
        safe = False
    if safe:
        args.ddp_find_unused_parameters = False
        object.__setattr__(args, _AUTO_FLAG, True)
    return args.ddp_find_unused_parameters



_COMM_HOOKS = ("bf16", "fp16")
_COMM_PRINTED = set()


def _note(key, message):
    if key not in _COMM_PRINTED:
        _COMM_PRINTED.add(key)
        print(message)


def _plain_ddp(accelerator):
    """True when accelerate would wrap the model in torch DDP (not FSDP / DeepSpeed / Megatron / XLA)."""
    distributed_type = str(getattr(getattr(accelerator, "distributed_type", None), "value", ""))
    if not distributed_type.startswith("MULTI_"):
        return False
    state = getattr(accelerator, "state", None)
    for plugin in ("fsdp_plugin", "deepspeed_plugin", "megatron_lm_plugin"):
        if getattr(state, plugin, None) is not None or getattr(accelerator, plugin, None) is not None:
            return False
    return True


def _is_full_finetune(model):
    if model is None or getattr(model, "peft_config", None) is not None:
        return False
    return any(p.requires_grad for p in model.parameters())


def _prepare_ddp_handler(accelerator, model):
    """Construction-time DDP kwargs, set on accelerate's handler right before it builds DDP."""
    from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs

    handler = accelerator.ddp_handler
    if handler is None:
        handler = DistributedDataParallelKwargs()
    if not isinstance(handler, DistributedDataParallelKwargs):
        return
    changed = False
    # Full finetuning: gradients become views into the all-reduce buckets, so DDP keeps one copy of
    # every gradient instead of two (Qwen3-0.6B on 2x T4: 13.34 -> 11.12 GiB peak, same step time,
    # same losses). LoRA gradients are tiny and stay as they are.
    if (
        os.environ.get("UNSLOTH_DDP_GRADIENT_AS_BUCKET_VIEW", "1") != "0"
        and not handler.gradient_as_bucket_view
        and _is_full_finetune(model)
    ):
        handler.gradient_as_bucket_view = True
        changed = True
        _note("view", "Unsloth: DDP gradient_as_bucket_view = True (full finetuning).")
    # Opt-in: halves the all-reduce bytes for float32 gradients, but rounds them before the average.
    hook = os.environ.get("UNSLOTH_DDP_COMM_HOOK", "").strip().lower()
    if hook in _COMM_HOOKS and handler.comm_hook == DDPCommunicationHookType.NO:
        handler.comm_hook = (
            DDPCommunicationHookType.BF16 if hook == "bf16" else DDPCommunicationHookType.FP16
        )
        changed = True
        _note("hook", f"Unsloth: DDP gradient all-reduce in {hook} (UNSLOTH_DDP_COMM_HOOK={hook}).")
    elif hook and hook not in _COMM_HOOKS and hook not in ("0", "none", "off"):
        _note("hook_bad", f"Unsloth: UNSLOTH_DDP_COMM_HOOK={hook} ignored (use bf16 or fp16).")
    if changed:
        accelerator.ddp_handler = handler


def install_unsloth_ddp_comm_defaults(trainer):
    """Wrap ``accelerator.prepare_model`` so plain DDP gets Unsloth's construction-time defaults.

    Applied inside prepare_model since transformers 4.x replaces ``ddp_handler`` in ``_wrap_model``.
    Returns True when the wrapper is installed.
    """
    accelerator = getattr(trainer, "accelerator", None)
    try:
        if accelerator is None or not _plain_ddp(accelerator):
            return False
        prepare_model = accelerator.prepare_model
        if getattr(prepare_model, "_unsloth_ddp_comm_defaults", False):
            return True

        def _prepare_model_ddp_defaults(*args, **kwargs):
            model = args[0] if args else kwargs.get("model")
            # Native FSDP2 shards in Trainer.__init__; accelerate never builds DDP for it.
            if getattr(model, "_unsloth_fsdp2_native", False) is not True:
                try:
                    _prepare_ddp_handler(accelerator, model)
                except Exception as e:
                    _note("err_pre", f"Unsloth: DDP defaults skipped ({e}).")
            return prepare_model(*args, **kwargs)

        _prepare_model_ddp_defaults._unsloth_ddp_comm_defaults = True
        _prepare_model_ddp_defaults.__wrapped__ = prepare_model
        accelerator.prepare_model = _prepare_model_ddp_defaults
        return True
    except Exception as e:
        _note("err", f"Unsloth: DDP defaults skipped ({e}).")
        return False
