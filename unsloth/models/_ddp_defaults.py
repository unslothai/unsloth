# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""DDP defaults for Unsloth trainers.

transformers' Trainer passes ``find_unused_parameters=True`` to DDP whenever the model is not a
``PreTrainedModel`` and ``ddp_find_unused_parameters`` is None, which is every PEFT model. Under reentrant
gradient checkpointing that crashes on the first backward ("marked as ready twice"): the checkpointed
forward runs without grad, DDP marks every LoRA weight unused, then the recompute marks them again.
transformers already picks False for a ``PreTrainedModel`` under checkpointing; this does the same for a
plain LoRA adapter on a dense text model, where every trainable parameter gets a gradient each step,
including continued pretraining's trainable embed_tokens / lm_head copies (modules_to_save).
Without checkpointing True works (and tolerates an adapter a step never runs), so it is left alone.

Conversely a fully finetuned VLM gets False from transformers, and a text-only batch leaves its
trainable tower without gradients; Unsloth checkpoints VLMs non-reentrantly, so True is picked (as TRL).
Gemma 4 / Gemma 3N stay reentrant, where True crashes: with False their towers are kept out of DDP and averaged
after each synced backward instead.
"""

import os

import torch

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
_CLASSIFIER_HEADS = ("score", "classifier")
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
        # Mllama text: its cross-attention decoder layers are skipped without image states.
        if getattr(c, "cross_attention_layers", None):
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


def _trainable_lora_config(model):
    """Config of the active adapter owning every trainable LoRA tensor; a frozen extra (DPO "ref") does not count."""
    peft_config = getattr(model, "peft_config", None)
    if not isinstance(peft_config, dict) or not peft_config:
        return None
    if len(peft_config) == 1:
        return next(iter(peft_config.values()))
    owners = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            for marker in _LORA_PARAM_MARKERS:
                if marker in name:
                    owners.add(name.split(marker, 1)[1].split(".", 1)[0])
    active = getattr(model, "active_adapter", None)
    if isinstance(active, (list, tuple)):
        active = active[0] if len(active) == 1 else None
    if owners != {active}:
        return None
    return peft_config.get(active)


def _lora_uses_every_trainable_param(model):
    """True only when we can show every trainable tensor is a LoRA weight run on each forward."""
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        # transformers already decides for these (False under gradient checkpointing).
        return False
    cfg = _trainable_lora_config(model)
    if cfg is None:
        return False
    if (
        str(getattr(getattr(cfg, "peft_type", None), "value", getattr(cfg, "peft_type", "")))
        != "LORA"
    ):
        return False
    base = getattr(model, "get_base_model", lambda: None)()
    classifier = type(base).__name__.endswith("ForSequenceClassification")
    # modules_to_save that run every forward: a classifier head, or embed_tokens / lm_head (continued pretraining).
    always_run = set(_CLASSIFIER_HEADS) if classifier else set()
    always_run |= _embedding_leaf_names(base)
    for attr in ("modules_to_save", "target_parameters", "trainable_token_indices"):
        value = getattr(cfg, attr, None)
        if attr == "modules_to_save" and value and set(value) <= always_run:
            continue
        if value:
            return False
    config = getattr(base, "config", None) or getattr(model, "config", None)
    # Decoder-only causal LMs only: encoder-decoder and speech models can leave adapters idle (t2u_model).
    if getattr(config, "is_encoder_decoder", False) or not (
        classifier or type(base).__name__.endswith(("ForCausalLM", "LMHeadModel"))
    ):
        return False
    if _config_may_skip(config):
        return False
    # The output head under any name (output, output_layer, embed_out): fused losses read its weight.
    head_params = _module_param_ids(base, "get_output_embeddings")
    embedding_copies = _modules_to_save_copy_ids(
        base, ("get_input_embeddings", "get_output_embeddings")
    )
    n_trainable = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n_trainable += 1
        if id(param) in embedding_copies:
            continue
        if id(param) in head_params:
            return False
        if not any(m in name for m in _LORA_PARAM_MARKERS) and not (
            classifier
            and ".modules_to_save." in name
            and any(f".{h}.modules_to_save." in f".{name}" for h in _CLASSIFIER_HEADS)
        ):
            return False
        lowered = name.lower()
        if any(m in lowered for m in _SKIPPABLE_NAME_MARKERS):
            return False
    return n_trainable > 0


def _module_param_ids(base, getter):
    try:
        module = getattr(base, getter)()
        return {id(p) for p in module.parameters()} if module is not None else set()
    except Exception:
        return set()


def _embedding_leaf_names(base):
    """Leaf names (embed_tokens, lm_head, wte, ...) of the input / output embeddings."""
    targets = []
    for getter in ("get_input_embeddings", "get_output_embeddings"):
        try:
            module = getattr(base, getter)()
        except Exception:
            module = None
        if module is not None:
            targets.append(module)
    names = set()
    if not targets:
        return names
    try:
        for name, module in base.named_modules():
            if name and any(module is t for t in targets):
                names.add(name.rsplit(".", 1)[-1])
    except Exception:
        pass
    return names


def _modules_to_save_copy_ids(base, getters):
    """Parameters of PEFT's trainable modules_to_save copies of the input / output embeddings."""
    ids = set()
    for getter in getters:
        try:
            module = getattr(base, getter)()
        except Exception:
            continue
        copies = getattr(module, "modules_to_save", None)
        if module is None or not isinstance(copies, torch.nn.ModuleDict):
            continue
        for copy in copies.values():
            ids.update(id(p) for p in copy.parameters())
    return ids


# Towers a text-only batch never runs.
_TOWER_NAME_MARKERS = ("vision", "visual", "audio", "image", "projector")
# vision.py forces these back to reentrant checkpointing, where True fails ("marked ready twice").
_REENTRANT_VLM_MODEL_TYPES = ("gemma3n", "gemma4")


def _unwrapped_models(model):
    out = []
    while model is not None and not any(model is m for m in out):
        out.append(model)
        model = getattr(model, "module", None)
    if out:
        get_base = getattr(out[-1], "get_base_model", None)
        if callable(get_base):
            base = get_base()
            if base is not None and not any(base is m for m in out):
                out.append(base)
    return out


def _checkpoint_forced_non_reentrant():
    # vision.py's DDP wrapper (marked _unsloth_original) forces every checkpoint call non-reentrant.
    try:
        import torch.utils.checkpoint as torch_checkpoint
    except Exception:
        return False
    return getattr(torch_checkpoint.checkpoint, "_unsloth_original", None) is not None


def _vlm_tower_needs_unused_search(args, model):
    """Trainable tower a text-only batch skips, under ``_unsloth_vlm_gc_non_reentrant``."""
    models = _unwrapped_models(model)
    if not any(getattr(m, "_unsloth_vlm_gc_non_reentrant", False) is True for m in models):
        return False
    kwargs = getattr(args, "gradient_checkpointing_kwargs", None) or {}
    # Unsloth's generated configs pin use_reentrant = True; under the DDP wrapper that pin is not what runs.
    if (
        isinstance(kwargs, dict)
        and kwargs.get("use_reentrant") is True
        and not _checkpoint_forced_non_reentrant()
    ):
        return False
    config = getattr(models[-1], "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if any(t in model_type for t in _REENTRANT_VLM_MODEL_TYPES):
        return False
    if not any(
        getattr(c, key, None) is not None for c in _configs(config) for key in _SUBMODEL_CONFIG_KEYS
    ):
        return False
    return any(
        p.requires_grad and any(m in name.lower() for m in _TOWER_NAME_MARKERS)
        for name, p in model.named_parameters()
    )


def _reentrant_vlm_text_lora(model):
    """A LoRA adapter on a VLM that vision.py checkpoints reentrantly (Gemma 4).

    Under reentrant checkpointing transformers' True crashes on the first backward ("mark a variable ready only
    once"), and a language-layer adapter gets a gradient on every step, so False is the setting that trains;
    vision / audio tower adapters are kept out of DDP and averaged by ``_all_reduce_tower_grads``. A dense model
    only: routed experts, embedding / head adapters and extra trainable tensors keep the default."""
    from transformers import PreTrainedModel

    if isinstance(model, PreTrainedModel):
        return False
    models = _unwrapped_models(model)
    config = getattr(models[-1], "config", None)
    model_type = str(getattr(config, "model_type", "") or "").lower()
    if not model_type.startswith("gemma4"):
        return False
    cfg = _trainable_lora_config(model)
    if cfg is None:
        return False
    if (
        str(getattr(getattr(cfg, "peft_type", None), "value", getattr(cfg, "peft_type", "")))
        != "LORA"
    ):
        return False
    for attr in ("modules_to_save", "target_parameters", "trainable_token_indices"):
        if getattr(cfg, attr, None):
            return False
    for c in _configs(config):
        if getattr(c, "enable_moe_block", False) is True:
            return False
        for key in _MOE_CONFIG_KEYS:
            value = getattr(c, key, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return False
    towers_synced = os.environ.get("UNSLOTH_DDP_TOWER_SYNC", "1") != "0"
    n_trainable = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if not any(m in name for m in _LORA_PARAM_MARKERS):
            return False
        lowered = name.lower()
        if towers_synced and any(m in lowered for m in _TOWER_NAME_MARKERS):
            continue
        n_trainable += 1
        if any(m in lowered for m in _SKIPPABLE_NAME_MARKERS):
            return False
    return n_trainable > 0


def unsloth_default_ddp_find_unused_parameters(args, model):
    """Set ``args.ddp_find_unused_parameters = False`` for a plain LoRA model under DDP with gradient
    checkpointing when the user left it None.

    Kept at None (transformers' True) for anything a step may skip: MoE / expert LoRA, vision or audio
    towers, MTP heads, lm_head / embedding LoRA adapters, modules_to_save other than a classifier head or
    the embed_tokens / lm_head copies, several adapters, non-LoRA PEFT.
    False too for a LoRA on Gemma 4, which checkpoints reentrantly (True crashes there).
    Set to True for a VLM with a trainable tower under non-reentrant checkpointing. Returns the value now on ``args``.
    """
    if args is None or not hasattr(args, "ddp_find_unused_parameters"):
        return None
    auto = getattr(args, _AUTO_FLAG, None)
    if isinstance(auto, tuple):
        # Reused args object: our earlier value is not a user choice, re-decide; a value set since then wins.
        if args.ddp_find_unused_parameters is auto[0]:
            args.ddp_find_unused_parameters = None
        object.__setattr__(args, _AUTO_FLAG, None)
    if args.ddp_find_unused_parameters is not None:
        return args.ddp_find_unused_parameters
    if os.environ.get("UNSLOTH_DDP_FIND_UNUSED_PARAMETERS_AUTO", "1") == "0":
        return None
    if not _is_distributed():
        return None
    try:
        gc_on = _gradient_checkpointing_on(args, model)
        safe = gc_on and (
            _lora_uses_every_trainable_param(model) or _reentrant_vlm_text_lora(model)
        )
        tower = gc_on and not safe and _vlm_tower_needs_unused_search(args, model)
    except Exception:
        safe = tower = False
    if safe or tower:
        args.ddp_find_unused_parameters = not safe
        object.__setattr__(args, _AUTO_FLAG, (not safe,))
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
        if (
            getattr(state, plugin, None) is not None
            or getattr(accelerator, plugin, None) is not None
        ):
            return False
    return True


def _is_full_finetune(model):
    if model is None or getattr(model, "peft_config", None) is not None:
        return False
    return any(p.requires_grad for p in model.parameters())


def _tower_params(model):
    """Trainable vision / audio tower weights of a VLM."""
    config = getattr(_unwrapped_models(model)[-1], "config", None)
    if not any(
        getattr(c, key, None) is not None for c in _configs(config) for key in _SUBMODEL_CONFIG_KEYS
    ):
        return [], []
    names, params = [], []
    for name, p in model.named_parameters():
        if p.requires_grad and any(m in name.lower() for m in _TOWER_NAME_MARKERS):
            names.append(name)
            params.append(p)
    return names, params


def _exclude_towers_from_ddp(model):
    """Without unused-parameter search (reentrant checkpointing cannot use it) a skipped tower stalls DDP's
    reduction. Keep the towers out of DDP; ``_all_reduce_tower_grads`` averages them after each synced backward."""
    if os.environ.get("UNSLOTH_DDP_TOWER_SYNC", "1") == "0":
        return False
    names, params = _tower_params(model)
    if not params:
        return False
    from torch.nn.parallel import DistributedDataParallel

    ignored = list(getattr(model, "_ddp_params_and_buffers_to_ignore", None) or [])
    seen = set(ignored)
    DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
        model, ignored + [n for n in names if n not in seen]
    )
    model._unsloth_ddp_tower_params = params
    _note(
        "tower",
        f"Unsloth: DDP averages {len(params)} vision / audio tower weights itself "
        "(a text-only batch leaves them unused).",
    )
    return True


def _exclude_frozen_from_ddp(model):
    """DDP's start broadcasts every parameter from rank 0, flattening each into a bucket: a second copy of the
    largest tensor. A LoRA model's frozen weights come from the same checkpoint on every rank and get no
    gradient, so skip them (Gemma4 / Gemma3n per-layer embeddings, 4.4 GiB, ran a T4 out of memory)."""
    if os.environ.get("UNSLOTH_DDP_SYNC_FROZEN", "0") == "1":
        return 0
    if getattr(model, "peft_config", None) is None:
        return 0
    ignored = list(getattr(model, "_ddp_params_and_buffers_to_ignore", None) or [])
    seen = set(ignored)
    added = [
        name for name, p in model.named_parameters() if not p.requires_grad and name not in seen
    ]
    if not added:
        return 0
    from torch.nn.parallel import DistributedDataParallel

    DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, ignored + added)
    return len(added)


def _no_dynamo_ddp_graph_split():
    """Dynamo's DDPOptimizer splits every graph it compiles while a DDP forward is active at gradient bucket
    boundaries. Unsloth compiles functions inside the layers, not the DDP model, and a non-reentrant checkpoint
    recompute runs them again in backward, outside that forward, unsplit: it saves different tensors than the
    forward did and raises CheckpointError (Gemma4 E2B LoRA under DDP). Gradients still reach DDP's buckets
    per parameter. UNSLOTH_DDP_GRAPH_SPLIT=1 keeps the split."""
    if os.environ.get("UNSLOTH_DDP_GRAPH_SPLIT", "0") == "1":
        return False
    try:
        import torch._dynamo
    except Exception:
        return False
    config = torch._dynamo.config
    if getattr(config, "optimize_ddp", False) in (False, "no_optimization"):
        return False
    config.optimize_ddp = False
    _note(
        "graph_split",
        "Unsloth: torch.compile does not split graphs at DDP buckets (gradient checkpointing).",
    )
    return True


def _all_reduce_tower_grads(params, group = None):
    """Average tower gradients across ranks like DDP(find_unused_parameters = True): a weight no rank used
    keeps ``grad = None``; one used on some rank is averaged with zeros from the others."""
    import torch
    import torch.distributed as dist

    world = dist.get_world_size(group)
    used = torch.tensor(
        [p.grad is not None for p in params], dtype = torch.int32, device = params[0].device
    )
    dist.all_reduce(used, group = group)
    buckets, size = [[]], 0
    for p, u in zip(params, used.tolist()):
        if not u:
            continue
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if size > (1 << 26) or (buckets[-1] and buckets[-1][-1].dtype != p.grad.dtype):
            buckets.append([])
            size = 0
        buckets[-1].append(p.grad)
        size += p.grad.numel()
    for grads in buckets:
        if not grads:
            continue
        flat = torch._utils._flatten_dense_tensors(grads)
        flat.div_(world)  # before the sum, like DDP: fp16 tower grads would overflow summed first
        dist.all_reduce(flat, group = group)
        for g, s in zip(grads, torch._utils._unflatten_dense_tensors(flat, grads)):
            g.copy_(s)
    _TOWER_SYNC_COUNT[0] += 1


_TOWER_SYNC_COUNT = [0]


def _broadcast_tower_params(params, group = None):
    """DDP broadcasts the parameters it owns from rank 0 at construction; the excluded towers need the same."""
    import torch.distributed as dist

    src = dist.get_global_rank(group, 0) if group is not None else 0
    for p in params:
        dist.broadcast(p.data, src = src, group = group)


def _install_tower_grad_sync(accelerator, ddp_model):
    _broadcast_tower_params(ddp_model.module._unsloth_ddp_tower_params, ddp_model.process_group)
    wrapped = getattr(accelerator, "_unsloth_tower_ddp", None)
    if wrapped is None:
        wrapped = accelerator._unsloth_tower_ddp = []
    wrapped.append(ddp_model)
    backward = accelerator.backward
    if getattr(backward, "_unsloth_tower_sync", False):
        return

    def _backward_tower_sync(*args, **kwargs):
        out = backward(*args, **kwargs)
        for m in accelerator._unsloth_tower_ddp:
            # Inside no_sync (gradient accumulation) DDP defers its all-reduce; so do the towers.
            if getattr(m, "require_backward_grad_sync", False):
                _all_reduce_tower_grads(m.module._unsloth_ddp_tower_params, m.process_group)
        return out

    _backward_tower_sync._unsloth_tower_sync = True
    _backward_tower_sync.__wrapped__ = backward
    accelerator.backward = _backward_tower_sync


def _prepare_ddp_handler(accelerator, model):
    from accelerate.utils import DDPCommunicationHookType, DistributedDataParallelKwargs

    handler = accelerator.ddp_handler
    if handler is None:
        handler = DistributedDataParallelKwargs()
    if not isinstance(handler, DistributedDataParallelKwargs):
        return
    if not handler.find_unused_parameters:
        _exclude_towers_from_ddp(model)
    _exclude_frozen_from_ddp(model)
    _no_dynamo_ddp_graph_split()
    changed = False
    # Full finetuning only: gradients become bucket views, one copy instead of two.
    if (
        os.environ.get("UNSLOTH_DDP_GRADIENT_AS_BUCKET_VIEW", "1") != "0"
        and not handler.gradient_as_bucket_view
        and _is_full_finetune(model)
    ):
        handler.gradient_as_bucket_view = True
        changed = True
        _note("view", "Unsloth: DDP gradient_as_bucket_view = True (full finetuning).")
    # Opt-in: rounds gradients before the average.
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
    """Wrap ``accelerator.prepare_model``: transformers 4.x replaces ``ddp_handler`` in ``_wrap_model``."""
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
            out = prepare_model(*args, **kwargs)
            ddp = getattr(out, "_orig_mod", out)
            if (
                getattr(model, "_unsloth_ddp_tower_params", None)
                and getattr(ddp, "module", None) is model
            ):
                _install_tower_grad_sync(accelerator, ddp)
            return out

        _prepare_model_ddp_defaults._unsloth_ddp_comm_defaults = True
        _prepare_model_ddp_defaults.__wrapped__ = prepare_model
        accelerator.prepare_model = _prepare_model_ddp_defaults
        return True
    except Exception as e:
        _note("err", f"Unsloth: DDP defaults skipped ({e}).")
        return False
