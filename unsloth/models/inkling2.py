# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Thinking Machines Inkling 2 (``inkling2``: Inkling-2-Small, ``inkling2_loop``: Inkling-2-Mini).

transformers >= 5.24 ships both natively. Two things need Unsloth's help:

- Attention projections are wrapper modules (``q_proj`` = ``Inkling2QueryProjection`` holding ``.linear`` plus a
  scaled L2 clip; ``o_proj`` holds ``.linear`` and the output ``gate_proj``), so the usual LoRA names would land on
  modules PEFT cannot wrap. ``unsloth_lora_target_modules`` maps them to the real Linears.
- The released checkpoints declare ``quantization_config = {"quant_method": "modelopt", "quant_algo": "NVFP4"}``,
  but their layout is Inkling's own (``w13_weight`` + ``_scale`` / ``_amax`` / ``_outer_scale_denominator``), which
  transformers' Inkling conversion mapping decodes itself. The generic ModelOpt NVFP4 rewrite must not touch it.
  transformers 5.24 dequantizes those experts for ``inkling2_loop`` only; ``inkling2`` (Small) maps them as if they
  were plain / FP8 tensors, so ``register_inkling2_expert_dequantization`` gives it the loop's transforms.
"""

import re

__all__ = [
    "INKLING2_MODEL_TYPES",
    "is_inkling2_config",
    "inkling2_owns_modelopt_quantization",
    "drop_inkling2_quantization_config",
    "register_inkling2_expert_dequantization",
    "patch_inkling2_model",
    "checkpoint_inkling2_message_inputs",
    "inkling2_lora_target_modules",
    "raise_if_merging_inkling2",
    "raise_if_inkling2_parallel_unsupported",
    "INKLING2_TP_REASON",
    "INKLING2_PP_REASON",
    "INKLING2_PARALLEL_ALTERNATIVE",
]

INKLING2_MODEL_TYPES = frozenset(
    {
        "inkling2",
        "inkling2_text",
        "inkling2_loop",
        "inkling2_loop_text",
    }
)

# Usual LoRA alias -> Linear leaves under one language-model attention block.
_ATTENTION_TARGETS = {
    "q_proj": ("self_attn.q_proj.linear",),
    "k_proj": ("self_attn.k_proj.linear",),
    "v_proj": ("self_attn.v_proj.linear",),
    # Inkling2GatedOutputProjection: sigmoid(gate_proj(x)) * attn, then linear.
    "o_proj": ("self_attn.o_proj.linear", "self_attn.o_proj.gate_proj"),
}
# Routed experts are 3D parameters, trained through target_parameters by get_peft_model.
_EXPERT_ALIASES = frozenset({"gate_proj", "up_proj", "down_proj", "gate_up_proj"})
# Vision tower leaves (Inkling2VisionAttention / Inkling2VisionMLP are plain Linears).
_VISION_ATTENTION = {
    "q_proj": "q_proj",
    "k_proj": "k_proj",
    "v_proj": "v_proj",
    "o_proj": "out_proj",
    "out_proj": "out_proj",
}
_VISION_MLP = {"fc1": "fc1", "fc2": "fc2", "gate_proj": "fc1", "up_proj": "fc1", "down_proj": "fc2"}
_DEFAULT = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _model_types(config):
    """model_type of a config (object or dict) and of its text_config."""
    out = set()
    for cfg in (config, _get(config, "text_config")):
        model_type = _get(cfg, "model_type")
        if model_type:
            out.add(str(model_type).lower())
    return out


def _get(cfg, key):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(key)
    try:
        return getattr(cfg, key, None)
    except Exception:
        return None


def is_inkling2_config(config) -> bool:
    return bool(_model_types(config) & INKLING2_MODEL_TYPES)


def raise_if_merging_inkling2(model, save_method):
    """Merged and GGUF exports re-read the base checkpoint and add the LoRA deltas to its tensors. The released Inkling 2
    checkpoints name them in Inkling's own layout (``attn.wq_du``, ``controllers.*``; transformers renames them on
    load) and pack the experts in NVFP4 / MXFP8, so the merge places no LoRA weight and writes the base unchanged."""
    if str(save_method).strip().lower() == "lora":
        return
    if not is_inkling2_config(getattr(model, "config", None)):
        return
    raise NotImplementedError(
        f'Unsloth: `save_method = "{save_method}"` (merged and GGUF exports) does not support Inkling 2 yet: its '
        "checkpoint stores the weights under Inkling's own names with the MoE experts packed in NVFP4 / MXFP8, so the "
        "merge cannot place the LoRA weights and would write the base model unchanged.\n"
        'Save the LoRA adapter instead (`model.save_pretrained(...)` or `save_method = "lora"`): it reloads with '
        "Unsloth, and with transformers through `peft.PeftModel.from_pretrained`."
    )


INKLING2_TP_REASON = (
    "its attention projections are wrapper modules (q / k / v / o hold the Linear beside scaled L2 clips, short "
    "convolutions over the key / value channels and a sigmoid output gate), next to per-head attention sink and "
    "hidden-state-conditioned relative position bias projections, and tensor parallel has no sharding rule for them"
)
INKLING2_PP_REASON = (
    "its decoder carries a stack of block states (block attention residuals: every layer and the final norm read "
    "learned queries over the block outputs) beside hidden_states, and Inkling-2-Mini also runs layers 6-17 three "
    "times per forward with message gates feeding pass k-1 into pass k, so a pipeline stage boundary would cut "
    "state the pipeline does not carry"
)
INKLING2_PARALLEL_ALTERNATIVE = (
    'train data parallel instead: UNSLOTH_PARALLEL="dp=N" (DDP, the whole model on every GPU) or '
    'UNSLOTH_PARALLEL="fsdp=1" (native FSDP2) under torchrun, or split the model over the GPUs of one process '
    'with device_map = "balanced"'
)


def raise_if_inkling2_parallel_unsupported(config, model_types = ()) -> None:
    """Tensor / pipeline parallel (UNSLOTH_TP_SIZE / UNSLOTH_PP_SIZE, which UNSLOTH_PARALLEL tp= / pp= and
    unsloth.parallelize set) on Inkling 2, refused from the config before any weight loads. Left alone, tensor
    parallel stops at the first wrapper projection and pipeline parallel at the residual modules, both after a full
    load on every rank when quantizing on load (the Mini checkpoint is 14.4 GB in 4-bit)."""
    types = {str(t).lower() for t in (model_types or ())}
    if not (is_inkling2_config(config) or types & INKLING2_MODEL_TYPES):
        return
    import os

    def size(name):
        try:
            return int(str(os.environ.get(name) or "1").strip())
        except ValueError:
            return 1  # parallel_env reports the malformed value itself

    tp, pp = size("UNSLOTH_TP_SIZE"), size("UNSLOTH_PP_SIZE")
    asked = [
        (f"tensor parallel (UNSLOTH_TP_SIZE={tp})", INKLING2_TP_REASON) if tp > 1 else None,
        (f"pipeline parallel (UNSLOTH_PP_SIZE={pp})", INKLING2_PP_REASON) if pp > 1 else None,
    ]
    asked = [x for x in asked if x is not None]
    if not asked:
        return
    raise NotImplementedError(
        "Unsloth: "
        + "; ".join(f"{what} does not support Inkling 2 yet: {why}" for what, why in asked)
        + ". Expert parallel is not supported either. Unset "
        + " / ".join(what.split("(")[1].split("=")[0] for what, _ in asked)
        + " (or drop tp / pp from UNSLOTH_PARALLEL) and "
        + INKLING2_PARALLEL_ALTERNATIVE
        + "."
    )


def inkling2_owns_modelopt_quantization(config) -> bool:
    """True for an Inkling 2 config whose ModelOpt quantization_config transformers decodes in its own mapping."""
    if not is_inkling2_config(config):
        return False
    quant = _get(config, "quantization_config")
    if quant is not None and not isinstance(quant, dict):
        quant = quant.to_dict() if hasattr(quant, "to_dict") else vars(quant)
    return isinstance(quant, dict) and str(quant.get("quant_method", "")).lower() == "modelopt"


def drop_inkling2_quantization_config(
    config,
    verbose = True,
    load_in_4bit = False,
) -> None:
    """Remove the released checkpoint's ModelOpt marker from the config (and its text_config)."""
    for cfg in (config, _get(config, "text_config")):
        if cfg is None or _get(cfg, "quantization_config") is None:
            continue
        try:
            delattr(cfg, "quantization_config")
        except Exception:
            pass
        if _get(cfg, "quantization_config") is not None:
            try:
                cfg.quantization_config = None
            except Exception:
                pass
    if verbose:
        print(
            "Unsloth: Inkling 2 stores its MoE experts in NVFP4 / MXFP8; transformers dequantizes them while "
            "loading" + (", then bitsandbytes quantizes them to 4bit." if load_in_4bit else ".")
        )


def _is_expert_transform(transform) -> bool:
    patterns = getattr(transform, "_original_source_patterns", None) or ()
    return any(str(p).startswith(r"mlp\.experts\.w") for p in patterns)


def _has_dequant_op(transform) -> bool:
    return any(
        type(op).__name__ == "Inkling2LoopDequantizeExperts"
        for op in getattr(transform, "operations", ())
    )


def register_inkling2_expert_dequantization() -> bool:
    """Load the released NVFP4 / MXFP8 experts of ``inkling2`` checkpoints the way transformers loads ``inkling2_loop``.

    Upstream builds ``inkling2_loop_text`` as its two dequantizing expert converters plus ``inkling2_text`` without
    its expert transforms; this registers ``inkling2_text`` the same way. Returns False (and changes nothing) when
    either mapping no longer has that shape."""
    try:
        from transformers import conversion_mapping as cm
    except Exception:
        return False
    get = getattr(cm, "get_checkpoint_conversion_mapping", None)
    register = getattr(cm, "register_checkpoint_conversion_mapping", None)
    if get is None or register is None:
        return False
    try:
        loop_text, text = get("inkling2_loop_text"), get("inkling2_text")
    except Exception:
        return False
    if not loop_text or not text:
        return False
    dequant = [t for t in loop_text if _is_expert_transform(t)]
    if len(dequant) != 2 or not all(_has_dequant_op(t) for t in dequant):
        return False
    if any(_has_dequant_op(t) for t in text):
        return True  # already registered
    rest = [t for t in text if not _is_expert_transform(t)]
    if len(rest) == len(text):
        return False  # upstream no longer maps the experts here: nothing to replace
    register("inkling2_text", dequant + rest, overwrite = True)
    return True


def inkling2_lora_target_modules(
    model,
    target_modules,
    attention = True,
    mlp = True,
    vision = False,
    language = True,
):
    """Exact module names for the usual LoRA aliases. Unknown names pass through as PEFT suffixes."""
    if isinstance(target_modules, str) and target_modules != "all-linear":
        return target_modules
    if target_modules is None or target_modules == "all-linear":
        wanted = list(_DEFAULT)
    else:
        wanted = [str(x) for x in target_modules]

    language_on = language
    language, vision_layers = [], []
    for name, _ in model.named_modules():
        if re.search(r"(?:^|\.)vision_tower\.layers\.\d+$", name):
            vision_layers.append(name)
        elif (
            re.search(r"(?:^|\.)layers\.\d+$", name)
            and "vision_tower" not in name
            and ".mtp" not in name
        ):
            language.append(name)
    modules = dict(model.named_modules())

    out, passthrough = [], []
    for alias in wanted:
        known = False
        if alias in _ATTENTION_TARGETS:
            known = True
            if attention and language_on:
                for layer in language:
                    for leaf in _ATTENTION_TARGETS[alias]:
                        if f"{layer}.{leaf}" in modules:
                            out.append(f"{layer}.{leaf}")
        if alias in _EXPERT_ALIASES:
            known = True
        if vision and alias in _VISION_ATTENTION and attention:
            known = True
            for layer in vision_layers:
                name = f"{layer}.self_attn.{_VISION_ATTENTION[alias]}"
                if name in modules:
                    out.append(name)
        if vision and alias in _VISION_MLP and mlp:
            known = True
            for layer in vision_layers:
                name = f"{layer}.mlp.{_VISION_MLP[alias]}"
                if name in modules:
                    out.append(name)
        if alias in ("out_proj", "fc1", "fc2"):
            known = True
        if not known:
            passthrough.append(alias)
    result = []
    for name in out + passthrough:
        if name not in result:
            result.append(name)
    return result


def _unsloth_lora_target_modules(
    self,
    target_modules,
    attention = True,
    mlp = True,
    vision = False,
    language = True,
):
    return inkling2_lora_target_modules(
        self, target_modules, attention = attention, mlp = mlp, vision = vision, language = language
    )


def _message_input_math(query, eps, scaling, block_states, partial_block_states, loop_layer_idx):
    """`Inkling2LoopMessageAttentionResidual.forward` (transformers 5.24) as plain tensor ops: the eager path the module
    takes, op for op, so values and gradients are the module's bit for bit."""
    import torch

    stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim = -2).float()
    stacked = stacked * torch.rsqrt(
        stacked.square().mean(-1, keepdim = True) + eps
    )  # unweighted RMSNorm, fp32
    query_states = query[loop_layer_idx].float().expand(*stacked.shape[:-2], -1, -1)
    attn_weights = torch.matmul(query_states, stacked.transpose(2, 3)) * scaling
    attn_weights = torch.nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float32).to(
        query_states.dtype
    )
    attn_output = torch.matmul(attn_weights, stacked).transpose(1, 2).contiguous()
    return attn_output.transpose(1, 2)


def _checkpoint_use_reentrant(text_model):
    """The ``use_reentrant`` the text model's layers checkpoint with (transformers keeps it on the partial it installs
    as ``_gradient_checkpointing_func``; 5.24 defaults to False). Unknown: torch's default, True."""
    import functools

    function = getattr(text_model, "_gradient_checkpointing_func", None)
    while isinstance(function, functools.partial):
        if "use_reentrant" in function.keywords:
            return bool(function.keywords["use_reentrant"])
        function = function.func
    return True


def _weights_offloaded(module):
    """True when accelerate keeps the module's weights elsewhere until its hook runs (CPU / disk offload: the
    parameter sits on ``meta``) or the hook executes on another device than the parameter's."""
    query = getattr(module, "query", None)
    if query is None or query.device.type == "meta":
        return True
    hook = getattr(module, "_hf_hook", None)
    for h in getattr(hook, "hooks", None) or ([hook] if hook is not None else []):
        if getattr(h, "offload", False):
            return True
        execution_device = getattr(h, "execution_device", None)
        if execution_device is not None:
            try:
                import torch
                if torch.device(execution_device) != query.device:
                    return True
            except Exception:
                return True
    return False


def _checkpointed_message_input(text_model, message_input, forward):
    """Inkling-2-Mini's loop message residual runs in the text model's forward, between the checkpointed decoder
    layers, so everything its fp32 attention saves for backward stays on the GPU for the whole step: about 50 MB per
    call at 1024 tokens, 24 calls (12 loop layers x 2 later passes), so 1.2 GB. Under gradient checkpointing run it
    checkpointed: only its inputs (block states the layers keep anyway) are held and backward recomputes the small
    attention, with the same values and gradients.

    The recompute is the module's math as plain tensor ops (_message_input_math), not a module call: recomputing
    through the module (its accelerate device hook and compiled-module wrapper) from the autograd engine gave
    run-to-run different first-step gradients on a model split over two GPUs, where the engine runs each device's
    backward on its own thread. The checkpoint is reentrant or not as the model's layers are (``autograd.grad`` needs
    non-reentrant), and a module whose weights accelerate offloads keeps its hooked forward (the hook loads them).
    UNSLOTH_INKLING2_MESSAGE_CHECKPOINT=0 keeps the stock residual."""
    import os
    import weakref

    import torch
    from torch.utils.checkpoint import checkpoint

    owner = weakref.ref(text_model)
    module = weakref.ref(message_input)

    def message_input_forward(*args, **kwargs):
        tm, mi = owner(), module()
        if (
            tm is None
            or mi is None
            or kwargs
            or len(args) != 3
            or not getattr(tm, "gradient_checkpointing", False)
            or not tm.training
            or not torch.is_grad_enabled()
            or torch.compiler.is_compiling()
            or getattr(getattr(mi, "config", None), "_attn_implementation", "eager") != "eager"
            or os.environ.get("UNSLOTH_INKLING2_MESSAGE_CHECKPOINT", "1") == "0"
            or _weights_offloaded(mi)
        ):
            return forward(*args, **kwargs)
        block_states, partial_block_states, loop_layer_idx = args
        device = mi.query.device
        # Where accelerate's hook would put the inputs (a split model); autograd follows the moves.
        block_states = block_states.to(device)
        partial_block_states = partial_block_states.to(device)
        scaling = block_states.shape[-1] ** -0.5
        return checkpoint(
            _message_input_math,
            mi.query,
            float(mi.norm.eps),
            scaling,
            block_states,
            partial_block_states,
            int(loop_layer_idx),
            use_reentrant = _checkpoint_use_reentrant(tm),
            preserve_rng_state = False,
        )

    message_input_forward._unsloth_checkpointed_message_input = True
    return message_input_forward


def checkpoint_inkling2_message_inputs(model) -> int:
    """Wrap every loop text model's `message_input` call (see _checkpointed_message_input). Returns the count."""
    n = 0
    for module in model.modules():
        message_input = getattr(module, "message_input", None)
        if message_input is None or not type(message_input).__name__.endswith(
            "LoopMessageAttentionResidual"
        ):
            continue
        if getattr(message_input.forward, "_unsloth_checkpointed_message_input", False):
            continue
        # The instance forward (accelerate's device hook when the model is split) stays inside the checkpoint.
        message_input.forward = _checkpointed_message_input(
            module, message_input, message_input.forward
        )
        n += 1
    return n


def patch_inkling2_model(model) -> bool:
    """Give a loaded Inkling 2 model Unsloth's LoRA target mapping. Returns True when the model is Inkling 2."""
    config = getattr(model, "config", None)
    if not is_inkling2_config(config):
        return False
    cls = type(model)
    if getattr(cls, "unsloth_lora_target_modules", None) is not _unsloth_lora_target_modules:
        cls.unsloth_lora_target_modules = _unsloth_lora_target_modules
    checkpoint_inkling2_message_inputs(model)
    return True
