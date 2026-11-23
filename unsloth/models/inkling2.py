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
    "inkling2_pipeline_adapter",
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

    tp = size("UNSLOTH_TP_SIZE")
    asked = [
        (f"tensor parallel (UNSLOTH_TP_SIZE={tp})", INKLING2_TP_REASON) if tp > 1 else None,
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


def _message_input_math(query, norm, scaling, block_states, partial_block_states, loop_layer_idx):
    """`Inkling2LoopMessageAttentionResidual.forward` (transformers 5.24) as plain tensor ops: the eager path the module
    takes, op for op, so values and gradients are the module's bit for bit. ``norm`` is the stock unweighted RMSNorm's
    eps (inlined, no module call in the recompute), or the residual's own `norm` module when it has hooks or a forward
    other than the stock one (see _message_norm), so those apply to the original forward and the recomputation alike."""
    import torch

    stacked = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim = -2).float()
    if isinstance(norm, float):
        stacked = stacked * torch.rsqrt(
            stacked.square().mean(-1, keepdim = True) + norm
        )  # unweighted RMSNorm, fp32
    else:
        stacked = norm(stacked)
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


def _message_norm(norm):
    """What _message_input_math normalizes with: the stock unweighted RMSNorm (no hooks, no instance or class forward
    override, no parameters) as its eps, inlined so the recompute stays plain tensor ops; anything else as the module,
    called like transformers does so its hooks and forward apply."""
    from torch.nn.modules import module as torch_module

    stock = type(norm).__name__ in (
        "Inkling2LoopUnweightedRMSNorm",
        "Inkling2UnweightedRMSNorm",
    ) and getattr(type(norm).forward, "__module__", "").startswith("transformers.")
    if (
        stock
        and "forward" not in norm.__dict__
        and not norm._forward_hooks
        and not norm._forward_pre_hooks
        and not torch_module._global_forward_hooks
        and not torch_module._global_forward_pre_hooks
        and not any(True for _ in norm.parameters())
        and isinstance(getattr(norm, "eps", None), (int, float))
    ):
        return float(norm.eps)
    return norm


def _weak_or_dead(obj):
    import weakref
    return (lambda: None) if obj is None else weakref.ref(obj)


class _CheckpointedMessageInput:
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
    UNSLOTH_INKLING2_MESSAGE_CHECKPOINT=0 keeps the stock residual.

    Installed as the residual's instance ``forward``. ``copy.deepcopy`` of the residual (or the model) binds the copy to
    the copied residual and owner, and the residual pickles: the residual and owner are weak references at run time,
    remapped through the deepcopy memo and pickled as strong references."""

    _unsloth_checkpointed_message_input = True

    def __init__(self, text_model, message_input, forward):
        from torch.utils.checkpoint import checkpoint

        self._checkpoint = checkpoint  # bound at install, as the layers bind theirs at enable
        self._owner = _weak_or_dead(text_model)
        self._module = _weak_or_dead(message_input)
        # The stock bound forward is kept as its importable function, anything else (accelerate's device hook) as is.
        if getattr(forward, "__self__", None) is message_input and hasattr(forward, "__func__"):
            self._function, self._forward = forward.__func__, None
        else:
            self._function, self._forward = None, forward

    def _call_original(self, mi, *args, **kwargs):
        if self._forward is not None:
            return self._forward(*args, **kwargs)
        return self._function(mi, *args, **kwargs)

    def __deepcopy__(self, memo):
        import copy

        new = object.__new__(type(self))
        memo[id(self)] = new
        owner, module = self._owner(), self._module()
        new._owner = _weak_or_dead(memo.get(id(owner), owner))
        new._module = _weak_or_dead(memo.get(id(module), module))
        new._function = self._function
        new._checkpoint = self._checkpoint
        new._forward = copy.deepcopy(self._forward, memo)
        return new

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_owner"], state["_module"] = self._owner(), self._module()
        return state

    def __setstate__(self, state):
        state = dict(state)
        self._owner = _weak_or_dead(state.pop("_owner"))
        self._module = _weak_or_dead(state.pop("_module"))
        self.__dict__.update(state)

    def __call__(self, *args, **kwargs):
        import os

        import torch

        tm, mi = self._owner(), self._module()
        if mi is None:
            raise RuntimeError(
                "Unsloth: the Inkling 2 message residual this forward belongs to was freed."
            )
        if (
            tm is None
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
            return self._call_original(mi, *args, **kwargs)
        block_states, partial_block_states, loop_layer_idx = args
        device = mi.query.device
        # Where accelerate's hook would put the inputs (a split model); autograd follows the moves.
        block_states = block_states.to(device)
        partial_block_states = partial_block_states.to(device)
        scaling = block_states.shape[-1] ** -0.5
        return self._checkpoint(
            _message_input_math,
            mi.query,
            _message_norm(mi.norm),
            scaling,
            block_states,
            partial_block_states,
            int(loop_layer_idx),
            use_reentrant = _checkpoint_use_reentrant(tm),
            preserve_rng_state = False,
        )


def _checkpointed_message_input(text_model, message_input, forward):
    """See _CheckpointedMessageInput."""
    return _CheckpointedMessageInput(text_model, message_input, forward)


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


# ------------------------------------------------------------------------------------------- pipeline parallel


class Inkling2PipelineAdapter:
    """How pipeline parallelism (models/pp_native.py) splits an Inkling 2 text model.

    Every decoder layer reads ``block_states`` (block attention residuals: the inputs of the block boundary layers,
    ``layer_idx % n_layers_per_attn_res_block == 0``) beside ``hidden_states``, the final norm reads them through
    ``final_input``, and Inkling-2-Mini runs layers ``[loop_layer_index_start, loop_layer_index_end)`` several times,
    each later pass restarting from the pre-loop states with messages (``message_input``) from the previous pass.
    So a stage boundary carries two tensors, ``hidden_states`` [batch, seq, hidden] and ``block_states``
    [batch, seq, blocks, hidden] (the block count is fixed by the cut), the looped layers stay on one stage (a
    stage starts at or before the loop start or at or after the loop end), ``final_input`` lives with the last
    stage and ``message_input`` with the stage holding the loop. The text model's forward runs only this stage's
    layers (``stage_forward``, the transformers 5.24 forward restricted to ``[start, end)``)."""

    def __init__(self, decoder):
        self.config = decoder.config
        config = self.config
        self.num_layers = int(config.num_hidden_layers)
        self.block = int(getattr(config, "n_layers_per_attn_res_block", 0) or 0)
        start = getattr(config, "loop_layer_index_start", None)
        end = getattr(config, "loop_layer_index_end", None)
        passes = getattr(config, "loop_num_passes", None)
        self.loop = None
        if hasattr(decoder, "message_input") and None not in (start, end, passes):
            self.loop = (int(start), int(end), int(passes))
        # Decoder children beside embed / layers / norm, with the layer whose stage owns them (None = the last).
        self.extra_modules = {"final_input": None}
        if self.loop is not None:
            self.extra_modules["message_input"] = self.loop[0]

    def cut_points(self):
        """Layer indices where a stage may start: anywhere but strictly inside the looped layers."""
        points = range(1, self.num_layers)
        if self.loop is None:
            return list(points)
        start, end, _ = self.loop
        return [i for i in points if not start < i < end]

    def blocks_before(self, cut):
        """Entries of block_states when the stage starting at ``cut`` begins (the boundary layers before it)."""
        return sum(1 for i in range(min(cut, self.num_layers)) if i % self.block == 0)

    def carry(self, cut):
        """Tensors crossing the boundary before layer ``cut``: [(trailing shape, dtype)], dtype None = the activation
        dtype (both stay in the model's hidden dtype in transformers' forward)."""
        hidden = int(self.config.hidden_size)
        return [((hidden,), None), ((self.blocks_before(cut), hidden), None)]

    def owner_layer(self, name):
        return self.extra_modules.get(name)

    def install(self, decoder, ctx, stage_output):
        install_inkling2_stage_forward(decoder, ctx, stage_output)


def inkling2_pipeline_adapter(decoder):
    """An Inkling2PipelineAdapter for an Inkling 2 text model whose forward is the one stage_forward mirrors, else None
    (not Inkling 2). Raises ValueError for an Inkling 2 text model whose layout it does not know."""
    if decoder is None or not type(decoder).__name__.startswith("Inkling2"):
        return None
    if not is_inkling2_config(getattr(decoder, "config", None)):
        return None
    config = decoder.config
    layers = getattr(decoder, "layers", None)
    missing = [
        name
        for name in ("embed_tokens", "layers", "norm", "final_input")
        if getattr(decoder, name, None) is None
    ]
    block = getattr(config, "n_layers_per_attn_res_block", None)
    if (
        missing
        or not isinstance(block, int)
        or block < 1
        or len(layers) != config.num_hidden_layers
    ):
        raise ValueError(
            f"Unsloth: pipeline parallelism does not know this Inkling 2 text model layout ({type(decoder).__name__}"
            f"{', missing ' + str(missing) if missing else ''}); it mirrors transformers 5.24's."
        )
    adapter = Inkling2PipelineAdapter(decoder)
    if adapter.loop is None and getattr(config, "loop_layer_index_start", None) is not None:
        raise ValueError(
            f"Unsloth: pipeline parallelism found Inkling 2 loop settings without a message_input module in "
            f"{type(decoder).__name__}; this layout is not supported."
        )
    if adapter.loop is not None:
        start, end, passes = adapter.loop
        if not (0 < start < end <= adapter.num_layers) or passes < 1:
            raise ValueError(
                f"Unsloth: pipeline parallelism does not support the Inkling 2 loop [{start}, {end}) x {passes}."
            )
    return adapter


def _modeling(decoder):
    import sys
    return sys.modules[type(decoder).__module__]


def inkling2_stage_layers(decoder, start, end, hidden_states, block_states, layer_kwargs):
    """Decoder layers ``[start, end)`` of an Inkling 2 text model on (hidden_states, block_states), as transformers
    5.24's text forward runs them (Mini: the looped layers' passes and messages when the loop starts here)."""
    import torch

    config = decoder.config
    layers = decoder.layers
    loop = None
    if (
        hasattr(decoder, "message_input")
        and getattr(config, "loop_layer_index_start", None) is not None
    ):
        loop = (config.loop_layer_index_start, config.loop_layer_index_end, config.loop_num_passes)

    def run(layer, h, b, *extra, **more):
        new = layer(h, b, *extra, **layer_kwargs[layer.attention_type], **more)
        if layer.is_block_boundary:
            b = torch.cat([b.to(h.device), h.unsqueeze(-2)], dim = -2)
        return new, b

    h, b = hidden_states, block_states
    i = start
    while i < end:
        if loop is not None and loop[0] < i < loop[1]:
            raise RuntimeError(
                f"Unsloth: an Inkling 2 pipeline stage starts at layer {i}, inside the looped layers "
                f"[{loop[0]}, {loop[1]})."
            )
        if loop is None or i != loop[0]:
            h, b = run(layers[i], h, b)
            i += 1
            continue
        loop_start, loop_end, passes = loop
        if end < loop_end:
            raise RuntimeError(
                f"Unsloth: an Inkling 2 pipeline stage ends at layer {end}, inside the looped layers "
                f"[{loop_start}, {loop_end})."
            )
        num_loop_layers = loop_end - loop_start
        pre_h, pre_b = h, b
        num_pre_loop_blocks = pre_b.shape[-2]
        for layer in layers[loop_start:loop_end]:
            h, b = run(layer, h, b)
        for loop_idx in range(1, passes):
            previous_b = b[..., num_pre_loop_blocks + 1 :, :]
            previous_h = h
            h, b = pre_h, pre_b
            cache_offset = config.num_hidden_layers + (loop_idx - 1) * num_loop_layers - loop_start
            for loop_layer_idx, layer in enumerate(layers[loop_start:loop_end]):
                messages = decoder.message_input(previous_b, previous_h, loop_layer_idx)
                h, b = run(layer, h, b, messages, cache_offset = cache_offset)
        i = loop_end
    return h, b


# Model-level flags the text forward's output recorder consumes; never handed to the layers.
_STAGE_DROPPED_KWARGS = (
    "output_attentions",
    "output_hidden_states",
    "output_router_logits",
    "return_dict",
)


def _stage_text_forward(
    decoder, ctx, stage_output, input_ids, attention_mask, position_ids, inputs_embeds, kwargs
):
    import torch

    modeling = _modeling(decoder)
    if inputs_embeds is None:
        inputs_embeds = decoder.embed_tokens(input_ids)
    if position_ids is None:
        position_ids = torch.arange(inputs_embeds.shape[1], device = inputs_embeds.device).unsqueeze(
            0
        )
    for key in _STAGE_DROPPED_KWARGS:
        kwargs.pop(key, None)
    if ctx.is_first:
        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(
            *inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1]
        )
    else:
        hidden_states, block_states = ctx.stage_input
    mapping = attention_mask
    if not isinstance(mapping, dict):
        mask_kwargs = {
            "config": decoder.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache": None,
            "position_ids": position_ids,
        }
        mapping = {
            "full_attention": modeling.create_causal_mask(**mask_kwargs),
            "sliding_attention": modeling.create_sliding_window_causal_mask(**mask_kwargs),
            "linear_attention": modeling.create_recurrent_attention_mask(**mask_kwargs),
        }
    distance = {
        "full_attention": modeling.get_relative_distance(inputs_embeds, None, is_sliding = False),
        "sliding_attention": modeling.get_relative_distance(inputs_embeds, None, is_sliding = True),
    }
    layer_kwargs = {
        kind: dict(
            attention_mask = mapping[kind],
            relative_distance = distance[kind],
            conv_mask = mapping["linear_attention"],
            cache = None,
            position_ids = position_ids,
            **kwargs,
        )
        for kind in ("full_attention", "sliding_attention")
    }
    hidden_states, block_states = inkling2_stage_layers(
        decoder, ctx.start, ctx.end, hidden_states, block_states, layer_kwargs
    )
    if not ctx.is_last:
        # The layers' short convolutions hand out time-major (transposed) outputs; the wire needs dense rows.
        hidden_states, block_states = hidden_states.contiguous(), block_states.contiguous()
        if torch.is_grad_enabled() and not block_states.requires_grad:
            # Only frozen embeddings in it (a cut before the second block): a leaf, so the next stage's gradient has
            # somewhere to go.
            block_states = block_states.detach().requires_grad_()
        raise stage_output((hidden_states, block_states))
    hidden_states = decoder.final_input(block_states, hidden_states)[0]
    hidden_states = decoder.norm(hidden_states).to(inputs_embeds.dtype)
    return modeling.MoeModelOutputWithPast(last_hidden_state = hidden_states, cache = None)


def install_inkling2_stage_forward(decoder, ctx, stage_output):
    """Run only this pipeline stage's layers in the text model's forward while ``ctx`` is active (a pipeline step);
    the stock forward otherwise (a replica put back for generate / exports). ``stage_output`` is the exception a
    stage before the last raises with (hidden_states, block_states)."""
    import functools

    original = decoder.__dict__.get("forward")
    stock = type(decoder).forward

    def stage(
        self,
        input_ids = None,
        attention_mask = None,
        position_ids = None,
        cache = None,
        inputs_embeds = None,
        use_cache = None,
        **kwargs,
    ):
        return _stage_text_forward(
            self, ctx, stage_output, input_ids, attention_mask, position_ids, inputs_embeds, kwargs
        )

    try:  # FlexAttention keeps a per-forward block mask cache (inkling2_flex)
        from .inkling2_flex import _wrap_text_forward
        stage = _wrap_text_forward(stage)
    except Exception:
        pass

    @functools.wraps(stock)
    def forward(*args, **kwargs):
        if not ctx.active:
            if original is not None:
                return original(*args, **kwargs)
            return stock(decoder, *args, **kwargs)
        if args:
            raise RuntimeError(
                "Unsloth: the Inkling 2 pipeline stage forward takes keyword arguments only "
                f"(got {len(args)} positional)."
            )
        if (
            kwargs.get("cache") is not None
            or kwargs.get("past_key_values") is not None
            or kwargs.get("use_cache")
        ):
            raise RuntimeError(
                "Unsloth: the Inkling 2 pipeline stage forward trains without a KV cache."
            )
        kwargs.pop("past_key_values", None)
        return stage(decoder, **kwargs)

    forward._unsloth_inkling2_pipeline_stage = True
    decoder.forward = forward
    return forward
