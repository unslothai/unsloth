# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Absorbing-mask diffusion LMs over one bidirectional sequence: LLaDA, LLaDA-MoE, Dream, Nemotron-Labs-Diffusion.

Each response token is replaced by [MASK] with probability p, the clean prompt stays visible, and the
loss is cross-entropy at the masked positions weighted by 1 / p. The families differ only in the knobs
below, each taken from that family's published training code.
"""

import functools
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .diffusion_profiles import (
    DiffusionProfile,
    register_diffusion_profile,
    response_mask,
)

__all__ = ["MaskedDiffusionProfile"]


def _ensure_post_init(cls):
    """4.x remote code that never calls post_init() loads on transformers 5 only once it does."""
    if getattr(cls, "_unsloth_post_init_guard", False):
        return cls
    original_init = cls.__init__

    @functools.wraps(original_init)
    def __init__(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "all_tied_weights_keys") and hasattr(self, "post_init"):
            self.post_init()

    cls.__init__ = __init__
    cls._unsloth_post_init_guard = True
    return cls


def _accept_new_validate_kwargs(cls):
    """Dream's GenerationConfig.validate(self, is_init) predates the keywords transformers 5 passes it."""
    import inspect
    import sys

    from transformers import GenerationConfig

    module = sys.modules.get(cls.__module__)
    for value in list(vars(module).values()) if module is not None else ():
        if (
            not (isinstance(value, type) and issubclass(value, GenerationConfig))
            or value is GenerationConfig
        ):
            continue
        validate = value.__dict__.get("validate")
        if validate is None or getattr(validate, "_unsloth_tolerant", False):
            continue
        parameters = inspect.signature(validate).parameters
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
            continue
        accepted = set(parameters)

        @functools.wraps(validate)
        def tolerant(
            self,
            *args,
            _validate = validate,
            _accepted = accepted,
            **kwargs,
        ):
            return _validate(self, *args, **{k: v for k, v in kwargs.items() if k in _accepted})

        tolerant._unsloth_tolerant = True
        value.validate = tolerant


def _default_diffusion_generate_config(model):
    """Dream's diffusion_generate() without a generation_config rebuilds one through
    GenerationConfig.from_model_config, which transformers 5 breaks on Dream-only fields (eps)."""
    generate = getattr(model, "diffusion_generate", None)
    if generate is None or getattr(generate, "_unsloth_default_config", False):
        return

    @functools.wraps(generate)
    def diffusion_generate(
        inputs = None,
        generation_config = None,
        **kwargs,
    ):
        if generation_config is None:
            import copy
            generation_config = copy.deepcopy(model.generation_config)
            # from_model_config used to fill these from the model config; Dream ships mask_token_id null.
            for name in ("mask_token_id", "eos_token_id", "pad_token_id", "bos_token_id"):
                if getattr(generation_config, name, None) is None:
                    setattr(generation_config, name, getattr(model.config, name, None))
        return generate(inputs, generation_config = generation_config, **kwargs)

    diffusion_generate._unsloth_default_config = True
    model.diffusion_generate = diffusion_generate


def _expose_mask_signatures():
    """unsloth_zoo wraps the transformers mask builders as (*args, **kwargs); remote code that filters its
    arguments by inspect.signature (Nemotron-Labs-Diffusion's causal prefill) then passes nothing."""
    import inspect

    try:
        import transformers.masking_utils as masking_utils
    except ImportError:
        return
    for name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        function = getattr(masking_utils, name, None)
        if (
            function is None
            or "__signature__" in vars(function)
            or hasattr(function, "__wrapped__")
        ):
            continue
        for cell_name, cell in zip(function.__code__.co_freevars, function.__closure__ or ()):
            if cell_name == "signature" and isinstance(cell.cell_contents, inspect.Signature):
                function.__signature__ = cell.cell_contents
                break


def _bridge_native_checkpointing(model):
    """LLaDA checkpoints through its own set_activation_checkpointing, not the transformers API."""
    inner = getattr(model, "model", None)
    if getattr(model, "supports_gradient_checkpointing", False) or not hasattr(
        inner, "set_activation_checkpointing"
    ):
        return

    def gradient_checkpointing_enable(gradient_checkpointing_kwargs = None, **kwargs):
        inner.set_activation_checkpointing("whole_layer")

    def gradient_checkpointing_disable():
        inner.set_activation_checkpointing(None)

    model.gradient_checkpointing_enable = gradient_checkpointing_enable
    model.gradient_checkpointing_disable = gradient_checkpointing_disable
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads = lambda: None


@dataclass
class MaskedDiffusionProfile(DiffusionProfile):
    # Dream is adapted from an AR model: position i's logits predict token i + 1 (Dream src/trainer).
    shift_logits: bool = False
    # "answer_length": LLaDA GUIDELINES.md SFT, each example / its answer length, mean over the batch.
    # "masked_tokens": Dream / Nemotron, summed then divided by the number of masked tokens.
    loss_normalization: str = "answer_length"
    # LLaDA and Dream train on EOS-filled, attended padding so the model learns where answers stop.
    pad_with_eos: bool = True
    auto_class: str = "AutoModelForCausalLM"
    default_mask_token_id: int = None

    def __post_init__(self):
        self.noise = "mask"
        self.defaults = {
            "diffusion_eps": 1e-3,
            "diffusion_time_weighting": "inverse_t",
            **self.defaults,
        }

    def model_class(self, config, trust_remote_code):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        auto_map = getattr(config, "auto_map", None) or {}
        class_ref = (
            auto_map.get(self.auto_class)
            or auto_map.get("AutoModelForCausalLM")
            or auto_map.get("AutoModel")
        )
        if class_ref is None or not trust_remote_code:
            return super().model_class(config, trust_remote_code)
        cls = get_class_from_dynamic_module(
            class_ref,
            config._name_or_path,
            revision = getattr(config, "_commit_hash", None),
        )
        _accept_new_validate_kwargs(cls)
        return _ensure_post_init(cls)

    def prepare_model(self, model, tokenizer):
        config = model.config
        # transformers 5 dropped the PretrainedConfig.use_cache default that LLaDA's forward reads.
        if getattr(config, "use_cache", None) is None:
            config.use_cache = False
        mask_id = getattr(config, "mask_token_id", None)
        if mask_id is None and tokenizer is not None:
            mask_id = getattr(getattr(tokenizer, "tokenizer", tokenizer), "mask_token_id", None)
        if mask_id is None:
            mask_id = self.default_mask_token_id
        eos_id = getattr(config, "eos_token_id", None)
        if isinstance(eos_id, (list, tuple)):
            eos_id = eos_id[0] if eos_id else None
        if eos_id is None and tokenizer is not None:
            eos_id = getattr(getattr(tokenizer, "tokenizer", tokenizer), "eos_token_id", None)
        model._unsloth_mask_token_id = mask_id
        model._unsloth_eos_token_id = eos_id
        _bridge_native_checkpointing(model)
        _default_diffusion_generate_config(model)
        _expose_mask_signatures()
        return model

    def _token_ids(self, model, args):
        mask_id = self.option(args, "diffusion_mask_token_id")
        if mask_id is None:
            mask_id = getattr(model, "_unsloth_mask_token_id", None)
        if mask_id is None:
            mask_id = getattr(model.config, "mask_token_id", None) or self.default_mask_token_id
        if mask_id is None:
            raise ValueError(
                f"Unsloth: {self.name} needs a [MASK] token id. Set DiffusionConfig(diffusion_mask_token_id = ...)."
            )
        eos_id = getattr(model, "_unsloth_eos_token_id", None)
        if eos_id is None:
            eos_id = getattr(model.config, "eos_token_id", None)
            if isinstance(eos_id, (list, tuple)):
                eos_id = eos_id[0]
        return mask_id, eos_id

    def compute_loss(
        self,
        model,
        inputs,
        args,
        num_items_in_batch = None,
    ):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        batch, length = input_ids.shape
        device = input_ids.device
        mask_id, eos_id = self._token_ids(model, args)
        eps = float(self.option(args, "diffusion_eps"))
        weighting = self.option(args, "diffusion_time_weighting")

        maskable = response_mask(inputs)
        forward_mask = attention_mask
        if self.pad_with_eos and attention_mask is not None and eos_id is not None:
            padding = ~attention_mask.bool()
            input_ids = input_ids.masked_fill(padding, eos_id)
            if inputs.get("labels") is not None:
                maskable = maskable | padding
            forward_mask = None

        # Same draw order as the references (t, then the per-token draw) so a seed reproduces their batch.
        t = torch.rand(batch, device = device)
        p_mask = ((1 - eps) * t + eps)[:, None].expand(batch, length)
        masked = (torch.rand((batch, length), device = device) < p_mask) & maskable
        noisy = torch.where(masked, mask_id, input_ids)

        model_inputs = {"input_ids": noisy}
        if forward_mask is not None and not bool(forward_mask.all()):
            model_inputs["attention_mask"] = forward_mask
        logits = model(**model_inputs).logits
        if self.shift_logits:
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim = 1)

        if not bool(masked.any()):
            zero = logits.sum() * 0.0
            return zero, None, {"masked_fraction": 0.0}

        token_loss = F.cross_entropy(logits[masked].float(), input_ids[masked], reduction = "none")
        if weighting == "inverse_t":
            token_loss = token_loss / p_mask[masked]
        elif weighting not in ("none", None):
            raise ValueError(
                f"Unsloth: {self.name} does not support diffusion_time_weighting={weighting!r}."
            )

        if self.loss_normalization == "answer_length":
            answer_length = maskable.sum(dim = 1, keepdim = True).clamp_min(1).expand(batch, length)
            loss = (token_loss / answer_length[masked]).sum() / batch
        else:
            loss = token_loss.sum() / masked.sum()

        metrics = {
            "masked_fraction": (masked.sum() / maskable.sum().clamp_min(1)).item(),
            "masked_ce": F.cross_entropy(logits[masked].float(), input_ids[masked]).item(),
        }
        return loss, None, metrics


_ATTENTION_AND_MLP = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# LLaDA-MoE shares model_type "llada" with the dense model, so it is matched first by architecture.
register_diffusion_profile(
    MaskedDiffusionProfile(
        name = "llada_moe",
        architectures = ("LLaDAMoEModel", "LLaDAMoEModelLM"),
        requires_remote_code = True,
        # Every layer is MoE: adapt attention only, never the router or the experts.
        lora_target_modules = r".*\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|o_proj)",
        default_mask_token_id = 156895,  # model card generate(mask_id = 156895); absent from config.json
    )
)
register_diffusion_profile(
    MaskedDiffusionProfile(
        name = "llada",
        model_types = ("llada",),
        architectures = ("LLaDAModelLM",),
        requires_remote_code = True,
        # OLMo-style names; anchored to blocks so the output head (transformer.ff_out) stays frozen.
        lora_target_modules = r".*\.blocks\.\d+\.(q_proj|k_proj|v_proj|attn_out|ff_proj|up_proj|ff_out)",
        quant_skip_modules = ["transformer.ff_out", "transformer.wte", "lm_head"],
        default_mask_token_id = 126336,
    )
)
register_diffusion_profile(
    MaskedDiffusionProfile(
        name = "dream",
        model_types = ("Dream", "dream"),
        architectures = ("DreamModel",),
        requires_remote_code = True,
        lora_target_modules = list(_ATTENTION_AND_MLP),
        shift_logits = True,
        loss_normalization = "masked_tokens",
        auto_class = "AutoModel",
        # Dream q_sample draws t ~ U(0, 1) with no floor and weights by 1 / t (time_reweighting "original").
        defaults = {"diffusion_eps": 0.0},
    )
)
register_diffusion_profile(
    MaskedDiffusionProfile(
        name = "nemotron_labs_diffusion",
        model_types = ("nemotron_labs_diffusion",),
        architectures = ("NemotronLabsDiffusionModel",),
        requires_remote_code = True,
        lora_target_modules = list(_ATTENTION_AND_MLP),
        loss_normalization = "masked_tokens",
        auto_class = "AutoModel",
    )
)
