# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""Per-family training recipes for text-diffusion language models.

A profile names the checkpoints it handles (``model_types`` / ``architectures``), how they load, which
LoRA targets are safe, and the training objective. ``DiffusionTrainer`` calls ``profile.compute_loss``
on ordinary SFT batches (``input_ids``, ``attention_mask``, ``labels`` with -100 outside the response).
"""

from dataclasses import dataclass, field
from typing import Optional

import torch

__all__ = [
    "DiffusionProfile",
    "register_diffusion_profile",
    "resolve_diffusion_profile",
    "diffusion_profiles",
    "diffusion_model_types",
    "response_mask",
    "sample_noise_level",
    "restore_rotary_buffers",
]


@dataclass
class DiffusionProfile:
    name: str
    model_types: tuple = ()
    architectures: tuple = ()
    # "mask" = absorbing [MASK] noise (LLaDA, Dream, Nemotron); "uniform" = random-token noise (DiffusionGemma).
    noise: str = "mask"
    requires_remote_code: bool = False
    lora_target_modules: object = field(
        default_factory = lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    lora_exclude_modules: Optional[str] = None
    # bitsandbytes keeps these in full precision; None = the shared default list.
    quant_skip_modules: Optional[list] = None
    # Profile-specific defaults a DiffusionConfig field left at None falls back to.
    defaults: dict = field(default_factory = dict)

    def matches(self, config) -> bool:
        model_type = getattr(config, "model_type", None)
        if model_type in self.model_types:
            return True
        archs = getattr(config, "architectures", None) or ()
        return any(arch in self.architectures for arch in archs)

    def model_class(self, config, trust_remote_code):
        """The class ``from_pretrained`` is called on. Remote-code families use the auto class."""
        from transformers import AutoModel, AutoModelForCausalLM

        auto_map = getattr(config, "auto_map", None) or {}
        if "AutoModelForCausalLM" in auto_map or not auto_map:
            return AutoModelForCausalLM
        return AutoModel

    def prepare_model(self, model, tokenizer):
        """Post-load fixes (remote-code compat, mask-token bookkeeping). Must be idempotent."""
        return model

    def option(
        self,
        args,
        name,
        default = None,
    ):
        value = getattr(args, name, None) if args is not None else None
        if value is None:
            value = self.defaults.get(name, default)
        return value

    def compute_loss(
        self,
        model,
        inputs,
        args,
        num_items_in_batch = None,
    ):
        """Returns (loss, outputs_or_None, metrics: dict[str, float])."""
        raise NotImplementedError(
            f"Unsloth: diffusion profile {self.name} has no training objective."
        )


_PROFILES = []


def register_diffusion_profile(profile):
    for i, existing in enumerate(_PROFILES):
        if existing.name == profile.name:
            _PROFILES[i] = profile
            return profile
    _PROFILES.append(profile)
    return profile


def diffusion_profiles():
    _load_builtin_profiles()
    return tuple(_PROFILES)


def diffusion_model_types():
    return tuple(mt for p in diffusion_profiles() for mt in p.model_types)


def resolve_diffusion_profile(config):
    if config is None:
        return None
    for profile in diffusion_profiles():
        if profile.matches(config):
            return profile
    return None


_BUILTINS_LOADED = False


def _load_builtin_profiles():
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import importlib

    for module in ("diffusion_gemma_objective", "diffusion_masked", "diffusion_block"):
        try:
            importlib.import_module(f"{__package__}.{module}")
        except ModuleNotFoundError as e:
            if e.name != f"{__package__}.{module}":
                raise


def response_mask(inputs):
    """Bool [B, L]: supervised positions. No labels -> every attended token (pretraining-style)."""
    labels = inputs.get("labels")
    attention_mask = inputs.get("attention_mask")
    if labels is not None:
        mask = labels != -100
        if attention_mask is not None:
            mask = mask & attention_mask.bool()
        return mask
    if attention_mask is not None:
        return attention_mask.bool()
    return torch.ones_like(inputs["input_ids"], dtype = torch.bool)


def sample_noise_level(
    batch_size,
    eps,
    device,
    low = None,
    high = None,
):
    """Per-example t ~ U(low, high), default U(eps, 1 - eps) as in the published recipes."""
    low = eps if low is None else low
    high = 1.0 - eps if high is None else high
    return low + (high - low) * torch.rand(batch_size, device = device)


def restore_rotary_buffers(model):
    """4.x remote code keeps RoPE frequencies in a non-persistent buffer filled in __init__; transformers 5 builds
    on the meta device and only re-fills it through `_init_weights`, which these classes override, so `inv_freq`
    comes back zero (GPU) or uninitialised (CPU) and positions are silently ignored. Recompute it."""
    repaired = 0
    for module in model.modules():
        init_fn = getattr(module, "rope_init_fn", None)
        inv_freq = getattr(module, "inv_freq", None)
        config = getattr(module, "config", None) or getattr(model, "config", None)
        if (
            init_fn is None
            or config is None
            or not isinstance(inv_freq, torch.Tensor)
            or inv_freq.is_meta
        ):
            continue
        try:
            expected, scaling = init_fn(
                config, inv_freq.device, **getattr(module, "rope_kwargs", {})
            )
        except Exception:
            continue
        expected = expected.to(device = inv_freq.device, dtype = inv_freq.dtype)
        if expected.shape != inv_freq.shape or torch.equal(inv_freq, expected):
            continue
        with torch.no_grad():
            inv_freq.copy_(expected)
        if isinstance(getattr(module, "original_inv_freq", None), torch.Tensor):
            module.original_inv_freq = inv_freq.clone()
        if hasattr(module, "attention_scaling"):
            module.attention_scaling = scaling
        repaired += 1
    return repaired
