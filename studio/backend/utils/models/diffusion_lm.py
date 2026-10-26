# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Text-diffusion language models (LLaDA, Dream, SDAR, DiffusionGemma, ...), detected from config.json alone.

Torch-free mirror of the profile matching in ``unsloth/models/diffusion_profiles.py``; same names, same
order (a profile's ``model_types`` and ``architectures`` are both tried before the next profile, and
LLaDA-MoE precedes LLaDA because both report ``model_type = "llada"``). ``test_diffusion_lm_studio.py``
keeps the two in sync. Not to be confused with Studio's image / video diffusion pipelines.
"""

from pathlib import Path
from typing import Mapping, Optional

from loggers import get_logger

logger = get_logger(__name__)

# (profile name, model_types, architectures), in core registry order.
DIFFUSION_LM_PROFILES: tuple = (
    ("llada2", ("llada2_moe",), ("LLaDA2MoeModelLM",)),
    ("sdar", ("sdar",), ("SDARForCausalLM",)),
    (
        "diffusion_gemma",
        ("diffusion_gemma", "diffusion_gemma4"),
        (
            "DiffusionGemmaForBlockDiffusion",
            "DiffusionGemma4ModelForBlockDiffusion",
            "DiffusionGemma4ForBlockDiffusion",
        ),
    ),
    ("llada_moe", (), ("LLaDAMoEModel", "LLaDAMoEModelLM")),
    ("llada", ("llada",), ("LLaDAModelLM",)),
    ("dream", ("Dream", "dream"), ("DreamModel",)),
    ("nemotron_labs_diffusion", ("nemotron_labs_diffusion",), ("NemotronLabsDiffusionModel",)),
)

# Tied encoder / decoder base weights with separate adapters: merging would add both deltas to both.
NO_MERGE_PROFILES = frozenset({"diffusion_gemma"})

# Rough training-memory factor over an autoregressive model of the same size: LLaDA2 / SDAR attend over
# [noisy ; clean] (twice the length), DiffusionGemma runs a self-conditioning pass plus the encoder.
ACTIVATION_FACTOR = {"llada2": 2.0, "sdar": 2.0, "diffusion_gemma": 2.0}


def diffusion_lm_profile(config: Optional[Mapping]) -> Optional[str]:
    """Profile name for a raw config.json dict, or None for any other model."""
    if not isinstance(config, Mapping):
        return None
    model_type = config.get("model_type")
    architectures = config.get("architectures") or ()
    if isinstance(architectures, str):
        architectures = (architectures,)
    for name, model_types, archs in DIFFUSION_LM_PROFILES:
        if model_type in model_types or any(a in archs for a in architectures):
            return name
    return None


def _local_adapter_base(model_name: str) -> Optional[str]:
    try:
        import json

        path = Path(model_name).expanduser() / "adapter_config.json"
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding = "utf-8-sig")).get("base_model_name_or_path")
    except Exception:
        return None


def _hub_adapter_base(
    model_name: str,
    hf_token = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    try:
        from utils.paths import is_local_path

        if is_local_path(model_name):
            return None
        import json
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(model_name, "adapter_config.json", token = hf_token, revision = revision)
        return json.loads(Path(path).read_text(encoding = "utf-8-sig")).get("base_model_name_or_path")
    except Exception:
        return None


def diffusion_lm_profile_for_model(
    model_name: str,
    hf_token: Optional[str] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    _depth: int = 0,
) -> Optional[str]:
    """Profile name for a model id / local path (a local LoRA folder resolves to its base). Never raises:
    an unreadable config means "not detected here", and the loaded model is checked again after load."""
    try:
        from utils.models.model_config import _read_raw_config_json
        from utils.paths import is_local_path, normalize_path

        if is_local_path(model_name):
            local = Path(normalize_path(model_name)).expanduser()
            if not (local / "config.json").is_file():
                base = _local_adapter_base(str(local))
                if not base or _depth >= 2 or base == model_name:
                    return None
                return diffusion_lm_profile_for_model(
                    base, hf_token, local_files_only, None, _depth = _depth + 1
                )
        return diffusion_lm_profile(
            _read_raw_config_json(model_name, hf_token, local_files_only, revision)
        )
    except Exception as exc:
        logger.debug("Diffusion LM detection skipped for '%s': %s", model_name, exc)
        return None


def mlx_refusal(
    model_name: str,
    hf_token = None,
    revision: Optional[str] = None,
) -> Optional[str]:
    """Why the MLX backend cannot take this model, or None: it has no denoising objective or sampler, so
    it would train / decode a text-diffusion LM as an autoregressive one."""
    profile = diffusion_lm_profile_for_model(model_name, hf_token, revision = revision)
    if not profile:
        # A Hub LoRA has no config.json; off MLX the post-load check catches it, here nothing loads first.
        base = _hub_adapter_base(model_name, hf_token, revision)
        profile = base and diffusion_lm_profile_for_model(base, hf_token)
    if not profile:
        return None
    return (
        f"Text-diffusion models ({profile}) are not supported by the MLX backend on Apple Silicon yet: "
        "it has no denoising objective or sampler. Use an NVIDIA or AMD GPU for this model."
    )


def loaded_diffusion_profile(model) -> Optional[str]:
    """Profile name of a model loaded through Unsloth's diffusion loader (``_unsloth_slow_diffusion``), else
    None. Resolved from the model's config with the core registry, then this mirror; PEFT wrappers forward
    both attributes."""
    if model is None or not getattr(model, "_unsloth_slow_diffusion", False):
        return None
    config = getattr(model, "config", None)
    try:
        from unsloth.models.diffusion_profiles import resolve_diffusion_profile
        profile = resolve_diffusion_profile(config)
        if profile is not None:
            return profile.name
    except Exception as exc:
        logger.debug("Core diffusion profile lookup failed: %s", exc)
    return diffusion_lm_profile(_config_dict(config)) or "unknown"


def _config_dict(config) -> Optional[dict]:
    if config is None:
        return None
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
    }


GGUF_REFUSAL = (
    "GGUF export is not available for text-diffusion models ({profile}): llama.cpp would serve them as "
    "ordinary left-to-right models. Export the LoRA adapter instead."
)
MERGE_REFUSAL = (
    "Merged export is not available for DiffusionGemma: its encoder and decoder share base weights but "
    "train separate adapters, so merging would add both deltas to both. Export the LoRA adapter instead."
)
# The diffusion loader keeps the plain transformers model, which has no Unsloth merged-save path.
MERGE_REFUSAL_GENERIC = (
    "Merged export is not available for text-diffusion models ({profile}) yet. Export the LoRA adapter "
    "instead; it loads on top of the base model with Unsloth or PEFT."
)


def export_refusal(profile: Optional[str], method: str) -> Optional[str]:
    """Why ``method`` ("gguf", "merged", "lora", "base") cannot export a ``profile`` model; None if it can."""
    if not profile:
        return None
    if method == "gguf":
        return GGUF_REFUSAL.format(profile = profile)
    if method == "merged":
        if profile in NO_MERGE_PROFILES:
            return MERGE_REFUSAL
        return MERGE_REFUSAL_GENERIC.format(profile = profile)
    return None


def diffusion_lm_export_limits(checkpoint_path: str, hf_token = None) -> Optional[dict]:
    """``{profile, disabled_methods, reason}`` for a text-diffusion checkpoint folder, else None. Cache-only:
    the base model was read when it trained, and a picker probe must not reach the network. ``hf_token``
    is the caller's (False = denied the ambient credential), so the cache guard sees the same boundary."""
    profile = diffusion_lm_profile_for_model(checkpoint_path, hf_token, local_files_only = True)
    if not profile:
        return None
    disabled = [m for m in ("merged", "gguf") if export_refusal(profile, m)]
    return {
        "profile": profile,
        "disabled_methods": disabled,
        "reason": " ".join(export_refusal(profile, m) for m in disabled),
    }


def require_diffusion_api(name: str):
    """A core-Unsloth diffusion helper, or a clear error naming the Unsloth version needed."""
    import unsloth

    obj = getattr(unsloth, name, None)
    if obj is None:
        try:
            from unsloth.models import diffusion as _diffusion
            obj = getattr(_diffusion, name, None)
        except Exception:
            obj = None
    if obj is None:
        raise RuntimeError(
            f"Unsloth: this Unsloth Studio build trains text-diffusion models but the installed Unsloth has no "
            f"`{name}`. Update with `unsloth studio update` (or `pip install -U unsloth unsloth_zoo`)."
        )
    return obj
