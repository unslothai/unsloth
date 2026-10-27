# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HiDream-I1 Llama text-encoder assembly.

The HiDream-ai/HiDream-I1-* repos name ``text_encoder_4`` (LlamaForCausalLM) and
``tokenizer_4`` in their model_index but do NOT ship the weights: the official example
loads meta-llama/Meta-Llama-3.1-8B-Instruct separately and passes both components into
``HiDreamImagePipeline.from_pretrained``. That upstream repo is Hub-gated (manual
approval), so Unsloth loads the open unsloth mirror instead -- byte-identical weights,
no license wall at load time, and the unsloth org is already inside the loader's
non-GGUF trust gate. ``output_hidden_states=True`` matches the official example: the
pipeline's prompt encoder consumes the Llama hidden states, not the logits.
"""

from __future__ import annotations

from typing import Any, Optional

from loggers import get_logger

logger = get_logger(__name__)

HIDREAM_FAMILY_NAME = "hidream-i1"

# Open mirror of the gated meta-llama/Meta-Llama-3.1-8B-Instruct the pipeline expects.
HIDREAM_LLAMA_REPO = "unsloth/Meta-Llama-3.1-8B-Instruct"
HIDREAM_LLAMA_BF16_BYTES = 16_060_556_376


def hidream_te4_kwargs(
    dtype: Any,
    hf_token: Optional[str] = None,
    *,
    fam: Any = None,
    te_quant_mode: Optional[str] = None,
    target: Any = None,
    local_files_only: bool = False,
) -> dict[str, Any]:
    """Returns {text_encoder_4, tokenizer_4} kwargs; TE4 is fp8-cast only under layerwise fp8; local_files_only raises instead of downloading."""
    import torch  # noqa: F401 -- dtype values are torch dtypes; import keeps parity with callers
    from transformers import AutoTokenizer, LlamaForCausalLM

    # Live hub root: an unpinned lookup after a cache-folder change misses the cached encoder.
    from utils.hf_cache_settings import active_hf_hub_cache

    cache_dir = active_hf_hub_cache()

    tokenizer_4 = AutoTokenizer.from_pretrained(
        HIDREAM_LLAMA_REPO,
        token = hf_token,
        local_files_only = local_files_only,
        cache_dir = cache_dir,
    )

    fp8_engages = False
    hosted_int8 = False
    if target is not None:
        try:
            from . import diffusion_precision as precision
            from .diffusion_precision import (
                TE_QUANT_FP8,
                normalize_te_quant,
                te_quant_supported,
            )
            from .diffusion_te_prequant import int8_convrot_te_runs_on

            mode = normalize_te_quant(te_quant_mode)
            denied = getattr(precision, "_te_family_denied", None)
            allowed = not (callable(denied) and denied(getattr(fam, "name", None), mode))
            # fp8 (the hosted fp8 file, else a dense load and cast) needs fp8 support; int8 takes the hosted int8
            # ConvRot file, which Apple Silicon runs as plain tensors where fp8 does not.
            fp8_engages = (
                mode in (TE_QUANT_FP8, "int8")
                and te_quant_supported(target, TE_QUANT_FP8)
                and allowed
            )
            hosted_int8 = mode == "int8" and allowed and int8_convrot_te_runs_on(target)
        except Exception:  # noqa: BLE001 -- quant probe failure keeps the dense bf16 path
            fp8_engages = hosted_int8 = False

    if (fp8_engages or hosted_int8) and fam is not None:
        from .diffusion_te_prequant import (
            load_prequant_text_encoder,
            te_prequant_sources_for_base,
        )

        source = te_prequant_sources_for_base(
            fam,
            HIDREAM_LLAMA_REPO,
            te_quant_mode = te_quant_mode,
            target = target,
            components = ("text_encoder_4",),
            standalone_component_bases = {"text_encoder_4": HIDREAM_LLAMA_REPO},
        ).get("text_encoder_4")
        if source is not None:
            scheme = "int8" if normalize_te_quant(te_quant_mode) == "int8" else "fp8"
            encoder = load_prequant_text_encoder(
                HIDREAM_LLAMA_REPO,
                "text_encoder_4",
                source,
                dtype = dtype,
                hf_token = hf_token,
                scheme = scheme,
                logger = logger,
                # The int8 file carries no lm_head; the pipeline reads hidden states only.
                trim_lm_head = scheme == "int8",
                # The Llama TE4 lives in its own standalone repo (config at the root), and the pipeline needs hidden
                # states/attentions from its forward.
                config_subfolder = "",
                config_overrides = {
                    "output_hidden_states": True,
                    "output_attentions": True,
                },
                local_files_only = local_files_only,
            )
            if encoder is not None:
                return {"text_encoder_4": encoder, "tokenizer_4": tokenizer_4}
        if hosted_int8 and not fp8_engages:
            from .diffusion_precision import te_quant_picked_automatically
            if not te_quant_picked_automatically(te_quant_mode):
                # Apple Silicon: the plan priced the hosted int8 TE4 and nothing else runs here, so the 16 GB dense
                # Llama is not loaded in its place (as _te_prequant_pipe_kwargs refuses for the other encoders).
                raise RuntimeError(
                    f"The pre-quantized text_encoder_4 for {HIDREAM_LLAMA_REPO} could not be loaded "
                    f"({'not available' if source is None else 'see the server log'}). The memory check sized "
                    "this load for it, so the larger dense encoder is not loaded in its place. Set Text encoder "
                    "precision to Dense (bf16) to load the dense encoder instead."
                )

    logger.info("diffusion.hidream: loading Llama TE4 from %s", HIDREAM_LLAMA_REPO)
    from .safetensors_pread import pread_safetensors

    # loaded before the pipeline's own scope: a 16 GB bf16 Llama converted on Windows reads through pread too
    with pread_safetensors(convert_to = dtype):
        text_encoder_4 = LlamaForCausalLM.from_pretrained(
            HIDREAM_LLAMA_REPO,
            output_hidden_states = True,
            output_attentions = True,
            torch_dtype = dtype,
            token = hf_token,
            local_files_only = local_files_only,
            cache_dir = cache_dir,
        )
    if fp8_engages:
        try:
            from .diffusion_precision import _cast_fp8

            class _Target:
                pass

            cast_target = _Target()
            cast_target.dtype = dtype
            _cast_fp8(text_encoder_4, cast_target)
            from .diffusion_precision import TE_RUNTIME_SCHEME_ATTR

            setattr(text_encoder_4, TE_RUNTIME_SCHEME_ATTR, "fp8")
            logger.info("diffusion.hidream: TE4 layerwise fp8 cast engaged")
        except Exception as exc:  # noqa: BLE001 -- best-effort like the generic TE pass
            # A half-cast encoder cannot run dense, so rebuild it fresh.
            logger.warning("diffusion.hidream: TE4 fp8 cast failed, reloading dense: %s", exc)
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                HIDREAM_LLAMA_REPO,
                output_hidden_states = True,
                output_attentions = True,
                torch_dtype = dtype,
                local_files_only = local_files_only,
                token = hf_token,
                cache_dir = cache_dir,
            )
    return {"text_encoder_4": text_encoder_4, "tokenizer_4": tokenizer_4}
