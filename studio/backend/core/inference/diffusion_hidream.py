# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HiDream-I1 Llama text-encoder assembly.

The HiDream-ai/HiDream-I1-* repos name ``text_encoder_4`` (LlamaForCausalLM) and
``tokenizer_4`` in their model_index but do NOT ship the weights: the official example
loads meta-llama/Meta-Llama-3.1-8B-Instruct separately and passes both components into
``HiDreamImagePipeline.from_pretrained``. That upstream repo is Hub-gated (manual
approval), so Studio loads the open unsloth mirror instead -- byte-identical weights,
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


def hidream_te4_kwargs(
    dtype: Any,
    hf_token: Optional[str] = None,
    *,
    fam: Any = None,
    te_quant_mode: Optional[str] = None,
    target: Any = None,
) -> dict[str, Any]:
    """``{text_encoder_4, tokenizer_4}`` kwargs for a HiDream pipeline ``from_pretrained``.

    Loaded eagerly (~16 GB bf16) before the pipeline call so a failure surfaces as a
    clear error instead of a half-built pipeline.

    The generic ``quantize_text_encoders`` pass only covers ``text_encoder``..``_3``, so
    TE4 -- HiDream's HEAVIEST encoder -- is handled here: when the requested TE quant is
    layerwise fp8 (and the device/family qualify, same gates as the runtime cast), TE4 is
    fp8-cast too, preferring the hosted pre-cast checkpoint (~half the download) and
    falling back to dense-load-then-cast. Any other mode keeps today's dense bf16 TE4."""
    import torch  # noqa: F401 -- dtype values are torch dtypes; import keeps parity with callers
    from transformers import AutoTokenizer, LlamaForCausalLM

    tokenizer_4 = AutoTokenizer.from_pretrained(HIDREAM_LLAMA_REPO, token = hf_token)

    fp8_engages = False
    if target is not None:
        try:
            from . import diffusion_precision as precision
            from .diffusion_precision import (
                TE_QUANT_FP8,
                normalize_te_quant,
                te_quant_supported,
            )

            mode = normalize_te_quant(te_quant_mode)
            denied = getattr(precision, "_te_family_denied", None)
            fp8_engages = (
                mode == TE_QUANT_FP8
                and te_quant_supported(target, mode)
                and not (callable(denied) and denied(getattr(fam, "name", None), mode))
            )
        except Exception:  # noqa: BLE001 -- quant probe failure keeps the dense bf16 path
            fp8_engages = False

    if fp8_engages and fam is not None:
        from .diffusion_te_prequant import (
            load_prequant_text_encoder,
            resolve_te_prequant_source,
        )

        source = resolve_te_prequant_source(fam, "text_encoder_4", "fp8")
        if source is not None:
            encoder = load_prequant_text_encoder(
                HIDREAM_LLAMA_REPO,
                "text_encoder_4",
                source,
                dtype = dtype,
                hf_token = hf_token,
                scheme = "fp8",
                logger = logger,
                # The Llama TE4 lives in its own standalone repo (config at the root), and
                # the pipeline needs hidden states/attentions from its forward.
                config_subfolder = "",
                config_overrides = {
                    "output_hidden_states": True,
                    "output_attentions": True,
                },
            )
            if encoder is not None:
                return {"text_encoder_4": encoder, "tokenizer_4": tokenizer_4}

    logger.info("diffusion.hidream: loading Llama TE4 from %s", HIDREAM_LLAMA_REPO)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        HIDREAM_LLAMA_REPO,
        output_hidden_states = True,
        output_attentions = True,
        torch_dtype = dtype,
        token = hf_token,
    )
    if fp8_engages:
        try:
            from .diffusion_precision import _cast_fp8

            class _Target:
                pass

            cast_target = _Target()
            cast_target.dtype = dtype
            _cast_fp8(text_encoder_4, cast_target)
            logger.info("diffusion.hidream: TE4 layerwise fp8 cast engaged")
        except Exception as exc:  # noqa: BLE001 -- best-effort like the generic TE pass
            # A mid-pass failure can leave fp8 storage / upcast hooks behind; a half-cast
            # encoder cannot run as dense, so rebuild it fresh instead of shipping partial state.
            logger.warning("diffusion.hidream: TE4 fp8 cast failed, reloading dense: %s", exc)
            text_encoder_4 = LlamaForCausalLM.from_pretrained(
                HIDREAM_LLAMA_REPO,
                output_hidden_states = True,
                output_attentions = True,
                torch_dtype = dtype,
                token = hf_token,
            )
    return {"text_encoder_4": text_encoder_4, "tokenizer_4": tokenizer_4}
