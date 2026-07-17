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


def hidream_te4_kwargs(dtype: Any, hf_token: Optional[str] = None) -> dict[str, Any]:
    """``{text_encoder_4, tokenizer_4}`` kwargs for a HiDream pipeline ``from_pretrained``.

    Loaded eagerly (~16 GB bf16) before the pipeline call so a failure surfaces as a
    clear error instead of a half-built pipeline."""
    import torch  # noqa: F401 -- dtype values are torch dtypes; import keeps parity with callers
    from transformers import AutoTokenizer, LlamaForCausalLM

    logger.info("diffusion.hidream: loading Llama TE4 from %s", HIDREAM_LLAMA_REPO)
    tokenizer_4 = AutoTokenizer.from_pretrained(HIDREAM_LLAMA_REPO, token = hf_token)
    text_encoder_4 = LlamaForCausalLM.from_pretrained(
        HIDREAM_LLAMA_REPO,
        output_hidden_states = True,
        output_attentions = True,
        torch_dtype = dtype,
        token = hf_token,
    )
    return {"text_encoder_4": text_encoder_4, "tokenizer_4": tokenizer_4}
