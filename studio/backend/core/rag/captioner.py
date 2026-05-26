# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Figure captioning via a small generative VLM (PaddleOCR-VL, 4-bit).

Defensive by design: any load or per-image failure returns an empty
string so the ingestion pipeline can fall back to its prior page-text
caption behaviour rather than crashing.

Lifecycle: lives inside the ingestion subprocess (`_subprocess_worker`
in `ingestion.py`). The model loads on first `caption_images` call and
is released when the subprocess exits — no impact on the parent chat
model.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)

# Pre-quantized Unsloth bnb-4bit repo (see unsloth/models/mapper.py).
# Using this name directly skips the FLOAT_TO_INT_MAPPER redirect so the
# initial download fetches the 4-bit weights (~1.5 GB) instead of bf16
# (~5 GB followed by on-the-fly quantization).
_CAPTION_MODEL_NAME = "unsloth/Qwen3-VL-2B-Instruct-unsloth-bnb-4bit"
_PROMPT = (
    "Describe this figure in <=60 words. Focus on factual content "
    "(axes, labels, captions, visible text, main objects). "
    "Do not speculate beyond what is visible."
)
_MAX_NEW_TOKENS = 120

_lock = threading.Lock()
_model: Any | None = None
_processor: Any | None = None
_load_failed: bool = False


def _load() -> tuple[Any, Any] | None:
    """Lazy-load PaddleOCR-VL. Returns (model, processor) or None on failure.

    Sentinel-cached: once a load failure occurs in this subprocess we
    don't keep retrying for every batch of images.
    """
    global _model, _processor, _load_failed
    if _load_failed:
        return None
    with _lock:
        if _model is not None and _processor is not None:
            return _model, _processor
        try:
            from unsloth import FastVisionModel

            logger.info(
                "Loading RAG captioner: %s (4-bit via Unsloth)",
                _CAPTION_MODEL_NAME,
            )
            # FastVisionModel handles 4-bit quantization, dtype selection,
            # and inference-mode wiring. Returns (model, processor) — for
            # vision models the "tokenizer" slot carries the processor.
            model, processor = FastVisionModel.from_pretrained(
                model_name = _CAPTION_MODEL_NAME,
                load_in_4bit = True,
                trust_remote_code = True,
            )
            FastVisionModel.for_inference(model)
            _model = model
            _processor = processor
            return _model, _processor
        except Exception as exc:
            logger.warning(
                "RAG captioner %s failed to load (%s). Falling back to "
                "page-text captions for this ingestion.",
                _CAPTION_MODEL_NAME,
                exc,
            )
            _load_failed = True
            return None


def caption_images(image_bytes_list: list[bytes]) -> list[str]:
    """Generate one short caption per image. Same-length output.

    Returns ``""`` for any image whose captioning failed (or all images
    if the model couldn't load). Never raises — ingestion must stay
    resilient to VLM unavailability.
    """
    if not image_bytes_list:
        return []
    loaded = _load()
    if loaded is None:
        return ["" for _ in image_bytes_list]

    model, processor = loaded
    from io import BytesIO

    import torch
    from PIL import Image

    out: list[str] = []
    for blob in image_bytes_list:
        try:
            image = Image.open(BytesIO(blob)).convert("RGB")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ]
            # Qwen3-VL's processor pipes image + text through the chat
            # template in one call when tokenize=True / return_dict=True.
            inputs = processor.apply_chat_template(
                messages,
                tokenize = True,
                add_generation_prompt = True,
                return_dict = True,
                return_tensors = "pt",
            )
            inputs = {
                k: (v.to(model.device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            input_ids_len = int(inputs["input_ids"].shape[1])
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens = _MAX_NEW_TOKENS,
                    do_sample = False,
                )
            # Decode only the newly-generated tokens, not the prompt echo.
            new_tokens = output_ids[:, input_ids_len:]
            caption = processor.batch_decode(
                new_tokens,
                skip_special_tokens = True,
            )[0]
            out.append(caption.strip())
        except Exception as exc:  # noqa: BLE001
            logger.warning("caption_images: skipping one image: %s", exc)
            out.append("")
    return out
