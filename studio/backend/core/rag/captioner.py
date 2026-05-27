# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Figure captioning for RAG ingestion.

Two captioning sources, tried in order:

1. The user's currently-loaded chat VLM — when ``vlm_url`` / ``vlm_model``
   are provided by the parent process (it probes ``llama_cpp.is_vision``
   at enqueue time). Captions go through that model's OpenAI-compatible
   ``/v1/chat/completions`` endpoint as base64 ``image_url``.

2. A helper llama-server fallback that loads the pre-cached
   ``unsloth/gemma-4-E2B-it-GGUF`` (gemma-3n family, multimodal) with
   its mmproj for vision. Spawned for the lifetime of a
   ``caption_images`` call, unloaded before return so no llama-server
   process leaks past ingestion.

Defensive: any per-image failure returns an empty string; total
captioner unavailability (no chat VLM + helper load failure) returns
empty strings for every image. The caller (``_stream_image_chunks``)
falls back to the parser's page-text caption in that case.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROMPT = (
    "Describe this figure in <=60 words. Focus on factual content "
    "(axes, labels, captions, visible text, main objects). "
    "Do not speculate beyond what is visible."
)
_MAX_NEW_TOKENS = 120
# Downscale large images so the base64 payload stays manageable; the chat
# model's prefill cost scales with image-tile count, not pixel count, but
# very large inputs still bloat the JSON body. 1600 px on the long side
# matches PR #5351's chat-composer extractor.
_MAX_IMAGE_SIZE = 1600
_REQUEST_TIMEOUT_SECONDS = 120.0

# Helper VLM (used when no vision-capable chat model is loaded).
# Matches the model pre-cached by precache_helper_gguf() at studio
# startup so the captioner doesn't have to wait on a fresh download.
_HELPER_REPO = "unsloth/gemma-4-E2B-it-GGUF"
_HELPER_VARIANT = "UD-Q4_K_XL"
_HELPER_MODEL_NAME = "helper"


def _image_to_data_url(blob: bytes) -> str:
    from PIL import Image

    img = Image.open(BytesIO(blob)).convert("RGB")
    if max(img.size) > _MAX_IMAGE_SIZE:
        img.thumbnail((_MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE))
    buf = BytesIO()
    img.save(buf, format = "JPEG", quality = 88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _load_helper_vlm() -> Optional[tuple[Any, str, str]]:
    """Spawn a private LlamaCppBackend with the helper VLM + mmproj.

    Returns ``(backend, base_url, model_name)`` on success, ``None`` on
    failure. The caller is responsible for unloading the backend when
    done (so the helper doesn't outlive the ingestion subprocess).
    """
    try:
        from core.inference.llama_cpp import LlamaCppBackend

        # kill_orphans=False is critical: the global singleton is
        # already running the user's chat-model llama-server. Killing
        # "orphans" here would reap that healthy chat process because
        # the orphan-killer can't tell two LlamaCppBackend instances
        # apart by PID ownership.
        backend = LlamaCppBackend(kill_orphans = False)
        logger.info(
            "RAG captioner: loading helper VLM %s (%s) as fallback",
            _HELPER_REPO,
            _HELPER_VARIANT,
        )
        ok = backend.load_model(
            hf_repo = _HELPER_REPO,
            hf_variant = _HELPER_VARIANT,
            model_identifier = f"rag-captioner:{_HELPER_REPO}:{_HELPER_VARIANT}",
            is_vision = True,
            n_ctx = 4096,
            n_gpu_layers = -1,
        )
        if not ok:
            logger.warning("RAG captioner: helper VLM failed to start")
            return None
        return backend, backend.base_url, _HELPER_MODEL_NAME
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG captioner: helper VLM load raised: %s", exc)
        return None


def _post_one(client: Any, endpoint: str, model: str, blob: bytes) -> str:
    """POST one image to the OpenAI-compatible endpoint, return caption."""
    data_url = _image_to_data_url(blob)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    },
                ],
            }
        ],
        "max_tokens": _MAX_NEW_TOKENS,
        "temperature": 0.0,
    }
    response = client.post(endpoint, json = payload)
    response.raise_for_status()
    data = response.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return content.strip() if isinstance(content, str) else ""


def caption_images(
    image_bytes_list: list[bytes],
    *,
    vlm_url: Optional[str] = None,
    vlm_model: Optional[str] = None,
) -> list[str]:
    """Generate one short caption per image; same-length output.

    Tries the loaded chat VLM first (``vlm_url`` + ``vlm_model``). If
    those are missing, spawns the helper VLM, captions, and unloads it
    before returning. On any failure returns ``""`` for the affected
    image. Never raises.
    """
    if not image_bytes_list:
        return []

    import httpx

    helper_backend: Optional[Any] = None
    try:
        # Resolve endpoint + model: chat VLM if available, else helper.
        if vlm_url and vlm_model:
            endpoint = f"{vlm_url.rstrip('/')}/v1/chat/completions"
            model_name = vlm_model
        else:
            loaded = _load_helper_vlm()
            if loaded is None:
                # No chat VLM and helper failed → all empty strings;
                # caller falls back to page-text captions.
                return ["" for _ in image_bytes_list]
            helper_backend, helper_base_url, helper_model_name = loaded
            endpoint = f"{helper_base_url.rstrip('/')}/v1/chat/completions"
            model_name = helper_model_name

        out: list[str] = []
        with httpx.Client(timeout = _REQUEST_TIMEOUT_SECONDS) as client:
            for blob in image_bytes_list:
                try:
                    out.append(_post_one(client, endpoint, model_name, blob))
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "caption_images: per-image request to %s failed: %s",
                        endpoint,
                        exc,
                    )
                    out.append("")
        return out
    finally:
        # Always tear down the helper if we spawned one. Chat VLM (when
        # provided by the parent) is left alone — it's not ours to manage.
        if helper_backend is not None:
            try:
                helper_backend.unload_model()
                logger.info("RAG captioner: helper VLM unloaded")
            except Exception as exc:  # noqa: BLE001
                logger.warning("RAG captioner: helper unload failed: %s", exc)
