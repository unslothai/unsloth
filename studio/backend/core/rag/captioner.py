# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Figure captioning via the user's currently-loaded chat VLM.

No separate vision model is loaded. If the chat model is vision-capable
(detected at ingestion-enqueue time and passed in as ``vlm_url`` /
``vlm_model``), we call its OpenAI-compatible ``/v1/chat/completions``
with the figure as a base64 ``image_url``. If no vision-capable chat
model is loaded, captioning is skipped entirely — the ingestion falls
back to the parser's page-text ``nearest_caption``.

Defensive: any per-image request failure returns an empty string so
ingestion stays resilient.
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import Optional

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


def _image_to_data_url(blob: bytes) -> str:
    from PIL import Image

    img = Image.open(BytesIO(blob)).convert("RGB")
    if max(img.size) > _MAX_IMAGE_SIZE:
        img.thumbnail((_MAX_IMAGE_SIZE, _MAX_IMAGE_SIZE))
    buf = BytesIO()
    img.save(buf, format = "JPEG", quality = 88)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def caption_images(
    image_bytes_list: list[bytes],
    *,
    vlm_url: Optional[str] = None,
    vlm_model: Optional[str] = None,
) -> list[str]:
    """Caption each image via the loaded chat VLM.

    ``vlm_url`` / ``vlm_model`` come from the parent's chat-backend probe.
    When either is missing (no model loaded, or loaded model is text-only),
    returns an empty string per image so the caller falls back to its
    parser-provided caption. Never raises.
    """
    if not image_bytes_list:
        return []
    if not vlm_url or not vlm_model:
        return ["" for _ in image_bytes_list]

    import httpx

    endpoint = f"{vlm_url.rstrip('/')}/v1/chat/completions"
    out: list[str] = []
    with httpx.Client(timeout = _REQUEST_TIMEOUT_SECONDS) as client:
        for blob in image_bytes_list:
            try:
                data_url = _image_to_data_url(blob)
                payload = {
                    "model": vlm_model,
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
                content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                out.append(content.strip() if isinstance(content, str) else "")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "caption_images: per-image request to %s failed: %s",
                    endpoint,
                    exc,
                )
                out.append("")
    return out
