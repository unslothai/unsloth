# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Caption figures with the loaded vision model and splice the text into the
page, making images searchable via the normal FTS5 + dense path (no image
vector space).

No-op (never raises) without a vision model or on failure. Gated by
``config.CAPTION_IMAGES`` (off by default) since each caption is a model call.
"""

from __future__ import annotations

import base64
import logging

from . import config

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Describe this figure or image from a document in one or two concise "
    "sentences, for search indexing. State what it depicts (e.g. a diagram, "
    "chart, table or photo) and its key content. Do not add commentary."
)


def vision_endpoint() -> tuple[str, str] | None:
    """``(base_url, model)`` for a loaded vision GGUF model, else None.
    Lazy import so the RAG core needs no inference stack."""
    try:
        from routes.inference import get_llama_cpp_backend

        backend = get_llama_cpp_backend()
        if getattr(backend, "is_loaded", False) and getattr(
            backend, "is_vision", False
        ):
            return backend.base_url, "local"
    except Exception:  # noqa: BLE001 - never let discovery break ingestion
        return None
    return None


def _caption_one(
    base_url: str, model: str, image_bytes: bytes, timeout: float
) -> str | None:
    import httpx

    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _CAPTION_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 200,
        "temperature": 0.2,
        "stream": False,
        # Off: thinking models otherwise spend the budget reasoning, returning "".
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json = payload, timeout = timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip() or None
    except Exception:  # noqa: BLE001 - a failed caption is non-fatal
        logger.debug("caption request failed", exc_info = True)
        return None


def caption_images(
    images: list, *, endpoint: tuple[str, str] | None = None
) -> dict[int, list[str]]:
    """Caption ``ParsedImage`` objects, keyed by 1-based page number; ``{}`` when
    disabled, no vision model, or no images. ``endpoint`` is injectable for tests;
    bounded by ``CAPTION_MAX_IMAGES``."""
    if not config.CAPTION_IMAGES or not images:
        return {}
    ep = endpoint or vision_endpoint()
    if ep is None:
        return {}
    base_url, model = ep

    out: dict[int, list[str]] = {}
    for img in images[: config.CAPTION_MAX_IMAGES]:
        image_bytes = getattr(img, "image_bytes", None)
        if not image_bytes:
            continue
        caption = _caption_one(base_url, model, image_bytes, config.CAPTION_TIMEOUT_S)
        if caption:
            page = getattr(img, "page_number", None) or 0
            out.setdefault(int(page), []).append(caption)
    return out


def splice_captions(pages: list, captions: dict[int, list[str]]) -> list:
    """Append captions to their page's text so the chunker indexes them; the
    marker keeps figures attributable in retrieved chunks. Returns new ``Page``
    objects (uncaptioned pages unchanged)."""
    if not captions:
        return pages
    from .parsers import Page

    out: list = []
    for page in pages:
        caps = captions.get(page.page_number or 0)
        if not caps:
            out.append(page)
            continue
        extra = "".join(f"\n\n[Figure on page {page.page_number}: {c}]" for c in caps)
        text = page.text + extra
        out.append(Page(text = text, page_number = page.page_number, char_count = len(text)))
    return out
