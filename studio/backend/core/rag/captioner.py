# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vision-model helpers for ingestion: figure captioning and scanned-page OCR.

Both turn pixels into text the normal FTS5 + dense path can index. Each is a no-op
(never raises) without a loaded vision model or on failure: captioning is gated by
``config.CAPTION_IMAGES``, OCR by ``config.OCR_SCANNED``."""

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

_OCR_PROMPT = (
    "Transcribe all text on this document page exactly as it appears, in reading "
    "order. Output only the transcribed text, with no commentary, labels, or code "
    "fences. Preserve headings, lists, and line breaks. If the page has no readable "
    "text, output nothing."
)


def vision_endpoint() -> tuple[str, str] | None:
    """``(base_url, model)`` for a loaded vision GGUF model, else None."""
    try:
        from routes.inference import get_llama_cpp_backend
        backend = get_llama_cpp_backend()
        if getattr(backend, "is_loaded", False) and getattr(backend, "is_vision", False):
            return backend.base_url, "local"
    except Exception:  # noqa: BLE001 - never let discovery break ingestion
        return None
    return None


def _vision_complete(
    base_url: str, model: str, image_bytes: bytes, *, prompt: str, timeout: float, max_tokens: int
) -> str | None:
    """One image-in / text-out call to the loaded vision model's OpenAI-compatible
    endpoint. Returns the stripped text or ``None`` on empty/failure (non-fatal)."""
    import httpx

    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
        # Off: thinking models would spend the budget reasoning, returning "".
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = httpx.post(f"{base_url}/v1/chat/completions", json = payload, timeout = timeout)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip() or None
    except Exception:  # noqa: BLE001 - a failed vision call is non-fatal
        logger.debug("vision request failed", exc_info = True)
        return None


def _caption_one(base_url: str, model: str, image_bytes: bytes, timeout: float) -> str | None:
    return _vision_complete(
        base_url, model, image_bytes, prompt = _CAPTION_PROMPT, timeout = timeout, max_tokens = 200
    )


def _ocr_one(base_url: str, model: str, image_bytes: bytes, timeout: float) -> str | None:
    return _vision_complete(
        base_url,
        model,
        image_bytes,
        prompt = _OCR_PROMPT,
        timeout = timeout,
        max_tokens = config.OCR_MAX_TOKENS,
    )


def caption_images(
    images: list, *, endpoint: tuple[str, str] | None = None
) -> dict[int, list[str]]:
    """Caption ``ParsedImage`` objects, keyed by 1-based page number; ``{}`` when
    disabled, no vision model, or no images. Bounded by ``CAPTION_MAX_IMAGES``."""
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


def ocr_pages(
    page_pngs: dict[int, bytes], *, endpoint: tuple[str, str] | None = None
) -> dict[int, str]:
    """OCR rendered page PNGs (keyed by 1-based page number) to text via the loaded
    vision model. Returns ``{}`` when disabled, no vision model, or no pages.
    Bounded by ``OCR_MAX_PAGES``."""
    if not config.OCR_SCANNED or not page_pngs:
        return {}
    ep = endpoint or vision_endpoint()
    if ep is None:
        return {}
    base_url, model = ep

    out: dict[int, str] = {}
    for page_num in sorted(page_pngs)[: config.OCR_MAX_PAGES]:
        text = _ocr_one(base_url, model, page_pngs[page_num], config.OCR_TIMEOUT_S)
        if text:
            out[int(page_num)] = text
    return out


def splice_captions(pages: list, captions: dict[int, list[str]]) -> list:
    """Append captions to their page's text so the chunker indexes them, keeping
    figures attributable in retrieved chunks. Returns new ``Page`` objects."""
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
