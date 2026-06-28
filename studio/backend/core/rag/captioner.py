# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Vision-model helpers for ingestion: figure captioning and scanned-page OCR.

Both turn pixels into text the normal FTS5 + dense path can index. Each is a no-op
(never raises) without a loaded vision model or on failure: captioning is gated by
``config.CAPTION_IMAGES``, OCR by ``config.OCR_SCANNED``.

These reuse the loaded chat model's vision endpoint, so the model must be served
with a micro-batch large enough for one image's tokens (llama-server
``--ubatch-size`` >= image tokens): some vision encoders (e.g. Gemma-family) attend
non-causally over the whole image and abort the server otherwise. Studio's vision
chat already requires this, so OCR inherits it."""

from __future__ import annotations

import base64
import logging

from . import config

logger = logging.getLogger(__name__)

_CAPTION_PROMPT = (
    "Read this figure or image from a document for search indexing.\n"
    "First, on a line 'TEXT:', transcribe every piece of visible text exactly as "
    "written, in reading order: the title, axis labels and units, legend and series "
    "names, EVERY box / node / arrow label, table headers and cells, equations, and "
    "footnotes. List each distinct label even if it is small.\n"
    "Then, on a line 'SUMMARY:', add one or two sentences on what it shows (chart "
    "type and trend, diagram subject, table topic, or photo content).\n"
    "Report only what is visible. Transcribe exactly; do not invent or guess any "
    "text, label, or number."
)

_OCR_PROMPT = (
    "Transcribe all text on this document page exactly as it appears, in reading "
    "order, including any text inside figures, diagrams, charts, and tables (keep "
    "table rows readable). Output only the transcribed text, with no commentary or "
    "code fences. Preserve headings, lists, and line breaks. If the page has no "
    "readable text, output nothing."
)


def _collapse_runaway(
    text: str,
    max_repeat: int = 3,
    max_total: int = 8,
) -> str:
    """Bound pathological repetition: some vision models loop on sparse images and
    emit the same line many times, consecutively or interleaved with others. Cap each
    distinct non-empty line at ``max_repeat`` in a row and ``max_total`` overall, and
    collapse blank-line floods, so a degenerate page cannot flood the index.
    ``max_total`` is generous so legitimate repeats (e.g. a short table column) and
    short repeats survive; only egregious looping is trimmed."""
    out: list[str] = []
    seen: dict[str, int] = {}
    prev: str | None = None
    run = 0
    for line in text.splitlines():
        key = line.strip()
        if not key:
            if prev == "":  # collapse runs of blank lines to a single separator
                continue
            prev = ""
            out.append("")
            continue
        run = run + 1 if key == prev else 1
        prev = key
        seen[key] = seen.get(key, 0) + 1
        if run > max_repeat or seen[key] > max_total:
            continue
        out.append(line)
    return "\n".join(out)


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


def _vision_auth_headers() -> dict | None:
    """Authorization header for the loaded backend's HTTP API, or None. Vision calls
    hit the same llama-server endpoint as chat, so in direct-stream mode (the server
    is started with ``--api-key``) they need the same bearer or they 401. None when no
    key is set, so unauthenticated servers don't get a spurious header."""
    try:
        from routes.inference import get_llama_cpp_backend
        return get_llama_cpp_backend()._auth_headers or None
    except Exception:  # noqa: BLE001 - auth discovery must never break ingestion
        return None


def _vision_complete(
    base_url: str,
    model: str,
    image_bytes: bytes,
    *,
    prompt: str,
    timeout: float,
    max_tokens: int,
    temperature: float = 0.0,
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
        # Deterministic by default: transcription must not randomly drop labels.
        "temperature": temperature,
        "stream": False,
        # Off: thinking models would spend the budget reasoning, returning "".
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        r = httpx.post(
            f"{base_url}/v1/chat/completions",
            json = payload,
            timeout = timeout,
            headers = _vision_auth_headers(),
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        return text.strip() or None
    except Exception:  # noqa: BLE001 - a failed vision call is non-fatal
        logger.debug("vision request failed", exc_info = True)
        return None


def _caption_one(base_url: str, model: str, image_bytes: bytes, timeout: float) -> str | None:
    return _vision_complete(
        base_url,
        model,
        image_bytes,
        prompt = _CAPTION_PROMPT,
        timeout = timeout,
        max_tokens = config.CAPTION_MAX_TOKENS,
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
    """Caption ``ParsedImage`` objects, keyed by 1-based page number. Returns ``{}``
    when there are no images or no vision model is loaded. The caller
    (`ingestion._run`) owns the on/off policy (config default plus the per-upload
    toggle), so this does not re-check it. Bounded by ``CAPTION_MAX_IMAGES``; each
    caption is passed through ``_collapse_runaway`` so a looping vision model cannot
    flood the index."""
    if not images:
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
            out.setdefault(int(page), []).append(_collapse_runaway(caption))
    return out


def ocr_pages(
    page_pngs: dict[int, bytes], *, endpoint: tuple[str, str] | None = None
) -> dict[int, str]:
    """OCR rendered page PNGs (keyed by 1-based page number) to text via the loaded
    vision model. Returns ``{}`` when there is no vision model or no pages. The caller
    (`ingestion._ocr_scanned_pages`) owns the on/off policy (config default plus the
    per-upload override), so this does not re-check it. Bounded by ``OCR_MAX_PAGES``."""
    if not page_pngs:
        return {}
    ep = endpoint or vision_endpoint()
    if ep is None:
        return {}
    base_url, model = ep

    out: dict[int, str] = {}
    for page_num in sorted(page_pngs)[: config.OCR_MAX_PAGES]:
        text = _ocr_one(base_url, model, page_pngs[page_num], config.OCR_TIMEOUT_S)
        if text:
            out[int(page_num)] = _collapse_runaway(text)
    return out


def merge_page_captions(captions: dict[int, list[str]]) -> dict[int, list[str]]:
    """Combine a page's per-tile captions into one deduped block. Overlapping tiles
    repeat labels, so identical non-empty lines are dropped (first occurrence kept,
    order preserved) and the result passes through ``_collapse_runaway``. Returns one
    merged string per page so ``splice_captions`` adds a single figure block."""
    out: dict[int, list[str]] = {}
    for page, caps in captions.items():
        seen: set[str] = set()
        lines: list[str] = []
        for cap in caps:
            for line in (cap or "").splitlines():
                stripped = line.strip()
                key = stripped.lower()
                if not stripped or key in seen:
                    continue
                seen.add(key)
                lines.append(stripped)
        merged = _collapse_runaway("\n".join(lines))
        if merged.strip():
            out[page] = [merged]
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
