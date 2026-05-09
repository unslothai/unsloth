# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Document extractor for the Chat composer.

Given raw file bytes (PDF / DOCX / HTML / MD / TXT), produce Markdown
suitable to splice into an outgoing chat message. When a vision-capable
model is loaded, selected figures are captioned through our OpenAI-compatible
``/v1/chat/completions`` surface after conversion.

This build uses **PyMuPDF4LLM** (via ``pymupdf4llm`` / ``pymupdf``) for PDF
parsing and **mammoth** for DOCX conversion. Plain-text and Markdown inputs
are decoded as UTF-8 with replacement; HTML inputs are converted to Markdown.

Notes and limitations:

* **OCR is disabled.** There is no local OCR pass in this build, so scanned
  PDFs without a text layer will yield empty or near-empty Markdown. The
  ``use_vlm_ocr`` flag is still accepted for API compatibility; when set it
  renders bounded page images so a loaded vision model can describe them.
* **PPTX is not supported** in this build. ``SUPPORTED_SUFFIXES`` and
  ``SUPPORTED_MIME_TYPES`` no longer advertise the PowerPoint types.
* Parser dependencies are checked per format so plain-text, Markdown, and HTML
  still work when optional PDF or DOCX libraries are missing.
* If the loaded model is not vision-capable, image description is silently
  skipped and ``figures`` comes back with captions set to ``None``;
  ``describe_skipped_reason`` carries the diagnostic text.
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import io
import logging
import math
import multiprocessing
import os
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from typing import Any, Awaitable, Callable, Literal, List, Optional

from .vlm_capability import VlmCapability, detect_loaded_vlm


logger = logging.getLogger(__name__)


SUPPORTED_MIME_TYPES = frozenset(
    {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/json",
        "application/x-ndjson",
        "application/xml",
        "application/yaml",
        "application/javascript",
        "text/html",
        "text/markdown",
        "text/plain",
        "text/csv",
        "text/css",
        "text/javascript",
        "text/xml",
        "text/yaml",
    }
)

SUPPORTED_SUFFIXES = frozenset(
    {
        ".pdf", ".docx", ".html", ".htm", ".md", ".txt",
        ".csv", ".json", ".jsonl", ".yaml", ".yml",
        ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java",
        ".c", ".cpp", ".h", ".hpp", ".cs", ".php", ".rb", ".swift",
        ".kt", ".kts", ".scala", ".sh", ".bash", ".zsh", ".ps1",
        ".sql", ".toml", ".ini", ".cfg", ".log", ".xml", ".css", ".scss",
    }
)


_DESCRIBE_PROMPT = (
    "Describe this figure in <=60 words. Focus on factual content "
    "(axes, labels, captions, visible text, main objects). Do not "
    "speculate beyond what is visible."
)


DEFAULT_DOCUMENT_VISUAL_PAYLOADS = 3
MAX_DOCUMENT_VISUAL_PAYLOADS = 10
_MAX_ENCODED_VISUALS = DEFAULT_DOCUMENT_VISUAL_PAYLOADS
_EXTRACT_TIMEOUT_SECONDS = 120
_VLM_CAPTION_TOTAL_TIMEOUT_SECONDS = 180
_LOCAL_VLM_CAPTION_CONCURRENCY = 1
_DEFAULT_VLM_CAPTION_CONCURRENCY = 3
_EXTRACT_CONCURRENCY = max(
    1, int(os.environ.get("UNSLOTH_STUDIO_EXTRACT_CONCURRENCY", "2"))
)
_EXTRACT_SEMAPHORE = threading.BoundedSemaphore(_EXTRACT_CONCURRENCY)
# Bounded queue wait: callers park here for a slot instead of failing fast
# with 503 when the worker pool is saturated. Tuned so a fast burst (e.g.
# multi-select 4 PDFs) drains naturally without surfacing busy errors,
# while truly stuck workers still time out via _EXTRACT_TIMEOUT_SECONDS.
_EXTRACT_QUEUE_WAIT_SECONDS = max(
    0.0,
    float(os.environ.get("UNSLOTH_STUDIO_EXTRACT_QUEUE_WAIT", "60")),
)
_PAGE_RENDER_DPI = 150
_MAX_PAGE_RENDER_PIXELS = 4_000_000
_MIME_TO_SUFFIX = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/json": ".json",
    "application/x-ndjson": ".jsonl",
    "application/xml": ".xml",
    "application/yaml": ".yaml",
    "application/javascript": ".js",
    "text/html": ".html",
    "text/markdown": ".md",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/css": ".css",
    "text/javascript": ".js",
    "text/xml": ".xml",
    "text/yaml": ".yaml",
}

_PLAIN_TEXT_SUFFIXES = SUPPORTED_SUFFIXES - {".pdf", ".docx", ".html", ".htm"}


def _normalized_suffix(filename: str, content_type: str = "") -> str:
    suffix = os.path.splitext(filename)[1].lower()
    if suffix in SUPPORTED_SUFFIXES:
        return suffix
    mime = (content_type or "").split(";", 1)[0].strip().lower()
    return _MIME_TO_SUFFIX.get(mime, suffix)


class DocumentExtractionUnavailable(RuntimeError):
    """Document extraction backend is not installed or failed to import.

    The backend is PyMuPDF4LLM + mammoth for parsed document formats.
    """


class DocumentExtractionTimeout(RuntimeError):
    """Raised when document parsing exceeds the 120-second worker limit."""


class DocumentExtractionBusy(RuntimeError):
    """Raised when the bounded document extraction worker pool is saturated."""


class DocumentExtractionCancelled(RuntimeError):
    """Raised when the caller cancels an in-flight extraction."""


class DocumentExtractionEncrypted(RuntimeError):
    """Raised when a PDF is encrypted and cannot be parsed without a password."""


try:  # pragma: no cover - presence depends on optional install
    import pymupdf  # type: ignore
    import pymupdf4llm  # type: ignore
except Exception as _pdf_extract_exc:  # pragma: no cover
    pymupdf = None  # type: ignore[assignment]
    pymupdf4llm = None  # type: ignore[assignment]
    _PDF_EXTRACTION_IMPORT_ERROR: Optional[BaseException] = _pdf_extract_exc
else:
    _PDF_EXTRACTION_IMPORT_ERROR = None

try:  # pragma: no cover - presence depends on optional install
    import mammoth  # type: ignore
except Exception as _docx_extract_exc:  # pragma: no cover
    mammoth = None  # type: ignore[assignment]
    _DOCX_EXTRACTION_IMPORT_ERROR: Optional[BaseException] = _docx_extract_exc
else:
    _DOCX_EXTRACTION_IMPORT_ERROR = None

# The dispatcher can still extract plain text / code / data files when PDF or
# DOCX optional parsers are missing. Format-specific helpers raise
# DocumentExtractionUnavailable only when that format is actually requested.
DOCUMENT_EXTRACTION_AVAILABLE = True
_DOCUMENT_EXTRACTION_IMPORT_ERROR: Optional[BaseException] = (
    _PDF_EXTRACTION_IMPORT_ERROR or _DOCX_EXTRACTION_IMPORT_ERROR
)


def document_parser_support() -> dict[str, bool]:
    return {
        "pdf": _PDF_EXTRACTION_IMPORT_ERROR is None,
        "docx": _DOCX_EXTRACTION_IMPORT_ERROR is None,
        "html": True,
        "text": True,
        "data": True,
        "code": True,
    }


def document_parser_unavailable_reasons() -> dict[str, str]:
    reasons: dict[str, str] = {}
    if _PDF_EXTRACTION_IMPORT_ERROR is not None:
        reasons["pdf"] = "PDF extraction requires pymupdf and pymupdf4llm."
    if _DOCX_EXTRACTION_IMPORT_ERROR is not None:
        reasons["docx"] = "DOCX extraction requires mammoth."
    return reasons


@dataclass
class ExtractedFigure:
    id: str
    page: Optional[int]
    caption: Optional[str]
    error: Optional[str] = None
    kind: Literal["figure", "page"] = "figure"
    image_mime: Optional[str] = None
    image_base64: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None


@dataclass
class ExtractResult:
    markdown: str
    figures: List[ExtractedFigure] = field(default_factory = list)
    page_count: int = 0
    tokens_est: int = 0
    describe_skipped_reason: Optional[str] = None
    vlm_source: Optional[str] = None
    vlm_model: Optional[str] = None
    image_input_available: bool = False
    warnings: List[str] = field(default_factory = list)


ProgressCb = Callable[[dict], Awaitable[None]]


def _ensure_pdf_backend() -> None:
    if pymupdf is None or pymupdf4llm is None:
        if _PDF_EXTRACTION_IMPORT_ERROR is not None:
            logger.debug(
                "PDF extraction parser import failed: %s",
                _PDF_EXTRACTION_IMPORT_ERROR,
            )
        raise DocumentExtractionUnavailable(
            "PDF extraction requires pymupdf and pymupdf4llm. Re-run Studio "
            "setup to install the parser dependencies from "
            "studio/backend/requirements/single-env/data-designer-deps.txt"
        )


def _ensure_docx_backend() -> None:
    if mammoth is None:
        if _DOCX_EXTRACTION_IMPORT_ERROR is not None:
            logger.debug(
                "DOCX extraction parser import failed: %s",
                _DOCX_EXTRACTION_IMPORT_ERROR,
            )
        raise DocumentExtractionUnavailable(
            "DOCX extraction requires mammoth. Re-run Studio setup to install "
            "the parser dependencies from "
            "studio/backend/requirements/single-env/data-designer-deps.txt"
        )


def _estimate_tokens(text: str) -> int:
    return max(0, len(text) // 4)


def _encode_pil_image_for_chat(image: Any) -> tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    if image is None:
        return None, None, None, None
    try:
        from PIL import Image as PILImage

        img = image.copy()
        img.thumbnail((1600, 1600))
        if img.mode in ("RGBA", "LA"):
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            alpha = img.getchannel("A")
            background.paste(img.convert("RGB"), mask = alpha)
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        out = io.BytesIO()
        img.save(out, format = "JPEG", quality = 88, optimize = True)
        encoded = base64.b64encode(out.getvalue()).decode("ascii")
        return encoded, img.width, img.height, "image/jpeg"
    except (ImportError, AttributeError, ValueError, OSError) as exc:
        logger.warning("Failed to encode extracted document image", exc_info=exc)
        return None, None, None, None


async def _describe_image_via_vlm(
    *,
    image_base64: str,
    image_mime: str,
    endpoint_url: str,
    model_name: str,
    authorization_header: Optional[str],
    timeout_seconds: float,
) -> tuple[Optional[str], Optional[str]]:
    try:
        import httpx
    except Exception as exc:
        return None, f"httpx unavailable: {exc}"

    headers = {"Content-Type": "application/json"}
    if authorization_header:
        headers["Authorization"] = authorization_header

    data_url = f"data:{image_mime};base64,{image_base64}"
    payload = {
        "model": model_name,
        "stream": False,
        "max_tokens": 512,
        "temperature": 0.2,
        "top_p": 0.9,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _DESCRIBE_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    }
    try:
        async with httpx.AsyncClient(timeout = timeout_seconds) as client:
            response = await client.post(
                endpoint_url.rstrip("/") + "/v1/chat/completions",
                headers = headers,
                json = payload,
            )
        if response.status_code >= 400:
            return None, (
                f"VLM caption request failed with HTTP "
                f"{response.status_code}"
            )
        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        finish_reason = choice.get("finish_reason")

        # Some chat templates (Gemma 3/3n via llama-server, Qwen3 always-think)
        # route the entire visible reply into ``reasoning_content`` and leave
        # ``content`` empty.  The chat UI handles this in its streaming
        # consumer (see ``llama_cpp._chat_completion``); mirror that fallback
        # here so non-streaming callers see the same answer.
        candidates: list[Any] = [
            message.get("content"),
            message.get("reasoning_content"),
            message.get("text"),
        ]
        # Some servers return content as a list of parts (OpenAI multimodal);
        # join any text parts into one string before checking emptiness.
        normalized: list[str] = []
        for raw in candidates:
            if isinstance(raw, str):
                if raw.strip():
                    normalized.append(raw.strip())
            elif isinstance(raw, list):
                parts = [
                    part.get("text", "")
                    for part in raw
                    if isinstance(part, dict)
                    and isinstance(part.get("text"), str)
                ]
                joined = "".join(parts).strip()
                if joined:
                    normalized.append(joined)

        if not normalized:
            logger.warning(
                "VLM caption empty: finish_reason=%r message_keys=%s",
                finish_reason,
                list(message.keys()),
            )
            return None, (
                f"VLM caption empty (finish_reason={finish_reason!r})"
            )
        # Prefer the first non-empty candidate
        # (content > reasoning_content > text).
        return normalized[0], None
    except Exception as exc:
        logger.debug("VLM caption request failed", exc_info = True)
        return None, f"VLM caption request failed: {type(exc).__name__}"


def _build_extract_options(
    *,
    extract_images: bool,
    use_vlm_ocr: bool,
    max_visual_payloads: int,
) -> tuple[dict, list[str]]:
    """Return ``(options, build_warnings)``.

    The options dict is a simple bag of flags consumed by the synchronous
    extract dispatcher. There is no local OCR pass available in this build;
    ``use_vlm_ocr=True`` is implemented as a bounded full-page visual
    extraction fallback for VLM captioning.
    """
    build_warnings: list[str] = []
    if use_vlm_ocr:
        build_warnings.append(
            "Full-page OCR was requested, but this build has no local OCR "
            "engine; rendered page images will be sent to the loaded vision "
            "model when image description is enabled."
        )
    options = {
        "extract_images": bool(extract_images),
        "use_vlm_ocr": bool(use_vlm_ocr),
        "max_visual_payloads": max(0, max_visual_payloads),
    }
    return options, build_warnings


def _pymupdf4llm_markdown_kwargs() -> dict[str, Any]:
    """Return kwargs supported by the installed pymupdf4llm.to_markdown()."""
    preferred = {
        "write_images": False,
        "show_progress": False,
        "ignore_images": True,
        "table_strategy": "lines_strict",
        "use_ocr": False,
        "force_ocr": False,
    }
    try:
        signature = inspect.signature(pymupdf4llm.to_markdown)
    except (TypeError, ValueError):
        return {
            key: value
            for key, value in preferred.items()
            if key not in {"use_ocr", "force_ocr"}
        }
    params = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()):
        return preferred
    return {key: value for key, value in preferred.items() if key in params}


def _safe_page_pixmap(page: Any) -> Any:
    rect = getattr(page, "rect", None)
    width_pt = max(float(getattr(rect, "width", 0) or 0), 1.0)
    height_pt = max(float(getattr(rect, "height", 0) or 0), 1.0)
    scale = _PAGE_RENDER_DPI / 72.0
    projected_pixels = width_pt * scale * height_pt * scale
    if projected_pixels > _MAX_PAGE_RENDER_PIXELS:
        scale *= math.sqrt(_MAX_PAGE_RENDER_PIXELS / projected_pixels)
    scale = max(scale, 0.05)
    matrix = pymupdf.Matrix(scale, scale)  # type: ignore[union-attr]
    return page.get_pixmap(matrix = matrix, alpha = False)


def _append_page_image_figure(
    doc: Any,
    figures_out: list[ExtractedFigure],
    *,
    page_index: int,
    max_figures: int,
    encode_image: bool = True,
) -> bool:
    if len(figures_out) >= max_figures:
        return False
    if not encode_image:
        figures_out.append(
            ExtractedFigure(
                id = f"page-{page_index + 1}",
                page = page_index + 1,
                caption = None,
                error = None,
                kind = "page",
            )
        )
        return True
    try:
        from PIL import Image as PILImage

        pix = _safe_page_pixmap(doc[page_index])
        png_bytes = pix.tobytes("png")
        page_image = PILImage.open(io.BytesIO(png_bytes))
        image_base64, image_width, image_height, image_mime = (
            _encode_pil_image_for_chat(page_image)
        )
        if not image_base64:
            return False
        figures_out.append(
            ExtractedFigure(
                id = f"page-{page_index + 1}",
                page = page_index + 1,
                caption = None,
                error = None,
                kind = "page",
                image_mime = image_mime,
                image_base64 = image_base64,
                image_width = image_width,
                image_height = image_height,
            )
        )
        return True
    except (
        ImportError,
        MemoryError,
        OverflowError,
        ValueError,
        OSError,
        RuntimeError,
    ) as exc:
        logger.warning(
            "Failed to render page %d preview for PDF",
            page_index + 1,
            exc_info = exc,
        )
        return False


def _extract_pdf(
    file_bytes: bytes,
    max_figures: int,
    use_vlm_ocr: bool,
    max_visual_payloads: int,
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    """Extract Markdown + figures from a PDF via PyMuPDF4LLM.

    Returns ``(markdown, figures, page_count, truncated_count, seen)``.
    """
    _ensure_pdf_backend()
    assert pymupdf is not None and pymupdf4llm is not None  # for type-checkers

    doc = pymupdf.open(stream = file_bytes, filetype = "pdf")
    try:
        if getattr(doc, "is_encrypted", False) or getattr(doc, "needs_pass", False):
            raise DocumentExtractionEncrypted(
                "Encrypted PDF; provide a password before extracting it."
            )
        markdown = pymupdf4llm.to_markdown(doc, **_pymupdf4llm_markdown_kwargs())

        figures_out: list[ExtractedFigure] = []
        encoded_visuals = 0
        seen = 0
        truncated_count = 0
        page_count = len(doc)

        if max_figures > 0 and page_count > 0:
            if use_vlm_ocr:
                for page_index in range(page_count):
                    if len(figures_out) >= max_figures:
                        truncated_count += page_count - page_index
                        break
                    if _append_page_image_figure(
                        doc,
                        figures_out,
                        page_index = page_index,
                        max_figures = max_figures,
                        encode_image = encoded_visuals < max_visual_payloads,
                    ):
                        if figures_out[-1].image_base64:
                            encoded_visuals += 1
                        seen += 1
            elif _append_page_image_figure(
                doc,
                figures_out,
                page_index = 0,
                max_figures = max_figures,
                encode_image = encoded_visuals < max_visual_payloads,
            ):
                if figures_out[-1].image_base64:
                    encoded_visuals += 1

            if not use_vlm_ocr:
                try:
                    from PIL import Image as PILImage

                    for page_index in range(page_count):
                        page = doc[page_index]
                        try:
                            images = page.get_images(full = True)
                        except (ValueError, RuntimeError) as exc:
                            logger.debug(
                                "page.get_images failed on page %d",
                                page_index + 1,
                                exc_info = exc,
                            )
                            continue
                        for img_info in images:
                            xref = img_info[0] if img_info else 0
                            if not xref:
                                continue
                            try:
                                extracted = doc.extract_image(xref)
                            except (ValueError, RuntimeError) as exc:
                                logger.debug(
                                    "doc.extract_image failed for xref %s",
                                    xref,
                                    exc_info = exc,
                                )
                                continue
                            if not extracted:
                                continue
                            raw_bytes = extracted.get("image")
                            if not raw_bytes:
                                continue
                            try:
                                pil_img = PILImage.open(io.BytesIO(raw_bytes))
                                pil_img.load()
                            except (OSError, ValueError) as exc:
                                logger.debug(
                                    "PIL failed to decode extracted image xref %s",
                                    xref,
                                    exc_info = exc,
                                )
                                continue
                            if pil_img.width < 50 or pil_img.height < 50:
                                continue
                            seen += 1
                            if len(figures_out) >= max_figures:
                                truncated_count += 1
                                continue
                            image_base64 = None
                            image_width = None
                            image_height = None
                            image_mime = None
                            if encoded_visuals < max_visual_payloads:
                                (
                                    image_base64,
                                    image_width,
                                    image_height,
                                    image_mime,
                                ) = _encode_pil_image_for_chat(pil_img)
                                if image_base64:
                                    encoded_visuals += 1
                            figures_out.append(
                                ExtractedFigure(
                                    id = f"fig-{len(figures_out)}",
                                    page = page_index + 1,
                                    caption = None,
                                    error = None,
                                    kind = "figure",
                                    image_mime = image_mime,
                                    image_base64 = image_base64,
                                    image_width = image_width,
                                    image_height = image_height,
                                )
                            )
                except ImportError as exc:
                    logger.warning(
                        "Pillow is unavailable; skipping embedded-image extraction",
                        exc_info = exc,
                    )

        return markdown, figures_out, page_count, truncated_count, seen
    finally:
        try:
            doc.close()
        except Exception:  # pragma: no cover - defensive
            logger.debug("pymupdf doc.close() raised", exc_info = True)


def _extract_docx(
    file_bytes: bytes,
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    _ensure_docx_backend()
    assert mammoth is not None  # for type-checkers
    stream = io.BytesIO(file_bytes)
    result = mammoth.convert_to_markdown(stream)
    markdown = result.value or ""
    return markdown, [], 0, 0, 0


def _extract_plaintext(
    file_bytes: bytes,
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    text = file_bytes.decode("utf-8", errors = "replace")
    return text, [], 0, 0, 0


def _extract_html(
    file_bytes: bytes,
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    html = file_bytes.decode("utf-8", errors = "replace")
    try:
        from core.inference._html_to_md import html_to_markdown
    except Exception as exc:
        logger.warning(
            "HTML-to-Markdown converter unavailable; using raw HTML",
            exc_info = exc,
        )
        return html, [], 0, 0, 0
    return html_to_markdown(html), [], 0, 0, 0


def _run_extract_sync(
    file_bytes: bytes,
    filename: str,
    options: dict,
    content_type: str = "",
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    """Synchronous dispatch by file suffix.

    Returns ``(markdown, figures, page_count, truncated_count, seen)``.
    """
    suffix = _normalized_suffix(filename, content_type)
    extract_images = bool(options.get("extract_images"))
    use_vlm_ocr = bool(options.get("use_vlm_ocr"))
    max_figures = int(options.get("max_figures", 0)) if extract_images else 0
    max_visual_payloads = int(
        options.get("max_visual_payloads", DEFAULT_DOCUMENT_VISUAL_PAYLOADS)
    )

    if suffix == ".pdf":
        return _extract_pdf(file_bytes, max_figures, use_vlm_ocr, max_visual_payloads)
    if suffix == ".docx":
        return _extract_docx(file_bytes)
    if suffix in {".html", ".htm"}:
        return _extract_html(file_bytes)
    if suffix in _PLAIN_TEXT_SUFFIXES:
        return _extract_plaintext(file_bytes)
    raise ValueError(f"Unsupported file type: {filename}")


_RUN_EXTRACT_SYNC_ORIGINAL = _run_extract_sync


def _run_extract_worker(
    result_queue: Any,
    file_bytes: bytes,
    filename: str,
    options: dict,
    content_type: str,
) -> None:
    try:
        result_queue.put(
            ("ok", _run_extract_sync(file_bytes, filename, options, content_type))
        )
    except DocumentExtractionUnavailable as exc:
        result_queue.put(("extraction_unavailable", str(exc)))
    except DocumentExtractionEncrypted as exc:
        result_queue.put(("encrypted", str(exc)))
    except ValueError as exc:
        result_queue.put(("value_error", str(exc)))
    except BaseException as exc:
        result_queue.put(("error", type(exc).__name__, str(exc)))


def _drain_future_exception(fut: Any) -> None:
    """Retrieve a future's exception (if any) so asyncio's gc-time
    "Future exception was never retrieved" warning stays quiet when the
    awaiting task is cancelled mid-flight (e.g. client disconnect or
    AbortController abort)."""
    try:
        if fut.cancelled():
            return
        fut.exception()
    except BaseException:
        # Never let a drain hook itself raise — best effort only.
        pass


def _terminate_extract_process(proc: multiprocessing.Process) -> None:
    if not proc.is_alive():
        return
    proc.terminate()
    proc.join(5)
    if proc.is_alive() and hasattr(proc, "kill"):
        proc.kill()
        proc.join(2)


def _run_extract_process_sync(
    file_bytes: bytes,
    filename: str,
    options: dict,
    content_type: str,
    timeout_seconds: int,
    cancel_event: Optional[threading.Event] = None,
) -> tuple[str, list[ExtractedFigure], int, int, int]:
    if cancel_event is not None and cancel_event.is_set():
        raise DocumentExtractionCancelled("document extraction was cancelled")
    # Park up to _EXTRACT_QUEUE_WAIT_SECONDS waiting for a slot, polling
    # cancel_event so a client disconnect during the wait short-circuits
    # cleanly instead of holding the request open.
    deadline = time.monotonic() + _EXTRACT_QUEUE_WAIT_SECONDS
    acquired = _EXTRACT_SEMAPHORE.acquire(blocking = False)
    while True:
        if acquired:
            break
        if cancel_event is not None and cancel_event.is_set():
            raise DocumentExtractionCancelled(
                "document extraction was cancelled"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        wait = min(remaining, 0.5)
        if _EXTRACT_SEMAPHORE.acquire(timeout = wait):
            acquired = True
            break
    if not acquired:
        raise DocumentExtractionBusy("document extraction is busy")

    ctx = multiprocessing.get_context("spawn" if os.name == "nt" else "fork")
    result_queue = ctx.Queue(maxsize = 1)
    proc = ctx.Process(
        target = _run_extract_worker,
        args = (result_queue, file_bytes, filename, options, content_type),
        daemon = True,
    )
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise DocumentExtractionCancelled("document extraction was cancelled")
        proc.start()
        deadline = time.monotonic() + timeout_seconds
        message = None
        while message is None:
            try:
                message = result_queue.get(timeout = 0.1)
                break
            except queue.Empty:
                if cancel_event is not None and cancel_event.is_set():
                    _terminate_extract_process(proc)
                    raise DocumentExtractionCancelled(
                        "document extraction was cancelled"
                    )
                if not proc.is_alive():
                    break
                if time.monotonic() >= deadline:
                    _terminate_extract_process(proc)
                    raise DocumentExtractionTimeout(
                        "document parsing exceeded the 120-second worker limit"
                    )

        proc.join(2)
        if proc.is_alive():
            proc.terminate()
            proc.join(2)
        if message is None:
            raise RuntimeError(
                f"document extraction worker exited without a result "
                f"(exitcode={proc.exitcode})"
            )

        kind = message[0]
        if kind == "ok":
            return message[1]
        if kind == "extraction_unavailable":
            raise DocumentExtractionUnavailable(message[1])
        if kind == "encrypted":
            raise DocumentExtractionEncrypted(message[1])
        if kind == "value_error":
            raise ValueError(message[1])
        if kind == "error":
            raise RuntimeError(f"{message[1]}: {message[2]}")
        raise RuntimeError(f"unexpected document worker result: {kind!r}")
    finally:
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception:
            pass
        _EXTRACT_SEMAPHORE.release()


async def extract_document(
    file_bytes: bytes,
    filename: str,
    *,
    content_type: str = "",
    describe_images: bool = True,
    use_vlm_ocr: bool = False,
    max_figures: int = 40,
    max_visual_payloads: int = DEFAULT_DOCUMENT_VISUAL_PAYLOADS,
    vlm_timeout_seconds: float = 60.0,
    capability: Optional[VlmCapability] = None,
    self_base_url: Optional[str] = None,
    authorization_header: Optional[str] = None,
    progress_cb: Optional[ProgressCb] = None,
    cancel_event: Optional[threading.Event] = None,
) -> ExtractResult:
    """Extract layout-aware Markdown plus figure metadata.

    When ``describe_images`` is True and the active model is
    vision-capable, the selected visual references are captioned via the
    OpenAI-compat ``/v1/chat/completions`` surface after extraction.
    Otherwise figures come back with ``caption=None`` and
    ``describe_skipped_reason`` carries the human-readable reason.
    """
    async def _emit(**event: Any) -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise DocumentExtractionCancelled("document extraction was cancelled")
        if progress_cb is not None:
            try:
                await progress_cb(event)
            except Exception:
                logger.debug("progress_cb raised; continuing", exc_info = True)

    max_figures = max(0, max_figures)
    max_visual_payloads = max(0, min(max_visual_payloads, max_figures))
    cap = capability if capability is not None else detect_loaded_vlm(self_base_url)
    image_input_available = bool(cap.is_vlm and cap.endpoint_url and cap.model_name)
    describe_available = bool(
        describe_images and cap.is_vlm and cap.endpoint_url and cap.model_name
    )
    effective_describe = (
        describe_available and max_figures > 0 and max_visual_payloads > 0
    )
    extract_images = max_figures > 0

    skipped_reason: Optional[str] = None
    if describe_images and not effective_describe:
        if describe_available and max_figures <= 0:
            skipped_reason = "figure description disabled because max_figures is 0"
        elif describe_available and max_visual_payloads <= 0:
            skipped_reason = (
                "figure description disabled because max_visual_payloads is 0"
            )
        else:
            skipped_reason = cap.reason or "no_vlm"

    await _emit(stage = "parsing")

    options, build_warnings = _build_extract_options(
        extract_images = extract_images,
        use_vlm_ocr = use_vlm_ocr,
        max_visual_payloads = max_visual_payloads,
    )
    options["max_figures"] = max_figures

    try:
        if _run_extract_sync is _RUN_EXTRACT_SYNC_ORIGINAL:
            # Drive run_in_executor directly (rather than asyncio.to_thread)
            # so we can attach a done-callback that retrieves the future's
            # exception even when the awaiting task is cancelled — silences
            # "Future exception was never retrieved" noise on busy/cancel.
            loop = asyncio.get_running_loop()
            extract_future = loop.run_in_executor(
                None,
                _run_extract_process_sync,
                file_bytes,
                filename,
                options,
                content_type,
                _EXTRACT_TIMEOUT_SECONDS,
                cancel_event,
            )
            extract_future.add_done_callback(_drain_future_exception)
            markdown, figures_out, page_count, truncated_count, seen = (
                await extract_future
            )
        else:
            # Tests monkeypatch _run_extract_sync directly; preserve that seam
            # without forcing patched callables through multiprocessing spawn.
            loop = asyncio.get_running_loop()
            markdown, figures_out, page_count, truncated_count, seen = (
                await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        _run_extract_sync,
                        file_bytes,
                        filename,
                        options,
                        content_type,
                    ),
                    timeout = _EXTRACT_TIMEOUT_SECONDS,
                )
            )
    except asyncio.TimeoutError:
        raise DocumentExtractionTimeout(
            "document parsing exceeded the 120-second worker limit"
        )
    except DocumentExtractionTimeout:
        raise
    except DocumentExtractionBusy:
        raise
    except DocumentExtractionCancelled:
        raise
    except DocumentExtractionEncrypted:
        raise
    except DocumentExtractionUnavailable:
        raise
    except ValueError:
        # Unsupported file type — surface unchanged so the route can map to 415.
        raise
    except Exception as exc:
        logger.exception("document extraction failed for %s", filename)
        raise RuntimeError("document extraction failed") from exc

    caption_deadline_hit = False
    if effective_describe:
        caption_concurrency = (
            _LOCAL_VLM_CAPTION_CONCURRENCY
            if cap.source in {"transformers", "unsloth"}
            else _DEFAULT_VLM_CAPTION_CONCURRENCY
        )
        sem = asyncio.Semaphore(caption_concurrency)

        captionable_total = sum(
            1
            for fig in figures_out[:max_figures]
            if fig.image_base64 and fig.image_mime
        )
        captioned_completed = 0
        await _emit(
            stage = "captioning",
            current = 0,
            total = captionable_total,
            page = None,
            total_pages = page_count,
        )

        async def _describe_one(index: int, figure: ExtractedFigure) -> None:
            nonlocal captioned_completed
            if figure.caption or not figure.image_base64 or not figure.image_mime:
                return
            if cancel_event is not None and cancel_event.is_set():
                raise DocumentExtractionCancelled("document extraction was cancelled")
            async with sem:
                if cancel_event is not None and cancel_event.is_set():
                    raise DocumentExtractionCancelled(
                        "document extraction was cancelled"
                    )
                try:
                    caption, error = await _describe_image_via_vlm(
                        image_base64 = figure.image_base64,
                        image_mime = figure.image_mime,
                        endpoint_url = cap.endpoint_url or "",
                        model_name = cap.model_name or "",
                        authorization_header = authorization_header,
                        timeout_seconds = vlm_timeout_seconds,
                    )
                    figures_out[index] = replace(
                        figure,
                        caption = caption,
                        error = error,
                    )
                except asyncio.TimeoutError as exc:
                    logger.warning(
                        "VLM describe timed out for figure %s", figure.id, exc_info=exc
                    )
                    figures_out[index] = replace(
                        figure,
                        error = f"VLM describe timed out: {type(exc).__name__}",
                    )
                except Exception as exc:
                    logger.warning(
                        "VLM describe failed for figure %s", figure.id, exc_info=exc
                    )
                    figures_out[index] = replace(
                        figure,
                        error = f"VLM describe failed: {type(exc).__name__}",
                    )
                finally:
                    captioned_completed += 1
                    await _emit(
                        stage = "captioning",
                        current = captioned_completed,
                        total = captionable_total,
                        page = figure.page,
                        total_pages = page_count,
                    )

        tasks = [
            _describe_one(index, fig)
            for index, fig in enumerate(figures_out[:max_figures])
            if fig.image_base64 and fig.image_mime
        ]
        if tasks:
            try:
                caption_timeout_seconds = _VLM_CAPTION_TOTAL_TIMEOUT_SECONDS
                if cap.source in {"transformers", "unsloth"}:
                    caption_timeout_seconds = max(
                        caption_timeout_seconds,
                        len(tasks) * vlm_timeout_seconds + 15,
                    )
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout = caption_timeout_seconds,
                )
                for result in results:
                    if isinstance(
                        result,
                        (DocumentExtractionCancelled, asyncio.CancelledError),
                    ):
                        raise result
            except asyncio.TimeoutError:
                caption_deadline_hit = True
                for index, figure in enumerate(figures_out):
                    if figure.image_base64 and not figure.caption and not figure.error:
                        figures_out[index] = replace(
                            figure,
                            error = "VLM caption deadline exceeded",
                        )

    warnings: List[str] = list(build_warnings)
    if truncated_count > 0:
        warnings.append(
            f"Document has {seen} figures; showing the first {max_figures} "
            f"({truncated_count} truncated)."
        )
    visual_payload_count = sum(1 for figure in figures_out if figure.image_base64)
    if (
        visual_payload_count >= max_visual_payloads
        and len(figures_out) > visual_payload_count
    ):
        warnings.append(
            f"Only the first {max_visual_payloads} visual payloads "
            "were attached; remaining figure references are text-only."
        )
    if effective_describe and figures_out and all(f.caption is None for f in figures_out):
        error_samples: list[str] = []
        seen_errors: set[str] = set()
        for figure in figures_out:
            if not figure.error or figure.error in seen_errors:
                continue
            seen_errors.add(figure.error)
            error_samples.append(f"{figure.id}: {figure.error}")
            if len(error_samples) >= 3:
                break
        sample_suffix = (
            " Examples: " + "; ".join(error_samples) + "."
            if error_samples
            else ""
        )
        warnings.append(
            "Figure descriptions were requested but none were produced — "
            "check that the loaded model accepts image inputs via /v1."
            f"{sample_suffix}"
        )
    if caption_deadline_hit:
        warnings.append(
            "Figure captioning reached the inline timeout; some image "
            "descriptions were skipped."
        )

    await _emit(stage = "done")

    return ExtractResult(
        markdown = markdown,
        figures = figures_out,
        page_count = page_count,
        tokens_est = _estimate_tokens(markdown),
        describe_skipped_reason = skipped_reason,
        vlm_source = cap.source,
        vlm_model = cap.model_name,
        image_input_available = image_input_available,
        warnings = warnings,
    )
