# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG config; every value is env-overridable."""

from __future__ import annotations

import os
import re

DEFAULT_EMBEDDING_MODEL = "unsloth/bge-small-en-v1.5"
EMBEDDING_MODEL = os.environ.get("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
# Keep <= embedder_max - ~12: bge's 512 limit plus 2 special tokens (llama-server 500s on overflow, ST truncates).
CHUNK_TOKENS = int(os.environ.get("RAG_CHUNK_TOKENS", "500"))
CHUNK_OVERLAP = int(os.environ.get("RAG_CHUNK_OVERLAP", "64"))
TOP_K_LEXICAL = int(os.environ.get("RAG_TOP_K_LEXICAL", "30"))
TOP_K_DENSE = int(os.environ.get("RAG_TOP_K_DENSE", "30"))
TOP_K_HYBRID = int(os.environ.get("RAG_TOP_K_HYBRID", "10"))
RRF_K = int(os.environ.get("RAG_RRF_K", "60"))

# Whole-document context: a file under this budget is injected in full instead of top-K retrieval.
THREAD_WHOLE_DOC = os.environ.get("RAG_THREAD_WHOLE_DOC", "1") == "1"
WHOLE_DOC_MAX_TOKENS = int(os.environ.get("RAG_WHOLE_DOC_MAX_TOKENS", "6000"))

# Off, evicted turns are simply dropped and the recall reserve is not taken. Only applies once the
# window evicts, itself opt-in per request via context_overflow="truncate_oldest"; the compaction
# headroom and sticky boundary belong to the window (ROLLING_COMPACTION_HEADROOM_RATIO), not here.
CONVERSATION_ARCHIVE = os.environ.get("RAG_CONVERSATION_ARCHIVE", "1") == "1"
CONVERSATION_ARCHIVE_TOP_K = int(os.environ.get("RAG_CONVERSATION_ARCHIVE_TOP_K", "4"))
# Sized to CONVERSATION_ARCHIVE_TOP_K * CHUNK_TOKENS with slack for the wrapper text.
CONVERSATION_RECALL_RESERVE_TOKENS = int(
    os.environ.get("RAG_CONVERSATION_RECALL_RESERVE_TOKENS", "2048")
)
# Off restores the plain OR-of-every-token query other scopes use.
CONVERSATION_QUERY_FOCUS = os.environ.get("RAG_CONVERSATION_QUERY_FOCUS", "1") == "1"
# Presentation only: neither ordering changes which turns are selected.
CONVERSATION_RECALL_ORDER = os.environ.get("RAG_CONVERSATION_RECALL_ORDER", "chronological")
# Automatic recall only, never a search the model asked for; default 0.0 since a weak match is often
# still the right turn.
CONVERSATION_FORCED_MIN_SCORE = float(os.environ.get("RAG_CONVERSATION_FORCED_MIN_SCORE", "0.0"))

UPLOAD_EXTS = {".pdf", ".txt", ".md", ".markdown", ".docx", ".html", ".htm"}
# 0 disables the cap; bounds parse + vision work at ingest.
MAX_UPLOAD_BYTES = int(os.environ.get("RAG_MAX_UPLOAD_BYTES", str(200 * 1024 * 1024)))

# Caps prevent an accidentally broad linked folder from becoming an unbounded ingestion queue.
FOLDER_SYNC_INTERVAL_S = float(os.environ.get("RAG_FOLDER_SYNC_INTERVAL_S", "30"))
FOLDER_MAX_FILES = int(os.environ.get("RAG_FOLDER_MAX_FILES", "10000"))
FOLDER_JOB_HISTORY_LIMIT = int(os.environ.get("RAG_FOLDER_JOB_HISTORY_LIMIT", "200"))

# Falls back to plain PyMuPDF text when off, when pymupdf4llm is missing, or when extraction fails.
PDF_MARKDOWN = os.environ.get("RAG_PDF_MARKDOWN", "1") == "1"

# No-op without a vision model; the chat's "Describe figures & charts" toggle overrides it per upload.
CAPTION_IMAGES = os.environ.get("RAG_CAPTION_IMAGES", "1") == "1"
# Total per-document tile budget (figure-bearing pages are tiled, see below).
CAPTION_MAX_IMAGES = int(os.environ.get("RAG_CAPTION_MAX_IMAGES", "24"))
CAPTION_TIMEOUT_S = float(os.environ.get("RAG_CAPTION_TIMEOUT_S", "60"))
# Captions transcribe every label, and FIGURE_DPI keeps small box/axis labels legible when tiled.
CAPTION_MAX_TOKENS = int(os.environ.get("RAG_CAPTION_MAX_TOKENS", "768"))
FIGURE_DPI = int(os.environ.get("RAG_FIGURE_DPI", "200"))
# Overlapping tile grid covers sub-figures and small labels without exact region detection.
FIGURE_TILE_ROWS = int(os.environ.get("RAG_FIGURE_TILE_ROWS", "2"))
FIGURE_TILE_COLS = int(os.environ.get("RAG_FIGURE_TILE_COLS", "2"))
FIGURE_TILE_OVERLAP = float(os.environ.get("RAG_FIGURE_TILE_OVERLAP", "0.12"))
FIGURE_FULLPAGE = os.environ.get("RAG_FIGURE_FULLPAGE", "1") == "1"
CAPTION_MAX_PAGES = int(os.environ.get("RAG_CAPTION_MAX_PAGES", "4"))

# Needs a vision model, else the page stays empty; MIN_CHARS is the text length below which a page counts as scanned.
OCR_SCANNED = os.environ.get("RAG_OCR_SCANNED", "1") == "1"
OCR_MIN_CHARS = int(os.environ.get("RAG_OCR_MIN_CHARS", "16"))
OCR_MAX_PAGES = int(os.environ.get("RAG_OCR_MAX_PAGES", "20"))
OCR_DPI = int(os.environ.get("RAG_OCR_DPI", "150"))
OCR_TIMEOUT_S = float(os.environ.get("RAG_OCR_TIMEOUT_S", "60"))
OCR_MAX_TOKENS = int(os.environ.get("RAG_OCR_MAX_TOKENS", "2048"))

# Switching backends changes the vectors, so the index must be rebuilt.
EMBED_BACKEND = os.environ.get("RAG_EMBED_BACKEND", "auto")

# The model name alone is not the embedding space: llama-server embeds through the GGUF companion and pools its own way.
EMBEDDING_IDENTITY_TAGS = ("sentence-transformers", "llama-server")


def _escape_identity_segment(value: str) -> str:
    """Colons separate the segments, and a model can be a local path that contains one
    (``C:\\models\\bge``), which would otherwise read back as ``C``. A repo id contains
    neither character, so the identity of a normal model is unchanged."""
    return value.replace("%", "%25").replace(":", "%3A")


def _unescape_identity_segment(value: str) -> str:
    return value.replace("%3A", ":").replace("%25", "%")


def embedding_identity(
    backend: str,
    model: str,
    *,
    gguf_repo: str | None = None,
) -> str:
    """Tagged identity for ``documents.embedding_model``.

    The configured model comes first so a row written before identities carried a tag
    still compares equal on it. llama-server appends the GGUF repo it actually embeds
    through, which is the part that can differ from the model's ST form."""
    model = _escape_identity_segment(model)
    if gguf_repo is None:
        return f"{backend}:{model}"
    return f"{backend}:{model}:{_escape_identity_segment(gguf_repo)}"


def embedding_identity_model(identity: str | None) -> str | None:
    """The configured model inside a tagged identity, or None when untagged."""
    for tag in EMBEDDING_IDENTITY_TAGS:
        if identity and identity.startswith(f"{tag}:"):
            return _unescape_identity_segment(identity[len(tag) + 1 :].split(":", 1)[0])
    return None


def embedding_identity_matches(stored: str | None, current: str) -> bool:
    """Whether ``stored``'s vectors can answer a query embedded under ``current``.

    NULL is still assumed current. An untagged row predates the tag and we cannot know
    which backend wrote it, so it matches on the model name alone, exactly as it did
    before: dropping those would empty dense search over every corpus indexed so far.
    They are reported instead (``store.count_untagged_documents``)."""
    if stored is None:
        return True
    if embedding_identity_model(stored) is not None:
        return stored == current
    return stored == (embedding_identity_model(current) or current)


def effective_embedding_model() -> str:
    """The embedding model actually in use: the persisted Settings override when
    one is stored, else ``EMBEDDING_MODEL`` (env/default). Read at call time so a
    Settings change applies without a restart."""
    try:
        from utils.embedding_model_settings import get_rag_embedding_model
        return get_rag_embedding_model()
    except Exception:  # noqa: BLE001 - settings store unavailable (tests, early boot)
        return EMBEDDING_MODEL


def _names_gguf(model: str) -> bool:
    """True when "gguf" appears as a whole name segment, so plain substrings
    like "bigguf" don't count."""
    return "gguf" in re.split(r"[^a-z0-9]+", model.lower())


# Suffixes unsloth puts on an unquantized re-upload; the GGUF sits on the base name.
_QUANT_SUFFIX_RE = re.compile(r"(?:-qat)?(?:-q\d+_\d+[a-z]*)?-unquantized$", re.I)


def gguf_repo_candidates(model: str) -> list[str]:
    """Repos that may hold ``model``'s GGUF, in preference order. Shared by the
    loader and the settings resolve endpoint so they cannot pick different repos."""
    if "RAG_EMBED_GGUF_REPO" in os.environ:
        return [EMBED_GGUF_REPO]
    if _names_gguf(model):
        return [model]
    owner, _, name = model.rpartition("/")
    out = [f"{model}-GGUF"]
    base = _QUANT_SUFFIX_RE.sub("", name)
    if base != name:
        out.append(f"{owner}/{base}-GGUF" if owner else f"{base}-GGUF")
    out.append(model)
    return list(dict.fromkeys(out))


def gguf_repo_is_explicit() -> bool:
    """Whether one GGUF repository was explicitly pinned for every embedder."""
    return "RAG_EMBED_GGUF_REPO" in os.environ


def gguf_repo_for_embedding_model(model: str) -> str:
    """GGUF repo for ``model``, honoring an explicit companion override."""
    if "RAG_EMBED_GGUF_REPO" in os.environ:
        return EMBED_GGUF_REPO
    if model == DEFAULT_EMBEDDING_MODEL:
        return EMBED_GGUF_REPO
    if _names_gguf(model):
        return model
    return f"{model}-GGUF"


def default_gguf_repo() -> str:
    """GGUF companion for the env/default embedding model."""
    return gguf_repo_for_embedding_model(EMBEDDING_MODEL)


def effective_gguf_repo() -> str:
    """GGUF repo for the llama-server backend, tracking the effective model.

    An explicit ``RAG_EMBED_GGUF_REPO`` env always wins, then the repo the picker
    resolved and stored for this model (which need not follow any naming rule),
    then the ``-GGUF`` companion convention.
    """
    return effective_gguf_repo_for_embedding_model(effective_embedding_model())


def effective_gguf_repo_for_embedding_model(model: str) -> str:
    """GGUF repo the loader/identity use for ``model``.

    The resolved repo is part of the vector space identity, not merely a load
    location: two different conversions of the same source model need separate
    tags or their document/query vectors can be mixed.
    """
    if "RAG_EMBED_GGUF_REPO" in os.environ:
        return EMBED_GGUF_REPO
    try:
        from utils.embedding_model_settings import get_stored_gguf_repo, remembered_gguf_repo
        stored = get_stored_gguf_repo(model)
        if stored is None:
            # One stored record, so saving another model would move a pinned job's derived identity mid-run and
            # split one document set across two tags.
            stored = remembered_gguf_repo(model)
    except Exception:  # noqa: BLE001 - store unavailable: fall back to the convention
        stored = None
    return stored or gguf_repo_for_embedding_model(model)


# F16 over Q8_0: faster (no per-block dequant at this size) and exact, for ~30MB more on disk.
EMBED_GGUF_REPO = os.environ.get("RAG_EMBED_GGUF_REPO", "unsloth/bge-small-en-v1.5-GGUF")
EMBED_GGUF_VARIANT = os.environ.get("RAG_EMBED_GGUF_VARIANT", "F16")
# "auto" differs per backend: llama-server offloads inside its own subprocess, while
# sentence-transformers would pin a CUDA primary context (712 MiB on a B200), so it stays on CPU.
EMBED_DEVICE = os.environ.get("RAG_EMBED_DEVICE", "auto")


def embed_device_preference() -> str:
    """``EMBED_DEVICE`` normalized to exactly ``gpu``, ``cpu`` or ``auto``.

    One reader for both backends, because they used to disagree about the same
    string: the llama path compared a bare ``.lower()``, so ``" gpu "`` fell through
    to auto, and an Intel user writing the accelerator's own name (``xpu``) got CPU
    from a setting that named their device. Anything that is not recognizably a
    request for CPU or for an accelerator is ``auto``, so a typo degrades to each
    backend's default rather than to silence.
    """
    value = (EMBED_DEVICE or "").strip().lower()
    if value in ("gpu", "cuda", "rocm", "hip", "xpu", "mps", "metal"):
        return "gpu"
    if value == "cpu":
        return "cpu"
    return "auto"


def embed_device_requires_gpu() -> bool:
    """True when a failed GPU start must raise instead of retrying on CPU.

    Only the literal documented value is that hard a request, which is what it has
    always meant. The spellings we newly began honoring above -- padding, or the
    accelerator's own name -- used to fall through to ``auto``, and ``auto`` falls
    back, so reading them as fatal would take RAG away from hosts it worked on. They
    still opt into the GPU; they just do not insist on it."""
    return (EMBED_DEVICE or "").lower() == "gpu"


EMBED_HOST = os.environ.get("RAG_EMBED_HOST", "127.0.0.1")
EMBED_PORT = int(os.environ.get("RAG_EMBED_PORT", "0"))
EMBED_BATCH = int(os.environ.get("RAG_EMBED_BATCH", "64"))
EMBED_STARTUP_TIMEOUT_S = float(os.environ.get("RAG_EMBED_STARTUP_TIMEOUT_S", "120"))
EMBED_REQUEST_TIMEOUT_S = float(os.environ.get("RAG_EMBED_REQUEST_TIMEOUT_S", "60"))
