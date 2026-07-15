# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Infra-only model detection shared by the model routes and the hub
inventory. Lives directly under ``utils`` (not ``utils.models``) so the hub
cache scanner can import it without pulling in ``utils/models/__init__.py``,
which eagerly loads the model-config/checkpoint stack, and without importing
``routes.models`` (import-time side effects, would cycle)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Hub repo id shape ("owner/name", no leading separator); anything else is
# treated as a local filesystem path.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")

# The llama.cpp install-validation probe repo. Always hidden.
_PROBE_REPO_ID = "ggml-org/models"
# The probe's on-disk filename. Carries the ".gguf" so it stays specific and
# does not hide unrelated repos like ``user/stories260K-finetune-GGUF``.
_PROBE_FILENAME = "stories260k.gguf"


def _safe_resolve(path: Path) -> Optional[str]:
    """resolve() to a string, or None when the path is inaccessible."""
    try:
        return str(path.resolve())
    except OSError:
        return None


def is_hidden_model(*values: str | None) -> bool:
    """True if any id/path is the RAG embedding model (the effective embedder
    or its GGUF companion repo) or the llama.cpp install validation probe
    (ggml-org/models / stories260K), so pickers hide them (GGUF and non-GGUF).
    None are usable chat models; the probe can be cached as a side effect of
    installing the prebuilt llama-server and otherwise sorts smallest, so it
    would be auto-selected.

    Hub repo ids are matched EXACTLY (case-insensitive full "owner/name"), so a
    custom embedder with a generic basename like "org/model" cannot substring
    hide unrelated cached repos such as "user/model-chat" or "org/model-GGUF".
    A local-path embedder is matched by exact resolved path, and the probe by
    its exact on-disk filename."""
    from core.rag import config as rag_config

    hidden_repo_ids = {_PROBE_REPO_ID.lower()}
    exact_paths: list[str] = []
    for model in (
        rag_config.effective_embedding_model(),
        rag_config.effective_gguf_repo(),
    ):
        if _HF_REPO_ID_RE.match(model):
            hidden_repo_ids.add(model.lower())
        else:
            resolved = _safe_resolve(Path(model).expanduser())
            if resolved:
                exact_paths.append(resolved.lower())
    for v in values:
        if not v:
            continue
        low = v.lower()
        if _HF_REPO_ID_RE.match(v):
            # A repo id ("owner/name"): match the hidden set exactly. It is
            # never a filesystem path, so skip the path/filename checks.
            if low in hidden_repo_ids:
                return True
            continue
        # Anything else is treated as a filesystem path (the cached snapshot
        # path, or a local model id). Match the probe by its exact filename and
        # any configured local-path embedder by exact resolved path. Split on
        # both separators so a Windows-style path ("...\\stories260K.gguf") is
        # matched even when this runs on a POSIX interpreter (and vice versa).
        if low.replace("\\", "/").rsplit("/", 1)[-1] == _PROBE_FILENAME:
            return True
        if exact_paths:
            resolved = _safe_resolve(Path(v).expanduser())
            if resolved and resolved.lower() in exact_paths:
                return True
    return False
