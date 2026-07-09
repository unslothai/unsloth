# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Infra-only model detection shared by the model routes and the hub
inventory. Lives in utils so hub services can use it without importing
routes.models (which has import-time side effects and would cycle)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

# Hub repo id shape ("owner/name", no leading separator); anything else is
# treated as a local filesystem path.
_HF_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def _safe_resolve(path: Path) -> Optional[str]:
    """resolve() to a string, or None when the path is inaccessible."""
    try:
        return str(path.resolve())
    except OSError:
        return None


def is_hidden_model(*values: str | None) -> bool:
    """True if any id/path is the RAG embedding model (EMBEDDING_MODEL or
    EMBED_GGUF_REPO basename) or the llama.cpp install validation probe
    (ggml-org/models / stories260K), so pickers hide them (GGUF and non-GGUF).
    None are usable chat models; the probe can be cached as a side effect of
    installing the prebuilt llama-server and otherwise sorts smallest, so it
    would be auto-selected. A local-path embedder is matched by exact resolved
    path only: a generic basename like "model" must not substring-hide
    unrelated chat models."""
    from core.rag import config as rag_config

    needles = [
        # The validation probe's repo (matches the cached repo id) and its exact
        # filename (matches the on-disk path). The filename carries the .gguf so
        # it does not hide unrelated repos like ``user/stories260K-finetune-GGUF``.
        "ggml-org/models",
        "stories260k.gguf",
    ]
    exact_paths: list[str] = []
    for model in (
        rag_config.effective_embedding_model(),
        rag_config.effective_gguf_repo(),
    ):
        if _HF_REPO_ID_RE.match(model):
            needles.append(model.split("/")[-1].lower())
        else:
            resolved = _safe_resolve(Path(model).expanduser())
            if resolved:
                exact_paths.append(resolved.lower())
    for v in values:
        if not v:
            continue
        low = v.lower()
        if any(n in low for n in needles):
            return True
        if exact_paths:
            resolved = _safe_resolve(Path(v).expanduser())
            if resolved and resolved.lower() in exact_paths:
                return True
    return False
