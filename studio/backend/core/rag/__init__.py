# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG core package for Unsloth Studio.

Callers import submodules lazily (``from core.rag import store``), so this
package pulls in no heavy deps (sentence-transformers, torch, PyMuPDF). Keep it
free of top-level submodule imports.
"""

__all__ = [
    "config",
    "parsers",
    "chunking",
    "embeddings",
    "store",
    "retrieval",
    "tool",
    "ingestion",
]
