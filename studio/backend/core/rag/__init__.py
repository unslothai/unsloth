# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Retrieval-augmented-generation (RAG) core package for Unsloth Studio.

Submodules are imported lazily by callers (e.g. ``from core.rag import store``)
so importing this package never pulls in heavy optional dependencies such as
sentence-transformers, torch or PyMuPDF. Keep this file free of top-level
imports of those submodules.
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
