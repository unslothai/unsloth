# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""RAG core package. Import submodules lazily; keep this free of top-level
submodule imports to avoid pulling in heavy deps."""

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
