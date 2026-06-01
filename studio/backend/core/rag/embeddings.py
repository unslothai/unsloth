# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder singleton: thread-safe lazy SentenceTransformer keyed by model
name. ``token_counter`` reuses the model's tokenizer so chunk sizing matches the
embedder; ``warm()`` primes the load off the request path.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from . import config

logger = logging.getLogger(__name__)

# Default "false" also silences the fast tokenizer's "forked after parallelism"
# warning. encode() flips it to "true" only during a batch tokenize (rayon
# speedup) and restores it, keeping the speedup without the warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_lock = threading.Lock()
# Serializes encode/tokenize: the HF fast tokenizer uses interior mutability and
# is NOT thread-safe, so concurrent ingestion threads sharing this singleton
# panic with "Already borrowed". Separate from _lock so a long encode never
# blocks a (rare) reload.
_compute_lock = threading.Lock()
_model = None
_name: str | None = None


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - torch may be missing or broken
        return False


def _get(model_name: str | None = None):
    """Return the cached SentenceTransformer, (re)loading if the name changed."""
    global _model, _name
    name = model_name or config.EMBEDDING_MODEL
    with _lock:
        if _model is None or _name != name:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if _cuda() else "cpu"
            logger.info("loading embedding model %s on %s", name, device)
            _model = SentenceTransformer(name, device = device)
            _name = name
        return _model


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get(model_name)


def encode(texts: list[str], *, model_name: str | None = None, normalize: bool = True):
    """Embed texts into an (N, dim) numpy array. Serialized so concurrent
    ingestion threads don't trip the fast tokenizer's borrow check. Rayon
    parallelism is enabled only for this call and restored afterward."""
    model = _get(model_name)
    with _compute_lock:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            return model.encode(
                texts,
                normalize_embeddings = normalize,
                convert_to_numpy = True,
                show_progress_bar = False,
            )
        finally:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"


def dim(model_name: str | None = None) -> int:
    """Embedding dimension for the (loaded) model."""
    return _get(model_name).get_sentence_embedding_dimension()


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Callable counting tokens with the model's tokenizer. Counts under the
    compute lock since the same fast tokenizer backs encode and is not
    thread-safe."""
    tok = _get(model_name).tokenizer

    def _count(t: str) -> int:
        with _compute_lock:
            return len(tok.encode(t, add_special_tokens = False))

    return _count
