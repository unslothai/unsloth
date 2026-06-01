# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder singleton: thread-safe lazy SentenceTransformer keyed by model
name. ``token_counter`` reuses the model's tokenizer so chunk sizing matches the
embedder; ``warm()`` primes the load off the request path at startup.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

from . import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
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
    """Embed a batch of texts into an (N, dim) numpy array."""
    return _get(model_name).encode(
        texts,
        normalize_embeddings = normalize,
        convert_to_numpy = True,
        show_progress_bar = False,
    )


def dim(model_name: str | None = None) -> int:
    """Embedding dimension for the (loaded) model."""
    return _get(model_name).get_sentence_embedding_dimension()


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Return a callable counting tokens with the model's own tokenizer."""
    tok = _get(model_name).tokenizer
    return lambda t: len(tok.encode(t, add_special_tokens = False))
