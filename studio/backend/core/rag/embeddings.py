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
from functools import lru_cache
from typing import Callable

from utils.hardware.hardware import DeviceType, get_device

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


# Map Studio's detected backend (utils.hardware) to a torch/sentence-transformers
# device string. MLX (Apple via the MLX framework, not torch) has no torch
# device, so the torch-based embedder falls back to CPU there.
_TORCH_DEVICE = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}
_GPU_DEVICES = frozenset({"cuda", "xpu"})


def _device() -> str:
    """Embedder device from Studio's hardware detection (CUDA/XPU/CPU)."""
    return _TORCH_DEVICE.get(get_device(), "cpu")


def _get(model_name: str | None = None):
    """Return the cached SentenceTransformer, (re)loading if the name changed.
    On a GPU backend the model runs in fp16 (``half()``) for a ~1.5x encode
    speedup at negligible accuracy loss; CPU stays fp32 (half is unsupported
    there)."""
    global _model, _name
    name = model_name or config.EMBEDDING_MODEL
    with _lock:
        if _model is None or _name != name:
            from sentence_transformers import SentenceTransformer

            device = _device()
            logger.info("loading embedding model %s on %s", name, device)
            _model = SentenceTransformer(name, device = device)
            if device in _GPU_DEVICES:
                _model = _model.half()
            _name = name
        return _model


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get(model_name)


@lru_cache(maxsize = 1)
def _inference_ctx_factory():
    """Pick the encode context once (cached): ``torch.inference_mode`` if torch
    is importable, else ``contextlib.nullcontext``. Returns the factory, not an
    instance, so each call still gets a fresh single-use guard."""
    try:
        import torch

        return torch.inference_mode
    except Exception:  # noqa: BLE001 - torch may be missing or broken
        from contextlib import nullcontext

        return nullcontext


def _inference_ctx():
    """Fresh inference (or no-op) context from the cached factory."""
    return _inference_ctx_factory()()


def encode(texts: list[str], *, model_name: str | None = None, normalize: bool = True):
    """Embed texts into an (N, dim) float32 numpy array. Serialized so concurrent
    ingestion threads don't trip the fast tokenizer's borrow check; runs under
    inference_mode when torch is available. Rayon parallelism is enabled only for
    this call and restored afterward."""
    model = _get(model_name)
    with _compute_lock:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        try:
            with _inference_ctx():
                out = model.encode(
                    texts,
                    normalize_embeddings = normalize,
                    convert_to_numpy = True,
                    show_progress_bar = False,
                )
        finally:
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # fp16 weights yield fp16 output; store float32 for sqlite-vec + stable cosine.
    if hasattr(out, "astype"):
        out = out.astype("float32", copy = False)
    return out


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
