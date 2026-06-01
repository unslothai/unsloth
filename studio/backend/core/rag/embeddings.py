# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder facade. Public functions (``encode``, ``token_counter``,
``dim``, ``warm``) dispatch to a process-wide backend chosen by
``config.EMBED_BACKEND``:

* ``sentence-transformers`` (default) -- thread-safe lazy SentenceTransformer
  singleton keyed by model name; needs torch. ``token_counter`` reuses the
  model's tokenizer so chunk sizing matches the embedder.
* ``llama-server`` -- a GGUF embedder served over HTTP by the bundled
  llama.cpp; no torch (see ``embed_llama_server``).

Backends produce different vectors, so switching ``RAG_EMBED_BACKEND`` requires
rebuilding the index (drop/re-create KBs, re-upload thread docs). Startup
failure of the selected backend raises -- never a silent cross-backend fallback,
which would corrupt retrieval by mixing vector spaces.
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


def _st_encode(
    texts: list[str], *, model_name: str | None = None, normalize: bool = True
):
    """SentenceTransformers encode -> (N, dim) float32. Serialized so concurrent
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


def _st_dim(model_name: str | None = None) -> int:
    """SentenceTransformers embedding dimension for the (loaded) model."""
    return _get(model_name).get_sentence_embedding_dimension()


def _st_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Callable counting tokens with the model's tokenizer. Counts under the
    compute lock since the same fast tokenizer backs encode and is not
    thread-safe."""
    tok = _get(model_name).tokenizer

    def _count(t: str) -> int:
        with _compute_lock:
            return len(tok.encode(t, add_special_tokens = False))

    return _count


class _SentenceTransformersBackend:
    """Default backend: delegates to the module-level ST helpers above so the
    existing ``_get`` monkeypatch in tests keeps working."""

    def encode(self, texts, *, model_name = None, normalize = True):
        return _st_encode(texts, model_name = model_name, normalize = normalize)

    def token_counter(self, *, model_name = None):
        return _st_token_counter(model_name)

    def dim(self, *, model_name = None):
        return _st_dim(model_name)

    def warm(self, *, model_name = None):
        _get(model_name)


# ── Backend selection ─────────────────────────────────────────────

_backend_lock = threading.Lock()
_backend = None
_backend_key: str | None = None

_ST_ALIASES = frozenset({"sentence-transformers", "sentence_transformers", "st"})
_LLAMA_ALIASES = frozenset(
    {"llama-server", "llama_server", "llama", "llama.cpp", "llamacpp", "gguf"}
)
_AUTO_ALIASES = frozenset({"auto", ""})


def _resolve_auto() -> str:
    """Pick a backend for ``auto``: sentence-transformers when a CUDA/ROCm GPU is
    present (torch fp16 wins bulk indexing there), else the torch-free GGUF
    llama-server -- unless its binary is missing, then fall back to ST. Reuses the
    chat backend's static probes (nvidia-smi first, so the GPU check stays
    torch-free)."""
    from core.inference.llama_cpp import LlamaCppBackend

    if LlamaCppBackend._get_gpu_free_memory():
        return "sentence-transformers"
    if LlamaCppBackend._find_llama_server_binary():
        return "llama-server"
    return "sentence-transformers"


def _get_backend():
    """Return the process-wide embedding backend for ``config.EMBED_BACKEND``,
    building it once. Cached by the raw config value so ``auto`` detection runs
    only on a cache miss; rebuilds when the config changes (e.g. in tests)."""
    global _backend, _backend_key
    raw = (config.EMBED_BACKEND or "auto").strip().lower()
    with _backend_lock:
        if _backend is not None and _backend_key == raw:
            return _backend
        key = _resolve_auto() if raw in _AUTO_ALIASES else raw
        if key in _ST_ALIASES:
            _backend = _SentenceTransformersBackend()
        elif key in _LLAMA_ALIASES:
            # Imported lazily so the ST path never imports llama plumbing.
            from .embed_llama_server import LlamaServerBackend

            _backend = LlamaServerBackend()
        else:
            raise ValueError(
                f"Unknown RAG_EMBED_BACKEND={config.EMBED_BACKEND!r}; expected "
                "'auto', 'sentence-transformers' or 'llama-server'"
            )
        _backend_key = raw
        return _backend


def _reset_backend() -> None:
    """Drop the cached backend (test teardown / explicit re-init)."""
    global _backend, _backend_key
    with _backend_lock:
        _backend = None
        _backend_key = None


# ── Public API (dispatches to the selected backend) ───────────────


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get_backend().warm(model_name = model_name)


def encode(texts: list[str], *, model_name: str | None = None, normalize: bool = True):
    """Embed texts into an (N, dim) float32 numpy array."""
    return _get_backend().encode(texts, model_name = model_name, normalize = normalize)


def dim(model_name: str | None = None) -> int:
    """Embedding dimension for the (loaded) model."""
    return _get_backend().dim(model_name = model_name)


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Callable counting tokens with the embedder's own tokenizer."""
    return _get_backend().token_counter(model_name = model_name)
