# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder facade dispatching to a process-wide backend from
``config.EMBED_BACKEND`` (``auto`` picks by hardware): ``sentence-transformers``
(torch) or ``llama-server`` (GGUF, no torch).

Backends produce different vectors, so switching requires rebuilding the index. We
degrade to llama.cpp rather than crash when ST breaks on a machine: an init-time
probe falls back before any vector is produced (so spaces can't mix), and a
runtime ``encode`` failure swaps the process to llama-server for the rest of its
life (KBs already embedded with ST should then be reindexed).
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

# "false" silences the fast tokenizer's fork warning; encode() flips it to "true"
# only during a batch tokenize (rayon speedup), then restores it.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_lock = threading.Lock()
# Serializes encode/tokenize (HF fast tokenizer isn't thread-safe). Separate from
# _lock so a long encode never blocks a reload.
_compute_lock = threading.Lock()
_model = None
_name: str | None = None


# Studio device -> torch device string. Apple has no torch device -> CPU.
_TORCH_DEVICE = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}


def _device() -> str:
    return _TORCH_DEVICE.get(get_device(), "cpu")


_torchao_stub_done = False


def _install_torchao_stub_once() -> None:
    """Neutralize torchao before importing sentence-transformers. On Windows ROCm,
    torchao (pulled in by transformers.quantizers) imports an absent c10d backend
    and aborts, dropping the embedder to llama-server. Workers stub it too; the
    embedder runs in the main process. No-op elsewhere; runs once under ``_lock``."""
    global _torchao_stub_done
    if _torchao_stub_done:
        return
    _torchao_stub_done = True
    from core._torchao_stub import install_torchao_windows_rocm_stub

    install_torchao_windows_rocm_stub()


def _get(model_name: str | None = None):
    """Cached SentenceTransformer, (re)loading on a name change. Loaded in fp16
    for a ~1.5x speedup at negligible accuracy loss."""
    global _model, _name
    name = model_name or config.EMBEDDING_MODEL
    with _lock:
        if _model is None or _name != name:
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer

            device = _device()
            logger.info("loading embedding model %s on %s", name, device)
            _model = SentenceTransformer(
                name, device = device, model_kwargs = {"torch_dtype": "float16"}
            )
            _name = name
        return _model


@lru_cache(maxsize = 1)
def _inference_ctx_factory():
    """``torch.inference_mode`` if torch imports, else ``nullcontext``. Returns the
    factory so each call gets a fresh single-use guard."""
    try:
        import torch
        return torch.inference_mode
    except Exception:  # noqa: BLE001 - torch may be missing or broken
        from contextlib import nullcontext
        return nullcontext


def _inference_ctx():
    return _inference_ctx_factory()()


def _st_encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """ST encode -> (N, dim) float32. Serialized (fast-tokenizer borrow check),
    under inference_mode when torch is present, with rayon enabled for the call."""
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
    return _get(model_name).get_sentence_embedding_dimension()


def _st_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Token counter using the model's tokenizer, under the compute lock (the same
    fast tokenizer backs encode and isn't thread-safe), with rayon enabled for the
    call. Mirrors ``_st_encode``."""
    tok = _get(model_name).tokenizer

    def _count(t: str) -> int:
        with _compute_lock:
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            try:
                return len(tok.encode(t, add_special_tokens = False))
            finally:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return _count


def _st_unload() -> None:
    """Drop the cached ST model and free its GPU memory. The empty_cache call only
    runs when a model was loaded (a CUDA context already exists), so it never
    creates one on a process that never embedded."""
    global _model, _name
    with _lock:
        had_model = _model is not None
        _model = None
        _name = None
    if not had_model:
        return
    import gc

    gc.collect()  # drop the model now so the freed VRAM is visible to callers
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - torch may be missing or broken
        pass


class _SentenceTransformersBackend:
    """Default backend; delegates to the module-level ST helpers so the ``_get``
    monkeypatch in tests keeps working."""

    def encode(
        self,
        texts,
        *,
        model_name = None,
        normalize = True,
    ):
        try:
            return _st_encode(texts, model_name = model_name, normalize = normalize)
        except Exception as st_err:  # noqa: BLE001 - runtime ST/CUDA encode failure
            # ST loaded but this encode blew up; swap the process to the llama-server
            # embedder (so later encodes stay in one space) and retry.
            fallback = _switch_to_llama_fallback(st_err)
            if fallback is None:
                raise
            return fallback.encode(texts, model_name = model_name, normalize = normalize)

    def token_counter(self, *, model_name = None):
        return _st_token_counter(model_name)

    def dim(self, *, model_name = None):
        return _st_dim(model_name)

    def warm(self, *, model_name = None):
        _get(model_name)

    def unload(self):
        """Drop the cached ST model and free its GPU memory (see module unload)."""
        _st_unload()


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
    present (torch fp16 wins bulk indexing), else the torch-free GGUF llama-server
    -- or ST if its binary is missing. GPU check is torch-free (nvidia-smi)."""
    from core.inference.llama_cpp import LlamaCppBackend

    if LlamaCppBackend._get_gpu_free_memory():
        return "sentence-transformers"
    if LlamaCppBackend._find_llama_server_binary():
        return "llama-server"
    return "sentence-transformers"


def _try_make_llama_backend():
    """A llama-server GGUF embedding backend if its binary is present, else None.
    Construction is lazy -- no server starts until warm."""
    from core.inference.llama_cpp import LlamaCppBackend

    if not LlamaCppBackend._find_llama_server_binary():
        return None
    from .embed_llama_server import LlamaServerBackend

    return LlamaServerBackend()


def _build_st_backend_or_fallback():
    """Build the ST backend, probing it by loading the model now. If the probe
    raises (no torch, CUDA mismatch, bad wheel) and the GGUF llama-server embedder
    is available, fall back to it. The probe runs before any vector is produced, so
    this never mixes spaces. Re-raises if no embedder can start."""
    backend = _SentenceTransformersBackend()
    try:
        backend.warm(model_name = None)
        return backend
    except Exception as st_err:  # noqa: BLE001 - any ST/torch import or load failure
        fallback = _try_make_llama_backend()
        if fallback is None:
            raise
        logger.warning(
            "sentence-transformers embedder unavailable (%s); falling back to the "
            "llama-server GGUF embedder",
            st_err,
        )
        return fallback


def _switch_to_llama_fallback(err):
    """An ST encode failed at runtime even though the model had loaded. Swap the
    process embedder to llama-server so every later encode stays in one space, and
    return it (None if no binary). Vectors written before the swap were ST, so any
    KB already embedded with ST should be reindexed."""
    global _backend, _backend_key
    with _backend_lock:
        if not isinstance(_backend, _SentenceTransformersBackend):
            return _backend  # another thread already swapped (or was never ST)
        fallback = _try_make_llama_backend()
        if fallback is None:
            return None
        logger.warning(
            "sentence-transformers encode failed (%s); switching to the llama-server "
            "embedder for the rest of this process. Reindex any knowledge base that "
            "was already embedded with sentence-transformers.",
            err,
        )
        _backend = fallback
        _backend_key = (config.EMBED_BACKEND or "auto").strip().lower()
        return fallback


def _get_backend():
    """The process-wide embedding backend for ``config.EMBED_BACKEND``, built once.
    Cached by the raw config value, so ``auto`` detection runs only on a miss and a
    config change rebuilds it."""
    global _backend, _backend_key
    raw = (config.EMBED_BACKEND or "auto").strip().lower()
    with _backend_lock:
        if _backend is not None and _backend_key == raw:
            return _backend
        key = _resolve_auto() if raw in _AUTO_ALIASES else raw
        if key in _ST_ALIASES:
            _backend = _build_st_backend_or_fallback()
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
    """Drop the cached backend (test teardown / re-init)."""
    global _backend, _backend_key
    with _backend_lock:
        _backend = None
        _backend_key = None


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get_backend().warm(model_name = model_name)


def unload() -> None:
    """Free the active embedder's GPU memory so a GGUF inference model can allocate
    with full VRAM (a resident embedder otherwise pushes the auto-context pick past
    the driver's spill threshold). Dispatches to the backend: sentence-transformers
    drops its cached model, the llama-server embedder kills its subprocess. A no-op
    when no backend is built; the embedder reloads lazily on the next embed/warm. A
    concurrent in-flight encode keeps its own reference, so its VRAM frees once that
    call returns."""
    with _backend_lock:
        backend = _backend
    if backend is not None:
        backend.unload()


def encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """Embed texts into an (N, dim) float32 numpy array."""
    return _get_backend().encode(texts, model_name = model_name, normalize = normalize)


def dim(model_name: str | None = None) -> int:
    """Embedding dimension for the (loaded) model."""
    return _get_backend().dim(model_name = model_name)


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Callable counting tokens with the embedder's own tokenizer."""
    return _get_backend().token_counter(model_name = model_name)
