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
from utils.transformers_dtype import dtype_kwargs

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


# Unsloth device -> torch device string. Apple has no torch device -> CPU.
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


class UnsafeEmbeddingModelError(RuntimeError):
    """Raised when the embedding model repo is flagged unsafe. A distinct type so the
    llama-server fallback paths re-raise it instead of masking a security block as a
    routine ST failure."""


def _ambient_hf_token() -> str | None:
    """The HF token the loader itself would use (HF_TOKEN env or the cached login), so
    the scan can reach a gated/private repo instead of failing open. None if unavailable."""
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _st_module_subdirs(name: str, token: str | None, local_only: bool) -> tuple[str, ...]:
    """The module directories a SentenceTransformer load reads weights from, so the online
    security scan scopes them as load roots (a flagged pickle directly under one must block).
    Two sources, both repo-controlled:

    * each module's non-empty ``path`` in ``modules.json`` (e.g. ``0_Transformer``), from which
      ST deserializes ``pytorch_model.bin``;
    * the child sub-modules of any ``Router`` (legacy ``Asym``) module. A Router declares its
      children only in ``router_config.json`` (``types`` maps ``{route}_{idx}_{ClassName}`` ->
      class), NOT in ``modules.json``, and ``Router.load()`` deserializes each from its own
      subdir -- so a flagged pickle in a child (``query_0_WordEmbeddings/pytorch_model.bin``) is
      a load root too. This mirrors the offline gate's ``_router_child_dirs`` expansion; without
      it the online scan would drop that child pickle as an unreferenced nested shard while the
      loader still deserialized it. The config is read only for a Router-typed module, so a plain
      embedder pays no extra fetch.

    Returns () on any failure (no modules.json, offline, malformed) so the guard never bricks the
    embedder. A traversing declared path (``0/../evil``) is dropped, matching the offline gate.

    ``local_only`` MUST be the value the caller captured for the load, not a fresh env read
    (``_hf_offline_if_dns_dead()`` flips the offline vars mid-load): re-reading could force
    this probe local-only, return (), and leave the scan with no module roots while the
    loader still deserializes a flagged pickle under ``0_Transformer/``.
    """
    try:
        import json
        import posixpath

        from utils.paths import is_local_path

        local_root = None
        if is_local_path(name):
            from pathlib import Path
            from utils.paths import normalize_path

            local_root = Path(normalize_path(name)).expanduser()

        def _read_repo_json(rel: str):
            """Parse a repo-relative JSON file (local dir or Hub download honoring
            ``local_only``), or None when missing / malformed / unreachable."""
            try:
                if local_root is not None:
                    path = local_root.joinpath(*rel.split("/"))
                    if not path.is_file():
                        return None
                    return json.loads(path.read_text())
                from huggingface_hub import hf_hub_download
                from huggingface_hub.utils import EntryNotFoundError

                try:
                    # CAPTURED predicate, never a fresh env read (see docstring); hf honors only
                    # HF_HUB_OFFLINE, so offline would otherwise block on timeouts.
                    local = hf_hub_download(
                        name, rel, token = token or None, local_files_only = local_only
                    )
                except EntryNotFoundError:
                    return None
                return json.loads(open(local).read())
            except Exception:
                return None

        def _safe_subdir(rel) -> str | None:
            """Canonical repo-relative subdir (``""`` = root), or None if it escapes the repo.
            Mirrors the offline gate so a traversing declared path cannot read or scope a
            directory outside the repo."""
            norm = posixpath.normpath(str(rel).strip().strip("/"))
            if norm in ("", "."):
                return ""
            if norm == ".." or norm.startswith("../"):
                return None
            return norm

        data = _read_repo_json("modules.json")
        if not isinstance(data, list):
            return ()
        subdirs = []
        for module in data:
            if not isinstance(module, dict):
                continue
            sub = str(module.get("path", "")).strip().strip("/")
            if sub:
                subdirs.append(sub)
            # Only a Router/Asym module hides load roots in router_config.json; the read is
            # scoped to it so a plain embedder incurs no extra fetch.
            if str(module.get("type", "")).rsplit(".", 1)[-1].lower() not in ("router", "asym"):
                continue
            prefix = _safe_subdir(sub)
            if prefix is None:
                continue
            cfg = _read_repo_json(posixpath.join(prefix, "router_config.json"))
            types = cfg.get("types") if isinstance(cfg, dict) else None
            if not isinstance(types, dict):
                continue
            for model_id in types:
                child = _safe_subdir(model_id)
                if child:
                    subdirs.append(posixpath.join(prefix, child))
        return tuple(dict.fromkeys(subdirs))
    except Exception:
        return ()


def _guard_model_security(name: str, local_only_load: bool) -> None:
    """Refuse to load a repo HF flagged as unsafe: a poisoned pickle deserializes inside
    SentenceTransformer regardless of trust_remote_code. Defense in depth behind the
    /settings gate (a name can also arrive via env/default); local paths and unreachable
    scans fail open inside evaluate_file_security. Never bricks the embedder on a gate error.

    ``local_only_load`` MUST equal the ``local_files_only`` the caller passes to
    SentenceTransformer -- it is what licenses skipping the Hub scan. ``_hf_offline_if_dns_dead()``
    mutates then restores the offline vars, so two reads can disagree and the scan could be
    skipped while the constructor fetched the unscanned repo.
    """
    try:
        from utils.security import evaluate_file_security, security_load_subdirs

        token = _ambient_hf_token()
        # Union the audio-model load roots with the ST module dirs so a flagged pickle
        # directly under a Transformer module dir (0_Transformer/) blocks instead of
        # passing as an unreferenced nested shard.
        load_subdirs = tuple(
            dict.fromkeys(
                (
                    *security_load_subdirs(name, token),
                    *_st_module_subdirs(name, token, local_only_load),
                )
            )
        )
        blocked = evaluate_file_security(
            name,
            hf_token = token,
            load_subdirs = load_subdirs,
            local_only_load = local_only_load,
        ).blocked
    except Exception:
        return
    if blocked:
        raise UnsafeEmbeddingModelError(
            f"Embedding model {name!r} is flagged as unsafe by Hugging Face's security "
            "scan; refusing to load. Set a different RAG embedding model."
        )


def _get(model_name: str | None = None):
    """Cached SentenceTransformer, (re)loading on a name change. Loaded in fp16
    for a ~1.5x speedup at negligible accuracy loss."""
    global _model, _name
    name = model_name or config.effective_embedding_model()
    with _lock:
        if _model is None or _name != name:
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer

            from utils.utils import hf_env_offline

            device = _device()
            # Resolve to the exact cache casing the offline local_files_only load needs. The
            # /settings route persists that spelling for a custom override but leaves the
            # configured default verbatim, so resolve here to cover the default too. No-op
            # for a local path or nothing cached.
            load_name = name
            from utils.paths import is_local_path

            if not is_local_path(load_name):
                from utils.models import resolve_st_cached_repo_id_case
                load_name = resolve_st_cached_repo_id_case(load_name)
            logger.info("loading embedding model %s on %s", load_name, device)
            # Read the offline state ONCE for both the gate and the loader:
            # _hf_offline_if_dns_dead() mutates then restores the offline vars, so re-reading
            # for the constructor could yield False after the guard skipped the Hub scan on
            # True.
            local_only = hf_env_offline()
            _guard_model_security(load_name, local_only)
            # huggingface_hub honors only HF_HUB_OFFLINE, so a TRANSFORMERS_OFFLINE-only
            # session would otherwise still fetch missing repo files.
            _model = SentenceTransformer(
                load_name,
                device = device,
                model_kwargs = dtype_kwargs("float16"),
                local_files_only = local_only,
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
        except UnsafeEmbeddingModelError:
            raise  # a security block must hard-fail, not fall back to llama-server
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
    except UnsafeEmbeddingModelError:
        raise  # a security block must hard-fail, not fall back to llama-server
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


def active_backend_is_llama() -> bool:
    """True when this process actually embeds via the llama-server (GGUF) backend.

    Reflects the ACTUAL built backend once one exists: an ``auto`` install that
    resolves to sentence-transformers but then falls back to llama-server at
    runtime (``_build_st_backend_or_fallback`` on a torch/CUDA load failure, or
    ``_switch_to_llama_fallback`` on an encode failure) loads only inert GGUF, so
    callers gating on the ST pickle must see llama here. Before any backend is
    built, defers to the resolver (``auto`` -> ``_resolve_auto()``, else the raw
    key) exactly as a fresh process would. Never raises: a backend probe must not
    block saving a model."""
    try:
        with _backend_lock:
            backend = _backend
        if backend is not None:
            # A backend exists: report what it ACTUALLY is. A concrete
            # sentence-transformers backend must return False even if the
            # resolver would now pick llama, so its pickle stays gated. If the
            # llama import fails we cannot be llama, so fall to the safe False.
            try:
                from .embed_llama_server import LlamaServerBackend
            except Exception:  # noqa: BLE001 - llama plumbing import must never block
                return False
            return isinstance(backend, LlamaServerBackend)
        raw = (config.EMBED_BACKEND or "auto").strip().lower()
        key = _resolve_auto() if raw in _AUTO_ALIASES else raw
        return key in _LLAMA_ALIASES
    except Exception:  # noqa: BLE001 - a backend probe must never block saving
        return False


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get_backend().warm(model_name = model_name)


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
