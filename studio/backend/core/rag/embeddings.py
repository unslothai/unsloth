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

Torch driver faults bypass Python handlers, so ``_load_device`` probes allocation
in a child and falls back to CPU without changing the embedding space.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from contextlib import contextmanager
from functools import lru_cache
from typing import Callable

from utils.hardware.hardware import DeviceType, get_device
from utils.transformers_dtype import dtype_kwargs
from utils.utils import hf_env_offline

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


class TorchDeviceUnusableError(RuntimeError):
    """Raised when torch cannot allocate safely on the accelerator or CPU."""


def _load_device() -> str:
    """Choose a device after probing for fatal torch driver failures in a child.

    Fall back to CPU to preserve the embedding space. Raise only if CPU also
    crashes, allowing the caller to select the GGUF backend."""
    device = _device()
    if device == "cpu":
        return device

    from utils.torch_device_probe import device_can_allocate

    if device_can_allocate(device):
        return device
    if device_can_allocate("cpu"):
        logger.warning(
            "torch cannot allocate on %s without crashing; loading the embedding model "
            "on CPU instead. This install's torch build does not match this machine.",
            device,
        )
        return "cpu"
    raise TorchDeviceUnusableError(
        f"torch crashes when allocating on {device}; this install's torch build does "
        "not match this machine"
    )


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


def _st_module_subdirs(name: str, token: str | None) -> tuple[str, ...]:
    """The module directories a SentenceTransformer load reads weights from, taken from
    the repo's ``modules.json`` (each module's non-empty ``path``, e.g. ``0_Transformer``).
    ST deserializes ``pytorch_model.bin`` from these dirs, so they are load roots for the
    security scan: a flagged pickle directly under one must block. Returns () on any
    failure (no modules.json, offline, malformed) so the guard never bricks the embedder.
    """
    try:
        import json

        from utils.paths import is_local_path

        if is_local_path(name):
            from pathlib import Path
            from utils.paths import normalize_path

            path = Path(normalize_path(name)).expanduser() / "modules.json"
            if not path.is_file():
                return ()
            data = json.loads(path.read_text(encoding = "utf-8-sig"))
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError
            from utils.hf_cache_settings import active_hf_hub_cache

            try:
                local = hf_hub_download(
                    name,
                    "modules.json",
                    token = token or None,
                    cache_dir = active_hf_hub_cache(),
                )
            except EntryNotFoundError:
                return ()
            data = json.loads(open(local, encoding = "utf-8-sig").read())
        subdirs = []
        for module in data or ():
            sub = str((module or {}).get("path", "")).strip().strip("/")
            if sub:
                subdirs.append(sub)
        return tuple(dict.fromkeys(subdirs))
    except Exception:
        return ()


def _guard_model_security(name: str, local_only: bool = False) -> None:
    """Refuse to load a repo HF flagged as unsafe: a poisoned pickle deserializes inside
    SentenceTransformer regardless of trust_remote_code. Defense in depth behind the
    /settings gate (a name can also arrive via env/default); local paths and unreachable
    scans fail open inside evaluate_file_security. Never bricks the embedder on a gate error.

    ``local_only`` (offline) inspects the local cache; subdir probes are skipped (they'd hit the
    network and hang, and the offline gate walks the whole snapshot anyway).
    """
    try:
        from utils.security import evaluate_file_security, security_load_subdirs

        token = _ambient_hf_token()
        if local_only:
            load_subdirs = ()
        else:
            # Union audio-model load roots with ST module dirs so a flagged pickle under a
            # Transformer module dir blocks instead of passing as an unreferenced nested shard.
            load_subdirs = tuple(
                dict.fromkeys(
                    (*security_load_subdirs(name, token), *_st_module_subdirs(name, token))
                )
            )
        blocked = evaluate_file_security(
            name, hf_token = token, load_subdirs = load_subdirs, local_only_load = local_only
        ).blocked
    except Exception:
        return
    if blocked:
        reason = (
            "has cached pickle weights that cannot be security-scanned offline and no "
            "safetensors alternative"
            if local_only
            else "is flagged as unsafe by Hugging Face's security scan"
        )
        raise UnsafeEmbeddingModelError(
            f"Embedding model {name!r} {reason}; refusing to load. "
            "Set a different RAG embedding model."
        )


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _CaptureLoadReport(logging.Filter):
    """Swallow transformers' multi-line "<Model> LOAD REPORT" table, keeping the text.

    transformers >= 5 emits the report through ``logger.warning`` with embedded ANSI
    colour codes, so it lands in the server log as ~7 unstructured lines that break
    every JSON consumer. It fires on every boot for the RAG embedder because
    bge-small-en-v1.5 ships a legacy ``embeddings.position_ids`` key that the current
    BertModel does not expect, which is benign and identical every time.

    Nothing is lost: the caller re-emits the report (see ``_quiet_transformers_load``)
    at debug when it only reports that known legacy key, and at warning when it
    mentions anything that could change the model's behaviour.
    """

    _SERIOUS = ("MISSING", "MISMATCH", "CONVERSION")
    # The only UNEXPECTED key worth downgrading is the legacy buffer every BERT-era
    # sentence-transformer ships. Any other discarded weight can genuinely change
    # retrieval quality, so it stays a warning. Matched on the whole key, not as a
    # substring: "encoder.position_ids_projection.weight" is a real discarded weight.
    _KNOWN_BENIGN_UNEXPECTED = "embeddings.position_ids"

    def __init__(self) -> None:
        super().__init__()
        self.reports: list[str] = []
        # These filters sit on process-global loggers, so a concurrent load on another
        # thread would otherwise have its report swallowed and attributed here. Only
        # capture what the thread that opened the context emits.
        self.thread_id = threading.get_ident()

    def filter(self, record: logging.LogRecord) -> bool:
        if threading.get_ident() != self.thread_id:
            return True
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001 - a broken record must not break loading
            return True
        if "LOAD REPORT" not in msg:
            return True
        self.reports.append(msg)
        return False

    def is_serious(self) -> bool:
        for report in self.reports:
            if any(tag in report for tag in self._SERIOUS):
                return True
            # Row by row over the table only. transformers appends a "Notes:" section
            # explaining each status ("- UNEXPECTED: can be ignored when loading from
            # a different task/architecture"), and reading that as a key row would make
            # every unexpected report serious, including the benign one.
            table = report.split("Notes:", 1)[0]
            for row in table.splitlines():
                if "UNEXPECTED" not in row:
                    continue
                key = _ANSI_RE.sub("", row).split("|", 1)[0].strip()
                if key != self._KNOWN_BENIGN_UNEXPECTED and not key.endswith(
                    "." + self._KNOWN_BENIGN_UNEXPECTED
                ):
                    return True
        return False


_LOAD_REPORT_LOGGERS = (
    "transformers.utils.loading_report",
    "transformers.modeling_utils",
    # An adapter-backed embedding model reports through the PEFT integration's own
    # logger, which is not a descendant of either of the above.
    "transformers.integrations.peft",
)


@contextmanager
def _quiet_transformers_load():
    """Keep a transformers weight load from writing raw ANSI/tqdm output to stdout.

    Scoped to the embedder load only, so a user-visible model load keeps its normal
    progress bar and report. Restores the progress-bar setting exactly as found, so
    a caller that had already disabled bars stays disabled.
    """
    capture = _CaptureLoadReport()
    attached = []
    for name in _LOAD_REPORT_LOGGERS:
        log = logging.getLogger(name)
        log.addFilter(capture)
        attached.append(log)

    # The weight-load bar is transformers.utils.logging.tqdm, so disable_progress_bar()
    # reaches it. The "is it on right now" probe has been spelled both ways across
    # versions (is_progress_bar_enabled in 4.x/5.x, are_progress_bars_disabled in
    # some builds), so accept either and skip the restore when neither exists rather
    # than re-enabling a bar the caller had deliberately turned off.
    # transformers' enable_progress_bar() also calls the Hub's enable_progress_bars(),
    # so restoring the transformers flag would clobber a Hub-only disable that someone
    # else installed (unsloth does exactly that in patch_ipykernel_hf_xet for the
    # broken hf-xet/ipykernel pair). Snapshot the Hub state separately and put it back.
    hub_bars_off = None
    try:
        from huggingface_hub.utils import are_progress_bars_disabled
        hub_bars_off = bool(are_progress_bars_disabled())
    except Exception:  # noqa: BLE001 - no Hub, or a version without the probe
        hub_bars_off = None

    reenable = False
    hf_logging = None
    try:
        from transformers.utils import logging as hf_logging
        if hasattr(hf_logging, "is_progress_bar_enabled"):
            was_on = bool(hf_logging.is_progress_bar_enabled())
        elif hasattr(hf_logging, "are_progress_bars_disabled"):
            was_on = not bool(hf_logging.are_progress_bars_disabled())
        else:
            was_on = False
        if was_on:
            hf_logging.disable_progress_bar()
            reenable = True
    except Exception:  # noqa: BLE001 - older/absent transformers: nothing to disable
        hf_logging = None

    try:
        yield capture
    finally:
        for log in attached:
            log.removeFilter(capture)
        if reenable and hf_logging is not None:
            try:
                hf_logging.enable_progress_bar()
            except Exception:  # noqa: BLE001
                pass
            if hub_bars_off:
                try:
                    from huggingface_hub.utils import disable_progress_bars
                    disable_progress_bars()
                except Exception:  # noqa: BLE001
                    pass


def _one_line(text: str) -> str:
    """The report as a single plain-text line.

    This module logs through the stdlib logger, not structlog, so re-emitting the
    captured table verbatim would put the ANSI escapes and embedded newlines straight
    back into the server log, which is the thing being fixed.
    """
    plain = _ANSI_RE.sub("", text)
    return " | ".join(part.strip() for part in plain.splitlines() if part.strip())


def _emit_load_reports(report) -> None:
    """Re-emit what the filter swallowed, as one record on our own logger: debug for
    the expected legacy-key notice, warning for anything that could change the
    embeddings. Drains the list so a retry does not report the same lines twice."""
    serious = report.is_serious()
    for text in report.reports:
        if serious:
            logger.warning("embedding model load report: %s", _one_line(text))
        else:
            logger.debug("embedding model load report: %s", _one_line(text))
    report.reports.clear()


def _st_accepts_local_files_only(st_cls) -> bool:
    """Whether this SentenceTransformer version accepts local_files_only; passing it to an
    older constructor raises, so gate on the signature."""
    try:
        import inspect
        return "local_files_only" in inspect.signature(st_cls.__init__).parameters
    except Exception:
        return False


def _get(model_name: str | None = None):
    """Cached SentenceTransformer, (re)loading on a name change. Loaded in fp16
    for a ~1.5x speedup at negligible accuracy loss."""
    global _model, _name
    name = model_name or config.effective_embedding_model()
    # Capture offline state once so the gate and the load agree (no window where the gate is
    # skipped as offline but the constructor then reaches the network).
    local_only = hf_env_offline()
    with _lock:
        if _model is None or _name != name:
            # Probe before loading sentence-transformers on the selected device.
            device = _load_device()
            degraded_to_cpu = device == "cpu" and _device() != "cpu"
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer
            from utils.hf_cache_settings import active_hf_hub_cache

            logger.info("loading embedding model %s on %s", name, device)
            _guard_model_security(name, local_only)
            st_kwargs = dict(
                device = device,
                cache_folder = active_hf_hub_cache(),
                model_kwargs = dtype_kwargs("float32" if degraded_to_cpu else "float16"),
            )
            load_target = name
            if local_only:
                from utils.utils import hf_cache_snapshot_dir
                snapshot = hf_cache_snapshot_dir(name)
                if snapshot is not None:
                    # Load from the local snapshot dir: a local path never touches the Hub, so
                    # this is offline-safe on ANY sentence-transformers version (even ones
                    # predating local_files_only).
                    load_target = str(snapshot)
                elif _st_accepts_local_files_only(SentenceTransformer):
                    st_kwargs["local_files_only"] = True
            with _quiet_transformers_load() as report:
                # The re-emit runs in finally: a load that raises after transformers
                # wrote its report is exactly when a MISSING or MISMATCH line matters,
                # and letting the exception skip the loop would swallow it.
                try:
                    _model = SentenceTransformer(load_target, **st_kwargs)
                finally:
                    _emit_load_reports(report)
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

    # Unfiltered probe on purpose: the winner here runs under PyTorch, so the
    # ROCm arch gate (which asks what the installed llama.cpp prebuilt was built
    # for, #7624) must not apply. A device that prebuilt lacks kernels for is
    # usually still a perfectly good sentence-transformers device.
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
