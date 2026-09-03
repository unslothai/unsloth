# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder facade dispatching to a process-wide backend from
``config.EMBED_BACKEND`` (``auto`` picks by hardware): ``sentence-transformers``
(torch) or ``llama-server`` (GGUF, no torch).

Either way the embedder stays off the GPU unless asked: this one runs in the backend
process, where a CUDA context outlives every unload, and the other runs in a child.
See ``_device``.

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

# "false" silences the fast tokenizer's fork warning; encode() flips it only during a batch tokenize.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_lock = threading.Lock()
# Serializes encode/tokenize (the HF fast tokenizer is not thread-safe); separate from _lock so a
# long encode never blocks a reload.
_compute_lock = threading.Lock()
_model = None
_name: str | None = None
# The process embedder can swap between an encode returning and the caller asking, so the answer
# must be the backend actually used. See encode_with_identity.
_served_by = threading.local()


# Unsloth device -> torch device string. Apple has no torch device -> CPU.
_TORCH_DEVICE = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}


def _device() -> str:
    """Torch device for the in-process embedder. CPU unless asked otherwise.

    Defaulting a GPU machine to CPU is deliberate. This embedder runs inside the
    backend process, and the first CUDA allocation there creates a primary context
    that is never returned while the process lives: measured at 712 MiB on a B200,
    against 74 MiB for bge-small's own weights. So ingesting one document used to
    cost most of a gigabyte of VRAM for the rest of the session, on a machine where
    the user had loaded no model at all, and no amount of unloading gets it back --
    ``del model; torch.cuda.empty_cache()`` returns none of it.

    The trade is real but small at the sizes this runs at. bge-small is a 33M parameter
    BERT: on the same host, one 128-token chunk takes 18.7ms on CPU against 5.2ms on
    CUDA, which is noise next to parsing and chunking the document it came from. Bulk
    indexing is where it shows, at batch 64: 445 chunks/s on CPU against 3174/s on CUDA.
    ``RAG_EMBED_DEVICE=gpu`` opts back in for a large corpus.

    This reads the same setting as the llama-server backend but resolves ``auto``
    differently, which is intended: that backend offloads inside its own subprocess,
    where the context dies with the child and costs the backend nothing.
    """
    if config.embed_device_preference() != "gpu":
        return "cpu"
    # Still a table lookup: asking for a GPU on a host without one lands on CPU rather than on a device
    # string torch cannot open.
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
# Its own lock, not _lock: that one is held across a whole model construction, so borrowing it made
# a preflight probe wait out someone else's download.
_stub_lock = threading.Lock()


def _install_torchao_stub_once() -> None:
    """Neutralize torchao before importing sentence-transformers. On Windows ROCm,
    torchao (pulled in by transformers.quantizers) imports an absent c10d backend
    and aborts, dropping the embedder to llama-server. Workers stub it too; the
    embedder runs in the main process. No-op elsewhere; runs once."""
    global _torchao_stub_done
    with _stub_lock:
        if _torchao_stub_done:
            return
        _torchao_stub_done = True
        from core._torchao_stub import install_torchao_windows_rocm_stub

        install_torchao_windows_rocm_stub()


class UnsafeEmbeddingModelError(RuntimeError):
    """Raised when the embedding model repo is flagged unsafe. A distinct type so the
    llama-server fallback paths re-raise it instead of masking a security block as a
    routine ST failure."""


class EmbeddingModelDownloadRequiredError(RuntimeError):
    """The picker activated a model whose explicit transfer is still pending."""


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
            # Union the load roots so a flagged pickle under a Transformer module dir blocks instead of passing
            # as an unreferenced nested shard.
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
    # Only the legacy BERT-era buffer is downgraded, matched on the whole key: any other discarded
    # weight can genuinely change retrieval quality.
    _KNOWN_BENIGN_UNEXPECTED = "embeddings.position_ids"

    def __init__(self) -> None:
        super().__init__()
        self.reports: list[str] = []
        # These filters sit on process-global loggers, so capture only what the thread that opened the context emits.
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
            # Row by row over the table only: transformers appends a "Notes:" section explaining each status
            # ("- UNEXPECTED: can be ignored ..."), whose lines would read as serious key rows.
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
    # An adapter-backed embedding model reports through the PEFT integration's own logger, not a
    # descendant of either above.
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

    # The "is it enabled" probe is spelled both ways across transformers versions, so accept either and
    # skip the restore when neither exists.
    # transformers' enable_progress_bar() also calls the Hub's, so snapshot and restore the Hub state
    # separately or a Hub-only disable is clobbered.
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
    """Cached SentenceTransformer, (re)loading on a name change. Loaded in fp16 on an
    accelerator for a ~1.5x speedup at negligible accuracy loss, fp32 on CPU."""
    global _model, _name
    name = model_name or config.effective_embedding_model()
    # Capture offline state once so the gate and the load agree.
    try:
        from utils.embedding_model_settings import get_stored_download_pending
        download_pending = get_stored_download_pending(name)
    except Exception:  # noqa: BLE001 - old/unavailable settings store
        download_pending = False
    offline = hf_env_offline()
    # local_only means "load from cache"; offline is what the security gate needs, and claiming offline
    # while online rejects a .bin-only repo the resolver just scanned.
    local_only = offline or download_pending
    with _lock:
        if _model is None or _name != name:
            # Probe before loading sentence-transformers on the selected device.
            device = _load_device()
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer
            from utils.hf_cache_settings import active_hf_hub_cache

            logger.info("loading embedding model %s on %s", name, device)
            st_kwargs = dict(
                device = device,
                cache_folder = active_hf_hub_cache(),
                # Keyed on the device we load on: fp16 BERT on CPU raises "not implemented for Half", which encode()
                # answers by swapping the whole process to llama-server.
                model_kwargs = dtype_kwargs("float32" if device == "cpu" else "float16"),
            )
            load_target = name
            from utils.paths import is_local_path
            from utils.utils import cached_st_source, hf_cache_snapshot_dir

            # The repo AND the directory that supplied the weights, together: ST weights alone are satisfied by
            # the first finalized shard of a transfer still in flight.
            # Repo ids only: a local folder named all-MiniLM-L6-v2 would otherwise load the Hub's weights under
            # the local path's identity.
            st_source = None if is_local_path(name) else cached_st_source(name)
            if not local_only and st_source is not None:
                # Load the snapshot that was called cached: the repo id lets ST reach the Hub for a newer revision
                # during the first index, changing the vectors without changing their identity.
                load_target = str(st_source[1])
            if local_only:
                # ST-specific AND complete: a hybrid repo's cached GGUF, or a transfer that finalized only its first
                # shard, would otherwise retire the marker.
                if download_pending and st_source is None:
                    # Defensive: a loadable check and snapshot lookup share no lock, so eviction between them is
                    # still a pending model.
                    raise EmbeddingModelDownloadRequiredError(
                        f"Embedding model {name!r} is not downloaded yet. "
                        "Finish its Settings download before indexing documents."
                    )
                snapshot = st_source[1] if st_source else hf_cache_snapshot_dir(name)
                if snapshot is not None:
                    # A local path never touches the Hub, so this is offline-safe on ANY sentence-transformers
                    # version, even ones predating local_files_only.
                    load_target = str(snapshot)
                elif download_pending:
                    raise EmbeddingModelDownloadRequiredError(
                        f"Embedding model {name!r} is not downloaded yet. "
                        "Finish its Settings download before indexing documents."
                    )
                elif _st_accepts_local_files_only(SentenceTransformer):
                    st_kwargs["local_files_only"] = True
            # Scan after load_target is settled: on the repo id it checked the Hub's current commit while the
            # load opened an older cached one. evaluate_file_security recovers the repo and exact commit from
            # a snapshot path.
            _guard_model_security(load_target, offline)
            with _quiet_transformers_load() as report:
                # Re-emit in finally: a load that raises after transformers wrote its report is exactly when a
                # MISSING or MISMATCH line matters.
                try:
                    _model = SentenceTransformer(load_target, **st_kwargs)
                finally:
                    _emit_load_reports(report)
            _name = name
            if download_pending:
                # Retire only once the model is constructed: retiring earlier let a failed construction fall through
                # to llama-server with no marker, freeing the fallback to fetch the GGUF companion.
                try:
                    from utils.embedding_model_settings import clear_stored_download_pending
                    clear_stored_download_pending(name)
                except Exception:  # noqa: BLE001 - a settings write must not fail a load
                    pass
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
    with _compute_lock:
        # Admission and model lookup are one lease: lookup first would let unload clear the globals while
        # this call still held a strong reference.
        model = _get(model_name)
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
    with _compute_lock:
        return _get(model_name).get_sentence_embedding_dimension()


def _st_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Token counter using the model's tokenizer, under the compute lock (the same
    fast tokenizer backs encode and isn't thread-safe), with rayon enabled for the
    call. Mirrors ``_st_encode``: admission and model lookup are one lease, so the
    tokenizer is read per call inside the lock rather than captured here. Chunking
    holds this callable for a whole document, and a tokenizer captured up front
    outlives the unload that retired it -- counting on with weights nobody can
    reach, while the endpoint reports the model as gone."""

    def _count(t: str) -> int:
        with _compute_lock:
            tok = _get(model_name).tokenizer
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            try:
                return len(tok.encode(t, add_special_tokens = False))
            finally:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"

    return _count


def _release_st_model() -> bool:
    """Drop the module-level SentenceTransformer without racing an encode."""
    global _model, _name
    with _compute_lock:
        with _lock:
            released = _model is not None
            _model = None
            _name = None
    return released


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
        except (UnsafeEmbeddingModelError, EmbeddingModelDownloadRequiredError):
            raise
        except Exception as st_err:  # noqa: BLE001 - runtime ST/CUDA encode failure
            # ST loaded but this encode failed: swap the process to llama-server so later encodes stay in one
            # space, then retry.
            fallback = _switch_to_llama_fallback(st_err, model_name)
            if fallback is None:
                raise
            _served_by.backend = fallback
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
# Per model, keyed by model: one (key, model) pair let a second failing model erase the first one's
# pin and send a running job back to ST.
# Read WITHOUT _backend_lock: _get_backend holds it across a whole model load; dict.get is atomic
# and a pin landing mid-probe is answered on the next call.
_forced_backends: dict[str, str] = {}

_ST_ALIASES = frozenset({"sentence-transformers", "sentence_transformers", "st"})
_LLAMA_ALIASES = frozenset(
    {"llama-server", "llama_server", "llama", "llama.cpp", "llamacpp", "gguf"}
)
_AUTO_ALIASES = frozenset({"auto", ""})


def _resolve_auto() -> str:
    """Pick a backend for ``auto``: sentence-transformers when a CUDA/ROCm GPU is
    present (torch fp16 wins bulk indexing), else the torch-free GGUF llama-server
    -- or ST if its binary is missing. The GPU check goes through an smi tool
    (nvidia-smi, then amd-smi), so it costs no CUDA/HIP context unless neither
    is installed."""
    from core.inference.llama_cpp import LlamaCppBackend

    # Unfiltered probe: the winner runs under PyTorch, so the ROCm arch gate for the installed llama.cpp
    # prebuilt (#7624) must not apply.
    if LlamaCppBackend._get_gpu_free_memory():
        return "sentence-transformers"
    if LlamaCppBackend._find_llama_server_binary():
        return "llama-server"
    return "sentence-transformers"


def _model_is_local_gguf(model: str | None) -> bool:
    """Whether ``model`` names a local .gguf file, or a folder holding one.

    Gated on ``is_local_path`` first: a plain repo id costs no filesystem walk on
    the hot ``_get_backend`` path."""
    if not model:
        return False
    try:
        from utils.paths import is_local_path

        if not is_local_path(model):
            return False
        from core.rag.embed_llama_server import LlamaServerBackend

        return LlamaServerBackend._resolve_local_gguf(model) is not None
    except Exception:  # noqa: BLE001 - filesystem oddity is not a llama signal
        return False


def _model_names_gguf_repo(model: str | None) -> bool:
    """Whether ``model`` is a repo id that names GGUF weights.

    The `-GGUF` companion suffix is the convention the resolver derives and the
    picker's on-device dot follows, so such a repo publishes no safetensors for
    sentence-transformers to open. A pure name test: the remote counterpart of
    ``_model_is_local_gguf``, which cannot see a repo that is not on disk yet."""
    if not model:
        return False
    try:
        from utils.paths import is_local_path

        # A directory may be named anything, so only the filesystem can say that ~/models/my-gguf holding
        # safetensors is a sentence-transformers model.
        if is_local_path(model):
            return False
    except Exception:  # noqa: BLE001 - unparseable path is not a repo id either
        return False
    # config's predicate, not a second opinion: gguf_repo_candidates already counts "gguf" as a whole
    # name segment, so owner/GGUF-model is a GGUF repo there and must not be one here.
    return config._names_gguf(model.strip().rstrip("/").rsplit("/", 1)[-1])


def _resolve_auto_for_model(model_name: str | None = None) -> str:
    """``auto``, but honouring the backend recorded for the saved model.

    An embedder with no GGUF still runs on sentence-transformers, so the picker
    records that choice; the hardware default would send it to llama-server,
    which has nothing to open."""
    model = model_name or config.effective_embedding_model()
    # Ahead of the stored record, since the filesystem was asked rather than guessed at; only auto
    # consults this, so an explicit RAG_EMBED_BACKEND still wins.
    if _model_is_local_gguf(model):
        return "llama-server"
    try:
        from utils.embedding_model_settings import get_stored_backend
        stored = get_stored_backend(model)
    except Exception:  # noqa: BLE001 - store unavailable: fall back to hardware
        stored = None
    if stored:
        key = stored.strip().lower()
        if key in _ST_ALIASES or key in _LLAMA_ALIASES:
            return key
    # Below the stored record, since a name is only a guess: a repo with a torn GGUF family and usable
    # safetensors has a validated ST plan.
    if _model_names_gguf_repo(model):
        return "llama-server"
    return _resolve_auto()


def sentence_transformers_runtime_available() -> bool:
    """Whether the ST backend can reach the model-loading step in this process.

    This deliberately mirrors the environment-dependent prefix of ``_get`` but
    does not construct a model (which could download the snapshot the picker is
    still planning). It catches missing/broken torch or sentence-transformers
    installs and the fatal device mismatch that ``_build_st_backend_or_fallback``
    would otherwise discover only after an ST-only plan was persisted.
    """
    try:
        _load_device()
        # Not under _lock: _get holds it across an entire SentenceTransformer construction, download
        # included, so sharing it blocked Settings for the length of a slow first load.
        _install_torchao_stub_once()
        from sentence_transformers import SentenceTransformer

        return callable(SentenceTransformer)
    except Exception as exc:  # noqa: BLE001 - any failed runtime import selects the fallback
        logger.debug("sentence-transformers runtime preflight failed: %s", exc)
        return False


def _llama_server_runtime_available() -> bool:
    """Whether the fallback that ST construction would use can be built."""
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return bool(LlamaCppBackend._find_llama_server_binary())
    except Exception:  # noqa: BLE001 - an unavailable fallback cannot be planned
        return False


def resolved_backend_for_model(model_name: str) -> str:
    """Backend a fresh operation for ``model_name`` would actually select."""
    raw = _raw_backend()
    forced = _forced_backends.get(model_name)
    key = forced or (_resolve_auto_for_model(model_name) if raw in _AUTO_ALIASES else raw)
    if key in _ST_ALIASES and not sentence_transformers_runtime_available():
        # Without a real llama binary ST is the only possible plan, and its eventual error is more useful
        # than a fabricated GGUF destination.
        if _llama_server_runtime_available():
            key = "llama-server"
    if key in _LLAMA_ALIASES:
        return "llama-server"
    if key in _ST_ALIASES:
        return "sentence-transformers"
    raise ValueError(
        f"Unknown RAG_EMBED_BACKEND={config.EMBED_BACKEND!r}; expected "
        "'auto', 'sentence-transformers' or 'llama-server'"
    )


def _try_make_llama_backend():
    """A llama-server GGUF embedding backend if its binary is present, else None.
    Construction is lazy -- no server starts until warm."""
    from core.inference.llama_cpp import LlamaCppBackend

    if not LlamaCppBackend._find_llama_server_binary():
        return None
    from .embed_llama_server import LlamaServerBackend

    return LlamaServerBackend()


def _build_st_backend_or_fallback(model_name: str | None = None):
    """Build the ST backend, probing it by loading the model now. If the probe
    raises (no torch, CUDA mismatch, bad wheel) and the GGUF llama-server embedder
    is available, fall back to it. The probe runs before any vector is produced, so
    this never mixes spaces. Re-raises if no embedder can start.

    ``model_name`` is the model the caller pinned. Warming ``None`` reads the live
    setting, so a job pinned to A probed B once Settings moved, failing the valid
    A job before its first encode."""
    backend = _SentenceTransformersBackend()
    try:
        backend.warm(model_name = model_name)
        return backend
    except (UnsafeEmbeddingModelError, EmbeddingModelDownloadRequiredError):
        raise
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


def _switch_to_llama_fallback(err, model_name: str | None = None):
    """An ST encode failed at runtime even though the model had loaded. Swap the
    process embedder to llama-server so every later encode stays in one space, and
    return it (None if no binary). Vectors written before the swap were ST, so any
    KB already embedded with ST should be reindexed."""
    global _backend, _backend_key
    failed_model = model_name or config.effective_embedding_model()
    old = None
    with _backend_lock:
        if not isinstance(_backend, _SentenceTransformersBackend):
            return _backend
        fallback = _try_make_llama_backend()
        if fallback is None:
            return None
        logger.warning(
            "sentence-transformers encode failed (%s); switching to the llama-server "
            "embedder for the rest of this process. Reindex any knowledge base that "
            "was already embedded with sentence-transformers.",
            err,
        )
        old, _backend = _backend, fallback
        _forced_backends[failed_model] = "llama-server"
        _backend_key = _backend_cache_key(_raw_backend(), "llama-server")
    # The failed ST wrapper is no longer published, but its module-level model would survive even a
    # later unload of the llama replacement.
    _dispose_replaced_backend(old, fallback)
    return fallback


def _raw_backend() -> str:
    return (config.EMBED_BACKEND or "auto").strip().lower()


def sentence_transformers_fallback_allowed(model_name: str | None = None) -> bool:
    """Whether a resolved ST plan can actually be selected for a new model.

    An explicit llama configuration ignores the per-model stored backend, and
    a runtime ST failure deliberately pins llama until unload. In either state,
    offering safetensors would save a model the first index cannot load.
    """
    raw = _raw_backend()
    model = model_name or config.effective_embedding_model()
    if _forced_backends.get(model) in _LLAMA_ALIASES:
        return False
    return raw in _AUTO_ALIASES or raw in _ST_ALIASES


def _current_backend_key() -> str:
    """The cache key the backend in use should carry right now. Tests that install a
    stub backend set ``_backend_key`` from this so it is not rebuilt under them."""
    raw = _raw_backend()
    forced = _forced_backends.get(config.effective_embedding_model())
    if forced:
        return _backend_cache_key(raw, forced)
    key = _resolve_auto_for_model() if raw in _AUTO_ALIASES else raw
    return _backend_cache_key(raw, key)


def _backend_cache_key(raw: str, key: str) -> str:
    """Cache key for a built backend. It carries the RESOLVED choice, not just the
    raw config, so saving a model that needs the other backend rebuilds instead of
    serving the one already built for the previous model."""
    return f"{raw}\x00{key}"


def _dispose_replaced_backend(old, new = None) -> None:
    """Release resources owned by a backend that is no longer published."""
    if old is None or old is new:
        return
    if isinstance(old, _SentenceTransformersBackend):
        # Two ST wrappers share the module-level model and the replacement is already warmed, so clearing it
        # here would discard the model just selected.
        if not isinstance(new, _SentenceTransformersBackend):
            _release_st_model()
        return
    shutdown = getattr(old, "_shutdown", None)
    if callable(shutdown):
        try:
            shutdown()
        except Exception:  # noqa: BLE001 - replacement is already selected
            logger.warning("replaced embedding backend shutdown failed", exc_info = True)


def _get_backend(model_name: str | None = None):
    """The process-wide embedding backend for ``config.EMBED_BACKEND``, built once.
    Cached by the resolved choice, so ``auto`` detection runs only on a miss and a
    config or saved-model change rebuilds it.

    ``model_name`` is the model the caller is embedding for, defaulting to the live
    setting. A job pins its model once and passes it down, and per-model stored
    backends mean two models can resolve differently: reading the setting here
    instead would let a Settings change mid-job build the NEW model's backend while
    ``encode_with_identity`` goes on labelling the vectors with the pinned one.
    """
    global _backend, _backend_key
    raw = _raw_backend()
    old = None
    new = None
    with _backend_lock:
        model = model_name or config.effective_embedding_model()
        forced = _forced_backends.get(model)
        key = forced or (_resolve_auto_for_model(model) if raw in _AUTO_ALIASES else raw)
        if _backend is not None and _backend_key == _backend_cache_key(raw, key):
            return _backend
        old = _backend
        if key in _ST_ALIASES:
            new = _build_st_backend_or_fallback(model)
        elif key in _LLAMA_ALIASES:
            # Imported lazily so the ST path never imports llama plumbing.
            from .embed_llama_server import LlamaServerBackend
            new = LlamaServerBackend()
        else:
            raise ValueError(
                f"Unknown RAG_EMBED_BACKEND={config.EMBED_BACKEND!r}; expected "
                "'auto', 'sentence-transformers' or 'llama-server'"
            )
        _backend = new
        if key in _ST_ALIASES and _is_llama_backend(new):
            # Pin the backend the warm probe actually fell back to, but let a different model retry ST.
            key = "llama-server"
            _forced_backends[model] = key
        _backend_key = _backend_cache_key(raw, key)
    # A llama shutdown can wait for an in-flight encode, so keep that wait out of the global publication lock.
    _dispose_replaced_backend(old, new)
    return new


def _reset_backend() -> None:
    """Drop the cached backend (test teardown / re-init)."""
    global _backend, _backend_key
    with _backend_lock:
        _forced_backends.clear()
        _backend = None
        _backend_key = None


def backend_is_loaded(model_name: str | None = None) -> bool:
    """Whether ``model_name`` is resident, or any embedder when omitted.

    Deliberately lock-free: ``_backend_lock`` and ``_lock`` are both held across a
    whole model load, so taking either here made GET, PUT, reset and unload wait it
    out. Both reads are single attribute loads, and the pre- or post-load value is
    equally true for "is something resident right now".
    """
    backend = _backend
    if backend is None:
        # No published backend does not mean nothing is loaded: answering False stranded module-level
        # weights, since release_backend returns on the same test.
        if model_name is None:
            return _model is not None
        return _model is not None and _name == model_name
    if model_name is None:
        # A llama backend whose process is gone is not resident, whichever model was asked about.
        if _is_llama_backend(backend):
            try:
                return bool(backend._process_alive())
            except Exception:  # noqa: BLE001 - a status probe must never block settings
                return False
        return True
    if isinstance(backend, _SentenceTransformersBackend):
        return _model is not None and _name == model_name
    if _is_llama_backend(backend):
        try:
            # The object keeps _model_repo after the subprocess exits, so a repo match alone would call a dead
            # server resident.
            if not backend._process_alive():
                return False
            return backend._model_repo == config.effective_gguf_repo_for_embedding_model(model_name)
        except Exception:  # noqa: BLE001 - a status probe must never block settings
            return False
    return False


def release_backend() -> bool:
    """Drop the embedder and stop its llama-server, if one is running. Returns
    whether anything was released.

    Safe mid-ingestion: the next embed rebuilds, and the llama backend's own POST
    retry already covers a server that went away under it."""
    global _backend, _backend_key
    with _backend_lock:
        # Unload is an explicit fresh start, so a past runtime fallback stops pinning the choice and the saved
        # model picks its backend again.
        _forced_backends.clear()
        backend, _backend, _backend_key = _backend, None, None
    if backend is None:
        # Nothing published, but the module-level model can still be there (see backend_is_loaded);
        # freeing it here is what keeps that leak from being permanent.
        return _release_st_model()
    _dispose_replaced_backend(backend)
    return True


def active_backend_is_llama(model_name: str | None = None) -> bool:
    """True when this process actually embeds via the llama-server (GGUF) backend.

    Reflects the ACTUAL built backend once one exists: an ``auto`` install that
    resolves to sentence-transformers but then falls back to llama-server at
    runtime (``_build_st_backend_or_fallback`` on a torch/CUDA load failure, or
    ``_switch_to_llama_fallback`` on an encode failure) loads only inert GGUF, so
    callers gating on the ST pickle must see llama here. Before any backend is
    built, defers to the resolver (``auto`` -> ``_resolve_auto_for_model()``, else
    the raw key) exactly as a fresh process would.

    ``model_name`` names the model to resolve for, defaulting to the live setting.
    A caller embedding under a model pinned for the length of a job passes it, so
    the answer cannot drift when the setting changes underneath that job. Never
    raises: a backend probe must not block saving a model."""
    try:
        with _backend_lock:
            backend = _backend
        if backend is not None:
            # Report what the backend ACTUALLY is: a concrete sentence-transformers backend must return False
            # even if the resolver would now pick llama, so its pickle stays gated.
            try:
                from .embed_llama_server import LlamaServerBackend
            except Exception:  # noqa: BLE001 - llama plumbing import must never block
                return False
            return isinstance(backend, LlamaServerBackend)
        raw = (config.EMBED_BACKEND or "auto").strip().lower()
        key = _resolve_auto_for_model(model_name) if raw in _AUTO_ALIASES else raw
        return key in _LLAMA_ALIASES
    except Exception:  # noqa: BLE001 - a backend probe must never block saving
        return False


def _identity(is_llama: bool, name: str) -> str:
    if is_llama:
        return config.embedding_identity(
            "llama-server",
            name,
            gguf_repo = config.effective_gguf_repo_for_embedding_model(name),
        )
    return config.embedding_identity("sentence-transformers", name)


def _identity_backend_is_llama(name: str) -> bool:
    """Backend the next encode for ``name`` will use.

    The security-facing active-backend probe deliberately reports a resident
    backend even when Settings has just selected another one. Identity prediction
    is different: ``_get_backend`` will replace a resident backend whose cache key
    no longer matches the stored per-model resolution, so admission/deduplication
    must predict that replacement before the first encode happens.

    ``name`` is threaded into the probe rather than left to default: it may be a
    model pinned for the length of one job (a linked-folder reconcile resolves the
    model once and embeds every file under it), and re-reading the live setting per
    file would let a Settings change mid-job tag two files in one folder with two
    different identities.
    """
    try:
        raw = _raw_backend()
        with _backend_lock:
            backend = _backend
            cached_key = _backend_key
        forced = _forced_backends.get(name)
        resolved = forced or (_resolve_auto_for_model(name) if raw in _AUTO_ALIASES else raw)
        expected_key = _backend_cache_key(raw, resolved)
        if backend is None or cached_key == expected_key:
            return active_backend_is_llama(name)
        return resolved in _LLAMA_ALIASES
    except Exception:  # noqa: BLE001 - identity prediction must not block ingestion
        return active_backend_is_llama(name)


def embedding_identity(model_name: str | None = None) -> str:
    """Identity of the vectors this process produces right now.

    Recorded on every document, because the model name alone does not name the
    embedding space: llama-server ignores the name and embeds through the GGUF
    companion with its own pooling, and this process can switch to it at runtime. Two
    spaces under one label is an index that answers with the wrong documents and says
    nothing about it."""
    name = model_name or config.effective_embedding_model()
    return _identity(_identity_backend_is_llama(name), name)


def _is_llama_backend(backend) -> bool:
    """Whether a concrete backend object embeds through llama-server."""
    try:
        from .embed_llama_server import LlamaServerBackend
    except Exception:  # noqa: BLE001 - llama plumbing import must never block
        return False
    return isinstance(backend, LlamaServerBackend)


def encode_with_identity(
    texts: list[str],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """``(vectors, identity)``, the identity taken from the encode that produced them.

    Not from the process embedder read afterwards: a concurrent ST encode failure
    swaps that between the two, so the vectors would be labelled with a space they
    were never in, and a query then searches (or a document is stored against) the
    wrong half of the index."""
    _served_by.backend = None
    vectors = encode(texts, model_name = model_name, normalize = normalize)
    served = getattr(_served_by, "backend", None)
    name = model_name or config.effective_embedding_model()
    if served is None:
        return vectors, embedding_identity(name)
    return vectors, _identity(_is_llama_backend(served), name)


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get_backend(model_name).warm(model_name = model_name)


def encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """Embed texts into an (N, dim) float32 numpy array.

    An explicit unload can retire the llama backend between resolving it and using
    it, so that one lifecycle failure reacquires the newly published backend, the
    same way ``token_counter`` does for a counter held across chunks. Without it
    ``release_backend`` fails the in-flight document rather than rebuilding for it.
    """
    backend = _get_backend(model_name)
    _served_by.backend = backend
    try:
        return backend.encode(texts, model_name = model_name, normalize = normalize)
    except RuntimeError:
        if not (_is_llama_backend(backend) and getattr(backend, "_closed", False)):
            raise
    replacement = _get_backend(model_name)
    if replacement is backend:
        raise RuntimeError("llama-server embedding backend was unloaded")
    _served_by.backend = replacement
    return replacement.encode(texts, model_name = model_name, normalize = normalize)


def dim(model_name: str | None = None) -> int:
    """Embedding dimension for the (loaded) model."""
    return _get_backend(model_name).dim(model_name = model_name)


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Callable counting tokens with the embedder's own tokenizer.

    Chunking keeps this callable for the whole document. An explicit unload can
    retire its llama backend between two calls, so lazily reacquire the newly
    published backend only for that precise lifecycle failure. Other tokenizer
    errors still propagate unchanged.
    """
    backend = _get_backend(model_name)
    state = (backend, backend.token_counter(model_name = model_name))
    counter_lock = threading.Lock()

    def _count(text: str) -> int:
        nonlocal state
        served_backend, served_count = state
        try:
            return served_count(text)
        except RuntimeError:
            if not (
                _is_llama_backend(served_backend) and getattr(served_backend, "_closed", False)
            ):
                raise
        with counter_lock:
            # Another counting thread may already have replaced the retired counter.
            if state[0] is served_backend:
                replacement = _get_backend(model_name)
                if replacement is served_backend:
                    raise RuntimeError("llama-server embedding backend was unloaded")
                state = (
                    replacement,
                    replacement.token_counter(model_name = model_name),
                )
            retry = state[1]
        return retry(text)

    return _count
