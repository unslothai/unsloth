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
import re
import sys
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


def _rocm_gpu_is_fatal() -> bool:
    """True when this host's ROCm stack faults on its first real GPU allocation.

    ROCm is ``DeviceType.CUDA`` internally, so nothing here distinguishes a healthy CUDA
    host from a ROCm wheel built for the wrong gfx arch. That cannot be decided in this
    process: the mismatch SIGSEGVs inside the HIP runtime and takes uvicorn with it (#8474,
    host in #7331). So a child allocates first and its death is the answer. Asked once per
    process, and never off ROCm, so every other host spawns nothing.

    Only ``_safe_torch_device`` calls this, after ``_device()`` has settled detection, so
    the unknown case below is not a ROCm host going unprobed.
    """
    if _host_is_rocm() is not True:
        return False
    from utils.device_allocation_probe import probe_torch_device_allocation

    return not probe_torch_device_allocation("cuda:0").ok


def _host_is_rocm() -> bool | None:
    """True/False once hardware detection has settled, None while it has not.

    Tri-state on purpose. Detection imports torch and this is reached from
    ``active_backend_is_llama()`` on the settings request path, so forcing it is not an
    option; but collapsing "not yet known" into False would let that path fall through to
    the in-process AMD GPU query. Callers decide what "unknown" means for them.
    """
    try:
        from utils.hardware import hardware as hardware_mod
        if not hardware_mod.DETECTION_COMPLETE.is_set():
            return None
        return bool(hardware_mod.IS_ROCM)
    except Exception:  # noqa: BLE001 - if we cannot even tell, leave behaviour untouched
        return False


def _rocm_is_possible() -> bool:
    """Could this host be ROCm at all, answered without torch and without detection.

    Keeps the "detection has not settled" caution narrow. Treating every unsettled host as
    possibly-ROCm is safe for the crash but too broad: it hands a CPU or macOS host the
    provisional answer, which ``routes/settings.py`` reads as "not llama" and drops its
    GGUF handling, so a valid local .gguf briefly 409s on a host with no AMD GPU.

    ``torch.version.hip`` settles it exactly: a build attribute, read without initialising
    the driver (``hardware.py::apply_gpu_ids`` relies on that), and the ROCm-ness the
    dangerous query turns on. Consulted only when torch is already imported, then read off
    disk, and only then does each platform fall back to a file test or to no claim:
      * macOS never has ROCm.
      * Linux: ROCm's own kernel driver publishes this topology directory, and
        ``utils/hardware/hardware.py`` already reads it (``_rocm_kfd_gpu_pci_ids``).
      * Windows: nothing equivalent, and an installed HIP SDK is NOT evidence -- ``main.py``
        refuses to let HIP_PATH/ROCM_PATH alone pick a backend for the same reason.
    """
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            if getattr(getattr(torch, "version", None), "hip", None) is not None:
                return True
            # AMD SDK wheels leave torch.version.hip unset and only say rocm in the version
            # string; hardware.py's own ROCm test carries the same fallback. And a torch
            # that is in sys.modules but still executing has neither yet, which is absence
            # of evidence, not evidence of absence, so that stays possible.
            version_string = getattr(torch, "__version__", None)
            if not isinstance(version_string, str) or not version_string:
                return True
            return "rocm" in version_string.lower()
        installed = _installed_torch_is_rocm()
        if installed is not None:
            return installed
        if sys.platform == "darwin":
            return False
        if sys.platform == "win32":
            # Nothing torch-free left to read, and Windows has no KFD node. An installed
            # HIP SDK is not evidence (main.py makes the same point), but neither is its
            # absence, so keep the caution rather than the convenience.
            return True
        return os.path.isdir("/sys/class/kfd/kfd/topology/nodes")
    except Exception:  # noqa: BLE001 - unsure means possible; the caution is the safe side
        return True


@lru_cache(maxsize = 1)
def _installed_torch_is_rocm() -> bool | None:
    """Is the INSTALLED torch a ROCm build, answered without importing it.

    ``torch/version.py`` holds generated literals and ``find_spec`` locates the package
    without executing it, so this reads the same two facts detection reads, long before
    torch is imported. That matters on Windows, which has no KFD node.

    False also covers torch being absent, which is a definite answer: no torch, no ROCm
    torch path. None is reserved for an installation that is there but unreadable. Cached,
    being filesystem I/O on a request path.
    """
    try:
        import importlib.util

        spec = importlib.util.find_spec("torch")
        if spec is None or not spec.origin:
            # Absent, not unreadable. There is no ROCm torch path to protect on a
            # --no-torch install, and saying "unknown" here would make Windows cautious
            # forever and cost that install the llama/GGUF answer it actually resolves to.
            return False
        version_py = os.path.join(os.path.dirname(spec.origin), "version.py")
        with open(version_py, encoding = "utf-8") as handle:
            source = handle.read()
    except Exception:  # noqa: BLE001 - no answer, and the caller decides what that means
        return None

    # hip = '6.3.42134' on a ROCm build, hip: Optional[str] = None otherwise.
    if re.search(r"^hip\b[^=\n]*=\s*['\"]", source, re.MULTILINE):
        return True
    match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)", source, re.MULTILINE)
    if match:
        # AMD SDK wheels leave hip unset and only say rocm here; same fallback as
        # utils/hardware/hardware.py.
        return "rocm" in match.group(1).lower()
    return None


def _safe_torch_device() -> str:
    """``_device()``, except that a ROCm GPU which cannot allocate resolves to CPU.

    Degrading to CPU rather than to the llama-server GGUF embedder is deliberate: same
    backend, same model, so the vectors are unchanged and no knowledge base needs
    reindexing. bge-small on CPU is a slowdown, not a loss of function.
    """
    device = _device()
    if device != "cuda":
        return device
    if _rocm_gpu_is_fatal():
        logger.warning(
            "this host's ROCm GPU crashed an isolated allocation probe, so it cannot be "
            "used in this process without killing the backend; loading the embedding "
            "model on CPU instead. Embeddings are unchanged, so no knowledge base needs "
            "reindexing. Usually a torch wheel built for a different gfx arch."
        )
        return "cpu"
    return device


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
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer
            from utils.hf_cache_settings import active_hf_hub_cache

            device = _safe_torch_device()
            # A GPU we were pushed off is not a host that never had one: fp16 wins on the
            # GPU and is slow and patchily supported on CPU. Only the degraded case
            # switches, so a genuine CPU/Apple host loads exactly as before.
            degraded_to_cpu = device == "cpu" and _device() == "cuda"
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
    -- or ST if its binary is missing.

    The GPU check is torch-free on NVIDIA (nvidia-smi) but NOT on AMD, where
    ``_get_gpu_free_memory`` falls back to torch ``mem_get_info`` in this process. So
    anything that might be ROCm answers without asking it, which is what stops this
    function steering an affected host into the crash it exists to route around. The answer
    is the same either way: a working ROCm GPU takes sentence-transformers, a condemned one
    is placed on CPU by ``_safe_torch_device``. No allocation probe runs here; the device
    decision belongs to the load.

    Settles detection if it has not settled, so the answer is final and safe for
    ``_get_backend`` to cache. ``active_backend_is_llama`` must not block on detection and
    so decides the ROCm case itself rather than coming here.
    """
    is_rocm = _host_is_rocm()
    if is_rocm is None:
        # get_device() imports torch under its own lock: affordable for the builder,
        # not for the request path, which does not come here.
        try:
            get_device()
            is_rocm = _host_is_rocm()
        except Exception:  # noqa: BLE001 - detection failing is not this function's problem
            is_rocm = None

    if is_rocm is True or (is_rocm is None and _rocm_is_possible()):
        # None here is detection that would not settle. Hold back the AMD query only for a
        # host that could be ROCm, so a CPU or macOS host keeps its real answer, and with
        # it its GGUF classification.
        return "sentence-transformers"

    return _resolve_by_gpu_and_binary()


def _resolve_by_gpu_and_binary() -> str:
    """The non-ROCm half of ``auto``: GPU present -> ST, else the GGUF llama-server, or ST
    if its binary is missing. Exactly what ``_resolve_auto`` was before this file learned
    about ROCm. Needs no hardware detection, so the request path calls it directly instead
    of going through the resolver and parking on the detection lock.
    """
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
        if raw in _AUTO_ALIASES:
            # Runs inside PUT /embedding-model, so it must not sit behind hardware
            # detection. A ROCm host, or an unsettled one that could be ROCm, answers
            # without asking: auto resolves to sentence-transformers on every ROCm host, so
            # False is the real answer, and it leaves the ST pickle gate ENGAGED, the
            # conservative direction for a security check. Everything else keeps the
            # llama/GGUF classification settings.py needs to accept a local .gguf.
            is_rocm = _host_is_rocm()
            if is_rocm is True or (is_rocm is None and _rocm_is_possible()):
                return False
            # ensure_hardware_detected() holds _DETECT_LOCK across a cold torch import, so
            # _resolve_auto here would park the request on the wait this branch avoids.
            key = _resolve_auto() if is_rocm is False else _resolve_by_gpu_and_binary()
        else:
            key = raw
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
