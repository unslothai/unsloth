# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Dense embedder facade over config.EMBED_BACKEND (sentence-transformers or llama-server); off GPU
unless asked; backends differ, so switching needs reindex; ST failure falls back to llama-server;
_load_device probes in a child and falls back to CPU without changing the embedding space."""

from __future__ import annotations

from core.training.account_jobs import account_hf_token, account_path, managed_account
import logging
import os
import re
import threading
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Callable

from utils.cpu_threads import fit_threads_to_affinity, runtime_capped_openblas_loaded
from utils.hardware.hardware import DeviceType, get_device
from utils.transformers_dtype import dtype_kwargs
from utils.utils import hf_env_offline

from . import config

logger = logging.getLogger(__name__)

# "false" silences the fast tokenizer's fork warning; encode() flips it only during a batch tokenize.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_lock = threading.Lock()
# HF fast tokenizer is not thread-safe; separate from _lock so encode never blocks reload.
_compute_lock = threading.Lock()
_model = None
_name: str | None = None
# Table lookup: a GPU request with no GPU maps to cpu, not to a device string torch cannot open.
_served_by = threading.local()


# Unsloth device -> torch device string. Apple has no torch device -> CPU.
_TORCH_DEVICE = {DeviceType.CUDA: "cuda", DeviceType.XPU: "xpu"}


def _device() -> str:
    """Torch device for the in-process embedder. CPU unless asked otherwise.

    Defaulting a GPU machine to CPU is deliberate: this embedder runs inside the backend process,
    and the first CUDA allocation there creates a primary context never returned while the process
    lives (712 MiB on a B200 against 74 MiB for bge-small's own weights), so ingesting one document
    cost most of a gigabyte of VRAM for the session and no amount of unloading gets it back.

    The trade is small at these sizes: one 128-token chunk takes 18.7ms on CPU against 5.2ms on
    CUDA. Bulk indexing is where it shows (batch 64: 445 chunks/s on CPU against 3174/s), and
    Settings (GPU) or ``RAG_EMBED_DEVICE=gpu`` opts back in for a large corpus.

    This reads the same setting as the llama-server backend but resolves ``auto`` differently, which
    is intended: that backend offloads inside its own subprocess, where the context dies with the
    child.
    """
    if config.embed_device_preference() != "gpu":
        return "cpu"
    # Asking for a GPU without one falls back to CPU, never to a device torch cannot open.
    return _TORCH_DEVICE.get(get_device(), "cpu")


class TorchDeviceUnusableError(RuntimeError):
    """Raised when torch cannot allocate safely on the accelerator or CPU."""


def _load_device() -> str:
    """Falls back to CPU to keep the embedding space; re-raises only if CPU also crashes."""
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


# Matched as a substring of the lowercased model name. EmbeddingGemma's activations overflow float16 and every
# vector comes back NaN; its model card says to use float32 or bfloat16.
_FLOAT16_UNSAFE_MODELS = ("embeddinggemma",)


def _load_dtype(device: str, name: str) -> str:
    """float32 on CPU, where fp16 BERT raises "not implemented for Half" and encode() answers by swapping the
    whole process to llama-server. float16 on an accelerator, except for models that cannot run in it: those
    get bf16 where the device has it natively, else float32."""
    if device == "cpu":
        return "float32"
    if not any(m in name.lower() for m in _FLOAT16_UNSAFE_MODELS):
        return "float16"
    from core.training.diffusion_train_common import (
        native_bf16_supported,
        native_bf16_supported_xpu,
    )

    native = native_bf16_supported_xpu() if device == "xpu" else native_bf16_supported()
    return "bfloat16" if native else "float32"


_torchao_stub_done = False
# Own lock: _lock is held across a whole model load, so a probe would wait out a download.
_stub_lock = threading.Lock()


def _install_torchao_stub_once() -> None:
    """Stub torchao before sentence-transformers loads: on Windows ROCm its absent c10d backend aborts."""
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
    if managed_account():
        return False
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None


def _repo_json(name: str, filename: str, token: str | bool | None) -> Any:
    """``filename`` parsed from the local folder or Hub repo ``name``, None when the file is absent.
    Raises whatever reading or parsing raises."""
    import json

    from utils.paths import is_local_path

    if is_local_path(name):
        from pathlib import Path
        from utils.paths import normalize_path

        path = Path(normalize_path(name)).expanduser() / filename
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding = "utf-8-sig"))
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from utils.hf_cache_settings import active_hf_hub_cache

    try:
        local = hf_hub_download(
            name,
            filename,
            token = account_hf_token(token),
            cache_dir = active_hf_hub_cache(),
        )
    except EntryNotFoundError:
        return None
    with open(local, encoding = "utf-8-sig") as f:
        return json.load(f)


def _st_module_subdirs(name: str, token: str | None) -> tuple[str, ...]:
    """Module subdirs from modules.json, the load roots for the security scan; () on any failure."""
    try:
        data = _repo_json(name, "modules.json", token)
        subdirs = []
        for module in data or ():
            sub = str((module or {}).get("path", "")).strip().strip("/")
            if sub:
                subdirs.append(sub)
        return tuple(dict.fromkeys(subdirs))
    except Exception:
        return ()


def _guard_model_security(
    name: str,
    local_only: bool = False,
    display: str | None = None,
) -> None:
    """Refuses repos HF flags unsafe: a poisoned pickle loads regardless of trust_remote_code."""
    try:
        from utils.security import evaluate_file_security, security_load_subdirs

        token = _ambient_hf_token()
        if local_only:
            load_subdirs = ()
        else:
            # Union of load roots, so a flagged pickle under a module dir still blocks.
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
            f"Embedding model {(display or name)!r} {reason}; refusing to load. "
            "Set a different RAG embedding model."
        )


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


class _CaptureLoadReport(logging.Filter):
    """Swallow transformers' multi-line "<Model> LOAD REPORT" table, keeping the text.

    transformers >= 5 emits the report through ``logger.warning`` with embedded ANSI colour codes,
    so it lands in the server log as ~7 unstructured lines that break every JSON consumer. It fires
    on every boot for the RAG embedder because bge-small-en-v1.5 ships a legacy
    ``embeddings.position_ids`` key the current BertModel does not expect, which is benign and
    identical every time.

    Nothing is lost: the caller re-emits the report (see ``_quiet_transformers_load``) at debug when
    it only reports that known legacy key, and at warning when it mentions anything that could
    change the model's behaviour.
    """

    _SERIOUS = ("MISSING", "MISMATCH", "CONVERSION")
    # Only this legacy BERT buffer is benign; other discarded weights can change retrieval.
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
            # Skip transformers' trailing Notes: section, whose lines look like key rows.
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
    # Adapter models log via the PEFT integration logger, not a child of the loggers above.
    "transformers.integrations.peft",
)


@contextmanager
def _quiet_transformers_load():
    """Keeps transformers load progress and ANSI off stdout; restores the progress-bar setting as found."""
    capture = _CaptureLoadReport()
    attached = []
    for name in _LOAD_REPORT_LOGGERS:
        log = logging.getLogger(name)
        log.addFilter(capture)
        attached.append(log)

    # Progress-bar probe name differs across transformers versions; restore Hub state separately.
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
    """Flattens the report to one plain line, since the stdlib logger would echo ANSI and newlines."""
    plain = _ANSI_RE.sub("", text)
    return " | ".join(part.strip() for part in plain.splitlines() if part.strip())


def _emit_load_reports(report) -> None:
    """Re-emits swallowed load reports: debug for the legacy-key notice, warning if embeddings may
    change; then drains the list so a retry does not report the same lines twice."""
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


_ST_PACKAGE_PREFIX = "sentence_transformers."
_ST_GATE_MARKER = "_unsloth_custom_module_gate"


def _refuse_custom_module(
    class_ref,
    model_name_or_path,
    trust_remote_code,
    prefix = _ST_PACKAGE_PREFIX,
) -> None:
    if (
        isinstance(class_ref, str)
        and not class_ref.startswith(prefix)
        and model_name_or_path is not None
        and not trust_remote_code
    ):
        raise ValueError(
            f"The model {model_name_or_path} references the module class {class_ref!r}, which is not "
            "part of Sentence Transformers. Importing it executes third-party code, so Unsloth refuses "
            "to load it as an embedding model."
        )


def _gate_st_custom_modules() -> None:
    """Backport sentence-transformers 6.0's trust gate (CVE-2026-68770): before 6.0 a cached
    model's modules.json could import repo-hosted classes without trust_remote_code."""
    try:
        import inspect
        import sentence_transformers as st
        from packaging.version import Version

        if Version(st.__version__).major >= 6:
            return
        from sentence_transformers import SentenceTransformer
    except Exception:
        return
    # 5.0-5.4 resolve on the model class; 5.5+ also via util.misc.import_module_class.
    owner = next(
        (c for c in SentenceTransformer.__mro__ if "_load_module_class_from_ref" in vars(c)),
        None,
    )
    if owner is not None:
        original = vars(owner)["_load_module_class_from_ref"]
        if not getattr(original, _ST_GATE_MARKER, False):
            signature = inspect.signature(original)

            def _load_module_class_from_ref(self, *args, **kwargs):
                bound = signature.bind_partial(self, *args, **kwargs).arguments
                _refuse_custom_module(
                    bound.get("class_ref"),
                    bound.get("model_name_or_path"),
                    bound.get("trust_remote_code"),
                )
                return original(self, *args, **kwargs)

            setattr(_load_module_class_from_ref, _ST_GATE_MARKER, True)
            _load_module_class_from_ref.__wrapped__ = original
            setattr(owner, "_load_module_class_from_ref", _load_module_class_from_ref)
    # Below 6.0 these module loaders import a dotted path out of their own config via
    # import_from_string, past both resolvers: Router its types, Dense its
    # activation_function (and calls it), WordEmbeddings its tokenizer_class.
    for module_path, prefix, where in (
        ("sentence_transformers.models.Router", _ST_PACKAGE_PREFIX, "behind this Router"),
        ("sentence_transformers.models.Dense", "torch.", "in a Dense module"),
        (
            "sentence_transformers.models.WordEmbeddings",
            _ST_PACKAGE_PREFIX,
            "in a WordEmbeddings module",
        ),
    ):
        try:
            import importlib

            module = importlib.import_module(module_path)
            original_from_string = getattr(module, "import_from_string", None)
            if original_from_string is None or getattr(
                original_from_string, _ST_GATE_MARKER, False
            ):
                continue

            def import_from_string(
                dotted_path,
                *args,
                _original = original_from_string,
                _prefix = prefix,
                _where = where,
                **kwargs,
            ):
                _refuse_custom_module(dotted_path, _where, False, prefix = _prefix)
                return _original(dotted_path, *args, **kwargs)

            setattr(import_from_string, _ST_GATE_MARKER, True)
            import_from_string.__wrapped__ = original_from_string
            module.import_from_string = import_from_string
        except Exception:
            pass
    try:
        import sys
        from sentence_transformers.util import misc
    except Exception:
        return
    original_import = getattr(misc, "import_module_class", None)
    if original_import is None or getattr(original_import, _ST_GATE_MARKER, False):
        return

    def import_module_class(
        class_ref,
        model_name_or_path = None,
        *args,
        **kwargs,
    ):
        _refuse_custom_module(class_ref, model_name_or_path, kwargs.get("trust_remote_code"))
        return original_import(class_ref, model_name_or_path, *args, **kwargs)

    setattr(import_module_class, _ST_GATE_MARKER, True)
    import_module_class.__wrapped__ = original_import
    # Rebind every module that imported the function by name.
    for module in list(sys.modules.values()):
        if (
            getattr(module, "__name__", "").startswith("sentence_transformers")
            and getattr(module, "import_module_class", None) is original_import
        ):
            setattr(module, "import_module_class", import_module_class)


def _route_linears_through_onednn(model) -> int:
    """Run a CPU model's fp32 Linear layers in oneDNN when torch's BLAS is Windows ROCm's OpenBLAS.

    That DLL does ~4 GFLOPS per core against MKL's ~130, so bge-small indexed ~7x slower on all cores
    (#14313). oneDNN ships in the same wheel; the vectors match to ~2e-7, so the index stays valid.
    Probed on the model's own first layer; any doubt keeps the stock path. Returns layers rerouted.
    """
    if not runtime_capped_openblas_loaded():
        return 0
    try:
        import torch

        if not getattr(torch.version, "hip", None) or not torch.backends.mkldnn.is_available():
            return 0
        linears = [
            m
            for m in model.modules()
            if type(m) is torch.nn.Linear
            and m.weight.device.type == "cpu"
            and m.weight.dtype == torch.float32
        ]
        if not linears:
            return 0

        class OneDnnLinear(torch.nn.Linear):
            def forward(self, x):
                return torch.ops.mkldnn._linear_pointwise(x, self.weight, self.bias, "none", [], "")

        probe = linears[0]
        with torch.inference_mode():
            x = torch.randn(2, 5, probe.in_features)
            expected = torch.nn.functional.linear(x, probe.weight, probe.bias)
            got = OneDnnLinear.forward(probe, x)
        if got.shape != expected.shape or not torch.allclose(got, expected, rtol = 1e-4, atol = 1e-5):
            return 0
    except Exception:  # noqa: BLE001 -- the stock path is always correct
        return 0
    for module in linears:
        module.__class__ = OneDnnLinear
    logger.info(
        "embedding model: %d Linear layers run in oneDNN (ROCm OpenBLAS is slow on CPU)",
        len(linears),
    )
    return len(linears)


def _get(model_name: str | None = None):
    """Cached SentenceTransformer, (re)loading on a name change. Loaded in fp16 on an
    accelerator for a ~1.5x speedup at negligible accuracy loss, fp32 on CPU; see ``_load_dtype``
    for the models that cannot run in fp16."""
    account_path(model_name, reference = True)
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
            device = _load_device()
            _install_torchao_stub_once()
            from sentence_transformers import SentenceTransformer
            from utils.hf_cache_settings import active_hf_hub_cache

            _gate_st_custom_modules()

            logger.info("loading embedding model %s on %s", name, device)
            st_kwargs = dict(
                device = device,
                cache_folder = active_hf_hub_cache(),
                model_kwargs = dtype_kwargs(_load_dtype(device, name)),
            )
            if managed_account():
                st_kwargs["token"] = False
            load_target = name
            from utils.paths import is_local_path
            from utils.utils import cached_st_source, hf_cache_snapshot_dir

            # Repo ids only: a local folder with a Hub-like name must not load the Hub's weights.
            st_source = None if is_local_path(name) else cached_st_source(name)
            if not local_only and st_source is not None:
                # Load the cached snapshot path, not the repo id, so ST cannot fetch a newer revision.
                load_target = str(st_source[1])
            if local_only:
                # ST-specific AND complete: a hybrid repo's cached GGUF, or a transfer that finalized only its first
                # shard, would otherwise retire the marker.
                if download_pending and st_source is None:
                    # Check and lookup share no lock: eviction between them is still pending.
                    raise EmbeddingModelDownloadRequiredError(
                        f"Embedding model {name!r} is not downloaded yet. "
                        "Finish its Settings download before indexing documents."
                    )
                snapshot = st_source[1] if st_source else hf_cache_snapshot_dir(name)
                if snapshot is not None:
                    # Re-emit in finally: MISSING or MISMATCH lines matter most when the load raises.
                    load_target = str(snapshot)
                elif download_pending:
                    raise EmbeddingModelDownloadRequiredError(
                        f"Embedding model {name!r} is not downloaded yet. "
                        "Finish its Settings download before indexing documents."
                    )
                elif _st_accepts_local_files_only(SentenceTransformer):
                    st_kwargs["local_files_only"] = True
            # Scan only after load_target is settled, so the scanned commit is the loaded one.
            _guard_model_security(load_target, offline, display = name)
            with _quiet_transformers_load() as report:
                # Retire only after construction succeeds, or a failed load falls to llama-server with no marker.
                try:
                    try:
                        _model = SentenceTransformer(load_target, **st_kwargs)
                    except Exception as exc:  # noqa: BLE001 - e.g. CUDA OOM beside a resident chat model
                        # Retry on CPU before encode()'s llama-server fallback, which changes the embedding space.
                        if device == "cpu" or config.embed_device_requires_gpu():
                            raise
                        logger.warning(
                            "loading embedding model %s on %s failed (%s); retrying on CPU",
                            name,
                            device,
                            exc,
                        )
                        st_kwargs["device"] = "cpu"
                        st_kwargs["model_kwargs"] = dtype_kwargs(_load_dtype("cpu", name))
                        _model = SentenceTransformer(load_target, **st_kwargs)
                finally:
                    _emit_load_reports(report)
            _route_linears_through_onednn(_model)
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
    """ST encode to (N, dim) float32, serialised for the fast-tokenizer borrow check."""
    with _compute_lock:
        # ST encode failed after load: switch to llama-server so later encodes share one vector space.
        model = _get(model_name)
        # Affinity narrowed in Task Manager mid-index must reach this thread's pool (#14313).
        fit_threads_to_affinity()
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


def _st_max_tokens(model_name: str | None = None) -> int | None:
    with _compute_lock:
        model = _get(model_name)
        limit = getattr(model, "max_seq_length", None)
        if not limit:
            return None
        tokenizer = getattr(model, "tokenizer", None)
        specials = getattr(tokenizer, "num_special_tokens_to_add", None)
        reserved = int(specials()) if callable(specials) else 0
        prompts = getattr(model, "prompts", None) or {}
        prompt = prompts.get(getattr(model, "default_prompt_name", None))
        if prompt and tokenizer is not None:
            reserved += len(tokenizer.encode(prompt, add_special_tokens = False))
        return max(1, int(limit) - reserved)


_ST_COUNT_CACHE_SIZE = 4096


def _st_token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Reads the tokenizer per call under _compute_lock; caching it would outlive an unload.
    The count memo is tied to that tokenizer by a weakref: a switch empties it, an unload frees it."""
    counts: dict[str, int] = {}
    counted_by: list = [None]

    def _count(t: str) -> int:
        with _compute_lock:
            tok = _get(model_name).tokenizer
            owner = counted_by[0]
            if owner is None or owner() is not tok:
                counts.clear()
                try:
                    counted_by[0] = weakref.ref(tok)
                except TypeError:
                    counted_by[0] = None
            memo = counted_by[0] is not None and type(t) is str
            known = counts.get(t) if memo else None
            if known is not None:
                # Leave the environment a tokenize would.
                if os.environ.get("TOKENIZERS_PARALLELISM") != "false":
                    os.environ["TOKENIZERS_PARALLELISM"] = "false"
                return known
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            try:
                n = len(tok.encode(t, add_special_tokens = False))
            finally:
                os.environ["TOKENIZERS_PARALLELISM"] = "false"
            if memo:
                if len(counts) >= _ST_COUNT_CACHE_SIZE:
                    counts.pop(next(iter(counts)))
                counts[t] = n
            return n

    return _count


def _release_st_model() -> bool:
    """Drop the module-level SentenceTransformer without racing an encode."""
    global _model, _name
    with _compute_lock:
        with _lock:
            released = _model is not None
            _model = None
            _name = None
    if released:
        # Dropping the reference leaves its blocks reserved; gated, since before detection clear_gpu_cache() runs it.
        try:
            from utils import hardware
            if hardware.hardware.DEVICE is not None:
                hardware.clear_gpu_cache()
        except Exception as e:  # noqa: BLE001 - freeing is best effort
            logger.debug("GPU cache clear after embedder release failed: %s", e)
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
            # Directory names prove nothing; only the filesystem shows that a dir holding safetensors is ST.
            fallback = _switch_to_llama_fallback(st_err, model_name)
            if fallback is None:
                raise
            _served_by.backend = fallback
            return fallback.encode(texts, model_name = model_name, normalize = normalize)

    def token_counter(self, *, model_name = None):
        return _st_token_counter(model_name)

    def dim(self, *, model_name = None):
        return _st_dim(model_name)

    def max_tokens(self, *, model_name = None):
        return _st_max_tokens(model_name)

    def warm(self, *, model_name = None):
        _get(model_name)


_backend_lock = threading.Lock()
_backend = None
_backend_key: str | None = None
# Read without _backend_lock (held across model loads); dict.get is atomic.
_forced_backends: dict[str, str] = {}

_ST_ALIASES = frozenset({"sentence-transformers", "sentence_transformers", "st"})
_LLAMA_ALIASES = frozenset(
    {"llama-server", "llama_server", "llama", "llama.cpp", "llamacpp", "gguf"}
)
_AUTO_ALIASES = frozenset({"auto", ""})


def _resolve_auto() -> str:
    """Auto picks sentence-transformers on a CUDA/ROCm GPU, else llama-server; GPU detection uses smi."""
    from core.inference.llama_cpp import LlamaCppBackend

    # Unfiltered probe: the llama.cpp ROCm arch gate (#7624) does not apply to PyTorch.
    if LlamaCppBackend._get_gpu_free_memory():
        return "sentence-transformers"
    if LlamaCppBackend._find_llama_server_binary():
        return "llama-server"
    return "sentence-transformers"


# Cached: the auto GPU probe is a subprocess (#10390).
_resident_hardware: tuple[object, str] | None = None
_hardware_probe = threading.local()


def _resident_hardware_choice() -> str:
    """_resolve_auto, asked once for the backend that is published."""
    held = _resident_hardware
    if held is not None and _backend is not None and held[0] is _backend:
        return held[1]
    _hardware_probe.choice = _resolve_auto()
    return _hardware_probe.choice


def _keep_hardware_choice(backend) -> None:
    """Keep the answer only when THIS call resolved it; see the clear in _get_backend."""
    global _resident_hardware
    choice = getattr(_hardware_probe, "choice", None)
    if choice is not None:
        _resident_hardware = (backend, choice)


def _model_is_local_gguf(model: str | None) -> bool:
    """True for a local .gguf file or folder; the cheap is_local_path check runs first on the hot path."""
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
    """Name-only test for a -GGUF repo: it has no safetensors for sentence-transformers to open."""
    if not model:
        return False
    try:
        from utils.paths import is_local_path

        # Before the stored record: asks the filesystem, not a guess; RAG_EMBED_BACKEND still overrides.
        if is_local_path(model):
            return False
    except Exception:  # noqa: BLE001 - unparseable path is not a repo id either
        return False
    # Must agree with config's predicate used by gguf_repo_candidates.
    return config._names_gguf(model.strip().rstrip("/").rsplit("/", 1)[-1])


# ST under auto stays on the CPU in float32 (_device); llama-server offloads. Exact ids: a fine-tune may lack the -GGUF.
_LLAMA_SERVER_PREFERRED_MODELS = frozenset(
    {"unsloth/embeddinggemma-300m", "unsloth/embeddinggemma-2"}
)


def _plan_prefers_llama_server(model: str) -> bool:
    """Picker plan only: the runtime follows the saved record, so an env-configured model keeps its embedding space."""
    if model.strip().rstrip("/").lower() not in _LLAMA_SERVER_PREFERRED_MODELS:
        return False
    # The runtime serves the unrecorded model in effect on the hardware default; another plan would be a dead download.
    if model == config.effective_embedding_model():
        return False
    # A pinned GGUF repo serves other weights; an explicit device already chose where torch runs.
    if config.gguf_repo_is_explicit() or config.embed_device_preference() != "auto":
        return False
    try:
        from utils.embedding_model_settings import get_stored_backend
        return not get_stored_backend(model)
    except Exception:  # noqa: BLE001 - store unavailable: keep the hardware plan
        return False


def _resolve_auto_for_model(model_name: str | None = None) -> str:
    """Auto, but honouring the saved backend: an embedder with no GGUF stays on sentence-transformers."""
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
    return _resident_hardware_choice()


def sentence_transformers_runtime_available() -> bool:
    """Checks torch, sentence-transformers and the device, without building a model that may download."""
    try:
        _load_device()
        # Not under _lock: _get holds it across a whole model download/construction.
        _install_torchao_stub_once()
        from sentence_transformers import SentenceTransformer

        return callable(SentenceTransformer)
    except Exception as exc:  # noqa: BLE001 - any failed runtime import selects the fallback
        logger.debug("sentence-transformers runtime preflight failed: %s", exc)
        return False


def _llama_server_runtime_available() -> bool:
    try:
        from core.inference.llama_cpp import LlamaCppBackend
        return bool(LlamaCppBackend._find_llama_server_binary())
    except Exception:  # noqa: BLE001 - an unavailable fallback cannot be planned
        return False


_ST_LOAD_PREFLIGHT_TIMEOUT_S = 5.0
# Proofs last one Settings resolution: a folder can be fixed, a repo republished, or the next caller lack access.
_st_unloadable_proofs: ContextVar[set[str] | None] = ContextVar(
    "st-unloadable-proofs", default = None
)


@contextmanager
def st_load_proof_scope():
    """Keep ``sentence_transformers_can_load`` proofs for the enclosed resolution. Nested scopes share one."""
    marker = _st_unloadable_proofs.set(set()) if _st_unloadable_proofs.get() is None else None
    try:
        yield
    finally:
        if marker is not None:
            _st_unloadable_proofs.reset(marker)


def sentence_transformers_known_unloadable(model_name: str) -> bool:
    """Whether a check earlier in this scope proved ``model_name`` unloadable. No network."""
    proofs = _st_unloadable_proofs.get()
    return proofs is not None and model_name in proofs


def sentence_transformers_can_load(model_name: str, token: str | bool | None = None) -> bool:
    """False when ``modules.json`` names a sentence_transformers class this install lacks (embeddinggemma-2 names
    6.x ``sentence_transformers.base`` modules) or ``config.json`` a ``model_type`` transformers does not know.
    True when either file cannot be read, so an unreachable repo keeps its plan."""
    if sentence_transformers_known_unloadable(model_name):
        return False
    if _st_load_preflight(model_name, token):
        return True
    proofs = _st_unloadable_proofs.get()
    if proofs is not None:
        proofs.add(model_name)
    return False


def _st_load_preflight(model_name: str, token: str | bool | None = None) -> bool:
    try:
        from sentence_transformers.util import import_from_string
        from utils.utils import call_with_deadline

        def _metadata():
            modules = _repo_json(model_name, "modules.json", token)
            config_files = ["config.json"] if modules is None else []
            for module in modules or ():
                ref = str((module or {}).get("type", ""))
                if ref.rsplit(".", 1)[-1] != "Transformer":
                    continue
                path = str((module or {}).get("path", "")).strip("/")
                filename = f"{path}/config.json" if path else "config.json"
                if filename not in config_files:
                    config_files.append(filename)
            return modules, {
                filename: _repo_json(model_name, filename, token) for filename in config_files
            }

        # bounded with one deadline so nested module metadata cannot multiply the wait
        modules, model_configs = call_with_deadline(
            _metadata,
            _ST_LOAD_PREFLIGHT_TIMEOUT_S,
            name = "embed-st-load-preflight",
        )
        for module in modules or ():
            ref = str((module or {}).get("type", ""))
            # Other classes are third-party code, which _gate_st_custom_modules refuses on its own.
            if not ref.startswith(_ST_PACKAGE_PREFIX):
                continue
            try:
                # How ST's own loader resolves sentence_transformers.* refs.
                import_from_string(ref)
            except ImportError:
                return False
        for model_config in model_configs.values():
            model_type = (model_config or {}).get("model_type")
            if isinstance(model_type, str):
                # CONFIG_MAPPING, not CONFIG_MAPPING_NAMES, so types registered at runtime count. An auto_map does not
                # help an unknown type: the loader never passes trust_remote_code.
                from transformers import CONFIG_MAPPING
                if model_type not in CONFIG_MAPPING:
                    return False
    except Exception as exc:  # noqa: BLE001 - offline, gated, missing or malformed: no proof either way
        logger.debug("sentence-transformers load preflight for %s failed: %s", model_name, exc)
    return True


def resolved_backend_for_model(model_name: str, token: str | bool | None = None) -> str:
    """Backend a fresh operation for ``model_name`` would actually select."""
    raw = _raw_backend()
    forced = _forced_backends.get(model_name)
    key = forced or (_resolve_auto_for_model(model_name) if raw in _AUTO_ALIASES else raw)
    # Without a llama binary ST is the only plan; its error beats a fabricated GGUF destination. Stat before Hub.
    if key in _ST_ALIASES and _llama_server_runtime_available():
        use_llama = (
            # Auto only: an explicit ST policy ignores both the preference and the stored backend.
            (raw in _AUTO_ALIASES and _plan_prefers_llama_server(model_name))
            or not sentence_transformers_runtime_available()
            # An ST plan for it never loads, and its pending marker blocks the llama fallback's GGUF.
            or (raw in _AUTO_ALIASES and not sentence_transformers_can_load(model_name, token))
        )
        if use_llama:
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
    """A llama-server backend if its binary exists, else None; no server starts until warm."""
    from core.inference.llama_cpp import LlamaCppBackend

    if not LlamaCppBackend._find_llama_server_binary():
        return None
    from .embed_llama_server import LlamaServerBackend

    return LlamaServerBackend()


def _build_st_backend_or_fallback(model_name: str | None = None):
    """Warms the pinned model_name, not the live setting; a failed ST probe falls back to llama-server."""
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
    """After a runtime ST encode failure, swaps to llama-server; earlier ST vectors need a reindex."""
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
    # ST wrappers share one module-level model; the warmed replacement would lose it if cleared here.
    _dispose_replaced_backend(old, fallback)
    return fallback


def _raw_backend() -> str:
    return (config.EMBED_BACKEND or "auto").strip().lower()


def sentence_transformers_fallback_allowed(model_name: str | None = None) -> bool:
    """Under an explicit llama config or a runtime ST failure pin, ST cannot be offered for a new model."""
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
    if old is None or old is new:
        return
    if isinstance(old, _SentenceTransformersBackend):
        # ST wrappers share one model; releasing it would discard the replacement just warmed.
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
    """Uses the job's pinned model_name, not the live setting; a Settings change would mislabel vectors."""
    global _backend, _backend_key
    raw = _raw_backend()
    old = None
    new = None
    with _backend_lock:
        model = model_name or config.effective_embedding_model()
        forced = _forced_backends.get(model)
        # Load-bearing: other probes resolve auto outside this lock and would keep a stale answer.
        _hardware_probe.choice = None
        key = forced or (_resolve_auto_for_model(model) if raw in _AUTO_ALIASES else raw)
        if _backend is not None and _backend_key == _backend_cache_key(raw, key):
            _keep_hardware_choice(_backend)
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
        _keep_hardware_choice(new)
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
    global _backend, _backend_key, _resident_hardware
    with _backend_lock:
        _forced_backends.clear()
        _backend = None
        _backend_key = None
        _resident_hardware = None


def backend_is_loaded(model_name: str | None = None) -> bool:
    """Lock-free on purpose: the load locks are held across whole model loads, which would stall status."""
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
            # Unload is a fresh start: clearing _forced_backends lets the saved model pick its backend again.
            if not backend._process_alive():
                return False
            return backend._model_repo == config.effective_gguf_repo_for_embedding_model(model_name)
        except Exception:  # noqa: BLE001 - a status probe must never block settings
            return False
    return False


def release_backend() -> bool:
    """Drops the embedder, stopping its llama-server; safe mid-ingestion as the next embed rebuilds it."""
    global _backend, _backend_key, _resident_hardware
    with _backend_lock:
        # Unload is a fresh start: a past runtime fallback stops pinning the backend choice.
        _forced_backends.clear()
        backend, _backend, _backend_key = _backend, None, None
        _resident_hardware = None
    if backend is None:
        # Live servers keep spawn-time pooling; a stopped backend re-reads the file, so dedup must match.
        return _release_st_model()
    _dispose_replaced_backend(backend)
    return True


def active_backend_is_llama(model_name: str | None = None) -> bool:
    """True when the built backend is llama-server, including runtime fallbacks; never raises."""
    try:
        with _backend_lock:
            backend = _backend
        if backend is not None:
            # Report the actual backend, so a concrete ST backend's pickle stays gated.
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


def _llama_pooling(name: str, served = None) -> str | None:
    try:
        from .embed_llama_server import LlamaServerBackend, _gguf_pooling
    except Exception:  # noqa: BLE001 - llama plumbing import must never block
        return None
    # The file the running server loaded wins over a cache search: moving the HF cache leaves it serving the old path.
    backend = served if served is not None else _backend
    desired = config.effective_gguf_repo_for_embedding_model(name)
    if isinstance(backend, LlamaServerBackend):
        path, repo, captured, alive = backend.pooling_identity_snapshot()
    else:
        path, repo, captured, alive = None, None, None, False
    if path and repo == desired:
        # AppleDouble files keep the extension, so a text parser would embed and cite them as chunks.
        if served is not None or alive:
            pooling = captured or _gguf_pooling(path)
        elif os.path.isfile(path):
            pooling = _gguf_pooling(path)
        else:
            pooling = LlamaServerBackend.cached_pooling(name)
    else:
        pooling = LlamaServerBackend.cached_pooling(name)
    if pooling is None:
        # Nothing on disk to read, so match no stored row: pre-encode dedupe must not take forced-CLS vectors as current.
        return "unresolved"
    return None if pooling == "cls" else pooling


def _identity(
    is_llama: bool,
    name: str,
    served = None,
    served_identity: tuple[str | None, str | None] | None = None,
) -> str:
    if is_llama:
        if served_identity is None:
            repo = config.effective_gguf_repo_for_embedding_model(name)
            pooling = _llama_pooling(name, served)
        else:
            repo, captured_pooling = served_identity
            pooling = None if captured_pooling == "cls" else captured_pooling
        return config.embedding_identity(
            "llama-server",
            name,
            gguf_repo = repo,
            pooling = pooling,
        )
    return config.embedding_identity("sentence-transformers", name)


def _identity_backend_is_llama(name: str) -> bool:
    """Predicts the backend the next encode for name uses, including any replacement it will trigger."""
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
    """Identity of the vector space, not the model name: llama-server uses the GGUF's own pooling."""
    name = model_name or config.effective_embedding_model()
    return _identity(_identity_backend_is_llama(name), name)


def _is_llama_backend(backend) -> bool:
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
    """Takes identity from the encode that made the vectors; a concurrent ST swap could mislabel them."""
    _served_by.backend = None
    _served_by.llama_identity = None
    vectors = encode(texts, model_name = model_name, normalize = normalize)
    served = getattr(_served_by, "backend", None)
    served_identity = getattr(_served_by, "llama_identity", None)
    name = model_name or config.effective_embedding_model()
    if served is None:
        return vectors, embedding_identity(name)
    return vectors, _identity(_is_llama_backend(served), name, served, served_identity)


def warm(model_name: str | None = None) -> None:
    """Eagerly load the embedder so the first real request isn't slow."""
    _get_backend(model_name).warm(model_name = model_name)


def encode(
    texts: list[str],
    *,
    model_name: str | None = None,
    normalize: bool = True,
):
    """An explicit unload retires the backend; reacquire the new one rather than fail the document."""
    backend = _get_backend(model_name)
    _served_by.backend = backend
    _served_by.llama_identity = None
    try:
        vectors = backend.encode(texts, model_name = model_name, normalize = normalize)
        served = getattr(_served_by, "backend", None) or backend
        _served_by.llama_identity = getattr(served, "served_embedding_identity", lambda: None)()
        return vectors
    except RuntimeError:
        if not (_is_llama_backend(backend) and getattr(backend, "_closed", False)):
            raise
    replacement = _get_backend(model_name)
    if replacement is backend:
        raise RuntimeError("llama-server embedding backend was unloaded")
    _served_by.backend = replacement
    vectors = replacement.encode(texts, model_name = model_name, normalize = normalize)
    _served_by.llama_identity = getattr(replacement, "served_embedding_identity", lambda: None)()
    return vectors


def dim(model_name: str | None = None) -> int:
    return _get_backend(model_name).dim(model_name = model_name)


def max_tokens(model_name: str | None = None) -> int | None:
    return _get_backend(model_name).max_tokens(model_name = model_name)


def clip_to_limit(text: str, count, limit: int | None) -> tuple[str, bool]:
    """The longest prefix of ``text`` within ``limit`` tokens by ``count``, and whether it was cut."""
    if not limit or count(text) <= limit:
        return text, False
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo], True


def chunk_tokens(model_name: str | None = None) -> int:
    limit = max_tokens(model_name)
    return min(config.CHUNK_TOKENS, limit) if limit else config.CHUNK_TOKENS


def token_counter(model_name: str | None = None) -> Callable[[str], int]:
    """Reacquires the backend only for an unload's lifecycle failure; other tokenizer errors propagate."""
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
