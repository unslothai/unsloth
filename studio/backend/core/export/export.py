# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export backend - exports models in various formats."""

import glob
import json
import structlog
import tempfile
from loggers import get_logger
import os
import sys
import shutil
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List

# unsloth imports torch, so a --no-torch install raises here; stay importable and return a clean
# "PyTorch is not installed" error.
try:
    from unsloth import FastLanguageModel, FastVisionModel, _IS_MLX
    _UNSLOTH_IMPORT_ERROR = None
except Exception as _unsloth_exc:
    FastLanguageModel = None
    FastVisionModel = None
    _IS_MLX = False
    _UNSLOTH_IMPORT_ERROR = _unsloth_exc

from huggingface_hub import HfApi, ModelCard
from hub.utils.hf_tokens import HfTokenArg, apply_token_to_child_env, is_anonymous, normalize_token
from utils.hardware import clear_gpu_cache

from utils.models import is_vision_model, get_base_model_from_lora
from utils.models.model_identity import restore_hf_cache_repo_identity
from utils.models.model_config import detect_audio_type
from utils.paths import (
    ensure_dir,
    outputs_root,
    resolve_export_write_dir,
    resolve_output_dir,
)
from core.inference import get_inference_backend
from core.export import q4nx
from utils.paths.path_utils import any_not_appledouble_metadata, drop_appledouble_metadata

# GPU/PyTorch-only imports, skipped on MLX and --no-torch installs so the module stays importable.
torch = None
_TORCH_IMPORT_ERROR: Optional[BaseException] = None
if not _IS_MLX:
    try:
        from peft import PeftModel, PeftModelForCausalLM
        import torch
    except Exception as _torch_exc:
        _TORCH_IMPORT_ERROR = _torch_exc

logger = get_logger(__name__)

_ADAPTER_WEIGHT_NAMES = {
    "mlx": frozenset({"adapters.safetensors"}),
    "peft": frozenset({"adapter_model.safetensors", "adapter_model.bin"}),
}
_ZOO_UPGRADE_MESSAGE = (
    "PEFT adapter conversion requires an updated unsloth-zoo. Upgrade "
    "unsloth-zoo, then retry adapter_format='peft' (or GGUF adapter export)."
)


def _other_adapter_weight_names(resolved_format: str) -> frozenset:
    return _ADAPTER_WEIGHT_NAMES["peft" if resolved_format == "mlx" else "mlx"]


def _resolve_adapter_format(adapter_format: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (format, error); omission resolves to the platform's native format."""
    if adapter_format is None:
        return ("mlx" if _IS_MLX else "peft"), None
    if adapter_format not in _ADAPTER_WEIGHT_NAMES:
        return None, f"Invalid adapter_format '{adapter_format}'. Choose 'mlx' or 'peft'."
    if adapter_format == "mlx" and not _IS_MLX:
        return None, (
            "adapter_format='mlx' is only available on Apple-silicon MLX "
            "servers; this server exports the native PEFT format."
        )
    return adapter_format, None


def _load_in_4bit_kwargs(load_in_4bit: bool) -> dict:
    # True is the loaders' default; passing it reads as an explicit request to requantize fp8 checkpoints to NF4.
    return {} if load_in_4bit else {"load_in_4bit": False}


def _export_runtime_available() -> bool:
    """True if export can run: MLX active, or Unsloth imported (only succeeds on a GPU host)."""
    return bool(_IS_MLX) or (FastLanguageModel is not None)


def _export_runtime_message() -> str:
    """Precise reason the export runtime is unavailable, mirroring hardware.export_capability()."""
    if torch is None:
        return (
            "PyTorch is not installed. Model export requires PyTorch with a supported accelerator "
            "(NVIDIA, AMD, or Intel GPU) or Apple Silicon (MLX). Install PyTorch to enable export."
        )
    return (
        "Export requires an NVIDIA, AMD, or Intel GPU, or Apple Silicon (MLX). No supported "
        "accelerator was found on this host. (PyTorch is installed, but Unsloth cannot export on "
        "CPU only.)"
    )


# Kept for call sites / tests referencing the PyTorch-missing text.
_PYTORCH_MISSING_MESSAGE = (
    "PyTorch is not installed. Model export requires PyTorch with a supported accelerator "
    "(NVIDIA, AMD, or Intel GPU) or Apple Silicon (MLX). Install PyTorch to enable export."
)

_LLAMA_CPP_SCRIPTS_WARNING_EMITTED = False


@contextlib.contextmanager
def _llama_cpp_scripts_pin():
    """Pin convert_hf_to_gguf.py to setup.sh's llama.cpp ref for one conversion.

    Scoped and marked internal because UNSLOTH_LLAMA_CPP_SCRIPTS_DIR is read as the
    user's own choice: it outranks UNSLOTH_LLAMA_CPP_CONVERTER_TAG, and it exempts the
    converter from the UNSLOTH_CONVERTER_SCAN_STRICT refusal. A pin the user set is left
    exactly as it is; that one carries their exemption.
    """
    global _LLAMA_CPP_SCRIPTS_WARNING_EMITTED
    if _IS_MLX:
        # The MLX save path pins for itself, under a plain threading.Lock held across the
        # conversion: entering it here too nests, and the second entry never returns.
        yield
        return
    try:
        from unsloth_zoo.llama_cpp import (
            LLAMA_CPP_DEFAULT_DIR,
            _resolve_local_convert_script,  # noqa: F401
        )
    except Exception:
        # Not just ImportError: a half-built unsloth_zoo raises RuntimeError or AttributeError.
        if not _LLAMA_CPP_SCRIPTS_WARNING_EMITTED:
            logger.warning(
                "Unsloth: installed unsloth_zoo does not honor "
                "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR; convert_hf_to_gguf.py will "
                "still be downloaded from llama.cpp master and may drift "
                "past the pinned llama-quantize binary. Upgrade unsloth_zoo "
                "to activate the local script pin."
            )
            _LLAMA_CPP_SCRIPTS_WARNING_EMITTED = True
        yield
        return

    if os.environ.get("UNSLOTH_LLAMA_CPP_CONVERTER_TAG", "").strip():
        # The pin outranks the tag, so pinning here is what made setting a tag do nothing.
        yield
        return

    try:
        from unsloth_zoo.llama_cpp import _converter_dir_is_incomplete
        incomplete = _converter_dir_is_incomplete(LLAMA_CPP_DEFAULT_DIR)
    except Exception:
        # An older unsloth_zoo has no such check, and it only ever skips the pin.
        incomplete = False
    if incomplete:
        # Pinning an entrypoint with no conversion/ beside it is what stops the staged
        # resolver from fetching a co-versioned set that runs.
        yield
        return

    try:
        from unsloth_zoo.llama_cpp import internal_scripts_dir_pin
    except ImportError:
        internal_scripts_dir_pin = None

    if internal_scripts_dir_pin is not None:
        with internal_scripts_dir_pin(LLAMA_CPP_DEFAULT_DIR):
            yield
        return

    # Older unsloth_zoo, no internal pin: scope the variable by hand so it cannot leak.
    # Strict mode still takes the exemption there; upgrading unsloth_zoo is the fix.
    existing = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
    if existing is not None:
        yield
        return
    os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = LLAMA_CPP_DEFAULT_DIR
    try:
        yield
    finally:
        os.environ.pop("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", None)


def _multi_gpu_device_map_kwargs() -> dict:
    """``device_map`` kwargs for sharding a checkpoint across every visible GPU.

    unsloth's ``from_pretrained`` defaults to ``device_map="sequential"``, which stacks
    the whole model on GPU0 and OOMs multi-GPU hosts whose other GPUs sit empty (#7053).
    Returns a sharding map only on a real multi-GPU CUDA/ROCm host (mirroring the
    inference loader's ``get_device_map``), else empty so single-GPU, CPU and MLX loads
    keep the loader default."""
    if _IS_MLX:
        return {}
    try:
        from utils.hardware import get_device_map, get_parent_visible_gpu_ids

        visible = get_parent_visible_gpu_ids()
        if len(visible) > 1:
            device_map = get_device_map(visible)
        elif not visible:
            # UUID/MIG masks resolve to no numeric ids; get_device_map(None) falls back to the visible count.
            device_map = get_device_map(None)
        else:
            return {}
        # A "balanced"-only whitelist dropped the map the moment CUDA began asking for a planned one.
        if device_map in ("balanced", "unsloth", "unsloth_balanced"):
            return {"device_map": device_map}
    except Exception as exc:
        logger.debug(f"multi-GPU device_map resolution failed; using loader default: {exc}")
    return {}


def _is_oom_error(exc: BaseException) -> bool:
    """True for an accelerator OOM, however it is spelled.

    accelerate and transformers re-raise it as a plain ``RuntimeError`` on several paths
    and ROCm/XPU use their own classes, so match the message too.
    """
    if torch is not None:
        oom_types = tuple(
            t
            for t in (
                getattr(torch, "OutOfMemoryError", None),
                getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
                getattr(getattr(torch, "xpu", None), "OutOfMemoryError", None),
            )
            if isinstance(t, type)
        )
        if oom_types and isinstance(exc, oom_types):
            return True
    return "out of memory" in f"{type(exc).__name__}: {exc}".lower()


def _is_cpu_spill_rejection(exc: BaseException) -> bool:
    """bitsandbytes refuses a map that spills to CPU/disk with a plain ``ValueError``.

    Busy secondary GPUs can make ``balanced`` spill to CPU even where the old sequential
    load fit on GPU0, and that message says nothing about memory, so the retry has to
    match it explicitly. See transformers ``quantizers/quantizer_bnb_4bit.py``.
    """
    return "dispatched on the cpu or the disk" in str(exc).lower()


def _is_device_map_infeasible(exc: BaseException) -> bool:
    """The planner refusing to place the model, matched by class name.

    It raises rather than spilling a bitsandbytes model to CPU, budgeting from free
    memory read before this process opens a context -- so a training or chat job
    holding the other cards can make it refuse a model the single-device loader
    still fits. By name, so the export does not require a version defining it.
    """
    return type(exc).__name__ == "DeviceMapInfeasible"


class _CpuSpillRetry(Exception):
    """A multi-GPU load that succeeded but left modules offloaded to CPU/disk."""


def _cpu_offloaded_modules(model) -> int:
    """Count the modules a load parked on CPU or disk.

    Only bitsandbytes refuses such a map; a full-precision load accepts it, leaves the
    parameters on meta and dies much later in safetensors with "Cannot copy out of meta
    tensor". Nothing raises at load time, so inspect the map directly. PEFT re-dispatches
    when attaching an adapter, so in practice this catches merged checkpoints.
    """
    device_map = getattr(model, "hf_device_map", None) or {}
    return sum(1 for target in device_map.values() if str(target) in ("cpu", "disk"))


def _accepts_by_keyword(params, name):
    """True if `name` is passable as a keyword, not merely named.

    Every call site passes by keyword, so a positional-only parameter is not support: counting
    it turns a clean refusal into a TypeError.
    """
    import inspect

    parameter = params.get(name)
    return parameter is not None and parameter.kind is not inspect.Parameter.POSITIONAL_ONLY


def _supports_kwarg(fn, name):
    """True if `fn` accepts keyword `name` directly or via **kwargs."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return _accepts_by_keyword(params, name) or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _imatrix_export_supported(save_fn):
    """True when this build can apply an imatrix, not merely swallow the keyword: the MLX binding
    takes `**kwargs` and filters them, so only unsloth_zoo itself settles it."""
    import inspect

    try:
        params = inspect.signature(save_fn).parameters
    except (TypeError, ValueError):
        return False
    if _accepts_by_keyword(params, "imatrix_file"):
        return True
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return False
    try:
        from unsloth_zoo.llama_cpp import resolve_imatrix_file  # noqa: F401
    except Exception:
        # Runs before export_gguf's exception boundary, which must return a tuple, not raise.
        return False
    return True


def _reported_gguf_files(result):
    """Absolute GGUF paths unsloth reported writing, or None if it reported nothing.

    None means "fall back to the legacy heuristics", which covers every older shape:
    pre-2025.10 unsloth returned nothing, is_main_process=False returns None, and
    save_method="lora" returns a str. An empty result is None too, since a stale
    manifest is indistinguishable from an old build.
    """
    if not isinstance(result, dict):
        return None
    files = result.get("gguf_files")
    if not isinstance(files, (list, tuple)):
        return None

    resolved = []
    for entry in files:
        # A malformed payload must not be half-trusted.
        if not isinstance(entry, (str, os.PathLike)):
            return None
        path = os.path.abspath(os.fspath(entry))
        if path.lower().endswith(".gguf") and os.path.isfile(path):
            resolved.append(path)
    return resolved or None


def _materialized_imatrix_path(model_dir, imatrix_file):
    """Where unsloth copies a `*.gguf_file` imatrix beside the model, else None.

    `_materialize_imatrix` drops that copy in the model directory, so the owned-root scan
    would otherwise relocate it as if it were a converted model. Callers compare the whole path
    (see `_is_imatrix`): a basename match would suppress a real output of the same name.
    """
    if imatrix_file is True:
        name = "imatrix_unsloth.gguf"
    elif isinstance(imatrix_file, (str, os.PathLike)):
        base = os.path.basename(os.fspath(imatrix_file))
        if not base.endswith(".gguf_file"):
            return None
        name = base[: -len(".gguf_file")] + ".gguf"
    else:
        return None
    return Path(model_dir) / name


def _is_imatrix(path, imatrix_path):
    """True when `path` is the materialized imatrix, asked of the filesystem rather than `==`.

    The on-disk spelling is the filesystem's to choose (a folding mount changes case, APFS
    stores NFD), so byte-exact `Path.__eq__` misses and the imatrix is relocated as a model.
    """
    if imatrix_path is None:
        return False
    try:
        return os.path.samefile(path, imatrix_path)
    except OSError:
        # Either side may be gone by cleanup time; fall back to a folded comparison.
        return _folded(path) == _folded(imatrix_path)


def _folded(path):
    import unicodedata
    return unicodedata.normalize("NFC", os.path.normcase(os.fspath(path)))


def _compressed_export_supported():
    """True if the installed unsloth build can do FP8/NVFP4 compressed-tensors export."""
    try:
        import unsloth.save as _us
        return hasattr(_us, "_normalize_compressed_method")
    except Exception:
        return False


def _torchao_export_supported():
    """True if the installed unsloth build has the portable torchao FP8/INT8 export path and a real
    torchao to run it; False where torchao is stubbed (its config classes return None)."""
    try:
        if _torchao_runtime_unavailable():
            return False
        import unsloth.save as _us
        return hasattr(_us, "_normalize_torchao_method")
    except Exception:
        return False


def _torchao_runtime_unavailable():
    try:
        from core._torchao_stub import _is_windows_rocm, is_stubbed
        if is_stubbed("torchao"):
            return True
        # Windows ROCm only gets real torchao through install_torchao_windows_rocm_real_or_stub().
        return _is_windows_rocm() and "torchao" not in sys.modules
    except Exception:
        return False


def _is_torchao_alias(alias):
    """Any torchao spelling unsloth's normalizer accepts, else a torchao_ prefix, so the Windows
    ROCm guard catches it before it is misread as compressed-tensors."""
    if not alias:
        return False
    try:
        import unsloth.save as _us
        if _us._normalize_torchao_method(alias) is not None:
            return True
    except Exception:
        pass
    return str(alias).lower().startswith("torchao")


def _has_nvidia_gpu():
    """True only on a real NVIDIA CUDA box (not ROCm/XPU/CPU/MLX); compressed-tensors needs it."""
    try:
        from utils.hardware import hardware as _hw
        return _hw.DEVICE == _hw.DeviceType.CUDA and not _hw.IS_ROCM
    except Exception:
        try:
            import torch
            return bool(torch.cuda.is_available()) and getattr(torch.version, "hip", None) is None
        except Exception:
            return False


def _hf_offline(timeout = 3):
    """True if export should avoid the Hub: honors the HF offline env vars, else does one
    cheap TCP reachability probe so a network-down load uses local files / the HF cache
    instead of hanging on connection timeouts. Proxy-aware (probes the proxy egress when
    one is configured); disable the probe with UNSLOTH_OFFLINE_PROBE=0."""
    _offline = {"1", "true", "yes", "on"}
    if (
        os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in _offline
        or os.environ.get("TRANSFORMERS_OFFLINE", "").strip().lower() in _offline
    ):
        return True
    if os.environ.get("UNSLOTH_OFFLINE_PROBE", "1").strip().lower() in {"0", "false", "no", "off"}:
        return False

    # Shared bounded, proxy-aware probe (also used by the export worker before version activation).
    from utils.transformers_version import hf_endpoint_unreachable

    if hf_endpoint_unreachable(timeout):
        logger.warning("Hugging Face endpoint unreachable; loading checkpoint in offline mode")
        return True
    return False


try:
    from unsloth.models.loader_utils import _force_hf_offline
except Exception:
    import contextlib as _contextlib

    @_contextlib.contextmanager
    def _force_hf_offline():
        yield


def _offline_window_if(local_files_only):
    """Forced-offline window when offline was detected, else a no-op context."""
    return _force_hf_offline() if local_files_only else contextlib.nullcontext()


def _is_wsl():
    """Detect if running under Windows Subsystem for Linux."""
    try:
        return "microsoft" in open("/proc/version", encoding = "utf-8").read().lower()
    except Exception:
        return False


def _apply_wsl_sudo_patch():
    """On WSL, monkey-patch do_we_need_sudo() to return False.

    WSL lacks passwordless sudo and do_we_need_sudo()'s `sudo apt-get update`
    hangs on a stdin password; setup.sh pre-installs the build deps anyway.
    """
    if not _is_wsl():
        return

    try:
        import unsloth_zoo.llama_cpp as llama_cpp_module

        def _wsl_do_we_need_sudo(system_type = "debian"):
            logger.info("WSL detected — skipping sudo check (build deps pre-installed by setup.sh)")
            return False

        llama_cpp_module.do_we_need_sudo = _wsl_do_we_need_sudo
        logger.info("Applied WSL sudo patch to unsloth_zoo.llama_cpp.do_we_need_sudo")
    except Exception as e:
        logger.warning(f"Could not apply WSL sudo patch: {e}")


MODEL_CARD = """---
base_model: {base_model}
tags:
- text-generation-inference
- transformers
- unsloth
- {model_type}
- {extra}
license: apache-2.0
language:
- en
---

# Uploaded finetuned {method} model

- **Developed by:** {username}
- **License:** apache-2.0
- **Finetuned from model :** {base_model}

This {model_type} model was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth) and Huggingface's TRL library.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
"""


# Both llama.cpp spellings: the Hub filters on the exact string, and upstream's add_tags is a no-op.
GGUF_MODEL_CARD = """---
tags:
- gguf
- llama.cpp
- llama-cpp
- unsloth{vlm_tag}
---

# {name} : GGUF

This model was converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth).

**Example usage**:
- For text only LLMs:    `llama-cli -hf {repo_id} --jinja`
- For multimodal models: `llama-mtmd-cli -hf {repo_id} --jinja`

## Available model files:
{files}
"""


# export_metadata.json is local bookkeeping whose base model can be a local path; the GGUF push
# keeps it out of the repo too.
_HUB_UPLOAD_IGNORE = ["export_metadata.json", "._*"]

_STAGING_PREFIX = "unsloth-hub-upload-"


def _staging_dir(export_parent):
    """Stage a copy of an export where it fits: merge_and_overwrite_lora refuses a save its
    destination cannot hold, and neither the temporary directory nor the export's own filesystem is
    reliably the roomier, or even writable — only the export directory itself has to be.
    """
    roomiest = []
    for parent in (Path(tempfile.gettempdir()), Path(export_parent)):
        try:
            roomiest.append((shutil.disk_usage(parent).free, parent))
        except OSError:
            continue
    roomiest.sort(key = lambda candidate: -candidate[0])
    for _, parent in roomiest:
        try:
            return tempfile.TemporaryDirectory(prefix = _STAGING_PREFIX, dir = parent)
        except OSError:
            continue
    return tempfile.TemporaryDirectory(prefix = _STAGING_PREFIX)


def _dir_is_fresh(directory):
    # Finder metadata does not count; the upload drops it anyway.
    return not (
        Path(directory).is_dir() and any_not_appledouble_metadata(Path(directory).iterdir())
    )


def _holds_checkpoint_weights(directory):
    return Path(directory).is_dir() and any(
        entry.suffix in (".safetensors", ".bin") for entry in Path(directory).iterdir()
    )


def _ensure_hub_repo_private(hf_api, repo_id):
    """Tighten an existing repo to private: create_repo sets `private` only at creation."""
    try:
        hf_api.update_repo_settings(repo_id = repo_id, private = True, repo_type = "model")
        return
    except Exception as exception:
        try:
            info = hf_api.repo_info(repo_id = repo_id, repo_type = "model")
            if bool(getattr(info, "private", False)):
                return
        except Exception:
            pass
        raise RuntimeError(
            f"private=True was requested but {repo_id!r} could not be confirmed private "
            "(the token likely lacks `write:repo_settings`, or the repository belongs to "
            "someone else). Refusing to upload rather than publish to a public repository."
        ) from exception


def _open_hub_repo(hf_api, repo_id, private):
    """Create or reuse the repo we are about to upload into, private before anything lands.

    Call this immediately before the upload, not earlier: it is what turns a failure after
    this point into an empty repo.
    """
    repo_url = hf_api.create_repo(repo_id, private = private, exist_ok = True)
    repo_id = getattr(repo_url, "repo_id", repo_id)
    if private:
        _ensure_hub_repo_private(hf_api, repo_id)
    return repo_id


def _push_mlx_merged(
    model, tokenizer, *, save_method, output_path, output_is_fresh, repo_id, hf_token, private
):
    """Upload an MLX merged save; returns the repo id the Hub resolved."""
    with contextlib.ExitStack() as stack:
        upload_dir = output_path
        if not output_is_fresh:
            # A reused folder can hold leftovers, so upload a clean second save; without a
            # local save this is the only one.
            upload_dir = stack.enter_context(
                _staging_dir(Path(output_path).parent)
                if output_path
                else tempfile.TemporaryDirectory()
            )
            model.save_pretrained_merged(upload_dir, tokenizer, save_method = save_method)
        hf_api = HfApi(token = hf_token)
        repo_id = _open_hub_repo(hf_api, repo_id, private)
        hf_api.upload_folder(
            folder_path = upload_dir,
            repo_id = repo_id,
            repo_type = "model",
            ignore_patterns = _HUB_UPLOAD_IGNORE,
        )
    return repo_id


def _publish_unsloth_model_card(hf_api, repo_id, model, hf_token):
    """Write the card the delegated push can no longer write for itself.

    Unsloth's `upload_to_huggingface` only writes it when its own `create_repo(exist_ok=False)`
    finds the repo absent, so opening the repo first silently costs a fresh push its card.
    Best-effort, and an existing card is kept, exactly as the merged and base paths do.
    """
    try:
        if hf_api.file_exists(repo_id, "README.md", repo_type = "model"):
            return
        config = getattr(model, "config", None)
        if config is None:
            return
        base_model = getattr(config, "_name_or_path", "unknown") or "unknown"
        # method/extra reproduce what upload_to_huggingface passed for this path
        # ("finetuned", "trl"): the template already carries the unsloth tag, so `extra`
        # is where trl goes, and the heading already reads "Uploaded finetuned ... model".
        content = MODEL_CARD.format(
            username = repo_id.split("/")[0],
            base_model = repo_id if os.path.isdir(base_model) else base_model,
            model_type = getattr(config, "model_type", "llm"),
            method = "",
            extra = "trl",
        )
        ModelCard(content).push_to_hub(repo_id, token = hf_token, commit_message = "Unsloth Model Card")
    except Exception as exception:
        logger.warning(f"Could not publish the model card: {exception}")


class ExportBackend:
    # {"layout", "adapter_only"} for a decision checkpoint (GGUF only), else None.
    decision: Optional[dict] = None

    def __init__(self):
        self.inference_backend = get_inference_backend()
        self.current_checkpoint = None
        self.current_model = None
        self.current_tokenizer = None
        self.is_vision = False
        self.is_peft = False
        self._audio_type = None
        self.decision = None

    def cleanup_memory(self):
        """Offload and delete all models from memory"""
        try:
            logger.info("Starting memory cleanup...")

            model_names = list(self.inference_backend.models.keys())
            for model_name in model_names:
                self.inference_backend.unload_model(model_name)

            self.current_model = None
            self.current_tokenizer = None
            self.current_checkpoint = None
            self._audio_type = None
            self.decision = None

            clear_gpu_cache()

            logger.info("Memory cleanup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False

    def scan_checkpoints(
        self, outputs_dir: Optional[str] = None
    ) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """
        Scan outputs folder for training runs and their checkpoints.

        Returns: [(model_name, [(display_name, checkpoint_path), ...]), ...]
        """
        if outputs_dir is None:
            outputs_dir = str(outputs_root())
        from utils.models.checkpoints import scan_checkpoints

        return scan_checkpoints(outputs_dir = outputs_dir)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        hf_token: HfTokenArg = None,
        _device_map_override: Optional[dict] = None,
        base_model: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Load a checkpoint for export.

        ``base_model`` is the caller's authorized adapter base; it wins over adapter_config.json.

        ``hf_token`` authenticates the actual weight load for gated/private checkpoints, matching
        the token the worker used for the security preflight (otherwise a gated repo passes scanning
        then 401s at from_pretrained).

        ``False`` (denied the ambient token) must travel all the way down: ``None`` reads as "go and
        find a credential" (``if token is None: get_token()`` in ``save.py`` and ``hf_login``), and
        ``get_token()`` reads the operator's stored login off disk.
        """
        token = normalize_token(hf_token)
        # Loaders only: the probes' cache guards refuse an anonymous read, which offline
        # misreads a cached VLM as a text model.
        probe_token = token or None
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            self.cleanup_memory()

            # Before the base / audio / vision probes: a decision run never reaches the chat loaders.
            from core.export.decision import decision_kind

            decision = decision_kind(checkpoint_path)
            if decision is not None:
                return self._load_decision_checkpoint(
                    str(Path(checkpoint_path).expanduser()), *decision, token = token
                )

            checkpoint_path_obj = Path(checkpoint_path)

            adapter_config = checkpoint_path_obj / "adapter_config.json"
            if adapter_config.exists():
                base_model = base_model or get_base_model_from_lora(checkpoint_path)
                if not base_model:
                    return False, "Could not determine base model for adapter"
            else:
                base_model = None

            model_id = base_model or checkpoint_path

            # Skip the Hub when offline so a no-internet export uses the local cache.
            local_files_only = _hf_offline()

            # Shard across every visible GPU instead of stacking on GPU0 (#7053); {} on single-GPU/CPU/MLX.
            _device_map_kw = (
                _multi_gpu_device_map_kwargs()
                if _device_map_override is None
                else _device_map_override
            )

            # Run the type-detection probes inside the forced-offline window, else a gated base 404s: it
            # covers is_vision_model's Hub reads, and local_files_only makes detect_audio_type's requests.get
            # skip.
            with _offline_window_if(local_files_only):
                self._audio_type = detect_audio_type(
                    model_id, hf_token = probe_token, local_files_only = local_files_only
                )
                self.is_vision = not self._audio_type and is_vision_model(
                    model_id, hf_token = probe_token, local_files_only = local_files_only
                )

            if self._audio_type == "csm":
                from unsloth import FastModel
                from transformers import CsmForConditionalGeneration

                logger.info("Loading as CSM audio model...")
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    auto_model = CsmForConditionalGeneration,
                    load_in_4bit = False,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            elif self._audio_type == "whisper":
                from unsloth import FastModel
                from transformers import WhisperForConditionalGeneration

                logger.info("Loading as Whisper audio model...")
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    dtype = None,
                    load_in_4bit = False,
                    auto_model = WhisperForConditionalGeneration,
                    whisper_language = "English",
                    whisper_task = "transcribe",
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            elif self._audio_type == "snac":
                logger.info("Loading as SNAC (Orpheus) audio model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    **_load_in_4bit_kwargs(load_in_4bit),
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            elif self._audio_type == "bicodec":
                from unsloth import FastModel
                logger.info("Loading as BiCodec (Spark-TTS) audio model...")
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None if _IS_MLX else torch.float32,
                    load_in_4bit = False,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            elif self._audio_type == "dac":
                from unsloth import FastModel
                logger.info("Loading as DAC (OuteTTS) audio model...")
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    load_in_4bit = False,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            elif self.is_vision:
                logger.info("Loading as vision model...")
                model, processor = FastVisionModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    **_load_in_4bit_kwargs(load_in_4bit),
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )
                tokenizer = processor

            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    **_load_in_4bit_kwargs(load_in_4bit),
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
                )

            # Only for the multi-GPU map: a single-GPU host has no second placement to retry on.
            _offloaded = _cpu_offloaded_modules(model) if _device_map_kw else 0
            if _device_map_override is None and _offloaded:
                del model
                raise _CpuSpillRetry(f"{_offloaded} module(s) offloaded to CPU/disk")

            if _IS_MLX:
                # MLX doesn't use PeftModel - detect LoRA via adapter_config.json
                self.is_peft = adapter_config.exists()
            else:
                self.is_peft = isinstance(model, (PeftModel, PeftModelForCausalLM))

            restored_repo_id = restore_hf_cache_repo_identity(model, base_model)
            if restored_repo_id:
                logger.info(
                    f"Restored Hub model identity for legacy adapter export: {restored_repo_id}"
                )

            # The MLX GGUF LoRA converter honors the approved load decision.
            self.trust_remote_code = bool(trust_remote_code)
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_checkpoint = checkpoint_path

            if self._audio_type:
                model_type = f"Audio ({self._audio_type})"
            elif self.is_vision:
                model_type = "Vision"
            else:
                model_type = "Text"
            peft_info = " (PEFT Adapter)" if self.is_peft else " (Merged Model)"

            logger.info(f"Successfully loaded {model_type} model{peft_info}")
            return True, f"Loaded {model_type} model{peft_info} successfully"

        except Exception as e:
            # "balanced" budgets from free memory read BEFORE this process opens a CUDA context, so the shard
            # can OOM when another job owns the other GPUs; fall back once.
            if (
                _device_map_override is None
                and (
                    isinstance(e, _CpuSpillRetry)
                    or _is_oom_error(e)
                    or _is_cpu_spill_rejection(e)
                    or _is_device_map_infeasible(e)
                )
                and _multi_gpu_device_map_kwargs()
            ):
                # Retry outside this block: the live traceback pins the half-built model's frames, so an in-block
                # retry inherits the exhausted device.
                retry_reason = str(e)
            else:
                logger.error(f"Error loading checkpoint: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return False, f"Failed to load checkpoint: {str(e)}"

        logger.warning(
            f"Multi-GPU export load unusable ({retry_reason}); retrying on "
            f"the sequential loader default."
        )
        self.cleanup_memory()
        return self.load_checkpoint(
            checkpoint_path,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            trust_remote_code = trust_remote_code,
            hf_token = hf_token,
            # Name the map: an omitted one is unsloth's DEFAULT_DEVICE_MAP, which requested_device_map
            # upgrades back to the planner, re-running the placement that just failed.
            _device_map_override = {"device_map": "sequential"},
        )

    def _load_decision_checkpoint(
        self,
        checkpoint_path: str,
        layout: str,
        adapter_only: bool,
        token: HfTokenArg = None,
    ) -> Tuple[bool, str]:
        """Records a decision checkpoint; its weights load (adapters) or convert (merged) at export."""
        from core.export.decision import DecisionExportError, check_decision_eligibility

        if not _export_runtime_available():
            return False, _export_runtime_message()
        try:
            check_decision_eligibility(checkpoint_path)
        except DecisionExportError as exc:
            return False, str(exc)
        self.decision = {"layout": layout, "adapter_only": adapter_only}
        # An adapter folder loads its base at export time, under the credential of this load.
        self._decision_token = token
        self.is_vision = False
        self.is_peft = adapter_only
        self.current_checkpoint = checkpoint_path
        name = "Clef" if layout == "clef" else "Laya"
        kind = "LoRA adapters" if adapter_only else "merged"
        logger.info(f"Decision checkpoint ({name}, {kind}) ready for GGUF export")
        return True, f"Loaded {name} decision model ({kind}); it exports to GGUF only"

    def _export_decision_gguf(
        self, quantization_method, push_to_hub: bool, imatrix_file, npu_q4nx: bool
    ) -> Tuple[bool, str, Optional[str]]:
        if push_to_hub:
            return (
                False,
                "Decision model GGUF export saves to the run folder only; Hub upload is not supported.",
                None,
            )
        if imatrix_file or npu_q4nx:
            return (
                False,
                "Decision model GGUF export does not support imatrix or Q4NX conversion.",
                None,
            )
        from core.export.decision import DecisionExportError, run_decision_gguf_export

        try:
            data = run_decision_gguf_export(
                self.current_checkpoint,
                quantization_method,
                local_files_only = _hf_offline(),
                print_output = True,
                token = getattr(self, "_decision_token", None),
            )
        except (DecisionExportError, ValueError, RuntimeError) as exc:
            logger.error(f"Decision GGUF export failed: {exc}")
            return False, str(exc), None
        output_dir = str(Path(self.current_checkpoint).resolve() / "gguf")
        quants = ", ".join((data or {}).get("quantizations") or [])
        return True, f"Decision model exported to GGUF ({quants}) in {output_dir}", output_dir

    def _decision_only_gguf(self) -> Tuple[bool, str, Optional[str]]:
        return False, "Decision models export to GGUF only.", None

    def _write_export_metadata(self, save_directory: str):
        """Write export_metadata.json with base model info for Chat page discovery."""
        try:
            base_model = (
                get_base_model_from_lora(self.current_checkpoint)
                if self.current_checkpoint
                else None
            )
            source = self.current_checkpoint
            metadata = {
                "base_model": base_model,
                "source_checkpoint": str(Path(source).resolve())
                if source and Path(source).exists()
                else None,
            }
            metadata_path = os.path.join(save_directory, "export_metadata.json")
            with open(metadata_path, "w", encoding = "utf-8") as f:
                json.dump(metadata, f, indent = 2)
            logger.info(f"Wrote export metadata to {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not write export metadata: {e}")

    def export_merged_model(
        self,
        save_directory: str,
        format_type: str = "16-bit (FP16)",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        compressed_method: Optional[str] = None,
        install_missing_dependencies: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export a merged model (a no-op merge for non-PEFT base models).

        ``format_type`` is "16-bit (FP16)", "4-bit (FP4)", or a compressed-tensors label.
        ``compressed_method`` is an optional compressed-tensors scheme alias (fp8, fp8_static, w8a8,
        w4a16, mxfp4, mxfp8, nvfp4); it overrides ``format_type`` and is resolved against
        unsloth.save COMPRESSED_EXPORT_SCHEMES.
        """
        if self.decision is not None:
            return self._decision_only_gguf()
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        # save_pretrained_merged is a no-op merge for non-PEFT base models, so one path covers both.

        output_path: Optional[str] = None
        save_dir_was_empty = False
        # Two backends: compressed-tensors (llm-compressor, NVIDIA-only) and portable torchao FP8/INT8.
        # The alias comes from compressed_method (the "all formats" dropdown) or the format_type label.
        _LABEL_TO_ALIAS = {
            "FP8 (compressed-tensors)": "fp8",
            "NVFP4 (compressed-tensors)": "nvfp4",
        }
        compressed_alias = compressed_method or _LABEL_TO_ALIAS.get(format_type)

        # Fail fast: a stubbed torchao otherwise crashes in transformers with TorchAoConfig(quant_type=None).
        if _is_torchao_alias(compressed_alias) and _torchao_runtime_unavailable():
            return (
                False,
                "Portable torchao FP8/INT8 export needs torchao, which could not be loaded "
                "on this Windows ROCm build. Update Unsloth to install it, or use "
                "16-bit merged or GGUF quantization instead.",
                None,
            )

        compressed_suffix: Optional[str] = None
        # Classify the alias: torchao-portable vs compressed-tensors.
        torchao_info = None
        if compressed_alias and _torchao_export_supported():
            try:
                import unsloth.save as _us_t
                torchao_info = _us_t._normalize_torchao_method(compressed_alias)
            except Exception:
                torchao_info = None
        is_torchao = torchao_info is not None
        is_compressed = compressed_alias is not None and not is_torchao
        try:
            if _IS_MLX and (is_compressed or is_torchao):
                return (
                    False,
                    "Quantized (FP8/FP4/INT) export is not supported on macOS/MLX. "
                    "Use 16-bit or GGUF.",
                    None,
                )

            if is_torchao:
                # Portable torchao: no NVIDIA GPU, no calibration.
                compressed_suffix = torchao_info[1]

            if is_compressed:
                # compressed-tensors needs CUDA; enforce in the backend even if the UI gate is bypassed.
                if not _has_nvidia_gpu():
                    return (
                        False,
                        "Compressed-tensors (FP8/FP4) export requires an NVIDIA GPU. On other "
                        "hardware use the portable FP8/INT8 (torchao) formats or 16-bit.",
                        None,
                    )
                if not _compressed_export_supported():
                    return (
                        False,
                        "Compressed-tensors (FP8/FP4) export requires an Unsloth build with "
                        "compressed-tensors support. Upgrade unsloth, or choose 16-bit.",
                        None,
                    )
                import unsloth.save as _us

                # Prefer the llm-compressor-main shadow (transformers 5.x): the shipped 0.10.x cannot quantize newer
                # models.
                _shadow_pp = None
                _shadow_offered = False
                try:
                    from utils.transformers_version import (
                        _env_offline,
                        _llmcompressor_main_disabled,
                        llmcompressor_shadow_pythonpath,
                    )

                    # Same rule as the consent probe: the dialog names the shadow whenever it can be provisioned.
                    _shadow_offered = not _llmcompressor_main_disabled() and not _env_offline()
                    _shadow_pp = llmcompressor_shadow_pythonpath(
                        allow_provision = install_missing_dependencies,
                    )
                except Exception as e:
                    logger.warning(f"llm-compressor-main shadow unavailable: {e}")
                if _shadow_pp:
                    os.environ[_us._COMPRESSED_QUANTIZE_PYTHONPATH_ENV] = _shadow_pp
                else:
                    # Consent for the shadow does not cover installing into this interpreter instead.
                    if _shadow_offered:
                        install_missing_dependencies = False
                    # The workspace 0.10.x cannot exceed its transformers ceiling, so fail fast for sidecar models.
                    os.environ.pop(_us._COMPRESSED_QUANTIZE_PYTHONPATH_ENV, None)
                    _exceeds, _tf_ver = _us._transformers_exceeds_llm_compressor_ceiling()
                    if _exceeds:
                        return (
                            False,
                            "FP8/FP4 compressed-tensors export is not available for this model: it "
                            f"runs under transformers {_tf_ver}, but the installed llm-compressor "
                            f"supports transformers <= {_us._LLM_COMPRESSOR_MAX_TRANSFORMERS} and the "
                            "llm-compressor-main runtime is not set up (install not approved, "
                            "offline, UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN, or provisioning failed). "
                            "Approve the install, or export to GGUF or 16-bit instead.",
                            None,
                        )

                try:
                    info = _us._normalize_compressed_method(compressed_alias)
                except Exception as e:
                    return False, f"Unsupported compressed export '{compressed_alias}': {e}", None
                if info is None:
                    return (
                        False,
                        f"'{compressed_alias}' is not a recognized compressed-tensors export.",
                        None,
                    )
                compressed_suffix = info[2]

            if _IS_MLX:
                mlx_save_method = "merged_4bit" if format_type == "4-bit (FP4)" else "merged_16bit"
            elif is_compressed or is_torchao:
                save_method = compressed_alias
            elif format_type == "4-bit (FP4)":
                save_method = "merged_4bit_forced"
            else:
                save_method = "merged_16bit"

            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving merged model locally to: {save_directory}")
                # Leftovers in a reused folder would be uploaded too, so only a fresh one is
                # pushed as is.
                save_dir_was_empty = _dir_is_fresh(save_directory)
                ensure_dir(Path(save_directory))

                # No push, but the merge resolves the base repo and save.py turns None into
                # get_token(), so the credential still has to be spelled out.
                merged_token_kw = (
                    {"token": hf_token}
                    if (hf_token or is_anonymous(hf_token))
                    and _supports_kwarg(self.current_model.save_pretrained_merged, "token")
                    else {}
                )
                # Always explicit: the library defaults to auto-installing, so an unconsented export must say False.
                consent_kw = (
                    {"install_missing_dependencies": bool(install_missing_dependencies)}
                    if _supports_kwarg(
                        self.current_model.save_pretrained_merged, "install_missing_dependencies"
                    )
                    else {}
                )
                if _IS_MLX:
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                        save_method = mlx_save_method,
                        **merged_token_kw,
                    )
                else:
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                        save_method = save_method,
                        **consent_kw,
                        **merged_token_kw,
                    )

                # Compressed / torchao writes to the "<dir>-<suffix>" sibling; report that as output.
                final_dir = (
                    f"{save_directory}-{compressed_suffix}"
                    if (is_compressed or is_torchao)
                    else save_directory
                )
                self._write_export_metadata(final_dir)
                logger.info(f"Model saved successfully to {final_dir}")
                output_path = str(Path(final_dir).resolve())

            if push_to_hub:
                if not repo_id or not hf_token:
                    return (
                        False,
                        "Repository ID and Hugging Face token required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing merged model to Hub: {repo_id}")

                if _IS_MLX:
                    repo_id = _push_mlx_merged(
                        self.current_model,
                        self.current_tokenizer,
                        save_method = mlx_save_method,
                        output_path = output_path,
                        output_is_fresh = save_dir_was_empty,
                        repo_id = repo_id,
                        hf_token = hf_token,
                        private = private,
                    )
                else:
                    uploaded = False
                    if output_path and Path(output_path).is_dir():
                        # Upload the artifact already built in output_path; push_to_hub_merged(save_method=...) would
                        # redo the expensive merge and quantization.
                        with contextlib.ExitStack() as stack:
                            upload_dir = output_path
                            if not (is_compressed or is_torchao or save_dir_was_empty):
                                # A reused folder can hold leftovers, so upload a clean second save
                                # instead.
                                upload_dir = stack.enter_context(
                                    _staging_dir(Path(output_path).parent)
                                )
                                self.current_model.save_pretrained_merged(
                                    upload_dir,
                                    self.current_tokenizer,
                                    save_method = save_method,
                                    **merged_token_kw,
                                )
                            # Whatever was built, only weights are worth a repo; without them the
                            # merging push below runs instead, as it did before this was uploaded.
                            if _holds_checkpoint_weights(upload_dir):
                                hf_api = HfApi(token = hf_token)
                                repo_url = hf_api.create_repo(
                                    repo_id, private = private, exist_ok = True
                                )
                                repo_id = getattr(repo_url, "repo_id", repo_id)
                                if private:
                                    _ensure_hub_repo_private(hf_api, repo_id)
                                hf_api.upload_folder(
                                    folder_path = upload_dir,
                                    repo_id = repo_id,
                                    repo_type = "model",
                                    ignore_patterns = _HUB_UPLOAD_IGNORE,
                                )
                                uploaded = True
                    if uploaded:
                        # Last and best-effort like the GGUF card; an existing card is kept, as
                        # push_to_hub_merged does.
                        try:
                            if not hf_api.file_exists(repo_id, "README.md", repo_type = "model"):
                                base_model = getattr(
                                    self.current_model.config, "_name_or_path", "unknown"
                                )
                                content = MODEL_CARD.format(
                                    username = repo_id.split("/")[0],
                                    base_model = repo_id if os.path.isdir(base_model) else base_model,
                                    model_type = getattr(
                                        self.current_model.config, "model_type", "llm"
                                    ),
                                    method = compressed_alias or format_type,
                                    extra = "unsloth",
                                )
                                ModelCard(content).push_to_hub(
                                    repo_id, token = hf_token, commit_message = "Unsloth Model Card"
                                )
                        except Exception as exception:
                            logger.warning(f"Could not publish the model card: {exception}")
                    else:
                        self.current_model.push_to_hub_merged(
                            repo_id,
                            self.current_tokenizer,
                            save_method = save_method,
                            token = hf_token,
                            private = private,
                            **(
                                {"install_missing_dependencies": bool(install_missing_dependencies)}
                                if _supports_kwarg(
                                    self.current_model.push_to_hub_merged,
                                    "install_missing_dependencies",
                                )
                                else {}
                            ),
                        )
                logger.info(f"Model pushed successfully to {repo_id}")

            return True, "Model exported successfully", output_path

        except Exception as e:
            logger.error(f"Error exporting merged model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Export failed: {str(e)}", None

    def export_base_model(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        base_model_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        if self.decision is not None:
            return self._decision_only_gguf()
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        if self.is_peft:
            return (
                False,
                "This is a PEFT model. Use 'Merged Model' export type instead.",
                None,
            )

        output_path: Optional[str] = None
        save_dir_was_empty = False
        try:
            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving base model locally to: {save_directory}")
                save_dir_was_empty = _dir_is_fresh(save_directory)
                ensure_dir(Path(save_directory))

                if _IS_MLX:
                    # fuse() is a no-op without LoRA layers, so this handles non-LoRA models too.
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                        save_method = "merged_16bit",
                    )
                else:
                    self.current_model.save_pretrained(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)

                self._write_export_metadata(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            if push_to_hub:
                if not repo_id or not hf_token:
                    return (
                        False,
                        "Repository ID and Hugging Face token required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing base model to Hub: {repo_id}")

                if _IS_MLX:
                    _push_mlx_merged(
                        self.current_model,
                        self.current_tokenizer,
                        save_method = "merged_16bit",
                        output_path = output_path,
                        output_is_fresh = save_dir_was_empty,
                        repo_id = repo_id,
                        hf_token = hf_token,
                        private = private,
                    )
                else:
                    base_model = (
                        base_model_id or self.current_model.config._name_or_path or "unknown"
                    )

                    with contextlib.ExitStack() as stack:
                        upload_dir = save_directory
                        if save_directory and not save_dir_was_empty:
                            # A reused folder can hold leftovers, so upload a clean second save
                            # instead.
                            upload_dir = stack.enter_context(
                                _staging_dir(Path(save_directory).parent)
                            )
                            self.current_model.save_pretrained(upload_dir)
                            self.current_tokenizer.save_pretrained(upload_dir)
                        hf_api = HfApi(token = hf_token)
                        repo_url = hf_api.create_repo(repo_id, private = private, exist_ok = True)
                        repo_id = getattr(repo_url, "repo_id", repo_id)
                        if private:
                            _ensure_hub_repo_private(hf_api, repo_id)
                        username = repo_id.split("/")[0]

                        content = MODEL_CARD.format(
                            username = username,
                            base_model = base_model,
                            model_type = self.current_model.config.model_type,
                            method = "",
                            extra = "unsloth",
                        )
                        card = ModelCard(content)
                        card.push_to_hub(
                            repo_id, token = hf_token, commit_message = "Unsloth Model Card"
                        )

                        if save_directory:
                            hf_api.upload_folder(
                                folder_path = upload_dir,
                                repo_id = repo_id,
                                repo_type = "model",
                                ignore_patterns = _HUB_UPLOAD_IGNORE,
                            )
                            logger.info(f"Model pushed successfully to {repo_id}")
                        else:
                            return (
                                False,
                                "Local save directory required for Hub upload",
                                None,
                            )

            return True, "Model exported successfully", output_path

        except Exception as e:
            logger.error(f"Error exporting base model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Export failed: {str(e)}", None

    def export_gguf(
        self,
        save_directory: str,
        quantization_method = "Q4_K_M",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        imatrix_file = None,
        private: bool = False,
        npu_q4nx: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export the model in GGUF format.

        ``quantization_method`` is a single GGUF quant method ("Q4_K_M") or a list of them; a list
        produces one GGUF per quant from a single model load, since unsloth save_to_gguf loops
        internally. ``imatrix_file`` is an importance matrix path or boolean. ``npu_q4nx`` also
        converts one Q4_0 / Q4_1 / Q4_K_M GGUF to FastFlowLM's Q4NX for the AMD Ryzen AI NPU.
        """
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if self.decision is not None:
            return self._export_decision_gguf(
                quantization_method, push_to_hub, imatrix_file, npu_q4nx
            )
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        # Older unsloth builds raise on an unexpected imatrix_file kwarg even for a plain no-imatrix export.
        if imatrix_file and not _imatrix_export_supported(self.current_model.save_pretrained_gguf):
            return (
                False,
                "This Unsloth build does not support GGUF imatrix export. "
                "Upgrade unsloth and unsloth_zoo, or disable the imatrix option.",
                None,
            )
        # Truthiness, as above: a disabled imatrix must not reach an exporter without the kwarg.
        imatrix_kw = {"imatrix_file": imatrix_file} if imatrix_file else {}
        # Resolution reads a Hub repo, so the local save needs the token; kept out of imatrix_kw, which the
        # push shares and names token= itself.
        local_token_kw = (
            {"token": hf_token}
            if imatrix_file
            and (hf_token or is_anonymous(hf_token))
            and _supports_kwarg(self.current_model.save_pretrained_gguf, "token")
            else {}
        )

        output_path: Optional[str] = None
        exported_ggufs: List[str] = []
        exported_modelfile = False
        exported_modelfile_bytes: Optional[bytes] = None
        exported_is_vlm = False
        exported_config: Optional[bytes] = None
        try:
            # Normalize to a lowercased list so multiple quants come from one model load.
            if isinstance(quantization_method, (list, tuple)):
                quant_methods = [str(q).lower() for q in quantization_method if str(q).strip()]
            else:
                quant_methods = [str(quantization_method).lower()]
            if not quant_methods:
                quant_methods = ["q4_k_m"]
            quant_method = quant_methods if len(quant_methods) > 1 else quant_methods[0]
            if npu_q4nx and not save_directory:
                return False, "The AMD NPU (Q4NX) export needs a local save directory.", None
            if npu_q4nx and not any(q in q4nx.SOURCE_QUANTS for q in quant_methods):
                return (
                    False,
                    "The AMD NPU (Q4NX) export needs a Q4_0, Q4_1 or Q4_K_M GGUF in the selection.",
                    None,
                )

            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                # Keep unsloth relative-path internals anchored to the repo cwd.
                abs_save_dir = os.path.abspath(save_directory)
                logger.info(f"Saving GGUF model locally to: {abs_save_dir}")

                ensure_dir(Path(abs_save_dir))

                # On WSL, patch out sudo check before llama.cpp build
                _apply_wsl_sudo_patch()

                # Keep all intermediates under an export-owned root.
                model_tmp_root = tempfile.mkdtemp(prefix = "_tmp_model_", dir = abs_save_dir)
                model_tmp_path = Path(model_tmp_root)
                _model_tmp = os.path.join(model_tmp_root, "model")
                # Resolve before anything can raise; the cleanup below needs it too.
                imatrix_path = _materialized_imatrix_path(_model_tmp, imatrix_file)
                try:
                    # Pinned to setup.sh's llama.cpp ref so the converter cannot drift past the pinned
                    # llama-quantize gguf API; scoped to the conversion.
                    with _llama_cpp_scripts_pin():
                        result = self.current_model.save_pretrained_gguf(
                            _model_tmp,
                            self.current_tokenizer,
                            quantization_method = quant_method,
                            **imatrix_kw,
                            **local_token_kw,
                        )

                    # Scan only the owned root; exact reported paths cover external outputs.
                    reported = result if isinstance(result, dict) else {}
                    produced = {p for p in model_tmp_path.rglob("*.gguf") if p.is_file()}
                    produced.update(Path(f) for f in _reported_gguf_files(result) or [])
                    produced = {p for p in produced if not _is_imatrix(p, imatrix_path)}
                    modelfiles = {p for p in model_tmp_path.rglob("Modelfile") if p.is_file()}
                    reported_modelfile = reported.get("modelfile_location")
                    if reported_modelfile and Path(reported_modelfile).is_file():
                        modelfiles.add(Path(os.path.abspath(os.fspath(reported_modelfile))))

                    relocated_ggufs = []
                    for src in sorted(produced):
                        if src.is_symlink():
                            raise RuntimeError(
                                f"GGUF conversion produced a symlink, refusing to relocate it: {src}"
                            )
                        dest = os.path.join(abs_save_dir, src.name)
                        if os.path.isdir(dest):
                            # move would nest the file, and the allow-list would match neither.
                            raise RuntimeError(
                                f"Cannot relocate {src.name}: a directory of that name is "
                                f"already in {abs_save_dir}"
                            )
                        shutil.move(str(src), dest)
                        relocated_ggufs.append(dest)
                        logger.info(f"Relocated GGUF: {src.name} → {abs_save_dir}/")
                    if not relocated_ggufs:
                        raise RuntimeError(
                            "GGUF conversion produced no files: no .gguf outputs for "
                            f"{abs_save_dir}"
                        )
                    exported_ggufs = [str(f) for f in drop_appledouble_metadata(relocated_ggufs)]
                    if not exported_ggufs:
                        # final_ggufs below would pass on an earlier export's .gguf left here.
                        raise RuntimeError(
                            "GGUF conversion produced only AppleDouble metadata companions "
                            f"and no usable .gguf file for {abs_save_dir}"
                        )
                    exported_is_vlm = bool(
                        reported.get("is_vlm", getattr(self, "is_vision", False))
                    )
                    # In memory: the temp root goes, and a config.json here reads as a checkpoint.
                    merged_config = (
                        Path(reported.get("save_directory") or _model_tmp) / "config.json"
                    )
                    if merged_config.is_file():
                        exported_config = merged_config.read_bytes()

                    if modelfiles:
                        modelfile = sorted(modelfiles)[0]
                        if modelfile.is_symlink():
                            raise RuntimeError(
                                "GGUF conversion produced a symlinked Modelfile, "
                                f"refusing to relocate it: {modelfile}"
                            )
                        # Optional: a blocked destination publishes from memory, never fails.
                        modelfile_dest = os.path.join(abs_save_dir, "Modelfile")
                        try:
                            if os.path.isdir(modelfile_dest):
                                raise OSError(
                                    f"a directory named Modelfile is already in {abs_save_dir}"
                                )
                            shutil.move(str(modelfile), modelfile_dest)
                            exported_modelfile = True
                            logger.info(f"Relocated Modelfile → {abs_save_dir}/")
                        except OSError as exception:
                            logger.warning(f"Could not relocate the Modelfile: {exception}")
                            try:
                                exported_modelfile_bytes = modelfile.read_bytes()
                            except OSError as read_exception:
                                logger.warning(
                                    f"Could not read the Modelfile to upload it: {read_exception}"
                                )
                finally:
                    # The imatrix is an input, so counting it would retain the merged checkpoint on every such export.
                    unrelocated = []
                    if model_tmp_path.is_dir():
                        unrelocated = sorted(
                            str(p)
                            for p in model_tmp_path.rglob("*.gguf")
                            if not _is_imatrix(p, imatrix_path)
                        )
                    if unrelocated:
                        logger.error(
                            "Kept GGUF files that could not be relocated: %s",
                            ", ".join(unrelocated),
                        )
                    else:
                        shutil.rmtree(model_tmp_root, ignore_errors = True)

                # iterdir, not glob.glob: glob hides dot-leading names, so an empty model stem's ".Q4_K_M.gguf"
                # read as "(none)". This list is the success gate.
                final_ggufs = sorted(
                    str(p)
                    for p in drop_appledouble_metadata(list(Path(abs_save_dir).iterdir()))
                    if p.is_file() and p.name.lower().endswith(".gguf")
                )
                logger.info(
                    "GGUF export complete. Final files in %s:\n  %s",
                    abs_save_dir,
                    "\n  ".join(os.path.basename(f) for f in final_ggufs) or "(none)",
                )
                if not final_ggufs:
                    # Reporting success over an empty directory is what hid #7897.
                    return (
                        False,
                        f"GGUF conversion reported success but wrote no .gguf file to "
                        f"{abs_save_dir}. Check the export log for the path the "
                        f"converter actually used, then upgrade with "
                        f"`pip install --upgrade unsloth unsloth_zoo` and retry.",
                        None,
                    )

                # Only write metadata once an artifact is actually present.
                self._write_export_metadata(abs_save_dir)
                output_path = str(Path(abs_save_dir).resolve())

                if npu_q4nx:
                    source = q4nx.source_gguf(exported_ggufs, quant_methods)
                    try:
                        if source is None:
                            raise RuntimeError("no Q4_0, Q4_1 or Q4_K_M GGUF was written")
                        with q4nx.staged_output(Path(abs_save_dir) / "npu-q4nx") as staging:
                            q4nx.convert_gguf_to_q4nx(source, staging)
                            self._write_q4nx_companions(staging, exported_config)
                    except Exception as exception:
                        logger.error(f"Q4NX conversion failed: {exception}")
                        return (
                            False,
                            f"GGUF files were saved to {output_path}, but the AMD NPU (Q4NX) "
                            f"conversion failed: {exception}",
                            output_path,
                        )

            if push_to_hub:
                if not repo_id or not hf_token:
                    return (
                        False,
                        "Repository ID and Hugging Face token required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing GGUF model to Hub: {repo_id}")

                if output_path and Path(output_path).is_dir():
                    # These are already built; push_to_hub_gguf would convert the model again.
                    hf_api = HfApi(token = hf_token)
                    repo_url = hf_api.create_repo(repo_id, private = private, exist_ok = True)
                    repo_id = getattr(repo_url, "repo_id", repo_id)
                    if private:
                        _ensure_hub_repo_private(hf_api, repo_id)
                    # Allow-list, not the folder; glob.escape keeps "model[v2].gguf" a literal.
                    hf_api.upload_folder(
                        folder_path = output_path,
                        repo_id = repo_id,
                        repo_type = "model",
                        allow_patterns = [
                            *(glob.escape(os.path.basename(f)) for f in exported_ggufs),
                            *(["Modelfile"] if exported_modelfile else []),
                        ],
                    )
                    if exported_config is not None:
                        hf_api.upload_file(
                            path_or_fileobj = exported_config,
                            path_in_repo = "config.json",
                            repo_id = repo_id,
                            repo_type = "model",
                            commit_message = "Unsloth config.json",
                        )
                    if exported_modelfile_bytes is not None:
                        hf_api.upload_file(
                            path_or_fileobj = exported_modelfile_bytes,
                            path_in_repo = "Modelfile",
                            repo_id = repo_id,
                            repo_type = "model",
                            commit_message = "Unsloth Ollama Modelfile",
                        )
                    # Last (advertises the files), best-effort: RepoCard hardcodes huggingface.co.
                    try:
                        ModelCard(
                            GGUF_MODEL_CARD.format(
                                name = repo_id.split("/")[-1],
                                repo_id = repo_id,
                                vlm_tag = "\n- vision-language-model" if exported_is_vlm else "",
                                files = "\n".join(
                                    f"- `{os.path.basename(f)}`" for f in exported_ggufs
                                ),
                            )
                        ).push_to_hub(repo_id, token = hf_token, commit_message = "Unsloth Model Card")
                    except Exception as exception:
                        logger.warning(f"Could not publish the model card: {exception}")
                else:
                    # Converts as well as pushes, so it needs the same scoped pin.
                    with _llama_cpp_scripts_pin():
                        self.current_model.push_to_hub_gguf(
                            repo_id,
                            self.current_tokenizer,
                            quantization_method = quant_method,
                            token = hf_token,
                            private = private,
                            **imatrix_kw,
                        )
                logger.info(f"GGUF model pushed successfully to {repo_id}")

            return (
                True,
                f"GGUF model exported successfully ({', '.join(quant_methods)})",
                output_path,
            )

        except Exception as e:
            logger.error(f"Error exporting GGUF model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            if output_path:
                # Only the Hub leg can raise once output_path is set, so the files are on disk.
                return (
                    False,
                    f"GGUF files were saved to {output_path}, but the Hub upload failed: {e}",
                    output_path,
                )
            return False, f"GGUF export failed: {str(e)}", None

    def _write_q4nx_companions(self, q4nx_dir: Path, config: Optional[bytes]) -> None:
        """The tokenizer files FastFlowLM loads next to model.q4nx (see q4nx.CONFIG_FILES)."""
        with tempfile.TemporaryDirectory(prefix = "_tmp_tokenizer_", dir = q4nx_dir) as scratch:
            self.current_tokenizer.save_pretrained(scratch)
            for name in q4nx.TOKENIZER_FILES:
                # The converter rebuilds tokenizer.json from the GGUF; the HF one is what FLM ships.
                if (Path(scratch) / name).is_file():
                    shutil.copyfile(Path(scratch) / name, q4nx_dir / name)
        # generation_config holds stop ids config.json lacks (Phi-4-mini's <|end|>, 200020).
        generation = getattr(self.current_model, "generation_config", None)
        q4nx.write_flm_tokenizer_config(
            q4nx_dir,
            json.loads(config) if config else None,
            {"eos_token_id": getattr(generation, "eos_token_id", None)},
        )

    def _save_mlx_adapter(
        self,
        destination: str,
        resolved_format: str,
        gguf_outtype: Optional[str] = None,
        hf_token: HfTokenArg = None,
    ) -> None:
        """Save the loaded MLX adapter as mlx or peft, plus a GGUF LoRA when gguf_outtype is set."""
        if resolved_format == "mlx":
            self.current_model.save_lora_adapters(destination)
            self.current_tokenizer.save_pretrained(destination)
            return
        saver = getattr(self.current_model, "save_lora_adapters", None)
        if not _supports_kwarg(saver, "adapter_format"):
            raise RuntimeError(_ZOO_UPGRADE_MESSAGE)
        # The zoo converter refuses an existing path; stage fresh so a repeat export overwrites.
        parent = Path(destination).parent
        ensure_dir(parent)
        with tempfile.TemporaryDirectory(prefix = _STAGING_PREFIX, dir = parent) as tmp_dir:
            staged = os.path.join(tmp_dir, "adapter")
            saver(staged, adapter_format = "peft")
            self.current_tokenizer.save_pretrained(staged)
            if gguf_outtype:
                self._convert_peft_dir_to_gguf(staged, gguf_outtype, hf_token)
            ensure_dir(Path(destination))
            for name in os.listdir(staged):
                src, dst = os.path.join(staged, name), os.path.join(destination, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok = True)
                else:
                    os.replace(src, dst)

    def _convert_peft_dir_to_gguf(
        self, save_directory: str, outtype: str, hf_token: HfTokenArg
    ) -> None:
        """Convert a PEFT adapter with llama.cpp's convert_lora_to_gguf.py, as the CUDA path does."""
        import importlib.util
        import subprocess
        import sys as _sys

        with open(os.path.join(save_directory, "adapter_config.json"), "r", encoding = "utf-8") as f:
            peft_cfg = json.load(f)
        rejects = []
        if peft_cfg.get("alpha_pattern"):
            rejects.append("per-module alpha values (GGUF LoRA stores one global alpha)")
        if peft_cfg.get("use_rslora") and peft_cfg.get("rank_pattern"):
            rejects.append("rsLoRA with per-module ranks (bakes to per-module alphas)")
        if peft_cfg.get("use_dora"):
            rejects.append("DoRA magnitudes (no GGUF LoRA representation)")
        if peft_cfg.get("modules_to_save") or getattr(
            self.current_model, "_unsloth_full_state_modules", None
        ):
            rejects.append("full-module state (modules_to_save or replaced embeddings)")
        if peft_cfg.get("target_parameters"):
            rejects.append("expert-parameter adapters (llama.cpp has no expert handling)")
        if rejects:
            raise RuntimeError(
                "This adapter cannot be exported as a GGUF LoRA: "
                + "; ".join(rejects)
                + ". Export the PEFT safetensors adapter instead."
            )

        if importlib.util.find_spec("torch") is None:
            raise RuntimeError(
                "GGUF adapter export needs the 'torch' Python package for "
                "llama.cpp's converter; install it and retry."
            )

        from unsloth_zoo import llama_cpp as _zoo_llama_cpp

        default_dir = os.path.normpath(_zoo_llama_cpp.LLAMA_CPP_DEFAULT_DIR)
        source_dir = os.path.join(os.path.dirname(default_dir), "llama.cpp-source")
        # A user-set scripts dir is authoritative and is checked before any network revision lookup.
        pinned_dir = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", "").strip()
        if pinned_dir:
            converter = os.path.join(os.path.expanduser(pinned_dir), "convert_lora_to_gguf.py")
            if not os.path.exists(converter):
                raise RuntimeError(
                    f"UNSLOTH_LLAMA_CPP_SCRIPTS_DIR={pinned_dir} has no convert_lora_to_gguf.py; point it "
                    "at a full llama.cpp checkout or unset it."
                )
        elif os.path.exists(os.path.join(default_dir, "convert_lora_to_gguf.py")):
            converter = os.path.join(default_dir, "convert_lora_to_gguf.py")
        else:
            # Pinned to the installed binaries' revision (else the latest release).
            try:
                _repo, tag = _zoo_llama_cpp._resolve_converter_revision(default_dir)
            except Exception:
                tag = None
            tag = tag.split("-mix-")[0] if tag else None
            if tag:
                source_dir = f"{source_dir}-{tag}"
            converter = os.path.join(source_dir, "convert_lora_to_gguf.py")
            if not os.path.exists(converter):
                converter = None
        if converter is None:
            if not getattr(_zoo_llama_cpp, "_converter_network_allowed", lambda: True)():
                raise RuntimeError(
                    "GGUF adapter export needs llama.cpp's convert_lora_to_gguf.py, which is not "
                    f"installed, and offline mode forbids cloning it; clone llama.cpp into {source_dir}."
                )
            if not getattr(_zoo_llama_cpp, "_auto_install_enabled", lambda: True)():
                raise RuntimeError(
                    "GGUF adapter export needs a llama.cpp source checkout and automatic "
                    f"installation was declined (UNSLOTH_AUTO_INSTALL=0); clone llama.cpp into {source_dir}."
                )
            # Not install_llama_cpp: it probes apt-get even when only cloning, which fails on macOS.
            ensure_dir(Path(source_dir).parent)
            with tempfile.TemporaryDirectory(dir = Path(source_dir).parent) as tmp_dir:
                clone = os.path.join(tmp_dir, "llama.cpp")
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        *(["--branch", tag] if tag else []),
                        "https://github.com/ggml-org/llama.cpp",
                        clone,
                    ],
                    check = True,
                    capture_output = True,
                    text = True,
                    encoding = "utf-8",
                    errors = "replace",
                )
                if not os.path.exists(source_dir):
                    os.replace(clone, source_dir)
            converter = os.path.join(source_dir, "convert_lora_to_gguf.py")
            if not os.path.exists(converter):
                raise RuntimeError(
                    f"convert_lora_to_gguf.py is missing from the llama.cpp clone at {source_dir}."
                )
        if importlib.util.find_spec("gguf") is None and not os.path.isdir(
            os.path.join(os.path.dirname(converter), "gguf-py")
        ):
            raise RuntimeError(
                "GGUF adapter export needs the 'gguf' Python package (or a "
                "full llama.cpp checkout with gguf-py); install it and retry."
            )

        base_model_id = peft_cfg.get("base_model_name_or_path") or getattr(
            self.current_model, "_hf_repo", None
        )
        if not base_model_id:
            raise RuntimeError(
                "Could not determine the adapter's base model for GGUF "
                "conversion (no base_model_name_or_path)."
            )
        model_name = str(base_model_id).replace("\\", "/").rstrip("/").split("/")[-1] or "model"
        out_gguf = os.path.join(save_directory, f"{model_name}-lora-{outtype}.gguf")
        cmd = [
            _sys.executable,
            converter,
            save_directory,
            "--outfile",
            out_gguf,
            "--outtype",
            outtype,
        ]
        # The snapshot the model was loaded from carries the adapter's pinned base revision.
        loaded_base = next(
            (
                str(p)
                for p in (
                    getattr(self.current_model, "_config_src_path", None),
                    getattr(self.current_model, "_src_path", None),
                    base_model_id,
                )
                if isinstance(p, (str, os.PathLike))
                and os.path.isfile(os.path.join(str(p), "config.json"))
            ),
            None,
        )
        if loaded_base:
            cmd += ["--base", loaded_base]
        else:
            cmd += ["--base-model-id", str(base_model_id)]
        if getattr(self, "trust_remote_code", False):
            cmd.append("--trust-remote-code")
        env = os.environ.copy()
        apply_token_to_child_env(env, normalize_token(hf_token))
        logger.info(f"Converting adapter at '{save_directory}' to GGUF -> '{out_gguf}'")
        result = subprocess.run(
            cmd,
            env = env,
            capture_output = True,
            text = True,
            encoding = "utf-8",
            errors = "replace",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"LoRA -> GGUF conversion failed (exit {result.returncode}): "
                + (result.stderr or result.stdout or "").strip()[-2000:]
            )

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: HfTokenArg = None,
        private: bool = False,
        gguf: bool = False,
        gguf_outtype: str = "q8_0",
        adapter_format: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """Export the LoRA adapter only, not merged.

        ``gguf`` also converts the adapter to a GGUF LoRA file (llama.cpp convert_lora_to_gguf.py),
        loadable with `llama-cli --lora ...`; ``gguf_outtype`` is its output float type, one of
        q8_0/f16/bf16/f32. ``adapter_format`` is 'mlx' or 'peft' (MLX servers only offer both);
        omitted resolves to the platform's native format.
        """
        if self.decision is not None:
            return self._decision_only_gguf()
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        if not self.is_peft:
            return False, "This is not a PEFT model. No adapter to export.", None

        resolved_format, format_error = _resolve_adapter_format(adapter_format)
        if format_error:
            return False, format_error, None

        _GGUF_LORA_OUTTYPES = ("q8_0", "f16", "bf16", "f32")
        if gguf:
            if adapter_format == "mlx":
                return (
                    False,
                    "GGUF LoRA files are built from a PEFT-format adapter; "
                    "omit adapter_format or use 'peft' with gguf=True.",
                    None,
                )
            resolved_format = "peft"
            # convert_lora_to_gguf.py reads only the standard lora_A/lora_B delta and ignores DoRA's
            # lora_magnitude_vector, so a DoRA export silently drops magnitude rescaling.
            _peft_config = getattr(self.current_model, "peft_config", {}).get("default")
            if getattr(_peft_config, "use_dora", False):
                return (
                    False,
                    "GGUF LoRA export is not supported for DoRA adapters: the GGUF LoRA "
                    "format has no way to represent DoRA's magnitude vectors, so the "
                    "exported file would silently lose the DoRA behavior. Use the "
                    "safetensors adapter instead, or merge to a full GGUF model.",
                    None,
                )
            outtype = str(gguf_outtype).lower()
            if outtype not in _GGUF_LORA_OUTTYPES:
                return (
                    False,
                    f"Invalid GGUF LoRA outtype '{gguf_outtype}'. "
                    f"Choose one of {', '.join(_GGUF_LORA_OUTTYPES)}.",
                    None,
                )
            if _IS_MLX:
                if not _supports_kwarg(
                    getattr(self.current_model, "save_lora_adapters", None), "adapter_format"
                ):
                    return False, _ZOO_UPGRADE_MESSAGE, None
            else:
                # getattr so an older build without save_pretrained_gguf returns a clean message instead of a generic 500.
                _save_gguf_fn = getattr(self.current_model, "save_pretrained_gguf", None)
                if _save_gguf_fn is None or not _supports_kwarg(_save_gguf_fn, "save_method"):
                    return (
                        False,
                        "This Unsloth build does not support GGUF LoRA adapter export. "
                        "Upgrade unsloth and unsloth_zoo, or export the safetensors adapter.",
                        None,
                    )

        def save_lora_gguf(directory):
            # Writes the adapter files plus "<base>-lora-<outtype>.gguf".
            if _IS_MLX:
                self._save_mlx_adapter(directory, "peft", gguf_outtype = outtype, hf_token = hf_token)
                return
            self.current_model.save_pretrained_gguf(
                directory,
                self.current_tokenizer,
                save_method = "lora",
                quantization_method = outtype,
                # A token fetches a gated base's config; False keeps a denied caller
                # off get_token().
                token = normalize_token(hf_token),
            )

        output_path: Optional[str] = None
        save_dir_was_empty = False
        try:
            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving LoRA adapter locally to: {save_directory}")
                save_dir_was_empty = _dir_is_fresh(save_directory)
                # One folder, one format (recursive: peft nests named adapters).
                if _IS_MLX and Path(save_directory).is_dir():
                    other = _other_adapter_weight_names(resolved_format)
                    if any(
                        name in other for _, _, names in os.walk(save_directory) for name in names
                    ):
                        return (
                            False,
                            f"'{save_directory}' already holds the other format's adapter "
                            "weights; refusing to mix MLX- and PEFT-format files in one "
                            "directory. Use a different directory or remove the old adapter first.",
                            None,
                        )
                ensure_dir(Path(save_directory))

                if gguf:
                    _apply_wsl_sudo_patch()
                    save_lora_gguf(save_directory)
                    # iterdir, not glob.glob: glob hides dot-leading names.
                    final_ggufs = sorted(
                        str(p)
                        for p in drop_appledouble_metadata(list(Path(save_directory).iterdir()))
                        if p.is_file() and p.name.lower().endswith(".gguf")
                    )
                    logger.info(
                        "LoRA GGUF export complete. Files in %s:\n  %s",
                        save_directory,
                        "\n  ".join(os.path.basename(f) for f in final_ggufs) or "(none)",
                    )
                elif _IS_MLX:
                    self._save_mlx_adapter(save_directory, resolved_format)
                else:
                    self.current_model.save_pretrained(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)
                self._write_export_metadata(save_directory)
                logger.info(f"Adapter saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            if push_to_hub:
                if not repo_id or not hf_token:
                    return (
                        False,
                        "Repository ID and Hugging Face token required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing LoRA adapter to Hub: {repo_id}")

                # Needs a local save_directory so the conversion is not re-run.
                if gguf and not (output_path and Path(output_path).is_dir()):
                    return (
                        False,
                        "GGUF LoRA Hub upload requires a local save directory; set one and retry.",
                        None,
                    )

                hf_api = HfApi(token = hf_token)

                if gguf:
                    with contextlib.ExitStack() as stack:
                        upload_dir = output_path
                        if not save_dir_was_empty:
                            # A reused folder can hold leftovers, so upload a clean second save
                            # instead. The converter names the GGUF's model after its folder.
                            upload_dir = os.path.join(
                                stack.enter_context(_staging_dir(Path(output_path).parent)),
                                Path(output_path).name,
                            )
                            save_lora_gguf(upload_dir)
                        repo_id = _open_hub_repo(hf_api, repo_id, private)
                        hf_api.upload_folder(
                            folder_path = upload_dir,
                            repo_id = repo_id,
                            repo_type = "model",
                            ignore_patterns = _HUB_UPLOAD_IGNORE,
                        )
                elif _IS_MLX:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        # Serialise first: opening the repo before this would leave an empty
                        # one behind whenever the adapter or tokenizer fails to write.
                        self._save_mlx_adapter(tmp_dir, resolved_format)
                        repo_id = _open_hub_repo(hf_api, repo_id, private)
                        hf_api.upload_folder(
                            folder_path = tmp_dir,
                            repo_id = repo_id,
                            repo_type = "model",
                        )
                else:
                    # Opened here rather than left to push_to_hub: a repo that does not exist
                    # yet is one another client can create public first, and `private` cannot
                    # change an existing repo's visibility, so the adapter would land in it.
                    repo_id = _open_hub_repo(hf_api, repo_id, private)
                    _publish_unsloth_model_card(hf_api, repo_id, self.current_model, hf_token)
                    self.current_model.push_to_hub(repo_id, token = hf_token, private = private)
                    self.current_tokenizer.push_to_hub(repo_id, token = hf_token, private = private)
                logger.info(f"Adapter pushed successfully to {repo_id}")

            return True, "LoRA adapter exported successfully", output_path

        except Exception as e:
            logger.error(f"Error exporting LoRA adapter: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Adapter export failed: {str(e)}", None


_export_backend = None


def get_export_backend() -> ExportBackend:
    global _export_backend
    if _export_backend is None:
        _export_backend = ExportBackend()
    return _export_backend
