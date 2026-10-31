# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export backend - exports models in various formats."""

import glob
import json
import structlog
import tempfile
from loggers import get_logger
import os
import re
import sys
import shutil
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List

# unsloth imports torch, so --no-torch installs raise here; stay importable and return a clean PyTorch is not installed error
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
from utils.models.model_config import (
    detect_audio_type,
    load_mlx_adapter_processor,
    load_mlx_adapter_tokenizer,
)
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


# Export merges one module at a time and trains nothing, so the auto plan keeps 1 GiB for that, not a training step.
_EXPORT_AUTO_OFFLOAD_KWARGS = {
    "offload_layers": "auto",
    "device_map_planner_kwargs": {"reserve_bytes": 1 << 30},
}


def _offload_layers_kwargs(offload_layers, device_map_kw: Optional[dict] = None) -> dict:
    if not offload_layers or offload_layers == "auto":
        return {}
    # A fixed count loads onto one card; core raises on a multi-GPU map.
    if (device_map_kw or {}).get("device_map") in ("unsloth_balanced", "balanced"):
        return {}
    return {"offload_layers": offload_layers}


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
    """Pins convert_hf_to_gguf.py to the setup.sh llama.cpp ref; a user-set UNSLOTH_LLAMA_CPP_SCRIPTS_DIR outranks the pin and skips the strict scan"""
    global _LLAMA_CPP_SCRIPTS_WARNING_EMITTED
    if _IS_MLX:
        # The MLX save path pins itself under a non-reentrant lock; nesting here deadlocks.
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
        # Pinning an entrypoint with no conversion/ beside it stops the staged resolver fetching a co-versioned set that runs
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

    # Older unsloth_zoo has no internal pin, so the variable is scoped by hand; strict mode still exempts there, and upgrading unsloth_zoo is the fix
    existing = os.environ.get("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR")
    if existing is not None:
        yield
        return
    os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = LLAMA_CPP_DEFAULT_DIR
    try:
        yield
    finally:
        os.environ.pop("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", None)


@contextlib.contextmanager
def _converter_remote_code(approved: bool):
    """Keep convert_hf_to_gguf.py to the load's consent-gated trust decision for one conversion.

    The converter loads several architectures' tokenizers with trust_remote_code=True, so a
    checkpoint loaded with built-in classes (never seen by the consent gate) could run its
    auto_map code there. Transformers imports that code only from its HF_MODULES_CACHE copy,
    which the converter child inherits: a path no directory can be created at refuses it.
    Covers the MLX save path too, which has no guard of its own.
    """
    if approved:
        yield
        return
    existing = os.environ.get("HF_MODULES_CACHE")
    os.environ["HF_MODULES_CACHE"] = os.path.join(os.path.abspath(__file__), "no_remote_code")
    try:
        yield
    finally:
        if existing is None:
            os.environ.pop("HF_MODULES_CACHE", None)
        else:
            os.environ["HF_MODULES_CACHE"] = existing


def _multi_gpu_device_map_kwargs() -> dict:
    """Shards only on multi-GPU hosts, since unsloth's sequential device_map stacks the model on GPU0."""
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
    """True for accelerator OOM by exception type or message: accelerate and transformers re-raise plain RuntimeError, ROCm and XPU use own classes"""
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
    match it explicitly. See transformers ``quantizers/quantizer_bnb_4bit.py``. Unsloth
    re-raises it as "does not fit in GPU memory" with the original as ``__cause__``.
    """
    while exc is not None:
        if "dispatched on the cpu or the disk" in str(exc).lower():
            return True
        exc = exc.__cause__
    return False


def _is_device_map_infeasible(exc: BaseException) -> bool:
    """Matched by class name so no version defining it is needed; can fire when other jobs hold the GPUs although the model still fits"""
    return type(exc).__name__ == "DeviceMapInfeasible"


class _CpuSpillRetry(Exception):
    """A multi-GPU load that succeeded but left modules offloaded to CPU/disk."""


def _cpu_offloaded_modules(model) -> int:
    """Full-precision loads accept a CPU/disk map silently, then die on meta tensors; inspect the map."""
    device_map = getattr(model, "hf_device_map", None) or {}
    return sum(1 for target in device_map.values() if str(target) in ("cpu", "disk"))


def _accepts_by_keyword(params, name):
    """Positional-only params do not count: call sites pass by keyword; counting them gives a TypeError."""
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
    """None means use legacy heuristics; an empty result is None too, as it may be a stale manifest."""
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
    """Imatrix copy the owned-root scan must skip; compare full paths, since basenames can hide outputs."""
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
    """Uses os.path.samefile, not ==: case folding or APFS NFD spelling makes byte-exact comparison miss."""
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


def _torchao_too_old():
    """Installed torchao version when it is below what transformers' TorchAoConfig accepts, else None."""
    try:
        from core._torchao_stub import torchao_too_old_for_export
        return torchao_too_old_for_export()
    except Exception:
        return None


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
    """Honors HF offline env vars, else a TCP probe avoids hangs; set UNSLOTH_OFFLINE_PROBE=0 to skip it."""
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
    """WSL: do_we_need_sudo()'s apt-get update hangs on a password prompt; setup.sh pre-installs deps."""
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


# Local bookkeeping whose base model can be a local path; keep it out of the Hub upload, including the GGUF push
_HUB_UPLOAD_IGNORE = ["export_metadata.json", "._*"]
# REAP's report: uploaded with a pruned export, and deleted from a reused repo by any other merged
# or base upload (GGUF files carry one each) (upload_folder never removes remote files on its own; one it uploads in the
# same commit is kept).
_REAP_REPORT = "reap_report.json"

_STAGING_PREFIX = "unsloth-hub-upload-"


def _card_base_model(model) -> str:
    """The base a Hub card names: after REAP the config points at a staging copy, so the id the
    pruned model came from is used instead."""
    source = getattr(model, "_unsloth_reap_source", None)
    if source:
        return source
    return getattr(getattr(model, "config", None), "_name_or_path", "unknown") or "unknown"


def _staging_dir(export_parent):
    """Temp dir and export filesystem are not reliably roomy or writable; only the export dir must be."""
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
    """Call right before the upload: once the repo exists, a later failure leaves it empty; create it private before anything lands"""
    repo_url = hf_api.create_repo(repo_id, private = private, exist_ok = True)
    repo_id = getattr(repo_url, "repo_id", repo_id)
    if private:
        _ensure_hub_repo_private(hf_api, repo_id)
    return repo_id


def _hub_push_token(hf_token: HfTokenArg) -> HfTokenArg:
    if hf_token is None:
        from huggingface_hub import get_token
        return get_token()
    return hf_token


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
            delete_patterns = [_REAP_REPORT],
        )
    return repo_id


def _publish_unsloth_model_card(hf_api, repo_id, model, hf_token):
    """Writes the card upload_to_huggingface skips once the repo is opened first; best-effort, and an existing README is kept"""
    try:
        if hf_api.file_exists(repo_id, "README.md", repo_type = "model"):
            return
        config = getattr(model, "config", None)
        if config is None:
            return
        base_model = _card_base_model(model)
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


def _checkpoint_load_refusal(checkpoint_path, pinned_base) -> Optional[str]:
    """Why a local checkpoint must not load now: a pinned adapter base it no longer names, or
    weight entries that point outside the folder."""
    from core.export.decision import adapter_base_changed
    from utils.security.file_security import escaping_weight_references

    if pinned_base and adapter_base_changed(checkpoint_path, pinned_base):
        return "The adapter's base model changed after it was checked. Load it again."
    if escaping_weight_references(checkpoint_path):
        return "This checkpoint's weight index names files outside its folder."
    # A local base loads too: its own index must stay inside its folder as well.
    if pinned_base and os.path.isdir(pinned_base) and escaping_weight_references(pinned_base):
        return "The base model's weight index names files outside its folder."
    return None


def _with_base_processor(tokenizer, base_model: str, **load_kw):
    """A text-only VLM adapter saves a bare tokenizer; the export still needs the base's processor (mmproj)."""
    if tokenizer is None or hasattr(tokenizer, "tokenizer"):
        return tokenizer
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(base_model, **load_kw)
    except Exception as e:
        logger.info(f"No processor from base model {base_model}: {e}")
        return tokenizer
    if not hasattr(processor, "tokenizer"):
        return tokenizer
    processor.tokenizer = tokenizer
    if getattr(tokenizer, "chat_template", None):
        processor.chat_template = tokenizer.chat_template
    return processor


def _restore_use_cache(model) -> None:
    """The load prepares the model for training (use_cache=False on every config); export saves it."""
    try:
        from unsloth_zoo.training_utils import restore_use_cache
    except ImportError:
        return
    restore_use_cache(model)


def _cached_hub_adapter_commit(checkpoint_path: str) -> Optional[str]:
    """The commit of a Hub adapter's cached snapshot (the one its load just used), else None."""
    if not checkpoint_path or os.path.isdir(checkpoint_path):
        return None
    if os.path.isabs(checkpoint_path) or checkpoint_path.count("/") != 1:
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
        path = try_to_load_from_cache(checkpoint_path, "adapter_config.json")
    except Exception:
        return None
    if not isinstance(path, str):
        return None
    commit = os.path.basename(os.path.dirname(path))
    return commit if re.fullmatch(r"[0-9a-f]{40}", commit) else None


class ExportBackend:
    # {"layout", "adapter_only"} for a decision checkpoint (GGUF only), else None.
    decision: Optional[dict] = None

    def __init__(self):
        self.inference_backend = get_inference_backend()
        self.current_checkpoint = None
        self._checkpoint_token = None
        self._checkpoint_local_files_only = False
        self.current_model = None
        self.current_tokenizer = None
        self.is_vision = False
        self.is_peft = False
        self._audio_type = None
        self.decision = None
        self.diffusion_lm_profile = None
        # True from a load's start until cleanup_memory clears the GPU cache.
        self._gpu_cache_dirty = False

    def _diffusion_refusal(self, method: str) -> Optional[str]:
        profile = getattr(self, "diffusion_lm_profile", None)
        if not profile:
            return None
        from utils.models.diffusion_lm import export_refusal

        return export_refusal(profile, method)

    def cleanup_memory(self):
        """Offload and delete all models from memory"""
        try:
            logger.info("Starting memory cleanup...")

            model_names = list(self.inference_backend.models.keys())
            # Before the resets below; no flag (built without __init__) always clears.
            needs_clear = (
                bool(model_names)
                or self.current_model is not None
                or self.current_tokenizer is not None
                or getattr(self, "_gpu_cache_dirty", True)
            )
            for model_name in model_names:
                self.inference_backend.unload_model(model_name)

            self.current_model = None
            self.current_tokenizer = None
            self.current_checkpoint = None
            self._checkpoint_token = None
            self._checkpoint_local_files_only = False
            self._audio_type = None
            self.decision = None

            if needs_clear:
                clear_gpu_cache()
                self._gpu_cache_dirty = False

            logger.info("Memory cleanup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False
        finally:
            # Even when an unload above failed: a later load must never inherit this load's REAP
            # settings (they would read as "already pruned" for an unpruned model).
            try:
                from core.export.reap import discard_reap_staging
                discard_reap_staging(self)
            except Exception as e:
                logger.warning(f"Could not clear the REAP state: {e}")

    def scan_checkpoints(
        self, outputs_dir: Optional[str] = None
    ) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """Returns [(model_name, [(display_name, checkpoint_path), ...]), ...] for the outputs folder."""
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
        offload_layers = "auto",
        _offload_retry: bool = False,
    ) -> Tuple[bool, str]:
        """Load a checkpoint for export.

        ``base_model`` is the caller's authorized adapter base; it wins over adapter_config.json.

        ``hf_token`` authenticates the actual weight load for gated/private checkpoints, matching
        the token the worker used for the security preflight (otherwise a gated repo passes scanning
        then 401s at from_pretrained).

        ``offload_layers = "auto"`` only retries a single-GPU 4-bit load that spilled to the CPU (#13696).

        ``False`` (denied the ambient token) must travel all the way down: ``None`` reads as "go and
        find a credential" (``if token is None: get_token()`` in ``save.py`` and ``hf_login``), and
        ``get_token()`` reads the operator's stored login off disk.
        """
        token = normalize_token(hf_token)
        # Probe cache guards refuse anonymous reads, misreading a cached VLM offline.
        probe_token = token or None
        pinned_base = base_model
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            self.cleanup_memory()
            self._gpu_cache_dirty = True
            self.diffusion_lm_profile = None

            # Before the base / audio / vision probes: a decision run never reaches the chat loaders.
            from core.export.decision import decision_kind

            decision = decision_kind(checkpoint_path)
            if decision is not None:
                return self._load_decision_checkpoint(
                    str(Path(checkpoint_path).expanduser()),
                    *decision,
                    token = token,
                    base_model = base_model,
                )

            # The decision loader above checks its own base against the pin.
            refusal = _checkpoint_load_refusal(checkpoint_path, pinned_base)
            if refusal:
                return False, refusal

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
            _offload_kw = (
                _EXPORT_AUTO_OFFLOAD_KWARGS
                if _offload_retry
                else _offload_layers_kwargs(offload_layers, _device_map_kw)
            )

            # Probes run inside the forced-offline window, else a gated base 404s.
            with _offline_window_if(local_files_only):
                self._audio_type = detect_audio_type(
                    model_id, hf_token = probe_token, local_files_only = local_files_only
                )
                self.is_vision = not self._audio_type and is_vision_model(
                    model_id, hf_token = probe_token, local_files_only = local_files_only
                )
                from utils.models.diffusion_lm import diffusion_lm_profile_for_model

                self.diffusion_lm_profile = (
                    None
                    if self._audio_type
                    else diffusion_lm_profile_for_model(
                        model_id, hf_token = probe_token, local_files_only = local_files_only
                    )
                )
                if self.diffusion_lm_profile:
                    # DiffusionGemma ships a vision tower; its text-diffusion load is FastModel's.
                    self.is_vision = False

            # Again right before the load: the folder may have been edited during the probes.
            refusal = _checkpoint_load_refusal(checkpoint_path, pinned_base)
            if refusal:
                return False, refusal

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
                    **_offload_kw,
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
                    **_offload_kw,
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
                    **_offload_kw,
                )

            elif self._audio_type == "bicodec":
                from unsloth import FastModel
                from utils.models.spark_tts_adapter import spark_tts_adapter_load_target

                logger.info("Loading as BiCodec (Spark-TTS) audio model...")
                with spark_tts_adapter_load_target(
                    checkpoint_path, hf_token = token, local_files_only = local_files_only
                ) as load_target:
                    model, tokenizer = FastModel.from_pretrained(
                        model_name = load_target,
                        max_seq_length = max_seq_length,
                        dtype = None if _IS_MLX else torch.float32,
                        load_in_4bit = False,
                        trust_remote_code = trust_remote_code,
                        token = token,
                        local_files_only = local_files_only,
                        **_device_map_kw,
                        **_offload_kw,
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
                    **_offload_kw,
                )

            elif self.diffusion_lm_profile:
                from unsloth import FastModel
                logger.info(f"Loading as text-diffusion model ({self.diffusion_lm_profile})...")
                # Text-diffusion has no block swapping; skip offload like Training does.
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    **_load_in_4bit_kwargs(load_in_4bit),
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
                    **_offload_kw,
                )
                if _IS_MLX and adapter_config.exists():
                    processor = load_mlx_adapter_processor(processor, checkpoint_path)
                elif base_model:
                    with _offline_window_if(local_files_only):
                        processor = _with_base_processor(
                            processor,
                            base_model,
                            trust_remote_code = trust_remote_code,
                            token = token,
                            local_files_only = local_files_only,
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
                    **_offload_kw,
                )
                if _IS_MLX and adapter_config.exists():
                    tokenizer = load_mlx_adapter_tokenizer(tokenizer, checkpoint_path)

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
                _restore_use_cache(model)

            restored_repo_id = restore_hf_cache_repo_identity(model, base_model)
            if restored_repo_id:
                logger.info(
                    f"Restored Hub model identity for legacy adapter export: {restored_repo_id}"
                )

            # A Hub adapter has no config.json for the probe, but core loads a diffusion base through its
            # diffusion path: classify from what loaded, so the merged / GGUF refusals still apply.
            if not self.diffusion_lm_profile and not _IS_MLX:
                from utils.models.diffusion_lm import loaded_diffusion_profile
                self.diffusion_lm_profile = loaded_diffusion_profile(model)

            # The MLX GGUF LoRA converter honors the approved load decision.
            self.trust_remote_code = bool(trust_remote_code)
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_checkpoint = checkpoint_path
            self._checkpoint_token = token
            self._checkpoint_local_files_only = bool(local_files_only)
            # The base config's _commit_hash names the base repo, not the adapter's.
            self._adapter_commit = (
                (checkpoint_path, _cached_hub_adapter_commit(checkpoint_path))
                if self.is_peft
                else None
            )
            if not self.is_peft and not _IS_MLX:
                self._attach_full_checkpoint_mtp_head(checkpoint_path)

            if self._audio_type:
                model_type = f"Audio ({self._audio_type})"
            elif self.diffusion_lm_profile:
                model_type = f"Text-diffusion ({self.diffusion_lm_profile})"
            elif self.is_vision:
                model_type = "Vision"
            else:
                model_type = "Text"
            peft_info = " (PEFT Adapter)" if self.is_peft else " (Merged Model)"

            logger.info(f"Successfully loaded {model_type} model{peft_info}")
            return True, f"Loaded {model_type} model{peft_info} successfully"

        except Exception as e:
            # "balanced" budgets from free memory read before our CUDA context; can OOM on shared GPUs.
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
                # Retry outside this block: the live traceback pins the half-built model frames, so an in-block retry inherits the exhausted device
                retry_reason = str(e)
                # Name the map: an omitted one is unsloth's DEFAULT_DEVICE_MAP, which requested_device_map
                # upgrades back to the planner, re-running the placement that just failed.
                retry_kw = {"_device_map_override": {"device_map": "sequential"}}
            elif (
                offload_layers == "auto"
                and not _offload_retry
                and _device_map_override is None
                and not self.diffusion_lm_profile
                and _is_cpu_spill_rejection(e)
                and not _multi_gpu_device_map_kwargs()
            ):
                retry_reason = str(e)
                retry_kw = {"_offload_retry": True}
            else:
                logger.error(f"Error loading checkpoint: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return False, f"Failed to load checkpoint: {str(e)}"

        if "_offload_retry" in retry_kw:
            logger.warning(
                f"Export load does not fit in GPU memory ({retry_reason}); retrying with "
                f"offload_layers='auto' to keep the layers the GPU cannot hold in host RAM."
            )
        else:
            logger.warning(
                f"Multi-GPU export load unusable ({retry_reason}); retrying on "
                f"the sequential loader default."
            )
        self.cleanup_memory()
        return self.load_checkpoint(
            checkpoint_path,
            max_seq_length = max_seq_length,
            load_in_4bit = load_in_4bit,
            offload_layers = offload_layers,
            trust_remote_code = trust_remote_code,
            hf_token = hf_token,
            base_model = base_model,
            **retry_kw,
        )

    def _load_decision_checkpoint(
        self,
        checkpoint_path: str,
        layout: str,
        adapter_only: bool,
        token: HfTokenArg = None,
        base_model: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """Records a decision checkpoint; its weights load (adapters) or convert (merged) at export."""
        from core.export.decision import (
            DecisionExportError,
            adapter_base,
            check_decision_eligibility,
        )

        if not _export_runtime_available():
            return False, _export_runtime_message()
        if adapter_only:
            # The base FastDecisionModel will load must be the one the route authorized and scanned.
            try:
                resolved = adapter_base(checkpoint_path)
            except DecisionExportError as exc:
                return False, str(exc)
            if base_model and resolved != base_model:
                return False, (
                    f"This adapter loads {resolved}, not the authorized base model {base_model}."
                )
        try:
            check_decision_eligibility(checkpoint_path, token)
        except DecisionExportError as exc:
            return False, str(exc)
        self.decision = {"layout": layout, "adapter_only": adapter_only}
        # The base is loaded only at export: kept here and rechecked right before that load.
        self._decision_base_pin = resolved if adapter_only else None
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
        from core.export.decision import DecisionExportError, adapter_base, run_decision_gguf_export

        pinned = getattr(self, "_decision_base_pin", None)
        if pinned:
            try:
                current = adapter_base(self.current_checkpoint)
            except DecisionExportError as exc:
                return False, str(exc), None
            if current != pinned:
                return (
                    False,
                    "The adapter's base model changed after it was loaded. Load it again.",
                    None,
                )
        refusal = _checkpoint_load_refusal(self.current_checkpoint, pinned)
        if refusal:
            return False, refusal, None
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

    def _sync_gguf_reap_reports(self, folder, ggufs) -> List[str]:
        """One report beside each GGUF this export wrote (a reused folder keeps other files);
        returns their names, written or removed."""
        try:
            from core.export.reap import gguf_report_name, sync_reap_report

            names = []
            for gguf in ggufs:
                name = gguf_report_name(gguf)
                sync_reap_report(self, folder, name, [os.path.basename(gguf)])
                names.append(name)
            return names
        except Exception as exc:  # noqa: BLE001 - the report never blocks an export
            logger.warning(f"Could not update the REAP reports in {folder}: {exc}")
            return []

    def _sync_reap_report(self, folder) -> None:
        """reap_report.json in ``folder`` describes the weights it holds (written for a REAP pruned
        model, a stale one removed otherwise); run before a folder is uploaded."""
        try:
            from core.export.reap import sync_reap_report
            sync_reap_report(self, folder)
        except Exception as exc:  # noqa: BLE001 - the report never blocks an export
            logger.warning(f"Could not update reap_report.json in {folder}: {exc}")

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
        """Export a merged model; compressed_method (fp8, w8a8, nvfp4 and others) overrides format_type, resolved via unsloth.save COMPRESSED_EXPORT_SCHEMES"""
        if self.decision is not None:
            return self._decision_only_gguf()
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None
        refusal = self._diffusion_refusal("merged")
        if refusal:
            return False, refusal, None

        # save_pretrained_merged is a no-op merge for non-PEFT base models, so one path covers both.

        output_path: Optional[str] = None
        save_dir_was_empty = False
        # compressed-tensors (NVIDIA-only) vs portable torchao FP8/INT8.
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

        # transformers 5 refuses torchao < 0.15 as "requires torchao to be installed"; name the real cause.
        if _is_torchao_alias(compressed_alias):
            _old_torchao = _torchao_too_old()
            if _old_torchao:
                return (
                    False,
                    f"Portable torchao FP8/INT8 export needs torchao 0.15 or newer, but torchao "
                    f"{_old_torchao} is installed. Update Unsloth to install a newer torchao, or "
                    "use 16-bit merged or GGUF quantization instead.",
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

                # Prefer the llm-compressor-main shadow (transformers 5.x), since the shipped 0.10.x cannot quantize newer models
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
                # Reused folders may hold leftovers, so only a fresh one is pushed as is.
                save_dir_was_empty = _dir_is_fresh(save_directory)
                ensure_dir(Path(save_directory))

                # save.py turns None into get_token(), so the merge needs the token spelled out.
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
                if not _IS_MLX:
                    # Compressed / torchao siblings carry no stitched MTP head; an EAGLE-3 drafter ships.
                    self._package_drafter_merged(
                        final_dir,
                        repo_id if push_to_hub else None,
                        allow_mtp = not (is_compressed or is_torchao),
                    )
                logger.info(f"Model saved successfully to {final_dir}")
                output_path = str(Path(final_dir).resolve())
                # Complete on disk even if the Hub upload below fails (the worker syncs REAP's report).
                self._local_export_path = output_path
                # Before any upload: the folder's REAP report must describe the weights it now holds.
                self._sync_reap_report(output_path)

            if push_to_hub:
                hf_token = _hub_push_token(hf_token)
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
                        # Upload the built artifact; push_to_hub_merged would redo the merge and quantization.
                        with contextlib.ExitStack() as stack:
                            upload_dir = output_path
                            if not (is_compressed or is_torchao or save_dir_was_empty):
                                # A reused folder can hold leftovers, so upload from a clean staged second save
                                upload_dir = stack.enter_context(
                                    _staging_dir(Path(output_path).parent)
                                )
                                self.current_model.save_pretrained_merged(
                                    upload_dir,
                                    self.current_tokenizer,
                                    save_method = save_method,
                                    **merged_token_kw,
                                )
                                self._package_drafter_merged(upload_dir, repo_id)
                                self._sync_reap_report(upload_dir)
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
                                    delete_patterns = [_REAP_REPORT],
                                )
                                uploaded = True
                    if uploaded:
                        # Last and best-effort like the GGUF card; an existing card is kept, as push_to_hub_merged does
                        try:
                            if not hf_api.file_exists(repo_id, "README.md", repo_type = "model"):
                                base_model = _card_base_model(self.current_model)
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

        # A LoRA load stays one after REAP merged its adapters in memory for a pruned export.
        if self.is_peft or getattr(self, "_reap_merged_peft", False):
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
                    # A full fine-tune trained with a drafter ships it like a merged export.
                    self._package_drafter_merged(save_directory, repo_id if push_to_hub else None)

                self._write_export_metadata(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())
                self._local_export_path = output_path
                self._sync_reap_report(output_path)

            if push_to_hub:
                hf_token = _hub_push_token(hf_token)
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
                    base_model = base_model_id or _card_base_model(self.current_model)

                    with contextlib.ExitStack() as stack:
                        upload_dir = save_directory
                        if save_directory and not save_dir_was_empty:
                            # A reused folder can hold leftovers, so upload from a clean staged second save
                            upload_dir = stack.enter_context(
                                _staging_dir(Path(save_directory).parent)
                            )
                            self.current_model.save_pretrained(upload_dir)
                            self.current_tokenizer.save_pretrained(upload_dir)
                            self._package_drafter_merged(upload_dir, repo_id)
                            self._sync_reap_report(upload_dir)
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
                                delete_patterns = [_REAP_REPORT],
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
        """A list of ``quantization_method`` values yields one GGUF per quant from a single model load."""
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if self.decision is not None:
            return self._export_decision_gguf(
                quantization_method, push_to_hub, imatrix_file, npu_q4nx
            )
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None
        refusal = self._diffusion_refusal("gguf")
        if refusal:
            return False, refusal, None

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
        # True only after the load's consent gate passed; only then may the converter run auto_map code.
        remote_code_approved = bool(getattr(self, "trust_remote_code", False))
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
        gguf_reports: List[str] = []
        drafter_gguf_files: List[str] = []
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
                    with _llama_cpp_scripts_pin(), _converter_remote_code(remote_code_approved):
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
                    drafter_gguf_files = self._package_drafter_gguf(
                        reported.get("save_directory") or _model_tmp, abs_save_dir, exported_ggufs
                    )
                    exported_ggufs.extend(f for f in drafter_gguf_files if f.endswith(".gguf"))
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

                # iterdir, not glob.glob: glob hides dot-leading names, so an empty model stem .Q4_K_M.gguf reads as none; this list is the success gate
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
                # Before the upload below, which allow-lists each file's report when there is one.
                # The model's own GGUFs only: an EAGLE-3 drafter is converted separately, not pruned.
                drafter_names = {os.path.basename(f) for f in drafter_gguf_files}
                gguf_reports = self._sync_gguf_reap_reports(
                    output_path,
                    [f for f in exported_ggufs if os.path.basename(f) not in drafter_names],
                )

                if npu_q4nx:
                    source = q4nx.source_gguf(exported_ggufs, quant_methods)
                    try:
                        if source is None:
                            raise RuntimeError("no Q4_0, Q4_1 or Q4_K_M GGUF was written")
                        with q4nx.staged_output(Path(abs_save_dir) / "npu-q4nx") as staging:
                            q4nx.convert_gguf_to_q4nx(source, staging)
                            self._write_q4nx_companions(staging, exported_config)
                            # The NPU folder is deployed on its own, so it carries its own report.
                            self._sync_reap_report(staging)
                    except Exception as exception:
                        logger.error(f"Q4NX conversion failed: {exception}")
                        return (
                            False,
                            f"GGUF files were saved to {output_path}, but the AMD NPU (Q4NX) "
                            f"conversion failed: {exception}",
                            output_path,
                        )

            if push_to_hub:
                hf_token = _hub_push_token(hf_token)
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
                            *(["drafter.json"] if drafter_gguf_files else []),
                            *(
                                glob.escape(name)
                                for name in gguf_reports
                                if (Path(output_path) / name).is_file()
                            ),
                        ],
                        # A report an earlier pruned export pushed for these same files.
                        delete_patterns = [glob.escape(name) for name in gguf_reports] or None,
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
                    with _llama_cpp_scripts_pin(), _converter_remote_code(remote_code_approved):
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

    def _attach_full_checkpoint_mtp_head(self, checkpoint_path: str) -> None:
        """A full fine-tune keeps its trained MTP head only inside its shards, which transformers
        drops on load: attach it so every export writes it back."""
        try:
            from core.export.drafter_export import training_manifest

            token = getattr(self, "_checkpoint_token", None)
            manifest = training_manifest(
                checkpoint_path,
                token,
                revision = self._loaded_commit(),
                local_files_only = getattr(self, "_checkpoint_local_files_only", False),
            )
            if not manifest or manifest.get("strategy") != "mtp":
                return
            source = checkpoint_path
            if not os.path.isdir(source):
                source = self._loaded_hub_snapshot(checkpoint_path, token)
                if source is None:
                    logger.warning(
                        f"Could not find the cached files of {checkpoint_path} to attach its trained MTP head"
                    )
                    return
            if os.path.isfile(os.path.join(source, "adapter_config.json")):
                return
            from unsloth import FastModel

            FastModel.get_mtp_head(self.current_model, source = source)
            logger.info(f"Attached the trained MTP head from {checkpoint_path}")
        except Exception as exception:
            logger.warning(f"Could not attach the trained MTP head: {exception}")

    def _loaded_commit(self) -> Optional[str]:
        """The Hub commit current_checkpoint was loaded at; None for a local folder, or a Hub adapter
        whose commit is unknown (no pin beats the base's commit)."""
        if getattr(self, "is_peft", False):
            recorded = getattr(self, "_adapter_commit", None)
            if recorded and recorded[0] == getattr(self, "current_checkpoint", None):
                return recorded[1]
            return None
        return getattr(getattr(self.current_model, "config", None), "_commit_hash", None)

    def _loaded_manifest_kwargs(self) -> dict:
        """The load's commit and offline decision for a Hub id's drafter manifest read."""
        kwargs = {}
        revision = self._loaded_commit()
        if revision:
            kwargs["revision"] = revision
        if getattr(self, "_checkpoint_local_files_only", False):
            kwargs["local_files_only"] = True
        return kwargs

    def _loaded_hub_snapshot(self, repo_id: str, token) -> Optional[str]:
        """The local snapshot folder of a Hub checkpoint the current model was loaded from."""
        revision = self._loaded_commit()
        try:
            from huggingface_hub import hf_hub_download

            # Not snapshot_download: offline it refuses a snapshot lacking files the load never needed.
            config = hf_hub_download(
                repo_id, "config.json", revision = revision, token = token, local_files_only = True
            )
        except Exception:
            return None
        path = os.path.dirname(config)
        return path if os.path.isdir(path) else None

    def _package_drafter_merged(
        self,
        final_dir: str,
        repo_id: Optional[str],
        allow_mtp: bool = True,
    ) -> None:
        """Ship the drafter trained with this fine-tune next to a merged export (best-effort: the
        merged model is valid without it)."""
        try:
            from core.export.drafter_export import package_merged
            info = package_merged(
                self.current_checkpoint,
                final_dir,
                verifier = repo_id,
                allow_mtp = allow_mtp,
                token = self._checkpoint_token,
                **self._loaded_manifest_kwargs(),
            )
            if info:
                logger.info(f"Packaged the trained {info['strategy']} drafter into {final_dir}")
        except Exception as exception:
            logger.warning(f"Could not package the trained drafter: {exception}")

    def _package_drafter_lora(
        self,
        adapter_dir: str,
        head_saved: Optional[bool] = None,
    ) -> bool:
        try:
            from core.export.drafter_export import package_lora
            return (
                package_lora(
                    self.current_checkpoint,
                    adapter_dir,
                    head_saved,
                    token = self._checkpoint_token,
                    **self._loaded_manifest_kwargs(),
                )
                is not None
            )
        except Exception as exception:
            logger.warning(f"Could not carry the drafter manifest with the adapter: {exception}")
            return False

    def _push_drafter_lora_manifest(self, hf_api, repo_id: str) -> None:
        """push_to_hub uploads the head beside the adapter but not Studio's manifest."""
        try:
            from unsloth.save import _attached_mtp_head
            from core.export.drafter_export import DRAFTER_MANIFEST
            with tempfile.TemporaryDirectory() as tmp:
                head = _attached_mtp_head(self.current_model) is not None
                if self._package_drafter_lora(tmp, head_saved = head):
                    hf_api.upload_file(
                        path_or_fileobj = os.path.join(tmp, DRAFTER_MANIFEST),
                        path_in_repo = DRAFTER_MANIFEST,
                        repo_id = repo_id,
                        repo_type = "model",
                    )
        except Exception as exception:
            logger.warning(f"Could not upload the drafter manifest with the adapter: {exception}")

    def _package_drafter_gguf(self, merged_dir: str, gguf_dir: str, ggufs: List[str]) -> List[str]:
        try:
            from core.export.drafter_export import package_gguf

            info, files = package_gguf(
                self.current_checkpoint,
                merged_dir,
                gguf_dir,
                ggufs,
                token = self._checkpoint_token,
                **self._loaded_manifest_kwargs(),
            )
            if info:
                logger.info(f"Packaged the trained {info['strategy']} drafter with the GGUF export")
            return files
        except Exception as exception:
            logger.warning(
                f"Could not package the trained drafter with the GGUF export: {exception}"
            )
            return []

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
        from utils import llama_cpp_source

        default_dir = os.path.normpath(_zoo_llama_cpp.LLAMA_CPP_DEFAULT_DIR)
        source_dir = os.path.join(os.path.dirname(default_dir), "llama.cpp-source")
        base_source_dir = source_dir
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
            # The installed binaries' revision (else latest), from its unslothai/llama.cpp release.
            try:
                repo, tag = _zoo_llama_cpp._resolve_converter_revision(default_dir)
            except Exception:
                repo, tag = None, None
            converter = None
            if tag and llama_cpp_source.is_fork_release_tag(repo, tag):
                source_dir = f"{source_dir}-{tag}"
                if os.path.exists(os.path.join(source_dir, "convert_lora_to_gguf.py")):
                    converter = os.path.join(source_dir, "convert_lora_to_gguf.py")
            elif tag and not getattr(_zoo_llama_cpp, "_converter_network_allowed", lambda: True)():
                # Offline only: online, the download path resolves the stable mix release first.
                cached = sorted(
                    glob.glob(f"{glob.escape(source_dir)}-{glob.escape(tag)}-mix-*"),
                    key = os.path.getmtime,
                )
                cached = [
                    d for d in cached if os.path.isfile(os.path.join(d, "convert_lora_to_gguf.py"))
                ]
                if cached:
                    converter = os.path.join(cached[-1], "convert_lora_to_gguf.py")
            # Trees the previous git-clone exporter left: llama.cpp-source-<upstream tag>, or
            # llama.cpp-source when no revision was known.
            legacy = base_source_dir + (f"-{tag.split('-mix-')[0]}" if tag else "")
            if converter is None and os.path.isfile(
                os.path.join(legacy, "convert_lora_to_gguf.py")
            ):
                converter = os.path.join(legacy, "convert_lora_to_gguf.py")
            if converter is None and not tag:
                # No revision known (offline, no marker): the newest tree any exporter left.
                trees = [
                    d
                    for d in glob.glob(f"{glob.escape(base_source_dir)}-*")
                    if os.path.isfile(os.path.join(d, "convert_lora_to_gguf.py"))
                ]
                if trees:
                    converter = os.path.join(
                        max(trees, key = os.path.getmtime), "convert_lora_to_gguf.py"
                    )
        if converter is None:
            if not getattr(_zoo_llama_cpp, "_converter_network_allowed", lambda: True)():
                raise RuntimeError(
                    "GGUF adapter export needs llama.cpp's convert_lora_to_gguf.py, which is not "
                    "installed, and offline mode forbids downloading it; extract the "
                    f"{llama_cpp_source.FORK_REPO} release's llama.cpp-source-<tag>.tar.gz into "
                    f"{source_dir}."
                )
            if not getattr(_zoo_llama_cpp, "_auto_install_enabled", lambda: True)():
                raise RuntimeError(
                    "GGUF adapter export needs llama.cpp's converter sources and automatic "
                    "installation was declined (UNSLOTH_AUTO_INSTALL=0); extract the "
                    f"{llama_cpp_source.FORK_REPO} release's llama.cpp-source-<tag>.tar.gz into "
                    f"{source_dir}."
                )
            # Not install_llama_cpp: it probes apt-get even when only fetching sources, which fails on macOS.
            try:
                fetched = llama_cpp_source.download_converter_source(
                    repo, tag, Path(source_dir).parent
                )
            except Exception as exc:
                raise RuntimeError(
                    "Could not download llama.cpp's converter sources from the "
                    f"{llama_cpp_source.FORK_REPO} release: {exc}"
                ) from exc
            converter = os.path.join(str(fetched), "convert_lora_to_gguf.py")
            if not os.path.exists(converter):
                raise RuntimeError(f"convert_lora_to_gguf.py is missing from {fetched}.")
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
        """Adapter only, not merged; adapter_format is mlx or peft (omitted = platform native); gguf writes a LoRA GGUF (q8_0/f16/bf16/f32)"""
        if self.decision is not None:
            return self._decision_only_gguf()
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None
        refusal = self._diffusion_refusal("gguf") if gguf else None
        if refusal:
            return False, refusal, None

        if getattr(self, "_reap_merged_peft", False):
            return (
                False,
                "The LoRA adapters were merged into the model for a REAP export. Load the "
                "checkpoint again to export the adapter.",
                None,
            )
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
            # convert_lora_to_gguf.py ignores DoRA's lora_magnitude_vector, silently dropping it.
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
                # False keeps a denied caller off get_token().
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
                    self._package_drafter_lora(save_directory)
                self._write_export_metadata(save_directory)
                logger.info(f"Adapter saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            if push_to_hub:
                hf_token = _hub_push_token(hf_token)
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
                            # A reused folder can hold leftovers, so upload a clean second save; the converter names the GGUF model after its folder
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
                        # Serialise first so a failed write does not leave an empty repo behind.
                        self._save_mlx_adapter(tmp_dir, resolved_format)
                        repo_id = _open_hub_repo(hf_api, repo_id, private)
                        hf_api.upload_folder(
                            folder_path = tmp_dir,
                            repo_id = repo_id,
                            repo_type = "model",
                        )
                else:
                    # Create the repo here: another client could create it public first, and private cannot fix that.
                    repo_id = _open_hub_repo(hf_api, repo_id, private)
                    _publish_unsloth_model_card(hf_api, repo_id, self.current_model, hf_token)
                    self.current_model.push_to_hub(repo_id, token = hf_token, private = private)
                    self.current_tokenizer.push_to_hub(repo_id, token = hf_token, private = private)
                    self._push_drafter_lora_manifest(hf_api, repo_id)
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
