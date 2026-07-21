# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Export backend - exports models in various formats."""

import glob
import json
import structlog
import tempfile
from loggers import get_logger
import os
import shutil
import contextlib
from pathlib import Path
from typing import Optional, Tuple, List

# unsloth imports torch on non-MLX hosts, so a --no-torch install raises here. Stay importable
# (null the classes) so exports return a clean "PyTorch is not installed" error, not an import crash.
try:
    from unsloth import FastLanguageModel, FastVisionModel, _IS_MLX
    _UNSLOTH_IMPORT_ERROR = None
except (
    Exception
) as _unsloth_exc:  # ImportError (e.g. missing torch) or a broken native load
    FastLanguageModel = None
    FastVisionModel = None
    _IS_MLX = False
    _UNSLOTH_IMPORT_ERROR = _unsloth_exc

from huggingface_hub import HfApi, ModelCard
from utils.hardware import clear_gpu_cache

from utils.models import is_vision_model, get_base_model_from_lora
from utils.models.model_config import detect_audio_type
from utils.paths import (
    ensure_dir,
    outputs_root,
    resolve_export_write_dir,
    resolve_output_dir,
)
from core.inference import get_inference_backend

# GPU/PyTorch-only imports, skipped on MLX and on a --no-torch install so the module stays
# importable; export then degrades to a clear "PyTorch is not installed" error.
torch = None
_TORCH_IMPORT_ERROR: Optional[BaseException] = None
if not _IS_MLX:
    try:
        from peft import PeftModel, PeftModelForCausalLM
        from transformers.modeling_utils import PushToHubMixin
        import torch
    except Exception as _torch_exc:  # ImportError, or a broken native torch load
        _TORCH_IMPORT_ERROR = _torch_exc

logger = get_logger(__name__)


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


def _supports_kwarg(fn, name):
    """True if `fn` accepts keyword `name` directly or via **kwargs."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _compressed_export_supported():
    """True if the installed unsloth build can do FP8/NVFP4 compressed-tensors export."""
    try:
        import unsloth.save as _us
        return hasattr(_us, "_normalize_compressed_method")
    except Exception:
        return False


def _torchao_export_supported():
    """True if the installed unsloth build has the portable torchao FP8/INT8 export path."""
    try:
        import unsloth.save as _us
        return hasattr(_us, "_normalize_torchao_method")
    except Exception:
        return False


def _has_nvidia_gpu():
    """True only on a real NVIDIA CUDA box (not ROCm/XPU/CPU/MLX); compressed-tensors needs it."""
    try:
        from utils.hardware import hardware as _hw
        return _hw.DEVICE == _hw.DeviceType.CUDA and not _hw.IS_ROCM
    except Exception:
        try:
            import torch
            return (
                bool(torch.cuda.is_available())
                and getattr(torch.version, "hip", None) is None
            )
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
    if os.environ.get("UNSLOTH_OFFLINE_PROBE", "1").strip().lower() in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False  # probe disabled -> assume online; loads still pass local_files_only on env

    # Shared bounded, proxy-aware probe (also used by the export worker before version activation).
    from utils.transformers_version import hf_endpoint_unreachable

    if hf_endpoint_unreachable(timeout):
        logger.warning(
            "Hugging Face endpoint unreachable; loading checkpoint in offline mode"
        )
        return True
    return False


# Reuse Unsloth's lock-guarded forced-offline context; no-op fallback if it moves.
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
        return "microsoft" in open("/proc/version").read().lower()
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
            logger.info(
                "WSL detected — skipping sudo check (build deps pre-installed by setup.sh)"
            )
            return False

        llama_cpp_module.do_we_need_sudo = _wsl_do_we_need_sudo
        logger.info("Applied WSL sudo patch to unsloth_zoo.llama_cpp.do_we_need_sudo")
    except Exception as e:
        logger.warning(f"Could not apply WSL sudo patch: {e}")


# Model card template
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


class ExportBackend:
    """Handles model export operations"""

    def __init__(self):
        self.inference_backend = get_inference_backend()
        self.current_checkpoint = None
        self.current_model = None
        self.current_tokenizer = None
        self.is_vision = False
        self.is_peft = False
        self._audio_type = None

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

            clear_gpu_cache()

            logger.info("Memory cleanup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False

    def scan_checkpoints(
        self, outputs_dir: str = str(outputs_root())
    ) -> List[Tuple[str, List[Tuple[str, str]]]]:
        """
        Scan outputs folder for training runs and their checkpoints.

        Returns: [(model_name, [(display_name, checkpoint_path), ...]), ...]
        """
        from utils.models.checkpoints import scan_checkpoints
        return scan_checkpoints(outputs_dir = outputs_dir)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
        hf_token: Optional[str] = None,
    ) -> Tuple[bool, str]:
        """
        Load a checkpoint for export.

        ``hf_token`` authenticates the actual weight load for gated/private
        checkpoints, matching the token the worker used for the security preflight
        (otherwise a gated repo passes scanning then 401s at from_pretrained).

        Returns:
            Tuple of (success: bool, message: str)
        """
        token = hf_token if hf_token and hf_token.strip() else None
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            self.cleanup_memory()

            checkpoint_path_obj = Path(checkpoint_path)

            # Model identity for type detection
            adapter_config = checkpoint_path_obj / "adapter_config.json"
            base_model = None
            if adapter_config.exists():
                base_model = get_base_model_from_lora(checkpoint_path)
                if not base_model:
                    return False, "Could not determine base model for adapter"

            model_id = base_model or checkpoint_path

            # Skip the Hub when offline so a no-internet export uses the local cache.
            local_files_only = _hf_offline()

            # Run the type-detection probes in the forced-offline window (else a gated
            # base 404s); it covers is_vision_model's Hub reads + the transformers-5
            # subprocess, and local_files_only makes detect_audio_type's requests.get skip.
            with _offline_window_if(local_files_only):
                self._audio_type = detect_audio_type(
                    model_id, hf_token = token, local_files_only = local_files_only
                )
                self.is_vision = not self._audio_type and is_vision_model(
                    model_id, hf_token = token, local_files_only = local_files_only
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
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                )

            elif self._audio_type == "snac":
                logger.info("Loading as SNAC (Orpheus) audio model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
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
                )

            elif self.is_vision:
                logger.info("Loading as vision model...")
                model, processor = FastVisionModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                )
                tokenizer = processor  # vision: processor acts as tokenizer

            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                )

            if _IS_MLX:
                # MLX doesn't use PeftModel — detect LoRA via adapter_config.json
                self.is_peft = adapter_config.exists()
            else:
                self.is_peft = isinstance(model, (PeftModel, PeftModelForCausalLM))

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
            logger.error(f"Error loading checkpoint: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Failed to load checkpoint: {str(e)}"

    def _write_export_metadata(self, save_directory: str):
        """Write export_metadata.json with base model info for Chat page discovery."""
        try:
            base_model = (
                get_base_model_from_lora(self.current_checkpoint)
                if self.current_checkpoint
                else None
            )
            metadata = {"base_model": base_model}
            metadata_path = os.path.join(save_directory, "export_metadata.json")
            with open(metadata_path, "w") as f:
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
        hf_token: Optional[str] = None,
        private: bool = False,
        compressed_method: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export merged model (for PEFT models).

        Args:
            save_directory: Local directory to save model
            format_type: "16-bit (FP16)", "4-bit (FP4)", or a compressed-tensors label
            compressed_method: Optional compressed-tensors scheme alias (e.g. "fp8",
                "fp8_static", "w8a8", "w4a16", "mxfp4", "mxfp8", "nvfp4"). Overrides
                format_type and is resolved against unsloth.save COMPRESSED_EXPORT_SCHEMES.
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID (username/model-name)
            hf_token: Hugging Face token
            private: Whether to make the repo private

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        # Merged export works for PEFT adapters and non-PEFT Local/HF base models alike
        # (save_pretrained_merged is a no-op merge that just saves the base).

        output_path: Optional[str] = None
        # Quantized formats save to a sibling "<dir>-<suffix>". Two backends: compressed-tensors
        # (llm-compressor, NVIDIA-only) and portable torchao FP8/INT8 (device-agnostic). The alias
        # comes from `compressed_method` (the "all formats" dropdown) or the `format_type` label.
        _LABEL_TO_ALIAS = {
            "FP8 (compressed-tensors)": "fp8",
            "NVFP4 (compressed-tensors)": "nvfp4",
        }
        compressed_alias = compressed_method or _LABEL_TO_ALIAS.get(format_type)
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

                # Prefer the llm-compressor-main shadow (transformers 5.x): it quantizes newer models
                # (Qwen3.5, Gemma-4, ...) the shipped 0.10.x cannot. Route all compressed exports
                # through it when available; else fall back to the workspace 0.10.x path below.
                _shadow_pp = None
                try:
                    from utils.transformers_version import (
                        llmcompressor_shadow_pythonpath,
                    )
                    _shadow_pp = llmcompressor_shadow_pythonpath()
                except Exception as e:
                    logger.warning(f"llm-compressor-main shadow unavailable: {e}")
                if _shadow_pp:
                    os.environ[_us._COMPRESSED_QUANTIZE_PYTHONPATH_ENV] = _shadow_pp
                else:
                    # No shadow (disabled/offline/failed): the workspace 0.10.x cannot exceed its
                    # transformers ceiling, so fail fast for sidecar models; default-tier still works.
                    os.environ.pop(_us._COMPRESSED_QUANTIZE_PYTHONPATH_ENV, None)
                    _exceeds, _tf_ver = (
                        _us._transformers_exceeds_llm_compressor_ceiling()
                    )
                    if _exceeds:
                        return (
                            False,
                            "FP8/FP4 compressed-tensors export is not available for this model: it "
                            f"runs under transformers {_tf_ver}, but the installed llm-compressor "
                            f"supports transformers <= {_us._LLM_COMPRESSOR_MAX_TRANSFORMERS} and the "
                            "llm-compressor-main runtime could not be provisioned (offline or "
                            "UNSLOTH_DISABLE_LLMCOMPRESSOR_MAIN). Export to GGUF or 16-bit instead.",
                            None,
                        )

                try:
                    info = _us._normalize_compressed_method(compressed_alias)
                except Exception as e:
                    return (
                        False,
                        f"Unsupported compressed export '{compressed_alias}': {e}",
                        None,
                    )
                if info is None:
                    return (
                        False,
                        f"'{compressed_alias}' is not a recognized compressed-tensors export.",
                        None,
                    )
                compressed_suffix = info[2]

            if _IS_MLX:
                mlx_save_method = (
                    "merged_4bit" if format_type == "4-bit (FP4)" else "merged_16bit"
                )
            elif is_compressed or is_torchao:
                save_method = compressed_alias
            elif format_type == "4-bit (FP4)":
                save_method = "merged_4bit_forced"
            elif self._audio_type == "whisper":
                save_method = None
            else:
                save_method = "merged_16bit"

            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving merged model locally to: {save_directory}")
                ensure_dir(Path(save_directory))

                if _IS_MLX:
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                        save_method = mlx_save_method,
                    )
                else:
                    self.current_model.save_pretrained_merged(
                        save_directory, self.current_tokenizer, save_method = save_method
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
                    if save_directory:
                        self.current_model.push_to_hub_merged(
                            repo_id,
                            self.current_tokenizer,
                            save_directory = save_directory,
                            token = hf_token,
                            private = private,
                        )
                    else:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            self.current_model.save_pretrained_merged(
                                tmp_dir,
                                self.current_tokenizer,
                                save_method = mlx_save_method,
                            )
                            self.current_model.push_to_hub_merged(
                                repo_id,
                                self.current_tokenizer,
                                save_directory = tmp_dir,
                                token = hf_token,
                                private = private,
                            )
                elif (
                    (is_compressed or is_torchao)
                    and output_path
                    and Path(output_path).is_dir()
                ):
                    # Already built in output_path; upload it directly instead of re-running the
                    # expensive quantization that push_to_hub_merged(save_method=...) would redo.
                    hf_api = HfApi(token = hf_token)
                    repo_id = PushToHubMixin._create_repo(
                        PushToHubMixin,
                        repo_id = repo_id,
                        private = private,
                        token = hf_token,
                    )
                    content = MODEL_CARD.format(
                        username = repo_id.split("/")[0],
                        base_model = getattr(
                            self.current_model.config, "_name_or_path", "unknown"
                        ),
                        model_type = getattr(
                            self.current_model.config, "model_type", "llm"
                        ),
                        method = compressed_alias or format_type,
                        extra = "unsloth",
                    )
                    ModelCard(content).push_to_hub(
                        repo_id, token = hf_token, commit_message = "Unsloth Model Card"
                    )
                    hf_api.upload_folder(
                        folder_path = output_path,
                        repo_id = repo_id,
                        repo_type = "model",
                    )
                else:
                    hub_save_method = (
                        save_method if save_method is not None else "merged_16bit"
                    )
                    self.current_model.push_to_hub_merged(
                        repo_id,
                        self.current_tokenizer,
                        save_method = hub_save_method,
                        token = hf_token,
                        private = private,
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
        hf_token: Optional[str] = None,
        private: bool = False,
        base_model_id: Optional[str] = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export base model (for non-PEFT models).

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
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
        try:
            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving base model locally to: {save_directory}")
                ensure_dir(Path(save_directory))

                if _IS_MLX:
                    # MLX: save_pretrained_merged handles non-LoRA models too
                    # (fuse() is a no-op without LoRA layers)
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                        save_method = "merged_16bit",
                    )
                else:
                    self.current_model.save_pretrained(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)

                # Write export metadata so the Chat page can identify the base model
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
                    if save_directory:
                        self.current_model.push_to_hub_merged(
                            repo_id,
                            self.current_tokenizer,
                            save_directory = save_directory,
                            token = hf_token,
                            private = private,
                        )
                    else:
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            self.current_model.save_pretrained_merged(
                                tmp_dir,
                                self.current_tokenizer,
                                save_method = "merged_16bit",
                            )
                            self.current_model.push_to_hub_merged(
                                repo_id,
                                self.current_tokenizer,
                                save_directory = tmp_dir,
                                token = hf_token,
                                private = private,
                            )
                else:
                    # Base model name from request or model config
                    base_model = (
                        base_model_id
                        or self.current_model.config._name_or_path
                        or "unknown"
                    )

                    hf_api = HfApi(token = hf_token)
                    repo_id = PushToHubMixin._create_repo(
                        PushToHubMixin,
                        repo_id = repo_id,
                        private = private,
                        token = hf_token,
                    )
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
                            folder_path = save_directory,
                            repo_id = repo_id,
                            repo_type = "model",
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
        hf_token: Optional[str] = None,
        imatrix_file = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export model in GGUF format.

        Args:
            save_directory: Local directory to save model
            quantization_method: A single GGUF quant method (e.g., "Q4_K_M") or a list of them
                (e.g., ["Q4_K_M", "Q8_0"]). A list produces one GGUF per quant from a single
                model load (unsloth save_to_gguf loops internally).
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID
            hf_token: Hugging Face token

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        # Only forward imatrix_file to an unsloth build that accepts it, else older builds raise
        # an unexpected-keyword error even for a plain no-imatrix export.
        if imatrix_file is not None and not _supports_kwarg(
            self.current_model.save_pretrained_gguf, "imatrix_file"
        ):
            return (
                False,
                "This Unsloth build does not support GGUF imatrix export. "
                "Upgrade unsloth and unsloth_zoo, or disable the imatrix option.",
                None,
            )
        imatrix_kw = {"imatrix_file": imatrix_file} if imatrix_file is not None else {}

        output_path: Optional[str] = None
        model_tmp_to_cleanup: Optional[str] = None
        try:
            # Normalize to a lowercased list so multiple quants come from one model load.
            if isinstance(quantization_method, (list, tuple)):
                quant_methods = [
                    str(q).lower() for q in quantization_method if str(q).strip()
                ]
            else:
                quant_methods = [str(quantization_method).lower()]
            if not quant_methods:
                quant_methods = ["q4_k_m"]
            quant_method = quant_methods if len(quant_methods) > 1 else quant_methods[0]

            # Pin convert_hf_to_gguf.py to setup.sh's tagged llama.cpp ref so it
            # can't drift past the pinned llama-quantize binary's gguf API.
            global _LLAMA_CPP_SCRIPTS_WARNING_EMITTED
            try:
                from unsloth_zoo.llama_cpp import (
                    LLAMA_CPP_DEFAULT_DIR,
                    _resolve_local_convert_script,  # noqa: F401
                )
                os.environ.setdefault(
                    "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR
                )
            except ImportError:
                if not _LLAMA_CPP_SCRIPTS_WARNING_EMITTED:
                    logger.warning(
                        "Unsloth: installed unsloth_zoo does not honor "
                        "UNSLOTH_LLAMA_CPP_SCRIPTS_DIR; convert_hf_to_gguf.py will "
                        "still be downloaded from llama.cpp master and may drift "
                        "past the pinned llama-quantize binary. Upgrade unsloth_zoo "
                        "to activate the local script pin."
                    )
                    _LLAMA_CPP_SCRIPTS_WARNING_EMITTED = True

            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                # Keep unsloth relative-path internals anchored to the repo cwd.
                abs_save_dir = os.path.abspath(save_directory)
                logger.info(f"Saving GGUF model locally to: {abs_save_dir}")

                ensure_dir(Path(abs_save_dir))

                # On WSL, patch out sudo check before llama.cpp build
                _apply_wsl_sudo_patch()

                # convert_to_gguf writes output relative to cwd (repo root);
                # snapshot existing .gguf so we can diff and relocate afterwards.
                cwd = os.getcwd()
                pre_existing_ggufs = set(glob.glob(os.path.join(cwd, "*.gguf")))

                pre_existing_subs = {
                    d.name for d in Path(abs_save_dir).iterdir() if d.is_dir()
                }

                # Avoid clobbering an existing user-owned model/ directory.
                import uuid

                _model_tmp = os.path.join(
                    abs_save_dir, f"_tmp_model_{uuid.uuid4().hex[:8]}"
                )
                model_tmp_to_cleanup = _model_tmp
                self.current_model.save_pretrained_gguf(
                    _model_tmp,
                    self.current_tokenizer,
                    quantization_method = quant_method,
                    **imatrix_kw,
                )

                # Relocate the .gguf that convert_to_gguf wrote to cwd (repo root).
                new_ggufs = (
                    set(glob.glob(os.path.join(cwd, "*.gguf"))) - pre_existing_ggufs
                )
                for src in sorted(new_ggufs):
                    dest = os.path.join(abs_save_dir, os.path.basename(src))
                    shutil.move(src, dest)
                    logger.info(
                        f"Relocated GGUF: {os.path.basename(src)} → {abs_save_dir}/"
                    )

                # Flatten GGUF files from subdirs created during this export.
                for sub in list(Path(abs_save_dir).iterdir()):
                    if not sub.is_dir():
                        continue
                    if sub.name in pre_existing_subs:
                        continue
                    for src in sub.glob("*.gguf"):
                        dest = os.path.join(abs_save_dir, src.name)
                        shutil.move(str(src), dest)
                        logger.info(f"Relocated GGUF: {src.name} → {abs_save_dir}/")
                    shutil.rmtree(str(sub), ignore_errors = True)
                    logger.info(f"Cleaned up subdirectory: {sub.name}")

                # For non-PEFT models, save_pretrained_gguf leaves a *_gguf dir at
                # the checkpoint path; relocate its GGUFs and clean it up.
                if self.current_checkpoint:
                    ckpt = Path(self.current_checkpoint)
                    gguf_dir = ckpt.parent / f"{ckpt.name}_gguf"
                    if (
                        gguf_dir.is_dir()
                        and gguf_dir.resolve() != Path(abs_save_dir).resolve()
                    ):
                        for src in gguf_dir.glob("*.gguf"):
                            dest = os.path.join(abs_save_dir, src.name)
                            shutil.move(str(src), dest)
                            logger.info(f"Relocated GGUF: {src.name} → {abs_save_dir}/")
                        # Also relocate Ollama Modelfile if present
                        modelfile = gguf_dir / "Modelfile"
                        if modelfile.is_file():
                            shutil.move(
                                str(modelfile), os.path.join(abs_save_dir, "Modelfile")
                            )
                            logger.info(f"Relocated Modelfile → {abs_save_dir}/")
                        shutil.rmtree(str(gguf_dir), ignore_errors = True)
                        logger.info(f"Cleaned up intermediate GGUF dir: {gguf_dir}")

                # Write export metadata so the Chat page can identify the base model
                self._write_export_metadata(abs_save_dir)

                final_ggufs = sorted(glob.glob(os.path.join(abs_save_dir, "*.gguf")))
                logger.info(
                    "GGUF export complete. Final files in %s:\n  %s",
                    abs_save_dir,
                    "\n  ".join(os.path.basename(f) for f in final_ggufs) or "(none)",
                )
                output_path = str(Path(abs_save_dir).resolve())

            if push_to_hub:
                if not repo_id or not hf_token:
                    return (
                        False,
                        "Repository ID and Hugging Face token required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing GGUF model to Hub: {repo_id}")

                self.current_model.push_to_hub_gguf(
                    repo_id,
                    self.current_tokenizer,
                    quantization_method = quant_method,
                    token = hf_token,
                    **imatrix_kw,
                )
                logger.info(f"GGUF model pushed successfully to {repo_id}")

            return (
                True,
                f"GGUF model exported successfully ({', '.join(quant_methods)})",
                output_path,
            )

        except Exception as e:
            if model_tmp_to_cleanup:
                shutil.rmtree(model_tmp_to_cleanup, ignore_errors = True)
            logger.error(f"Error exporting GGUF model: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"GGUF export failed: {str(e)}", None

    def export_lora_adapter(
        self,
        save_directory: str,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
        gguf: bool = False,
        gguf_outtype: str = "q8_0",
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export LoRA adapter only (not merged).

        Args:
            gguf: If True, also convert the adapter to a GGUF LoRA file (llama.cpp
                convert_lora_to_gguf.py), loadable with `llama-cli --lora ...`.
            gguf_outtype: GGUF LoRA output float type; one of q8_0/f16/bf16/f32.

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not _export_runtime_available():
            return False, _export_runtime_message(), None
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        if not self.is_peft:
            return False, "This is not a PEFT model. No adapter to export.", None

        _GGUF_LORA_OUTTYPES = ("q8_0", "f16", "bf16", "f32")
        if gguf:
            if _IS_MLX:
                return (
                    False,
                    "GGUF LoRA adapter export is not supported on macOS/MLX. "
                    "Use the safetensors adapter instead.",
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
            # getattr so an older build without save_pretrained_gguf returns a clean message
            # instead of an AttributeError (a generic 500).
            _save_gguf_fn = getattr(self.current_model, "save_pretrained_gguf", None)
            if _save_gguf_fn is None or not _supports_kwarg(
                _save_gguf_fn, "save_method"
            ):
                return (
                    False,
                    "This Unsloth build does not support GGUF LoRA adapter export. "
                    "Upgrade unsloth and unsloth_zoo, or export the safetensors adapter.",
                    None,
                )

        output_path: Optional[str] = None
        try:
            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving LoRA adapter locally to: {save_directory}")
                ensure_dir(Path(save_directory))

                if gguf:
                    # Writes the adapter files plus "<base>-lora-<outtype>.gguf".
                    _apply_wsl_sudo_patch()
                    self.current_model.save_pretrained_gguf(
                        save_directory,
                        self.current_tokenizer,
                        save_method = "lora",
                        quantization_method = outtype,
                        # Forward the token so convert_lora_to_gguf.py can fetch a gated base's config.
                        token = hf_token or None,
                    )
                    final_ggufs = sorted(
                        glob.glob(os.path.join(save_directory, "*.gguf"))
                    )
                    logger.info(
                        "LoRA GGUF export complete. Files in %s:\n  %s",
                        save_directory,
                        "\n  ".join(os.path.basename(f) for f in final_ggufs)
                        or "(none)",
                    )
                elif _IS_MLX:
                    # MLX: save adapters.safetensors + tokenizer files
                    self.current_model.save_lora_adapters(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)
                else:
                    self.current_model.save_pretrained(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)
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

                if gguf:
                    # Upload the locally-built GGUF folder; needs a local save_directory so the
                    # conversion is not re-run.
                    if not (output_path and Path(output_path).is_dir()):
                        return (
                            False,
                            "GGUF LoRA Hub upload requires a local save directory; set one and "
                            "retry.",
                            None,
                        )
                    hf_api = HfApi(token = hf_token)
                    hf_api.create_repo(repo_id, private = private, exist_ok = True)
                    hf_api.upload_folder(
                        folder_path = output_path,
                        repo_id = repo_id,
                        repo_type = "model",
                    )
                elif _IS_MLX:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        self.current_model.save_lora_adapters(tmp_dir)
                        self.current_tokenizer.save_pretrained(tmp_dir)
                        hf_api = HfApi(token = hf_token)
                        hf_api.create_repo(repo_id, private = private, exist_ok = True)
                        hf_api.upload_folder(
                            folder_path = tmp_dir,
                            repo_id = repo_id,
                            repo_type = "model",
                        )
                else:
                    self.current_model.push_to_hub(
                        repo_id, token = hf_token, private = private
                    )
                    self.current_tokenizer.push_to_hub(
                        repo_id, token = hf_token, private = private
                    )
                logger.info(f"Adapter pushed successfully to {repo_id}")

            return True, "LoRA adapter exported successfully", output_path

        except Exception as e:
            logger.error(f"Error exporting LoRA adapter: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, f"Adapter export failed: {str(e)}", None


# Global export backend instance
_export_backend = None


def get_export_backend() -> ExportBackend:
    """Get or create the global export backend instance"""
    global _export_backend
    if _export_backend is None:
        _export_backend = ExportBackend()
    return _export_backend
