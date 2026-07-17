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
from unsloth import FastLanguageModel, FastVisionModel, _IS_MLX
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

# GPU-only imports — guarded for Apple Silicon where these aren't needed
if not _IS_MLX:
    from peft import PeftModel, PeftModelForCausalLM
    from transformers.modeling_utils import PushToHubMixin
    import torch

logger = get_logger(__name__)

_LLAMA_CPP_SCRIPTS_WARNING_EMITTED = False


def _multi_gpu_device_map_kwargs() -> dict:
    """``device_map`` kwargs for sharding a checkpoint across every visible GPU.

    unsloth's ``from_pretrained`` defaults to ``device_map="sequential"``, which
    stacks the whole model on GPU0 and OOMs multi-GPU hosts whose other GPUs sit
    empty (#7053). Return ``{"device_map": "balanced"}`` only when the CUDA/ROCm
    host actually exposes more than one GPU (mirrors the inference loader's
    ``get_device_map`` policy); empty otherwise -- single-GPU, CPU, and MLX
    loads keep the loader default untouched."""
    if _IS_MLX:
        return {}
    try:
        from utils.hardware import get_device_map, get_parent_visible_gpu_ids
        visible = get_parent_visible_gpu_ids()
        if len(visible) > 1:
            device_map = get_device_map(visible)
            if device_map == "balanced":
                return {"device_map": device_map}
    except Exception as exc:
        logger.debug(f"multi-GPU device_map resolution failed; using loader default: {exc}")
    return {}


def _supports_kwarg(fn, name):
    """True if `fn` accepts keyword `name` directly or via **kwargs."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def _compressed_export_supported():
    """True if the installed unsloth build can do FP8/NVFP4 compressed-tensors export."""
    try:
        import unsloth.save as _us
        return hasattr(_us, "_normalize_compressed_method")
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
        return False  # probe disabled -> assume online; loads still pass local_files_only on env

    # Shared bounded, proxy-aware probe (also used by the export worker before version activation).
    from utils.transformers_version import hf_endpoint_unreachable

    if hf_endpoint_unreachable(timeout):
        logger.warning("Hugging Face endpoint unreachable; loading checkpoint in offline mode")
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
            logger.info("WSL detected — skipping sudo check (build deps pre-installed by setup.sh)")
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

            # Shard across every visible GPU on a multi-GPU host instead of
            # stacking the checkpoint on GPU0 (#7053); {} on single-GPU/CPU/MLX.
            _device_map_kw = _multi_gpu_device_map_kwargs()

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
                    load_in_4bit = load_in_4bit,
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
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                    token = token,
                    local_files_only = local_files_only,
                    **_device_map_kw,
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
                    **_device_map_kw,
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
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export merged model (for PEFT models).

        Args:
            save_directory: Local directory to save model
            format_type: "16-bit (FP16)" or "4-bit (FP4)"
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID (username/model-name)
            hf_token: Hugging Face token
            private: Whether to make the repo private

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        if not self.is_peft:
            return (
                False,
                "This is not a PEFT model. Use 'Export Base Model' instead.",
                None,
            )

        output_path: Optional[str] = None
        # compressed-tensors formats run save_pretrained_merged with an FP8/FP4 save_method and
        # write to a sibling "<dir>-<suffix>" directory (for vLLM).
        _COMPRESSED = {
            "FP8 (compressed-tensors)": ("fp8", "fp8"),
            "NVFP4 (compressed-tensors)": ("nvfp4", "nvfp4"),
        }
        is_compressed = format_type in _COMPRESSED
        try:
            if _IS_MLX:
                if is_compressed:
                    return False, "Compressed-tensors export is not supported on macOS/MLX.", None
                mlx_save_method = "merged_4bit" if format_type == "4-bit (FP4)" else "merged_16bit"
            elif is_compressed:
                if not _compressed_export_supported():
                    return (
                        False,
                        "Compressed-tensors (FP8/NVFP4) export requires an Unsloth build with "
                        "compressed-tensors support. Upgrade unsloth, or choose 16-bit.",
                        None,
                    )
                save_method = _COMPRESSED[format_type][0]
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

                # Compressed export writes to the "<dir>-<suffix>" sibling; report that as output.
                final_dir = (
                    f"{save_directory}-{_COMPRESSED[format_type][1]}"
                    if is_compressed
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
                elif is_compressed and output_path and Path(output_path).is_dir():
                    # The compressed model was already built locally in output_path; upload it
                    # directly so we do not re-run the (expensive, OOM-prone) compression that
                    # push_to_hub_merged(save_method=fp8/nvfp4) would otherwise do a second time.
                    hf_api = HfApi(token = hf_token)
                    repo_id = PushToHubMixin._create_repo(
                        PushToHubMixin,
                        repo_id = repo_id,
                        private = private,
                        token = hf_token,
                    )
                    content = MODEL_CARD.format(
                        username = repo_id.split("/")[0],
                        base_model = getattr(self.current_model.config, "_name_or_path", "unknown"),
                        model_type = getattr(self.current_model.config, "model_type", "llm"),
                        method = format_type,
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
                    hub_save_method = save_method if save_method is not None else "merged_16bit"
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
                        base_model_id or self.current_model.config._name_or_path or "unknown"
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
                    card.push_to_hub(repo_id, token = hf_token, commit_message = "Unsloth Model Card")

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
        quantization_method: str = "Q4_K_M",
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        imatrix_file = None,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export model in GGUF format.

        Args:
            save_directory: Local directory to save model
            quantization_method: GGUF quantization method (e.g., "Q4_K_M")
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID
            hf_token: Hugging Face token

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        # Only forward imatrix_file to an unsloth build that accepts it; otherwise even a plain
        # no-imatrix export would fail with an unexpected-keyword error against an older unsloth.
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
            # unsloth expects lowercase quant method
            quant_method = quantization_method.lower()

            # Pin convert_hf_to_gguf.py to setup.sh's tagged llama.cpp ref so it
            # can't drift past the pinned llama-quantize binary's gguf API.
            global _LLAMA_CPP_SCRIPTS_WARNING_EMITTED
            try:
                from unsloth_zoo.llama_cpp import (
                    LLAMA_CPP_DEFAULT_DIR,
                    _resolve_local_convert_script,  # noqa: F401
                )
                os.environ.setdefault("UNSLOTH_LLAMA_CPP_SCRIPTS_DIR", LLAMA_CPP_DEFAULT_DIR)
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

                pre_existing_subs = {d.name for d in Path(abs_save_dir).iterdir() if d.is_dir()}

                # Avoid clobbering an existing user-owned model/ directory.
                import uuid

                _model_tmp = os.path.join(abs_save_dir, f"_tmp_model_{uuid.uuid4().hex[:8]}")
                model_tmp_to_cleanup = _model_tmp
                self.current_model.save_pretrained_gguf(
                    _model_tmp,
                    self.current_tokenizer,
                    quantization_method = quant_method,
                    **imatrix_kw,
                )

                # Relocate the .gguf that convert_to_gguf wrote to cwd (repo root).
                new_ggufs = set(glob.glob(os.path.join(cwd, "*.gguf"))) - pre_existing_ggufs
                for src in sorted(new_ggufs):
                    dest = os.path.join(abs_save_dir, os.path.basename(src))
                    shutil.move(src, dest)
                    logger.info(f"Relocated GGUF: {os.path.basename(src)} → {abs_save_dir}/")

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
                    if gguf_dir.is_dir() and gguf_dir.resolve() != Path(abs_save_dir).resolve():
                        for src in gguf_dir.glob("*.gguf"):
                            dest = os.path.join(abs_save_dir, src.name)
                            shutil.move(str(src), dest)
                            logger.info(f"Relocated GGUF: {src.name} → {abs_save_dir}/")
                        # Also relocate Ollama Modelfile if present
                        modelfile = gguf_dir / "Modelfile"
                        if modelfile.is_file():
                            shutil.move(str(modelfile), os.path.join(abs_save_dir, "Modelfile"))
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
                f"GGUF model exported successfully ({quantization_method})",
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
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export LoRA adapter only (not merged).

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        if not self.is_peft:
            return False, "This is not a PEFT model. No adapter to export.", None

        output_path: Optional[str] = None
        try:
            if save_directory:
                save_directory = str(resolve_export_write_dir(save_directory))
                logger.info(f"Saving LoRA adapter locally to: {save_directory}")
                ensure_dir(Path(save_directory))

                if _IS_MLX:
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

                if _IS_MLX:
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
                    self.current_model.push_to_hub(repo_id, token = hf_token, private = private)
                    self.current_tokenizer.push_to_hub(repo_id, token = hf_token, private = private)
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
