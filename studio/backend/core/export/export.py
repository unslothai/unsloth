# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

# backend/export.py
"""
Export backend - handles model exporting in various formats
"""

import glob
import json
import structlog
import tempfile
from loggers import get_logger
import os
import shutil
from pathlib import Path
from typing import Optional, Tuple, List
from unsloth import FastLanguageModel, FastVisionModel, _IS_MLX
from huggingface_hub import HfApi, ModelCard
from utils.hardware import clear_gpu_cache

from utils.models import is_vision_model, get_base_model_from_lora
from utils.models.model_config import detect_audio_type
from utils.paths import ensure_dir, outputs_root, resolve_export_dir, resolve_output_dir
from core.inference import get_inference_backend

# GPU-only imports — guarded for Apple Silicon where these aren't needed
if not _IS_MLX:
    from peft import PeftModel, PeftModelForCausalLM
    from transformers.modeling_utils import PushToHubMixin
    import torch

logger = get_logger(__name__)

_LLAMA_CPP_SCRIPTS_WARNING_EMITTED = False


def _is_wsl():
    """Detect if running under Windows Subsystem for Linux."""
    try:
        return "microsoft" in open("/proc/version").read().lower()
    except Exception:
        return False


def _apply_wsl_sudo_patch():
    """On WSL, monkey-patch do_we_need_sudo() to return False.

    WSL doesn't have passwordless sudo, and do_we_need_sudo() runs
    `sudo apt-get update` which hangs waiting for a stdin password
    inside a non-interactive subprocess. setup.sh pre-installs the
    build dependencies on WSL, so sudo is not needed at runtime.
    """
    if not _is_wsl():
        return

    try:
        import unsloth_zoo.llama_cpp as llama_cpp_module

        def _wsl_do_we_need_sudo(system_type = "debian"):
            logger.info(
                "WSL detected — skipping sudo check "
                "(build deps pre-installed by setup.sh)"
            )
            return False

        llama_cpp_module.do_we_need_sudo = _wsl_do_we_need_sudo
        logger.info(
            "Applied WSL sudo patch to " "unsloth_zoo.llama_cpp.do_we_need_sudo"
        )
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

            # Unload all models from inference backend
            model_names = list(self.inference_backend.models.keys())
            for model_name in model_names:
                self.inference_backend.unload_model(model_name)

            # Clear current export state
            self.current_model = None
            self.current_tokenizer = None
            self.current_checkpoint = None
            self._audio_type = None

            # Clear GPU memory cache (handles gc + backend-specific cleanup)
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

        Returns:
            List of tuples: [(model_name, [(display_name, checkpoint_path), ...]), ...]
        """
        from utils.models.checkpoints import scan_checkpoints

        return scan_checkpoints(outputs_dir = outputs_dir)

    def load_checkpoint(
        self,
        checkpoint_path: str,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
        trust_remote_code: bool = False,
    ) -> Tuple[bool, str]:
        """
        Load a checkpoint for export.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            # First, cleanup existing models
            self.cleanup_memory()

            checkpoint_path_obj = Path(checkpoint_path)

            # Determine the model identity for type detection
            adapter_config = checkpoint_path_obj / "adapter_config.json"
            base_model = None
            if adapter_config.exists():
                base_model = get_base_model_from_lora(checkpoint_path)
                if not base_model:
                    return False, "Could not determine base model for adapter"

            model_id = base_model or checkpoint_path

            # Detect audio type and vision
            self._audio_type = detect_audio_type(model_id)
            self.is_vision = not self._audio_type and is_vision_model(model_id)

            # Load model based on type
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
                )

            elif self._audio_type == "snac":
                logger.info("Loading as SNAC (Orpheus) audio model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
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
                )

            elif self._audio_type == "dac":
                from unsloth import FastModel

                logger.info("Loading as DAC (OuteTTS) audio model...")
                model, tokenizer = FastModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    load_in_4bit = False,
                    trust_remote_code = trust_remote_code,
                )

            elif self.is_vision:
                logger.info("Loading as vision model...")
                model, processor = FastVisionModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                )
                tokenizer = processor  # For vision models, processor acts as tokenizer

            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
                )

            # Check if PEFT / LoRA model
            if _IS_MLX:
                # MLX doesn't use PeftModel — detect LoRA via adapter_config.json
                self.is_peft = adapter_config.exists()
            else:
                self.is_peft = isinstance(model, (PeftModel, PeftModelForCausalLM))

            # Store loaded model
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
        try:
            if _IS_MLX:
                mlx_save_method = (
                    "merged_4bit" if format_type == "4-bit (FP4)" else "merged_16bit"
                )
            else:
                if format_type == "4-bit (FP4)":
                    save_method = "merged_4bit_forced"
                elif self._audio_type == "whisper":
                    save_method = None
                else:
                    save_method = "merged_16bit"

            # Save locally if requested
            if save_directory:
                save_directory = str(resolve_export_dir(save_directory))
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

                self._write_export_metadata(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            # Push to hub if requested
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
            # Save locally if requested
            if save_directory:
                save_directory = str(resolve_export_dir(save_directory))
                logger.info(f"Saving base model locally to: {save_directory}")
                ensure_dir(Path(save_directory))

                if _IS_MLX:
                    # MLX: save_pretrained_merged handles non-LoRA models too
                    # (fuse() is a no-op when there are no LoRA layers)
                    self.current_model.save_pretrained_merged(
                        save_directory,
                        self.current_tokenizer,
                    )
                else:
                    self.current_model.save_pretrained(save_directory)
                    self.current_tokenizer.save_pretrained(save_directory)

                # Write export metadata so the Chat page can identify the base model
                self._write_export_metadata(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            # Push to hub if requested
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
                            )
                            self.current_model.push_to_hub_merged(
                                repo_id,
                                self.current_tokenizer,
                                save_directory = tmp_dir,
                                token = hf_token,
                                private = private,
                            )
                else:
                    # Get base model name from request or model config
                    base_model = (
                        base_model_id
                        or self.current_model.config._name_or_path
                        or "unknown"
                    )

                    # Create repo
                    hf_api = HfApi(token = hf_token)
                    repo_id = PushToHubMixin._create_repo(
                        PushToHubMixin,
                        repo_id = repo_id,
                        private = private,
                        token = hf_token,
                    )
                    username = repo_id.split("/")[0]

                    # Create and push model card
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

                    # Upload model files
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

        output_path: Optional[str] = None
        try:
            # Convert quantization method to lowercase for unsloth
            quant_method = quantization_method.lower()

            # Pin convert_hf_to_gguf.py to the same llama.cpp ref as the
            # llama-quantize binary (Studio installs at a tagged ref via
            # setup.sh) so it can't drift past the pinned binary's gguf API.
            # Set before both branches; hub-only export has save_directory == "".
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

            # Save locally if requested
            if save_directory:
                save_directory = str(resolve_export_dir(save_directory))
                # Resolve to absolute path so unsloth's relative-path internals
                # (check_llama_cpp, use_local_gguf, _download_convert_hf_to_gguf)
                # all resolve against the repo root cwd, NOT the export directory.
                abs_save_dir = os.path.abspath(save_directory)
                logger.info(f"Saving GGUF model locally to: {abs_save_dir}")

                # Create the directory if it doesn't exist
                ensure_dir(Path(abs_save_dir))

                # On WSL, patch out sudo check before llama.cpp build
                _apply_wsl_sudo_patch()

                # Snapshot existing .gguf files in cwd before conversion.
                # unsloth's convert_to_gguf writes output files relative to
                # cwd (repo root), so we diff afterwards and relocate them.
                cwd = os.getcwd()
                pre_existing_ggufs = set(glob.glob(os.path.join(cwd, "*.gguf")))

                # Pass absolute path — no os.chdir needed.
                # unsloth saves intermediate HF model files into model_save_path.
                # unsloth-zoo's check_llama_cpp() uses ~/.unsloth/llama.cpp by default.
                model_save_path = os.path.join(abs_save_dir, "model")
                self.current_model.save_pretrained_gguf(
                    model_save_path,
                    self.current_tokenizer,
                    quantization_method = quant_method,
                )

                # Relocate GGUF artifacts into the export directory.
                # convert_to_gguf writes .gguf files to cwd (repo root)
                # because --outfile is a relative path like "model.Q4_K_M.gguf".
                new_ggufs = (
                    set(glob.glob(os.path.join(cwd, "*.gguf"))) - pre_existing_ggufs
                )
                for src in sorted(new_ggufs):
                    dest = os.path.join(abs_save_dir, os.path.basename(src))
                    shutil.move(src, dest)
                    logger.info(
                        f"Relocated GGUF: {os.path.basename(src)} → {abs_save_dir}/"
                    )

                # Flatten any .gguf files from subdirectories into abs_save_dir.
                # save_pretrained_gguf may create subdirs (e.g. model_gguf/)
                # with a name different from model_save_path.
                for sub in list(Path(abs_save_dir).iterdir()):
                    if not sub.is_dir():
                        continue
                    for src in sub.glob("*.gguf"):
                        dest = os.path.join(abs_save_dir, src.name)
                        shutil.move(str(src), dest)
                        logger.info(f"Relocated GGUF: {src.name} → {abs_save_dir}/")
                    # Clean up the subdirectory (intermediate HF files, etc.)
                    shutil.rmtree(str(sub), ignore_errors = True)
                    logger.info(f"Cleaned up subdirectory: {sub.name}")

                # For non-PEFT models, save_pretrained_gguf redirects to the
                # checkpoint path, leaving a *_gguf directory in outputs/.
                # Relocate any GGUFs from there and clean it up.
                if self.current_checkpoint:
                    ckpt = Path(self.current_checkpoint)
                    gguf_dir = ckpt.parent / f"{ckpt.name}_gguf"
                    if gguf_dir.is_dir():
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

                # Log final file locations (after relocation) so it's clear
                # where the GGUF files actually ended up.
                final_ggufs = sorted(glob.glob(os.path.join(abs_save_dir, "*.gguf")))
                logger.info(
                    "GGUF export complete. Final files in %s:\n  %s",
                    abs_save_dir,
                    "\n  ".join(os.path.basename(f) for f in final_ggufs) or "(none)",
                )
                output_path = str(Path(abs_save_dir).resolve())

            # Push to hub if requested
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
                )
                logger.info(f"GGUF model pushed successfully to {repo_id}")

            return (
                True,
                f"GGUF model exported successfully ({quantization_method})",
                output_path,
            )

        except Exception as e:
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
            # Save locally if requested
            if save_directory:
                save_directory = str(resolve_export_dir(save_directory))
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

            # Push to hub if requested
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
