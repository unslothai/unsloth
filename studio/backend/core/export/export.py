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
from pathlib import Path
from typing import Optional, Tuple, List, Union
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


def _normalize_gguf_quant_methods(
    quantization_method: Optional[Union[str, List[str]]] = None,
    quantization_methods: Optional[List[str]] = None,
) -> List[str]:
    """Resolve GGUF quant labels to a lowercase list for unsloth."""
    if quantization_methods:
        raw = quantization_methods
    elif isinstance(quantization_method, list):
        raw = quantization_method
    elif quantization_method:
        raw = [quantization_method]
    else:
        raw = ["Q4_K_M"]

    normalized: List[str] = []
    seen = set()
    for item in raw:
        label = str(item).strip()
        if not label:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(key)
    return normalized or ["q4_k_m"]


def _format_gguf_quant_label(quant_methods: List[str]) -> str:
    return ", ".join(method.upper() for method in quant_methods)


def _precheck_hub_or_fail(
    push_to_hub: bool,
    repo_id: Optional[str],
    hf_token: Optional[str],
    private: bool,
) -> Optional[Tuple[bool, str, None]]:
    """Run Hub preflight when push_to_hub is requested."""
    if not push_to_hub:
        return None
    if not repo_id:
        return False, "Repository ID is required for Hub upload", None
    from core.export.hf_precheck import precheck_hub_upload_tuple

    ok, message = precheck_hub_upload_tuple(
        repo_id = repo_id,
        hf_token = hf_token,
        private = private,
    )
    if not ok:
        return False, message, None
    return None


def _resolve_hub_repo_id(
    repo_id: str,
    hf_token: Optional[str],
    private: bool,
) -> str:
    """Create or resolve a Hub model repo and return the canonical repo id."""
    hf_api = HfApi(token = hf_token) if hf_token else HfApi()
    if _IS_MLX:
        hf_api.create_repo(repo_id, private = private, exist_ok = True, token = hf_token)
        return repo_id
    return PushToHubMixin._create_repo(
        PushToHubMixin,
        repo_id = repo_id,
        private = private,
        token = hf_token,
    )


def _push_hub_model_card(
    repo_id: str,
    hf_token: Optional[str],
    *,
    username: str,
    base_model: str,
    model_type: str,
    method: str = "",
    extra: str = "unsloth",
) -> None:
    content = MODEL_CARD.format(
        username = username,
        base_model = base_model,
        model_type = model_type,
        method = method,
        extra = extra,
    )
    card = ModelCard(content)
    card.push_to_hub(repo_id, token = hf_token, commit_message = "Unsloth Model Card")


def _ensure_full_hub_repo_id(repo_id: str, hf_token: Optional[str]) -> str:
    """Expand a short repo name to username/model when no namespace is given."""
    if "/" in repo_id:
        return repo_id
    hf_api = HfApi(token = hf_token) if hf_token else HfApi()
    username = hf_api.whoami(token = hf_token)["name"]
    return f"{username}/{repo_id}"


def _upload_model_folder_to_hub(
    folder_path: str,
    repo_id: str,
    hf_token: Optional[str],
    private: bool,
    *,
    repo_resolved: bool = False,
) -> None:
    """Upload an on-disk model folder without re-saving or re-merging."""
    full_repo_id = (
        repo_id
        if repo_resolved
        else _resolve_hub_repo_id(repo_id, hf_token, private)
    )
    hf_api = HfApi(token = hf_token) if hf_token else HfApi()
    hf_api.upload_folder(
        folder_path = folder_path,
        repo_id = full_repo_id,
        repo_type = "model",
        token = hf_token,
    )


def _find_export_artifact(save_directory: str, filename: str) -> Optional[str]:
    """Locate an export artifact in the save dir or a leftover _tmp_model_* subdir."""
    direct = os.path.join(save_directory, filename)
    if os.path.isfile(direct):
        return direct
    try:
        for entry in os.scandir(save_directory):
            if entry.is_dir() and entry.name.startswith("_tmp_model_"):
                candidate = os.path.join(entry.path, filename)
                if os.path.isfile(candidate):
                    return candidate
    except OSError:
        pass
    return None


def _gguf_path_in_repo(
    file_path: str,
    *,
    model_name: str,
    save_directory: str,
) -> str:
    """Match unsloth's Hub filename normalization for GGUF uploads."""
    original_name = os.path.basename(file_path)
    if "unsloth_gguf_" in original_name:
        quant_suffix = (
            original_name.split(".", 1)[1] if "." in original_name else original_name
        )
        return f"{model_name}.{quant_suffix}"

    save_base = os.path.basename(save_directory.rstrip("/\\"))
    if save_base and save_base in original_name:
        return original_name.replace(save_base, model_name)
    return original_name


def _build_gguf_hub_readme(
    *,
    repo_id: str,
    upload_names: List[str],
    is_vlm: bool,
    has_modelfile: bool,
) -> str:
    """Build a Hub README aligned with unsloth_push_to_hub_gguf."""
    model_slug = repo_id.split("/")[-1]
    readme_content = f"""---
tags:
- gguf
- llama.cpp
- unsloth
{"- vision-language-model" if is_vlm else ""}
---

# {model_slug} : GGUF

This model was finetuned and converted to GGUF format using [Unsloth](https://github.com/unslothai/unsloth).

**Example usage**:
- For text only LLMs:    `llama-cli -hf {repo_id} --jinja`
- For multimodal models: `llama-mtmd-cli -hf {repo_id} --jinja`

## Available Model files:
"""
    for name in upload_names:
        readme_content += f"- `{name}`\n"

    if is_vlm and has_modelfile:
        readme_content += "\n## ⚠️ Ollama Note for Vision Models\n"
        readme_content += "**Important:** Ollama currently does not support separate mmproj files for vision models.\n\n"
        readme_content += "To create an Ollama model from this vision model:\n"
        readme_content += "1. Place the `Modelfile` in the same directory as the finetuned bf16 merged model\n"
        readme_content += "3. Run: `ollama create model_name -f ./Modelfile`\n"
        readme_content += "   (Replace `model_name` with your desired name)\n\n"
        readme_content += "This will create a unified bf16 model that Ollama can use.\n"
    elif has_modelfile:
        readme_content += "\n## Ollama\n"
        readme_content += "An Ollama Modelfile is included for easy deployment.\n"

    readme_content += (
        "\nThis was trained 2x faster with [Unsloth](https://github.com/unslothai/unsloth)\n"
        '[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)\n'
    )
    return readme_content


def _upload_gguf_directory_to_hub(
    save_directory: str,
    repo_id: str,
    hf_token: Optional[str],
    private: bool = False,
    is_vlm: bool = False,
) -> None:
    """Upload already-converted GGUF artifacts without re-running conversion."""
    gguf_files = sorted(glob.glob(os.path.join(save_directory, "*.gguf")))
    if not gguf_files:
        raise RuntimeError(
            f"No GGUF files found in {save_directory}. Conversion may have failed."
        )

    hf_api = HfApi(token = hf_token) if hf_token else HfApi()
    full_repo_id = _resolve_hub_repo_id(
        _ensure_full_hub_repo_id(repo_id, hf_token),
        hf_token,
        private,
    )
    model_name = full_repo_id.split("/")[-1]

    upload_names: List[str] = []
    for file_path in gguf_files:
        path_in_repo = _gguf_path_in_repo(
            file_path,
            model_name = model_name,
            save_directory = save_directory,
        )
        upload_names.append(path_in_repo)
        logger.info("Uploading GGUF to Hub: %s", path_in_repo)
        hf_api.upload_file(
            path_or_fileobj = file_path,
            path_in_repo = path_in_repo,
            repo_id = full_repo_id,
            repo_type = "model",
            commit_message = "(Trained with Unsloth)",
            token = hf_token,
        )

    config_path = _find_export_artifact(save_directory, "config.json")
    if config_path:
        hf_api.upload_file(
            path_or_fileobj = config_path,
            path_in_repo = "config.json",
            repo_id = full_repo_id,
            repo_type = "model",
            commit_message = "(Trained with Unsloth) - config",
            token = hf_token,
        )

    modelfile_path = _find_export_artifact(save_directory, "Modelfile")
    has_modelfile = modelfile_path is not None
    if modelfile_path:
        hf_api.upload_file(
            path_or_fileobj = modelfile_path,
            path_in_repo = "Modelfile",
            repo_id = full_repo_id,
            repo_type = "model",
            commit_message = "(Trained with Unsloth) - Ollama Modelfile",
            token = hf_token,
        )

    readme_content = _build_gguf_hub_readme(
        repo_id = full_repo_id,
        upload_names = upload_names,
        is_vlm = is_vlm,
        has_modelfile = has_modelfile,
    )

    hf_api.upload_file(
        path_or_fileobj = readme_content.encode("utf-8"),
        path_in_repo = "README.md",
        repo_id = full_repo_id,
        repo_type = "model",
        commit_message = "Add README",
        token = hf_token,
    )

    tags = ["gguf", "llama-cpp", "unsloth"]
    if is_vlm:
        tags.append("vision-language-model")
    try:
        hf_api.add_tags(
            repo_id = full_repo_id,
            tags = tags,
            repo_type = "model",
        )
    except Exception as exc:
        logger.warning("Could not add Hub tags to %s: %s", full_repo_id, exc)


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
    ) -> Tuple[bool, str]:
        """
        Load a checkpoint for export.

        Returns:
            Tuple of (success: bool, message: str)
        """
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

            self._audio_type = detect_audio_type(model_id)
            self.is_vision = not self._audio_type and is_vision_model(model_id)

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
                tokenizer = processor  # vision: processor acts as tokenizer

            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = checkpoint_path,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = load_in_4bit,
                    trust_remote_code = trust_remote_code,
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
        try:
            precheck_failure = _precheck_hub_or_fail(
                push_to_hub, repo_id, hf_token, private
            )
            if precheck_failure is not None:
                return precheck_failure

            if _IS_MLX:
                mlx_save_method = "merged_4bit" if format_type == "4-bit (FP4)" else "merged_16bit"
            else:
                if format_type == "4-bit (FP4)":
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

                self._write_export_metadata(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")
                output_path = str(Path(save_directory).resolve())

            if push_to_hub:
                if not repo_id:
                    return (
                        False,
                        "Repository ID is required for Hub upload",
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
                    if output_path and os.path.isdir(output_path):
                        base_model = (
                            get_base_model_from_lora(self.current_checkpoint)
                            if self.current_checkpoint
                            else None
                        ) or getattr(self.current_model.config, "_name_or_path", None) or "unknown"
                        full_repo_id = _resolve_hub_repo_id(repo_id, hf_token, private)
                        _push_hub_model_card(
                            full_repo_id,
                            hf_token,
                            username = full_repo_id.split("/")[0],
                            base_model = base_model,
                            model_type = self.current_model.config.model_type,
                        )
                        _upload_model_folder_to_hub(
                            output_path,
                            full_repo_id,
                            hf_token,
                            private,
                            repo_resolved = True,
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
            precheck_failure = _precheck_hub_or_fail(
                push_to_hub, repo_id, hf_token, private
            )
            if precheck_failure is not None:
                return precheck_failure

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
                if not repo_id:
                    return (
                        False,
                        "Repository ID is required for Hub upload",
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
                    if output_path and os.path.isdir(output_path):
                        base_model = (
                            base_model_id or self.current_model.config._name_or_path or "unknown"
                        )
                        full_repo_id = _resolve_hub_repo_id(repo_id, hf_token, private)
                        _push_hub_model_card(
                            full_repo_id,
                            hf_token,
                            username = full_repo_id.split("/")[0],
                            base_model = base_model,
                            model_type = self.current_model.config.model_type,
                        )
                        _upload_model_folder_to_hub(
                            output_path,
                            full_repo_id,
                            hf_token,
                            private,
                            repo_resolved = True,
                        )
                        logger.info(f"Model pushed successfully to {full_repo_id}")
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
        quantization_method: Union[str, List[str], None] = "Q4_K_M",
        quantization_methods: Optional[List[str]] = None,
        push_to_hub: bool = False,
        repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        private: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Export model in GGUF format.

        Args:
            save_directory: Local directory to save model
            quantization_method: Single GGUF quant label (legacy)
            quantization_methods: Multiple GGUF quant labels in one conversion pass
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID
            hf_token: Hugging Face token
            private: Whether to create a private Hub repository

        Returns:
            Tuple of (success: bool, message: str, output_path: Optional[str])
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first.", None

        quant_methods = _normalize_gguf_quant_methods(
            quantization_method,
            quantization_methods,
        )
        quant_label = _format_gguf_quant_label(quant_methods)

        output_path: Optional[str] = None
        model_tmp_to_cleanup: Optional[str] = None
        try:
            precheck_failure = _precheck_hub_or_fail(
                push_to_hub, repo_id, hf_token, private
            )
            if precheck_failure is not None:
                return precheck_failure

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
                logger.info(
                    "Saving GGUF model locally to: %s (quantizations: %s)",
                    abs_save_dir,
                    quant_label,
                )

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
                    quantization_method = quant_methods,
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
                if not repo_id:
                    return (
                        False,
                        "Repository ID is required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing GGUF model to Hub: {repo_id}")

                if output_path and os.path.isdir(output_path):
                    _upload_gguf_directory_to_hub(
                        output_path,
                        repo_id,
                        hf_token,
                        private = private,
                        is_vlm = self.is_vision,
                    )
                else:
                    self.current_model.push_to_hub_gguf(
                        repo_id,
                        self.current_tokenizer,
                        quantization_method = quant_methods,
                        token = hf_token,
                        private = private,
                    )
                logger.info(f"GGUF model pushed successfully to {repo_id}")

            return (
                True,
                f"GGUF model exported successfully ({quant_label})",
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
            precheck_failure = _precheck_hub_or_fail(
                push_to_hub, repo_id, hf_token, private
            )
            if precheck_failure is not None:
                return precheck_failure

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
                if not repo_id:
                    return (
                        False,
                        "Repository ID is required for Hub upload",
                        None,
                    )

                logger.info(f"Pushing LoRA adapter to Hub: {repo_id}")

                if output_path and os.path.isdir(output_path):
                    full_repo_id = _resolve_hub_repo_id(repo_id, hf_token, private)
                    base_model = (
                        get_base_model_from_lora(self.current_checkpoint)
                        if self.current_checkpoint
                        else None
                    ) or getattr(self.current_model.config, "_name_or_path", None) or "unknown"
                    _push_hub_model_card(
                        full_repo_id,
                        hf_token,
                        username = full_repo_id.split("/")[0],
                        base_model = base_model,
                        model_type = getattr(self.current_model.config, "model_type", "unknown"),
                        method = "LoRA adapter",
                        extra = "lora",
                    )
                    _upload_model_folder_to_hub(
                        output_path,
                        full_repo_id,
                        hf_token,
                        private,
                        repo_resolved = True,
                    )
                elif _IS_MLX:
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        self.current_model.save_lora_adapters(tmp_dir)
                        self.current_tokenizer.save_pretrained(tmp_dir)
                        full_repo_id = _resolve_hub_repo_id(repo_id, hf_token, private)
                        base_model = (
                            get_base_model_from_lora(self.current_checkpoint)
                            if self.current_checkpoint
                            else None
                        ) or getattr(self.current_model.config, "_name_or_path", None) or "unknown"
                        _push_hub_model_card(
                            full_repo_id,
                            hf_token,
                            username = full_repo_id.split("/")[0],
                            base_model = base_model,
                            model_type = getattr(self.current_model.config, "model_type", "unknown"),
                            method = "LoRA adapter",
                            extra = "lora",
                        )
                        _upload_model_folder_to_hub(
                            tmp_dir,
                            full_repo_id,
                            hf_token,
                            private,
                            repo_resolved = True,
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
