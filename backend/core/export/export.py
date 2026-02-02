# backend/export.py
"""
Export backend - handles model exporting in various formats
"""
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, List
from peft import PeftModel, PeftModelForCausalLM
from unsloth import FastLanguageModel, FastVisionModel
from huggingface_hub import HfApi, ModelCard
from transformers.modeling_utils import PushToHubMixin
import torch

from utils.models import is_vision_model, get_base_model_from_lora
from core.inference import get_inference_backend

logger = logging.getLogger(__name__)

# Model card template
MODEL_CARD = \
"""---
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

            # Force garbage collection
            import gc
            gc.collect()

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

            logger.info("Memory cleanup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error during memory cleanup: {e}")
            return False

    def scan_checkpoints(self, outputs_dir: str = "./outputs") -> List[Tuple[str, str]]:
        """
        Scan outputs folder for model checkpoints.

        Returns:
            List of tuples: [(display_name, checkpoint_path), ...]
        """
        checkpoints = []
        outputs_path = Path(outputs_dir)

        if not outputs_path.exists():
            logger.warning(f"Outputs directory not found: {outputs_dir}")
            return checkpoints

        try:
            for item in outputs_path.iterdir():
                if item.is_dir():
                    # Check if this directory contains a model
                    config_file = item / "config.json"
                    adapter_config = item / "adapter_config.json"

                    if config_file.exists() or adapter_config.exists():
                        # This is a valid checkpoint
                        display_name = item.name
                        checkpoint_path = str(item)
                        checkpoints.append((display_name, checkpoint_path))
                        logger.debug(f"Found checkpoint: {display_name}")

            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: Path(x[1]).stat().st_mtime, reverse=True)

            logger.info(f"Found {len(checkpoints)} checkpoints in {outputs_dir}")
            return checkpoints

        except Exception as e:
            logger.error(f"Error scanning checkpoints: {e}")
            return []

    def load_checkpoint(self,
                       checkpoint_path: str,
                       max_seq_length: int = 2048,
                       load_in_4bit: bool = True) -> Tuple[bool, str]:
        """
        Load a checkpoint for export.

        Returns:
            Tuple of (success: bool, message: str)
        """
        try:
            logger.info(f"Loading checkpoint: {checkpoint_path}")

            # First, cleanup existing models
            self.cleanup_memory()

            # Detect if vision model
            checkpoint_path_obj = Path(checkpoint_path)

            # Check if it's a LoRA adapter
            adapter_config = checkpoint_path_obj / "adapter_config.json"
            if adapter_config.exists():
                # It's a LoRA - get base model to check vision
                base_model = get_base_model_from_lora(checkpoint_path)
                if base_model:
                    self.is_vision = is_vision_model(base_model)
                else:
                    return False, "Could not determine base model for adapter"
            else:
                # Check the model itself
                self.is_vision = is_vision_model(checkpoint_path)

            # Load model based on type
            if self.is_vision:
                logger.info("Loading as vision model...")
                model, processor = FastVisionModel.from_pretrained(
                    model_name=checkpoint_path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                )
                tokenizer = processor  # For vision models, processor acts as tokenizer
            else:
                logger.info("Loading as text model...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=checkpoint_path,
                    max_seq_length=max_seq_length,
                    dtype=None,
                    load_in_4bit=load_in_4bit,
                )

            # Check if PEFT model
            self.is_peft = isinstance(model, (PeftModel, PeftModelForCausalLM))

            # Store loaded model
            self.current_model = model
            self.current_tokenizer = tokenizer
            self.current_checkpoint = checkpoint_path

            model_type = "Vision" if self.is_vision else "Text"
            peft_info = " (PEFT Adapter)" if self.is_peft else " (Merged Model)"

            logger.info(f"Successfully loaded {model_type} model{peft_info}")
            return True, f"Loaded {model_type} model{peft_info} successfully"

        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"Failed to load checkpoint: {str(e)}"

    def export_merged_model(self,
                           save_directory: str,
                           format_type: str = "16-bit (FP16)",
                           push_to_hub: bool = False,
                           repo_id: Optional[str] = None,
                           hf_token: Optional[str] = None,
                           private: bool = False) -> Tuple[bool, str]:
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
            Tuple of (success: bool, message: str)
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first."

        if not self.is_peft:
            return False, "This is not a PEFT model. Use 'Export Base Model' instead."

        try:
            # Determine save method
            if format_type == "4-bit (FP4)":
                save_method = "merged_4bit_forced"
            else:  # 16-bit (FP16)
                save_method = "merged_16bit"

            # Save locally if requested
            if save_directory:
                logger.info(f"Saving merged model locally to: {save_directory}")
                os.makedirs(save_directory, exist_ok=True)

                self.current_model.save_pretrained_merged(
                    save_directory,
                    self.current_tokenizer,
                    save_method=save_method
                )
                logger.info(f"Model saved successfully to {save_directory}")

            # Push to hub if requested
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload"

                logger.info(f"Pushing merged model to Hub: {repo_id}")

                self.current_model.push_to_hub_merged(
                    repo_id,
                    self.current_tokenizer,
                    save_method=save_method,
                    token=hf_token,
                    private=private
                )
                logger.info(f"Model pushed successfully to {repo_id}")

            return True, "Model exported successfully"

        except Exception as e:
            logger.error(f"Error exporting merged model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"Export failed: {str(e)}"

    def export_base_model(self,
                         save_directory: str,
                         push_to_hub: bool = False,
                         repo_id: Optional[str] = None,
                         hf_token: Optional[str] = None,
                         private: bool = False) -> Tuple[bool, str]:
        """
        Export base model (for non-PEFT models).

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first."

        if self.is_peft:
            return False, "This is a PEFT model. Use 'Merged Model' export type instead."

        try:
            # Save locally if requested
            if save_directory:
                logger.info(f"Saving base model locally to: {save_directory}")
                os.makedirs(save_directory, exist_ok=True)

                self.current_model.save_pretrained(save_directory)
                self.current_tokenizer.save_pretrained(save_directory)
                logger.info(f"Model saved successfully to {save_directory}")

            # Push to hub if requested
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload"

                logger.info(f"Pushing base model to Hub: {repo_id}")

                # Get base model name
                base_model = self.current_model.config._name_or_path

                # Create repo
                hf_api = HfApi(token=hf_token)
                repo_id = PushToHubMixin._create_repo(
                    PushToHubMixin,
                    repo_id=repo_id,
                    private=private,
                    token=hf_token,
                )
                username = repo_id.split("/")[0]

                # Create and push model card
                content = MODEL_CARD.format(
                    username=username,
                    base_model=base_model,
                    model_type=self.current_model.config.model_type,
                    method="",
                    extra="unsloth",
                )
                card = ModelCard(content)
                card.push_to_hub(repo_id, token=hf_token, commit_message="Unsloth Model Card")

                # Upload model files
                if save_directory:
                    hf_api.upload_folder(
                        folder_path=save_directory,
                        repo_id=repo_id,
                        repo_type="model"
                    )
                    logger.info(f"Model pushed successfully to {repo_id}")
                else:
                    return False, "Local save directory required for Hub upload"

            return True, "Model exported successfully"

        except Exception as e:
            logger.error(f"Error exporting base model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"Export failed: {str(e)}"


    def export_gguf(self,
                    save_directory: str,
                    quantization_method: str = "Q4_K_M",
                    push_to_hub: bool = False,
                    repo_id: Optional[str] = None,
                    hf_token: Optional[str] = None) -> Tuple[bool, str]:
        """
        Export model in GGUF format.

        Args:
            save_directory: Local directory to save model
            quantization_method: GGUF quantization method (e.g., "Q4_K_M")
            push_to_hub: Whether to push to Hugging Face Hub
            repo_id: Hub repository ID
            hf_token: Hugging Face token

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first."

        try:
            # Convert quantization method to lowercase for unsloth
            quant_method = quantization_method.lower()

            # Save locally if requested
            if save_directory:
                logger.info(f"Saving GGUF model locally to: {save_directory}")

                # Create the directory if it doesn't exist
                os.makedirs(save_directory, exist_ok=True)

                # Get the base filename for the GGUF file
                import shutil
                original_dir = os.getcwd()

                try:
                    # Change to target directory
                    os.chdir(save_directory)
                    logger.info(f"Changed directory to: {save_directory}")

                    # Now save (will save in current directory)
                    self.current_model.save_pretrained_gguf(
                        "model",  # Base filename
                        self.current_tokenizer,
                        quantization_method=quant_method
                    )

                    logger.info(f"GGUF model saved successfully in {save_directory}")

                    # Check if llama.cpp directory was created here
                    llama_cpp_in_target = os.path.join(save_directory, "llama.cpp")
                    llama_cpp_in_original = os.path.join(original_dir, "llama.cpp")

                    if os.path.exists(llama_cpp_in_target):
                        logger.info(f"Found llama.cpp directory in {save_directory}")

                        # Remove llama.cpp from original directory if it exists
                        if os.path.exists(llama_cpp_in_original):
                            logger.info(f"Removing existing llama.cpp in {original_dir}")
                            shutil.rmtree(llama_cpp_in_original)

                        # Move llama.cpp back to original directory
                        logger.info(f"Moving llama.cpp to {original_dir}")
                        shutil.move(llama_cpp_in_target, llama_cpp_in_original)
                        logger.info(f"Successfully moved llama.cpp back to original directory")

                finally:
                    # Always change back to original directory
                    os.chdir(original_dir)
                    logger.info(f"Changed back to original directory: {original_dir}")

            # Push to hub if requested
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload"

                logger.info(f"Pushing GGUF model to Hub: {repo_id}")

                self.current_model.push_to_hub_gguf(
                    repo_id,
                    self.current_tokenizer,
                    quantization_method=quant_method,
                    token=hf_token
                )
                logger.info(f"GGUF model pushed successfully to {repo_id}")

            return True, f"GGUF model exported successfully ({quantization_method})"

        except Exception as e:
            logger.error(f"Error exporting GGUF model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"GGUF export failed: {str(e)}"

    def export_lora_adapter(self,
                           save_directory: str,
                           push_to_hub: bool = False,
                           repo_id: Optional[str] = None,
                           hf_token: Optional[str] = None,
                           private: bool = False) -> Tuple[bool, str]:
        """
        Export LoRA adapter only (not merged).

        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.current_model or not self.current_tokenizer:
            return False, "No model loaded. Please select a checkpoint first."

        if not self.is_peft:
            return False, "This is not a PEFT model. No adapter to export."

        try:
            # Save locally if requested
            if save_directory:
                logger.info(f"Saving LoRA adapter locally to: {save_directory}")
                os.makedirs(save_directory, exist_ok=True)

                self.current_model.save_pretrained(save_directory)
                self.current_tokenizer.save_pretrained(save_directory)
                logger.info(f"Adapter saved successfully to {save_directory}")

            # Push to hub if requested
            if push_to_hub:
                if not repo_id or not hf_token:
                    return False, "Repository ID and Hugging Face token required for Hub upload"

                logger.info(f"Pushing LoRA adapter to Hub: {repo_id}")

                self.current_model.push_to_hub(
                    repo_id,
                    token=hf_token,
                    private=private
                )
                self.current_tokenizer.push_to_hub(
                    repo_id,
                    token=hf_token,
                    private=private
                )
                logger.info(f"Adapter pushed successfully to {repo_id}")

            return True, "LoRA adapter exported successfully"

        except Exception as e:
            logger.error(f"Error exporting LoRA adapter: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, f"Adapter export failed: {str(e)}"


# Global export backend instance
_export_backend = None

def get_export_backend() -> ExportBackend:
    """Get or create the global export backend instance"""
    global _export_backend
    if _export_backend is None:
        _export_backend = ExportBackend()
    return _export_backend
