"""
Core inference backend - streamlined
"""
from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer
from peft import PeftModel, PeftModelForCausalLM

import sys
import torch
from typing import Optional, Generator, Tuple
from .model_config import ModelConfig, get_base_model_from_lora
from .path_utils import is_model_cached
from .utils import format_error_message, log_gpu_memory
from io import StringIO
import logging



logger = logging.getLogger(__name__)

class InferenceBackend:
    """Unified inference backend supporting text, vision, and LoRA models"""

    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []  # [(display_name, path), ...]
        self.default_models = [
            "unsloth/Qwen3-4B-Instruct-2507",
            "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
            "unsloth/Phi-3.5-mini-instruct",
            "unsloth/Gemma-3-4B-it",
            "unsloth/Qwen2-VL-2B-Instruct-bnb-4bit",
        ]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Thread safety
        import threading
        self._generation_lock = threading.RLock()
        self._model_state_lock = threading.Lock()

        logger.info(f"InferenceBackend initialized on {self.device}")

    def load_model(self,
                   config: ModelConfig,
                   max_seq_length: int = 2048,
                   dtype = None,
                   load_in_4bit: bool = True,
                   hf_token: Optional[str] = None) -> bool:
        """
        Load any model: base, LoRA adapter, text, or vision.
        """
        try:
            model_name = config.identifier

            # Check if already loaded
            if model_name in self.models and self.models[model_name].get("model"):
                logger.info(f"Model {model_name} already loaded")
                self.active_model_name = model_name
                return True

            # Check if currently loading
            if model_name in self.loading_models:
                logger.info(f"Model {model_name} is already being loaded")
                return False

            self.loading_models.add(model_name)

            self.models[model_name] = {
                "is_vision": config.is_vision,
                "is_lora": config.is_lora,
                "model_path": config.path,
                "base_model": config.base_model if config.is_lora else None,
                "loaded_adapters": {},
                "active_adapter": None,
            }

            model_type = "vision" if config.is_vision else "text"
            adapter_info = " (LoRA adapter)" if self.models[model_name]["is_lora"] else ""
            logger.info(f"Loading {model_type} model{adapter_info}: {model_name}")
            log_gpu_memory(f"Before loading {model_name}")

            # Load model - same approach for base models and LoRA adapters
            if config.is_vision:
                # Vision model (or vision LoRA adapter)
                model, processor = FastVisionModel.from_pretrained(
                    model_name=config.path,  # Can be base model OR LoRA adapter path
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    token=hf_token if hf_token and hf_token.strip() else None,
                )

                # Apply inference optimization
                FastVisionModel.for_inference(model)

                self.models[model_name]["model"] = model
                self.models[model_name]["tokenizer"] = processor
                self.models[model_name]["processor"] = processor

            else:
                # Text model (or text LoRA adapter)
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=config.path,  # Can be base model OR LoRA adapter path
                    max_seq_length=max_seq_length,
                    dtype=dtype,
                    load_in_4bit=load_in_4bit,
                    token=hf_token if hf_token and hf_token.strip() else None,
                )

                # Apply inference optimization
                FastLanguageModel.for_inference(model)

                self.models[model_name]["model"] = model
                self.models[model_name]["tokenizer"] = tokenizer

            # Load chat template info
            self._load_chat_template_info(model_name)

            self.active_model_name = model_name
            self.loading_models.discard(model_name)

            logger.info(f"Successfully loaded model: {model_name}")
            log_gpu_memory(f"After loading {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            error_msg = format_error_message(e, config.identifier)

            # Cleanup on failure
            if model_name in self.models:
                del self.models[model_name]
            self.loading_models.discard(model_name)

            raise Exception(error_msg)
    pass

    # Add this new function
    def unload_model(self, model_name: str) -> bool:
        """
        Completely removes a model from the registry and clears GPU memory.
        """
        if model_name in self.models:
            try:
                logger.info(f"Unloading model '{model_name}' from memory.")
                # Delete the model entry from our registry
                del self.models[model_name]

                # Clear the active model if it was the one being unloaded
                if self.active_model_name == model_name:
                    self.active_model_name = None

                # Use garbage collection and clear CUDA cache to release memory
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(f"Model '{model_name}' successfully unloaded.")
                return True
            except Exception as e:
                logger.error(f"Error while unloading model '{model_name}': {e}")
                return False
        else:
            logger.warning(f"Attempted to unload model '{model_name}', but it was not found in the registry.")
            return True
    pass

    def revert_to_base_model(self, base_model_name: str) -> bool:
        """
        Reverts the model to its pristine base state by unloading AND
        deleting all adapter configurations, as instructed.
        """
        if base_model_name not in self.models:
            return False

        model = self.models[base_model_name].get("model")

        try:
            # Step 1: Unload the adapter weights. This returns the base model object.
            # This step is only necessary if the model is currently a PeftModel instance.
            if isinstance(model, (PeftModel, PeftModelForCausalLM)):
                logger.info("Model is a PeftModel. Unloading adapters...")
                unwrapped_base_model = model.unload()
                self.models[base_model_name]["model"] = unwrapped_base_model
                model = unwrapped_base_model # Continue with the unwrapped model

            # Step 2: Delete any lingering adapter configurations from the object.
            # This is the crucial step you identified.
            if hasattr(model, 'peft_config') and model.peft_config:
                logger.info("Found lingering adapter configurations. Deleting them now...")
                # Create a static list of keys before iterating and deleting
                for name in list(model.peft_config.keys()):
                    logger.info(f"Deleting adapter config: '{name}'")
                    model.delete_adapter(name)

            logger.info("Model has been successfully reverted to a clean base state.")
            return True

        except Exception as e:
            logger.error(f"Failed to revert model to base state: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    pass

    def activate_lora_adapter(self, base_model_name: str, lora_path: str) -> Tuple[bool, Optional[str]]:
        """
        Activates a specific LoRA adapter on what is assumed to be a clean base model.
        """
        model = self.models[base_model_name].get("model")
        adapter_name_to_load = lora_path.split("/")[-1].replace(".", "_")

        try:
            # At this point, the model should be clean thanks to revert_to_base_model.
            # We can now safely load and set the new adapter.

            # Step 3: Load the new adapter.
            logger.info(f"Loading adapter '{adapter_name_to_load}' from '{lora_path}'")
            model.load_adapter(lora_path, adapter_name=adapter_name_to_load)

            # Step 4: Set the new adapter as active.
            logger.info(f"Setting '{adapter_name_to_load}' as the active adapter.")
            model.set_adapter(adapter_name_to_load)

            return True, adapter_name_to_load
        except Exception as e:
            # This will catch the "already exists" error if revert_to_base_model failed.
            logger.error(f"Failed to activate LoRA adapter '{adapter_name_to_load}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None
    pass

    def load_adapter(self, base_model_name: str, adapter_path: str, adapter_name: str = None) -> bool:
        """
        Load a LoRA adapter onto the base model if it's not already registered.
        This method is idempotent.
        """
        if base_model_name not in self.models:
            logger.error(f"Base model {base_model_name} not loaded")
            return False

        model = self.models[base_model_name].get("model")
        if model is None:
            logger.error(f"Model object for {base_model_name} is None.")
            return False

        if adapter_name is None:
            adapter_name = adapter_path.split("/")[-1].replace(".", "_")

        # If we've loaded this adapter before, we don't need to do anything.
        if adapter_name in self.models[base_model_name].get("loaded_adapters", {}):
            logger.info(f"Adapter '{adapter_name}' is already registered. Skipping.")
            return True

        try:
            logger.info(f"Loading new adapter '{adapter_name}' from '{adapter_path}' onto {base_model_name}")

            # Unsloth modifies the model in-place and returns None. Do NOT re-assign.
            model.load_adapter(adapter_path, adapter_name=adapter_name)

            # Update our internal registry so we don't load it again.
            self.models[base_model_name]["loaded_adapters"][adapter_name] = adapter_path

            total_adapters = len(getattr(model, 'peft_config', {}))
            logger.info(f"Adapter '{adapter_name}' loaded successfully. (Total adapters on model: {total_adapters})")
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter '{adapter_name}': {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    pass

    def enable_adapter(self, base_model_name: str, adapter_name: str) -> bool:
        """Enable specific adapter (for generation)"""
        if base_model_name not in self.models:
            return False

        model = self.models[base_model_name]["model"]

        try:
            logger.info(f"Enabling adapter: {adapter_name}")
            model.set_adapter(adapter_name)
            self.models[base_model_name]["active_adapter"] = adapter_name
            return True
        except Exception as e:
            logger.error(f"Failed to enable adapter: {e}")
            return False

    def disable_adapters(self, base_model_name: str) -> bool:
        """Disable all adapters (back to pure base model)"""
        if base_model_name not in self.models:
            return False

        model = self.models[base_model_name]["model"]

        try:
            logger.info(f"Disabling all adapters on {base_model_name}")
            model.disable_adapters()
            self.models[base_model_name]["active_adapter"] = None
            return True
        except Exception as e:
            logger.error(f"Failed to disable adapters: {e}")
            return False

    # In backend/inference.py

    def load_for_eval(self, lora_path: str, max_seq_length: int = 2048,
                     dtype = None, load_in_4bit: bool = True,
                     hf_token: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Prepare for eval: ensure base model and the specified adapter are loaded.
        """
        try:
            from .model_config import ModelConfig
            lora_config = ModelConfig.from_lora_path(lora_path, hf_token)
            if not lora_config:
                return False, None, None

            base_model_name = lora_config.base_model

            # 1. Load the base model if it's not already in memory (this logic is correct)
            if base_model_name not in self.models or not self.models[base_model_name].get("model"):
                logger.info(f"Base model '{base_model_name}' not loaded, loading now.")
                base_config = ModelConfig.from_ui_selection(base_model_name, None, is_lora=False)
                if not self.load_model(base_config, max_seq_length, dtype, load_in_4bit, hf_token):
                    return False, None, None
            else:
                logger.info(f"Base model '{base_model_name}' is already in memory.")

            self.active_model_name = base_model_name

            # 2. Delegate to our now-idempotent load_adapter function.
            # It will handle all cases: first adapter, or subsequent adapters.
            adapter_name = lora_path.split("/")[-1].replace(".", "_")
            adapter_success = self.load_adapter(
                base_model_name=base_model_name,
                adapter_path=lora_path,
                adapter_name=adapter_name
            )

            if not adapter_success:
                return False, base_model_name, None

            return True, base_model_name, adapter_name

        except Exception as e:
            logger.error(f"Error during load_for_eval: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None, None
    pass


    def load_for_eval(self, lora_path: str, max_seq_length: int = 2048,
                     dtype = None, load_in_4bit: bool = True,
                     hf_token: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Final Corrected Version:
        Ensures the base model and the specified adapter are loaded.
        This function is idempotent and handles all states correctly.
        """
        try:
            from .model_config import ModelConfig
            lora_config = ModelConfig.from_lora_path(lora_path, hf_token)
            if not lora_config:
                return False, None, None

            base_model_name = lora_config.base_model

            # 1. Load the base model if it's not already in memory
            if base_model_name not in self.models or not self.models[base_model_name].get("model"):
                logger.info(f"Base model '{base_model_name}' not loaded, loading now.")
                base_config = ModelConfig.from_ui_selection(base_model_name, None, is_lora=False)
                if not self.load_model(base_config, max_seq_length, dtype, load_in_4bit, hf_token):
                    return False, None, None

            self.active_model_name = base_model_name

            # 2. Determine the required adapter name from the user's selection
            adapter_name = lora_path.split("/")[-1].replace(".", "_")

            # 3. Call our robust load_adapter function to ensure this specific adapter is loaded.
            # It will only load from disk if the model doesn't already have it.
            adapter_success = self.load_adapter(
                base_model_name=base_model_name,
                adapter_path=lora_path,
                adapter_name=adapter_name
            )
            if not adapter_success:
                return False, base_model_name, None

            # 4. Return the correct, verified adapter name for the UI logic to use.
            return True, base_model_name, adapter_name

        except Exception as e:
            logger.error(f"Error during load_for_eval: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, None, None
    pass

    def load_adapter(self, base_model_name: str, adapter_path: str, adapter_name: str) -> bool:
        """
        Loads an adapter onto the model ONLY if it's not already attached.
        """
        model = self.models[base_model_name].get("model")

        # Check if this adapter name is already part of the model's config. This is the most reliable check.
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            logger.info(f"Adapter '{adapter_name}' is already attached to the model. Skipping load.")
            return True

        try:
            logger.info(f"Loading new adapter '{adapter_name}' from '{adapter_path}' onto {base_model_name}")
            model.load_adapter(adapter_path, adapter_name=adapter_name)

            # Update our internal registry ONLY after a successful load.
            if "loaded_adapters" not in self.models[base_model_name]:
                self.models[base_model_name]["loaded_adapters"] = {}
            self.models[base_model_name]["loaded_adapters"][adapter_name] = adapter_path

            total_adapters = len(getattr(model, 'peft_config', {}))
            logger.info(f"Adapter '{adapter_name}' loaded successfully. (Total unique adapters on model: {total_adapters})")
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter '{adapter_name}': {e}")
            return False
    pass

    def set_active_adapter(self, base_model_name: str, adapter_name: str) -> bool:
        """
        Sets the active adapter for generation. This replaces the flawed 'enable_adapter'.
        """
        model = self.models[base_model_name].get("model")
        try:
            logger.info(f"Setting active adapter to: '{adapter_name}'")
            model.set_adapter(adapter_name)
            self.models[base_model_name]["active_adapter"] = adapter_name
            return True
        except Exception as e:
            # This will catch the "adapter not found" error if something goes wrong.
            logger.error(f"Failed to set active adapter to '{adapter_name}': {e}")
            return False
    pass

    def generate_chat_response(self,
                          messages: list,
                          system_prompt: str,
                          image=None,
                          temperature: float = 0.7,
                          top_p: float = 0.9,
                          top_k: int = 40,
                          max_new_tokens: int = 256,
                          repetition_penalty: float = 1.1) -> Generator[str, None, None]:
        """
        Generate response for text or vision models.

        1. Messages are already in ChatML format (role/content)
        2. Apply get_chat_template() if model in mapper
        3. Apply tokenizer.apply_chat_template()
        4. Generate
        """
        if not self.active_model_name:
            yield "Error: No active model"
            return

        model_info = self.models[self.active_model_name]
        is_vision = model_info.get("is_vision", False)
        tokenizer = model_info.get("tokenizer") or model_info.get("processor")

        with self._generation_lock:
            if is_vision:
                # Vision model generation
                yield from self._generate_vision_response(
                    messages, system_prompt, image,
                    temperature, top_p, top_k, max_new_tokens, repetition_penalty
                )
            else:
                # Text model: Use training pipeline approach
                # Messages are already in ChatML format from eval.py

                # Step 1: Apply get_chat_template if model is in mapper
                try:
                    from backend.dataset_utils import MODEL_TO_TEMPLATE_MAPPER, get_tokenizer_chat_template

                    model_name_lower = self.active_model_name.lower()

                    # Check if model has a registered template
                    if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
                        template_name = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
                        logger.info(f"Applying chat template '{template_name}' for {self.active_model_name}")

                        # This modifies the tokenizer with the correct template
                        tokenizer = get_chat_template(
                            tokenizer,
                            self.active_model_name
                        )
                    else:
                        logger.info(f"No registered template for {self.active_model_name}, using tokenizer default")
                except Exception as e:
                    logger.warning(f"Could not apply get_chat_template: {e}")

                # Step 2: Format with tokenizer.apply_chat_template()
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    logger.debug(f"Formatted prompt: {formatted_prompt[:200]}...")
                except Exception as e:
                    logger.error(f"Error applying chat template: {e}")
                    # Fallback to manual formatting
                    formatted_prompt = self.format_chat_prompt(messages, system_prompt)

                # Step 3: Generate
                yield from self.generate_stream(
                    formatted_prompt, temperature, top_p, top_k, max_new_tokens, repetition_penalty
                )

    def _generate_vision_response(self, messages, system_prompt, image,
                                  temperature, top_p, top_k, max_new_tokens,
                                  repetition_penalty) -> Generator[str, None, None]:
        """Handle vision model generation."""
        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        processor = model_info["processor"]

        # Extract user message
        user_message = ""
        if messages and messages[-1]["role"] == "user":
            import re
            user_message = messages[-1]["content"]
            user_message = re.sub(r'<img[^>]*>', '', user_message).strip()

        if not user_message:
            user_message = "Describe this image." if image else "Hello"

        # Prepare vision messages
        if image:
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_message}
                    ],
                }
            ]

            input_text = processor.apply_chat_template(vision_messages, add_generation_prompt=True)
            inputs = processor(
                image,
                input_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).to("cuda")
        else:
            # Text-only for vision model
            formatted_prompt = self.format_chat_prompt(messages, system_prompt)
            inputs = processor.tokenizer(formatted_prompt, return_tensors="pt").to("cuda")

        # Generate with streaming
        captured_output = StringIO()
        original_stdout = sys.stdout

        try:
            sys.stdout = captured_output

            text_streamer = TextStreamer(processor.tokenizer, skip_prompt=True)
            model.generate(
                **inputs,
                streamer=text_streamer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

            sys.stdout = original_stdout
            generated_text = captured_output.getvalue()
            cleaned = self._clean_generated_text(generated_text)
            yield cleaned

        except Exception as e:
            sys.stdout = original_stdout
            logger.error(f"Vision generation error: {e}")
            yield f"Error: {str(e)}"
    pass

    def generate_stream(self,
                       prompt: str,
                       temperature: float = 0.7,
                       top_p: float = 0.9,
                       top_k: int = 40,
                       max_new_tokens: int = 256,
                       repetition_penalty: float = 1.1) -> Generator[str, None, None]:
        """Generate streaming text response (text models only)."""
        if not self.active_model_name:
            yield "Error: No active model"
            return

        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        tokenizer = model_info["tokenizer"]

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            from transformers import TextIteratorStreamer
            import threading

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id if tokenizer.pad_token_id is None else tokenizer.pad_token_id,
            )

            def generate_fn():
                try:
                    model.generate(**generation_kwargs)
                except Exception as e:
                    logger.error(f"Generation error: {e}")

            thread = threading.Thread(target=generate_fn)
            thread.start()

            output = ""
            for new_token in streamer:
                if new_token:
                    output += new_token
                    cleaned = self._clean_generated_text(output)
                    yield cleaned

            thread.join()

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield f"Error: {str(e)}"

    # ... other helper methods (format_chat_prompt, _clean_generated_text, etc.)
    pass

    def format_chat_prompt(self, messages: list, system_prompt: str = None) -> str:
        if not self.active_model_name or self.active_model_name not in self.models:
            logger.error("No active model available")
            return ""

        if self.models[self.active_model_name].get("tokenizer") is None:
            logger.error("Tokenizer not loaded for active model")
            return ""

        chat_template_info = self.models[self.active_model_name].get("chat_template_info", {})
        tokenizer = self.models[self.active_model_name]["tokenizer"]

        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        last_role = "system" if system_prompt else None

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role in ["system", "user", "assistant"] and content.strip():
                if role == last_role:
                    logger.debug(f"Skipping consecutive {role} message to maintain alternation")
                    continue

                if role == "user":
                    import re
                    clean_content = re.sub(r'<[^>]+>', '', content).strip()
                    if clean_content:
                        chat_messages.append({"role": role, "content": clean_content})
                        last_role = role
                elif role == "assistant" and content.strip():
                    chat_messages.append({"role": role, "content": content})
                    last_role = role
                elif role == "system":
                    continue

        if chat_messages and chat_messages[-1]["role"] == "assistant":
            logger.debug("Removing final assistant message to ensure proper alternation")
            chat_messages.pop()

        logger.info(f"Sending {len(chat_messages)} messages to tokenizer:")
        for i, msg in enumerate(chat_messages):
            logger.info(f"  {i}: {msg['role']} - {msg['content'][:50]}...")

        try:
            formatted_prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            logger.info(f"Successfully applied tokenizer's native chat template")
            return formatted_prompt
        except Exception as e:
            error_msg = str(e).lower()
            if "chat_template is not set" in error_msg or "no template argument" in error_msg:
                logger.info(f"Base model detected - no built-in chat template available, using fallback formatting")
            else:
                logger.warning(f"Failed to apply tokenizer chat template: {e}")
            logger.debug(f"""Failed with messages: {[f"{m['role']}: {m['content'][:30]}..." for m in chat_messages]}""")

        if chat_template_info.get("has_template", False):
            logger.info("Falling back to manual template formatting based on detected patterns")
            template_type = chat_template_info.get("format_type", "generic")
            manual_prompt = self._format_chat_manual(chat_messages, template_type, chat_template_info.get("special_tokens", {}))
            logger.info(f"Manual template result: {manual_prompt[:200]}...")
            return manual_prompt
        else:
            logger.info("Using generic chat formatting for base model")
            return self._format_generic_template(chat_messages, {})

    def _format_chat_manual(self, messages: list, template_type: str, special_tokens: dict) -> str:
        """
        Manual chat formatting fallback for when tokenizer template fails

        Args:
            messages: List of message dictionaries
            template_type: Detected template type
            special_tokens: Dictionary of special tokens

        Returns:
            str: Manually formatted prompt
        """
        if template_type == "llama3":
            return self._format_llama3_template(messages, special_tokens)
        elif template_type == "mistral":
            return self._format_mistral_template(messages, special_tokens)
        elif template_type == "chatml":
            return self._format_chatml_template(messages, special_tokens)
        elif template_type == "alpaca":
            return self._format_alpaca_template(messages, special_tokens)
        else:
            return self._format_generic_template(messages, special_tokens)

    def _format_llama3_template(self, messages: list, special_tokens: dict) -> str:
        """Format messages using Llama 3 template"""
        bos_token = special_tokens.get("bos_token", "<|begin_of_text|>")
        formatted = bos_token

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

        formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return formatted

    def _format_mistral_template(self, messages: list, special_tokens: dict) -> str:
        """Format messages using Mistral template"""
        bos_token = special_tokens.get("bos_token", "<s>")
        formatted = bos_token

        system_msg = None
        conversation = []

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                conversation.append(msg)

        i = 0
        while i < len(conversation):
            if conversation[i]["role"] == "user":
                user_content = conversation[i]["content"]

                if system_msg and i == 0:
                    user_content = f"{system_msg}\n\n{user_content}"

                formatted += f"[INST] {user_content} [/INST]"

                if i + 1 < len(conversation) and conversation[i + 1]["role"] == "assistant":
                    formatted += f" {conversation[i + 1]['content']}</s>"
                    i += 2
                else:
                    formatted += " "
                    break
            else:
                i += 1

        return formatted

    def _format_chatml_template(self, messages: list, special_tokens: dict) -> str:
        """Format messages using ChatML template"""
        formatted = ""

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        formatted += "<|im_start|>assistant\n"
        return formatted

    def _format_alpaca_template(self, messages: list, special_tokens: dict) -> str:
        """Format messages using Alpaca template"""
        formatted = ""
        system_msg = None

        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            elif msg["role"] == "user":
                if system_msg:
                    formatted += f"### Instruction:\n{system_msg}\n\n### Input:\n{msg['content']}\n\n### Response:\n"
                    system_msg = None
                else:
                    formatted += f"### Human:\n{msg['content']}\n\n### Assistant:\n"
            elif msg["role"] == "assistant":
                formatted += f"{msg['content']}\n\n"

        return formatted

    def _format_generic_template(self, messages: list, special_tokens: dict) -> str:
        """Generic fallback formatting"""
        formatted = ""

        for msg in messages:
            role = msg["role"].title()
            content = msg["content"]
            formatted += f"{role}: {content}\n"

        formatted += "Assistant: "
        return formatted

    def check_vision_model_compatibility(self, show_warning: bool = True) -> bool:
        """
        Check if current model supports vision and optionally show warning if image uploaded to text-only model

        Args:
            show_warning: Whether to show Gradio warning if vision not supported

        Returns:
            bool: True if current model supports vision, False otherwise
        """
        current_model = self.get_current_model()
        if current_model and current_model in self.models:
            is_vision = self.models[current_model].get("is_vision", False)
            if not is_vision and show_warning:
                import gradio as gr
                model_short = current_model.split('/')[-1] if '/' in current_model else current_model
                gr.Warning(f"Image uploaded, but {model_short} is a text-only model. Please select a vision model to analyze images.")
            return is_vision
        return False

    def _reset_model_generation_state(self, model_name: str):
        """Reset generation state for a specific model to prevent contamination."""
        if model_name not in self.models:
            return

        model = self.models[model_name].get("model")
        if not model:
            return

        try:
            # This is a common pattern for Unsloth/Hugging Face models
            if hasattr(model, 'past_key_values'):
                model.past_key_values = None
            if hasattr(model, 'generation_config'):
                if hasattr(model.generation_config, 'past_key_values'):
                     model.generation_config.past_key_values = None

            logger.debug(f"Reset generation state for model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not fully reset model state for {model_name}: {e}")
    pass

    def reset_generation_state(self):
        """Reset any cached generation state to prevent hanging after errors"""
        try:
            # Clear cached states for ALL loaded models
            for model_name in self.models.keys():
                self._reset_model_generation_state(model_name)

            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                logger.debug("Cleared CUDA cache and IPC resources")

            import gc
            gc.collect()
            logger.info("Performed comprehensive generation state reset")

        except Exception as e:
            logger.warning(f"Could not fully reset generation state: {e}")

    def resize_image(self, img, max_size: int = 800):
        """Resize image while maintaining aspect ratio if either dimension exceeds max_size"""
        if img is None:
            return None
        if img.size[0] > max_size or img.size[1] > max_size:
            from PIL import Image
            ratio = min(max_size/img.size[0], max_size/img.size[1])
            new_size = (int(img.size[0]*ratio), int(img.size[1]*ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def _clean_generated_text(self, text: str) -> str:
        import re

        text = re.sub(r'<\|start_header_id\|>.*?<\|end_header_id\|>', '', text)
        text = re.sub(r'<\|eot_id\|>', '', text)
        text = re.sub(r'<\|begin_of_text\|>', '', text)

        text = re.sub(r'\[INST\].*?\[/INST\]', '', text)
        text = re.sub(r'<s>|</s>', '', text)

        # Clean ChatML tokens (used by Qwen2-VL and similar models)
        text = re.sub(r'<\|im_start\|>.*?<\|im_end\|>', '', text)
        text = re.sub(r'<\|im_end\|>', '', text)
        text = re.sub(r'<\|im_start\|>', '', text)

        text = re.sub(r'^\s*(assistant|user|system):\s*', '', text, flags=re.IGNORECASE)
        text = text.strip()

        return text

    def _load_chat_template_info(self, model_name: str):
        if model_name not in self.models or not self.models[model_name].get("tokenizer"):
            return

        tokenizer = self.models[model_name]["tokenizer"]
        chat_template_info = {
            "has_template": False,
            "template": None,
            "format_type": "generic",
            "special_tokens": {},
            "template_name": None,
        }

        try:
            from backend.dataset_utils import MODEL_TO_TEMPLATE_MAPPER
            #Try exact match first
            model_name_lower = model_name.lower()
            if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
                chat_template_info["template_name"] = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
                logger.info(f"Detected template '{chat_template_info['template_name']}' for {model_name} from mapper")
            else:
                # Try partial match (for variants like model_name-bnb-4bit)
                for key in MODEL_TO_TEMPLATE_MAPPER:
                    if key in model_name_lower or model_name_lower in key:
                        chat_template_info["template_name"] = MODEL_TO_TEMPLATE_MAPPER[key]
                        logger.info(f"Detected template '{chat_template_info['template_name']}' for {model_name} (partial match)")
                        break
        except Exception as e:
            logger.warning(f"Could not detect template from mapper for {model_name}: {e}")

        try:
            if hasattr(tokenizer, 'chat_template') and tokenizer.chat_template:
                chat_template_info["has_template"] = True
                chat_template_info["template"] = tokenizer.chat_template

                template_str = tokenizer.chat_template.lower()

                if "start_header_id" in template_str and "end_header_id" in template_str:
                    chat_template_info["format_type"] = "llama3"
                elif "[inst]" in template_str and "[/inst]" in template_str:
                    chat_template_info["format_type"] = "mistral"
                elif "<|im_start|>" in template_str and "<|im_end|>" in template_str:
                    chat_template_info["format_type"] = "chatml"
                elif "### instruction:" in template_str or "### human:" in template_str:
                    chat_template_info["format_type"] = "alpaca"
                else:
                    chat_template_info["format_type"] = "custom"

                logger.info(f"Loaded chat template for {model_name} (detected as {chat_template_info['format_type']} format)")
                logger.debug(f"Template preview: {tokenizer.chat_template[:200]}...")

                special_tokens = {}
                if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token:
                    special_tokens["bos_token"] = tokenizer.bos_token
                if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token:
                    special_tokens["eos_token"] = tokenizer.eos_token
                if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token:
                    special_tokens["pad_token"] = tokenizer.pad_token

                chat_template_info["special_tokens"] = special_tokens

            else:
                logger.info(f"No chat template found for {model_name}, will use generic formatting")

        except Exception as e:
            logger.error(f"Error loading chat template info for {model_name}: {e}")

        self.models[model_name]["chat_template_info"] = chat_template_info

        if chat_template_info["has_template"]:
            logger.info(f"Chat template loaded for {model_name}: {chat_template_info['format_type']} format")
        else:
            logger.info(f"No built-in chat template for {model_name}, will use generic formatting")


    def get_current_model(self) -> Optional[str]:
        """Get currently active model name"""
        return self.active_model_name

    def is_model_loading(self) -> bool:
        """Check if any model is currently loading"""
        return len(self.loading_models) > 0

    def get_loading_model(self) -> Optional[str]:
        """Get name of currently loading model"""
        return next(iter(self.loading_models)) if self.loading_models else None

    def load_model_simple(self,
                         model_path: str,
                         hf_token: Optional[str] = None,
                         max_seq_length: int = 2048,
                         load_in_4bit: bool = True) -> bool:
        """
        Simple model loading wrapper for chat interface.
        Accepts model path as string and handles ModelConfig creation internally.

        Args:
            model_path: Model name or path (e.g., "unsloth/llama-3-8b")
            hf_token: HuggingFace token for gated models
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Create config from string path
            config = ModelConfig.from_ui_selection(
                model_path,
                lora_path=None,  # No LoRA for chat
                is_lora=False
            )

            # Call existing load_model with config
            return self.load_model(
                config=config,
                max_seq_length=max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=load_in_4bit,
                hf_token=hf_token
            )

        except Exception as e:
            logger.error(f"Error in load_model_simple: {e}")
            return False

    def add_local_model_to_dropdown(self, model_path: str):
        """Add successfully loaded local model to dropdown storage"""
        try:
            from pathlib import Path

            path_obj = Path(model_path)
            display_name = f"{path_obj.name}"

            # Check if already exists
            for existing_display, existing_path in self.loaded_local_models:
                if existing_path == model_path:
                    logger.debug(f"Local model already in dropdown: {model_path}")
                    return

            # Add to beginning of list
            self.loaded_local_models.insert(0, (display_name, model_path))
            logger.info(f"Added local model to dropdown: {display_name} -> {model_path}")

            # Keep only last 5
            if len(self.loaded_local_models) > 5:
                self.loaded_local_models.pop()

        except Exception as e:
            logger.error(f"Error adding local model to dropdown: {e}")

    def get_model_dropdown_choices(self, models: list = None) -> list:
        """Get model dropdown choices with status indicators"""
        if models is None:
            models = self.default_models

        try:
            active_model = self.active_model_name
            loading_model = self.get_loading_model()

            choices = []

            # Add local models first
            for local_display, local_path in self.loaded_local_models:
                if local_path == active_model:
                    choices.append((f"{local_display} (Active)", local_path))
                else:
                    choices.append((local_display, local_path))

            # Add default models
            for model in models:
                short_name = model.split('/')[-1] if '/' in model else model

                if model == active_model:
                    display_name = f"{short_name} (Active)"
                elif model == loading_model:
                    display_name = f"{short_name} (Loading...)"
                elif model in self.models and self.models[model].get("model"):
                # Model is loaded in memory
                    display_name = f"{short_name} (Ready)"
                # elif model in self.models:
                #     display_name = f"{short_name} (Ready)"
                elif is_model_cached(model):
                    # Model is downloaded but not loaded
                    display_name = f"{short_name} (Cached)"
                else:
                    display_name = f"↓ {short_name}"  # Not downloaded

                choices.append((display_name, model))

            return choices

        except Exception as e:
            logger.error(f"Error getting model choices: {e}")
            return [(model.split('/')[-1], model) for model in models]


    def update_model_dropdown(self, models: list = None):
        """Update model dropdown with current status"""
        try:
            import gradio as gr

            choices = self.get_model_dropdown_choices(models)
            active_model = self.active_model_name

            # Set value to active model if exists
            value = active_model if active_model else (choices[0][1] if choices else None)

            return gr.update(choices=choices, value=value)

        except Exception as e:
            logger.error(f"Error updating model dropdown: {e}")
            import gradio as gr
            return gr.update()

    def load_model_simple(self,
                     model_path: str,
                     hf_token: Optional[str] = None,
                     max_seq_length: int = 2048,
                     load_in_4bit: bool = True) -> bool:
        """
        Simple model loading wrapper for chat interface.
        Accepts model path as string and handles ModelConfig creation internally.

        Args:
            model_path: Model name or path (e.g., "unsloth/llama-3-8b")
            hf_token: HuggingFace token for gated models
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            from backend.model_config import ModelConfig

            logger.info(f"load_model_simple called with: {model_path}")

            # Create config from string path
            config = ModelConfig.from_ui_selection(
                model_path,
                lora_path=None,  # No LoRA for chat
                is_lora=False
            )

            logger.info(f"Created ModelConfig with identifier: {config.identifier}")

            # Call existing load_model with config
            return self.load_model(
                config=config,
                max_seq_length=max_seq_length,
                dtype=None,  # Auto-detect
                load_in_4bit=load_in_4bit,
                hf_token=hf_token
            )

        except Exception as e:
            logger.error(f"Error in load_model_simple: {e}")
            import traceback
            traceback.print_exc()
            return False

pass


# Global inference backend instance
inference_backend = InferenceBackend()

def get_inference_backend() -> InferenceBackend:
    return inference_backend
