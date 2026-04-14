# SPDX-License-Identifier: AGPL-3.0-only
"""MLX inference backend for Apple Silicon.

Drop-in replacement for InferenceBackend — same interface, uses mlx-lm/mlx-vlm
instead of torch/transformers for model loading and generation.
"""

import threading
from typing import Optional, Generator
from loggers import get_logger

logger = get_logger(__name__)


class MLXInferenceBackend:

    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []
        self.device = "mlx"
        self._generation_lock = threading.Lock()

        # MLX state
        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_vlm = False
        self._config = {}

    def load_model(self, config, max_seq_length=2048, load_in_4bit=True,
                   hf_token=None, trust_remote_code=False, gpu_ids=None,
                   dtype=None) -> bool:
        import mlx.core as mx

        model_name = config.identifier if hasattr(config, "identifier") else str(config)
        is_vision = getattr(config, "is_vision", False)

        if hf_token:
            import os
            os.environ["HF_TOKEN"] = hf_token

        if mx.metal.is_available():
            mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

        logger.info("Loading %s via %s", model_name, "mlx-vlm" if is_vision else "mlx-lm")

        if is_vision:
            from mlx_vlm import load as vlm_load
            model, processor = vlm_load(model_name)
            self._model = model
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", processor)
            self._is_vlm = True
        else:
            from mlx_lm import load as mlx_load
            tokenizer_config = {}
            if trust_remote_code:
                tokenizer_config["trust_remote_code"] = True
            model, tokenizer = mlx_load(
                model_name,
                tokenizer_config=tokenizer_config if tokenizer_config else None,
            )
            self._model = model
            self._tokenizer = tokenizer
            self._processor = None
            self._is_vlm = False

        self.active_model_name = model_name
        self.models[model_name] = {
            "model": self._model,
            "tokenizer": self._tokenizer,
            "processor": self._processor,
            "is_vision": is_vision,
            "is_lora": getattr(config, "is_lora", False),
            "is_audio": False,
            "audio_type": None,
            "has_audio_input": False,
        }

        logger.info("Model %s loaded successfully", model_name)
        return True

    def unload_model(self, model_name: str) -> bool:
        import mlx.core as mx
        import gc

        if model_name in self.models:
            del self.models[model_name]
        self._model = None
        self._tokenizer = None
        self._processor = None
        if self.active_model_name == model_name:
            self.active_model_name = None
        gc.collect()
        mx.clear_cache()
        logger.info("Model %s unloaded", model_name)
        return True

    def generate_chat_response(
        self,
        messages,
        system_prompt="",
        image=None,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        min_p=0.0,
        max_new_tokens=256,
        repetition_penalty=1.0,
        cancel_event=None,
    ) -> Generator[str, None, None]:

        if self._model is None:
            raise RuntimeError("No model loaded")

        # Build messages with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)

        # Inject image into the last user message for VLM
        if self._is_vlm and image is not None:
            for msg in reversed(full_messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        msg["content"] = [
                            {"type": "image"},
                            {"type": "text", "text": content},
                        ]
                    elif isinstance(content, list):
                        # Prepend image if not already there
                        has_image = any(p.get("type") == "image" for p in content if isinstance(p, dict))
                        if not has_image:
                            content.insert(0, {"type": "image"})
                    break

        if self._is_vlm:
            yield from self._generate_vlm(
                full_messages, image, temperature, top_p, top_k,
                max_new_tokens, repetition_penalty, cancel_event,
            )
        else:
            yield from self._generate_text(
                full_messages, temperature, top_p, top_k,
                max_new_tokens, repetition_penalty, cancel_event,
            )

    def _generate_text(self, messages, temperature, top_p, top_k,
                       max_new_tokens, repetition_penalty, cancel_event):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        if prompt is None:
            raise RuntimeError("apply_chat_template returned None — tokenizer may be incompatible")

        sampler = make_sampler(temp=temperature, top_p=top_p, min_tokens_to_keep=1)

        cumulative = ""
        logger.info("Generating: prompt_len=%d, max_tokens=%d, model=%s, tokenizer=%s",
                     len(prompt), max_new_tokens, type(self._model).__name__, type(self._tokenizer).__name__)
        with self._generation_lock:
            try:
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    sampler=sampler,
                ):
                    token_text = response.text if hasattr(response, "text") else str(response)
                    cumulative += token_text
                    yield cumulative

                    if cancel_event and cancel_event.is_set():
                        break
            except Exception as e:
                import traceback
                logger.error("stream_generate failed:\n%s", traceback.format_exc())
                raise

    def _generate_vlm(self, messages, image, temperature, top_p, top_k,
                      max_new_tokens, repetition_penalty, cancel_event):
        from mlx_vlm import stream_generate as vlm_stream

        # Apply chat template
        chat_fn = getattr(self._processor, "apply_chat_template", None)
        if chat_fn is None or not hasattr(self._processor, "chat_template") or self._processor.chat_template is None:
            tok = getattr(self._processor, "tokenizer", self._processor)
            chat_fn = tok.apply_chat_template

        prompt = chat_fn(messages, tokenize=False, add_generation_prompt=True)

        # For VLM: always use mlx_vlm's stream_generate which handles
        # pixel_values properly (passes None for text-only, image for VLM)
        images = [image] if image is not None else None

        cumulative = ""
        logger.info("VLM generating: prompt_len=%d, has_image=%s", len(prompt), image is not None)
        with self._generation_lock:
            for response in vlm_stream(
                self._model,
                self._processor,
                prompt,
                images,
                max_tokens=max_new_tokens,
                temp=temperature,
            ):
                token_text = response.text if hasattr(response, "text") else str(response)
                cumulative += token_text
                yield cumulative
                if cancel_event and cancel_event.is_set():
                    break

    def generate_with_adapter_control(self, use_adapter=None, cancel_event=None,
                                       **gen_kwargs) -> Generator[str, None, None]:
        # MLX LoRA adapter toggling not yet supported — generate normally
        yield from self.generate_chat_response(cancel_event=cancel_event, **gen_kwargs)

    def reset_generation_state(self):
        import mlx.core as mx
        import gc
        gc.collect()
        mx.clear_cache()
