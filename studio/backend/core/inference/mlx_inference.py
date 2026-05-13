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

        # Recorded for unload to release pinned memory back to the OS.
        self._memory_limits_applied = {}

    def _configure_memory_limits(self):
        """Apply Metal memory caps before loading a model.

        Mirrors MLXTrainer._configure_memory_limits's defaults:
        memory_limit = 85% of recommended working-set,
        wired_limit = min(recommended, memory_limit). Recorded so unload
        can lower wired_limit back to release pinned RAM.
        """
        import mlx.core as mx

        if not mx.metal.is_available():
            return
        info = mx.device_info()
        rec_bytes = info.get("max_recommended_working_set_size")
        if not rec_bytes or rec_bytes <= 0:
            return
        rec_gb = rec_bytes / 1e9
        memory_limit_gb = rec_gb * 0.85
        wired_limit_gb = min(rec_gb, memory_limit_gb)
        mx.set_memory_limit(int(memory_limit_gb * 1e9))
        mx.set_wired_limit(int(wired_limit_gb * 1e9))
        self._memory_limits_applied = {
            "memory_limit_gb": memory_limit_gb,
            "wired_limit_gb": wired_limit_gb,
            "recommended_gb": rec_gb,
        }
        logger.info(
            "MLX memory caps: memory_limit=%.2f GB, wired_limit=%.2f GB",
            memory_limit_gb,
            wired_limit_gb,
        )

    def load_model(
        self,
        config,
        max_seq_length = 2048,
        load_in_4bit = True,
        hf_token = None,
        trust_remote_code = False,
        gpu_ids = None,
        dtype = None,
    ) -> bool:
        import mlx.core as mx

        model_name = config.identifier if hasattr(config, "identifier") else str(config)
        is_vision = getattr(config, "is_vision", False)

        # GGUF guard. GGUF models are served via llama-server in the
        # parent process, NOT via mlx-lm in this MLX subprocess. The
        # route at studio/backend/routes/inference.py:592 (`if config.
        # is_gguf:`) is responsible for sending GGUF traffic to the
        # llama-server backend before reaching the MLX orchestrator.
        # If we end up here with is_gguf=True, the route's
        # `detect_gguf_model_remote` returned None on its first call
        # (transient HF Hub flake) but the subprocess re-detection
        # succeeded. The subprocess cannot reach into the parent's
        # llama-server, so all we can do is raise loudly so the caller
        # gets a clear error instead of a cryptic
        # "config.json does not exist" from mlx_lm.utils.load_model.
        if getattr(config, "is_gguf", False):
            raise RuntimeError(
                f"MLXInferenceBackend cannot load GGUF model '{model_name}': "
                f"GGUF models must be served by llama-server in the parent "
                f"process. The /api/inference/load route should have "
                f"detected this repo as GGUF before dispatching to the MLX "
                f"orchestrator -- this fallback indicates a transient HF "
                f"Hub failure during initial detection. Retry the request."
            )

        if hf_token:
            import os

            os.environ["HF_TOKEN"] = hf_token
        self._configure_memory_limits()

        is_lora = getattr(config, "is_lora", False)

        logger.info(
            "Loading %s via %s (is_lora=%s)",
            model_name,
            "mlx-vlm" if is_vision else "mlx-lm",
            is_lora,
        )

        try:
            from unsloth_zoo.mlx_loader import FastMLXModel
        except ImportError as e:
            raise ImportError(
                "Unsloth: MLX inference requires unsloth-zoo with the MLX modules "
                "(unsloth_zoo.mlx_loader). Reinstall via install.sh on Apple Silicon."
            ) from e

        model, tokenizer_or_processor = FastMLXModel.from_pretrained(
            model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
            token = hf_token,
            trust_remote_code = trust_remote_code,
            text_only = False if is_vision else True,
        )

        if is_vision:
            processor = tokenizer_or_processor
            self._model = model
            self._processor = processor
            self._tokenizer = getattr(processor, "tokenizer", processor)
            self._is_vlm = True
        else:
            tokenizer = tokenizer_or_processor
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

        if mx.metal.is_available() and self._memory_limits_applied and not self.models:
            try:
                mx.set_wired_limit(0)
                logger.info("MLX wired_limit released back to OS on unload")
            except Exception as e:
                logger.warning("Failed to release wired_limit: %s", e)
            self._memory_limits_applied = {}
        logger.info("Model %s unloaded", model_name)
        return True

    def generate_chat_response(
        self,
        messages,
        system_prompt = "",
        image = None,
        temperature = 0.7,
        top_p = 0.9,
        top_k = 40,
        min_p = 0.0,
        max_new_tokens = 256,
        repetition_penalty = 1.0,
        cancel_event = None,
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
                        has_image = any(
                            p.get("type") == "image"
                            for p in content
                            if isinstance(p, dict)
                        )
                        if not has_image:
                            content.insert(0, {"type": "image"})
                    break

        if self._is_vlm:
            yield from self._generate_vlm(
                full_messages,
                image,
                temperature,
                top_p,
                top_k,
                min_p,
                max_new_tokens,
                repetition_penalty,
                cancel_event,
            )
        else:
            yield from self._generate_text(
                full_messages,
                temperature,
                top_p,
                top_k,
                min_p,
                max_new_tokens,
                repetition_penalty,
                cancel_event,
            )

    def _generate_text(
        self,
        messages,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
    ):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        prompt = self._tokenizer.apply_chat_template(
            messages,
            tokenize = False,
            add_generation_prompt = True,
        )
        if prompt is None:
            raise RuntimeError(
                "apply_chat_template returned None — tokenizer may be incompatible"
            )

        sampler = make_sampler(
            temp = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
            min_tokens_to_keep = 1,
        )
        # Only build a logits processor when we actually have a non-trivial
        # repetition penalty (1.0 is the no-op value).
        logits_processors = None
        if repetition_penalty is not None and float(repetition_penalty) not in (
            0.0,
            1.0,
        ):
            logits_processors = make_logits_processors(
                repetition_penalty = float(repetition_penalty),
            )

        token_ids = []
        logger.info(
            "Generating: prompt_len=%d, max_tokens=%d, model=%s, tokenizer=%s",
            len(prompt),
            max_new_tokens,
            type(self._model).__name__,
            type(self._tokenizer).__name__,
        )
        with self._generation_lock:
            try:
                gen_kwargs = dict(
                    prompt = prompt,
                    max_tokens = max_new_tokens,
                    sampler = sampler,
                )
                if logits_processors is not None:
                    gen_kwargs["logits_processors"] = logits_processors
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    **gen_kwargs,
                ):
                    token_ids.append(response.token)
                    # Decode full sequence with skip_special_tokens — same as GPU
                    cumulative = self._tokenizer.decode(
                        token_ids,
                        skip_special_tokens = True,
                    )
                    yield cumulative

                    if cancel_event and cancel_event.is_set():
                        break
            except Exception as e:
                import traceback

                logger.error("stream_generate failed:\n%s", traceback.format_exc())
                raise

    def _generate_vlm(
        self,
        messages,
        image,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event,
    ):
        from mlx_vlm import stream_generate as vlm_stream

        # Apply chat template
        chat_fn = getattr(self._processor, "apply_chat_template", None)
        if (
            chat_fn is None
            or not hasattr(self._processor, "chat_template")
            or self._processor.chat_template is None
        ):
            tok = getattr(self._processor, "tokenizer", self._processor)
            chat_fn = tok.apply_chat_template

        prompt = chat_fn(messages, tokenize = False, add_generation_prompt = True)

        # For VLM: always use mlx_vlm's stream_generate which handles
        # pixel_values properly (passes None for text-only, image for VLM)
        images = [image] if image is not None else None

        cumulative = ""
        logger.info(
            "VLM generating: prompt_len=%d, has_image=%s",
            len(prompt),
            image is not None,
        )
        # mlx_vlm.stream_generate forwards **kwargs into generate_step, which
        # accepts temp/top_p/top_k/repetition_penalty (and builds the sampler
        # + logits_processors internally). Pass them through.
        # NOTE: mlx_vlm.generate_step expects ``temperature=`` (long form) —
        # passing ``temp=`` silently falls into **kwargs and is ignored,
        # leaving generation stuck at the default 0.0 (greedy).
        vlm_kwargs = dict(
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
        )
        if repetition_penalty is not None and float(repetition_penalty) not in (
            0.0,
            1.0,
        ):
            vlm_kwargs["repetition_penalty"] = float(repetition_penalty)

        with self._generation_lock:
            for response in vlm_stream(
                self._model,
                self._processor,
                prompt,
                images,
                **vlm_kwargs,
            ):
                token_text = (
                    response.text if hasattr(response, "text") else str(response)
                )
                cumulative += token_text
                yield cumulative
                if cancel_event and cancel_event.is_set():
                    break

    def generate_with_adapter_control(
        self, use_adapter = None, cancel_event = None, **gen_kwargs
    ) -> Generator[str, None, None]:
        # MLX LoRA adapter toggling not yet supported — generate normally
        yield from self.generate_chat_response(cancel_event = cancel_event, **gen_kwargs)

    def reset_generation_state(self):
        import mlx.core as mx
        import gc

        gc.collect()
        mx.clear_cache()
