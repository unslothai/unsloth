# SPDX-License-Identifier: AGPL-3.0-only
"""MLX inference backend for Apple Silicon.

Drop-in replacement for InferenceBackend — same interface, uses mlx-lm/mlx-vlm
instead of torch/transformers for model loading and generation.
"""

import threading
from typing import Optional, Generator
from core.inference.runtime_context import runtime_context_length
from loggers import get_logger

logger = get_logger(__name__)


def _build_generation_stats(prompt_n, prompt_tps, gen_n, gen_tps):
    """Map mlx stream stats onto the usage/timings shape llama-server emits."""
    prompt_n = int(prompt_n or 0)
    gen_n = int(gen_n or 0)
    prompt_tps = float(prompt_tps or 0.0)
    gen_tps = float(gen_tps or 0.0)
    prompt_ms = (prompt_n / prompt_tps * 1000.0) if prompt_tps > 0 else 0.0
    predicted_ms = (gen_n / gen_tps * 1000.0) if gen_tps > 0 else 0.0
    return {
        "usage": {
            "prompt_tokens": prompt_n,
            "completion_tokens": gen_n,
            "total_tokens": prompt_n + gen_n,
        },
        "timings": {
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prompt_per_token_ms": (prompt_ms / prompt_n) if prompt_n > 0 else 0.0,
            "prompt_per_second": prompt_tps,
            "predicted_n": gen_n,
            "predicted_ms": predicted_ms,
            "predicted_per_token_ms": (predicted_ms / gen_n) if gen_n > 0 else 0.0,
            "predicted_per_second": gen_tps,
            "cache_n": 0,
        },
    }


class MLXInferenceBackend:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []
        self.device = "mlx"
        self._generation_lock = threading.Lock()
        # usage/timings of the latest generation; shipped on gen_done.
        self.last_generation_stats = None

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_vlm = False
        self._config = {}

        # Recorded for unload to release pinned memory back to the OS.
        self._memory_limits_applied = {}

    def _configure_memory_limits(self):
        """Apply Metal memory caps before loading a model.

        memory_limit = 85% of recommended working-set;
        wired_limit = min(recommended, memory_limit). Recorded so unload can
        lower wired_limit back to release pinned RAM.
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

        # Keep the load token so the native-template fallback can fetch a
        # gated/private model's repo template later (during generation).
        self._hf_token = hf_token
        model_name = config.identifier if hasattr(config, "identifier") else str(config)
        is_vision = getattr(config, "is_vision", False)

        # GGUF guard. GGUF models are served by llama-server in the parent
        # process, not mlx-lm here. Reaching this with is_gguf=True means the
        # route's first detection flaked (transient HF Hub) but the subprocess
        # re-detected GGUF; raise loudly instead of a cryptic mlx_lm error.
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
            from unsloth_zoo.mlx.loader import FastMLXModel
        except ImportError as e:
            raise ImportError(
                "Unsloth: MLX inference requires unsloth-zoo with the MLX modules "
                "(unsloth_zoo.mlx.loader). Reinstall via install.sh on Apple Silicon."
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
            # Per-model load token for the native-template fallback (matches
            # the transformers backend; a later token-less load must not break
            # template fetches for a previously loaded gated model).
            "hf_token": hf_token,
            "model": self._model,
            "tokenizer": self._tokenizer,
            "processor": self._processor,
            "is_vision": is_vision,
            "is_lora": getattr(config, "is_lora", False),
            # For a LoRA adapter the native chat template lives on the base model;
            # without this the native-template fallback would load the adapter's own
            # (often template-less) tokenizer instead of the base model template.
            "base_model": getattr(config, "base_model", None)
            if getattr(config, "is_lora", False)
            else None,
            "is_audio": False,
            "audio_type": None,
            "has_audio_input": False,
            "context_length": runtime_context_length(self._model, max_seq_length),
        }
        # Capture chat_template_info so the worker IPC reply ships it back and
        # the route layer classifies capabilities like the other paths.
        self._populate_chat_template_info(model_name)

        logger.info("Model %s loaded successfully", model_name)
        return True

    def _populate_chat_template_info(self, model_name: str) -> None:
        """Mirror InferenceBackend._load_chat_template_info for MLX.

        Stores ``chat_template_info`` on ``self.models[model_name]`` with the
        resolved ``tokenizer.chat_template``."""
        entry = self.models.get(model_name)
        if not entry:
            return
        tok = entry.get("tokenizer")
        if tok is None:
            proc = entry.get("processor")
            tok = getattr(proc, "tokenizer", None) if proc else None
        info = {
            "has_template": False,
            "template": None,
            "format_type": "generic",
            "special_tokens": {},
            "template_name": None,
        }
        try:
            tpl = getattr(tok, "chat_template", None)
            if tpl:
                info["has_template"] = True
                info["template"] = tpl
                lower = tpl.lower()
                if "start_header_id" in lower and "end_header_id" in lower:
                    info["format_type"] = "llama3"
                elif "[inst]" in lower and "[/inst]" in lower:
                    info["format_type"] = "mistral"
                elif "<|im_start|>" in lower and "<|im_end|>" in lower:
                    info["format_type"] = "chatml"
                else:
                    info["format_type"] = "custom"
                special = {}
                for attr in ("bos_token", "eos_token", "pad_token"):
                    val = getattr(tok, attr, None)
                    if val:
                        special[attr] = val
                info["special_tokens"] = special
        except Exception as exc:
            logger.warning("MLX chat_template_info capture failed: %s", exc)
        entry["chat_template_info"] = info

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
        # Reasoning / tool kwargs forwarded by the route + worker; rendered via
        # apply_chat_template_for_generation like the transformers path.
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
    ) -> Generator[str, None, None]:
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Reset so a failed run cannot surface stale stats.
        self.last_generation_stats = None

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
                        # Prepend image if not already present
                        has_image = any(
                            p.get("type") == "image" for p in content if isinstance(p, dict)
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
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
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
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
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
        *,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
    ):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            render_with_native_template_fallback,
        )

        prompt = apply_chat_template_for_generation(
            self._tokenizer,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )
        if prompt is None:
            raise RuntimeError("apply_chat_template returned None — tokenizer may be incompatible")

        # If tools were requested but the (possibly overridden) template ignored
        # them, fall back to the model's native template -- same parity fix the
        # transformers backend applies, so MLX text models keep advertising tools.
        # ``self._tokenizer`` is this entry's ``model_info["tokenizer"]``, so the
        # probe and native render use the same renderer. (The VLM path renders via
        # the processor for image tokens and is intentionally not wired here.)
        model_info = self.models.get(self.active_model_name, {})
        prompt = render_with_native_template_fallback(
            formatted_prompt = prompt,
            tokenizer = self._tokenizer,
            model_info = model_info,
            active_model_name = self.active_model_name,
            messages = messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            hf_token = model_info.get("hf_token"),
        )

        sampler = make_sampler(
            temp = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
            min_tokens_to_keep = 1,
        )
        # Only build a logits processor for a non-trivial repetition penalty.
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
            final_response = None
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
                    final_response = response
                    token_ids.append(response.token)
                    # Decode full sequence with skip_special_tokens
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
            finally:
                # Latch final cumulative stats for the usage/timings chunk.
                if final_response is not None:
                    self.last_generation_stats = _build_generation_stats(
                        getattr(final_response, "prompt_tokens", 0),
                        getattr(final_response, "prompt_tps", 0.0),
                        getattr(final_response, "generation_tokens", 0),
                        getattr(final_response, "generation_tps", 0.0),
                    )

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
        *,
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
    ):
        from mlx_vlm import stream_generate as vlm_stream

        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
        )

        # Pick the chat-template-aware caller: processors with their own
        # apply_chat_template + chat_template (e.g. Qwen2.5-VL) use it
        # directly; else fall back to the nested tokenizer.
        chat_target = self._processor
        if (
            getattr(self._processor, "apply_chat_template", None) is None
            or not hasattr(self._processor, "chat_template")
            or self._processor.chat_template is None
        ):
            chat_target = getattr(self._processor, "tokenizer", self._processor)

        prompt = apply_chat_template_for_generation(
            chat_target,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )

        # mlx_vlm's stream_generate handles pixel_values (None for text-only)
        images = [image] if image is not None else None

        cumulative = ""
        logger.info(
            "VLM generating: prompt_len=%d, has_image=%s",
            len(prompt),
            image is not None,
        )
        # mlx_vlm.stream_generate forwards **kwargs into generate_step, which
        # builds the sampler + logits_processors internally.
        # GOTCHA: generate_step expects ``temperature=`` (long form); ``temp=``
        # silently falls into **kwargs and is ignored, stuck at greedy 0.0.
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
            final_response = None
            try:
                for response in vlm_stream(
                    self._model,
                    self._processor,
                    prompt,
                    images,
                    **vlm_kwargs,
                ):
                    final_response = response
                    token_text = response.text if hasattr(response, "text") else str(response)
                    cumulative += token_text
                    yield cumulative
                    if cancel_event and cancel_event.is_set():
                        break
            finally:
                # mlx_vlm exposes the same stats fields as mlx_lm.
                if final_response is not None:
                    self.last_generation_stats = _build_generation_stats(
                        getattr(final_response, "prompt_tokens", 0),
                        getattr(final_response, "prompt_tps", 0.0),
                        getattr(final_response, "generation_tokens", 0),
                        getattr(final_response, "generation_tps", 0.0),
                    )

    def generate_with_adapter_control(
        self,
        use_adapter = None,
        cancel_event = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        # MLX LoRA adapter toggling not yet supported — generate normally
        yield from self.generate_chat_response(cancel_event = cancel_event, **gen_kwargs)

    def reset_generation_state(self):
        import mlx.core as mx
        import gc

        gc.collect()
        mx.clear_cache()
