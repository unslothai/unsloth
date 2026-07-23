# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Core inference backend."""

from unsloth import FastLanguageModel, FastVisionModel
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer, TextStreamer
from peft import PeftModel, PeftModelForCausalLM

import json
import sys
import torch
from pathlib import Path
from typing import Optional, Union, Generator, Tuple
from utils.models import ModelConfig, get_base_model_from_lora
from utils.paths import is_model_cached
from utils.transformers_dtype import dtype_kwargs
from utils.utils import format_error_message
from utils.hardware import (
    get_device,
    clear_gpu_cache,
    log_gpu_memory,
    get_device_map,
    raise_if_offloaded,
    get_visible_gpu_count,
)
from core.inference.audio_codecs import AudioCodecManager
from core.inference.runtime_context import runtime_context_length
from core.inference.message_content import content_to_text
from core.inference.chat_eos import (
    chat_eos_repair,
    resolve_chat_turn_end_eos_ids_using,
)
from core.inference.chat_template_helpers import (
    ReasoningChannelNormalizer,
    detect_reasoning_channel_markers,
    detect_think_prefill,
)
from core.inference.presence_penalty import _make_presence_penalty_processor
from io import StringIO
import structlog
from loggers import get_logger


logger = get_logger(__name__)


class HarmonyTextStreamer:
    """Streaming text decoder for the gpt-oss harmony channel protocol.

    gpt-oss emits multi-channel output via ``<|channel|>analysis<|message|>...``
    / ``<|channel|>final<|message|>...``. Plain skip_special_tokens streaming
    glues channel names to content. This decodes with skip_special_tokens=False
    and parses statefully: emit ``<think>`` on first analysis, stream analysis,
    emit ``</think>`` on first final, stream final. Tracking per-channel lengths
    avoids the delta-on-transformed bug where wrapping tags shift position.
    Same put/end/iterator interface as TextIteratorStreamer.
    """

    import re as _re

    _HARMONY_RE = _re.compile(
        r"<\|channel\|>(\w+)<\|message\|>(.*?)(?=<\|end\|>|<\|channel\|>|\Z)",
        _re.DOTALL,
    )

    def __init__(
        self,
        tokenizer,
        *,
        skip_prompt: bool = True,
        timeout: float = 0.2,
    ):
        import queue

        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout

        self._queue: queue.Queue = queue.Queue()
        self._token_ids: list = []
        self._prompt_len: int = 0
        self._is_first_put: bool = True
        self._stop: bool = False

        # Stateful channel tracking avoids delta-on-transformed bugs
        self._emitted_think_open: bool = False
        self._emitted_think_close: bool = False
        self._analysis_emitted: int = 0  # chars of analysis content emitted
        self._final_emitted: int = 0  # chars of final content emitted

    # put / end — called from the generation thread

    def put(self, value):
        """Receive new token IDs from model.generate()."""
        import torch

        if isinstance(value, torch.Tensor):
            # shape (batch, seq) — take first batch element
            ids = value[0].tolist() if value.dim() > 1 else value.tolist()
        elif isinstance(value, (list, tuple)):
            ids = list(value)
        else:
            ids = [value]

        if self._is_first_put and self.skip_prompt:
            # First call is the full prompt; remember its length.
            self._prompt_len = len(ids)
            self._token_ids = list(ids)
            self._is_first_put = False
            return

        self._token_ids.extend(ids)

        # Decode only the generated part (after the prompt).
        gen_ids = self._token_ids[self._prompt_len :]
        raw = self.tokenizer.decode(gen_ids, skip_special_tokens = False)
        self._process_incremental(raw)

    def end(self):
        """Signal generation is complete."""
        # Final decode to capture remaining content.
        gen_ids = self._token_ids[self._prompt_len :]
        if gen_ids:
            raw = self.tokenizer.decode(gen_ids, skip_special_tokens = False)
            self._process_incremental(raw)

        # Close any open think tags.
        if self._emitted_think_open and not self._emitted_think_close:
            self._queue.put("</think>")
            self._emitted_think_close = True

        self._stop = True
        self._queue.put(None)  # sentinel

    # Iterator interface — consumed by the streaming loop

    def __iter__(self):
        return self

    def __next__(self):
        from queue import Empty
        while True:
            try:
                val = self._queue.get(timeout = self.timeout)
            except Empty:
                if self._stop:
                    raise StopIteration
                raise  # propagate Empty so caller can check thread liveness
            if val is None:
                raise StopIteration
            return val

    # Stateful incremental harmony protocol parsing

    def _process_incremental(self, raw: str) -> None:
        """Parse harmony channels and emit per-channel deltas (tracked by length, not whole-text diff)."""
        # If raw has <|channel|> but no complete channel+message pair yet, buffer.
        has_channel_token = "<|channel|>" in raw
        matches = list(self._HARMONY_RE.finditer(raw))

        if has_channel_token and not matches:
            # Partial harmony markup still building — wait for more tokens.
            return

        if not has_channel_token and not matches:
            return

        for m in matches:
            channel = m.group(1).lower()
            content = m.group(2)

            if channel == "analysis":
                if not self._emitted_think_open:
                    self._queue.put("<think>")
                    self._emitted_think_open = True

                new_content = content[self._analysis_emitted :]
                if new_content:
                    self._analysis_emitted = len(content)
                    self._queue.put(new_content)

            elif channel in ("final", "assistant"):
                if self._emitted_think_open and not self._emitted_think_close:
                    self._queue.put("</think>")
                    self._emitted_think_close = True

                new_content = content[self._final_emitted :]
                if new_content:
                    self._final_emitted = len(content)
                    self._queue.put(new_content)


class ReasoningTextIteratorStreamer(TextIteratorStreamer):
    """TextIteratorStreamer that preserves native channel tokens until parsed."""

    def __init__(
        self,
        tokenizer,
        *,
        markers: tuple[str, str],
        skip_prompt: bool = True,
        timeout: float = 0.2,
        cancel_event = None,
        **decode_kwargs,
    ):
        decode_kwargs["skip_special_tokens"] = False
        super().__init__(tokenizer, skip_prompt = skip_prompt, timeout = timeout, **decode_kwargs)
        self._normalizer = ReasoningChannelNormalizer(*markers)
        self._cancel_event = cancel_event
        self._aborted = False

    def abort(self):
        """Mark generation as failed so ``end`` drains without closing."""
        self._aborted = True

    def on_finalized_text(
        self,
        text: str,
        stream_end: bool = False,
    ):
        """Queue canonical deltas, closing only on natural stream completion."""
        delta = self._normalizer.feed(text)
        if delta:
            self.text_queue.put(delta, timeout = self.timeout)

        if stream_end:
            cancelled = self._aborted or (
                self._cancel_event is not None and self._cancel_event.is_set()
            )
            tail = self._normalizer.drain() if cancelled else self._normalizer.finish()
            if tail:
                self.text_queue.put(tail, timeout = self.timeout)
            self.text_queue.put(self.stop_signal, timeout = self.timeout)


class _GenerationThreadError(RuntimeError):
    """Generation worker failures that should propagate through stream routes."""


class InferenceBackend:
    """Unified inference backend supporting text, vision, and LoRA models"""

    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []  # [(display_name, path), ...]
        from core.inference.defaults import get_default_models

        self.default_models = get_default_models()
        self.device = get_device().value
        self._audio_codec_manager = AudioCodecManager()

        # _generation_lock serializes model.generate(). Plain Lock (NOT RLock):
        # RLock reentrancy would let concurrent compare-mode requests race on
        # the GPU. Acquired by the background generation thread, not the event-loop.
        import threading

        self._generation_lock = threading.Lock()
        self._model_state_lock = threading.Lock()

        logger.info(f"InferenceBackend initialized on {self.device}")

    @staticmethod
    def _normalize_top_k(top_k: int) -> int:
        # API uses -1 to disable top-k; transformers uses 0.
        return 0 if top_k < 0 else top_k

    def _resolve_chat_eos(self, model_name: str) -> None:
        """Resolve this chat model's assistant-turn-end stop tokens once at load,
        cache them in model_info, and repair generation_config so every
        ``.generate()`` path stops at the turn boundary.

        Some checkpoints (e.g. Qwen3.5 / Qwen3.6 small chat models) end turns with
        ``<|im_end|>`` but ship ``config.eos_token_id = <|endoftext|>`` and no
        ``generation_config.json``, so paths that read ``generation_config`` (the
        vision path, tool loops) run past the turn and loop. Turn-end markers are
        derived from the chat_template (see chat_eos.resolve_chat_turn_end_eos_ids),
        so base/coder models and harmony templates are left untouched.
        """
        info = self.models.get(model_name) or {}
        model = info.get("model")
        container = info.get("tokenizer")
        tokenizer = getattr(container, "tokenizer", container)  # unwrap processors
        if model is None or tokenizer is None:
            return
        # Vision models carry the chat_template on the processor, not the inner
        # tokenizer. Read markers from whichever has one, but resolve ids on the
        # generation tokenizer, else the vision path misses the turn-end token.
        template_source = container if getattr(container, "chat_template", None) else tokenizer
        try:
            turn_end_ids = resolve_chat_turn_end_eos_ids_using(template_source, tokenizer)
        except Exception as e:  # never block a load on eos resolution
            logger.warning("Chat turn-end eos resolution failed for %s: %s", model_name, e)
            return
        info["chat_turn_end_eos_ids"] = turn_end_ids

        gen = getattr(model, "generation_config", None)
        if gen is None:
            return
        repaired = chat_eos_repair(gen.eos_token_id, turn_end_ids)
        if repaired is None:
            return
        previous = gen.eos_token_id
        gen.eos_token_id = repaired
        logger.info(
            "Repaired generation_config.eos_token_id for %s: %s -> %s",
            model_name,
            previous,
            repaired,
        )

    def load_model(
        self,
        config: ModelConfig,
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
        trust_remote_code: bool = False,
        gpu_ids: Optional[list[int]] = None,
    ) -> bool:
        """Load any model: base, LoRA adapter, text, or vision."""
        # Keep the token so the native-template fallback can fetch a
        # gated model's repo template later during generation.
        self._hf_token = hf_token
        # GGUF uses max_seq_length=0 as "model default"; Unsloth crashes on it.
        if max_seq_length <= 0:
            max_seq_length = 2048

        try:
            model_name = config.identifier

            # Already loaded?
            if model_name in self.models and self.models[model_name].get("model"):
                logger.info(f"Model {model_name} already loaded")
                if hf_token:
                    self.models[model_name]["hf_token"] = hf_token
                self.active_model_name = model_name
                return True

            # Currently loading?
            if model_name in self.loading_models:
                logger.info(f"Model {model_name} is already being loaded")
                return False

            self.loading_models.add(model_name)
            device_map = get_device_map(gpu_ids)
            logger.info(
                f"Using device_map='{device_map}' ({get_visible_gpu_count()} GPU(s) visible)"
            )

            self.models[model_name] = {
                # Per-model token: the native-template fallback must use the
                # token this model was loaded with, not whichever loaded last.
                "hf_token": hf_token,
                # Per-model consent: the native-template reload must re-use the
                # exact trust_remote_code this model (and a LoRA's base) was loaded
                # with, so a custom-code tokenizer repo can be re-fetched without
                # executing any code the user did not already consent to.
                "trust_remote_code": trust_remote_code,
                "is_vision": config.is_vision,
                "is_lora": config.is_lora,
                "is_audio": config.is_audio,
                "audio_type": config.audio_type,
                "has_audio_input": config.has_audio_input,
                "model_path": config.path,
                "base_model": config.base_model if config.is_lora else None,
                "loaded_adapters": {},
                "active_adapter": None,
            }

            # ── Audio model loading path ──────────────────────────
            if config.is_audio:
                audio_type = config.audio_type
                adapter_info = " (LoRA adapter)" if config.is_lora else ""
                logger.info(f"Loading audio ({audio_type}) model{adapter_info}: {model_name}")
                log_gpu_memory(f"Before loading {model_name}")

                if audio_type == "csm":
                    from unsloth import FastModel
                    from transformers import CsmForConditionalGeneration

                    model, processor = FastModel.from_pretrained(
                        config.path,
                        auto_model = CsmForConditionalGeneration,
                        load_in_4bit = False,
                        device_map = device_map,
                        token = hf_token if hf_token and hf_token.strip() else None,
                        trust_remote_code = trust_remote_code,
                    )
                    FastModel.for_inference(model)
                    self.models[model_name]["model"] = model
                    self.models[model_name]["tokenizer"] = processor
                    self.models[model_name]["processor"] = processor
                elif audio_type == "bicodec":
                    import os
                    from unsloth import FastModel

                    if config.is_lora and config.base_model:
                        # LoRA adapter: base_model is .../Spark-TTS-0.5B/LLM;
                        # BiCodec weights live in the parent dir.
                        base_path = config.base_model
                        if os.path.isdir(base_path):
                            abs_repo_path = os.path.abspath(os.path.dirname(base_path))
                        else:
                            # base_model is an HF ID — download it.
                            from huggingface_hub import snapshot_download

                            local_dir = base_path.split("/")[-1]
                            repo_path = snapshot_download(base_path, local_dir = local_dir)
                            abs_repo_path = os.path.abspath(repo_path)

                        logger.info(
                            f"Spark-TTS LoRA: loading adapter from {config.path}, BiCodec from {abs_repo_path}"
                        )
                        model, tokenizer = FastModel.from_pretrained(
                            config.path,
                            dtype = torch.float32,
                            load_in_4bit = False,
                            device_map = device_map,
                            token = hf_token if hf_token and hf_token.strip() else None,
                            trust_remote_code = trust_remote_code,
                        )
                    else:
                        # Base model: download full HF repo, load from /LLM subfolder
                        from huggingface_hub import snapshot_download

                        hf_repo = config.path
                        local_dir = hf_repo.split("/")[-1]
                        repo_path = snapshot_download(hf_repo, local_dir = local_dir)
                        abs_repo_path = os.path.abspath(repo_path)
                        llm_path = os.path.join(abs_repo_path, "LLM")
                        logger.info(
                            f"Spark-TTS: downloaded repo to {repo_path}, loading LLM from {llm_path}"
                        )

                        model, tokenizer = FastModel.from_pretrained(
                            llm_path,
                            dtype = torch.float32,
                            load_in_4bit = False,
                            device_map = device_map,
                            token = hf_token if hf_token and hf_token.strip() else None,
                            trust_remote_code = trust_remote_code,
                        )

                    FastModel.for_inference(model)
                    self.models[model_name]["model"] = model
                    self.models[model_name]["tokenizer"] = tokenizer
                    self.models[model_name]["model_repo_path"] = abs_repo_path
                elif audio_type == "dac":
                    # OuteTTS uses FastModel (not FastLanguageModel)
                    from unsloth import FastModel

                    model, tokenizer = FastModel.from_pretrained(
                        config.path,
                        max_seq_length = max_seq_length,
                        load_in_4bit = False,
                        device_map = device_map,
                        token = hf_token if hf_token and hf_token.strip() else None,
                        trust_remote_code = trust_remote_code,
                    )
                    FastModel.for_inference(model)
                    self.models[model_name]["model"] = model
                    self.models[model_name]["tokenizer"] = tokenizer
                elif audio_type == "whisper":
                    # Whisper ASR — uses FastModel with WhisperForConditionalGeneration
                    from unsloth import FastModel
                    from transformers import WhisperForConditionalGeneration

                    model, tokenizer = FastModel.from_pretrained(
                        config.path,
                        auto_model = WhisperForConditionalGeneration,
                        whisper_language = "English",
                        whisper_task = "transcribe",
                        load_in_4bit = False,
                        device_map = device_map,
                        token = hf_token if hf_token and hf_token.strip() else None,
                        trust_remote_code = trust_remote_code,
                    )
                    FastModel.for_inference(model)
                    model.eval()

                    # ASR pipeline (per notebook)
                    from transformers import pipeline as hf_pipeline

                    whisper_pipe = hf_pipeline(
                        "automatic-speech-recognition",
                        model = model,
                        tokenizer = tokenizer.tokenizer,
                        feature_extractor = tokenizer.feature_extractor,
                        processor = tokenizer,
                        return_language = True,
                        **dtype_kwargs(torch.float16),
                    )
                    self.models[model_name]["model"] = model
                    self.models[model_name]["tokenizer"] = tokenizer
                    self.models[model_name]["whisper_pipeline"] = whisper_pipe
                else:
                    # SNAC (Orpheus) uses FastLanguageModel
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name = config.path,
                        max_seq_length = max_seq_length,
                        load_in_4bit = False,
                        device_map = device_map,
                        token = hf_token if hf_token and hf_token.strip() else None,
                        trust_remote_code = trust_remote_code,
                    )
                    FastLanguageModel.for_inference(model)
                    self.models[model_name]["model"] = model
                    self.models[model_name]["tokenizer"] = tokenizer

                # Load external codec for TTS audio types
                # (Whisper is ASR, audio_vlm is audio input — neither needs one)
                if audio_type not in ("whisper", "audio_vlm"):
                    model_repo_path = self.models[model_name].get("model_repo_path")
                    self._audio_codec_manager.load_codec(
                        audio_type, self.device, model_repo_path = model_repo_path
                    )

                # Reject CPU/disk offload for audio models too
                raise_if_offloaded(self.models[model_name]["model"], device_map, "Inference")
                self.models[model_name]["context_length"] = runtime_context_length(
                    self.models[model_name].get("model"),
                    max_seq_length,
                )

                self.active_model_name = model_name
                self.loading_models.discard(model_name)
                logger.info(f"Successfully loaded audio model: {model_name}")
                log_gpu_memory(f"After loading {model_name}")
                return True

            model_type = "vision" if config.is_vision else "text"
            adapter_info = " (LoRA adapter)" if self.models[model_name]["is_lora"] else ""
            logger.info(f"Loading {model_type} model{adapter_info}: {model_name}")
            log_gpu_memory(f"Before loading {model_name}")

            # Same load path for base models and LoRA adapters
            if config.is_vision:
                # Vision model (or vision LoRA adapter)
                model, processor = FastVisionModel.from_pretrained(
                    model_name = config.path,  # Can be base model OR LoRA adapter path
                    max_seq_length = max_seq_length,
                    dtype = dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    token = hf_token if hf_token and hf_token.strip() else None,
                    trust_remote_code = trust_remote_code,
                )

                FastVisionModel.for_inference(model)

                # FastVisionModel may return a raw tokenizer instead of a
                # Processor for some models (e.g. Gemma-3); load the real one.
                from transformers import ProcessorMixin

                if not (
                    isinstance(processor, ProcessorMixin) or hasattr(processor, "image_processor")
                ):
                    # LoRA adapters: use base model. Local merged exports: read base from export_metadata.json.
                    processor_source = config.base_model if config.is_lora else config.identifier
                    if not config.is_lora and config.is_local:
                        _meta_path = Path(config.path) / "export_metadata.json"
                        try:
                            if _meta_path.exists():
                                _meta = json.loads(_meta_path.read_text())
                                if _meta.get("base_model"):
                                    processor_source = _meta["base_model"]
                        except Exception:
                            pass
                    logger.warning(
                        f"FastVisionModel returned {type(processor).__name__} (no image_processor) "
                        f"for '{model_name}' — loading proper processor from '{processor_source}'"
                    )
                    from transformers import AutoProcessor

                    processor = AutoProcessor.from_pretrained(
                        processor_source,
                        token = hf_token if hf_token and hf_token.strip() else None,
                        trust_remote_code = trust_remote_code,
                    )
                    logger.info(f"Loaded {type(processor).__name__} from {processor_source}")

                self.models[model_name]["model"] = model
                self.models[model_name]["tokenizer"] = processor
                self.models[model_name]["processor"] = processor

            else:
                # Text model (or text LoRA adapter)
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = config.path,  # Can be base model OR LoRA adapter path
                    max_seq_length = max_seq_length,
                    dtype = dtype,
                    load_in_4bit = load_in_4bit,
                    device_map = device_map,
                    token = hf_token if hf_token and hf_token.strip() else None,
                    trust_remote_code = trust_remote_code,
                )

                FastLanguageModel.for_inference(model)

                self.models[model_name]["model"] = model
                self.models[model_name]["tokenizer"] = tokenizer

            raise_if_offloaded(self.models[model_name]["model"], device_map, "Inference")
            self.models[model_name]["context_length"] = runtime_context_length(
                self.models[model_name].get("model"),
                max_seq_length,
            )

            self._resolve_chat_eos(model_name)
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

    def unload_model(self, model_name: str) -> bool:
        """Remove a model from the registry and clear GPU memory."""
        if model_name in self.models:
            try:
                # Clean up codecs for audio models
                if self.models[model_name].get("is_audio"):
                    self._audio_codec_manager.unload()

                logger.info(f"Unloading model '{model_name}' from memory.")
                del self.models[model_name]

                # Clear the active model if it was the one unloaded
                if self.active_model_name == model_name:
                    self.active_model_name = None

                clear_gpu_cache()

                # Drop stale compiled cache for the next model. On spawn platforms,
                # preserve trainer files so concurrent dataset.map() workers can import them.
                import sys as _sys
                from utils.cache_cleanup import clear_unsloth_compiled_cache

                _preserve = ["Unsloth*Trainer.py"] if _sys.platform in ("win32", "darwin") else None
                clear_unsloth_compiled_cache(preserve_patterns = _preserve)

                logger.info(f"Model '{model_name}' successfully unloaded.")
                return True
            except Exception as e:
                logger.error(f"Error while unloading model '{model_name}': {e}")
                return False
        else:
            logger.warning(
                f"Attempted to unload model '{model_name}', but it was not found in the registry."
            )
            return True

    def revert_to_base_model(self, base_model_name: str) -> bool:
        """Revert the model to its pristine base state by unloading and
        deleting all adapter configurations."""
        if base_model_name not in self.models:
            return False

        model = self.models[base_model_name].get("model")

        try:
            # Unload adapter weights if model is a PeftModel.
            if isinstance(model, (PeftModel, PeftModelForCausalLM)):
                logger.info(f"Unloading LoRA adapters from '{base_model_name}'...")
                unwrapped_base_model = model.unload()
                self.models[base_model_name]["model"] = unwrapped_base_model
                model = unwrapped_base_model

            # model.unload() can leave a peft_config; removing it avoids
            # "multiple adapters" warnings on the next from_pretrained().
            if hasattr(model, "peft_config"):
                del model.peft_config

            logger.info(f"Model '{base_model_name}' reverted to clean base state.")
            return True

        except Exception as e:
            logger.error(f"Failed to revert model to base state: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False

    def load_for_eval(
        self,
        lora_path: str,
        max_seq_length: int = 2048,
        dtype = None,
        load_in_4bit: bool = True,
        hf_token: Optional[str] = None,
        gpu_ids: Optional[list[int]] = None,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Ensure the base model and the given adapter are loaded.
        Idempotent and handles all states correctly.
        """
        try:
            from utils.models import ModelConfig

            lora_config = ModelConfig.from_lora_path(lora_path, hf_token)
            if not lora_config:
                return False, None, None

            base_model_name = lora_config.base_model

            # 1. Load the base model if not already in memory
            if base_model_name not in self.models or not self.models[base_model_name].get("model"):
                logger.info(f"Base model '{base_model_name}' not loaded, loading now.")
                base_config = ModelConfig.from_ui_selection(base_model_name, None, is_lora = False)
                if not self.load_model(
                    base_config,
                    max_seq_length,
                    dtype,
                    load_in_4bit,
                    hf_token,
                    gpu_ids = gpu_ids,
                ):
                    return False, None, None

            self.active_model_name = base_model_name

            # 2. Derive adapter name from the user's selection
            adapter_name = lora_path.split("/")[-1].replace(".", "_")

            # 3. Ensure this adapter is loaded (load_adapter only reads from
            # disk if the model doesn't already have it).
            adapter_success = self.load_adapter(
                base_model_name = base_model_name,
                adapter_path = lora_path,
                adapter_name = adapter_name,
            )
            if not adapter_success:
                return False, base_model_name, None

            # 4. Return the verified adapter name for the UI.
            return True, base_model_name, adapter_name

        except Exception as e:
            logger.error(f"Error during load_for_eval: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return False, None, None

    def load_adapter(self, base_model_name: str, adapter_path: str, adapter_name: str) -> bool:
        """Load an adapter onto the model only if not already attached."""
        model = self.models[base_model_name].get("model")

        # Most reliable check: adapter name already in the model's config.
        if hasattr(model, "peft_config") and adapter_name in model.peft_config:
            logger.info(
                f"Adapter '{adapter_name}' is already attached to the model. Skipping load."
            )
            return True

        try:
            logger.info(
                f"Loading new adapter '{adapter_name}' from '{adapter_path}' onto {base_model_name}"
            )
            model.load_adapter(adapter_path, adapter_name = adapter_name)

            # Update the registry only after a successful load.
            if "loaded_adapters" not in self.models[base_model_name]:
                self.models[base_model_name]["loaded_adapters"] = {}
            self.models[base_model_name]["loaded_adapters"][adapter_name] = adapter_path

            total_adapters = len(getattr(model, "peft_config", {}))
            logger.info(
                f"Adapter '{adapter_name}' loaded successfully. (Total unique adapters on model: {total_adapters})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load adapter '{adapter_name}': {e}")
            return False

    def set_active_adapter(self, base_model_name: str, adapter_name: str) -> bool:
        """Set the active adapter for generation."""
        model = self.models[base_model_name].get("model")
        try:
            logger.info(f"Setting active adapter to: '{adapter_name}'")
            model.set_adapter(adapter_name)
            self.models[base_model_name]["active_adapter"] = adapter_name
            return True
        except Exception as e:
            # Catches "adapter not found" if something goes wrong.
            logger.error(f"Failed to set active adapter to '{adapter_name}': {e}")
            return False

    def _apply_adapter_state(self, use_adapter: Optional[Union[bool, str]]) -> None:
        """Apply adapter state before generation (must hold _generation_lock).

        Toggles PEFT enable/disable_adapter_layers (non-destructive, no reload).
        use_adapter: None = no change, False = base model, True = current adapter,
        str = named adapter.
        """
        if use_adapter is None:
            return

        base = self.active_model_name
        if not base or base not in self.models:
            return

        model_info = self.models[base]
        model = model_info.get("model")
        if model is None:
            return

        if use_adapter is False:
            # Disable LoRA layers -> base model output.
            if isinstance(model, (PeftModel, PeftModelForCausalLM)):
                logger.info(
                    f"Compare mode: disabling adapters on '{base}' for base model generation"
                )
                model.base_model.disable_adapter_layers()
            else:
                logger.info(f"Compare mode: model '{base}' is not a PeftModel, already base")

        elif use_adapter is True:
            # Re-enable LoRA layers -> adapter output.
            if isinstance(model, (PeftModel, PeftModelForCausalLM)):
                logger.info(f"Compare mode: enabling adapters on '{base}' for LoRA generation")
                model.base_model.enable_adapter_layers()
            else:
                logger.warning("use_adapter=true but model is not a PeftModel")

        elif isinstance(use_adapter, str):
            # Enable adapters and set the named one active.
            if isinstance(model, (PeftModel, PeftModelForCausalLM)):
                logger.info(f"Compare mode: enabling adapter '{use_adapter}' on '{base}'")
                model.base_model.enable_adapter_layers()
                self.set_active_adapter(base, use_adapter)
            else:
                logger.warning(f"use_adapter='{use_adapter}' but model is not a PeftModel")

    def generate_with_adapter_control(
        self,
        use_adapter: Optional[Union[bool, str]] = None,
        cancel_event = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        """Thread-safe generation with optional adapter toggling.

        Adapter toggle + model.generate() are serialized by _generation_lock in
        the background thread, avoiding the RLock-reentrant race when two async
        SSE handlers share one event-loop thread. use_adapter: see _apply_adapter_state.
        """
        yield from self._generate_chat_response_inner(
            cancel_event = cancel_event, _adapter_state = use_adapter, **gen_kwargs
        )

    def generate_chat_completion_with_tools(
        self,
        messages: list,
        tools: list,
        system_prompt: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        max_tool_iterations: int = 25,
        auto_heal_tool_calls: bool = True,
        nudge_tool_calls: Optional[bool] = None,
        tool_call_timeout: int = 300,
        session_id: Optional[str] = None,
        thread_id: Optional[str] = None,
        rag_scope: Optional[dict] = None,
        presence_penalty: float = 0.0,
        reasoning_prefilled: bool = False,
    ):
        """Run an agentic tool loop on top of ``generate_chat_response``.

        Yields the same event-dict protocol as the GGUF path so the route
        layer can stream both backends through one helper. Each event is one of:

        * ``{"type": "status", "text": ...}``
        * ``{"type": "content", "text": cumulative_text}``
        * ``{"type": "tool_start", "tool_name", "tool_call_id", "arguments"}``
        * ``{"type": "tool_end", "tool_name", "tool_call_id", "result"}``
        """
        from core.inference.safetensors_agentic import run_safetensors_tool_loop
        from core.inference.tools import execute_tool

        def _single_turn(conv: list, *, active_tools: Optional[list[dict]] = None):
            # conv already has the system message -- avoid double-prepend.
            # `active_tools` is supplied by run_safetensors_tool_loop so one-shot
            # tools such as render_html can be removed from later same-response prompts.
            turn_tools = active_tools if active_tools is not None else tools
            yield from self._generate_chat_response_inner(
                messages = conv,
                system_prompt = "",
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                max_new_tokens = max_new_tokens,
                repetition_penalty = repetition_penalty,
                cancel_event = cancel_event,
                tools = turn_tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                presence_penalty = presence_penalty,
            )

        initial = list(messages)
        if system_prompt:
            initial = [{"role": "system", "content": system_prompt}] + initial

        yield from run_safetensors_tool_loop(
            single_turn = _single_turn,
            messages = initial,
            tools = tools,
            execute_tool = execute_tool,
            cancel_event = cancel_event,
            auto_heal_tool_calls = auto_heal_tool_calls,
            nudge_tool_calls = nudge_tool_calls,
            max_tool_iterations = max_tool_iterations,
            tool_call_timeout = tool_call_timeout,
            session_id = session_id,
            thread_id = thread_id,
            rag_scope = rag_scope,
            reasoning_prefilled = reasoning_prefilled,
        )

    def generate_chat_response(
        self,
        messages: list,
        system_prompt: str,
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Generate response for text or vision models (lock held by background thread).

        ``tools`` / ``enable_thinking`` / ``reasoning_effort`` / ``preserve_thinking``
        are forwarded into ``apply_chat_template`` so templates that understand them
        (Qwen3, Llama 3.1+, gpt-oss harmony) advertise tool schemas / reasoning controls.
        ``presence_penalty`` matches the GGUF sampling path (0 disables it).
        """
        yield from self._generate_chat_response_inner(
            messages = messages,
            system_prompt = system_prompt,
            image = image,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            min_p = min_p,
            max_new_tokens = max_new_tokens,
            repetition_penalty = repetition_penalty,
            cancel_event = cancel_event,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
            presence_penalty = presence_penalty,
        )

    def _generate_chat_response_inner(
        self,
        messages: list,
        system_prompt: str = "",
        image = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        _adapter_state = None,
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Inner generation logic, called by generate_chat_response and
        generate_with_adapter_control.

        _adapter_state is passed to generate_stream/vision so the background
        thread can toggle adapters under the generation lock.
        """
        if not self.active_model_name:
            raise RuntimeError("No active model")

        model_info = self.models[self.active_model_name]
        is_vision = model_info.get("is_vision", False)
        tokenizer = model_info.get("tokenizer") or model_info.get("processor")
        # Unwrap processor -> raw tokenizer for VLMs on the text path.
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        top_k = self._normalize_top_k(top_k)

        if is_vision and image:
            # Verify the stored processor can handle images; FastVisionModel may
            # return a raw tokenizer instead of a ProcessorMixin (e.g. Gemma-3).
            from transformers import ProcessorMixin

            processor = model_info.get("processor")
            has_image_processing = processor is not None and (
                isinstance(processor, ProcessorMixin) or hasattr(processor, "image_processor")
            )
            if has_image_processing:
                yield from self._generate_vision_response(
                    messages,
                    system_prompt,
                    image,
                    temperature,
                    top_p,
                    top_k,
                    min_p,
                    max_new_tokens,
                    repetition_penalty,
                    cancel_event = cancel_event,
                    presence_penalty = presence_penalty,
                )
                return
            else:
                logger.warning(
                    f"Model '{self.active_model_name}' is marked as vision but its processor "
                    f"({type(processor).__name__}) has no image_processor — "
                    f"falling back to text-only generation (image will be ignored)."
                )

        # Text path: messages are already in ChatML format from eval.py.

        # Step 1: apply get_chat_template if model is in mapper.
        try:
            from utils.datasets import (
                MODEL_TO_TEMPLATE_MAPPER,
                get_tokenizer_chat_template,
            )
            model_name_lower = self.active_model_name.lower()

            if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
                template_name = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
                logger.info(
                    f"Applying chat template '{template_name}' for {self.active_model_name}"
                )

                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = template_name,
                )
                # The mapper installs the effective template only now, at generate
                # time, so re-resolve and UNION into the load-time cache (never
                # overwrite). get_chat_template can return a remapped tokenizer
                # (turn-end folded onto doc-eos) while generate_stream reads the
                # original, so take marker strings from the mapped template but
                # resolve their ids on the original.
                try:
                    _gen_tok = model_info.get("tokenizer") or tokenizer
                    refreshed = resolve_chat_turn_end_eos_ids_using(
                        getattr(tokenizer, "tokenizer", tokenizer),
                        getattr(_gen_tok, "tokenizer", _gen_tok),
                    )
                    existing = model_info.get("chat_turn_end_eos_ids") or []
                    model_info["chat_turn_end_eos_ids"] = sorted(set(existing) | set(refreshed))
                except Exception as e:
                    logger.warning(f"Could not refresh chat turn-end eos after template: {e}")
            else:
                logger.info(
                    f"No registered Unsloth template for {self.active_model_name}, using tokenizer default"
                )
        except Exception as e:
            logger.warning(f"Could not apply get_chat_template: {e}")

        # Step 2: format with tokenizer.apply_chat_template().
        if system_prompt:
            template_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            template_messages = messages
        reasoning_channel_markers_resolved = False
        try:
            if not (hasattr(tokenizer, "chat_template") and tokenizer.chat_template):
                raise ValueError(
                    f"Model '{self.active_model_name}' has no chat_template set in its "
                    f"tokenizer_config.json. This is usually a problem with the model's "
                    f"HuggingFace repository — it is missing a 'chat_template' key. "
                    f"Please use a model that includes a chat template, or manually set "
                    f"one via tokenizer.chat_template before inference."
                )
            reasoning_channel_markers = None
            formatted_prompt = self._apply_chat_template_for_generation(
                tokenizer,
                template_messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
            )

            # If tools were requested but the (possibly overridden) template ignored
            # them, fall back to the model's native template (shared with MLX).
            from core.inference.chat_template_helpers import (
                render_with_native_template_fallback,
            )

            render_result = render_with_native_template_fallback(
                formatted_prompt = formatted_prompt,
                tokenizer = tokenizer,
                model_info = model_info,
                active_model_name = self.active_model_name,
                messages = template_messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
                apply_fn = self._apply_chat_template_for_generation,
                hf_token = model_info.get("hf_token"),
                return_metadata = True,
            )
            formatted_prompt = render_result.prompt
            reasoning_channel_markers = render_result.reasoning_channel_markers
            reasoning_channel_markers_resolved = True

            logger.debug(f"Formatted prompt: {formatted_prompt[:200]}...")
        except Exception as e:
            logger.error(f"Error applying chat template: {e}")
            # Fall back to manual formatting
            from core.inference.chat_template_helpers import (
                neutralize_control_markup_in_messages,
                neutralize_non_assistant_control_markup,
            )

            safe_messages = neutralize_control_markup_in_messages(messages)
            safe_system = (
                neutralize_non_assistant_control_markup(system_prompt) if system_prompt else None
            )
            formatted_prompt = self.format_chat_prompt(safe_messages, safe_system)
            reasoning_channel_markers = None
            reasoning_channel_markers_resolved = True

        # Step 3: generate
        yield from self.generate_stream(
            formatted_prompt,
            temperature,
            top_p,
            top_k,
            min_p,
            max_new_tokens,
            repetition_penalty,
            cancel_event = cancel_event,
            _adapter_state = _adapter_state,
            presence_penalty = presence_penalty,
            reasoning_channel_markers = reasoning_channel_markers,
            reasoning_channel_markers_resolved = reasoning_channel_markers_resolved,
        )

    def _generate_vision_response(
        self,
        messages,
        system_prompt,
        image,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event = None,
        presence_penalty: float = 0.0,
    ) -> Generator[str, None, None]:
        """Handle vision model generation with true token-by-token streaming."""
        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        processor = model_info["processor"]
        # FastVisionModel may return a raw tokenizer (e.g. GemmaTokenizerFast)
        # for some models. Safe unwrap for tokenize-only ops.
        raw_tokenizer = getattr(processor, "tokenizer", processor)

        from core.inference.chat_template_helpers import (
            neutralize_control_markup_in_messages,
            neutralize_non_assistant_control_markup,
        )

        safe_messages = neutralize_control_markup_in_messages(messages)
        safe_system = (
            neutralize_non_assistant_control_markup(system_prompt) if system_prompt else None
        )

        # Extract user message (after neutralization so literal control markup is safe).
        user_message = ""
        if safe_messages and safe_messages[-1]["role"] == "user":
            import re
            user_message = content_to_text(safe_messages[-1]["content"])
            user_message = re.sub(r"<img[^>]*>", "", user_message).strip()

        if not user_message:
            user_message = "Describe this image." if image else "Hello"

        # Prepare vision messages
        if image:
            user_msg = {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_message},
                ],
            }
            if safe_system:
                vision_messages = [
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": safe_system}],
                    },
                    user_msg,
                ]
            else:
                vision_messages = [user_msg]

            try:
                input_text = processor.apply_chat_template(
                    vision_messages, add_generation_prompt = True, tokenize = False
                )
            except Exception as e:
                if system_prompt:
                    logger.warning(
                        f"Vision processor for '{self.active_model_name}' may not support "
                        f"system messages; retrying without. Original error: {e}"
                    )
                    vision_messages = [user_msg]
                    input_text = processor.apply_chat_template(
                        vision_messages, add_generation_prompt = True, tokenize = False
                    )
                else:
                    raise
            inputs = processor(
                image,
                input_text,
                add_special_tokens = False,
                return_tensors = "pt",
            ).to(model.device)
            prompt_text = input_text
        else:
            # Text-only path for a vision model
            formatted_prompt = self.format_chat_prompt(safe_messages, safe_system)
            inputs = raw_tokenizer(formatted_prompt, return_tensors = "pt").to(model.device)
            prompt_text = formatted_prompt

        # Stream with TextIteratorStreamer + background thread
        try:
            # Re-emit an open <think> prefill swallowed by skip_prompt (see
            # generate_stream).
            think_prefix = detect_think_prefill(
                prompt_text, getattr(raw_tokenizer, "all_special_tokens", None)
            )
            import threading

            streamer = self._make_text_streamer(
                raw_tokenizer,
                protocol_source = processor,
                # The text-only VLM fallback above did not render with the
                # processor template, so its native markers do not describe
                # this request's response protocol.
                reasoning_channel_markers = detect_reasoning_channel_markers(processor)
                if image
                else None,
                reasoning_channel_markers_resolved = True,
                skip_prompt = True,
                timeout = 0.2,
                cancel_event = cancel_event,
                use_harmony = self._is_gpt_oss_model(),
            )

            generation_kwargs = dict(
                **inputs,
                streamer = streamer,
                max_new_tokens = max_new_tokens,
                use_cache = True,
                do_sample = temperature > 0,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
            )
            # Presence penalty (GGUF parity) for VLM chat.
            _vision_input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
            if _vision_input_ids is not None:
                _pp = _make_presence_penalty_processor(
                    presence_penalty, int(_vision_input_ids.shape[1])
                )
                if _pp is not None:
                    generation_kwargs["logits_processor"] = _pp
            stopping_criteria = self._cancel_stopping_criteria(cancel_event)
            if stopping_criteria is not None:
                generation_kwargs["stopping_criteria"] = stopping_criteria
            active_stop_token_ids = self._generation_stop_token_ids(model, generation_kwargs)

            err: dict[str, str] = {}

            def generate_fn():
                with self._generation_lock:
                    try:
                        model.generate(**generation_kwargs)
                    except Exception as e:
                        err["msg"] = str(e)
                        if hasattr(streamer, "abort"):
                            streamer.abort()
                        logger.error(f"Vision generation error in thread: {e}")
                    finally:
                        try:
                            streamer.end()
                        except Exception:
                            pass

            thread = threading.Thread(target = generate_fn)
            thread.start()

            output = think_prefix
            # Emit the prefilled <think> before the first token so the block
            # renders during prompt prefill (which can take seconds).
            if think_prefix:
                yield think_prefix
            from queue import Empty
            import time

            generation_complete = False
            cancel_deadline = None
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        if cancel_deadline is None:
                            cancel_deadline = time.monotonic() + 10
                        elif time.monotonic() >= cancel_deadline:
                            break
                    try:
                        new_token = next(streamer)
                    except StopIteration:
                        generation_complete = True
                        break
                    except Empty:
                        if not thread.is_alive():
                            generation_complete = True
                            output = yield from self._drain_streamer_tail(
                                streamer, output, active_stop_token_ids
                            )
                            break
                        if cancel_deadline is not None:
                            remaining = cancel_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            thread.join(timeout = remaining)
                            if thread.is_alive():
                                break
                            generation_complete = True
                            output = yield from self._drain_streamer_tail(
                                streamer, output, active_stop_token_ids
                            )
                            break
                        continue
                    if new_token:
                        output, cleaned = self._append_stream_delta(
                            output, new_token, active_stop_token_ids
                        )
                        yield cleaned
            finally:
                if cancel_event is not None and not generation_complete:
                    cancel_event.set()
                join_timeout = 10
                if cancel_deadline is not None:
                    join_timeout = max(0, cancel_deadline - time.monotonic())
                thread.join(timeout = join_timeout)
                if thread.is_alive():
                    logger.warning(
                        "Vision generation thread did not exit after cancel/join timeout"
                    )

            if err.get("msg"):
                raise _GenerationThreadError(err["msg"])

        except _GenerationThreadError:
            raise
        except Exception as e:
            logger.error(f"Vision generation error: {e}")
            raise

    def generate_audio_input_response(
        self,
        messages,
        system_prompt,
        audio_array,
        temperature,
        top_p,
        top_k,
        min_p,
        max_new_tokens,
        repetition_penalty,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Audio-input (ASR) generation: takes an audio numpy array, streams text.

        Uses processor.apply_chat_template with audio embedded in messages (Gemma 3n pattern).
        """
        import threading
        import numpy as np

        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        processor = model_info.get("processor") or model_info.get("tokenizer")
        raw_tokenizer = getattr(processor, "tokenizer", processor)

        # Last user text; default matches the notebook prompt
        user_text = "Please transcribe this audio."
        if messages:
            for msg in reversed(messages):
                if msg["role"] == "user" and msg.get("content"):
                    user_text = content_to_text(msg["content"])
                    break

        # ASR-specific default system prompt if none set
        if not system_prompt:
            system_prompt = "You are an assistant that transcribes speech accurately."

        # Gemma 3n format — audio goes INTO apply_chat_template
        audio_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_array},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        # apply_chat_template does audio embedding + tokenization in one step
        inputs = processor.apply_chat_template(
            audio_messages,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensors = "pt",
            truncation = False,
        ).to(model.device)

        try:
            from transformers import TextIteratorStreamer
            from queue import Empty

            streamer = TextIteratorStreamer(
                raw_tokenizer,
                skip_prompt = True,
                skip_special_tokens = True,
                timeout = 0.2,
            )

            # Notebook uses do_sample=False (greedy) for ASR accuracy
            generation_kwargs = dict(
                **inputs,
                streamer = streamer,
                max_new_tokens = max_new_tokens,
                use_cache = True,
                do_sample = False,
            )

            err: dict[str, str] = {}

            def generate_fn():
                with self._generation_lock:
                    try:
                        model.generate(**generation_kwargs)
                    except Exception as e:
                        err["msg"] = str(e)
                        logger.error(f"Audio input generation error in thread: {e}")
                    finally:
                        try:
                            streamer.end()
                        except Exception:
                            pass

            thread = threading.Thread(target = generate_fn)
            thread.start()

            output = ""
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        break
                    try:
                        new_token = next(streamer)
                    except StopIteration:
                        break
                    except Empty:
                        if not thread.is_alive():
                            break
                        continue
                    if new_token:
                        output += new_token
                        yield new_token
            finally:
                if cancel_event is not None:
                    cancel_event.set()
                thread.join(timeout = 10)
                if thread.is_alive():
                    logger.warning(
                        "Audio input generation thread did not exit after cancel/join timeout"
                    )

            if err.get("msg"):
                raise _GenerationThreadError(err["msg"])

        except _GenerationThreadError:
            raise
        except Exception as e:
            logger.error(f"Audio input generation error: {e}")
            raise

    def generate_whisper_response(
        self,
        audio_array,
        cancel_event = None,
    ) -> Generator[str, None, None]:
        """Whisper ASR: takes an audio numpy array, yields transcribed text.

        Uses the pre-built transformers pipeline created at model load.
        """
        model_info = self.models[self.active_model_name]
        whisper_pipe = model_info.get("whisper_pipeline")
        if not whisper_pipe:
            yield "Error: Whisper pipeline not initialized"
            return

        try:
            with self._generation_lock:
                result = whisper_pipe({"raw": audio_array, "sampling_rate": 16000})

            text = result.get("text", "") if isinstance(result, dict) else str(result)
            if text:
                yield text
        except Exception as e:
            logger.error(f"Whisper ASR error: {e}")
            yield f"Error: {str(e)}"

    def _is_gpt_oss_model(self, model_name: str = None) -> bool:
        """Whether the given (or active) model uses the gpt-oss harmony protocol."""
        from utils.datasets import is_gpt_oss_model_name
        return is_gpt_oss_model_name(model_name or self.active_model_name or "")

    def _make_text_streamer(
        self,
        tokenizer,
        *,
        protocol_source = None,
        reasoning_channel_markers = None,
        reasoning_channel_markers_resolved: bool = False,
        skip_prompt: bool = True,
        timeout: float = 0.2,
        cancel_event = None,
        use_harmony: bool = False,
    ):
        """Create the streamer matching this model's native response protocol."""
        if use_harmony:
            try:
                return HarmonyTextStreamer(
                    tokenizer,
                    skip_prompt = skip_prompt,
                    timeout = timeout,
                )
            except Exception as e:
                logger.warning(f"HarmonyTextStreamer init failed, falling back: {e}")
                return TextIteratorStreamer(
                    tokenizer,
                    skip_prompt = skip_prompt,
                    skip_special_tokens = True,
                    timeout = timeout,
                )

        markers = (
            reasoning_channel_markers
            if reasoning_channel_markers_resolved
            else reasoning_channel_markers
            or detect_reasoning_channel_markers(protocol_source or tokenizer)
        )
        if markers is not None:
            return ReasoningTextIteratorStreamer(
                tokenizer,
                markers = markers,
                skip_prompt = skip_prompt,
                timeout = timeout,
                cancel_event = cancel_event,
            )
        return TextIteratorStreamer(
            tokenizer,
            skip_prompt = skip_prompt,
            skip_special_tokens = True,
            timeout = timeout,
        )

    def _append_stream_delta(
        self,
        output: str,
        new_token: str,
        stop_token_ids = None,
    ):
        """Append a streamer delta and apply response-boundary cleanup."""
        output += new_token
        return output, self._clean_generated_text(output, stop_token_ids = stop_token_ids)

    def _drain_streamer_tail(
        self,
        streamer,
        output: str,
        stop_token_ids = None,
    ):
        """Drain queued streamer text after the producer exits."""
        while True:
            try:
                new_token = next(streamer)
            except StopIteration:
                return output
            except Exception:
                return output
            if new_token:
                output, cleaned = self._append_stream_delta(
                    output, new_token, stop_token_ids = stop_token_ids
                )
                yield cleaned

    def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        min_p: float = 0.0,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
        cancel_event = None,
        _adapter_state = None,
        presence_penalty: float = 0.0,
        reasoning_channel_markers = None,
        reasoning_channel_markers_resolved: bool = False,
    ) -> Generator[str, None, None]:
        """Generate a streaming text response (text models only).

        _adapter_state: if not None, the background thread toggles adapters
        before model.generate(), under _generation_lock.
        ``presence_penalty`` matches the GGUF sampling path via a logits processor (0 disables it).
        """
        if not self.active_model_name:
            raise RuntimeError("No active model")

        model_info = self.models[self.active_model_name]
        model = model_info["model"]
        # For VLMs the stored "tokenizer" is actually the processor. Unwrap to
        # the real tokenizer so TextIteratorStreamer's skip_prompt /
        # skip_special_tokens work correctly.
        tokenizer = model_info["tokenizer"]
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        try:
            inputs = tokenizer(prompt, return_tensors = "pt").to(model.device)

            import threading

            # skip_prompt swallows an open <think> prefilled by the template;
            # re-emit it so the frontend can render the thinking block.
            # gpt-oss emits its own tags via HarmonyTextStreamer.
            think_prefix = (
                ""
                if self._is_gpt_oss_model()
                else detect_think_prefill(prompt, getattr(tokenizer, "all_special_tokens", None))
            )

            streamer = self._make_text_streamer(
                tokenizer,
                protocol_source = model_info.get("tokenizer"),
                reasoning_channel_markers = reasoning_channel_markers,
                reasoning_channel_markers_resolved = reasoning_channel_markers_resolved,
                skip_prompt = True,
                timeout = 0.2,
                cancel_event = cancel_event,
                use_harmony = self._is_gpt_oss_model(),
            )

            generation_kwargs = dict(
                **inputs,
                streamer = streamer,
                max_new_tokens = max_new_tokens,
                temperature = temperature,
                top_p = top_p,
                top_k = top_k,
                min_p = min_p,
                repetition_penalty = repetition_penalty,
                do_sample = temperature > 0,
                # Resolved once at load (chat_template-derived turn-end tokens).
                eos_token_id = model_info.get("chat_turn_end_eos_ids") or tokenizer.eos_token_id,
                pad_token_id = tokenizer.eos_token_id
                if tokenizer.pad_token_id is None
                else tokenizer.pad_token_id,
            )
            active_stop_token_ids = self._generation_stop_token_ids(model, generation_kwargs)
            # Presence penalty (GGUF parity); prompt_len excludes prompt tokens.
            _pp = _make_presence_penalty_processor(
                presence_penalty, int(inputs["input_ids"].shape[1])
            )
            if _pp is not None:
                generation_kwargs["logits_processor"] = _pp
            stopping_criteria = self._cancel_stopping_criteria(cancel_event)
            if stopping_criteria is not None:
                generation_kwargs["stopping_criteria"] = stopping_criteria

            def generate_fn():
                with self._generation_lock:
                    try:
                        if _adapter_state is not None:
                            self._apply_adapter_state(_adapter_state)
                        model.generate(**generation_kwargs)
                    except Exception as e:
                        err["msg"] = str(e)
                        if hasattr(streamer, "abort"):
                            streamer.abort()
                        logger.error(f"Generation error: {e}")
                    finally:
                        try:
                            streamer.end()
                        except Exception:
                            pass

            err: dict[str, str] = {}
            thread = threading.Thread(target = generate_fn)
            thread.start()

            output = think_prefix
            # Emit the prefilled <think> before the first token so the block
            # renders during prompt prefill (which can take seconds).
            if think_prefix:
                yield think_prefix
            from queue import Empty
            import time

            generation_complete = False
            cancel_deadline = None
            try:
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        if cancel_deadline is None:
                            cancel_deadline = time.monotonic() + 10
                        elif time.monotonic() >= cancel_deadline:
                            break
                    try:
                        new_token = next(streamer)
                    except StopIteration:
                        generation_complete = True
                        break
                    except Empty:
                        if not thread.is_alive():
                            generation_complete = True
                            output = yield from self._drain_streamer_tail(
                                streamer, output, active_stop_token_ids
                            )
                            break
                        if cancel_deadline is not None:
                            remaining = cancel_deadline - time.monotonic()
                            if remaining <= 0:
                                break
                            thread.join(timeout = remaining)
                            if thread.is_alive():
                                break
                            generation_complete = True
                            output = yield from self._drain_streamer_tail(
                                streamer, output, active_stop_token_ids
                            )
                            break
                        continue
                    if new_token:
                        output, cleaned = self._append_stream_delta(
                            output, new_token, active_stop_token_ids
                        )
                        yield cleaned
            finally:
                # Set cancel_event only on early exit (user cancel), NOT on
                # normal completion. It's a shared mp.Event; setting it
                # unconditionally would leave a stale cancel signal that could
                # disrupt the next serialized request (e.g. compare mode).
                if cancel_event is not None and not generation_complete:
                    cancel_event.set()
                join_timeout = 10
                if cancel_deadline is not None:
                    join_timeout = max(0, cancel_deadline - time.monotonic())
                thread.join(timeout = join_timeout)
                if thread.is_alive():
                    logger.warning("Generation thread did not exit after cancel/join timeout")

            if err.get("msg"):
                raise _GenerationThreadError(err["msg"])

        except _GenerationThreadError:
            raise
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise

    # ── Audio (TTS) Generation ────────────────────────────────────

    def generate_audio_response(
        self,
        text: str,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 50,
        min_p: float = 0.0,
        max_new_tokens: int = 2048,
        repetition_penalty: float = 1.0,
        use_adapter: Optional[Union[bool, str]] = None,
    ) -> Tuple[bytes, int]:
        """Generate audio from text for TTS models.
        Returns (wav_bytes, sample_rate). Blocking — full audio before return.
        """
        if not self.active_model_name:
            raise RuntimeError("No active model")

        model_info = self.models[self.active_model_name]
        audio_type = model_info.get("audio_type")
        model = model_info["model"]
        tokenizer = model_info.get("tokenizer")

        if not audio_type:
            raise RuntimeError(f"Model {self.active_model_name} is not an audio model")

        top_k = self._normalize_top_k(top_k)

        with self._generation_lock:
            if use_adapter is not None:
                self._apply_adapter_state(use_adapter)

            if audio_type == "snac":
                return self._generate_snac(
                    model,
                    tokenizer,
                    text,
                    temperature,
                    top_p,
                    max_new_tokens,
                    repetition_penalty,
                )
            elif audio_type == "csm":
                processor = model_info.get("processor", tokenizer)
                return self._generate_csm(model, processor, text, max_new_tokens)
            elif audio_type == "bicodec":
                return self._generate_bicodec(
                    model, tokenizer, text, temperature, top_k, max_new_tokens
                )
            elif audio_type == "dac":
                return self._generate_dac(
                    model,
                    tokenizer,
                    text,
                    temperature,
                    top_k,
                    top_p,
                    min_p,
                    max_new_tokens,
                    repetition_penalty,
                )
            else:
                raise RuntimeError(f"Unknown audio_type: {audio_type}")

    def _generate_snac(
        self, model, tokenizer, text, temperature, top_p, max_new_tokens, repetition_penalty
    ):
        """Generate audio using SNAC codec (Orpheus)."""
        device = model.device
        start_token = torch.tensor([[128259]], device = device)  # START_OF_HUMAN
        end_tokens = torch.tensor([[128009, 128260]], device = device)  # EOT, END_OF_HUMAN
        text_ids = tokenizer(text, return_tensors = "pt").input_ids.to(device)
        input_ids = torch.cat([start_token, text_ids, end_tokens], dim = 1)
        attention_mask = torch.ones_like(input_ids)

        generated = model.generate(
            input_ids = input_ids,
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_p = top_p,
            repetition_penalty = repetition_penalty,
            eos_token_id = 128258,  # END_OF_SPEECH
            use_cache = True,
        )
        return self._audio_codec_manager.decode_snac(generated, str(device))

    def _generate_csm(self, model, processor, text, max_new_tokens):
        """Generate audio using CSM (Sesame)."""
        speaker_id = 0
        inputs = processor(
            f"[{speaker_id}]{text}", add_special_tokens = True, return_tensors = "pt"
        ).to(model.device)
        audio_values = model.generate(**inputs, max_new_tokens = max_new_tokens, output_audio = True)
        return self._audio_codec_manager.decode_csm(audio_values)

    def _generate_bicodec(self, model, tokenizer, text, temperature, top_k, max_new_tokens):
        """Generate audio using BiCodec (Spark-TTS)."""
        prompt = "<|task_tts|><|start_content|>" + text + "<|end_content|><|start_global_token|>"
        inputs = tokenizer([prompt], return_tensors = "pt").to(model.device)
        generated = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample = True,
            temperature = temperature,
            top_k = top_k,
            eos_token_id = tokenizer.eos_token_id,
            pad_token_id = tokenizer.pad_token_id,
        )
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        decoded_text = tokenizer.batch_decode(new_tokens, skip_special_tokens = False)[0]
        return self._audio_codec_manager.decode_bicodec(decoded_text, str(model.device))

    def _generate_dac(
        self,
        model,
        tokenizer,
        text,
        temperature,
        top_k,
        top_p,
        min_p,
        max_new_tokens,
        repetition_penalty,
    ):
        """Generate audio using DAC (OuteTTS). Follows Oute_TTS_(1B).ipynb exactly."""
        # Monkey-patch RepetitionPenaltyLogitsProcessor with a 64-token window
        # (same as the OuteTTS notebook) to avoid degenerate repetition.
        self._patch_repetition_penalty_processor()

        prompt = (
            "<|im_start|>\n<|text_start|>"
            + text
            + "<|text_end|>\n<|audio_start|><|global_features_start|>\n"
        )
        with torch.inference_mode():
            with torch.amp.autocast("cuda", dtype = model.dtype):
                inputs = tokenizer([prompt], return_tensors = "pt").to(model.device)
                generated = model.generate(
                    **inputs,
                    temperature = temperature,
                    top_k = top_k,
                    top_p = top_p,
                    min_p = min_p,
                    repetition_penalty = repetition_penalty,
                    max_new_tokens = max_new_tokens,
                )
        decoded_text = tokenizer.batch_decode(generated, skip_special_tokens = False)[0]
        return self._audio_codec_manager.decode_dac(decoded_text, str(model.device))

    _repetition_penalty_patched = False

    @classmethod
    def _patch_repetition_penalty_processor(cls):
        """Monkey-patch transformers' RepetitionPenaltyLogitsProcessor with a
        64-token sliding-window variant (from the OuteTTS notebook).
        Applied once per process.
        """
        if cls._repetition_penalty_patched:
            return
        cls._repetition_penalty_patched = True

        from transformers import LogitsProcessor
        import transformers.generation.utils as generation_utils

        class RepetitionPenaltyLogitsProcessorPatch(LogitsProcessor):
            def __init__(self, penalty: float):
                self.penalty_last_n = 64
                if not isinstance(penalty, float) or penalty <= 0:
                    raise ValueError(f"`penalty` has to be a positive float, but is {penalty}")
                self.penalty = penalty

            @torch.no_grad()
            def __call__(
                self, input_ids: torch.LongTensor, scores: torch.FloatTensor
            ) -> torch.FloatTensor:
                if self.penalty_last_n == 0 or self.penalty == 1.0:
                    return scores
                batch_size, seq_len = input_ids.shape
                vocab_size = scores.shape[-1]
                for b in range(batch_size):
                    start_index = max(0, seq_len - self.penalty_last_n)
                    window_indices = input_ids[b, start_index:]
                    if window_indices.numel() == 0:
                        continue
                    for token_id in set(window_indices.tolist()):
                        if token_id >= vocab_size:
                            continue
                        logit = scores[b, token_id]
                        scores[b, token_id] = (
                            logit * self.penalty if logit <= 0 else logit / self.penalty
                        )
                return scores

        generation_utils.RepetitionPenaltyLogitsProcessor = RepetitionPenaltyLogitsProcessorPatch
        logger.info("Patched RepetitionPenaltyLogitsProcessor with 64-token window for OuteTTS")

    def _apply_chat_template_for_generation(
        self,
        tokenizer,
        messages: list,
        *,
        tools: Optional[list] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        preserve_thinking: Optional[bool] = None,
    ) -> str:
        """Render the chat prompt, peeling kwargs the template doesn't
        understand. Delegates to the dependency-light helper module so the
        fallback chain is unit-testable without pulling unsloth / torch into
        the test sandbox.
        """
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
        )
        return apply_chat_template_for_generation(
            tokenizer,
            messages,
            tools = tools,
            enable_thinking = enable_thinking,
            reasoning_effort = reasoning_effort,
            preserve_thinking = preserve_thinking,
        )

    def format_chat_prompt(
        self,
        messages: list,
        system_prompt: str = None,
    ) -> str:
        from core.inference.chat_template_helpers import (
            neutralize_control_markup_in_messages,
            neutralize_non_assistant_control_markup,
        )

        messages = neutralize_control_markup_in_messages(messages)
        if system_prompt:
            system_prompt = neutralize_non_assistant_control_markup(system_prompt)

        if not self.active_model_name or self.active_model_name not in self.models:
            logger.error("No active model available")
            return ""

        if self.models[self.active_model_name].get("tokenizer") is None:
            logger.error("Tokenizer not loaded for active model")
            return ""

        chat_template_info = self.models[self.active_model_name].get("chat_template_info", {})
        tokenizer = self.models[self.active_model_name]["tokenizer"]
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)

        chat_messages = []

        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})

        last_role = "system" if system_prompt else None

        for msg in messages:
            role = msg.get("role", "")
            content = content_to_text(msg.get("content", ""))

            if role in ["system", "user", "assistant"] and content.strip():
                if role == last_role:
                    logger.debug(f"Skipping consecutive {role} message to maintain alternation")
                    continue

                if role == "user":
                    import re
                    clean_content = re.sub(r"<[^>]+>", "", content).strip()
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
                chat_messages, tokenize = False, add_generation_prompt = True
            )
            logger.info(f"Successfully applied tokenizer's native chat template")
            return formatted_prompt
        except Exception as e:
            error_msg = str(e).lower()
            if "chat_template is not set" in error_msg or "no template argument" in error_msg:
                logger.info(
                    f"Base model detected - no built-in chat template available, using fallback formatting"
                )
            else:
                logger.warning(f"Failed to apply tokenizer chat template: {e}")
            logger.debug(
                f"""Failed with messages: {[f"{m['role']}: {m['content'][:30]}..." for m in chat_messages]}"""
            )

        if chat_template_info.get("has_template", False):
            logger.info("Falling back to manual template formatting based on detected patterns")
            template_type = chat_template_info.get("format_type", "generic")
            manual_prompt = self._format_chat_manual(
                chat_messages,
                template_type,
                chat_template_info.get("special_tokens", {}),
            )
            logger.info(f"Manual template result: {manual_prompt[:200]}...")
            return manual_prompt
        else:
            logger.info("Using generic chat formatting for base model")
            return self._format_generic_template(chat_messages, {})

    def _format_chat_manual(self, messages: list, template_type: str, special_tokens: dict) -> str:
        """Manual chat-formatting fallback when the tokenizer template fails.

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
            content = content_to_text(msg["content"])
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
                system_msg = content_to_text(msg["content"])
            else:
                conversation.append(msg)

        i = 0
        while i < len(conversation):
            if conversation[i]["role"] == "user":
                user_content = content_to_text(conversation[i]["content"])

                if system_msg and i == 0:
                    user_content = f"{system_msg}\n\n{user_content}"

                formatted += f"[INST] {user_content} [/INST]"

                if i + 1 < len(conversation) and conversation[i + 1]["role"] == "assistant":
                    formatted += f" {content_to_text(conversation[i + 1]['content'])}</s>"
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
            content = content_to_text(msg["content"])
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

        formatted += "<|im_start|>assistant\n"
        return formatted

    def _format_alpaca_template(self, messages: list, special_tokens: dict) -> str:
        """Format messages using Alpaca template"""
        formatted = ""
        system_msg = None

        for msg in messages:
            content = content_to_text(msg["content"])
            if msg["role"] == "system":
                system_msg = content
            elif msg["role"] == "user":
                if system_msg:
                    formatted += f"### Instruction:\n{system_msg}\n\n### Input:\n{content}\n\n### Response:\n"
                    system_msg = None
                else:
                    formatted += f"### Human:\n{content}\n\n### Assistant:\n"
            elif msg["role"] == "assistant":
                formatted += f"{content}\n\n"

        return formatted

    def _format_generic_template(self, messages: list, special_tokens: dict) -> str:
        """Generic fallback formatting"""
        formatted = ""

        for msg in messages:
            role = msg["role"].title()
            content = content_to_text(msg["content"])
            formatted += f"{role}: {content}\n"

        formatted += "Assistant: "
        return formatted

    def check_vision_model_compatibility(self) -> bool:
        """Whether the current model supports vision."""
        current_model = self.get_current_model()
        if current_model and current_model in self.models:
            return self.models[current_model].get("is_vision", False)
        return False

    def _reset_model_generation_state(self, model_name: str):
        """Reset generation state for a specific model to prevent contamination."""
        if model_name not in self.models:
            return

        model = self.models[model_name].get("model")
        if not model:
            return

        try:
            # Common pattern for Unsloth/Hugging Face models
            if hasattr(model, "past_key_values"):
                model.past_key_values = None
            if hasattr(model, "generation_config"):
                if hasattr(model.generation_config, "past_key_values"):
                    model.generation_config.past_key_values = None

            logger.debug(f"Reset generation state for model: {model_name}")
        except Exception as e:
            logger.warning(f"Could not fully reset model state for {model_name}: {e}")

    def reset_generation_state(self):
        """Reset any cached generation state to prevent hanging after errors"""
        try:
            # Clear cached state for ALL loaded models
            for model_name in self.models.keys():
                self._reset_model_generation_state(model_name)

            clear_gpu_cache()
            logger.debug("Cleared GPU cache")

            import gc

            gc.collect()
            logger.info("Performed comprehensive generation state reset")

        except Exception as e:
            logger.warning(f"Could not fully reset generation state: {e}")

    def resize_image(
        self,
        img,
        max_size: int = 800,
    ):
        """Resize image while maintaining aspect ratio if either dimension exceeds max_size"""
        if img is None:
            return None
        if img.size[0] > max_size or img.size[1] > max_size:
            from PIL import Image

            ratio = min(max_size / img.size[0], max_size / img.size[1])
            new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
            return img.resize(new_size, Image.Resampling.LANCZOS)
        return img

    def _generation_stop_token_ids(self, model, generation_kwargs: dict):
        """Return the stop-token ids active for a ``generate`` call."""
        if "eos_token_id" in generation_kwargs:
            return generation_kwargs.get("eos_token_id")
        generation_config = getattr(model, "generation_config", None)
        eos_token_id = getattr(generation_config, "eos_token_id", None)
        if eos_token_id is not None:
            return eos_token_id
        config = getattr(model, "config", None)
        return getattr(config, "eos_token_id", None)

    def _cancel_stopping_criteria(self, cancel_event):
        """Build a Transformers stopping criteria list for user cancellation."""
        if cancel_event is None:
            return None
        from transformers.generation.stopping_criteria import (
            StoppingCriteria,
            StoppingCriteriaList,
        )

        class _CancelCriteria(StoppingCriteria):
            def __init__(self, ev):
                self.ev = ev

            def __call__(self, input_ids, scores, **kwargs):
                return self.ev.is_set()

        return StoppingCriteriaList([_CancelCriteria(cancel_event)])

    def _clean_generated_text(
        self,
        text: str,
        *,
        stop_token_ids = None,
    ) -> str:
        """Strip leaked response-boundary tokens after streaming."""
        if self._is_gpt_oss_model():
            # HarmonyTextStreamer emits clean <think>...</think>. Strip any
            # harmony protocol tokens and other gpt-oss tokens (e.g.
            # <|return|>) that leak past the streamer.
            import re
            text = re.sub(r"<\|[a-z_]+\|>", "", text)
            return text.strip()

        tokenizer = self.models.get(self.active_model_name, {}).get("tokenizer")
        tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
        if tokenizer:
            if stop_token_ids is None:
                stop_token_ids = self.models.get(self.active_model_name, {}).get(
                    "chat_turn_end_eos_ids"
                )
            if isinstance(stop_token_ids, int):
                stop_token_ids = (stop_token_ids,)
            for token_id in stop_token_ids or ():
                try:
                    token = tokenizer.convert_ids_to_tokens(int(token_id))
                except Exception:
                    token = None
                if isinstance(token, str) and token and text.endswith(token):
                    text = text[: -len(token)]
                elif (
                    isinstance(token, str)
                    and token
                    and text.endswith("</think>")
                    and text[: -len("</think>")].endswith(token)
                ):
                    text = text[: -len("</think>") - len(token)] + "</think>"
        return text.strip()

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
            from utils.datasets import MODEL_TO_TEMPLATE_MAPPER

            # Exact match first
            model_name_lower = model_name.lower()
            if model_name_lower in MODEL_TO_TEMPLATE_MAPPER:
                chat_template_info["template_name"] = MODEL_TO_TEMPLATE_MAPPER[model_name_lower]
                logger.info(
                    f"Detected template '{chat_template_info['template_name']}' for {model_name} from mapper"
                )
            else:
                # Partial match (for variants like model_name-bnb-4bit)
                for key in MODEL_TO_TEMPLATE_MAPPER:
                    if key in model_name_lower or model_name_lower in key:
                        chat_template_info["template_name"] = MODEL_TO_TEMPLATE_MAPPER[key]
                        logger.info(
                            f"Detected template '{chat_template_info['template_name']}' for {model_name} (partial match)"
                        )
                        break
        except Exception as e:
            logger.warning(f"Could not detect template from mapper for {model_name}: {e}")

        try:
            if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
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

                logger.info(
                    f"Loaded chat template for {model_name} (detected as {chat_template_info['format_type']} format)"
                )
                logger.debug(f"Template preview: {tokenizer.chat_template[:200]}...")

                special_tokens = {}
                if hasattr(tokenizer, "bos_token") and tokenizer.bos_token:
                    special_tokens["bos_token"] = tokenizer.bos_token
                if hasattr(tokenizer, "eos_token") and tokenizer.eos_token:
                    special_tokens["eos_token"] = tokenizer.eos_token
                if hasattr(tokenizer, "pad_token") and tokenizer.pad_token:
                    special_tokens["pad_token"] = tokenizer.pad_token

                chat_template_info["special_tokens"] = special_tokens

            else:
                logger.info(f"No chat template found for {model_name}, will use generic formatting")

        except Exception as e:
            logger.error(f"Error loading chat template info for {model_name}: {e}")

        self.models[model_name]["chat_template_info"] = chat_template_info

        if chat_template_info["has_template"]:
            logger.info(
                f"Chat template loaded for {model_name}: {chat_template_info['format_type']} format"
            )
        else:
            logger.info(f"No built-in chat template for {model_name}, will use generic formatting")

    def get_current_model(self) -> Optional[str]:
        """Currently active model name."""
        return self.active_model_name

    def is_model_loading(self) -> bool:
        """Whether any model is currently loading."""
        return len(self.loading_models) > 0

    def get_loading_model(self) -> Optional[str]:
        """Name of the currently loading model."""
        return next(iter(self.loading_models)) if self.loading_models else None

    def load_model_simple(
        self,
        model_path: str,
        hf_token: Optional[str] = None,
        max_seq_length: int = 2048,
        load_in_4bit: bool = True,
    ) -> bool:
        """Simple model-loading wrapper for the chat interface. Takes a string
        path and builds the ModelConfig internally.

        Args:
            model_path: Model name or path (e.g., "unsloth/llama-3-8b")
            hf_token: HuggingFace token for gated models
            max_seq_length: Maximum sequence length
            load_in_4bit: Whether to use 4-bit quantization

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            config = ModelConfig.from_ui_selection(
                model_path,
                lora_path = None,  # No LoRA for chat
                is_lora = False,
            )

            return self.load_model(
                config = config,
                max_seq_length = max_seq_length,
                dtype = None,  # Auto-detect
                load_in_4bit = load_in_4bit,
                hf_token = hf_token,
            )

        except Exception as e:
            logger.error(f"Error in load_model_simple: {e}")
            return False


# Global inference backend instance
inference_backend = InferenceBackend()


def get_inference_backend() -> InferenceBackend:
    return inference_backend
