# SPDX-License-Identifier: AGPL-3.0-only
"""MLX inference backend for Apple Silicon.

Drop-in replacement for InferenceBackend — same interface, uses mlx-lm/mlx-vlm
instead of torch/transformers for model loading and generation.
"""

import json
import os
import threading
from contextlib import contextmanager
from typing import Optional, Generator
from core.inference.message_content import content_to_text
from core.inference.runtime_context import runtime_context_length
from core.inference.chat_template_helpers import (
    ReasoningChannelNormalizer,
    normalize_reasoning_snapshots,
)
from loggers import get_logger

logger = get_logger(__name__)


def _mlx_adapter_modules(model):
    """Return bypassable adapter entries and unsupported wrapper paths."""
    adapters = []
    unsupported = []
    for path, module in model.named_modules():
        if not path or not (hasattr(module, "lora_a") and hasattr(module, "lora_b")):
            continue
        base = getattr(module, "linear", None)
        if base is None:
            base = getattr(module, "embedding", None)
        if base is None:
            unsupported.append(path)
        else:
            adapters.append((path, module, base))
    return adapters, unsupported


@contextmanager
def _temporary_mlx_adapter_state(model, use_adapter):
    """Select base or adapter modules for one request, then restore the tree."""
    if use_adapter is None:
        yield
        return
    if isinstance(use_adapter, str):
        raise NotImplementedError(
            "Unsloth MLX: named adapter selection is not supported; use True for "
            "the loaded adapter or False for the base model."
        )
    if use_adapter is not True and use_adapter is not False:
        raise TypeError("Unsloth MLX: use_adapter must be None, True, False, or a string.")

    adapters, unsupported = _mlx_adapter_modules(model)
    if use_adapter is True:
        if not adapters and not unsupported:
            logger.warning("MLX adapter requested, but the active model has no adapter layers")
        yield
        return
    if unsupported:
        raise RuntimeError(
            "Unsloth MLX: cannot disable adapter layers without their base modules: "
            + ", ".join(unsupported[:5])
        )
    if not adapters:
        yield
        return

    from mlx.utils import tree_unflatten

    base_modules = tree_unflatten([(path, base) for path, _, base in adapters])
    adapter_modules = tree_unflatten([(path, wrapper) for path, wrapper, _ in adapters])
    try:
        model.update_modules(base_modules)
        yield
    finally:
        model.update_modules(adapter_modules)


def _mlx_vlm_model_config(model):
    """Return the loaded MLX model config and its type, preferring whichever of
    config / _config actually carries a model_type."""

    def _model_type(cfg):
        return cfg.get("model_type") if isinstance(cfg, dict) else getattr(cfg, "model_type", None)

    configs = [
        cfg
        for cfg in (getattr(model, "config", None), getattr(model, "_config", None))
        if cfg is not None
    ]
    for cfg in configs:
        model_type = _model_type(cfg)
        if model_type is not None:
            return cfg, model_type
    return (configs[0] if configs else None), None


def _render_registered_vlm_prompt(processor, model, messages, num_images):
    """Render through mlx-vlm when it declares a formatter for this model."""
    from mlx_vlm import prompt_utils

    config, model_type = _mlx_vlm_model_config(model)
    if config is None:
        return None
    if model_type not in getattr(prompt_utils, "MODEL_CONFIG", {}):
        return None

    rendered = prompt_utils.apply_chat_template(
        processor,
        config,
        messages,
        add_generation_prompt = True,
        num_images = num_images,
    )
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    raise RuntimeError("mlx-vlm's registered renderer returned an empty prompt.")


def _count_vlm_images(content):
    if isinstance(content, list):
        return sum(_count_vlm_images(item) for item in content)
    if not isinstance(content, dict):
        return 0
    if str(content.get("type", "")).lower() in ("image", "image_url", "input_image"):
        return 1
    return _count_vlm_images(content.get("content"))


def _vlm_media_reprs(content):
    if isinstance(content, list):
        values = (
            {str(content), json.dumps(content, ensure_ascii = False)}
            if _count_vlm_images(content)
            else set()
        )
        for item in content:
            values.update(_vlm_media_reprs(item))
        return values
    if not isinstance(content, dict):
        return set()
    if str(content.get("type", "")).lower() in ("image", "image_url", "input_image"):
        return {str(content), json.dumps(content, ensure_ascii = False)}
    return _vlm_media_reprs(content.get("content"))


def _prompt_serializes_vlm_media(prompt, messages):
    """Detect templates that embed the exact structured media object repr."""
    media_reprs = set()
    for message in messages:
        if isinstance(message, dict):
            media_reprs.update(_vlm_media_reprs(message.get("content")))
    text_content = [
        content_to_text(message.get("content")) for message in messages if isinstance(message, dict)
    ]
    return any(
        prompt.count(media_repr) > sum(content.count(media_repr) for content in text_content)
        for media_repr in media_reprs
    )


def _vlm_prompt_issue(prompt, messages):
    if not isinstance(prompt, str) or not prompt.strip():
        return "an empty prompt"
    if _prompt_serializes_vlm_media(prompt, messages):
        return "serialized structured image content"
    return None


def _vlm_messages_have_tool_history(messages):
    return any(
        isinstance(message, dict)
        and (
            message.get("role") == "tool"
            or message.get("tool_calls")
            or message.get("tool_call_id")
        )
        for message in messages
    )


def _build_generation_stats(
    prompt_n,
    prompt_tps,
    gen_n,
    gen_tps,
    cached_n = 0,
):
    """Map mlx stream stats onto the usage/timings shape llama-server emits."""
    prompt_n = int(prompt_n or 0)
    gen_n = int(gen_n or 0)
    cached_n = int(cached_n or 0)
    prompt_tps = float(prompt_tps or 0.0)
    gen_tps = float(gen_tps or 0.0)
    prompt_ms = (prompt_n / prompt_tps * 1000.0) if prompt_tps > 0 else 0.0
    predicted_ms = (gen_n / gen_tps * 1000.0) if gen_tps > 0 else 0.0
    total_prompt_n = prompt_n + cached_n
    return {
        "usage": {
            "prompt_tokens": total_prompt_n,
            "completion_tokens": gen_n,
            "total_tokens": total_prompt_n + gen_n,
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
            "cache_n": cached_n,
        },
    }


PROMPT_CACHE_ENTRIES = 6
PROMPT_CACHE_MEMORY_FRACTION = 0.15
PROMPT_CACHE_FALLBACK_BYTES = 2 * 1024**3


def _mlx_prompt_cache_api():
    try:
        from mlx_lm.models.cache import (
            LRUPromptCache,
            can_trim_prompt_cache,
            make_prompt_cache,
            trim_prompt_cache,
        )
    except ImportError:
        return None
    return LRUPromptCache, make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache


def _prompt_cache_max_bytes(recommended_gb = None):
    override = os.environ.get("UNSLOTH_MLX_PROMPT_CACHE_BYTES")
    if override:
        try:
            return max(int(override), 0)
        except ValueError:
            logger.warning("Ignoring non-integer UNSLOTH_MLX_PROMPT_CACHE_BYTES=%r", override)
    if recommended_gb:
        return int(recommended_gb * 1e9 * PROMPT_CACHE_MEMORY_FRACTION)
    return PROMPT_CACHE_FALLBACK_BYTES


class _MLXPromptCacheHistory:
    def __init__(self, max_entries, max_bytes):
        api = _mlx_prompt_cache_api()
        if api is None:
            raise RuntimeError("mlx-lm is too old for LRUPromptCache")
        lru_cls, make, can_trim, trim = api
        self._make_prompt_cache = make
        self._can_trim = can_trim
        self._trim = trim
        self._max_bytes = max_bytes
        self._lru = lru_cls(max_size = max_entries, max_bytes = max_bytes)

    def fetch(self, model, key, tokens):
        cache, rest = self._lru.fetch_nearest_cache(key, list(tokens))
        if cache is not None:
            if rest:
                return cache, list(rest)
            if self._can_trim(cache) and self._trim(cache, 1) == 1:
                return cache, list(tokens[-1:])
        if len(tokens) > 1:
            head = list(tokens[:-1])
            cache, rest = self._lru.fetch_nearest_cache(key, head)
            if cache is not None:
                covered = len(head) - len(rest)
                return cache, list(tokens[covered:])
        return self._make_prompt_cache(model), list(tokens)

    def insert(self, key, tokens, cache):
        # An over-budget entry evicts itself and every other conversation.
        nbytes = sum(getattr(entry, "nbytes", 0) for entry in cache)
        if nbytes > self._max_bytes:
            logger.debug(
                "MLX prompt cache: skipping %.2f GB entry over the %.2f GB budget",
                nbytes / 1e9,
                self._max_bytes / 1e9,
            )
            return
        self._lru.insert_cache(key, list(tokens), cache)


def _mlx_distributed_rank_size(group = None):
    """Return ``(rank, world_size)`` for an optional MLX distributed group."""
    if group is None:
        return 0, 1
    rank = int(group.rank())
    world_size = int(group.size())
    if world_size < 1:
        raise ValueError(f"Invalid MLX distributed world_size={world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"Invalid MLX distributed rank={rank} for world_size={world_size}.")
    return rank, world_size


def _mlx_distributed_backend_from_env():
    if os.environ.get("MLX_JACCL_COORDINATOR") and os.environ.get("MLX_IBV_DEVICES"):
        return "jaccl"
    return None


def _init_mlx_distributed():
    """Initialize MLX distributed state, falling back to singleton metadata."""
    import mlx.core as mx

    group = None
    rank = 0
    world_size = 1
    distributed = getattr(mx, "distributed", None)
    init = getattr(distributed, "init", None) if distributed is not None else None
    if callable(init):
        backend = _mlx_distributed_backend_from_env()
        if backend is None:
            group = init()
        else:
            try:
                group = init(backend = backend)
            except TypeError:
                group = init()
        if group is not None:
            rank, world_size = _mlx_distributed_rank_size(group)
    return group, rank, world_size


def _make_mlx_presence_penalty_processor(penalty: float):
    """Presence penalty as an mlx_lm/mlx_vlm logits processor, matching the safetensors path.

    generate_step calls processors as ``fn(tokens, logits)`` with ``tokens`` the
    full running sequence; the first call is prompt-only, so latch that length
    and penalize only after it.
    """
    state = {"prompt_len": None}

    def _processor(tokens, logits):
        if state["prompt_len"] is None:
            # First call is prompt-only; latch its length.
            state["prompt_len"] = int(tokens.shape[0])
            return logits
        generated = tokens[state["prompt_len"] :]
        if generated.size == 0:
            return logits
        import mlx.core as mx

        vocab = logits.shape[-1]
        # Bound ids to [0, vocab) before indexing logits: MLX does no bounds
        # checking and out-of-bounds indexing is undefined behavior (crash /
        # corruption), unlike torch's harmless negative wrap. MLX also lacks
        # boolean-mask filtering, so out-of-range/negative ids route to a
        # scratch slot at index vocab (dropped before the subtract) that never
        # collides with a real token: real ids (including 0) are penalized
        # once, strays ignored.
        valid = (generated >= 0) & (generated < vocab)
        safe = mx.where(valid, generated, vocab).astype(mx.int32)
        # Scatter penalty into a (vocab + 1)-wide mask: duplicate ids are
        # idempotent (presence applies once per token); scratch column dropped.
        mask = mx.zeros((vocab + 1,), dtype = logits.dtype)
        mask[safe] = penalty
        logits = logits - mask[:vocab]
        return logits

    return _processor


class MLXInferenceBackend:
    def __init__(self):
        self.models = {}
        self.active_model_name = None
        self.loading_models = set()
        self.loaded_local_models = []
        self.device = "mlx"
        self._generation_lock = threading.Lock()
        # usage/timings of the latest generation, shipped on gen_done.
        self.last_generation_stats = None

        self._model = None
        self._tokenizer = None
        self._processor = None
        self._is_vlm = False
        self._config = {}
        self._distributed_group = None
        self._distributed_rank = 0
        self._distributed_world_size = 1

        # Recorded for unload to release pinned memory back to the OS.
        self._memory_limits_applied = {}

        self._prompt_cache_history = None
        self._prompt_cache_unavailable = False

    def _prompt_cache(self):
        if self._prompt_cache_history is not None or self._prompt_cache_unavailable:
            return self._prompt_cache_history
        max_bytes = _prompt_cache_max_bytes(self._memory_limits_applied.get("recommended_gb"))
        if max_bytes <= 0:
            self._prompt_cache_unavailable = True
            logger.info("MLX prompt cache disabled by budget")
            return None
        try:
            self._prompt_cache_history = _MLXPromptCacheHistory(
                PROMPT_CACHE_ENTRIES,
                max_bytes,
            )
        except Exception as exc:
            self._prompt_cache_unavailable = True
            logger.info("MLX prompt cache unavailable (%s); prefilling every request", exc)
            return None
        logger.info(
            "MLX prompt cache: %d entries, %.2f GB budget",
            PROMPT_CACHE_ENTRIES,
            max_bytes / 1e9,
        )
        return self._prompt_cache_history

    def _clear_prompt_cache(self):
        self._prompt_cache_history = None
        self._prompt_cache_unavailable = False

    def _prepare_prompt_cache(self, prompt, adapter_state):
        history = self._prompt_cache()
        if history is None:
            return prompt, None, None, None, 0
        try:
            tokenizer = self._tokenizer
            bos = getattr(tokenizer, "bos_token", None)
            add_special_tokens = bos is None or not prompt.startswith(bos)
            tokens = list(tokenizer.encode(prompt, add_special_tokens = add_special_tokens))
            if not tokens:
                return prompt, None, None, None, 0
            key = f"{self.active_model_name}|{adapter_state!r}"
            cache, rest = history.fetch(self._model, key, tokens)
        except Exception as exc:
            logger.debug("MLX prompt cache lookup failed: %s", exc)
            return prompt, None, None, None, 0
        return rest, cache, key, tokens, len(tokens) - len(rest)

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
        parallel_mode = None,
        distributed_group = None,
    ) -> bool:
        import mlx.core as mx

        # Keep the token so the native-template fallback can fetch a gated
        # model's repo template during generation.
        self._hf_token = hf_token
        model_name = config.identifier if hasattr(config, "identifier") else str(config)
        is_vision = getattr(config, "is_vision", False)
        distributed_rank, distributed_size = _mlx_distributed_rank_size(distributed_group)
        is_distributed = distributed_group is not None and distributed_size > 1
        self._distributed_group = distributed_group
        self._distributed_rank = distributed_rank
        self._distributed_world_size = distributed_size

        # GGUF guard: GGUF is served by llama-server in the parent process,
        # not mlx-lm. Reaching here with is_gguf=True means the route's
        # detection flaked but the subprocess re-detected GGUF; raise loudly
        # instead of a cryptic mlx_lm error.
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
            "Loading %s via %s (is_lora=%s, distributed=%s, rank=%s/%s, mode=%s)",
            model_name,
            "mlx-vlm" if is_vision else "mlx-lm",
            is_lora,
            is_distributed,
            distributed_rank,
            distributed_size,
            parallel_mode,
        )
        if is_distributed and parallel_mode not in ("pipeline", "tensor"):
            raise ValueError(
                "Unsloth: distributed MLX inference requires parallel_mode='pipeline' "
                "or parallel_mode='tensor'."
            )
        if is_distributed and is_lora:
            raise ValueError(
                "Unsloth: distributed MLX inference for LoRA adapter repos "
                "is not supported yet. Merge/export the adapter into an MLX model "
                "before distributed inference."
            )

        try:
            from unsloth_zoo.mlx.loader import FastMLXModel
        except ImportError as e:
            raise ImportError(
                "Unsloth: MLX inference requires unsloth-zoo with the MLX modules "
                "(unsloth_zoo.mlx.loader). Reinstall via install.sh on Apple Silicon."
            ) from e

        load_kwargs = {
            "max_seq_length": max_seq_length,
            "dtype": dtype,
            "load_in_4bit": load_in_4bit,
            "token": hf_token,
            "trust_remote_code": trust_remote_code,
            "text_only": False if is_vision else True,
        }
        if is_distributed:
            if parallel_mode == "pipeline":
                load_kwargs["pipeline_group"] = distributed_group
            else:
                load_kwargs["tensor_group"] = distributed_group

        model, tokenizer_or_processor = FastMLXModel.from_pretrained(
            model_name,
            **load_kwargs,
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
            # Per-model token for the native-template fallback (matches transformers).
            "hf_token": hf_token,
            # Per-model trust_remote_code reused by the native-template reload (matches transformers).
            "trust_remote_code": trust_remote_code,
            "model": self._model,
            "tokenizer": self._tokenizer,
            "processor": self._processor,
            "is_vision": is_vision,
            "is_lora": getattr(config, "is_lora", False),
            # For a LoRA adapter the native chat template lives on the base model.
            "base_model": getattr(config, "base_model", None)
            if getattr(config, "is_lora", False)
            else None,
            "is_audio": False,
            "audio_type": None,
            "has_audio_input": False,
            "context_length": runtime_context_length(self._model, max_seq_length),
        }
        # Capture chat_template_info for the worker IPC reply and route capability classification.
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
        self._distributed_group = None
        self._distributed_rank = 0
        self._distributed_world_size = 1
        if self.active_model_name == model_name:
            self.active_model_name = None
        self._clear_prompt_cache()
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
        # Reasoning / tool kwargs, rendered via apply_chat_template_for_generation (transformers parity).
        tools = None,
        enable_thinking = None,
        reasoning_effort = None,
        preserve_thinking = None,
        presence_penalty = 0.0,
        _adapter_state = None,
    ) -> Generator[str, None, None]:
        if self._model is None:
            raise RuntimeError("No model loaded")

        # Reset so a failed run cannot surface stale stats.
        self.last_generation_stats = None

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
                        has_image = _count_vlm_images(content) > 0
                        if not has_image:
                            content.insert(0, {"type": "image"})
                    break

        if self._is_vlm:
            stream = self._generate_vlm(
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
                presence_penalty = presence_penalty,
                _adapter_state = _adapter_state,
            )
        else:
            stream = self._generate_text(
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
                presence_penalty = presence_penalty,
                _adapter_state = _adapter_state,
            )
        yield from stream

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
        presence_penalty = 0.0,
        _adapter_state = None,
    ):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler, make_logits_processors

        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            detect_think_prefill,
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

        # Parity with the transformers backend: if the template dropped the
        # requested tools, fall back to the native template so MLX text models
        # keep advertising them. self._tokenizer is this entry's tokenizer, so
        # probe and native render share a renderer. (VLM renders via the
        # processor for image tokens and is not wired here.)
        model_info = self.models.get(self.active_model_name, {})
        render_result = render_with_native_template_fallback(
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
            return_metadata = True,
        )
        prompt = render_result.prompt
        reasoning_channel_markers = render_result.reasoning_channel_markers

        # An open <think> prefilled by the template lives in the prompt, not
        # the generated tokens; re-emit it so the frontend renders the block.
        think_prefix = detect_think_prefill(
            prompt, getattr(self._tokenizer, "all_special_tokens", None)
        )
        sampler = make_sampler(
            temp = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
            min_tokens_to_keep = 1,
        )
        # Repetition and/or presence penalty processors (GGUF/safetensors parity).
        logits_processors = []
        if repetition_penalty is not None and float(repetition_penalty) not in (
            0.0,
            1.0,
        ):
            logits_processors.extend(
                make_logits_processors(
                    repetition_penalty = float(repetition_penalty),
                )
            )
        if presence_penalty:
            logits_processors.append(_make_mlx_presence_penalty_processor(float(presence_penalty)))
        if not logits_processors:
            logits_processors = None

        preserve_native_channels = reasoning_channel_markers is not None
        token_ids = []
        normalizer = (
            ReasoningChannelNormalizer(*reasoning_channel_markers)
            if reasoning_channel_markers is not None
            else None
        )
        # MLX consumers diff cumulative snapshots. Keep a prompt-prefilled
        # <think> prefix on every native-protocol snapshot just as the normal
        # decoding path does below.
        normalized_output = think_prefix
        with self._generation_lock, _temporary_mlx_adapter_state(self._model, _adapter_state):
            (
                gen_prompt,
                prompt_cache,
                cache_key,
                prompt_tokens,
                cached_n,
            ) = self._prepare_prompt_cache(prompt, _adapter_state)
            logger.info(
                "Generating: prompt_len=%d, cached=%d, max_tokens=%d, model=%s, tokenizer=%s",
                len(prompt),
                cached_n,
                max_new_tokens,
                type(self._model).__name__,
                type(self._tokenizer).__name__,
            )
            final_response = None
            try:
                # Enter request-scoped model state before yielding any response.
                if think_prefix:
                    yield think_prefix
                gen_kwargs = dict(
                    prompt = gen_prompt,
                    max_tokens = max_new_tokens,
                    sampler = sampler,
                )
                if prompt_cache is not None:
                    gen_kwargs["prompt_cache"] = prompt_cache
                if logits_processors is not None:
                    gen_kwargs["logits_processors"] = logits_processors
                for response in stream_generate(
                    self._model,
                    self._tokenizer,
                    **gen_kwargs,
                ):
                    final_response = response
                    token_ids.append(response.token)
                    if preserve_native_channels:
                        piece = getattr(response, "text", None) or ""
                        delta = normalizer.feed(piece)
                        if delta:
                            normalized_output += delta
                            yield normalized_output
                    else:
                        cumulative = self._tokenizer.decode(
                            token_ids,
                            skip_special_tokens = True,
                        )
                        yield think_prefix + cumulative

                    if cancel_event and cancel_event.is_set():
                        break
                if prompt_cache is not None and prompt_tokens is not None:
                    history = self._prompt_cache_history
                    if history is not None:
                        try:
                            history.insert(cache_key, prompt_tokens + token_ids, prompt_cache)
                        except Exception as exc:
                            logger.debug("MLX prompt cache insert failed: %s", exc)
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
                        cached_n,
                    )
        if normalizer is not None:
            cancelled = cancel_event is not None and cancel_event.is_set()
            tail = normalizer.drain() if cancelled else normalizer.finish()
            if tail:
                normalized_output += tail
                yield normalized_output

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
        presence_penalty = 0.0,
        _adapter_state = None,
    ):
        from mlx_vlm import stream_generate as vlm_stream

        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
        )

        # Pick the chat-template-aware caller: processors with their own
        # apply_chat_template + chat_template (e.g. Qwen2.5-VL), else the nested tokenizer.
        chat_target = self._processor
        if (
            getattr(self._processor, "apply_chat_template", None) is None
            or not hasattr(self._processor, "chat_template")
            or self._processor.chat_template is None
        ):
            chat_target = getattr(self._processor, "tokenizer", self._processor)

        # mlx_vlm's stream_generate handles pixel_values (None for text-only)
        images = [image] if image is not None else None
        attached_images = 0 if images is None else len(images)
        structured_images = sum(
            _count_vlm_images(message.get("content"))
            for message in messages
            if isinstance(message, dict)
        )
        if structured_images != attached_images:
            raise RuntimeError(
                f"VLM conversation contains {structured_images} structured image "
                f"item(s) for {attached_images} attached image(s)."
            )
        prompt = None
        has_tool_history = _vlm_messages_have_tool_history(messages)
        prompt_error = None
        try:
            prompt = apply_chat_template_for_generation(
                chat_target,
                messages,
                tools = tools,
                enable_thinking = enable_thinking,
                reasoning_effort = reasoning_effort,
                preserve_thinking = preserve_thinking,
            )
        except Exception as exc:
            if images is None or has_tool_history:
                raise
            prompt_error = exc
        prompt_issue = (
            _vlm_prompt_issue(prompt, messages) if prompt_error is None else "a rendering error"
        )
        if prompt_issue and has_tool_history:
            raise RuntimeError(
                f"VLM chat template returned {prompt_issue} and cannot be recovered "
                "without dropping tool-call history."
            ) from prompt_error

        if images is not None and prompt_issue:
            if tools or any(
                value is not None
                for value in (enable_thinking, reasoning_effort, preserve_thinking)
            ):
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue} and cannot be recovered "
                    "without dropping requested tools or reasoning controls."
                )
            try:
                recovered_prompt = _render_registered_vlm_prompt(
                    self._processor,
                    self._model,
                    messages,
                    len(images),
                )
            except Exception as recovery_error:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue}; model-aware "
                    f"recovery failed: {recovery_error}"
                ) from recovery_error
            if recovered_prompt is None:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue}, and no registered "
                    "MLX VLM renderer was available for this model."
                )
            recovered_issue = _vlm_prompt_issue(recovered_prompt, messages)
            if recovered_issue:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"Model-aware VLM rendering returned {recovered_issue} for "
                    f"{attached_images} attached image(s)."
                )
            prompt = recovered_prompt
        elif prompt_issue:
            raise RuntimeError(f"VLM chat template returned {prompt_issue}.") from prompt_error

        from core.inference.chat_template_helpers import detect_think_prefill

        # Re-emit an open <think> prefill from the prompt (see _generate_text).
        cumulative = detect_think_prefill(prompt, getattr(chat_target, "all_special_tokens", None))
        logger.info(
            "VLM generating: prompt_len=%d, has_image=%s",
            len(prompt),
            image is not None,
        )
        # stream_generate forwards **kwargs into generate_step (builds the
        # sampler + logits_processors internally). GOTCHA: generate_step expects
        # temperature= (long form); temp= is silently ignored, stuck at greedy 0.0.
        vlm_kwargs = dict(
            max_tokens = max_new_tokens,
            temperature = temperature,
            top_p = top_p,
            top_k = int(top_k or 0),
            min_p = float(min_p or 0.0),
        )
        _rep_active = repetition_penalty is not None and float(repetition_penalty) not in (
            0.0,
            1.0,
        )
        if presence_penalty:
            # Presence needs a custom processor: pass the full list (repetition +
            # presence) instead of the repetition_penalty shortcut so both apply.
            from mlx_lm.sample_utils import make_logits_processors

            _vlm_processors = []
            if _rep_active:
                _vlm_processors.extend(
                    make_logits_processors(repetition_penalty = float(repetition_penalty))
                )
            _vlm_processors.append(_make_mlx_presence_penalty_processor(float(presence_penalty)))
            vlm_kwargs["logits_processors"] = _vlm_processors
        elif _rep_active:
            vlm_kwargs["repetition_penalty"] = float(repetition_penalty)

        def _stream_vlm_snapshots():
            nonlocal cumulative
            # Hold the generation lock AND the request-scoped adapter state for the
            # whole stream so Base-vs-LoRA compare mode honors use_adapter and the
            # wrapper tree is restored on completion, cancellation, or close.
            with self._generation_lock, _temporary_mlx_adapter_state(self._model, _adapter_state):
                final_response = None
                try:
                    # Emit any prefilled <think> block before the first token so the
                    # UI renders it during prefill, matching _generate_text. Done
                    # inside the adapter context so an unsupported request raises
                    # before any output escapes.
                    if cumulative:
                        yield cumulative
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

        yield from normalize_reasoning_snapshots(
            _stream_vlm_snapshots(), chat_target, cancel_event, tools = tools
        )

    def generate_with_adapter_control(
        self,
        use_adapter = None,
        cancel_event = None,
        **gen_kwargs,
    ) -> Generator[str, None, None]:
        yield from self.generate_chat_response(
            cancel_event = cancel_event,
            _adapter_state = use_adapter,
            **gen_kwargs,
        )

    def reset_generation_state(self):
        import mlx.core as mx
        import gc

        gc.collect()
        mx.clear_cache()
