# SPDX-License-Identifier: AGPL-3.0-only
"""MLX inference backend for Apple Silicon.

Drop-in replacement for InferenceBackend — same interface, uses mlx-lm/mlx-vlm
instead of torch/transformers for model loading and generation.
"""

import copy
import hashlib
import importlib
import json
import os
import secrets
import threading
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Optional, Generator
from core.inference.message_content import content_to_text
from core.inference.runtime_context import runtime_context_length
from core.inference.chat_template_helpers import (
    ReasoningChannelNormalizer,
    normalize_reasoning_snapshots,
)
from utils.models.model_config import is_audio_input_type
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


def _vlm_runtime_reused_prefix(model, input_ids, cache_state):
    """True when this forward is the runtime's reuse of Studio's cut prefix.

    The runtime primes a position map over the whole prompt and then trims the
    ids to the uncached suffix, so on a reusing forward the primed map is exactly
    ``observed_prefix`` longer than the ids. A cold prefill passes the full prompt
    and fails that identity, which is what keeps a cold request unsuppressed.

    The map is cleared when the request begins, so a populated one can only have
    come from this request's priming; a length left over from an earlier request
    cannot satisfy the identity by coincidence.
    """
    prefix = int(getattr(cache_state, "observed_prefix", 0) or 0)
    if prefix <= 0 or input_ids is None:
        return False
    primed = getattr(getattr(model, "language_model", None), "_position_ids", None)
    if primed is None:
        return False
    try:
        return int(primed.shape[-1]) == int(input_ids.shape[-1]) + prefix
    except (AttributeError, IndexError, TypeError, ValueError):
        return False


@contextmanager
def _temporary_mlx_vlm_rope_suppression(model, cache_state):
    """Drop suffix-derived multimodal-RoPE state during a reusing request.

    On a cached-prefix continuation the runtime recomputes RoPE state from the
    uncached suffix — with no image grid, so it yields a zero delta — and merges
    it over the whole-prompt value it primed, positioning the suffix from the
    cache offset instead of the multimodal map the prefill used. Studio drops the
    two feature fields the runtime's own merge already treats as optional, so the
    primed whole-prompt value governs. Studio never computes RoPE state itself.

    Suppression must apply exactly when the runtime accepted the cached prefix,
    which is decided on the first forward. Reuse cannot be inferred from
    ``pixel_values`` alone: a text-only continuation is cache-eligible too and
    already passes ``None``. It is instead detected arithmetically — when the
    runtime reuses, it primes a whole-prompt position map and trims ``input_ids``
    to the uncached suffix, so the primed map is exactly ``observed_prefix``
    longer than this forward's ids. A cold prefill (priming failed, or the suffix
    still holds vision) passes the full prompt, so the identity does not hold and
    the request stays byte-identical to today.

    The override is installed on this model *instance* (not its class) for one
    request and removed on completion, cancellation, or error, so two backend
    instances sharing a model class, each under its own generation lock, cannot
    race on a shared method; any pre-existing instance override is restored
    exactly rather than deleted.
    """
    if cache_state is None or _vlm_mrope_reuse_arch(model) is None:
        yield
        return
    original = model.get_input_embeddings
    had_own_override = "get_input_embeddings" in vars(model)
    reuse = {"active": None}
    # Drop any map left by an earlier request so the identity below can only be
    # satisfied by this request's priming. A cold prefill recomputes it anyway.
    language_model = getattr(model, "language_model", None)
    if hasattr(language_model, "_position_ids"):
        language_model._position_ids = None
    position_setter = getattr(model, "_set_position_state", None)
    original_rope_index = getattr(language_model, "get_rope_index", None)
    had_own_rope_index = "get_rope_index" in getattr(language_model, "__dict__", {})

    def _suppressing_get_input_embeddings(
        input_ids = None,
        pixel_values = None,
        *args,
        **kwargs,
    ):
        # Decide before delegating: some runtimes clear the primed map inside
        # this call when no pixel values are present.
        if reuse["active"] is None:
            reuse["active"] = _vlm_runtime_reused_prefix(model, input_ids, cache_state)
        features = original(input_ids, pixel_values, *args, **kwargs)
        if reuse["active"] and pixel_values is None:
            features.rope_deltas = None
            features.position_ids = None
        return features

    rope_overridden = False
    embedding_overridden = False
    try:
        if callable(position_setter) and callable(original_rope_index):

            def _wrapper_position_state(input_ids, *_args, **_kwargs):
                position_setter(input_ids)
                return language_model._position_ids, language_model._rope_deltas

            language_model.get_rope_index = _wrapper_position_state
            rope_overridden = True
        model.get_input_embeddings = _suppressing_get_input_embeddings
        embedding_overridden = True
        yield
    finally:
        try:
            if embedding_overridden:
                if had_own_override:
                    model.get_input_embeddings = original
                else:
                    del model.get_input_embeddings
        finally:
            if rope_overridden:
                if had_own_rope_index:
                    language_model.get_rope_index = original_rope_index
                else:
                    del language_model.get_rope_index


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


def _flatten_registered_vlm_content(processor, content):
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return content_to_text(content)

    def _token(name, default):
        for target in (processor, getattr(processor, "tokenizer", None)):
            value = getattr(target, name, None)
            if isinstance(value, str) and value:
                return value
        return default

    image_token = _token("image_token", "<image>")
    audio_token = _token("audio_token", "<audio>")
    video_token = _token("video_token", "<video>")
    markers = {image_token, audio_token, video_token}
    parts = []
    for item in content:
        if not isinstance(item, dict):
            if item is not None:
                parts.append(str(item))
            continue
        item_type = str(item.get("type", "")).lower()
        if item_type in ("image", "image_url", "input_image"):
            parts.append(image_token)
        elif item_type in ("audio", "input_audio"):
            parts.append(audio_token)
        elif item_type in ("video", "input_video", "video_url"):
            parts.append(video_token)
        else:
            text = item.get("text", "") or item.get("content", "")
            if text:
                parts.append(str(text))

    flattened = []
    previous_was_marker = False
    for part in parts:
        if not part:
            continue
        is_marker = part in markers
        if previous_was_marker and not is_marker and not part[0].isspace():
            flattened.append(" ")
        flattened.append(part)
        previous_was_marker = is_marker
    return "".join(flattened)


class _VLMProcessorWithoutImagePadding:
    def __init__(self, processor):
        self._processor = processor

    def __getattr__(self, name):
        return getattr(self._processor, name)

    def process(
        self,
        text,
        images = None,
        return_tensors = "mlx",
        **kwargs,
    ):
        kwargs.pop("padding", None)
        return self._processor(
            text = text,
            images = images,
            return_tensors = return_tensors,
            **kwargs,
        )


def _vlm_rejects_image_padding(error):
    message = str(error)
    return "ImagesKwargs" in message and "unexpected keyword argument 'padding'" in message


def _vlm_text_probe(messages):
    nonce = secrets.token_hex(16)
    markers = []
    media_types = {
        "image",
        "image_url",
        "input_image",
        "audio",
        "input_audio",
        "video",
        "input_video",
        "video_url",
    }

    def replace(content):
        if isinstance(content, str):
            if not content:
                return content
            marker = f"UNSLOTH_VLM_TEXT_{nonce}_{len(markers):08x}"
            markers.append(marker)
            return marker
        if isinstance(content, list):
            return [replace(item) for item in content]
        if not isinstance(content, dict):
            return content
        replaced = dict(content)
        if str(content.get("type", "")).lower() in media_types:
            return replaced
        text_fields = [
            name
            for name in ("text", "content")
            if isinstance(content.get(name), str) and content.get(name)
        ]
        if text_fields:
            marker = f"UNSLOTH_VLM_TEXT_{nonce}_{len(markers):08x}"
            markers.append(marker)
            for name in text_fields:
                replaced[name] = marker
        elif "content" in content:
            replaced["content"] = replace(content["content"])
        return replaced

    return replace(messages), tuple(markers)


def _render_registered_vlm_prompt(
    processor,
    model,
    messages,
    num_images,
    num_audios = 0,
    *,
    enable_thinking = None,
    reasoning_effort = None,
    preserve_thinking = None,
):
    """Render through mlx-vlm when it declares a formatter for this model."""
    from mlx_vlm import prompt_utils

    config, model_type = _mlx_vlm_model_config(model)
    if config is None:
        return None
    if model_type not in getattr(prompt_utils, "MODEL_CONFIG", {}):
        return None

    kwargs = {
        name: value
        for name, value in (
            ("enable_thinking", enable_thinking),
            ("reasoning_effort", reasoning_effort),
            ("preserve_thinking", preserve_thinking),
        )
        if value is not None
    }

    def media_owner(
        counter,
        count,
        fallback = False,
    ):
        owner = next(
            (
                index
                for index, message in enumerate(messages)
                if isinstance(message, dict) and counter(message.get("content")) > 0
            ),
            None,
        )
        if owner is None and count and fallback:
            owner = next(
                (
                    index
                    for index in range(len(messages) - 1, -1, -1)
                    if isinstance(messages[index], dict)
                    and str(messages[index].get("role", "")).lower() == "user"
                ),
                None,
            )
        if count and (owner is None or str(messages[owner].get("role", "")).lower() != "user"):
            raise RuntimeError("Model-aware media recovery requires media on a user turn.")
        return owner

    image_owner = media_owner(_count_vlm_images, num_images)
    audio_owner = media_owner(_count_vlm_audios, num_audios, fallback = True)

    extract_text = getattr(prompt_utils, "extract_text_from_content", content_to_text)

    def extract_content(content):
        if not isinstance(content, list):
            return extract_text(content)
        text_parts = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") not in ("text", "input_text"):
                continue
            text = item.get("text", "") or item.get("content", "")
            if text:
                text_parts.append(text)
        return "".join(text_parts)

    def extract_messages(source_messages):
        return [
            (
                {**message, "content": extract_content(message.get("content"))}
                if isinstance(message, dict)
                else message
            )
            for message in source_messages
        ]

    text_messages = extract_messages(messages)

    def format_messages(source_messages):
        formatted = []
        for index, source_message in enumerate(source_messages):
            model_messages = prompt_utils.apply_chat_template(
                processor,
                config,
                source_message,
                add_generation_prompt = True,
                return_messages = True,
                num_images = num_images if index == image_owner else 0,
                num_audios = num_audios if index == audio_owner else 0,
                **kwargs,
            )
            if not isinstance(model_messages, list):
                raise RuntimeError("mlx-vlm's registered renderer returned invalid messages.")
            source_role = (
                source_message.get("role", "user") if isinstance(source_message, dict) else "user"
            )
            for message in model_messages:
                if isinstance(message, dict):
                    formatted.append(message)
                elif isinstance(message, str):
                    formatted.append({"role": source_role, "content": message})
                else:
                    raise RuntimeError("mlx-vlm's registered renderer returned invalid messages.")
        return formatted

    def candidates(formatted):
        def structured_content(content, field):
            if isinstance(content, str):
                return [{"type": "text", field: content}]
            if not isinstance(content, list):
                return content
            normalized = []
            for part in content:
                if not isinstance(part, dict):
                    normalized.append(part)
                    continue
                value = part.get("text", part.get("content"))
                if not isinstance(value, str):
                    normalized.append(part)
                    continue
                normalized.append(
                    {
                        **{
                            name: item
                            for name, item in part.items()
                            if name not in ("text", "content")
                        },
                        field: value,
                    }
                )
            return normalized

        def structured(field):
            return [
                {
                    **message,
                    "content": structured_content(message.get("content"), field),
                }
                for message in formatted
            ]

        flattened = [
            {
                **message,
                "content": _flatten_registered_vlm_content(processor, message.get("content")),
            }
            for message in formatted
        ]
        return formatted, structured("text"), structured("content"), flattened

    probe_source, markers = _vlm_text_probe(messages)
    probe_messages = extract_messages(probe_source)
    probe_candidates = candidates(format_messages(probe_messages))
    actual_candidates = candidates(format_messages(text_messages))
    for probe_candidate, candidate in zip(probe_candidates, actual_candidates):
        try:
            probe = prompt_utils.get_chat_template(
                processor,
                probe_candidate,
                True,
                **kwargs,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if _vlm_prompt_issue(probe, probe_messages, markers) is not None:
            continue
        try:
            rendered = prompt_utils.get_chat_template(
                processor,
                candidate,
                True,
                **kwargs,
            )
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if _vlm_prompt_issue(rendered, messages) is None:
            return rendered
    raise RuntimeError("mlx-vlm's registered renderer returned an invalid prompt.")


# Rate the chat route decodes uploads to; mlx-vlm does not resample arrays.
_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_PROBE_MESSAGES = [{"role": "user", "content": "audio"}]


def _classify_mlx_audio_type(
    model,
    processor,
    is_vision,
    config_audio_type = None,
):
    """audio_type for the model entry: "audio_vlm" (omni audio input; is_audio
    stays False — it means TTS and redirects in the chat route) or None.

    The checkpoint's own capability comes from unsloth_zoo, which answers it by
    observing whether audio content changes what the processor returns. Two
    things stay here because they are this backend's, not the checkpoint's: the
    waveform arrives at the rate the chat route decodes to, and the prompt is
    rendered through mlx-vlm's registry, where some families accept an audio
    count and silently drop it. The rendered prompt is what the capability call
    probes with, so a family whose marker only its own template emits is judged
    on the real thing.

    This probe speaks for "audio_vlm" and nothing else, so `config_audio_type`
    (the pre-load answer from detect_audio_type) is carried through untouched
    whenever the probe has no standing:

      * a non-vision checkpoint is never asked, so a TTS codec ("snac", "dac",
        "bicodec", "csm") or Whisper keeps the classification it arrived with —
        the worker mirrors this entry over the pre-load config, so returning a
        bare None here would silently strip the chat route's TTS redirect;
      * a probe that could not run (absent or older unsloth_zoo, a raising or
        unrecognised capability result) leaves the pre-load answer standing
        rather than downgrading a model on the strength of a missing
        dependency.

    Only a probe that actually ran and answered may retract "audio_vlm".
    """

    def _probe_says_no():
        # Authoritative for audio_vlm only; anything else passes through.
        return None if config_audio_type == "audio_vlm" else config_audio_type

    if not is_vision or processor is None:
        return config_audio_type
    # All of it inside the guard: a model load must never fail on a probe.
    # BaseException because any escape here aborts the load.
    try:
        from unsloth_zoo.mlx.utils import (
            audio_extractor_sampling_rate,
            audio_input_capability,
        )

        if audio_extractor_sampling_rate(processor) != _AUDIO_INPUT_SAMPLE_RATE:
            logger.info(
                "MLX audio input unavailable: feature extractor is not %d Hz, and "
                "the chat route decodes uploads to that rate.",
                _AUDIO_INPUT_SAMPLE_RATE,
            )
            return _probe_says_no()
        args = (processor, model, _AUDIO_PROBE_MESSAGES, 0)
        marked = _render_registered_vlm_prompt(*args, num_audios = 1)
        if not marked or marked == _render_registered_vlm_prompt(*args, num_audios = 0):
            logger.info(
                "MLX audio input unavailable: mlx-vlm's renderer for this family "
                "places no audio marker."
            )
            return _probe_says_no()
        capability = audio_input_capability(model, processor, texts = marked)
        if capability.capable:
            return "audio_vlm"
        logger.info("MLX audio input unavailable for this model: %s", capability.reason)
        return _probe_says_no()
    except BaseException as exc:
        logger.info(
            "MLX audio capability check did not run (%s); keeping the pre-load "
            "classification %r. Audio input needs an unsloth_zoo release providing "
            "audio_input_capability.",
            type(exc).__name__,
            config_audio_type,
        )
    return config_audio_type


def _count_vlm_images(content):
    if isinstance(content, list):
        return sum(_count_vlm_images(item) for item in content)
    if not isinstance(content, dict):
        return 0
    if str(content.get("type", "")).lower() in ("image", "image_url", "input_image"):
        return 1
    return _count_vlm_images(content.get("content"))


def _count_vlm_audios(content):
    if isinstance(content, list):
        return sum(_count_vlm_audios(item) for item in content)
    if not isinstance(content, dict):
        return 0
    if str(content.get("type", "")).lower() in ("audio", "input_audio"):
        return 1
    return _count_vlm_audios(content.get("content"))


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


def _vlm_prompt_issue(
    prompt,
    messages,
    text_markers = (),
):
    if not isinstance(prompt, str) or not prompt.strip():
        return "an empty prompt"
    if _prompt_serializes_vlm_media(prompt, messages):
        return "serialized structured image content"
    positions = []
    for marker in text_markers:
        if prompt.count(marker) != 1:
            return "dropped, duplicated, or reordered message text"
        positions.append(prompt.index(marker))
    if positions != sorted(positions):
        return "dropped, duplicated, or reordered message text"
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
MLX_VLM_PREFILL_STEP_SIZE = 2048
MLX_VLM_PROMPT_CACHE_MIN_VERSION = "0.6.8"
# mlx-vlm 0.6.8 produces different warm and stable-cold cache bytes for these.
_NONEXACT_MLX_VLM_CACHE_TYPES = frozenset({"idefics2", "kimi_vl", "llava", "llava_next"})


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


def _flatten_kv_entries(cache):
    for entry in cache:
        nested = getattr(entry, "caches", None)
        if nested is None:
            yield entry
        else:
            yield from _flatten_kv_entries(nested)


def _kv_prefix_coverage(cache):
    covered = None
    for entry in _flatten_kv_entries(cache):
        offset = getattr(entry, "offset", None)
        if offset is None:
            return None
        if getattr(entry, "start_position", 0):
            return None
        window = getattr(entry, "max_size", None)
        if window is not None and offset > window:
            return None
        if covered is None:
            covered = offset
        elif covered != offset:
            return None
    return covered


def _exact_vlm_cache_coverage(cache):
    offsets = []
    try:
        for entry in _flatten_kv_entries(cache):
            if hasattr(entry, "offset"):
                offsets.append(int(entry.offset))
    except Exception:
        return None
    if not offsets or any(offset != offsets[0] for offset in offsets[1:]):
        return None
    return offsets[0]


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
        covered = _kv_prefix_coverage(cache)
        if covered is None:
            logger.debug("MLX prompt cache: skipping cache with unverifiable prefix coverage")
            return
        tokens = list(tokens)
        if covered > len(tokens):
            logger.debug(
                "MLX prompt cache: cache covers %d tokens but only %d were tracked",
                covered,
                len(tokens),
            )
            return
        tokens = tokens[:covered]
        if not tokens:
            return
        self._lru.insert_cache(key, tokens, cache)


def _runtime_primes_rope():
    """True when the runtime exposes the whole-prompt RoPE priming entry point.

    The helper lives in ``mlx_vlm.generate`` in 0.5.0 and moved to
    ``mlx_vlm.generate.dispatch`` in the 0.6 package restructuring; 0.4.4 has no
    equivalent, so the suffix-derived delta cannot be corrected there and reuse
    stays refused. Its presence is a capability pre-filter, not a version claim.
    """
    for module_name in ("mlx_vlm.generate.dispatch", "mlx_vlm.generate"):
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        if hasattr(module, "_prime_cached_prefix_rope_state"):
            return True
    return False


def _vlm_mrope_reuse_arch(model):
    """Return the model type when the runtime exposes Qwen-style MRoPE priming."""
    _, arch = _mlx_vlm_model_config(model)
    language_model = getattr(model, "language_model", None)
    if not all(hasattr(language_model, attr) for attr in ("_rope_deltas", "_position_ids")):
        return None
    if not callable(getattr(language_model, "get_rope_index", None)):
        return None
    if any(hasattr(language_model, attr) for attr in ("_rope_delta", "_pos_hw", "_full_attn_mask")):
        return None
    if not _runtime_primes_rope():
        return None
    return arch


def _mlx_vlm_prompt_cache_version_supported():
    try:
        import mlx_vlm
        from packaging.version import Version
        return Version(mlx_vlm.__version__) >= Version(MLX_VLM_PROMPT_CACHE_MIN_VERSION)
    except Exception:
        return False


def _mlx_vlm_prompt_cache_api(model = None):
    """Return the latest exact-cache capabilities when continuation is safe.

    Some VLM language models keep multimodal position, attention, image, or
    generation metadata on the model object rather than in the KV cache.
    Image/generation state cannot be reconstructed by the cache contract.
    Position state is safe only when upstream can prime it from the full prompt
    before trimming to the uncached suffix. Unknown or older combinations
    deliberately fall back to cold prefill.
    """
    if not _mlx_vlm_prompt_cache_version_supported():
        return None
    try:
        from mlx_vlm import PromptCacheState
        from mlx_vlm.apc import _clone_prompt_cache_for_apc, model_apc_mode
    except (ImportError, AttributeError):
        return None
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        return None
    _, model_type = _mlx_vlm_model_config(model)
    if model_type in _NONEXACT_MLX_VLM_CACHE_TYPES:
        return None
    try:
        configs = (getattr(model, "config", None), getattr(model, "_config", None))
        for config in configs:
            text_config = (
                config.get("text_config")
                if isinstance(config, Mapping)
                else getattr(config, "text_config", None)
            )
            cross_attention_layers = (
                text_config.get("cross_attention_layers")
                if isinstance(text_config, Mapping)
                else getattr(text_config, "cross_attention_layers", None)
            )
            if cross_attention_layers:
                return None
    except Exception:
        return None
    try:
        cache_mode = model_apc_mode(language_model)
    except Exception:
        return None
    if cache_mode not in ("block", "exact"):
        return None
    has_unrestorable_generation_state = any(
        hasattr(language_model, attr)
        for attr in (
            "_image_cache",
            "_token_bounds",
            "_generated_ids_list",
            "_last_predicted_patch_id",
        )
    )
    if has_unrestorable_generation_state:
        return None
    has_multimodal_position_state = any(
        hasattr(language_model, attr)
        for attr in (
            "_rope_delta",
            "_rope_deltas",
            "_position_ids",
            "_pos_hw",
            "_full_attn_mask",
        )
    )
    if has_multimodal_position_state and _vlm_mrope_reuse_arch(model) is None:
        # Qwen-style position state is restorable through whole-prompt priming.
        # Other state layouts remain cold because Studio cannot reconstruct them.
        return None
    return PromptCacheState, cache_mode, _clone_prompt_cache_for_apc


def _vlm_media_fingerprint(media):
    """Hash the media that will be paired with a retained multimodal prefix.

    In-memory images include pixel representation and structural attributes.
    Paletted images additionally include their palette and transparency because
    equal palette indices can represent different colors. Media naming a file,
    including a bytes value the runtime would open as a pathname, is refused
    rather than hashed, because that file can change between this hash and the
    read. Any object whose identity cannot be established conservatively
    disables reuse for that request.
    """
    digest = hashlib.sha256()

    def update(value):
        if value is None:
            digest.update(b"none")
            return True
        if isinstance(value, (list, tuple)):
            digest.update(f"{type(value).__name__}:{len(value)}".encode())
            return all(update(item) for item in value)
        if isinstance(value, (str, bytes, os.PathLike)):
            # Studio decodes media in process, so a name here is not the payload
            # the model will actually read -- and a bytes value is opened as a
            # pathname too. The file can change between this hash and that read,
            # which would retain one image's state under another's identity, so
            # refuse rather than fingerprint external mutable state.
            return False
        if isinstance(value, (bytearray, memoryview)):
            digest.update(bytes(value))
            return True
        tobytes = getattr(value, "tobytes", None)
        if not callable(tobytes):
            return False
        digest.update(
            repr(
                (
                    type(value).__module__,
                    type(value).__qualname__,
                    getattr(value, "mode", None),
                    getattr(value, "size", None),
                    getattr(value, "shape", None),
                    str(getattr(value, "dtype", None)),
                )
            ).encode()
        )
        getpalette = getattr(value, "getpalette", None)
        if callable(getpalette):
            digest.update(repr(getpalette()).encode())
            digest.update(repr(getattr(value, "info", {}).get("transparency")).encode())
        digest.update(tobytes())
        return True

    try:
        return digest.hexdigest() if update(media) else None
    except Exception as exc:
        logger.debug("MLX VLM prompt cache: could not fingerprint media: %s", exc)
        return None


def _vlm_prompt_cache_state_nbytes(state):
    cache = getattr(state, "cache", None)
    if not cache:
        return None
    total = len(getattr(state, "token_ids", None) or ()) * 8
    try:
        for entry in _flatten_kv_entries(cache):
            nbytes = getattr(entry, "nbytes")
            total += int(nbytes)
    except (AttributeError, TypeError, ValueError):
        return None
    return total


# The fields Studio knows how to restore when it cuts a state. A plain entry
# carries only the first set; a sliding-window entry adds its window bounds and
# ring position. Anything else — wrapper tensors, nested caches, unknown fields —
# holds state derived from one sequence that no cut can rebuild.
_PLAIN_KV_ATTRIBUTES = frozenset({"keys", "values", "offset", "step", "nbytes"})
_ROTATING_KV_ATTRIBUTES = _PLAIN_KV_ATTRIBUTES | {"max_size", "keep", "_idx"}


def _vlm_cache_entry_shape(entry):
    """Classify an entry as ``"plain"``, ``"rotating"``, or ``None`` (refused)."""
    keys = getattr(entry, "keys", None)
    values = getattr(entry, "values", None)
    attributes = getattr(entry, "__dict__", None)
    if keys is None or values is None or attributes is None:
        return None
    # Both tensors must already hold every token the counters claim, or a cut
    # would leave the runtime a state asserting more cached tokens than it has.
    offset = getattr(entry, "offset", 0)
    if keys.shape[2] < offset or values.shape[2] < offset:
        return None
    names = set(attributes)
    if not names - _PLAIN_KV_ATTRIBUTES:
        return "plain"
    if names - _ROTATING_KV_ATTRIBUTES or not {"max_size", "_idx"} <= names:
        return None
    if entry.keep != 0 or _rotating_entry_is_wrapped(entry):
        return None
    return "rotating"


def _rotating_entry_is_wrapped(entry):
    """Whether the ring has already discarded its earliest tokens.

    A wrapped ring holds only its most recent window, so it can neither be cut
    to a shorter prefix nor be shown to continue identically to a fresh prefill.
    Such states are not retained, which leaves those conversations prefilling
    from cold exactly as before.
    """
    return entry._idx != entry.offset or entry.offset >= entry.max_size


def _vlm_cache_is_retainable(cache, allow_rotating = True):
    if not cache:
        return False
    for entry in cache:
        shape = _vlm_cache_entry_shape(entry)
        if shape is None or (shape == "rotating" and not allow_rotating):
            return False
    return True


def _cut_vlm_cache_to_prefix(cache, prefix):
    """Make every entry end at ``prefix``, or report that reuse must be abandoned.

    Retention already refuses wrapped rings, so every entry reaching here holds a
    contiguous run from the start and can be cut by slicing and resetting its
    counters — the restoration the runtime's own truncation omits.
    """
    plan = []
    for entry in cache:
        shape = _vlm_cache_entry_shape(entry)
        if shape is None or entry.offset < prefix:
            return False
        plan.append(entry)
    for entry in plan:
        if entry.keys.shape[2] > prefix:
            entry.keys = entry.keys[:, :, :prefix, :]
        if entry.values.shape[2] > prefix:
            entry.values = entry.values[:, :, :prefix, :]
        entry.offset = prefix
        if hasattr(entry, "_idx"):
            entry._idx = prefix
    return True


def _studio_prompt_cache_state_cls(base_cls, _cache = {}):
    """Subclass the runtime's state so its prefix callback applies Studio's cut."""
    if not isinstance(base_cls, type):
        return base_cls
    cls = _cache.get(base_cls)
    if cls is None:

        class _StudioPromptCacheState(base_cls):
            observed_prefix = 0
            prefix_checked = False

            def find_prefix_length(self, new_ids):
                prefix = int(super().find_prefix_length(new_ids))
                self.observed_prefix = 0
                self.prefix_checked = True
                prefix = prefix // MLX_VLM_PREFILL_STEP_SIZE * MLX_VLM_PREFILL_STEP_SIZE
                if len(new_ids) - prefix <= MLX_VLM_PREFILL_STEP_SIZE:
                    prefix -= MLX_VLM_PREFILL_STEP_SIZE
                if (
                    0 < prefix < len(new_ids)
                    and len(new_ids) - prefix > MLX_VLM_PREFILL_STEP_SIZE
                    and self.cache
                ):
                    if not _cut_vlm_cache_to_prefix(self.cache, prefix):
                        return 0
                    self.observed_prefix = prefix
                    return prefix
                return 0

        cls = _cache[base_cls] = _StudioPromptCacheState
    return cls


class _MLXVLMPromptCacheHistory:
    """Retain only committed, byte-bounded VLM state snapshots.

    Entries are addressed by content, not by conversation. Within one
    ``(model, adapter state, media fingerprint)`` scope a request reuses the
    retained entry whose prompt is the longest literal prefix of its own, so a
    retained state is never applied to a prompt that does not continue the one
    it was built from; upstream then revalidates the exact token prefix.

    ``mlx-vlm`` mutates KV objects while generation is still in progress.
    Fetching therefore returns a deep copy so cancellation, close, or an
    exception cannot corrupt the last committed snapshot. Insertions verify
    that every cache layer covers the same non-rotated prefix and that its
    memory can be measured before it enters the pool.
    """

    def __init__(
        self,
        max_entries,
        max_bytes,
        step_size = None,
    ):
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._step_size = step_size
        self._nbytes = 0
        self._entries = OrderedDict()
        self._next_id = 0

    def fetch(self, state_cls, scope, prompt):
        matched_id, matched_length = None, 0
        for entry_id, (_, _, entry_scope, entry_prompt) in self._entries.items():
            if entry_scope != scope or len(entry_prompt) <= matched_length:
                continue
            if prompt.startswith(entry_prompt):
                matched_id, matched_length = entry_id, len(entry_prompt)
        continued_id = matched_id
        if matched_id is None:
            matched_id = next(
                (
                    entry_id
                    for entry_id, (_, _, entry_scope, _) in reversed(self._entries.items())
                    if entry_scope == scope
                ),
                None,
            )
        if matched_id is None:
            return state_cls(), [], None
        state = copy.deepcopy(self._entries[matched_id][0])
        self._entries.move_to_end(matched_id)
        return state, list(getattr(state, "token_ids", None) or ()), continued_id

    def fetch_exact(self, scope, token_ids, min_prefix_tokens, max_prefix_tokens, clone):
        matched_id, matched_length = None, 0
        for entry_id, (state, _, entry_scope, _) in self._entries.items():
            stored = list(getattr(state, "token_ids", None) or ())
            if (
                entry_scope == scope
                and min_prefix_tokens <= len(stored) < len(token_ids)
                and (
                    max_prefix_tokens is None
                    or max_prefix_tokens <= 0
                    or len(stored) <= max_prefix_tokens
                )
                and len(stored) > matched_length
                and list(token_ids[: len(stored)]) == stored
            ):
                matched_id, matched_length = entry_id, len(stored)
        if matched_id is None:
            return None, 0, None
        state = self._entries[matched_id][0]
        if getattr(state, "_refresh_required", False):
            self._entries.move_to_end(matched_id)
            return None, matched_length, matched_id
        try:
            cache = clone(
                state.cache,
                min_capacity_tokens = len(token_ids) + 1,
            )
        except Exception:
            return None, 0, None
        if cache is None:
            return None, 0, None
        self._entries.move_to_end(matched_id)
        return cache, matched_length, matched_id

    def mark_exact_refresh(self, entry_id):
        entry = self._entries.get(entry_id)
        if entry is None:
            return False
        entry[0]._refresh_required = True
        return True

    def insert(
        self,
        commit,
        state,
        prompt_tokens = None,
        exact = False,
    ):
        scope, prompt, continued_id = commit
        tokens = list(getattr(state, "token_ids", None) or ())
        cache = getattr(state, "cache", None) or ()
        covered = _exact_vlm_cache_coverage(cache) if exact else _kv_prefix_coverage(cache)
        if covered is None or covered <= 0 or covered > len(tokens):
            logger.debug("MLX VLM prompt cache: skipping unverifiable prefix coverage")
            return False
        if not exact and not _vlm_cache_is_retainable(
            cache, allow_rotating = hasattr(state, "observed_prefix")
        ):
            logger.debug("MLX VLM prompt cache: skipping cache Studio cannot cut to a prefix")
            return False
        if not exact and self._step_size is not None:
            prompt_tokens = min(int(prompt_tokens or 0), covered)
            covered = ((prompt_tokens - 1) // self._step_size) * self._step_size
            if covered <= 0 or not _cut_vlm_cache_to_prefix(cache, covered):
                return False
        state.token_ids = tokens[:covered]
        nbytes = _vlm_prompt_cache_state_nbytes(state)
        if nbytes is None or nbytes > self._max_bytes:
            logger.debug("MLX VLM prompt cache: skipping unbounded or over-budget state")
            return False
        # Store first, then drop the entry this request continued, so a failure
        # here cannot leave the pool without either version. Superseding keeps a
        # linear conversation at one entry, while a prompt that continues none of
        # the retained entries is added beside them.
        self._entries[self._next_id] = (state, nbytes, scope, prompt)
        self._next_id += 1
        self._nbytes += nbytes
        previous = self._entries.pop(continued_id, None) if continued_id is not None else None
        if previous is not None:
            self._nbytes -= previous[1]
        while len(self._entries) > self._max_entries or self._nbytes > self._max_bytes:
            _, (_, evicted_bytes, _, _) = self._entries.popitem(last = False)
            self._nbytes -= evicted_bytes
        return True


class _StudioVLMExactSnapshot:
    pass


class _StudioVLMExactCacheManager:
    """Stage aligned recurrent snapshots until the request completes."""

    def __init__(self, history, scope, prompt, clone, step_size):
        self._history = history
        self._scope = scope
        self._prompt = prompt
        self._clone = clone
        self._step_size = step_size
        self._token_ids = []
        self._watermark = 0
        self._continued_id = None
        self._pending = None
        self._coordinate_valid = True
        self.observed_prefix = 0

    @property
    def exact_cache_guard_tokens(self):
        if self._watermark <= 0:
            return 0
        return len(self._token_ids) - self._watermark

    def lookup_exact_cache(
        self,
        token_ids,
        extra_hash = 0,
        max_prefix_tokens = None,
        min_prefix_tokens = 0,
    ):
        del extra_hash
        self._token_ids = list(token_ids)
        self._watermark = (
            (len(self._token_ids) - self._step_size - 1) // self._step_size
        ) * self._step_size
        self.observed_prefix = 0
        cache, prefix, entry_id = self._history.fetch_exact(
            self._scope,
            self._token_ids,
            min_prefix_tokens,
            max_prefix_tokens,
            self._clone,
        )
        self._continued_id = entry_id
        if cache is None or len(self._token_ids) - prefix <= self._step_size:
            return None, 0
        self.observed_prefix = prefix
        return cache, prefix

    def store_exact_cache(
        self,
        token_ids,
        prompt_cache,
        extra_hash = 0,
    ):
        del extra_hash
        token_ids = list(token_ids)
        is_watermark = len(token_ids) == self._watermark and self._watermark > 0
        # The full-prompt callback runs after its first decoded token updates KV.
        expected_coverage = len(token_ids) if is_watermark else len(token_ids) + 1
        if _exact_vlm_cache_coverage(prompt_cache) != expected_coverage:
            self._pending = None
            self._coordinate_valid = False
            return False
        if not self._coordinate_valid:
            return False
        if not is_watermark:
            return False
        try:
            cache = self._clone(prompt_cache)
        except Exception:
            return False
        if cache is None:
            return False
        state = _StudioVLMExactSnapshot()
        state.token_ids, state.cache = token_ids, cache
        self._pending = state
        return True

    def commit(self):
        if self._pending is not None:
            return self._history.insert(
                (self._scope, self._prompt, self._continued_id),
                self._pending,
                exact = True,
            )
        if self.observed_prefix > 0 and self._continued_id is not None:
            return self._history.mark_exact_refresh(self._continued_id)
        return False


def _legacy_vlm_cached_tokens(previous_ids, state, response):
    """Infer successful reuse for runtimes without explicit cache metadata.

    Legacy ``mlx-vlm`` accepts a cached prefix only when it is non-empty and
    shorter than the new prompt. The inference is reported only after normal
    exhaustion, when the upstream state contains the completed prompt.
    """
    token_ids = list(getattr(state, "token_ids", None) or ())
    generated_n = int(getattr(response, "generation_tokens", 0) or 0)
    prompt_n = max(0, len(token_ids) - generated_n)
    if not previous_ids or prompt_n <= 0:
        return 0
    current_prompt = token_ids[:prompt_n]
    common = 0
    for old, new in zip(previous_ids, current_prompt):
        if old != new:
            break
        common += 1
    return common if 0 < common < prompt_n else 0


def _build_vlm_generation_stats(response, cached_n):
    """Map total VLM prefill metrics onto Studio's cached/uncached split.

    Upstream measures prefill time against the total logical prompt even when
    part of it came from cache. Scaling throughput by the uncached fraction
    preserves that measured duration while exposing ``cache_n`` separately.
    """
    # mlx-vlm reports total prompt tokens and total/prefill-second even on a hit.
    total_prompt_n = int(getattr(response, "prompt_tokens", 0) or 0)
    prompt_n = max(total_prompt_n - cached_n, 0)
    total_prompt_tps = float(getattr(response, "prompt_tps", 0.0) or 0.0)
    prompt_tps = total_prompt_tps * prompt_n / total_prompt_n if total_prompt_n > 0 else 0.0
    return _build_generation_stats(
        prompt_n,
        prompt_tps,
        getattr(response, "generation_tokens", 0),
        getattr(response, "generation_tps", 0.0),
        cached_n,
    )


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
        self._vlm_prompt_cache_history = None
        self._vlm_prompt_cache_unavailable = False

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
        self._clear_vlm_prompt_cache()

    def _clear_vlm_prompt_cache(self):
        self._vlm_prompt_cache_history = None
        self._vlm_prompt_cache_unavailable = False

    def _prepare_vlm_prompt_cache(self, prompt, image, adapter_state):
        """Create a request-local state for one exact compatibility domain.

        A rendered prompt, loaded model, adapter selection, and media
        fingerprint are all required. Fetch/copy failure falls back to a fresh
        state with no commit context, preserving the prior snapshot even if the
        cold request succeeds.
        """
        if not prompt or not self.active_model_name:
            return None, None, [], None
        if self._vlm_prompt_cache_unavailable:
            return None, None, [], None
        media_fingerprint = _vlm_media_fingerprint(image)
        if media_fingerprint is None:
            return None, None, [], None
        api = _mlx_vlm_prompt_cache_api(self._model)
        if api is None:
            self._vlm_prompt_cache_unavailable = True
            return None, None, [], None
        state_cls, cache_mode, clone = api
        state_cls = _studio_prompt_cache_state_cls(state_cls)
        if self._vlm_prompt_cache_history is None:
            max_bytes = _prompt_cache_max_bytes(self._memory_limits_applied.get("recommended_gb"))
            if max_bytes <= 0:
                self._vlm_prompt_cache_unavailable = True
                return None, None, [], None
            self._vlm_prompt_cache_history = _MLXVLMPromptCacheHistory(
                PROMPT_CACHE_ENTRIES,
                max_bytes,
                MLX_VLM_PREFILL_STEP_SIZE,
            )
        scope = (
            self.active_model_name,
            repr(adapter_state),
            media_fingerprint,
        )
        if cache_mode == "exact":
            manager = _StudioVLMExactCacheManager(
                self._vlm_prompt_cache_history,
                scope,
                prompt,
                clone,
                MLX_VLM_PREFILL_STEP_SIZE,
            )
            return None, None, [], manager
        try:
            state, previous_ids, continued_id = self._vlm_prompt_cache_history.fetch(
                state_cls,
                scope,
                prompt,
            )
        except Exception as exc:
            logger.debug("MLX VLM prompt cache lookup failed: %s", exc)
            try:
                state = state_cls()
            except Exception:
                return None, None, [], None
            return state, None, [], None
        return state, (scope, prompt, continued_id), previous_ids, None

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
        # A successful reload may reuse the same public identifier for different
        # weights or adapters; no retained VLM state survives that model object swap.
        self._clear_vlm_prompt_cache()

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

        _audio_type = _classify_mlx_audio_type(
            model,
            self._processor,
            is_vision,
            config_audio_type = getattr(config, "audio_type", None),
        )
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
            # Mirrors utils.models.model_config semantics (is_audio == TTS).
            "is_audio": _audio_type is not None and _audio_type != "audio_vlm",
            "audio_type": _audio_type,
            "has_audio_input": is_audio_input_type(_audio_type),
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
        has_structured_image = any(
            _count_vlm_images(message.get("content")) > 0
            for message in full_messages
            if isinstance(message, dict)
        )
        if self._is_vlm and image is not None and not has_structured_image:
            for msg in reversed(full_messages):
                if msg.get("role") == "user":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        msg["content"] = [
                            {"type": "image"},
                            {"type": "text", "text": content},
                        ]
                    elif isinstance(content, list):
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
        if prompt_issue is None:
            probe_messages, text_markers = _vlm_text_probe(messages)
            if text_markers:
                try:
                    probe_prompt = apply_chat_template_for_generation(
                        chat_target,
                        probe_messages,
                        tools = tools,
                        enable_thinking = enable_thinking,
                        reasoning_effort = reasoning_effort,
                        preserve_thinking = preserve_thinking,
                    )
                except Exception as exc:
                    prompt_error = exc
                    prompt_issue = "a text-integrity rendering error"
                else:
                    prompt_issue = _vlm_prompt_issue(
                        probe_prompt,
                        probe_messages,
                        text_markers,
                    )
        if prompt_issue and has_tool_history:
            raise RuntimeError(
                f"VLM chat template returned {prompt_issue} and cannot be recovered "
                "without dropping tool-call history."
            ) from prompt_error

        if images is not None and prompt_issue:
            if tools:
                if prompt_error is not None:
                    raise prompt_error
                raise RuntimeError(
                    f"VLM chat template returned {prompt_issue} and cannot be recovered "
                    "without dropping requested tools."
                )
            try:
                recovered_prompt = _render_registered_vlm_prompt(
                    self._processor,
                    self._model,
                    messages,
                    len(images),
                    enable_thinking = enable_thinking,
                    reasoning_effort = reasoning_effort,
                    preserve_thinking = preserve_thinking,
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
                # Upstream seeds history-based logits processors only from the
                # uncached suffix. Reuse would therefore change penalty semantics.
                if _rep_active or presence_penalty:
                    cache_state, cache_commit, previous_ids, exact_manager = None, None, [], None
                else:
                    cache_state, cache_commit, previous_ids, exact_manager = (
                        self._prepare_vlm_prompt_cache(prompt, image, _adapter_state)
                    )
                request_kwargs = dict(vlm_kwargs)
                if cache_state is not None:
                    request_kwargs["prompt_cache_state"] = cache_state
                if exact_manager is not None:
                    request_kwargs["apc_manager"] = exact_manager
                if cache_state is not None or exact_manager is not None:
                    request_kwargs["prefill_step_size"] = MLX_VLM_PREFILL_STEP_SIZE
                final_response = None
                cached_n = 0
                completed = False
                cache_abandoned = False
                try:
                    # Emit any prefilled <think> block before the first token so the
                    # UI renders it during prefill, matching _generate_text. Done
                    # inside the adapter context so an unsupported request raises
                    # before any output escapes.
                    if cumulative:
                        yield cumulative
                    reuse_state = cache_state if cache_state is not None else exact_manager
                    with _temporary_mlx_vlm_rope_suppression(self._model, reuse_state):

                        def _responses():
                            nonlocal cache_abandoned
                            yielded = False
                            try:
                                for response in vlm_stream(
                                    self._model,
                                    self._processor,
                                    prompt,
                                    images,
                                    **request_kwargs,
                                ):
                                    yielded = True
                                    yield response
                            except ValueError as error:
                                if yielded or not _vlm_rejects_image_padding(error):
                                    raise
                                cache_abandoned = True
                                self._vlm_prompt_cache_history = None
                                self._vlm_prompt_cache_unavailable = True
                                cold_request_kwargs = dict(request_kwargs)
                                for name in (
                                    "prompt_cache_state",
                                    "apc_manager",
                                    "prefill_step_size",
                                ):
                                    cold_request_kwargs.pop(name, None)
                                yield from vlm_stream(
                                    self._model,
                                    _VLMProcessorWithoutImagePadding(self._processor),
                                    prompt,
                                    images,
                                    **cold_request_kwargs,
                                )

                        for response in _responses():
                            final_response = response
                            exposed_cached_n = getattr(response, "cached_tokens", None)
                            if exposed_cached_n is not None:
                                cached_n = int(exposed_cached_n or 0)
                            token_text = (
                                response.text if hasattr(response, "text") else str(response)
                            )
                            cumulative += token_text
                            yield cumulative
                            if cancel_event and cancel_event.is_set():
                                break
                        else:
                            completed = True
                    cancelled = cancel_event is not None and cancel_event.is_set()
                    if (
                        completed
                        and not cancelled
                        and not cache_abandoned
                        and cache_state is not None
                        and cache_commit is not None
                        and final_response is not None
                    ):
                        if not hasattr(final_response, "cached_tokens"):
                            if getattr(cache_state, "prefix_checked", False):
                                cached_n = cache_state.observed_prefix
                            else:
                                cached_n = _legacy_vlm_cached_tokens(
                                    previous_ids,
                                    cache_state,
                                    final_response,
                                )
                        history = self._vlm_prompt_cache_history
                        if history is not None:
                            try:
                                history.insert(
                                    cache_commit,
                                    cache_state,
                                    prompt_tokens = getattr(final_response, "prompt_tokens", 0),
                                )
                            except Exception as exc:
                                logger.debug("MLX VLM prompt cache insert failed: %s", exc)
                    if (
                        completed
                        and not cancelled
                        and not cache_abandoned
                        and exact_manager is not None
                    ):
                        try:
                            exact_manager.commit()
                        except Exception as exc:
                            logger.debug("MLX VLM exact prompt cache insert failed: %s", exc)
                finally:
                    # mlx_vlm exposes the same stats fields as mlx_lm.
                    if final_response is not None:
                        self.last_generation_stats = _build_vlm_generation_stats(
                            final_response,
                            cached_n,
                        )

        yield from normalize_reasoning_snapshots(
            _stream_vlm_snapshots(), chat_target, cancel_event, tools = tools
        )

    def generate_audio_input_response(
        self,
        messages,
        system_prompt,
        audio_array,
        max_new_tokens = 512,
        use_adapter = None,
        cancel_event = None,
        **_sampler,
    ):
        """Audio-input chat (omni models): waveform in, incremental text deltas
        out (the audio route forwards deltas, unlike the snapshot-diffing
        text/vision paths). Greedy, so the worker's sampler kwargs go unused."""
        entry = self.models.get(self.active_model_name) or {}
        if entry.get("audio_type") != "audio_vlm":
            raise RuntimeError(
                "Audio input is not supported for this model on the MLX backend: "
                "no verified audio-capable tower/processor was detected at load."
            )

        from mlx_vlm import stream_generate as vlm_stream

        # Only the CURRENT user turn may caption the audio; never older history.
        user_text = ""
        for msg in reversed(messages or []):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = content_to_text(msg.get("content") or "").strip()
                break
        if not user_text:
            user_text = "Please transcribe this audio."
        if not system_prompt:
            system_prompt = "You are an assistant that transcribes speech accurately."

        audio_messages = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": user_text}]},
        ]
        prompt = _render_registered_vlm_prompt(
            self._processor,
            self._model,
            audio_messages,
            num_images = 0,
            num_audios = 1,
        )
        if prompt is None:
            raise RuntimeError(
                "mlx-vlm has no registered prompt renderer for this model family; "
                "cannot build an audio prompt."
            )

        logger.info("MLX audio-input generating: prompt_len=%d", len(prompt))
        # Hold the adapter state for the whole stream, as text and vision do,
        # so Base-vs-LoRA compare doesn't run the adapter on both sides.
        with self._generation_lock, _temporary_mlx_adapter_state(self._model, use_adapter):
            final_response = None
            try:
                for response in vlm_stream(
                    self._model,
                    self._processor,
                    prompt,
                    audio = [audio_array],
                    max_tokens = max_new_tokens,
                    temperature = 0.0,
                ):
                    final_response = response
                    token_text = response.text if hasattr(response, "text") else str(response)
                    if token_text:
                        yield token_text
                    if cancel_event and cancel_event.is_set():
                        break
            finally:
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
        yield from self.generate_chat_response(
            cancel_event = cancel_event,
            _adapter_state = use_adapter,
            **gen_kwargs,
        )

    def reset_generation_state(self, caller_cancel_event = None):
        # caller_cancel_event: signature parity with the orchestrator; unused here.
        import mlx.core as mx
        import gc

        gc.collect()
        mx.clear_cache()
