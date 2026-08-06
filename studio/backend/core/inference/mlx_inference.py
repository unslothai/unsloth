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
    markup_for_tokenizer,
    neutralize_control_markup_in_messages,
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


def _render_registered_vlm_prompt(
    processor,
    model,
    messages,
    num_images,
    num_audios = 0,
):
    """Render through mlx-vlm when it declares a formatter for this model."""
    from mlx_vlm import prompt_utils

    config, model_type = _mlx_vlm_model_config(model)
    if config is None:
        return None
    if model_type not in getattr(prompt_utils, "MODEL_CONFIG", {}):
        return None

    # Recovery path: renders the caller's original list, not the neutralized copy (#7066).
    rendered = prompt_utils.apply_chat_template(
        processor,
        config,
        neutralize_control_markup_in_messages(messages, None, markup_for_tokenizer(processor)),
        add_generation_prompt = True,
        num_images = num_images,
        num_audios = num_audios,
    )
    if isinstance(rendered, str) and rendered.strip():
        return rendered
    raise RuntimeError("mlx-vlm's registered renderer returned an empty prompt.")


# Rate the chat route decodes uploads to; mlx-vlm does not resample arrays.
_AUDIO_INPUT_SAMPLE_RATE = 16000
_AUDIO_PROBE_MESSAGES = [{"role": "user", "content": "audio"}]
# Same turn with and without an image part, so a diff isolates the image marker.
_IMAGE_PROBE_MESSAGES = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "hi"}]}
]
_TEXT_PROBE_MESSAGES = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


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

# MLX takes a bit width, so this vocabulary is unrelated to llama.cpp's
# cache_type_kv names. Every width mx.quantize supports; mlx-lm adds no domain.
MLX_KV_BITS_CHOICES = (8, 6, 5, 4, 3, 2)
# mlx-lm/mlx-vlm quantize the cache in groups of this many elements; a head
# dimension that is not a multiple of it makes mx.quantize raise.
MLX_KV_GROUP_SIZE = 64
# Enabling quantization costs cross-turn prompt-cache reuse on some vision
# models; surfaced with the resolved setting so an API client sees it too.
MLX_KV_QUANT_NO_REUSE = (
    "The installed mlx-lm cannot measure a quantized cache entry, so prompt-cache "
    "reuse across turns is disabled while this is on."
)
MLX_KV_QUANT_VLM_CACHE_NOTE = (
    "On vision models, quantization starts once the cache reaches {start} tokens."
)


def _kv_entry_nbytes(entry):
    """Bytes held by one cache entry, or None when it cannot be measured.

    Read straight off the property, because that is what decides admission:
    upstream's LRUPromptCache.insert_cache sums ``c.nbytes`` itself, so an
    entry whose property raises (mlx-lm 0.31.2's QuantizedKVCache, a missing
    tree_reduce import) cannot enter the cache however else it could be sized.
    Measuring it another way here would only promise reuse that never happens.
    """
    try:
        return int(entry.nbytes)
    except Exception:
        return None


def _normalize_mlx_kv_bits(value):
    """Supported bit width, or None when unset or out of domain."""
    if value is None:
        return None
    try:
        bits = int(value)
    except (TypeError, ValueError):
        logger.warning("MLX kv_bits=%r is not an integer; ignoring", value)
        return None
    if bits not in MLX_KV_BITS_CHOICES:
        logger.warning(
            "MLX kv_bits=%s unsupported (choose %s); ignoring",
            bits,
            " or ".join(str(b) for b in MLX_KV_BITS_CHOICES),
        )
        return None
    return bits


def _kv_quant_probe(language_model, entries, bits):
    """Attempt the conversion the runtime will perform, on a real cache.

    Static proxies proved wrong in both directions: a model can declare a
    head_dim it does not use for the cache, and a window can be spelled
    differently from entry to entry. So populate one token and try the
    conversion that generation would do.

    Returns ``(converted, skipped, failure, retainable)``. ``retainable`` is
    False when a converted entry's size cannot be read, because the prompt
    cache budgets by size and upstream recomputes it internally.
    """
    import mlx.core as mx

    # The forward pass below runs the model's own layers, some of which draw
    # random numbers even in eval; restore the stream so merely asking about a
    # setting cannot change later sampled output.
    convertible = [entry for entry in entries if getattr(entry, "to_quantized", None) is not None]
    if not convertible:
        # Nothing the quantizer would convert, so the forward pass below would
        # cost a full model call to reach a verdict already known.
        return 0, len(entries), None, True
    if any(
        getattr(entry, name, None) is not None
        for entry in convertible
        for name in ("max_size", "window_size")
    ):
        # A bounded ring cannot be probed in its wrapped state, and once wrapped
        # its conversion keeps an absolute offset past its physical storage.
        # Knowable from the cache alone, so refuse before the forward pass.
        return 0, 0, "it uses a bounded sliding window", True

    # mx.random.state is a list mutated in place, so it must be snapshotted and
    # restored element-wise; rebinding the name restores nothing.
    rng_state = list(mx.random.state)
    try:
        try:
            language_model(mx.array([[0]]), cache = entries)
            mx.eval([getattr(entry, "state", None) for entry in entries])
        except Exception as exc:
            return 0, 0, f"its cache could not be exercised ({type(exc).__name__})", True

        converted = skipped = 0
        retainable = True
        for entry in entries:
            convert = getattr(entry, "to_quantized", None)
            if convert is None:
                skipped += 1
                continue
            try:
                quantized = convert(group_size = MLX_KV_GROUP_SIZE, bits = bits)
                mx.eval(quantized.state)
                converted += 1
            except Exception as exc:
                return converted, skipped, f"MLX cannot quantize it ({type(exc).__name__})", True
            # Through the same helper insertion uses: reading .nbytes directly
            # would call a converted entry unmeasurable on an mlx-lm whose
            # QuantizedKVCache.nbytes raises, and report a caveat that is false.
            if retainable and _kv_entry_nbytes(quantized) is None:
                retainable = False
        return converted, skipped, None, retainable
    finally:
        mx.random.state[:] = rng_state


def _kv_quant_eligibility(
    model,
    is_vlm,
    bits = MLX_KV_BITS_CHOICES[0],
):
    """Whether KV quantization can apply to this model, before generating.

    Returns ``(verdict, reason, retainable)``, verdict in
    full/partial/none/refused. Eligibility only: what the runtime converts stays
    its own decision. Refusing here is what stops an ineligible model raising
    mid-generation, once the leading entries are already converted.
    """
    language_model = getattr(model, "language_model", model) if is_vlm else model
    try:
        if is_vlm:
            from mlx_vlm.models import cache as vlm_cache
            entries = vlm_cache.make_prompt_cache(language_model)
        else:
            from mlx_lm.models import cache as lm_cache
            entries = lm_cache.make_prompt_cache(language_model)
    except Exception as exc:
        logger.warning("MLX KV quantization eligibility probe failed: %s", exc)
        return "none", "this model's KV cache layout could not be inspected", True

    if not entries:
        return "none", "this model builds no KV cache to quantize", True

    converted, skipped, failure, retainable = _kv_quant_probe(language_model, entries, bits)
    # The probe caches are transient. Release them only here, where the probe's
    # own locals are already gone, or their pages stay in the allocator.
    import mlx.core as mx

    entries.clear()
    del entries
    mx.clear_cache()
    if failure is not None:
        return "refused", f"this model's KV cache cannot be quantized: {failure}", True
    if not converted:
        return "none", "this model's KV cache layout cannot be quantized", True
    verdict = "partial" if skipped else "full"
    reason = "only some of this model's layers use a quantizable KV cache" if skipped else ""
    return verdict, reason, retainable


def _vlm_quantized_kv_start():
    """Token offset at which mlx-vlm begins quantizing, per its own default."""
    try:
        from mlx_vlm.generate.common import DEFAULT_QUANTIZED_KV_START
        return int(DEFAULT_QUANTIZED_KV_START)
    except Exception:
        return 5000


# An override replaces an existing template; it never creates one. Both render
# selectors pick their target by whether a template is present, so creating one
# would silently move the render to a different object.
_TEMPLATE_NOT_CAPTURED = object()
MLX_TEMPLATE_NO_TARGET = (
    "this model builds its prompt without a chat template, and supplying one "
    "would drop the markers that place its images and audio"
)
MLX_TEMPLATE_CALLABLE = (
    "the installed mlx-lm supplies this model's template as code rather than text, "
    "and prefers it over any override"
)
MLX_TEMPLATE_NAMED_SET = (
    "this model ships a set of named templates rather than one, and replacing the "
    "set with a single template would lose its tool-calling variant"
)
MLX_TEMPLATE_RENDER_FAILED = "it could not render a conversation: {error}"
MLX_TEMPLATE_NOT_SETTABLE = "this model's template cannot be replaced: {error}"
MLX_TEMPLATE_DROPS_IMAGE = (
    "it does not mark where images go, so this model could not accept image input with it"
)
MLX_TEMPLATE_DROPS_AUDIO = (
    "it does not mark where audio goes, so this model could not accept audio input with it"
)


def _template_install_targets(tokenizer, processor):
    """Objects whose chat_template would be read at render time.

    The processor and the tokenizer can be the same object when a processor
    exposes no nested tokenizer, so the result is de-duplicated by identity.
    """
    seen, targets = [], []
    for candidate in (processor, tokenizer):
        if candidate is None:
            continue
        if any(candidate is other for other in seen):
            continue
        seen.append(candidate)
        targets.append(candidate)
    return targets


def _template_render_targets(tokenizer, processor):
    """Objects a render selector would actually read the template from.

    Deferred to chat_render_target rather than restated (#7066): it also
    requires the processor to be able to render, and mlx-vlm's
    get_chat_template applies the same rule for audio.
    """
    from core.inference.chat_template_helpers import chat_render_target

    target = chat_render_target(processor, tokenizer)
    return [target] if target is not None else []


def _usable_template(value):
    """Whether a chat_template value is one this override can replace."""
    return isinstance(value, str) and value.strip() != ""


def _native_template_source(tokenizer, processor):
    """The object whose chat_template is this model's default.

    The render target, so the editor's "model default" is the text generation
    really uses. Falls back to the tokenizer when that target holds no usable
    string, keeping the nested template reported for a processor carrying a
    named set rather than reporting none at all.
    """
    for target in _template_render_targets(tokenizer, processor):
        if _usable_template(getattr(target, "chat_template", None)):
            return target
    return tokenizer


def _template_override_status(override, tokenizer, processor):
    """Resolve a requested override against this model, without applying it.

    Returns the targets to install onto and a status dict; an empty target list
    with a reason means the override cannot be honored.
    """
    status = {
        "requested": override,
        "applied": None,
        "reason": None,
        # Kept so a later check (the audio marker) can put the model back.
        "restore": [],
    }
    if not override or not override.strip():
        return [], status
    try:
        candidates = _template_install_targets(tokenizer, processor)
        existing = [(c, getattr(c, "chat_template", None)) for c in candidates]
        rendering = _template_render_targets(tokenizer, processor)
    except Exception as exc:
        status["reason"] = MLX_TEMPLATE_NOT_SETTABLE.format(error = exc)
        return [], status
    # Judge only the objects that render: refusing over an unreplaceable
    # template on an object nothing reads would reject a working override.
    blocked = [(c, getattr(c, "chat_template", None)) for c in rendering]
    if any(getattr(c, "_chat_template", None) is not None for c in rendering):
        # mlx-lm can hold the template as a callable, which apply_chat_template
        # prefers over the attribute, so an assignment would be inert.
        status["reason"] = MLX_TEMPLATE_CALLABLE
        return [], status
    if any(isinstance(value, (dict, list)) for _, value in blocked):
        status["reason"] = MLX_TEMPLATE_NAMED_SET
        return [], status
    if not any(_usable_template(value) for _, value in blocked):
        # Nothing to replace. Honorable on a text model, which cannot chat at
        # all without one. Not with a processor: creating a template takes the
        # render from mlx-vlm's fallback, which is what places the markers.
        if processor is not None or not blocked:
            status["reason"] = MLX_TEMPLATE_NO_TARGET
            return [], status
        return [c for c, _ in blocked], status
    # Install on every object holding a replaceable string, so both selectors
    # keep choosing what they chose before.
    return [c for c, value in existing if _usable_template(value)], status


def _audio_marker_survives(processor, model):
    """Whether the installed template still marks where audio goes."""
    if processor is None:
        return True
    args = (processor, model, _AUDIO_PROBE_MESSAGES, 0)
    try:
        marked = _render_registered_vlm_prompt(*args, num_audios = 1)
        return bool(marked) and marked != _render_registered_vlm_prompt(*args, num_audios = 0)
    except BaseException:
        # A load must never fail on a probe, matching _classify_mlx_audio_type.
        return False


def _image_placeholder(tokenizer, processor):
    """The model's own image placeholder, when it names one."""
    for source in (processor, tokenizer):
        token = getattr(source, "image_token", None)
        if isinstance(token, str) and token:
            return token
    return None


def _image_marker_survives(
    tokenizer,
    processor,
    placeholder = None,
):
    """Whether the installed template still marks where an image goes.

    Rendered through the target generation uses, so this sees what a real image
    request would. Three ways to fail: rendering an image the same as no image,
    emitting the structured content object instead of a marker, and dropping
    the placeholder the model names. A bare difference is not enough on its
    own, since a template can render the image as ordinary prose.
    """
    from core.inference.chat_template_helpers import apply_chat_template_for_generation

    targets = _template_render_targets(tokenizer, processor)
    if not targets:
        return True
    target = targets[0]
    try:
        marked = apply_chat_template_for_generation(target, _IMAGE_PROBE_MESSAGES)
        if not marked or marked == apply_chat_template_for_generation(target, _TEXT_PROBE_MESSAGES):
            return False
        if _vlm_prompt_issue(marked, _IMAGE_PROBE_MESSAGES):
            return False
        return placeholder is None or placeholder in marked
    except BaseException:
        # A load must never fail on a probe, matching _audio_marker_survives.
        return False


def _revoke_override_dropping(status, survives, reason):
    """Undo an installed override that stopped marking where media goes.

    Capability was classified against the native template, so an override that
    no longer renders the marker would leave the model advertising an input it
    can no longer place.
    """
    if not status["applied"] or survives():
        return status
    _restore_templates(status["restore"])
    status["applied"] = None
    status["restore"] = []
    status["reason"] = reason
    return status


def _revoke_override_that_drops_audio(status, processor, model):
    return _revoke_override_dropping(
        status,
        lambda: _audio_marker_survives(processor, model),
        MLX_TEMPLATE_DROPS_AUDIO,
    )


def _revoke_override_that_drops_image(
    status,
    tokenizer,
    processor,
    placeholder = None,
):
    return _revoke_override_dropping(
        status,
        lambda: _image_marker_survives(tokenizer, processor, placeholder),
        MLX_TEMPLATE_DROPS_IMAGE,
    )


def _restore_templates(installed):
    """Put back the templates an install replaced, newest first."""
    for target, value in reversed(installed):
        target.chat_template = value


def _install_template_override(override, tokenizer, processor, probe):
    """Install a chat template override, or report why it was not honored.

    ``probe`` renders a short conversation with whatever is installed; a
    template that cannot render would otherwise raise on every generation
    instead of once here, which is how a hand-edited template usually fails.
    """
    targets, status = _template_override_status(override, tokenizer, processor)
    if not targets:
        return status
    # Restore only what was actually assigned: a target that refused the
    # assignment would refuse the restore too, masking the real error.
    installed = []
    template = MLX_TEMPLATE_RENDER_FAILED
    try:
        for target in targets:
            original = target.chat_template
            template = MLX_TEMPLATE_NOT_SETTABLE
            target.chat_template = override
            template = MLX_TEMPLATE_RENDER_FAILED
            installed.append((target, original))
        probe()
    except Exception as exc:
        _restore_templates(installed)
        status["reason"] = template.format(error = exc)
        return status
    status["applied"] = override
    status["restore"] = installed
    return status


def _kv_quant_status(requested_bits, model, is_vlm):
    """Resolve a requested bit width against this model into a status dict."""
    status = {
        "requested_kv_bits": requested_bits,
        "kv_bits": None,
        "eligibility": None,
        "reason": "",
        "note": "",
    }
    if requested_bits is None:
        return status
    verdict, reason, retainable = _kv_quant_eligibility(model, is_vlm, requested_bits)
    status["eligibility"] = verdict
    status["reason"] = reason
    if verdict in ("full", "partial"):
        status["kv_bits"] = requested_bits
        notes = []
        if is_vlm:
            notes.append(MLX_KV_QUANT_VLM_CACHE_NOTE.format(start = _vlm_quantized_kv_start()))
        if not retainable:
            notes.append(MLX_KV_QUANT_NO_REUSE)
        status["note"] = " ".join(notes)
    else:
        logger.info("MLX KV quantization not applied: %s", reason)
    return status


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
        sizes = [_kv_entry_nbytes(entry) for entry in cache]
        if any(size is None for size in sizes):
            logger.debug("MLX prompt cache: skipping state whose size is unmeasurable")
            return
        nbytes = sum(sizes)
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

        # Load-time runtime knobs; every generation path reads them from here
        # rather than from per-request kwargs.
        self._kv_quant = _kv_quant_status(None, None, False)
        # Bound here too, so a load that fails before installing leaves the
        # entry writer and the reason log reading a dict rather than raising.
        self._template_override = _template_override_status(None, None, None)[1]

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

    def _kv_quant_generate_kwargs(self):
        """Load-time runtime knobs for a generate call, empty when unset.

        quantized_kv_start is deliberately not passed: mlx-lm and mlx-vlm ship
        different defaults (0 and 5000) and each runtime keeps its own.
        """
        kv_bits = (getattr(self, "_kv_quant", None) or {}).get("kv_bits")
        return {} if kv_bits is None else {"kv_bits": kv_bits}

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
        kv_bits = None,
        chat_template_override = None,
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

        _audio_type = _classify_mlx_audio_type(
            model,
            self._processor,
            is_vision,
            config_audio_type = getattr(config, "audio_type", None),
        )
        # Classify before the first generation: an ineligible cache would
        # otherwise raise inside maybe_quantize_kv_cache mid-stream, after it
        # had already converted the leading entries.
        self._kv_quant = _kv_quant_status(_normalize_mlx_kv_bits(kv_bits), self._model, is_vision)
        if self._kv_quant["kv_bits"] is not None:
            logger.info(
                "MLX KV cache quantization: %s-bit (%s eligibility)",
                self._kv_quant["kv_bits"],
                self._kv_quant["eligibility"],
            )

        # Captured before installing: chat_template_info must keep reporting
        # what the model shipped with, since the capability classification and
        # the editor's "model default" both read it. From the render target, not
        # the nested tokenizer: on a processor that owns its own template those
        # are different objects, and saving the wrong default back would install
        # the tokenizer's template over the processor.
        native_source = _native_template_source(self._tokenizer, self._processor)
        native_template = getattr(native_source, "chat_template", None)
        native_marks_audio = bool(
            chat_template_override
            and is_audio_input_type(_audio_type)
            and _audio_marker_survives(self._processor, self._model)
        )
        image_placeholder = _image_placeholder(self._tokenizer, self._processor)
        self._template_override = _install_template_override(
            chat_template_override,
            self._tokenizer,
            self._processor,
            lambda: self._render_template_probe(is_vision),
        )
        if native_marks_audio:
            _revoke_override_that_drops_audio(self._template_override, self._processor, self._model)
        # Unconditional for vision, unlike audio: a native template that places
        # nothing itself still renders images through _generate_vlm's registered
        # recovery, which _vlm_prompt_issue triggers. An override that renders
        # plain text triggers nothing, so the image is dropped in silence, and
        # gating this on the native template having marked images skipped
        # exactly the models that need the check.
        if is_vision:
            _revoke_override_that_drops_image(
                self._template_override, self._tokenizer, self._processor, image_placeholder
            )
        # Released once the media checks are done: the pairs strongly reference
        # the tokenizer and processor, and nothing reads them afterwards, so
        # keeping them would outlive the unload that nulls both.
        self._template_override["restore"] = []
        if self._template_override["reason"]:
            logger.info(
                "MLX chat template override not applied: %s",
                self._template_override["reason"],
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
            "mlx_kv_bits": self._kv_quant["kv_bits"],
            "mlx_kv_bits_requested": self._kv_quant["requested_kv_bits"],
            "mlx_kv_quant_eligibility": self._kv_quant["eligibility"],
            "mlx_kv_quant_reason": self._kv_quant["reason"],
            "mlx_kv_quant_note": self._kv_quant["note"],
            "chat_template_override_requested": self._template_override["requested"],
            "chat_template_override_reason": self._template_override["reason"],
        }
        # Capture chat_template_info for the worker IPC reply and route capability classification.
        self._populate_chat_template_info(model_name, native_template)

        logger.info("Model %s loaded successfully", model_name)
        return True

    def _render_template_probe(self, is_vision: bool) -> str:
        """Render a short conversation through the path generation will use.

        Must use the same target the real request does. The recovery renderer
        returns None instead of raising for a model outside mlx-vlm's family
        list, so probing it would pass a template that cannot render at all.
        """
        from core.inference.chat_template_helpers import (
            apply_chat_template_for_generation,
            chat_render_target,
        )

        messages = [{"role": "user", "content": "hi"}]
        target = (
            chat_render_target(self._processor)
            if is_vision and self._processor is not None
            else self._tokenizer
        )
        rendered = apply_chat_template_for_generation(target, messages)
        if not rendered or not rendered.strip():
            raise ValueError("the template produced an empty prompt")
        return rendered

    def _populate_chat_template_info(
        self,
        model_name: str,
        native_template = _TEMPLATE_NOT_CAPTURED,
    ) -> None:
        """Mirror InferenceBackend._load_chat_template_info for MLX.

        Stores ``chat_template_info`` on ``self.models[model_name]``. The
        template recorded is the one the model shipped with, not an override:
        the capability classification and the editor's notion of "default"
        both read it, so an override installed on the tokenizer must not
        show up here."""
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
            tpl = (
                getattr(tok, "chat_template", None)
                if native_template is _TEMPLATE_NOT_CAPTURED
                else native_template
            )
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
                gen_kwargs.update(self._kv_quant_generate_kwargs())
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
            chat_render_target,
        )

        # Pick the chat-template-aware caller: processors with their own
        # apply_chat_template + chat_template (e.g. Qwen2.5-VL), else the nested tokenizer.
        # Shared with the healing catalog the route builds ahead of this render, which has
        # to authorize against the same template this line selects (#7066).
        chat_target = chat_render_target(self._processor)

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
        vlm_kwargs.update(self._kv_quant_generate_kwargs())
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
                    # Sampling stays greedy; the runtime knobs below are load-time
                    # state, not the caller's sampler kwargs.
                    temperature = 0.0,
                    **self._kv_quant_generate_kwargs(),
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
