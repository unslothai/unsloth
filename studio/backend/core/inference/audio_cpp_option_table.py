# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Per-model request options and recommended values for the pinned audio.cpp runtime.

The published GGUFs often embed a spec older than the runtime, and some families declare no request
options at all even though their loaders read them. ``STUDIO_OPTION_LISTS`` names the options such a
family reads; it is used only when neither the runtime nor the GGUF declares any. ``OPTION_RULES`` then
adjusts whichever list a model ends up with: the value Unsloth Studio recommends (shown in Advanced
and sent when the user leaves it alone), tighter bounds, and options a workflow has no use for.

Recommended values come from audio.cpp's own web UI table (``webui/configs/model_params.json``),
else the runtime's default in the C++ loader, else the upstream model card; each one names its source.
A default of ``None`` sends nothing, for options the runtime works out per request. The table holds
only for the tags it was checked against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional

# The unslothai/audio.cpp releases this table was checked against (v0.9.1-unsloth.1 is the same
# commit as v0.9.0-unsloth.1).
AUDIO_CPP_OPTION_TABLE_TAGS = frozenset({"v0.9.0-unsloth.1", "v0.9.1-unsloth.1"})

# Enum domains a spec may name instead of listing (audio.cpp src/framework/model_spec/options.cpp).
ENUM_PRESETS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "weight_type_full": ("native", "f32", "f16", "bf16", "q8_0"),
        "weight_type_conv": ("native", "f32", "f16"),
        "weight_type_codec_q8": ("native", "f32", "f16", "q8_0"),
        "text_chunk_mode_full": ("word_budget", "tag_aware", "japanese", "endline"),
        "perf_mode_flash_attention": ("off", "flash_attention"),
        "best_of_n_language": ("auto", "en", "ja"),
    }
)

_EMPTY: Mapping[str, Any] = MappingProxyType({})


@dataclass(frozen = True)
class OptionRule:
    """How Unsloth Studio adjusts one family's options, for every model or only matching ones."""

    # Searched (case-insensitively) in the model's repo id, folder, variant key and file names; None
    # applies to the whole family. Every matching rule applies, in order.
    match: Optional[str] = None
    # Options the family reads but its list lacks, in the spec's request-option shape.
    extra: tuple[dict, ...] = ()
    # Recommended value per option name; None sends nothing and leaves the runtime to decide.
    defaults: Mapping[str, Any] = field(default_factory = lambda: _EMPTY)
    # (min, max) per option name; None keeps that side as declared.
    bounds: Mapping[str, tuple[Optional[float], Optional[float]]] = field(
        default_factory = lambda: _EMPTY
    )
    # Narrowed enum choices per option name.
    values: Mapping[str, tuple[str, ...]] = field(default_factory = lambda: _EMPTY)
    # Option names a workflow never shows or sends, keyed by workflow id or "*" for every workflow.
    hidden: Mapping[str, frozenset[str]] = field(default_factory = lambda: _EMPTY)
    # Where each recommended value comes from: a model_params.json key, a C++ file:line or a URL.
    sources: Mapping[str, str] = field(default_factory = lambda: _EMPTY)


def _opt(name: str, type_: str, description: str, **kw: Any) -> dict:
    return {"name": name, "type": type_, "description": description, **kw}


_CHUNK_MODES = ["default", "tag_aware", "japanese", "endline"]

# Shared by the four Qwen3-TTS packages (src/models/qwen3_tts/session.cpp:28-90).
_QWEN3_TTS_OPTIONS = (
    _opt("temperature", "float", "Talker sampling temperature.", min = 0.0, max = 2.0),
    _opt("top_k", "int", "Talker top-k sampling limit.", min = 0),
    _opt("top_p", "float", "Talker nucleus sampling.", min = 0.0, max = 1.0),
    _opt("repetition_penalty", "float", "Talker repetition penalty.", min = 1.0, max = 2.0),
    _opt("do_sample", "bool", "Sample the talker; off decodes greedily."),
    _opt("subtalker_do_sample", "bool", "Sample the code predictor; off decodes greedily."),
    _opt(
        "subtalker_temperature", "float", "Code predictor sampling temperature.", min = 0.0, max = 2.0
    ),
    _opt("subtalker_top_k", "int", "Code predictor top-k sampling limit.", min = 0),
    _opt("subtalker_top_p", "float", "Code predictor nucleus sampling.", min = 0.0, max = 1.0),
    _opt("max_tokens", "int", "Most codec frames to generate.", min = 1),
    _opt("text_chunk_size", "int", "Most characters per generated text chunk.", min = 1),
)

# MOSS-TTS-Nano and MOSS-TTS-Local read the same sampling names.
_MOSS_SAMPLING = (
    _opt("do_sample", "bool", "Sample audio tokens; off decodes greedily."),
    _opt("temperature", "float", "Audio token sampling temperature.", min = 0.01, max = 2.5),
    _opt("top_p", "float", "Audio token nucleus sampling.", min = 0.0, max = 1.0),
    _opt("top_k", "int", "Audio token top-k sampling limit.", min = 0),
    _opt("repetition_penalty", "float", "Audio token repetition penalty.", min = 1.0, max = 2.0),
    _opt("text_temperature", "float", "Text token sampling temperature.", min = 0.01),
    _opt("max_tokens", "int", "Most audio frames to generate.", min = 1),
)

# Request options a family's loader reads when no spec it ships declares any, keyed by family.
# Names are the C++ reader's; bounds come from the web UI table or the reader's own checks.
STUDIO_OPTION_LISTS: Mapping[str, tuple[dict, ...]] = MappingProxyType(
    {
        # src/models/pocket_tts/session.cpp:136-176, validated at :243-248.
        "pocket_tts": (
            _opt("temperature", "float", "Flow sampling temperature.", min = 0.01),
            _opt("eos_threshold", "float", "End-of-speech logit threshold."),
            _opt("frames_after_eos", "int", "Frames kept after end of speech (-1 = auto).", min = -1),
            _opt("noise_clamp", "float", "Clamp on sampling noise (-1 = off)."),
        ),
        # src/models/moss/moss_tts_nano/session.cpp:66-115.
        "moss_tts_nano": _MOSS_SAMPLING,
        # src/models/moss/moss_tts_local/session.cpp:193-225.
        "moss_tts_local": _MOSS_SAMPLING,
        # src/models/supertonic/session.cpp:190-250; the voice picker comes from the family's voices.
        "supertonic": (
            _opt("speed", "float", "Speaking rate.", min = 0.5, max = 2.0),
            _opt("num_inference_steps", "int", "Flow matching steps.", min = 1),
            _opt("text_chunk_size", "int", "Most characters per generated text chunk.", min = 1),
        ),
        # src/community_models/chatterbox_turbo/session.cpp:19-40, :97.
        "chatterbox_turbo": (
            _opt("temperature", "float", "Speech token sampling temperature.", min = 0.0),
            _opt("top_p", "float", "Speech token nucleus sampling.", min = 0.0, max = 1.0),
            _opt("top_k", "int", "Speech token top-k sampling limit.", min = 0),
            _opt("repetition_penalty", "float", "Speech token repetition penalty.", min = 0.0),
            _opt("text_chunk_size", "int", "Most characters per generated text chunk.", min = 1),
        ),
        # src/models/voxcpm2/session.cpp:336-341, :593-665.
        "voxcpm2": (
            _opt("num_inference_steps", "int", "Flow matching steps.", min = 1),
            _opt("guidance_scale", "float", "Classifier-free guidance.", min = 0.0, max = 5.0),
            _opt("min_tokens", "int", "Fewest audio patches to generate.", min = 0),
            _opt("max_tokens", "int", "Most audio patches to generate (0 = model limit).", min = 0),
            _opt("retry_badcase", "bool", "Retry a take that runs far too long."),
            _opt("retry_badcase_max_times", "int", "Most retries of a bad take.", min = 0),
            _opt(
                "retry_badcase_ratio_threshold",
                "float",
                "Audio-to-text length ratio that counts as a bad take.",
                min = 0.0,
            ),
            _opt(
                "text_chunk_mode", "enum", "How the text is split into chunks.", values = _CHUNK_MODES
            ),
        ),
        "qwen3_tts": _QWEN3_TTS_OPTIONS,
        # src/models/chatterbox/session.cpp:42-70. Exaggeration and guidance are clone tool options.
        "chatterbox": (
            _opt("temperature", "float", "Speech token sampling temperature.", min = 0.0, max = 2.0),
            _opt(
                "repetition_penalty", "float", "Speech token repetition penalty.", min = 1.0, max = 2.0
            ),
            _opt("min_p", "float", "Speech token min-p sampling.", min = 0.0, max = 1.0),
            _opt("top_p", "float", "Speech token nucleus sampling.", min = 0.0, max = 1.0),
        ),
        # src/models/index_tts2/request.cpp:94-195. The emotion options are clone tool options.
        "index_tts2": (
            _opt("use_random_emotion", "bool", "Pick a random emotion; lowers voice fidelity."),
            _opt("duration_factor", "float", "Length multiplier (>1 slower).", min = 0.5, max = 2.0),
            _opt("interval_silence_ms", "int", "Silence between text segments (ms).", min = 0),
            _opt("num_beams", "int", "Beam search width.", min = 1),
            _opt("max_tokens", "int", "Most mel tokens to generate.", min = 1),
            _opt("temperature", "float", "Sampling temperature.", min = 0.01),
            _opt("top_p", "float", "Nucleus sampling.", min = 0.01, max = 1.0),
            _opt("top_k", "int", "Top-k sampling limit.", min = 1),
            _opt("repetition_penalty", "float", "Repetition penalty.", min = 0.0),
            _opt("length_penalty", "float", "Beam search length penalty."),
            _opt("do_sample", "bool", "Sample tokens; off decodes greedily."),
        ),
        # src/models/vevo2/session.cpp:814-823, :1037-1085.
        "vevo2": (
            _opt("temperature", "float", "AR sampling temperature.", min = 0.01, max = 2.0),
            _opt("top_k", "int", "AR top-k sampling limit.", min = 0),
            _opt("top_p", "float", "AR nucleus sampling.", min = 0.0, max = 1.0),
            _opt("repetition_penalty", "float", "AR repetition penalty.", min = 0.01),
            _opt("max_tokens", "int", "Most AR tokens to generate.", min = 1),
            _opt("num_inference_steps", "int", "Flow matching steps.", min = 1, max = 100),
            _opt(
                "audio_chunk_duration_sec",
                "float",
                "Split long source audio into chunks this long (0 = off).",
                min = 0.0,
            ),
            _opt("cross_fade_duration_sec", "float", "Cross-fade between audio chunks.", min = 0.0),
        ),
        # src/models/higgs_audio_tts/session.cpp:98-130, :249.
        "higgs_audio_tts": (
            _opt("temperature", "float", "Sampling temperature.", min = 0.0),
            _opt("top_p", "float", "Nucleus sampling.", min = 0.0, max = 1.0),
            _opt("top_k", "int", "Top-k sampling limit.", min = 0),
            _opt("max_tokens", "int", "Most audio tokens to generate.", min = 1),
            _opt("text_chunk_size", "int", "Most characters per generated text chunk.", min = 1),
        ),
        # src/community_models/moss_voicegen/session.cpp:278-289.
        "moss_voicegen": (
            _opt("temperature", "float", "Audio token sampling temperature.", min = 0.0),
            _opt("top_p", "float", "Audio token nucleus sampling.", min = 0.0, max = 1.0),
            _opt("top_k", "int", "Audio token top-k sampling limit.", min = 0),
            _opt("repetition_penalty", "float", "Audio token repetition penalty.", min = 0.0),
        ),
        # src/models/omnivoice/session.cpp:77-121.
        "omnivoice": (
            _opt("num_inference_steps", "int", "Unmasking steps.", min = 1),
            _opt("guidance_scale", "float", "Classifier-free guidance.", min = 0.0, max = 5.0),
            _opt("speed", "float", "Speaking rate.", min = 0.5, max = 2.0),
            _opt("duration", "float", "Output length in seconds; overrides speed.", min = 0.0),
            _opt("t_shift", "float", "Timestep shift.", min = 0.0),
            _opt("denoise", "bool", "Denoise the reference clip."),
            _opt("preprocess_prompt", "bool", "Clean up the reference clip before cloning."),
            _opt("postprocess_output", "bool", "Trim silence from the output."),
            _opt("layer_penalty_factor", "float", "Penalty on deeper codebook layers.", min = 0.0),
            _opt("position_temperature", "float", "Unmasking position temperature.", min = 0.0),
            _opt("class_temperature", "float", "Token sampling temperature (0 = greedy).", min = 0.0),
            _opt(
                "audio_chunk_duration", "float", "Chunk length for long outputs (seconds).", min = 0.0
            ),
            _opt(
                "audio_chunk_threshold",
                "float",
                "Output length that switches chunking on (seconds).",
                min = 0.0,
            ),
        ),
        # src/models/vibevoice/session.cpp:70-120.
        "vibevoice": (
            _opt("num_inference_steps", "int", "Diffusion steps.", min = 1),
            _opt("guidance_scale", "float", "Classifier-free guidance.", min = 0.0, max = 5.0),
            _opt("max_length_times", "float", "Output length cap relative to the text.", min = 0.1),
            _opt("do_sample", "bool", "Sample tokens; off decodes greedily."),
            _opt("temperature", "float", "Sampling temperature.", min = 0.05, max = 2.0),
            _opt("top_p", "float", "Nucleus sampling.", min = 0.05, max = 1.0),
            _opt("top_k", "int", "Top-k sampling limit.", min = 0),
        ),
        # The pinned runtime's model_specs/fish_audio.json; the published GGUF embeds none.
        "fish_audio": (
            _opt("max_tokens", "int", "Most semantic tokens to generate per chunk.", min = 1),
            _opt("text_chunk_size", "int", "Most characters per generated text chunk.", min = 1),
            _opt(
                "text_chunk_mode",
                "enum",
                "How the text is split into chunks.",
                preset = "text_chunk_mode_full",
            ),
            _opt("top_p", "float", "Nucleus sampling.", min = 0.0, max = 1.0),
            _opt("top_k", "int", "Top-k sampling limit.", min = 1),
            # session.cpp:142 takes (0, 2) only.
            _opt(
                "temperature", "float", "Semantic token sampling temperature.", min = 0.01, max = 1.99
            ),
        ),
        # src/models/miotts/session.cpp:131-180.
        "miotts": (
            _opt("temperature", "float", "Sampling temperature.", min = 0.0, max = 2.0),
            _opt("top_p", "float", "Nucleus sampling.", min = 0.0, max = 1.0),
            _opt("top_k", "int", "Top-k sampling limit.", min = 0),
            _opt("repetition_penalty", "float", "Repetition penalty.", min = 1.0, max = 1.5),
            _opt("presence_penalty", "float", "Presence penalty.", min = 0.0, max = 1.0),
            _opt("frequency_penalty", "float", "Frequency penalty.", min = 0.0, max = 1.0),
            _opt("max_tokens", "int", "Most speech tokens to generate per chunk.", min = 1),
            _opt("do_sample", "bool", "Sample tokens; off decodes greedily."),
        ),
    }
)

# Adjustments per family, applied to whichever option list the model has.
OPTION_RULES: Mapping[str, tuple[OptionRule, ...]] = MappingProxyType(
    {
        "ace_step": (
            OptionRule(
                extra = (
                    _opt("lm_temperature", "float", "Planner LM sampling temperature.", min = 0.0),
                    _opt("lm_cfg_scale", "float", "Planner LM guidance scale.", min = 0.0),
                    _opt("lm_top_p", "float", "Planner LM nucleus sampling.", min = 0.0, max = 1.0),
                    _opt("lm_top_k", "int", "Planner LM top-k (0 = off).", min = 0),
                ),
                defaults = {
                    "lm_temperature": 0.85,
                    "lm_cfg_scale": 2.0,
                    "lm_top_p": 0.9,
                    "lm_top_k": 0,
                },
                sources = {
                    "lm_temperature": "audio.cpp src/models/ace_step/types.h:31-42",
                    "lm_cfg_scale": "audio.cpp src/models/ace_step/types.h:31-42",
                    "lm_top_p": "audio.cpp src/models/ace_step/types.h:31-42",
                    "lm_top_k": "audio.cpp src/models/ace_step/types.h:31-42",
                },
            ),
            # Turbo (anything but base): the runtime caps steps at 8 and skips guidance.
            OptionRule(
                match = r"^(?!.*(^|[\s/])base(/|\s|$))",
                defaults = {"num_inference_steps": 8, "shift": 3.0},
                bounds = {"num_inference_steps": (1, 8)},
                hidden = {"*": frozenset({"guidance_scale"})},
                sources = {
                    "num_inference_steps": "model_params.json ace_step; audio.cpp src/models/ace_step/session.cpp:79-80",
                    "shift": "model_params.json ace_step; https://github.com/ace-step/ACE-Step-1.5 docs/en/INFERENCE.md",
                },
            ),
            # Base takes 1-200 steps, 32-64 recommended, with guidance on.
            OptionRule(
                match = r"(^|[\s/])base(/|\s|$)",
                defaults = {"num_inference_steps": 32, "guidance_scale": 7.0, "shift": 1.0},
                bounds = {"num_inference_steps": (1, 200)},
                sources = {
                    "num_inference_steps": "https://github.com/ace-step/ACE-Step-1.5 docs/en/INFERENCE.md",
                    "guidance_scale": "https://github.com/ace-step/ACE-Step-1.5 docs/en/INFERENCE.md",
                    "shift": "audio.cpp src/models/ace_step/types.h",
                },
            ),
        ),
        # Pocket TTS: the runtime picks temperature per package (0.3 english, else 0.7).
        "pocket_tts": (
            OptionRule(
                defaults = {
                    "temperature": None,
                    "eos_threshold": -4.0,
                    "frames_after_eos": -1,
                    "noise_clamp": None,
                },
                sources = {
                    "eos_threshold": "audio.cpp include/engine/models/pocket_tts/types.h:55",
                    "frames_after_eos": "model_params.json pocket_tts",
                },
            ),
        ),
        "moss_tts_nano": (
            OptionRule(
                defaults = {
                    "do_sample": True,
                    "temperature": 1.7,
                    "top_p": 0.8,
                    "top_k": 25,
                    "repetition_penalty": 1.0,
                    "text_temperature": 1.5,
                    "max_tokens": None,
                },
                sources = {
                    "do_sample": "model_params.json moss_tts_nano",
                    "temperature": "model_params.json moss_tts_nano",
                    "top_p": "model_params.json moss_tts_nano",
                    "top_k": "model_params.json moss_tts_nano",
                    "repetition_penalty": "model_params.json moss_tts_nano",
                    "text_temperature": "audio.cpp include/engine/models/moss/moss_tts_nano/types.h:13",
                },
            ),
        ),
        # Supertonic: chunk size is 300 characters, 120 for Korean and Japanese.
        "supertonic": (
            OptionRule(
                defaults = {"speed": 1.05, "num_inference_steps": 8, "text_chunk_size": None},
                sources = {
                    "speed": "model_params.json supertonic (speaking_rate); audio.cpp include/engine/models/supertonic/session.h:20",
                    "num_inference_steps": "model_params.json supertonic",
                },
            ),
        ),
        "chatterbox_turbo": (
            OptionRule(
                defaults = {
                    "temperature": 0.8,
                    "top_p": 0.95,
                    "top_k": 1000,
                    "repetition_penalty": 1.2,
                    "text_chunk_size": 128,
                },
                sources = {
                    "temperature": "audio.cpp include/engine/community_models/chatterbox_turbo/tts.h:18",
                    "top_p": "audio.cpp include/engine/community_models/chatterbox_turbo/tts.h:19",
                    "top_k": "audio.cpp include/engine/community_models/chatterbox_turbo/tts.h:20",
                    "repetition_penalty": "audio.cpp include/engine/community_models/chatterbox_turbo/tts.h:21",
                    "text_chunk_size": "audio.cpp src/community_models/chatterbox_turbo/session.cpp:17",
                },
            ),
        ),
        "voxcpm2": (
            OptionRule(
                defaults = {
                    "num_inference_steps": 10,
                    "guidance_scale": 2.0,
                    "min_tokens": 2,
                    "max_tokens": None,
                    "retry_badcase": True,
                    "retry_badcase_max_times": 3,
                    "retry_badcase_ratio_threshold": 6.0,
                    "text_chunk_mode": "tag_aware",
                },
                sources = {
                    "num_inference_steps": "model_params.json voxcpm2",
                    "guidance_scale": "model_params.json voxcpm2",
                    "min_tokens": "model_params.json voxcpm2",
                    "retry_badcase": "model_params.json voxcpm2",
                    "retry_badcase_max_times": "audio.cpp include/engine/models/voxcpm2/types.h:22",
                    "retry_badcase_ratio_threshold": "audio.cpp include/engine/models/voxcpm2/types.h:23",
                    "text_chunk_mode": "model_params.json voxcpm2",
                },
            ),
        ),
        # Qwen3-TTS: max_tokens comes from the package's generation_config.json.
        "qwen3_tts": (
            OptionRule(
                defaults = {
                    "temperature": 0.9,
                    "top_k": 50,
                    "top_p": 1.0,
                    "repetition_penalty": 1.05,
                    "do_sample": True,
                    "subtalker_do_sample": True,
                    "subtalker_temperature": 0.9,
                    "subtalker_top_k": 50,
                    "subtalker_top_p": 1.0,
                    "max_tokens": None,
                    "text_chunk_size": 8192,
                },
                sources = {
                    "temperature": "model_params.json qwen3_tts",
                    "top_k": "model_params.json qwen3_tts",
                    "top_p": "model_params.json qwen3_tts",
                    "repetition_penalty": "model_params.json qwen3_tts",
                    "do_sample": "model_params.json qwen3_tts",
                    "subtalker_do_sample": "audio.cpp include/engine/models/qwen3_tts/types.h:26",
                    "subtalker_temperature": "audio.cpp include/engine/models/qwen3_tts/types.h:31",
                    "subtalker_top_k": "audio.cpp include/engine/models/qwen3_tts/types.h:32",
                    "subtalker_top_p": "audio.cpp include/engine/models/qwen3_tts/types.h:33",
                    "text_chunk_size": "audio.cpp src/models/qwen3_tts/session.cpp:20",
                },
            ),
        ),
        # Chatterbox converts in a separate vc session that reads none of the clone sampling options.
        "chatterbox": (
            OptionRule(
                defaults = {
                    "temperature": 0.8,
                    "repetition_penalty": 1.2,
                    "min_p": 0.05,
                    "top_p": 1.0,
                },
                hidden = {
                    "convert": frozenset({"temperature", "repetition_penalty", "min_p", "top_p"})
                },
                sources = {
                    "temperature": "model_params.json chatterbox",
                    "repetition_penalty": "model_params.json chatterbox",
                    "min_p": "audio.cpp include/engine/models/chatterbox/tts.h:24",
                    "top_p": "audio.cpp include/engine/models/chatterbox/tts.h:25",
                },
            ),
        ),
        "index_tts2": (
            OptionRule(
                defaults = {
                    "use_random_emotion": False,
                    "duration_factor": 1.0,
                    "interval_silence_ms": 200,
                    "num_beams": 3,
                    "max_tokens": 1500,
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "top_k": 30,
                    "repetition_penalty": 10.0,
                    "length_penalty": 0.0,
                    "do_sample": True,
                },
                sources = {
                    "use_random_emotion": "model_params.json index_tts2",
                    "duration_factor": "model_params.json index_tts2",
                    "interval_silence_ms": "model_params.json index_tts2",
                    "num_beams": "audio.cpp include/engine/models/index_tts2/types.h:143",
                    "max_tokens": "audio.cpp include/engine/models/index_tts2/types.h:145",
                    "temperature": "audio.cpp include/engine/models/index_tts2/types.h:141",
                    "top_p": "audio.cpp include/engine/models/index_tts2/types.h:139",
                    "top_k": "audio.cpp include/engine/models/index_tts2/types.h:140",
                    "repetition_penalty": "audio.cpp include/engine/models/index_tts2/types.h:144",
                    "length_penalty": "audio.cpp include/engine/models/index_tts2/types.h:142",
                    "do_sample": "audio.cpp include/engine/models/index_tts2/types.h:138",
                },
            ),
        ),
        # CosyVoice3 sizes both token limits from the text (20x and 2x) when they are unset.
        "cosyvoice3": (OptionRule(defaults = {"max_tokens": None, "min_tokens": None}),),
        # DotTTS-Edit: these only mean something on Edit, which claims them.
        "dots_tts": (
            OptionRule(
                match = r"(^|[-_ /.])edit([-_ ./]|$)",
                hidden = {
                    workflow: frozenset(
                        {"template_name", "source_text", "target_text", "use_xvector"}
                    )
                    for workflow in ("speak", "clone")
                },
            ),
        ),
        # Vevo2 chunks source audio only, which a Clone never has.
        "vevo2": (
            OptionRule(
                defaults = {
                    "temperature": 0.7,
                    "top_k": 20,
                    "top_p": 0.8,
                    "repetition_penalty": 1.1,
                    "max_tokens": 500,
                    "num_inference_steps": 32,
                    "audio_chunk_duration_sec": 0.0,
                    "cross_fade_duration_sec": 1.0,
                },
                hidden = {
                    "clone": frozenset({"audio_chunk_duration_sec", "cross_fade_duration_sec"})
                },
                sources = {
                    "temperature": "model_params.json vevo2",
                    "top_k": "model_params.json vevo2",
                    "top_p": "model_params.json vevo2",
                    "repetition_penalty": "audio.cpp include/engine/models/vevo2/types.h:36",
                    "max_tokens": "audio.cpp include/engine/models/vevo2/types.h:37",
                    "num_inference_steps": "model_params.json vevo2",
                    "audio_chunk_duration_sec": "model_params.json vevo2",
                    "cross_fade_duration_sec": "model_params.json vevo2",
                },
            ),
        ),
        # FireRedAudio reads these only for ASR (session.cpp:140-185); Clone has one template.
        "firered_audio": (
            OptionRule(
                hidden = {
                    "*": frozenset(
                        {"max_new_tokens", "enable_thinking", "top_k", "top_p", "temperature"}
                    ),
                    "clone": frozenset({"template_name"}),
                },
            ),
        ),
        # Base ignores template_name; on Instruct every other template breaks a Clone.
        "fireredtts3": (OptionRule(hidden = {"*": frozenset({"template_name"})}),),
        # Offered on Speak only, so there is no reference to trim.
        "dramabox": (OptionRule(hidden = {"*": frozenset({"reference_duration_sec"})}),),
        # Irodori: no_ref follows the workflow (a Clone sends a reference); the runtime predicts the
        # duration, keeps the three guidance scales unless overridden and sizes chunks from the model.
        "irodori_tts": (
            OptionRule(
                defaults = {"duration_sec": None, "guidance_scale": None, "text_chunk_size": None},
                hidden = {"*": frozenset({"no_ref"})},
            ),
            # The 500M v3 base checkpoint is not caption-conditioned.
            OptionRule(
                match = r"500m",
                hidden = {"*": frozenset({"caption", "caption_guidance_scale"})},
            ),
        ),
        "higgs_audio_tts": (
            OptionRule(
                defaults = {
                    "temperature": 0.8,
                    "top_p": 0.8,
                    "top_k": 30,
                    "max_tokens": 2048,
                    "text_chunk_size": 1024,
                },
                sources = {
                    "temperature": "audio.cpp src/models/higgs_audio_tts/session.cpp:103",
                    "top_p": "audio.cpp src/models/higgs_audio_tts/session.cpp:104",
                    "top_k": "audio.cpp src/models/higgs_audio_tts/session.cpp:105",
                    "max_tokens": "audio.cpp src/models/higgs_audio_tts/session.cpp:102",
                    "text_chunk_size": "audio.cpp src/models/higgs_audio_tts/session.cpp:22",
                },
            ),
        ),
        "moss_tts_local": (
            OptionRule(
                defaults = {
                    "do_sample": True,
                    "temperature": 1.7,
                    "top_p": 0.8,
                    "top_k": 25,
                    "repetition_penalty": 1.0,
                    "max_tokens": 4096,
                    "text_temperature": 1.0,
                },
                sources = {
                    "do_sample": "model_params.json moss_tts_local",
                    "temperature": "model_params.json moss_tts_local",
                    "top_p": "model_params.json moss_tts_local",
                    "top_k": "model_params.json moss_tts_local",
                    "repetition_penalty": "model_params.json moss_tts_local",
                    "max_tokens": "audio.cpp include/engine/models/moss/moss_tts_local/generator.h:24",
                    "text_temperature": "audio.cpp include/engine/models/moss/moss_tts_local/generator.h:30",
                },
            ),
        ),
        "moss_voicegen": (
            OptionRule(
                defaults = {"temperature": 1.5, "top_p": 0.6, "top_k": 50, "repetition_penalty": 1.1},
                sources = {
                    "temperature": "model_params.json moss_voicegen",
                    "top_p": "model_params.json moss_voicegen",
                    "top_k": "model_params.json moss_voicegen",
                    "repetition_penalty": "model_params.json moss_voicegen",
                },
            ),
        ),
        # OmniVoice: an unset duration follows speed.
        "omnivoice": (
            OptionRule(
                defaults = {
                    "num_inference_steps": 32,
                    "guidance_scale": 2.0,
                    "speed": 1.0,
                    "duration": None,
                    "t_shift": 0.1,
                    "denoise": True,
                    "preprocess_prompt": True,
                    "postprocess_output": True,
                    "layer_penalty_factor": 5.0,
                    "position_temperature": 5.0,
                    "class_temperature": 0.0,
                    "audio_chunk_duration": 15.0,
                    "audio_chunk_threshold": 30.0,
                },
                sources = {
                    "num_inference_steps": "model_params.json omnivoice",
                    "guidance_scale": "model_params.json omnivoice",
                    "speed": "model_params.json omnivoice",
                    "t_shift": "audio.cpp include/engine/models/omnivoice/types.h:25",
                    "denoise": "audio.cpp include/engine/models/omnivoice/types.h:26",
                    "preprocess_prompt": "audio.cpp include/engine/models/omnivoice/types.h:31",
                    "postprocess_output": "audio.cpp include/engine/models/omnivoice/types.h:27",
                    "layer_penalty_factor": "audio.cpp include/engine/models/omnivoice/types.h:28",
                    "position_temperature": "audio.cpp include/engine/models/omnivoice/types.h:29",
                    "class_temperature": "audio.cpp include/engine/models/omnivoice/types.h:30",
                    "audio_chunk_duration": "audio.cpp include/engine/models/omnivoice/types.h:32",
                    "audio_chunk_threshold": "audio.cpp include/engine/models/omnivoice/types.h:33",
                },
            ),
        ),
        # VibeVoice: unset steps fall back to the package config (20); the web UI runs 10.
        "vibevoice": (
            OptionRule(
                defaults = {
                    "num_inference_steps": 10,
                    "guidance_scale": 1.3,
                    "max_length_times": 2.0,
                    "do_sample": False,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 50,
                },
                sources = {
                    "num_inference_steps": "model_params.json vibevoice",
                    "guidance_scale": "model_params.json vibevoice",
                    "max_length_times": "model_params.json vibevoice",
                    "do_sample": "model_params.json vibevoice",
                    "temperature": "model_params.json vibevoice",
                    "top_p": "model_params.json vibevoice",
                    "top_k": "audio.cpp include/engine/models/vibevoice/types.h:20",
                },
            ),
        ),
        # VoxCPM1: the embedded spec's defaults are older than the runtime's; prompt_text is the
        # Clone transcript under another name.
        "voxcpm1": (
            OptionRule(
                defaults = {
                    "num_inference_steps": 10,
                    "guidance_scale": 2.0,
                    "min_tokens": 2,
                    "max_tokens": 4096,
                    "retry_badcase": True,
                    "retry_badcase_max_times": 3,
                    "retry_badcase_ratio_threshold": 6.0,
                },
                hidden = {"*": frozenset({"prompt_text"})},
                sources = {
                    "num_inference_steps": "model_params.json voxcpm1",
                    "guidance_scale": "model_params.json voxcpm1",
                    "min_tokens": "model_params.json voxcpm1",
                    "max_tokens": "audio.cpp include/engine/community_models/voxcpm1/types.h:14",
                    "retry_badcase": "model_params.json voxcpm1",
                    "retry_badcase_max_times": "audio.cpp include/engine/community_models/voxcpm1/types.h:18",
                    "retry_badcase_ratio_threshold": "audio.cpp include/engine/community_models/voxcpm1/types.h:19",
                },
            ),
        ),
        "fish_audio": (
            OptionRule(
                defaults = {
                    "max_tokens": 1024,
                    "text_chunk_size": 200,
                    "text_chunk_mode": "word_budget",
                    "top_p": 0.8,
                    "top_k": 30,
                    "temperature": 0.8,
                },
                sources = {
                    name: "audio.cpp model_specs/fish_audio.json"
                    for name in (
                        "max_tokens",
                        "text_chunk_size",
                        "text_chunk_mode",
                        "top_p",
                        "top_k",
                        "temperature",
                    )
                },
            ),
        ),
        "miotts": (
            OptionRule(
                defaults = {
                    "temperature": 0.8,
                    "top_p": 1.0,
                    "top_k": 50,
                    "repetition_penalty": 1.0,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "max_tokens": 700,
                    "do_sample": True,
                },
                sources = {
                    "temperature": "model_params.json miotts",
                    "top_p": "model_params.json miotts",
                    "top_k": "model_params.json miotts",
                    "repetition_penalty": "model_params.json miotts",
                    "presence_penalty": "audio.cpp include/engine/models/miotts/assets.h:37",
                    "frequency_penalty": "audio.cpp include/engine/models/miotts/assets.h:38",
                    "max_tokens": "audio.cpp src/models/miotts/assets.cpp:44",
                    "do_sample": "audio.cpp include/engine/models/miotts/assets.h:32",
                },
            ),
        ),
        # YuE2: guidance depends on cot (1.0 for melody/full, 1.01 for off), so it stays unset.
        "yue2": (
            OptionRule(
                extra = (
                    _opt(
                        "semantic_repetition_penalty",
                        "float",
                        "Semantic codec repetition penalty.",
                        min = 0.1,
                        max = 2.0,
                    ),
                    _opt(
                        "abc_temperature",
                        "float",
                        "ABC planning sampling temperature.",
                        min = 0.0,
                        max = 2.0,
                    ),
                    _opt("abc_top_p", "float", "ABC planning nucleus sampling.", min = 0.01, max = 1.0),
                    _opt("abc_top_k", "int", "ABC planning top-k sampling limit.", min = 1),
                    _opt(
                        "abc_repetition_penalty",
                        "float",
                        "ABC planning repetition penalty.",
                        min = 0.1,
                        max = 2.0,
                    ),
                ),
                defaults = {
                    "guidance_scale": None,
                    "semantic_temperature": 1.0,
                    "semantic_top_p": 0.95,
                    "semantic_top_k": 100,
                    "semantic_repetition_penalty": 1.2,
                    "abc_temperature": 0.7,
                    "abc_top_p": 0.9,
                    "abc_top_k": 30,
                    "abc_repetition_penalty": 1.005,
                },
                sources = {
                    name: "model_params.json yue2"
                    for name in (
                        "semantic_temperature",
                        "semantic_top_p",
                        "semantic_top_k",
                        "semantic_repetition_penalty",
                        "abc_temperature",
                        "abc_top_p",
                        "abc_top_k",
                        "abc_repetition_penalty",
                    )
                },
            ),
        ),
        # HeartMuLa: infinite mode and its two knobs mean nothing under the music form's length cap.
        "heartmula": (
            OptionRule(
                hidden = {
                    "*": frozenset(
                        {"infinite_mode", "text_chunk_size", "infinite_chunk_audio_duration_ms"}
                    )
                },
            ),
        ),
        # ControlFoley: masking applies to a video clip, which the music form never sends.
        "controlfoley": (OptionRule(hidden = {"*": frozenset({"mask_away_clip"})}),),
    }
)


def rules_for(family: str, text: str) -> tuple[OptionRule, ...]:
    """The rules of ``family`` that apply to a model described by ``text``."""
    return tuple(
        rule
        for rule in OPTION_RULES.get(family, ())
        if rule.match is None or re.search(rule.match, text, re.IGNORECASE)
    )
