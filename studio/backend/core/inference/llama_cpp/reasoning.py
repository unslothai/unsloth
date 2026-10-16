# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reasoning flags and effort levels."""

import os
import re
import threading

from typing import Optional

from .log import logger

# Canonical reasoning_effort levels, weakest -> strongest. Used to read the
# discrete set a template branches on (e.g. GLM-5.2 uses 'high' | 'max', Inkling
# uses the full 'none'..'max' ladder) so we only ever offer levels the template
# actually understands. Must stay in sync with REASONING_EFFORT_SCALE in
# studio/frontend/src/features/chat/provider-capabilities.ts.
_REASONING_EFFORT_SCALE = ("none", "minimal", "low", "medium", "high", "xhigh", "max")

# Match a Qwen3.8 path/repo segment without treating future names such as
# Qwen3.80 as part of the family. Official ids continue after the version with
# a dash (for example Qwen3.8-27B-GGUF).
_QWEN38_MODEL_RE = re.compile(r"(?:^|[^a-z0-9])qwen3\.8(?:$|[-_/\\])", re.IGNORECASE)


def _extract_reasoning_effort_levels(chat_template: str) -> list:
    """Return the reasoning_effort levels a template references, in canonical
    (weakest -> strongest) order.

    Looks for the quoted literals (e.g. ``'high'`` / ``"max"``) the template
    compares ``reasoning_effort`` against, so we surface exactly the levels it
    branches on and nothing else.
    """
    return [
        level
        for level in _REASONING_EFFORT_SCALE
        if f"'{level}'" in chat_template or f'"{level}"' in chat_template
    ]


# Log each substitution once per (levels, requested) pair.
_REASONING_EFFORT_CLAMPS_LOGGED: set = set()


def _clamp_reasoning_effort_to_levels(effort: Optional[str], levels) -> Optional[str]:
    """Match the UI clamp: prefer the nearest lower level, then the weakest.

    Use max for an unavailable xhigh if offered. Return None for an absent or
    unknown effort or an empty ladder. See clampReasoningEffortToLevels in
    studio/frontend/src/features/chat/provider-capabilities.ts.
    """
    if effort not in _REASONING_EFFORT_SCALE or not levels:
        return None
    if effort == "xhigh" and "xhigh" not in levels and "max" in levels:
        effort = "max"
    if effort in levels:
        return effort
    offered = [level for level in _REASONING_EFFORT_SCALE if level in levels and level != "none"]
    if not offered:
        return None
    rank = _REASONING_EFFORT_SCALE.index(effort)
    below = [level for level in offered if _REASONING_EFFORT_SCALE.index(level) < rank]
    return below[-1] if below else offered[0]


# Headers already described in the log, keyed on identity rather than name so a rebuilt
# GGUF at the same path is described again. Bounded; a dropped entry only re-logs once.
_GGUF_METADATA_LOGGED: "dict[tuple, bool]" = {}
_GGUF_METADATA_LOGGED_MAX = 32
_GGUF_METADATA_LOGGED_LOCK = threading.Lock()


def _note_gguf_metadata_read(gguf_path: str) -> bool:
    """True the first time this exact file is read, False for repeats.

    Falls open when the file cannot be stat'd: an unidentifiable read is better logged
    than silently swallowed.
    """
    try:
        st = os.stat(gguf_path)
        key = (os.path.abspath(gguf_path), st.st_size, st.st_mtime_ns)
    except Exception:
        return True
    with _GGUF_METADATA_LOGGED_LOCK:
        if key in _GGUF_METADATA_LOGGED:
            return False
        _GGUF_METADATA_LOGGED[key] = True
        while len(_GGUF_METADATA_LOGGED) > _GGUF_METADATA_LOGGED_MAX:
            # Insertion-ordered, so this drops the oldest.
            del _GGUF_METADATA_LOGGED[next(iter(_GGUF_METADATA_LOGGED))]
        return True


def detect_reasoning_flags(
    chat_template: Optional[str],
    model_identifier: Optional[str] = None,
    *,
    log_source: Optional[str] = None,
    log_level: str = "info",
) -> dict:
    """Classify a chat template's reasoning and tool-calling capabilities.

    Returns the same seven keys as the GGUF sniffer: ``supports_reasoning``,
    ``reasoning_style`` (``"enable_thinking"`` | ``"reasoning_effort"`` |
    ``"enable_thinking_effort"``), ``reasoning_always_on``,
    ``reasoning_effort_levels``, ``supports_preserve_thinking``,
    ``preserve_thinking_default``, ``supports_tools``. A falsy
    ``chat_template`` yields the all-default dict.
    Used by both the llama-server backend at load time and the
    safetensors/transformers paths in ``routes/inference`` so they agree on
    what the frontend sees.
    """
    flags = {
        "supports_reasoning": False,
        "reasoning_style": "enable_thinking",
        "reasoning_always_on": False,
        "reasoning_effort_levels": [],
        "supports_preserve_thinking": False,
        "preserve_thinking_default": False,
        "supports_tools": False,
    }
    if not chat_template:
        return flags
    tpl = chat_template
    prefix = f"{log_source}: " if log_source else ""
    # The GGUF sniffer re-derives these several times per request and the lines say the
    # same thing every time, so it asks for debug after the first read of a file.
    _log = logger.debug if log_level == "debug" else logger.info

    # 'none' is this style's disable sentinel, not an effort level: the off
    # switch is enable_thinking=false, and the Think menu already hides it.
    # Left in, it is the weakest level, so a stored effort the model does not
    # offer clamps onto it and silently disables thinking (Kimi-K3).
    effort_levels = (
        [lv for lv in _extract_reasoning_effort_levels(tpl) if lv != "none"]
        if ("reasoning_effort" in tpl and "enable_thinking" in tpl)
        else []
    )
    if effort_levels:
        # DeepSeek-V4's encoder accepts reasoning_effort {'high', 'max'} but its
        # template only branches on 'max', so the literal scan misses 'high'. Add it
        # (matched on whole repo-name segments, so 'deepseek-v40' won't false-match)
        # to expose the full high/max ladder instead of max alone.
        from core.inference.model_ids import hf_cache_repo_id

        # A snapshot path's last component is the revision, so read it as its repo id first.
        repo_name = hf_cache_repo_id(model_identifier) or model_identifier or ""
        segments = re.split(r"[-_.]", repo_name.lower().split("/")[-1])
        is_dsv4 = "deepseek4" in segments or any(
            a == "deepseek" and b == "v4" for a, b in zip(segments, segments[1:])
        )
        if is_dsv4 and "high" not in effort_levels:
            effort_levels = sorted(set(effort_levels) | {"high"}, key = _REASONING_EFFORT_SCALE.index)
        # GLM-5.2-style: an enable_thinking on/off gate PLUS a reasoning_effort
        # level among a discrete set (e.g. 'high' | 'max'). Distinct from
        # gpt-oss (reasoning_effort only, no on/off gate) and Qwen
        # (enable_thinking only). Disabling is enable_thinking=false; the levels
        # are the quoted effort literals the template actually branches on.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking_effort"
        flags["reasoning_effort_levels"] = effort_levels
        _log(
            f"{prefix}model supports reasoning "
            f"(enable_thinking + reasoning_effort: {effort_levels})"
        )
    elif "enable_thinking" in tpl:
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "enable_thinking"
        _log(f"{prefix}model supports reasoning (enable_thinking)")
    elif "reasoning_effort" in tpl:
        # gpt-oss / Harmony use reasoning_effort
        # ("low" | "medium" | "high"), not a boolean. Inkling maps a wider
        # named ladder ('none'..'max'); surface the levels the template
        # actually branches on so the Think menu offers exactly those and
        # nothing more ('none' renders as thinking off: effort 0). Guard:
        # trust the scan only when it includes the core 'low' and 'high'
        # literals, so a template that quotes e.g. 'none' for an unrelated
        # comparison keeps the default low/medium/high set.
        flags["supports_reasoning"] = True
        flags["reasoning_style"] = "reasoning_effort"
        scanned = _extract_reasoning_effort_levels(tpl)
        if "low" in scanned and "high" in scanned:
            flags["reasoning_effort_levels"] = scanned
        _log(
            f"{prefix}model supports reasoning "
            f"(reasoning_effort: {flags['reasoning_effort_levels'] or 'default levels'})"
        )
    elif "thinking" in tpl:
        # DeepSeek uses 'thinking', not 'enable_thinking'
        normalized_id = (model_identifier or "").lower()
        if "deepseek" in normalized_id:
            flags["supports_reasoning"] = True
            _log(f"{prefix}model supports reasoning (DeepSeek thinking)")

    # Hardcoded <think> tags or reasoning_content in the template mean
    # thinking is always on (no toggle).
    if not flags["supports_reasoning"]:
        if ("<think>" in tpl and "</think>" in tpl) or "reasoning_content" in tpl:
            flags["supports_reasoning"] = True
            flags["reasoning_always_on"] = True
            _log(f"{prefix}model always reasons (<think> tags in template)")

    # preserve_thinking: independent kwarg on some Qwen templates that
    # keeps historical <think> blocks in prior assistant turns.
    if "preserve_thinking" in tpl:
        flags["supports_preserve_thinking"] = True
        # Qwen3.8 deliberately changed the family default from the off behavior
        # used by Qwen3.6 and Gemma 4. Keep the product override family-scoped:
        # other templates that happen to expose the same kwarg stay off.
        flags["preserve_thinking_default"] = bool(_QWEN38_MODEL_RE.search(model_identifier or ""))
        _log(f"{prefix}model supports preserve_thinking")

    from core.inference.template_capabilities import template_supports_tools

    if isinstance(tpl, str) and template_supports_tools(tpl):
        flags["supports_tools"] = True
        _log(f"{prefix}model supports tool calling")

    return flags


# Inkling's template takes a numeric thinking-effort dial (0..0.99) and its
# float() coercion turns unrecognized named levels into 0, i.e. no thinking.
# Map OpenAI-style names to the values the model was trained on. Module-level
# so duck-typed engine stand-ins in tests do not need the attribute.
_INKLING_REASONING_EFFORT = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
    "max": 0.99,
}


# Muse Spark's template accepts only low/medium/high/xhigh and raise_exception()s on
# anything else, so a "none" or "max" request would fail the render. It cannot turn
# reasoning off: "none" and "minimal" take the smallest budget, "max" the largest.
_MUSE_SPARK_REASONING_EFFORT = {
    "none": "low",
    "minimal": "low",
    "max": "xhigh",
}


def _coerce_reasoning_effort(architecture, kwargs: dict) -> dict:
    if architecture == "inkling":
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str):
            mapped = _INKLING_REASONING_EFFORT.get(effort.strip().lower())
            if mapped is not None:
                kwargs["reasoning_effort"] = mapped
    elif architecture == "muse-spark":
        effort = kwargs.get("reasoning_effort")
        if isinstance(effort, str):
            mapped = _MUSE_SPARK_REASONING_EFFORT.get(effort.strip().lower())
            if mapped is not None:
                kwargs["reasoning_effort"] = mapped
    return kwargs
