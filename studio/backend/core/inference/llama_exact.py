# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ask llama-server for output that does not depend on the neighbours it decodes beside.

A llama-server built with unslothai/llama.cpp#194 reads ``LLAMA_EXACT_CONCURRENCY`` from its OWN
environment, and a sequence's generated tokens are then byte-identical however many chats share its
cache. There is no flag for it: the running server reports the mode as ``exact_concurrency`` in
``/props`` (unslothai/llama.cpp#197), and that field is the only evidence Studio accepts, a build
that ignores the variable starting all the same and saying nothing.

Three values, because two cannot express "I would like this" against "I require this": ``off`` (the
default, the mode costing about 9 per cent of solo decode), ``auto`` (relaunch once without it and
report ``unavailable``), and ``on`` (a refusal is a failed load, a caller that asked for
byte-identical output being unable to notice a downgrade).
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence


# Studio's own switch, overriding the request field and the stored setting.
EXACT_ENV = "UNSLOTH_LLAMA_EXACT_CONCURRENCY"
EXACT_AUTO = "auto"
EXACT_OFF = "off"
EXACT_ON = "on"
EXACT_SETTINGS = (EXACT_AUTO, EXACT_OFF, EXACT_ON)
DEFAULT_EXACT_SETTING = EXACT_OFF

# What the CHILD reads (unslothai/llama.cpp#194). Setting it on Studio is the workaround this replaces.
CHILD_ENV = "LLAMA_EXACT_CONCURRENCY"

# What the finished load reports. `unavailable` carries what a boolean cannot: the mode was asked
# for, and the chat runs anyway without it (the server refused, or Studio withheld it).
EXACT_STATE_ON = "on"
EXACT_STATE_OFF = "off"
EXACT_STATE_UNAVAILABLE = "unavailable"
EXACT_STATES = (EXACT_STATE_ON, EXACT_STATE_OFF, EXACT_STATE_UNAVAILABLE)


def normalize_setting(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in EXACT_SETTINGS else None


def _truthy(value: Optional[str]) -> bool:
    """llama.cpp reads this variable with ``atoi() != 0``, so match that, plus the spellings a
    person types where a C program would have read zero."""
    if value is None:
        return False
    text = value.strip().lower()
    if text in {"true", "yes", "on"}:
        return True
    try:
        return int(text, 10) != 0
    except ValueError:
        return False


def child_flag_set(environ: Mapping[str, str]) -> bool:
    return _truthy(environ.get(CHILD_ENV))


def child_flag_inherited(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the Studio process itself was started with ``LLAMA_EXACT_CONCURRENCY``. Read as the
    DEFAULT setting: ``off`` would silently disable the mode for the people who turned it on."""
    return child_flag_set(os.environ if environ is None else environ)


def exact_setting_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    source = os.environ if environ is None else environ
    return normalize_setting(source.get(EXACT_ENV))


# Sentinel for "read the store yourself", distinct from a stored None.
_MISSING = object()


def stored_exact_setting() -> Optional[str]:
    """The persisted setting, or None when nothing is stored or the store is unreadable.
    Imported inside the function: a load must never fail on an unreadable settings row."""
    try:
        from utils.exact_concurrency_settings import get_exact_concurrency
    except Exception:
        return None
    try:
        return normalize_setting(get_exact_concurrency())
    except Exception:
        return None


def resolve_exact_setting(
    requested: Any = None,
    *,
    stored: Any = _MISSING,
    environ: Optional[Mapping[str, str]] = None,
) -> str:
    """Which of auto/off/on this load runs under. Precedence, highest first:
    ``UNSLOTH_LLAMA_EXACT_CONCURRENCY``, the load request's field, the persisted setting, an
    inherited ``LLAMA_EXACT_CONCURRENCY`` (read as ``on``), then ``off``."""
    from_env = exact_setting_env(environ)
    if from_env is not None:
        return from_env
    from_request = normalize_setting(requested)
    if from_request is not None:
        return from_request
    from_store = normalize_setting(stored_exact_setting() if stored is _MISSING else stored)
    if from_store is not None:
        return from_store
    return EXACT_ON if child_flag_inherited(environ) else DEFAULT_EXACT_SETTING


def wants_exact(setting: Any) -> bool:
    return normalize_setting(setting) in (EXACT_AUTO, EXACT_ON)


# The refusals the server names, each printed on the way to exiting non-zero. Matching on the
# mode's own name keeps a build that adds a new refusal covered.
_REFUSAL_MARKERS = (
    # `throw std::runtime_error("exact concurrency: ...")`, from the KV cache and the graph.
    "exact concurrency:",
    # `LLAMA_EXACT_CONCURRENCY is set but ...`, and the column-bound refusal in
    # common_exact_concurrency_init.
    "llama_exact_concurrency",
)


def is_exact_refusal(text: Optional[str]) -> bool:
    """Whether this child output says the mode itself is why the server did not start. A build
    that predates #194 ignores the variable and starts normally, so this never fires there."""
    if not text:
        return False
    return any(marker in text.lower() for marker in _REFUSAL_MARKERS)


# What the mode cannot live beside, checked in llama.cpp itself so Studio can warn about its own
# launch line: `--cache-reuse` and `--context-shift` move positions while a cell's offset in its
# 256-cell page IS its position mod 256; `--no-kv-offload` drops a layer, and no flash attention
# leaves V transposed.
_CACHE_TYPE_FLAGS = ("--cache-type-k", "--cache-type-v", "-ctk", "-ctv")
_BARE_CONTRADICTIONS = (
    "--context-shift",
    "--no-kv-offload",
    "-nkvo",
    "--no-flash-attn",
    "--no-kv-unified",
    "-no-kvu",
)
# A later spelling of the same option replaces an earlier one, as llama-server applies argv,
# so `--flash-attn off --flash-attn on` runs with flash attention and is no contradiction.
_OPTION_FAMILY = {
    "-ctk": "--cache-type-k",
    "-ctv": "--cache-type-v",
    "-fa": "--flash-attn",
    "--no-flash-attn": "--flash-attn",
    "--no-context-shift": "--context-shift",
    "-nkvo": "--no-kv-offload",
    "-kvo": "--no-kv-offload",
    "--kv-offload": "--no-kv-offload",
    "-kvu": "--kv-unified",
    "-no-kvu": "--kv-unified",
    "--no-kv-unified": "--kv-unified",
}


def _flag_name(token: str) -> str:
    return token.split("=", 1)[0]


def _flag_value(token: str, following: Optional[str]) -> Optional[str]:
    if "=" in token:
        return token.split("=", 1)[1]
    return following


def contradicting_args(args: Optional[Sequence[str]]) -> list[str]:
    """The tokens in ``args`` that exact mode cannot run with, in the order they appear. Flag
    names, not values. A zero ``--cache-reuse 0`` and an ``f16`` cache type are the flag spelled
    as the default, not contradictions, and an option's LAST occurrence decides for it."""
    tokens = [str(a) for a in (args or ())]
    # family -> (name as last spelled, contradicts)
    final: dict[str, tuple[str, bool]] = {}
    for index, token in enumerate(tokens):
        name = _flag_name(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if name == "--cache-reuse":
            contradicts = _flag_value(token, following) not in ("0", None)
        elif name in _CACHE_TYPE_FLAGS:
            value = (_flag_value(token, following) or "").strip().lower()
            contradicts = bool(value) and value not in ("f16", "fp16", "float16")
        elif name in ("--flash-attn", "-fa"):
            value = (_flag_value(token, following) or "").strip().lower()
            contradicts = value in ("off", "0", "false", "disabled")
        elif name in _BARE_CONTRADICTIONS:
            contradicts = True
        elif name in ("--no-context-shift", "-kvo", "--kv-offload", "-kvu", "--kv-unified"):
            contradicts = False
        else:
            continue
        final[_OPTION_FAMILY.get(name, name)] = (name, contradicts)
    return [name for name, contradicts in final.values() if contradicts]


def apply_child_env(env: dict, *, on: bool) -> bool:
    """Put the child's variable in ``env`` (or take it out). True when ``env`` changed. Taking it
    OUT matters: ``env`` copies Studio's own, so an inherited value would outvote a load's ``off``."""
    if on:
        if env.get(CHILD_ENV) == "1":
            return False
        env[CHILD_ENV] = "1"
        return True
    return env.pop(CHILD_ENV, None) is not None
