# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Ask llama-server for output that does not depend on the neighbours it decodes beside.

A llama-server built with unslothai/llama.cpp#194 reads ``LLAMA_EXACT_CONCURRENCY`` from
its OWN environment. With it set, a sequence's generated tokens are byte-identical whether
it decodes alone, next to three other chats in one unified KV cache, or across a park and
restore that moved its cells: the CUDA dispatcher stops changing algorithm with batch
width, and a sequence's cells are placed in 256-cell pages so its attention rounds the same
way wherever its neighbours sit.

There is no command-line flag, nothing in ``--help``, and nothing in ``/props``: the knob
is an environment variable and the server does not advertise it. So a user who wants it
today has to set a llama.cpp variable on the Studio PROCESS, which is neither discoverable
nor per-load. This module is the Studio-side switch instead, and the launch is the only
probe there is -- a child started with the variable that comes up healthy has it, and one
that dies naming the mode does not.

Three values, because two cannot express the difference between "I would like this" and "I
require this":

``off``   the default. The mode costs about 9 per cent of solo decode on a dense model and
          more on a mixture of experts, and buys nothing for a single chat that never
          shares its cache, so nobody pays for it without asking.
``auto``  set the variable; if the load fails naming the mode, relaunch once without it and
          report ``unavailable``. For a user who wants determinism where the build offers
          it and a working server everywhere else.
``on``    require it. A build or a configuration that refuses is a failed load with the
          server's own message, not a quiet downgrade, because a caller that asked for
          byte-identical output and silently did not get it has no way to notice.

``UNSLOTH_LLAMA_EXACT_CONCURRENCY`` overrides the persisted setting and the load request,
the same shape as ``UNSLOTH_LLAMA_PREEMPT_MODE`` next door in ``llama_preemption``.
"""

from __future__ import annotations

import os
from typing import Any, Iterable, Mapping, Optional, Sequence


# Studio's own switch. Overrides the request field and the stored setting, so an operator
# can pin a machine without going through the GUI.
EXACT_ENV = "UNSLOTH_LLAMA_EXACT_CONCURRENCY"
EXACT_AUTO = "auto"
EXACT_OFF = "off"
EXACT_ON = "on"
EXACT_SETTINGS = (EXACT_AUTO, EXACT_OFF, EXACT_ON)
DEFAULT_EXACT_SETTING = EXACT_OFF

# What the CHILD reads (unslothai/llama.cpp#194). Set on the llama-server environment, not
# on Studio's: setting it on Studio is the workaround this module replaces.
CHILD_ENV = "LLAMA_EXACT_CONCURRENCY"

# What the finished load reports. `unavailable` is the one that carries information a
# boolean cannot: exact mode was asked for, the server would not give it, and the chat is
# running anyway.
EXACT_STATE_ON = "on"
EXACT_STATE_OFF = "off"
EXACT_STATE_UNAVAILABLE = "unavailable"
EXACT_STATES = (EXACT_STATE_ON, EXACT_STATE_OFF, EXACT_STATE_UNAVAILABLE)


def normalize_setting(value: Any) -> Optional[str]:
    """One of auto/off/on, or None for anything else (including None and "")."""
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in EXACT_SETTINGS else None


def _truthy(value: Optional[str]) -> bool:
    """llama.cpp reads this variable with ``atoi() != 0``, so match that, and additionally
    accept the spellings a person types when a C program would have read them as zero."""
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
    """Whether this environment mapping turns the mode on for a child that reads it."""
    return _truthy(environ.get(CHILD_ENV))


def child_flag_inherited(environ: Optional[Mapping[str, str]] = None) -> bool:
    """Whether the Studio process itself was started with ``LLAMA_EXACT_CONCURRENCY``.

    This is the workaround that exists today: the variable is inherited by every child, so
    a user who set it is already running in exact mode. It is read as the DEFAULT setting
    rather than ignored, because shipping a switch whose default is ``off`` would otherwise
    silently turn the mode off for exactly the people who had gone to the trouble of
    turning it on. An explicit ``off`` from the request, the stored setting or
    ``UNSLOTH_LLAMA_EXACT_CONCURRENCY`` still wins over it.
    """
    return child_flag_set(os.environ if environ is None else environ)


def exact_setting_env(environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """``UNSLOTH_LLAMA_EXACT_CONCURRENCY``, or None when unset or misspelled."""
    source = os.environ if environ is None else environ
    return normalize_setting(source.get(EXACT_ENV))


# Sentinel for "read the store yourself", distinct from a stored None.
_MISSING = object()


def stored_exact_setting() -> Optional[str]:
    """The persisted setting, or None when nothing is stored or the store is unreadable.

    Imported inside the function, like ``apply_model_memory_policy`` does: this module is
    imported by the backend, which unit tests construct without a database, and a load must
    never fail because a settings row could not be read.
    """
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
    """Which of auto/off/on this load runs under.

    Precedence, highest first: ``UNSLOTH_LLAMA_EXACT_CONCURRENCY``, the load request's
    field, the persisted setting, an inherited ``LLAMA_EXACT_CONCURRENCY`` (read as ``on``,
    see ``child_flag_inherited``), then ``off``.

    The environment is the override rather than the fallback because it is how an operator
    pins a machine that the GUI can also write to; the stored setting is what the GUI
    writes. ``stored`` is injectable so a test does not need a database.
    """
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
    """Whether this setting asks the child for exact mode. ``auto`` and ``on`` do."""
    return normalize_setting(setting) in (EXACT_AUTO, EXACT_ON)


# The refusals the server names. Every one of these is a message llama.cpp prints on the
# way to exiting non-zero, so seeing one in a dead child's output attributes the death to
# this mode and nothing else. Matching on the mode's own name rather than on each refusal
# keeps a build that adds a new refusal covered.
_REFUSAL_MARKERS = (
    # `throw std::runtime_error("exact concurrency: ...")`, from the KV cache and the graph.
    "exact concurrency:",
    # `LLAMA_EXACT_CONCURRENCY is set but ...` / `is set, so ...`, and the column-bound
    # refusal in common_exact_concurrency_init.
    "llama_exact_concurrency",
)


def is_exact_refusal(text: Optional[str]) -> bool:
    """Whether this child output says the mode itself is why the server did not start.

    A build that predates #194 ignores the variable entirely and starts normally, so this
    never fires there: absence of the feature reads as ``on`` and is reported as such,
    which is the honest answer available without a way to ask the server what it supports.
    A build that HAS the mode and refuses the configuration names it, and that is what
    ``auto`` relaunches without.
    """
    if not text:
        return False
    return any(marker in text.lower() for marker in _REFUSAL_MARKERS)


# What the mode cannot live beside. Each is checked in llama.cpp itself and refuses the
# load there; naming them here is what lets Studio warn about its own launch line instead
# of handing the user an error from a child they did not compose.
#
# `--cache-reuse` and `--context-shift` both move a sequence's positions, and a cell's
# offset inside its 256-cell page IS its position modulo 256, so the pool cannot do either.
# `--no-kv-offload` leaves a layer's cache off the CUDA backend, which has no paged
# attention. A quantized KV cache is not F16. No flash attention leaves V transposed.
_CACHE_TYPE_FLAGS = ("--cache-type-k", "--cache-type-v", "-ctk", "-ctv")
_BARE_CONTRADICTIONS = (
    "--context-shift",
    "--no-kv-offload",
    "-nkvo",
    "--no-flash-attn",
)


def _flag_name(token: str) -> str:
    return token.split("=", 1)[0]


def _flag_value(token: str, following: Optional[str]) -> Optional[str]:
    if "=" in token:
        return token.split("=", 1)[1]
    return following


def contradicting_args(args: Optional[Sequence[str]]) -> list[str]:
    """The tokens in ``args`` that exact mode cannot run with, in the order they appear.

    Returns the flag names, not the values, because that is what a warning has to name.
    A zero ``--cache-reuse 0`` and an ``f16`` cache type are not contradictions: they are
    the flag spelled as the default, which is what the mode already needs.
    """
    tokens = [str(a) for a in (args or ())]
    found: list[str] = []
    for index, token in enumerate(tokens):
        name = _flag_name(token)
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        if name == "--cache-reuse":
            if _flag_value(token, following) not in ("0", None):
                found.append(name)
            continue
        if name in _CACHE_TYPE_FLAGS:
            value = (_flag_value(token, following) or "").strip().lower()
            if value and value not in ("f16", "fp16", "float16"):
                found.append(name)
            continue
        if name in ("--flash-attn", "-fa"):
            value = (_flag_value(token, following) or "").strip().lower()
            if value in ("off", "0", "false", "disabled"):
                found.append(name)
            continue
        if name in _BARE_CONTRADICTIONS:
            found.append(name)
    return found


def apply_child_env(env: dict, *, on: bool) -> bool:
    """Put the child's variable in ``env`` (or take it out). True when ``env`` changed.

    Taking it OUT is not redundant. ``env`` starts as a copy of Studio's own environment,
    so a user who set ``LLAMA_EXACT_CONCURRENCY`` on Studio would otherwise keep getting
    exact mode from a load that explicitly resolved to ``off`` -- and an explicit ``off``
    is the one answer that has to be obeyed exactly, since it is what a user picks when the
    9 per cent is the thing they are trying to get rid of.
    """
    if on:
        if env.get(CHILD_ENV) == "1":
            return False
        env[CHILD_ENV] = "1"
        return True
    return env.pop(CHILD_ENV, None) is not None
