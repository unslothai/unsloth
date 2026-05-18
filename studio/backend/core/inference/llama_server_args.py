# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validator for user-supplied llama-server pass-through args.

Studio runs llama-server as a managed subprocess and lets callers pass
extra flags directly (CLI: ``unsloth run ... --top-k 20``; HTTP:
``LoadRequest.llama_extra_args``). This module is the boundary that
rejects only flags Studio fundamentally cannot share with the user --
model identity, the auth key, and the network endpoint Studio's HTTP
proxy targets. Anything else passes through.

User-supplied args are appended to ``cmd`` after Studio's auto-set
flags, so llama.cpp's last-wins CLI parsing makes the user's value
override the auto-set one. That covers tunable knobs the user might
reasonably want to override -- ``-c``/``--ctx-size``,
``-np``/``--parallel``, ``-fa``/``--flash-attn``,
``-ngl``/``--gpu-layers``, ``-t``/``--threads``, ``-fit``/``--fit*``,
``--cache-type-k/v``, ``--chat-template-file/-kwargs``,
``--spec-*``, ``--jinja``/``--no-jinja``,
``--no-context-shift``/``--context-shift``, sampling params, etc.

Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

from typing import Iterable, Optional

# Each group is the full set of aliases (short + long) for one
# hard-denied flag, taken from the llama-server README. If llama.cpp
# adds a new alias for an existing denied flag, extend the relevant
# group.
#
# Flags NOT in this list (e.g. -c, --parallel, --flash-attn, -ngl,
# -t/--threads, --jinja, --no-context-shift, --fit*, --cache-type-*,
# --chat-template-*, --spec-*) pass through and override Studio's
# auto-set version via llama.cpp's last-wins CLI parsing.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Model identity -- Studio resolves the model from LoadRequest and
    # passes -m / mmproj after downloading from HF if needed. A second
    # -m would point at a different model than the one Studio thinks
    # is loaded.
    frozenset({"-m", "--model"}),
    frozenset({"-mu", "--model-url"}),
    frozenset({"-dr", "--docker-repo"}),
    frozenset({"-hf", "-hfr", "--hf-repo"}),
    frozenset({"-hff", "--hf-file"}),
    frozenset({"-hfv", "-hfrv", "--hf-repo-v"}),
    frozenset({"-hffv", "--hf-file-v"}),
    frozenset({"-hft", "--hf-token"}),
    frozenset({"-mm", "--mmproj"}),
    frozenset({"-mmu", "--mmproj-url"}),
    # Networking -- Studio binds llama-server's port and reverse-proxies
    # HTTP traffic to it. Retargeting host/port/path/prefix would
    # orphan Studio's proxy and the UI would lose the server.
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    frozenset({"--reuse-port"}),
    # Auth / TLS -- Studio terminates auth at its own layer; an
    # upstream --api-key would shadow Studio's UNSLOTH_DIRECT_STREAM
    # key, and TLS on llama-server would break the local proxy hop.
    frozenset({"--api-key"}),
    frozenset({"--api-key-file"}),
    frozenset({"--ssl-key-file"}),
    frozenset({"--ssl-cert-file"}),
    # Single-model server -- Studio runs one model per llama-server
    # process and serves its own UI. Enabling multi-model loading or
    # llama-server's built-in web UI changes the surface clients see.
    # ``--webui``/``--no-webui`` are the legacy spelling; current
    # upstream uses ``--ui``/``--no-ui`` + ``--ui-*`` companions.
    # Keep both so the denylist matches old and new llama-server
    # binaries (Studio's prebuilt vs system-llama.cpp).
    frozenset({"--webui", "--no-webui"}),
    frozenset({"--ui", "--no-ui"}),
    frozenset({"--ui-config"}),
    frozenset({"--ui-config-file"}),
    frozenset({"--ui-mcp-proxy", "--no-ui-mcp-proxy"}),
    frozenset({"--models-dir"}),
    frozenset({"--models-preset"}),
    frozenset({"--models-max"}),
    frozenset({"--models-autoload", "--no-models-autoload"}),
)

_DENYLIST: frozenset[str] = frozenset().union(*_DENYLIST_GROUPS)


def _flag_name(token: str) -> Optional[str]:
    """Return the flag name for a token, or None if it isn't a flag.

    Peels ``--key=value`` to the bare ``--key``. Plain numeric values
    like ``-1`` or ``-0.5`` (e.g. ``--seed -1``) are values, not flags;
    llama-server short-form flags always start with a letter.
    """
    if not token.startswith("-") or token in {"-", "--"}:
        return None
    if len(token) >= 2 and (token[1].isdigit() or token[1] == "."):
        return None
    return token.split("=", 1)[0]


def validate_extra_args(args: Optional[Iterable[str]]) -> list[str]:
    """Validate user-supplied llama-server args.

    Returns the args as a flat list ready to extend the llama-server
    command. Raises ``ValueError`` (with the offending flag in the
    message) the moment a token resolves to a Studio-managed flag.
    """
    if not args:
        return []
    out: list[str] = []
    for raw in args:
        token = str(raw)
        flag = _flag_name(token)
        if flag is not None and flag in _DENYLIST:
            raise ValueError(
                f"llama-server flag '{flag}' is managed by Unsloth Studio "
                f"and cannot be passed as an extra arg"
            )
        out.append(token)
    return out


def is_managed_flag(flag: str) -> bool:
    """True if ``flag`` is a Studio-managed llama-server flag."""
    return flag in _DENYLIST


# Pass-through flags that shadow first-class ``LoadRequest`` fields
# (max_seq_length, cache_type_kv, speculative_type,
# chat_template_override). Stripped from inherited extras so they
# can't last-wins-override an Apply that re-sets the same first-class
# field.
_CONTEXT_FLAGS: frozenset[str] = frozenset({"-c", "--ctx-size"})
_CACHE_FLAGS: frozenset[str] = frozenset(
    {"-ctk", "--cache-type-k", "-ctv", "--cache-type-v"}
)
_SPEC_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--spec-type",
        "--spec-ngram-size-n",
        "--spec-ngram-size",
        "--draft-min",
        "--draft-max",
    }
)
_TEMPLATE_FLAGS: frozenset[str] = frozenset(
    {
        "--chat-template",
        "--chat-template-file",
        "--chat-template-kwargs",
        "--jinja",
        "--no-jinja",
    }
)

_SHADOWING_FLAGS: frozenset[str] = (
    _CONTEXT_FLAGS | _CACHE_FLAGS | _SPEC_FLAGS | _TEMPLATE_FLAGS
)

# Boolean flags inside _SHADOWING_FLAGS that take no value. The
# value-consuming heuristic in strip_shadowing_flags must skip just the
# flag for these, never the following token.
_BOOLEAN_SHADOWING_FLAGS: frozenset[str] = frozenset(
    {"--spec-default", "--jinja", "--no-jinja"}
)


def strip_shadowing_flags(
    args: Iterable[str],
    *,
    strip_context: bool = True,
    strip_cache: bool = True,
    strip_spec: bool = True,
    strip_template: bool = True,
) -> list[str]:
    """Strip flags that shadow first-class Studio settings.

    Used when the route inherits a previous load's ``llama_extra_args``
    so that an inherited ``-c 4096`` cannot override the current
    request's ``max_seq_length`` (and equivalents for cache /
    speculative / chat template). Each ``strip_*`` flag controls one
    group; the route only strips groups whose corresponding first-class
    field was actually supplied by the caller, so an inherited
    ``--chat-template-file`` survives an Apply that omits both
    ``llama_extra_args`` and ``chat_template_override``.
    """
    shadowing: set[str] = set()
    if strip_context:
        shadowing |= _CONTEXT_FLAGS
    if strip_cache:
        shadowing |= _CACHE_FLAGS
    if strip_spec:
        shadowing |= _SPEC_FLAGS
    if strip_template:
        shadowing |= _TEMPLATE_FLAGS

    tokens = [str(a) for a in (args or [])]
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in shadowing:
            out.append(tok)
            i += 1
            continue
        # Drop this token. Boolean shadowing flags never carry a value;
        # other shadowing flags consume the next token when it isn't a
        # flag and the value isn't already packed as ``--key=value``.
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out
