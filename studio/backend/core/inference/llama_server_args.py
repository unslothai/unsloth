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
    frozenset({"--webui", "--no-webui"}),
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
