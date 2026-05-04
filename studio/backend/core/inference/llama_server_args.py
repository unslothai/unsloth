# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Validator for user-supplied llama-server pass-through args.

Studio runs llama-server as a managed subprocess and lets callers pass
extra flags directly (CLI: ``unsloth studio run ... --top-k 20``;
HTTP: ``LoadRequest.llama_extra_args``). This module is the security
boundary that rejects flags Studio fundamentally controls -- the model
identity, network endpoint, auth key, GPU placement, and anything that
would break Studio's HTTP proxy or error reporting.

Tier-2 knobs that have a sibling ``LoadRequest`` field
(``--chat-template-file``, ``--cache-type-k/v``, ``--spec-type``,
sampling, etc.) are intentionally NOT denied. User-supplied args are
appended to ``cmd`` after Studio's auto-set flags, so llama.cpp's
last-wins parsing makes the user's value override the auto-set one.

Reference: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

from typing import Iterable, Optional

# Each group is the full set of aliases (short + long) for one
# Studio-managed flag, taken from the llama-server README. If
# llama.cpp adds a new alias for an existing managed flag, extend the
# relevant group.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Model identity -- Studio resolves the model from LoadRequest and
    # passes -m / mmproj after downloading from HF if needed.
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
    # Networking -- Studio binds llama-server's port and proxies HTTP.
    # Letting the user retarget host/port/path/prefix would orphan
    # Studio's proxy.
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    # Auth / TLS -- Studio terminates auth at its own layer; an
    # upstream --api-key would shadow Studio's UNSLOTH_DIRECT_STREAM
    # key, and TLS on llama-server would break the local proxy hop.
    frozenset({"--api-key"}),
    frozenset({"--api-key-file"}),
    frozenset({"--ssl-key-file"}),
    frozenset({"--ssl-cert-file"}),
    # Context length -- Studio computes effective_ctx from LoadRequest,
    # GPU-fit logic, and metadata, then surfaces it to the UI. Letting
    # the user pass -c would desync the UI's context slider from the
    # actual server.
    frozenset({"-c", "--ctx-size"}),
    # Server slot count -- Studio uses this for parallel inference
    # accounting and request scheduling.
    frozenset({"-np", "--parallel"}),
    # Forced perf / error-handling defaults that the rest of Studio
    # depends on. --flash-attn off would silently halve throughput;
    # --context-shift on would silently rotate the KV cache instead of
    # surfacing the "increase context length" error to the UI.
    frozenset({"-fa", "--flash-attn"}),
    frozenset({"--no-context-shift", "--context-shift"}),
    frozenset({"--jinja", "--no-jinja"}),
    # GPU placement -- Studio's _select_gpus / --fit logic owns this.
    frozenset({"-fit", "--fit"}),
    frozenset({"-fitt", "--fit-target"}),
    frozenset({"-fitc", "--fit-ctx"}),
    frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers"}),
    frozenset({"-t", "--threads"}),
    # Single-model server -- Studio runs one model per llama-server
    # process and serves its own UI. Multi-model / web-ui flags would
    # change the surface llama-server exposes.
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
