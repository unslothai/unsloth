# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Boundary validator for user-supplied llama-server pass-through args.

Reject only flags Studio manages (model identity, auth, network,
parallel slots). Everything else (sampling, ``-c``, ``-ngl``,
``--flash-attn``, ``--cache-type-*``, ``--spec-*``, ``--jinja``, ...)
is appended after Studio's auto-set flags so llama.cpp's last-wins
parser lets the user override.

Ref: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

from typing import Iterable, Optional

# Each group = every alias (short + long) of one hard-denied flag.
# Extend the matching group when llama.cpp adds a new alias.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Parallel slots: owned by typer --parallel; a pass-through would
    # desync app.state.llama_parallel_slots from llama-server.
    frozenset({"-np", "--parallel", "--n-parallel"}),
    # Model identity: Studio resolves it from LoadRequest; a second
    # -m would load a different model than Studio thinks it loaded.
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
    # Networking: Studio binds + proxies; retargeting orphans the proxy.
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    frozenset({"--reuse-port"}),
    # Auth / TLS: Studio terminates auth; upstream --api-key / TLS
    # shadows Studio's key and breaks the proxy hop.
    frozenset({"--api-key"}),
    frozenset({"--api-key-file"}),
    frozenset({"--ssl-key-file"}),
    frozenset({"--ssl-cert-file"}),
    # Built-in web UI. --webui/--no-webui is the legacy spelling;
    # upstream renamed to --ui/--no-ui + --ui-*. Keep both so prebuilt
    # and system llama.cpp binaries both match.
    frozenset({"--webui", "--no-webui"}),
    frozenset({"--ui", "--no-ui"}),
    frozenset({"--ui-config"}),
    frozenset({"--ui-config-file"}),
    frozenset({"--ui-mcp-proxy", "--no-ui-mcp-proxy"}),
    frozenset({"--models-dir"}),
    frozenset({"--models-preset"}),
    frozenset({"--models-max"}),
    frozenset({"--models-autoload", "--no-models-autoload"}),
    # Server-mode flips: --embedding / --rerank restrict llama-server to
    # those endpoints, breaking Studio's /v1/chat/completions hop.
    frozenset({"--embedding", "--embeddings"}),
    frozenset({"--rerank", "--reranking"}),
    # llama-server's own built-in tools flag would silently stack on top
    # of Studio's --enable-tools / --disable-tools policy resolver.
    frozenset({"--tools"}),
)

_DENYLIST: frozenset[str] = frozenset().union(*_DENYLIST_GROUPS)


def _flag_name(token: str) -> Optional[str]:
    """Flag name for ``token``, or None if it isn't a flag.

    Peels `--key=value` to `--key`, treats `-1` / `-0.5` as values
    (llama-server shorts always start with a letter), strips
    whitespace, and normalises attached `-np8` / signed `-np-1` /
    digit-prefix-junk `-np8x` to `-np`. Mirrors the CLI's
    `_expand_attached_np_short`.
    """
    token = token.strip()
    if not token.startswith("-") or token in {"-", "--"}:
        return None
    if len(token) >= 2 and (token[1].isdigit() or token[1] == "."):
        return None
    name = token.split("=", 1)[0]
    if len(name) > 3 and name.startswith("-np"):
        suffix = name[3:]
        if suffix[0].isdigit() or (
            len(suffix) > 1 and suffix[0] in {"-", "+"} and suffix[1].isdigit()
        ):
            return "-np"
    return name


def validate_extra_args(args: Optional[Iterable[str]]) -> list[str]:
    """Validate user-supplied llama-server args. Returns a flat list
    ready to extend the llama-server command; raises ``ValueError``
    naming the offending flag on the first managed token."""
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
    parse_ctx_override(out)
    return out


def is_managed_flag(flag: str) -> bool:
    """True if ``flag`` is Studio-managed. Normalises via ``_flag_name``
    so `-np8` / `--parallel=8` classify like the canonical tokens."""
    normalised = _flag_name(flag)
    return normalised is not None and normalised in _DENYLIST


# Pass-through flags that shadow first-class LoadRequest fields;
# stripped from inherited extras so they can't last-wins-override an
# Apply that re-sets the same field.
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
        # MTP path (llama.cpp #22673).
        "--spec-draft-n-max",
        "--spec-draft-n-min",
        "--spec-draft-p-min",
        "--spec-draft-p-split",
        "--spec-ngram-mod-n-match",
        "--spec-ngram-mod-n-min",
        "--spec-ngram-mod-n-max",
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

# Shadowing flags that take no value -- strip the flag only, never the
# following token.
_BOOLEAN_SHADOWING_FLAGS: frozenset[str] = frozenset(
    {"--spec-default", "--jinja", "--no-jinja"}
)


def parse_ctx_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``-c`` / ``--ctx-size`` value.

    Mirrors llama.cpp's last-wins flag parsing for the one pass-through
    numeric knob Studio's load-time fit logic needs to see.
    """
    if not args:
        return None

    tokens = [str(a) for a in args]
    override: Optional[int] = None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in _CONTEXT_FLAGS:
            i += 1
            continue

        if "=" in tok:
            raw_value = tok.split("=", 1)[1]
            i += 1
        else:
            if i + 1 >= n or _flag_name(tokens[i + 1]) is not None:
                raise ValueError(
                    f"llama-server flag '{flag}' requires an integer value"
                )
            raw_value = tokens[i + 1]
            i += 2

        try:
            value = int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(
                f"llama-server flag '{flag}' requires an integer value"
            ) from exc
        if value < 0:
            raise ValueError(
                f"llama-server flag '{flag}' requires a non-negative integer value"
            )
        override = value

    return override


def strip_shadowing_flags(
    args: Iterable[str],
    *,
    strip_context: bool = True,
    strip_cache: bool = True,
    strip_spec: bool = True,
    strip_template: bool = True,
) -> list[str]:
    """Strip flags that shadow first-class Studio settings.

    Used when inheriting a previous load's ``llama_extra_args`` so an
    inherited `-c 4096` can't override the current `max_seq_length`
    (same for cache / spec / template). Each ``strip_*`` toggle
    controls one group; the route only strips groups whose first-class
    field the caller actually supplied.
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
        # Drop the flag; consume the next token too unless it's
        # boolean, already inline (`-c=4096`), or another flag.
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out
