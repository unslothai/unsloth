# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Boundary validator for user-supplied llama-server pass-through args.

Reject only flags Unsloth manages (model identity, auth, network, parallel
slots). Everything else (sampling, ``-c``, ``-ngl``, ``--flash-attn``,
``--cache-type-*``, ``--spec-*``, ``--jinja``, ...) is appended after
Unsloth's auto-set flags so llama.cpp's last-wins parser lets the user override.

Ref: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
"""

from __future__ import annotations

import os
from typing import Iterable, Mapping, Optional

# Each group = every alias (short + long) of one hard-denied flag.
# Extend the matching group when llama.cpp adds a new alias.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Parallel slots: owned by typer --parallel; a pass-through would desync
    # app.state.llama_parallel_slots from llama-server.
    frozenset({"-np", "--parallel", "--n-parallel"}),
    # Model identity: Unsloth resolves it from LoadRequest; a second -m would
    # load a different model than Unsloth thinks it loaded.
    frozenset({"-m", "--model"}),
    # Public model id: Unsloth sets a sanitized --alias so the OpenAI API never
    # exposes the local .gguf path. A user-supplied alias is appended after
    # Unsloth's and, with llama.cpp's last-wins parsing, would reintroduce the
    # path leak this is meant to prevent.
    frozenset({"-a", "--alias"}),
    frozenset({"-mu", "--model-url"}),
    frozenset({"-dr", "--docker-repo"}),
    frozenset({"-hf", "-hfr", "--hf-repo"}),
    frozenset({"-hff", "--hf-file"}),
    frozenset({"-hfv", "-hfrv", "--hf-repo-v"}),
    frozenset({"-hffv", "--hf-file-v"}),
    frozenset({"-hft", "--hf-token"}),
    frozenset({"-mm", "--mmproj"}),
    frozenset({"-mmu", "--mmproj-url"}),
    # Networking: Unsloth binds + proxies; retargeting orphans the proxy.
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    frozenset({"--reuse-port"}),
    # Auth / TLS: Unsloth terminates auth; upstream --api-key / TLS shadows
    # Unsloth's key and breaks the proxy hop.
    frozenset({"--api-key"}),
    frozenset({"--api-key-file"}),
    frozenset({"--ssl-key-file"}),
    frozenset({"--ssl-cert-file"}),
    # Built-in web UI. --webui/--no-webui is the legacy spelling; upstream
    # renamed to --ui/--no-ui + --ui-*. Keep both so prebuilt and system
    # llama.cpp binaries match.
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
    # those endpoints, breaking Unsloth's /v1/chat/completions hop.
    frozenset({"--embedding", "--embeddings"}),
    frozenset({"--rerank", "--reranking"}),
    # llama-server's own built-in tools flag would silently stack on top of
    # Unsloth's --enable-tools / --disable-tools policy resolver.
    frozenset({"--tools"}),
    # Slot-state dir: Studio owns it for KV persistence across idle unload.
    frozenset({"--slot-save-path"}),
)

_DENYLIST: frozenset[str] = frozenset().union(*_DENYLIST_GROUPS)


def _flag_name(token: str) -> Optional[str]:
    """Flag name for ``token``, or None if it isn't a flag.

    Peels `--key=value` to `--key`, treats `-1`/`-0.5` as values (shorts
    always start with a letter), and normalises attached `-np8` / `-np-1` /
    `-np8x` to `-np`. Mirrors the CLI's `_expand_attached_np_short`.
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
    """Validate user-supplied llama-server args. Returns a flat list ready to
    extend the llama-server command; raises ``ValueError`` naming the
    offending flag on the first managed token."""
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
    parse_cache_override(out)
    parse_split_mode_override(out)
    return out


def is_managed_flag(flag: str) -> bool:
    """True if ``flag`` is Unsloth-managed. Normalises via ``_flag_name`` so
    `-np8` / `--parallel=8` classify like the canonical tokens."""
    normalised = _flag_name(flag)
    return normalised is not None and normalised in _DENYLIST


# Pass-through flags that shadow first-class LoadRequest fields; stripped
# from inherited extras so they can't last-wins-override an Apply that
# re-sets the same field.
_CONTEXT_FLAGS: frozenset[str] = frozenset({"-c", "--ctx-size"})
_CACHE_TYPE_K_FLAGS: frozenset[str] = frozenset({"-ctk", "--cache-type-k"})
_CACHE_TYPE_V_FLAGS: frozenset[str] = frozenset({"-ctv", "--cache-type-v"})
_CACHE_FLAGS: frozenset[str] = _CACHE_TYPE_K_FLAGS | _CACHE_TYPE_V_FLAGS
_SPEC_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--spec-type",
        "--spec-ngram-size-n",
        "--spec-ngram-size",
        "--draft-min",
        "--draft-max",
        # MTP path (llama.cpp #22673). The drafter selectors (local --model-draft
        # and HF --spec-draft-hf aliases) are Unsloth-managed since the separate-
        # drafter support (Gemma 4): an inherited copy must not last-wins-override
        # the auto-detected drafter. Explicit extras for the current load are never
        # stripped. The per-drafter tuning knobs (--spec-draft-type-*, -ngld,
        # --spec-draft-device) are deliberately NOT stripped: the VRAM budget reads
        # them via the same parsers the child honors, so they stay consistent on
        # inherit, and stripping them would silently move a CPU-offloaded drafter
        # back onto the GPU.
        "--model-draft",
        "-md",
        "--spec-draft-model",
        "--spec-draft-hf",
        "-hfd",
        "-hfrd",
        "--hf-repo-draft",
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
# Multi-GPU split mode shadows the Tensor Parallelism toggle
# (--split-mode tensor). Pass-through stays allowed so users keep the
# row/none/layer modes the toggle doesn't expose, but it's stripped on
# inherit and reconciled into the round-tripped tensor_parallel state.
# --tensor-split is coupled to the split mode and is stripped with it: Unsloth
# owns the tensor-mode split ratios, so an inherited/stale --tensor-split must
# not last-wins-override Unsloth's computed asymmetric split.
_SPLIT_MODE_FLAGS: frozenset[str] = frozenset({"-sm", "--split-mode"})
_TENSOR_SPLIT_FLAGS: frozenset[str] = frozenset({"-ts", "--tensor-split"})
_SPLIT_SHADOWING_FLAGS: frozenset[str] = _SPLIT_MODE_FLAGS | _TENSOR_SPLIT_FLAGS

# GPU-offload flags. Stripped only when the GPU Memory mode owns offload
# (manual emits --fit / --gpu-layers / --n-cpu-moe); in auto, a user's
# inherited -ngl is respected (the offload_overridden path), so this group is
# opt-in, not default. Layer flags are shared with llama_cpp's override
# detection; the MoE flags are strip-only (manual's --n-cpu-moe slider owns them).
_LAYER_OFFLOAD_FLAGS: frozenset[str] = frozenset(
    {"-ngl", "--gpu-layers", "--n-gpu-layers", "-fit", "--fit"}
)
_MOE_OFFLOAD_FLAGS: frozenset[str] = frozenset(
    {"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"}
)
_OFFLOAD_SHADOWING_FLAGS: frozenset[str] = _LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS

_SHADOWING_FLAGS: frozenset[str] = (
    _CONTEXT_FLAGS
    | _CACHE_FLAGS
    | _SPEC_FLAGS
    | _TEMPLATE_FLAGS
    | _SPLIT_SHADOWING_FLAGS
)

# Shadowing flags that take no value -- strip the flag only, not the next token.
_BOOLEAN_SHADOWING_FLAGS: frozenset[str] = frozenset(
    {"--spec-default", "--jinja", "--no-jinja", "-cmoe", "--cpu-moe"}
)


def parse_ctx_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``-c`` / ``--ctx-size`` value.

    Mirrors llama.cpp's last-wins parsing for the one numeric knob Unsloth's
    load-time fit logic needs.
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


def resolve_requested_ctx(args: Optional[Iterable[str]], fallback_n_ctx: int) -> int:
    """Return the context size load_model should treat as requested.

    Single source of truth for load_model's ctx-override conditional so
    tests don't reimplement and assert against their own logic.
    """
    override = parse_ctx_override(args)
    return override if override is not None else fallback_n_ctx


def _last_flag_value(
    args: Optional[Iterable[str]], flags: frozenset[str]
) -> Optional[str]:
    """Return the last-wins string value among ``flags`` in extras, or None.

    Handles both ``--flag=value`` and ``--flag value`` forms and raises if a
    matched flag has no (or an empty) value. Shared by the single-knob
    last-wins parsers (cache type, split mode).
    """
    if not args:
        return None

    tokens = [str(a) for a in args]
    override: Optional[str] = None
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None or flag not in flags:
            i += 1
            continue

        if "=" in tok:
            raw_value = tok.split("=", 1)[1]
            i += 1
        else:
            if i + 1 >= n or _flag_name(tokens[i + 1]) is not None:
                raise ValueError(f"llama-server flag '{flag}' requires a value")
            raw_value = tokens[i + 1]
            i += 2

        value = str(raw_value).strip()
        if not value:
            raise ValueError(f"llama-server flag '{flag}' requires a non-empty value")
        override = value

    return override


def parse_cache_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins cache type if extras pass cache flags.

    Mirrors parse_ctx_override but for cache type. Recognises both -ctk
    (key) and -ctv (value). When both flags appear, returns the last-wins
    value, treating key and value cache flags as the same setting because
    Unsloth's KV estimate has a single cache_type_kv knob.
    """
    return _last_flag_value(args, _CACHE_FLAGS)


def parse_cache_override_per_axis(
    args: Optional[Iterable[str]],
) -> tuple[Optional[str], Optional[str]]:
    """Last-wins --cache-type-k / --cache-type-v values kept apart, as (k, v).

    parse_cache_override collapses both axes to one last-wins value; this keeps
    them separate so an asymmetric K/V can be budgeted by its heavier axis.
    """
    return (
        _last_flag_value(args, _CACHE_TYPE_K_FLAGS),
        _last_flag_value(args, _CACHE_TYPE_V_FLAGS),
    )


def resolve_cache_type_kv(
    args: Optional[Iterable[str]], fallback_cache_type_kv: Optional[str]
) -> Optional[str]:
    """Return the cache type load_model should treat as requested.

    Single source of truth for ``load_model``'s cache override conditional.
    """
    override = parse_cache_override(args)
    return override if override is not None else fallback_cache_type_kv


def parse_split_mode_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins ``--split-mode`` / ``-sm`` value from extras.

    Mirrors parse_cache_override for the multi-GPU split mode. Returns the
    raw mode string (e.g. ``tensor`` / ``row`` / ``none`` / ``layer``), or
    None when extras don't set it.
    """
    return _last_flag_value(args, _SPLIT_MODE_FLAGS)


def resolve_tensor_parallel(
    args: Optional[Iterable[str]], fallback_tensor_parallel: bool
) -> bool:
    """Return the tensor-parallel state load_model should treat as requested.

    A user-supplied ``--split-mode`` in extras last-wins-overrides the
    toggle, so reconcile it back into the boolean: any explicit split mode
    means tensor-parallel is on iff that mode is ``tensor``. Falls back to
    the toggle value when extras don't set it.
    """
    override = parse_split_mode_override(args)
    if override is None:
        return fallback_tensor_parallel
    return override.strip().lower() == "tensor"


def _env_split_mode_is_tensor(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the inherited LLAMA_ARG_SPLIT_MODE env selects tensor. Unsloth
    emits --split-mode only on its tensor branch, so a tensor env on the layer
    path would run the child tensor-parallel unbudgeted; this flips the budget
    to tensor. Only tensor is heavier, so other modes are ignored."""
    raw = (os.environ if env is None else env).get("LLAMA_ARG_SPLIT_MODE")
    return bool(raw) and raw.strip().lower() == "tensor"


def _effective_tensor_parallel(
    extra_args: Optional[Iterable[str]],
    tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Tensor-parallel decision including the inherited LLAMA_ARG_SPLIT_MODE env.

    resolve_tensor_parallel (extras + toggle), flipped on when extras set no split
    mode but the child inherits a tensor split env. Shared by load_model (which
    budgets and launches it) and the tensor-fallback wrapper (so an env-only
    tensor crash still retries layer split)."""
    resolved = resolve_tensor_parallel(extra_args, tensor_parallel)
    if (
        not resolved
        and parse_split_mode_override(extra_args) is None
        and _env_split_mode_is_tensor(env)
    ):
        return True
    return resolved


def _tensor_parallel_matches_loaded(
    extra_args: Optional[Iterable[str]],
    requested_tensor_parallel: bool,
    loaded_tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Whether a duplicate load request matches a loaded server's tensor state.

    Env-only tensor mode is a launch hint load_model may downgrade to layer split
    (capacity/buffer), scrubbing the child env. So only let an inherited tensor env
    raise a match against a server that *actually* launched tensor; on a downgraded
    (layer) server the env is ignored, and an identical request would downgrade the
    same way -- avoiding an endless reload of a healthy server."""
    requested = resolve_tensor_parallel(extra_args, requested_tensor_parallel)
    if (
        loaded_tensor_parallel
        and not requested
        and parse_split_mode_override(extra_args) is None
        and _env_split_mode_is_tensor(env)
    ):
        requested = True
    return requested == loaded_tensor_parallel


_MMPROJ_DISABLE_FLAGS: frozenset[str] = frozenset({"--no-mmproj", "--no-mmproj-auto"})
_MMPROJ_ENABLE_FLAGS: frozenset[str] = frozenset({"--mmproj-auto"})


def extra_args_disable_mmproj(args: Optional[Iterable[str]]) -> bool:
    """True when pass-through args opt out of vision mmproj loading.

    llama-server parses --mmproj-auto / --no-mmproj / --no-mmproj-auto as one
    boolean with last-wins semantics; mirror that here.
    """
    if not args:
        return False
    disabled = False
    for raw in args:
        flag = _flag_name(str(raw))
        if flag in _MMPROJ_DISABLE_FLAGS:
            disabled = True
        elif flag in _MMPROJ_ENABLE_FLAGS:
            disabled = False
    return disabled


def strip_shadowing_flags(
    args: Iterable[str],
    *,
    strip_context: bool = True,
    strip_cache: bool = True,
    strip_spec: bool = True,
    strip_template: bool = True,
    strip_split_mode: bool = True,
    strip_tensor_split: bool = False,
    strip_offload: bool = False,
) -> list[str]:
    """Strip flags that shadow first-class Unsloth settings.

    Used when inheriting a previous load's ``llama_extra_args`` so an
    inherited `-c 4096` can't override the current `max_seq_length`
    (same for cache / spec / template / split-mode). Each ``strip_*``
    toggle controls one group; the route only strips groups whose
    first-class field the caller actually supplied.

    ``strip_split_mode`` removes both ``--split-mode`` and the coupled
    ``--tensor-split`` (the Tensor Parallelism toggle owns the whole split).
    ``strip_tensor_split`` removes ``--tensor-split`` *alone*, so manual mode can
    replace an inherited per-GPU ratio while leaving the user's ``--split-mode``
    row/none/layer choice intact.
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
    if strip_split_mode:
        shadowing |= _SPLIT_SHADOWING_FLAGS
    if strip_tensor_split:
        shadowing |= _TENSOR_SPLIT_FLAGS
    if strip_offload:
        shadowing |= _OFFLOAD_SHADOWING_FLAGS

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
        # Drop the flag; also consume the next token unless it's boolean,
        # already inline (`-c=4096`), or another flag.
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out


def strip_split_mode_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Remove the split-mode group (``--split-mode`` / ``-sm`` and the coupled
    ``--tensor-split`` / ``-ts``) from ``args``, keeping every other shadow flag.
    Preserves a None/empty input so the inherit-vs-explicit-empty distinction
    survives. Used where tensor mode is being forced off (downgrade / fallback)."""
    if not args:
        return args
    return strip_shadowing_flags(
        args,
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = True,
    )
