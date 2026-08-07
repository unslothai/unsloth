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

# Valid llama-server --parallel range, shared with LoadRequest.n_parallel.
# Mirrored by callers that cannot import this: run.py and unsloth_cli/commands/
# studio.py (_PARALLEL_MIN/MAX), per-model-config.ts (N_PARALLEL_MIN/MAX);
# test_parallel_slots_per_load.py pins them together.
PARALLEL_MIN = 1
PARALLEL_MAX = 64

# Each group = every alias (short + long) of one hard-denied flag.
# Extend the matching group when llama.cpp adds a new alias.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Parallel slots: owned by typer --parallel and LoadRequest.n_parallel; a
    # pass-through would desync the slot bookkeeping from llama-server.
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

    Peels `--key=value` to `--key`, normalises long-option underscores like
    llama.cpp, treats `-1`/`-0.5` as values (shorts always start with a letter),
    and normalises attached `-np8` / `-np-1` / `-np8x` to `-np`. Mirrors the
    CLI's `_expand_attached_np_short`.
    """
    token = token.strip()
    if not token.startswith("-") or token in {"-", "--"}:
        return None
    if len(token) >= 2 and (token[1].isdigit() or token[1] == "."):
        return None
    name = token.split("=", 1)[0]
    if name.startswith("--"):
        name = name.replace("_", "-")
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
    parse_gpu_layers_override(out)
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
# llama.cpp placement flags. Opt-in (users may pass them under auto-select):
# stripped only when gpu_ids is set, so they cannot override the selected pool
# or choose a main GPU outside it (#7188).
_DEVICE_FLAGS: frozenset[str] = frozenset({"--device", "-dev", "--main-gpu", "-mg"})

# GPU-offload flags. Stripped only when the GPU Memory mode owns offload
# (manual emits --fit / --gpu-layers / --n-cpu-moe); in auto, a user's
# inherited -ngl is respected (the offload_overridden path), so this group is
# opt-in, not default. Layer flags are shared with llama_cpp's override
# detection; the MoE flags are strip-only (manual's --n-cpu-moe slider owns them).
_GPU_LAYER_FLAGS: frozenset[str] = frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers"})
_FIT_FLAGS: frozenset[str] = frozenset({"-fit", "--fit"})
_LAYER_OFFLOAD_FLAGS: frozenset[str] = _GPU_LAYER_FLAGS | _FIT_FLAGS
_MOE_OFFLOAD_FLAGS: frozenset[str] = frozenset({"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"})
_OFFLOAD_SHADOWING_FLAGS: frozenset[str] = _LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS

# Host-memory placement flags. Both are full-model RAM reservations (--mlock pins
# it, --no-mmap mallocs a copy), so the Model Memory settings own them: stripped
# only when a toggle vetoes them, never unconditionally.
_MLOCK_FLAGS: frozenset[str] = frozenset({"--mlock", "-mlock"})
# Modern spelling of both, as an enum value. Takes a value, so NOT boolean.
_LOAD_MODE_FLAGS: frozenset[str] = frozenset({"--load-mode", "-lm"})
_NO_MMAP_FLAGS: frozenset[str] = frozenset({"--no-mmap", "-no-mmap"})
# Deprecated selectors for the same load-mode enum. Measured: ANY of them
# trailing the managed flag resets the WHOLE mode and drops the mlock, in both
# polarities ("--mmap" and "--no-direct-io" do it too). Affirmative dio streams
# and holds no full copy; the negative spellings are NOT plain mmap, upstream
# maps them to mode `none` like --no-mmap, so no-reserve must veto those too.
_DIO_ON_FLAGS: frozenset[str] = frozenset({"--direct-io", "-dio"})
_DIO_OFF_FLAGS: frozenset[str] = frozenset({"--no-direct-io", "-ndio"})
_DIO_FLAGS: frozenset[str] = _DIO_ON_FLAGS | _DIO_OFF_FLAGS
_LOAD_MODE_ALIAS_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | frozenset({"--mmap"}) | _DIO_FLAGS
# Every spelling that asks for a full-model host buffer.
_RAM_RESERVING_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | _DIO_OFF_FLAGS
# llama.cpp reads these before argv, so an inherited value survives stripping the
# equivalent tokens. Scrubbed whenever a toggle is on, like the spec/placement
# env groups, so the setting owns memory placement outright.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "LLAMA_ARG_MLOCK",
    "LLAMA_ARG_MMAP",
    "LLAMA_ARG_LOAD_MODE",
    "LLAMA_ARG_DIO",
    # Legacy negative aliases, honoured by PRESENCE whatever the value.
    "LLAMA_ARG_NO_MMAP",
    "LLAMA_ARG_NO_DIO",
)

_SHADOWING_FLAGS: frozenset[str] = (
    _CONTEXT_FLAGS | _CACHE_FLAGS | _SPEC_FLAGS | _TEMPLATE_FLAGS | _SPLIT_SHADOWING_FLAGS
)

# Shadowing flags that take no value -- strip the flag only, not the next token.
_BOOLEAN_SHADOWING_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--jinja",
        "--no-jinja",
        "-cmoe",
        "--cpu-moe",
        "--mlock",
        "-mlock",
        "--no-mmap",
        "-no-mmap",
        "--mmap",
        "--direct-io",
        "-dio",
        "--no-direct-io",
        "-ndio",
    }
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
                raise ValueError(f"llama-server flag '{flag}' requires an integer value")
            raw_value = tokens[i + 1]
            i += 2

        try:
            value = int(str(raw_value).strip())
        except ValueError as exc:
            raise ValueError(f"llama-server flag '{flag}' requires an integer value") from exc
        if value < 0:
            raise ValueError(f"llama-server flag '{flag}' requires a non-negative integer value")
        override = value

    return override


def resolve_requested_ctx(args: Optional[Iterable[str]], fallback_n_ctx: int) -> int:
    """Return the context size load_model should treat as requested.

    Single source of truth for load_model's ctx-override conditional so
    tests don't reimplement and assert against their own logic.
    """
    override = parse_ctx_override(args)
    return override if override is not None else fallback_n_ctx


def _last_flag_value(args: Optional[Iterable[str]], flags: frozenset[str]) -> Optional[str]:
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


def parse_gpu_layers_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied GPU layer count from extras.

    Manual GPU memory mode strips llama.cpp offload flags because the
    first-class load fields own them. Callers use this parser first to preserve
    an explicit ``-ngl`` / ``--gpu-layers`` / ``--n-gpu-layers`` value when
    translating the extras into those fields.
    """
    raw_value = _last_flag_value(args, _GPU_LAYER_FLAGS)
    if raw_value is None:
        return None
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("llama-server GPU layers flag requires an integer value") from exc
    if value < -1:
        raise ValueError("llama-server GPU layers flag requires an integer value of at least -1")
    return value


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


def resolve_tensor_parallel(args: Optional[Iterable[str]], fallback_tensor_parallel: bool) -> bool:
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
    strip_device: bool = False,
    strip_mlock: bool = False,
    strip_no_mmap: bool = False,
    strip_load_mode_aliases: bool = False,
    strip_load_mode: bool = False,
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
    row/none/layer choice intact. ``strip_device`` is enabled when ``gpu_ids``
    owns placement.

    ``strip_mlock`` / ``strip_no_mmap`` are enabled by the Model Memory settings
    so a RAM-reservation flag cannot survive a load the user asked to keep
    RAM-free. ``strip_no_mmap`` covers every spelling of mode `none`, so the
    negative DirectIO forms go with it. All boolean: only the token is dropped.
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
    if strip_device:
        shadowing |= _DEVICE_FLAGS
    if strip_mlock:
        shadowing |= _MLOCK_FLAGS
    if strip_no_mmap:
        shadowing |= _RAM_RESERVING_FLAGS
    if strip_load_mode_aliases:
        shadowing |= _LOAD_MODE_ALIAS_FLAGS
    if strip_load_mode:
        shadowing |= _LOAD_MODE_FLAGS

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


def apply_model_memory_policy(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool = False,
    weights_in_host_memory: bool = True,
) -> tuple[list[str], list[str]]:
    """Resolve the Model Memory settings into llama-server flags.

    Returns ``(managed_flags, extras)``: what Unsloth emits itself, and the
    user's extras with any vetoed flag removed.

    "Keep model in GPU memory" page-locks the weights (``--load-mode mmap+mlock``,
    or the deprecated ``--mlock``) but ONLY when ``weights_in_host_memory``.
    mlock pins a whole mapping in host RAM, so for a model fully offloaded to a
    discrete GPU it would hold a second full copy of the weights in system RAM
    without doing anything for VRAM residency; there, residency is carried by
    the idle-unload veto alone. Every other load-mode-bearing flag is stripped
    from the emitted extras, because a trailing one resets the whole mode and
    would drop the mlock.

    "Don't reserve system RAM" drops ``--mlock`` / ``--no-mmap``, leaving the
    default mmap path. With both off nothing is stripped, so a hand-typed flag
    still applies.
    """
    try:
        from utils.model_memory_settings import get_no_ram_reserve, should_mlock
    except Exception:
        # Settings unavailable (bare unit-test import): behave as before.
        return [], list(extra_args or [])

    no_ram_reserve = get_no_ram_reserve()
    tokens = list(extra_args or [])
    if no_ram_reserve:
        tokens = strip_shadowing_flags(
            tokens,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_mlock = True,
            strip_no_mmap = True,
        )
        tokens = _strip_reserving_load_modes(tokens)

    managed: list[str] = []
    if should_mlock() and weights_in_host_memory:
        # Before the extras, like the rest of the managed block. mmap+mlock, not
        # bare mlock: it matches what --mlock meant alongside the default mmap.
        managed.extend(["--load-mode", "mmap+mlock"] if supports_load_mode else ["--mlock"])
        tokens = strip_shadowing_flags(
            tokens,
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_mlock = True,
            strip_load_mode_aliases = True,
            strip_load_mode = True,
        )
    return managed, tokens


def _strip_reserving_load_modes(tokens: list[str]) -> list[str]:
    """Drop only ``--load-mode`` values that lock or reserve host RAM.

    No-reserve vetoes the reservation, not the loader. ``mmap`` and ``dio``
    hold no full host copy, so a DirectIO preset survives instead of silently
    falling back to mmap. Unknown values are left alone rather than rewritten.
    """
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        if _flag_name(token) not in _LOAD_MODE_FLAGS:
            out.append(token)
            i += 1
            continue
        if "=" in token:
            value, step = token.split("=", 1)[1], 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            value, step = tokens[i + 1], 2
        else:
            value, step = "", 1
        value = value.strip().lower()
        if value in _LOAD_MODE_MLOCK_VALUES or value in _LOAD_MODE_RESERVING_VALUES:
            i += step
            continue
        out.extend(tokens[i : i + step])
        i += step
    return out


def model_memory_owns_placement() -> bool:
    """True when either toggle is on, so the child env must be scrubbed."""
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return False
    return get_keep_resident() or get_no_ram_reserve()


def scrub_memory_env(env: dict) -> list[str]:
    """Drop inherited memory env vars when the settings own placement.

    Returns the names removed, for logging. A no-op with both toggles off, so
    an existing LLAMA_ARG_MLOCK deployment keeps working untouched.
    """
    if not model_memory_owns_placement():
        return []
    return [name for name in MEMORY_ENV_VARS if env.pop(name, None) is not None]


# Mirrors llama_cpp's _LLAMA_ARG_TRUE/FALSE_VALUES; duplicated so this module
# stays dependency-free (llama_cpp imports from here, not the other way).
_ENV_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_ENV_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})

_LOAD_MODE_MLOCK_VALUES = frozenset({"mlock", "mmap+mlock"})
# Modes that read the weights into a full host buffer. "dio" streams via
# DirectIO and "mmap" maps, so neither reserves RAM for the whole model.
_LOAD_MODE_RESERVING_VALUES = frozenset({"none", "mlock"})


def resolve_effective_memory_state(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[bool, bool]:
    """``(mlock, reserves_ram)`` the child will actually run with.

    Mirrors llama.cpp: env supplies defaults, argv overrides last-wins. Used to
    compare a running process against the current settings, so the reload hint
    reflects the launched state rather than only what Unsloth emitted.
    """
    env = env or {}
    mlock = False
    reserves_ram = False
    # Each var runs the SAME handler as its flag, so it assigns the whole mode
    # and a later one overwrites an earlier one, in llama.cpp's registration
    # order. Measured: LLAMA_ARG_MLOCK=1 with LLAMA_ARG_MMAP=on or
    # LLAMA_ARG_DIO=0 leaves the child unlocked.
    # Only the mlock bit, like the argv --mlock below: "mlock" vs "mmap+mlock"
    # is not observable and changes no decision.
    if str(env.get("LLAMA_ARG_MLOCK", "")).strip().lower() in _ENV_TRUE_VALUES:
        mlock = True
    # Every option with a negative form also answers to LLAMA_ARG_NO_<NAME>:
    # upstream rewrites the name and, if that var EXISTS, forces the value
    # falsey whatever it says, before reading the affirmative one. Measured:
    # LLAMA_ARG_NO_MMAP=0 still disables mmap, and it beats LLAMA_ARG_MMAP=on.
    # --mlock has no negative form, so LLAMA_ARG_NO_MLOCK does nothing.
    # LLAMA_ARG_MMAP is whether to mmap, so "off" means mmap disabled ("none").
    _mmap_env = "0" if "LLAMA_ARG_NO_MMAP" in env else str(env.get("LLAMA_ARG_MMAP", ""))
    _mmap_env = _mmap_env.strip().lower()
    if _mmap_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram = False, False
    elif _mmap_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram = False, True
    # LLAMA_ARG_DIO likewise: on selects DirectIO, off selects "none".
    _dio_env = "0" if "LLAMA_ARG_NO_DIO" in env else str(env.get("LLAMA_ARG_DIO", ""))
    _dio_env = _dio_env.strip().lower()
    if _dio_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram = False, False
    elif _dio_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram = False, True
    _mode_env = str(env.get("LLAMA_ARG_LOAD_MODE", "")).strip().lower()
    if _mode_env:
        mlock = _mode_env in _LOAD_MODE_MLOCK_VALUES
        reserves_ram = _mode_env in _LOAD_MODE_RESERVING_VALUES

    tokens = [str(a) for a in (argv or [])]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None:
            i += 1
            continue
        if flag in _MLOCK_FLAGS:
            # Only the mlock bit: the enum has both "mlock" and "mmap+mlock" and
            # which one this maps to is not observable. It changes no decision,
            # since mlock alone already counts as a reservation for no-reserve.
            mlock = True
            i += 1
        elif flag in _NO_MMAP_FLAGS:
            # Deprecated selector for the whole "none" mode, so it clears the
            # mlock too: measured, "--mlock --no-mmap" leaves the child unlocked
            # while "--no-mmap --mlock" locks it.
            mlock = False
            reserves_ram = True
            i += 1
        elif flag in _DIO_ON_FLAGS:
            # Deprecated load-mode selector: resets the mode, so the mlock goes.
            # DirectIO streams the weights, so it holds no full host copy.
            mlock = False
            reserves_ram = False
            i += 1
        elif flag in _DIO_OFF_FLAGS:
            # Not "plain mmap": upstream maps these to mode `none`, like
            # --no-mmap, which reads the weights into a full host buffer.
            mlock = False
            reserves_ram = True
            i += 1
        elif flag == "--mmap":
            mlock = False
            reserves_ram = False
            i += 1
        elif flag in _LOAD_MODE_FLAGS:
            if "=" in tok:
                value, step = tok.split("=", 1)[1], 1
            elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
                value, step = tokens[i + 1], 2
            else:
                value, step = "", 1
            value = value.strip().lower()
            if value:
                mlock = value in _LOAD_MODE_MLOCK_VALUES
                reserves_ram = value in _LOAD_MODE_RESERVING_VALUES
            i += step
        else:
            i += 1
    return mlock, reserves_ram


def memory_state_satisfies_settings(
    state: Optional[tuple[bool, bool]],
    policy_active: bool = False,
    mlock_applicable: bool = True,
) -> bool:
    """True when a launched ``(mlock, reserves_ram)`` matches the settings.

    Shared by the duplicate-load comparator (so toggling a setting forces a real
    relaunch instead of returning already-loaded) and the settings route (so the
    reload hint agrees with it).

    ``state`` is None for a process this policy does not govern, such as the
    diffusion runner, which has no load-mode of its own; nothing about it can
    contradict the settings, so it always matches.

    ``policy_active`` says the launch differed from an unmanaged one, because a
    flag was emitted, a requested one suppressed, or an inherited env var
    scrubbed. With both toggles off the policy no longer applies, so any of
    those has to be undone on the next launch, while a launch it never touched
    is left alone.

    ``mlock_applicable`` is False when the launch is fully offloaded to a
    discrete GPU, where page-locking host RAM buys nothing and is deliberately
    not emitted. Residency there is the idle-unload veto, which needs no
    relaunch, so demanding mlock would ask for a reload that can never satisfy
    the check.
    """
    if state is None:
        return True
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return True
    mlock, reserves_ram = state
    if get_no_ram_reserve():
        # mlock_applicable only excuses a MISSING lock; a live reservation
        # still has to go, wherever the weights are.
        return not (mlock or reserves_ram)
    if get_keep_resident():
        return mlock or not mlock_applicable
    return not policy_active
