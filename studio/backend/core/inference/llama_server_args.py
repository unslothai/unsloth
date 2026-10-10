# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Boundary validator for user-supplied llama-server pass-through args. Reject only flags Unsloth
manages (model identity, auth, network, parallel slots). Everything else (sampling, ``-c``,
``-ngl``, ``--flash-attn``, ``--cache-type-*``, ``--spec-*``, ``--jinja``, ...) is appended after
Unsloth's auto-set flags so llama.cpp's last-wins parser lets the user override. Ref:
https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md"""

from __future__ import annotations

import logging
import math
import os
import re
import struct
import sys
from typing import Any, Callable, Iterable, Mapping, Optional

from utils.reasoning_budget import validate_reasoning_budget_message

logger = logging.getLogger(__name__)

# Valid --parallel range; mirrored in run.py, unsloth_cli studio.py, per-model-config.ts (test pins).
PARALLEL_MIN = 1
PARALLEL_MAX = 64

PARALLEL_DEFAULT = 4


def clamp_parallel_slots(n_parallel) -> int:
    if n_parallel is None:
        return PARALLEL_DEFAULT
    try:
        asked = int(n_parallel)
    except (TypeError, ValueError):
        return PARALLEL_DEFAULT
    return max(PARALLEL_MIN, min(PARALLEL_MAX, asked))


# Mirrored by N_BATCH_MIN/MAX in per-model-config.ts
BATCH_MIN = 1
BATCH_MAX = 65536

# Sanity bounds, not upstream ones. --cache-ram -1 = no limit, 0 disables.
# Mirrored by CTX_CHECKPOINTS_MAX / CACHE_RAM_MAX in per-model-config.ts.
CTX_CHECKPOINTS_MAX = 256
CACHE_RAM_MAX_MIB = 1024 * 1024

# llama.cpp allocates this default even when Studio emits no flag.
LLAMA_CTX_CHECKPOINTS_DEFAULT = 32

# Checkpoints live in host RAM: a hybrid's whole recurrent state, or an SWA model's window.
CTX_CHECKPOINT_HOST_BUDGET_FRACTION = 0.05
CTX_CHECKPOINT_HOST_BUDGET_FLOOR_BYTES = 1024**3
# Zero forces a full prompt re-ingest after divergence.
CTX_CHECKPOINTS_MIN_USEFUL = 2

# Single source of slot-count aliases for the denial, its hint and the single-sequence retry.
_PARALLEL_FLAGS: frozenset[str] = frozenset({"-np", "--parallel", "--n-parallel"})

# Each group = every alias of one denied flag; extend when llama.cpp adds an alias.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    _PARALLEL_FLAGS,
    frozenset({"-m", "--model"}),
    # Unsloth sets a sanitized --alias; a user alias (last-wins) would leak the local .gguf path.
    frozenset({"-a", "--alias"}),
    frozenset({"-mu", "--model-url"}),
    frozenset({"-dr", "--docker-repo"}),
    frozenset({"-hf", "-hfr", "--hf-repo"}),
    frozenset({"-hff", "--hf-file"}),
    frozenset({"-hfv", "-hfrv", "--hf-repo-v"}),
    frozenset({"-hffv", "--hf-file-v"}),
    frozenset({"-hft", "--hf-token"}),
    frozenset({"-mmu", "--mmproj-url"}),
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    frozenset({"--reuse-port"}),
    # Unsloth terminates auth; upstream --api-key / TLS breaks the proxy hop
    frozenset({"--api-key"}),
    frozenset({"--api-key-file"}),
    frozenset({"--ssl-key-file"}),
    frozenset({"--ssl-cert-file"}),
    # --webui is the legacy spelling of --ui; keep both for old and new llama.cpp binaries.
    frozenset({"--webui", "--no-webui"}),
    frozenset({"--ui", "--no-ui"}),
    frozenset({"--ui-config", "--webui-config"}),
    frozenset({"--ui-config-file", "--webui-config-file"}),
    frozenset({"--ui-mcp-proxy", "--webui-mcp-proxy", "--no-ui-mcp-proxy", "--no-webui-mcp-proxy"}),
    frozenset({"--models-dir"}),
    frozenset({"--models-preset"}),
    frozenset({"--models-max"}),
    frozenset({"--models-autoload", "--no-models-autoload"}),
    frozenset({"--embedding", "--embeddings"}),
    frozenset({"--rerank", "--reranking"}),
    # Pooling decides whether the embedding launch is safe; an override could pick NONE/RANK.
    frozenset({"--pooling"}),
    frozenset({"--tools"}),
    # --agent enables ALL built-in tools (incl. exec_shell_command), same as --tools.
    frozenset({"-ag", "--agent", "-no-ag", "--no-agent"}),
    # Where those tools run: docker:/podman: spins up a container, ssh:<target> runs them on another host entirely.
    frozenset({"--tools-runtime"}),
    # MCP servers: upstream says do not enable in untrusted environments.
    frozenset({"--mcp-servers-config"}),
    frozenset({"--mcp-servers-json"}),
    # Widening CORS would bypass the origin boundary the proxy enforces.
    frozenset({"--cors-origins"}),
    frozenset({"--cors-headers"}),
    frozenset({"--cors-methods"}),
    frozenset({"--cors-credentials", "--no-cors-credentials"}),
    frozenset({"--media-path"}),
    # _classify_llama_start_failure reads startup output; redirecting it hides the cause.
    frozenset({"--log-file"}),
    frozenset({"--log-disable"}),
    # Unsloth owns the slot-state dir. --slots/--props are deliberately NOT denied (Unsloth reads only /props).
    frozenset({"--slot-save-path"}),
    # These print and exit, so the load would only time out later
    frozenset({"-h", "--help", "--usage"}),
    frozenset({"--version"}),
    frozenset({"--list-devices"}),
    frozenset({"-cl", "--cache-list"}),
    frozenset({"--completion-bash"}),
)

_DENYLIST: frozenset[str] = frozenset().union(*_DENYLIST_GROUPS)

# Path-valued flags the child opens; account access is not checked for them, so owner only.
OWNER_ONLY_PATH_FLAGS: frozenset[str] = frozenset(
    {
        "-mm",
        "--mmproj",
        "--spec-draft-model",
        "-md",
        "--model-draft",
        "-mv",
        "--model-vocoder",
        "--lora",
        "--lora-scaled",
        "--control-vector",
        "--control-vector-scaled",
        "--chat-template-file",
        "--grammar-file",
        "-jf",
        "--json-schema-file",
        "-lcs",
        "--lookup-cache-static",
        "-lcd",
        "--lookup-cache-dynamic",
        "--log-prompts-dir",
        "--video-ffmpeg-dir",
        # Not a path: llama-server sends the model's tensors to these hosts.
        "--rpc",
    }
)


def owner_only_path_args(args: Optional[Iterable[str]]) -> list[tuple[str, str]]:
    """``(flag, value)`` for each OWNER_ONLY_PATH_FLAGS occurrence in ``args``, in order."""
    tokens = [str(a) for a in args or ()]
    found: list[tuple[str, str]] = []
    for i, raw in enumerate(tokens):
        flag = _flag_name(raw)
        if flag in OWNER_ONLY_PATH_FLAGS:
            _, eq, inline = raw.partition("=")
            found.append((flag, inline if eq else (tokens[i + 1] if i + 1 < len(tokens) else "")))
    return found


def owner_only_path_flags(args: Optional[Iterable[str]]) -> list[str]:
    """The OWNER_ONLY_PATH_FLAGS present in ``args``, in first-seen order."""
    return list(dict.fromkeys(flag for flag, _ in owner_only_path_args(args)))


# Flags taking TWO values (from llama-server --help); so the positional check allows the 2nd.
_TWO_VALUE_FLAGS: frozenset[str] = frozenset({"--control-vector-layer-range"})

# Second value optional: new builds use FNAME:SCALE, old ones FNAME SCALE.
_OPTIONAL_SECOND_VALUE_FLAGS: frozenset[str] = frozenset(
    {"--lora-scaled", "--control-vector-scaled"}
)

MAX_EXTRA_ARG_TOKENS = 256
MAX_EXTRA_ARGS_BYTES = 32 * 1024
# CreateProcess caps the whole command line at 32767 chars; leave room for Unsloth's own args.
MAX_EXTRA_ARGS_BYTES_WINDOWS = 24 * 1024


WINDOWS_COMMAND_LIMIT = 32767
WINDOWS_COMMAND_RESERVE = 8192


def windows_command_length(args: list) -> int:
    """Characters ``subprocess`` would put on a Windows command line for ``args``. list2cmdline is
    the exact serializer Popen uses there, and it is not a sum of lengths: a value needing quotes
    has its backslashes doubled, so an escape-heavy grammar can nearly double. Measuring it is
    the only honest check."""
    import subprocess
    return len(subprocess.list2cmdline([str(a) for a in args]))


def max_extra_args_bytes() -> int:
    """The size cap for this platform."""
    return MAX_EXTRA_ARGS_BYTES_WINDOWS if sys.platform == "win32" else MAX_EXTRA_ARGS_BYTES


def _flag_name(token: str) -> Optional[str]:
    """Flag name for ``token``, or None if it isn't a flag. Peels `--key=value` to `--key`,
    normalises long-option underscores like llama.cpp, treats `-1`/`-0.5` as values (shorts
    always start with a letter), and normalises attached `-np8` / `-np-1` / `-np8x` to `-np`.
    Mirrors the CLI's `_expand_attached_np_short`."""
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


def _value_is_attached(token: str, flag: str) -> bool:
    """Whether this token carries its own value, rather than expecting the next one. Not "the name
    changed": _flag_name also folds llama.cpp's underscore spelling (--ctx_size is --ctx-size to
    it, and the binary takes both), so comparing the normalised name against the raw token read
    "--ctx_size 4096" as attached and then refused the 4096 as a bare value. Only "=" and an
    attached short like -np8 are values in the same token."""
    raw = token.strip()
    if "=" in raw:
        return True
    return raw.replace("_", "-") != flag


def _is_spawnable(token: str) -> bool:
    """Whether execve could carry this token at all (no unpaired surrogates)."""
    try:
        token.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _has_control_characters(token: str) -> bool:
    """A NUL, or any C0 control other than tab and newline."""
    return any(ch == "\x00" or (ord(ch) < 32 and ch not in "\t\n") for ch in token)


def validate_extra_args(args: Optional[Iterable[str]]) -> list[str]:
    """Validate user-supplied llama-server args. Returns a flat list ready to extend the
    llama-server command; raises ``ValueError`` naming the offending flag on the first managed
    token."""
    if not args:
        return []
    out: list[str] = []
    total_bytes = 0
    pending_values = 0
    pending_two_value = 0
    two_value_flag = ""
    for raw in args:
        token = str(raw)
        if len(out) >= MAX_EXTRA_ARG_TOKENS:
            raise ValueError(
                f"too many extra llama-server args (limit {MAX_EXTRA_ARG_TOKENS} tokens)"
            )
        # Cap the whole list (grammars are long single tokens). Unpaired surrogates make Popen raise later.
        try:
            encoded = token.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError(
                "extra llama-server args cannot contain unpaired surrogate characters"
            ) from error
        total_bytes += len(encoded)
        limit = max_extra_args_bytes()
        if total_bytes > limit:
            raise ValueError(f"extra llama-server args are too large (limit {limit} bytes)")
        # execve rejects NUL; other control chars would reach the child's parser as garbage
        if _has_control_characters(token):
            raise ValueError("extra llama-server args cannot contain control characters")
        flag = _flag_name(token)
        if flag is not None and flag in _DENYLIST:
            message = (
                f"llama-server flag '{flag}' is managed by Unsloth Studio "
                f"and cannot be passed as an extra arg"
            )
            if flag in _PARALLEL_FLAGS:
                message += "; set n_parallel on the load request (parallel decode slots) instead"
            raise ValueError(message)
        if flag is None:
            # A bare positional fails the launch, or worse is read as the model path (bypasses -m deny).
            if pending_values <= 0:
                raise ValueError(
                    "extra llama-server args cannot contain a bare value "
                    f"('{token[:64]}'); every value must follow its flag"
                )
            pending_values -= 1
            if pending_two_value > 0:
                pending_two_value -= 1
        elif token != token.strip():
            # _flag_name strips before lookup, so a padded "--top-k " would slip past the denylist
            raise ValueError(
                f"llama-server does not accept the spaces around '{token[:64]}': "
                f"write it as '{flag}'"
            )
        elif "=" in token:
            # llama.cpp does not accept --flag=value (exits 'invalid argument'), so refuse it here.
            value = token.partition("=")[2]
            raise ValueError(
                f"llama-server does not read an attached value: write '{flag}' and "
                f"'{value[:32]}' as two separate arguments, not '{token[:64]}'"
            )
        else:
            attached = _value_is_attached(token, flag)
            if pending_two_value > 0:
                raise ValueError(f"llama-server flag '{two_value_flag}' takes two values")
            # An attached value is only ONE of the two; START= still owes END.
            if flag in _TWO_VALUE_FLAGS:
                pending_values = 1 if attached else 2
                pending_two_value = pending_values
            elif flag in _OPTIONAL_SECOND_VALUE_FLAGS:
                pending_values = 1 if attached else 2
                pending_two_value = 0
            else:
                pending_values = 0 if attached else 1
                pending_two_value = 0
            two_value_flag = flag
        out.append(token)
    if pending_two_value > 0:
        raise ValueError(f"llama-server flag '{two_value_flag}' takes two values")
    if sys.platform == "win32":
        serialized = windows_command_length(out)
        budget = WINDOWS_COMMAND_LIMIT - WINDOWS_COMMAND_RESERVE
        if serialized > budget:
            raise ValueError(
                "extra llama-server args are too long for a Windows command line "
                f"({serialized} characters after quoting, limit {budget})"
            )
    parse_ctx_override(out)
    parse_cache_override(out)
    parse_split_mode_override(out)
    parse_gpu_layers_override(out)
    parse_tensor_split_override(out)
    parse_reasoning_budget_override(out)
    parse_reasoning_budget_message_override(out)
    return out


def drop_managed_flags(args: Optional[Iterable[str]]) -> tuple[list[str], list[str]]:
    """Split stored args into what still loads and the flag names removed.

    For the paths that CARRY OVER an existing value rather than receive a new one. The denylist
    grows (``--agent`` and the MCP flags were added once a text box made them one paste away), so an
    override saved by an older build can hold a name that is refused today. Refusing there punishes
    a user for a decision made later: the load, or the save of an unrelated setting, fails naming a
    flag they may not remember writing. Dropping is the same judgement applied quietly.

    A flag takes its value with it, or ``--log-file /var/log/x`` would leave a bare ``/var/log/x``
    behind, which llama.cpp reads as a positional model path. The bounds and the control-character
    rule are enforced by re-validating what is left, so the result is always something
    ``validate_extra_args`` accepts.
    """
    tokens = [str(raw) for raw in (args or [])]

    def _takes_next(
        index: int,
        token: str,
        flag: str,
        source: list = None,
    ) -> bool:
        """True when the token's value is the NEXT token rather than its own. ``source`` defaults to
        the input list; the trimming loop passes the list it is shortening, where "the next
        token" means the one just removed."""
        if _value_is_attached(token, flag):
            return False
        seq = tokens if source is None else source
        if source is not None:
            return True
        following = seq[index + 1] if index + 1 < len(seq) else None
        return following is not None and _flag_name(following) is None

    kept: list[str] = []
    dropped: list[str] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        flag = _flag_name(token)
        if flag is not None and flag in _DENYLIST:
            dropped.append(flag)
            skip_next = _takes_next(index, token, flag)
            continue
        # A poisoned value takes its flag too, else the flag would eat the next token.
        if _has_control_characters(token) or not _is_spawnable(token):
            # Placeholder, not the token: this list is logged and the token has control characters.
            dropped.append("<flag>" if flag is not None else "<value>")
            if flag is not None:
                # Drop the value too: an orphan is a bare positional, read as the model path.
                skip_next = _takes_next(index, token, flag)
            elif kept:
                owner = _flag_name(kept[-1])
                if owner is not None:
                    dropped.append(owner)
                    kept.pop()
            continue
        if flag is not None and "=" in token:
            # Dropped here, not by the tail-trimming loop, so one bad token does not cost later flags.
            # After the control-character check, so a poisoned name is logged as a placeholder.
            dropped.append(flag)
            continue
        if flag is not None and token != token.strip():
            dropped.append(flag)
            skip_next = _takes_next(index, token, flag)
            continue
        if (
            flag is not None
            and _takes_next(index, token, flag)
            and (_has_control_characters(tokens[index + 1]) or not _is_spawnable(tokens[index + 1]))
        ):
            dropped.append(flag)
            continue
        kept.append(token)

    while kept:
        try:
            return validate_extra_args(kept), dropped
        except ValueError:
            # Shed from the tail; drop a flag whose value just went, else llama-server refuses it. Log names.
            dropped.append(_flag_name(kept[-1]) or "<value>")
            kept = kept[:-1]
            last_flag = _flag_name(kept[-1]) if kept else None
            if last_flag is not None and _takes_next(len(kept) - 1, kept[-1], last_flag, kept):
                dropped.append(last_flag)
                kept = kept[:-1]
            # Drop a two-value flag whole; half of it fails at startup.
            while len(kept) >= 2:
                owner = _flag_name(kept[-2])
                if (
                    owner in _TWO_VALUE_FLAGS
                    and _flag_name(kept[-1]) is None
                    and "=" not in kept[-2]
                ):
                    dropped.append(owner)
                    kept = kept[:-2]
                    continue
                break
    return [], dropped


def sorted_managed_flags() -> list[str]:
    """Every denied flag, sorted, for a UI that wants to explain a rejection before the request is
    made. The validator stays the authority; this is only a mirror."""
    return sorted(_DENYLIST)


def is_managed_flag(flag: str) -> bool:
    """True if ``flag`` is Unsloth-managed. Normalises via ``_flag_name`` so `-np8` / `--parallel=8`
    classify like the canonical tokens."""
    normalised = _flag_name(flag)
    return normalised is not None and normalised in _DENYLIST


# Shadow LoadRequest fields; stripped from inherited extras so they can't last-wins-override.
_CONTEXT_FLAGS: frozenset[str] = frozenset({"-c", "--ctx-size"})
_CACHE_TYPE_K_FLAGS: frozenset[str] = frozenset({"-ctk", "--cache-type-k"})
_CACHE_TYPE_V_FLAGS: frozenset[str] = frozenset({"-ctv", "--cache-type-v"})
_CACHE_FLAGS: frozenset[str] = _CACHE_TYPE_K_FLAGS | _CACHE_TYPE_V_FLAGS
_REASONING_BUDGET_FLAGS: frozenset[str] = frozenset({"--reasoning-budget"})
_REASONING_BUDGET_MESSAGE_FLAGS: frozenset[str] = frozenset({"--reasoning-budget-message"})
_REASONING_BUDGET_MAX = 2_147_483_647
_SPEC_FLAGS: frozenset[str] = frozenset(
    {
        "--spec-default",
        "--spec-type",
        "--spec-ngram-size-n",
        "--spec-ngram-size",
        "--draft-min",
        "--draft-max",
        # Drafter selectors are Unsloth-managed: strip inherited copies. Tuning knobs (-ngld etc) are not
        # stripped: the VRAM budget reads them, and stripping would move a CPU drafter onto the GPU.
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
        # enable_thinking's new spelling; a template override recomputes it. Takes a value, so not boolean.
        "--reasoning",
        "-rea",
        "--jinja",
        "--no-jinja",
    }
)
# Shadows the Tensor Parallelism toggle; stripped on inherit. --tensor-split goes with it.
_SPLIT_MODE_FLAGS: frozenset[str] = frozenset({"-sm", "--split-mode"})
_TENSOR_SPLIT_FLAGS: frozenset[str] = frozenset({"-ts", "--tensor-split"})
_SPLIT_SHADOWING_FLAGS: frozenset[str] = _SPLIT_MODE_FLAGS | _TENSOR_SPLIT_FLAGS
# Stripped only when gpu_ids is set, so they cannot pick a GPU outside the selected pool.
_DEVICE_FLAGS: frozenset[str] = frozenset({"--device", "-dev", "--main-gpu", "-mg"})

# Stripped only when manual GPU Memory mode owns offload; auto respects an inherited -ngl.
_GPU_LAYER_FLAGS: frozenset[str] = frozenset({"-ngl", "--gpu-layers", "--n-gpu-layers"})
# Stripped only when the matching field is set
_BATCH_FLAGS: frozenset[str] = frozenset({"-b", "--batch-size"})
_UBATCH_FLAGS: frozenset[str] = frozenset({"-ub", "--ubatch-size"})
# --swa-checkpoints is upstream's older spelling of --ctx-checkpoints
_CTX_CHECKPOINTS_FLAGS: frozenset[str] = frozenset(
    {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}
)
_CACHE_RAM_FLAGS: frozenset[str] = frozenset({"-cram", "--cache-ram"})
# One group: the control sets a single dtype, so an inherited pair that split K from V has to go whole
_SPEC_DRAFT_CACHE_K_FLAGS: frozenset[str] = frozenset(
    {"-ctkd", "--cache-type-k-draft", "--spec-draft-type-k"}
)
_SPEC_DRAFT_CACHE_V_FLAGS: frozenset[str] = frozenset(
    {"-ctvd", "--cache-type-v-draft", "--spec-draft-type-v"}
)
_SPEC_DRAFT_CACHE_FLAGS: frozenset[str] = _SPEC_DRAFT_CACHE_K_FLAGS | _SPEC_DRAFT_CACHE_V_FLAGS
_FIT_FLAGS: frozenset[str] = frozenset({"-fit", "--fit"})
# Never stripped (last-wins), so a pass-through value is what the child keeps free.
_FIT_TARGET_FLAGS: frozenset[str] = frozenset({"-fitt", "--fit-target"})
_FIT_CTX_FLAGS: frozenset[str] = frozenset({"-fitc", "--fit-ctx"})
_LAYER_OFFLOAD_FLAGS: frozenset[str] = _GPU_LAYER_FLAGS | _FIT_FLAGS
_MOE_OFFLOAD_FLAGS: frozenset[str] = frozenset({"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"})
_OFFLOAD_SHADOWING_FLAGS: frozenset[str] = _LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS

# Full-model RAM reservations; stripped only when a Model Memory toggle vetoes them.
_MLOCK_FLAGS: frozenset[str] = frozenset({"--mlock", "-mlock"})
# Takes a value, so NOT boolean.
_LOAD_MODE_FLAGS: frozenset[str] = frozenset({"--load-mode", "-lm"})
_NO_MMAP_FLAGS: frozenset[str] = frozenset({"--no-mmap", "-no-mmap"})
# Deprecated selectors reset the WHOLE mode (dropping mlock). Negative spellings map to `none`,
# so no-reserve must veto those too.
_DIO_ON_FLAGS: frozenset[str] = frozenset({"--direct-io", "-dio"})
_DIO_OFF_FLAGS: frozenset[str] = frozenset({"--no-direct-io", "-ndio"})
_DIO_FLAGS: frozenset[str] = _DIO_ON_FLAGS | _DIO_OFF_FLAGS
_LOAD_MODE_ALIAS_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | frozenset({"--mmap"}) | _DIO_FLAGS
_RAM_RESERVING_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | _DIO_OFF_FLAGS
# llama.cpp reads these before argv, so inherited values survive token stripping.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "LLAMA_ARG_MLOCK",
    "LLAMA_ARG_MMAP",
    "LLAMA_ARG_LOAD_MODE",
    "LLAMA_ARG_DIO",
    # Honoured by PRESENCE whatever the value.
    "LLAMA_ARG_NO_MMAP",
    "LLAMA_ARG_NO_DIO",
)

_SHADOWING_FLAGS: frozenset[str] = (
    _CONTEXT_FLAGS | _CACHE_FLAGS | _SPEC_FLAGS | _TEMPLATE_FLAGS | _SPLIT_SHADOWING_FLAGS
)

# Take no value: strip the flag only, not the next token.
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
    """Return the last user-supplied ``-c`` / ``--ctx-size`` value. Mirrors llama.cpp's last-wins
    parsing for the one numeric knob Unsloth's load-time fit logic needs."""
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


def parse_ctx_checkpoints_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``--ctx-checkpoints`` value, or None. The control emits its
    flag before the extras, so a copy typed for this load last-wins at launch. Sizing has to
    price that value, not the field, or a ``--ctx-checkpoints 256`` in the extras allocates 256
    per-slot snapshots against a fit that budgeted the field's count."""
    value = _last_flag_value(args, _CTX_CHECKPOINTS_FLAGS)
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return max(0, parsed)


def resolve_ctx_checkpoints(args: Optional[Iterable[str]], requested: Optional[int]) -> int:
    """Resolve explicit counts only, with extra arguments taking precedence."""
    override = parse_ctx_checkpoints_override(args)
    return int(override if override is not None else (requested or 0))


def ctx_checkpoints_within_host_budget(
    per_checkpoint_bytes: int,
    n_parallel: int,
    total_host_bytes: Optional[int],
    *,
    upstream_default: Optional[int] = None,
) -> int:
    """Fit checkpoints per slot within the host budget and this build's own default.

    Unknown sizes keep the default. The minimum useful count may exceed the target budget
    but never the default: this caps what the child would keep, it never raises it.
    """
    default = (
        LLAMA_CTX_CHECKPOINTS_DEFAULT if upstream_default is None else max(0, int(upstream_default))
    )
    if per_checkpoint_bytes <= 0 or not total_host_bytes or total_host_bytes <= 0:
        return default
    budget = max(
        CTX_CHECKPOINT_HOST_BUDGET_FLOOR_BYTES,
        int(total_host_bytes * CTX_CHECKPOINT_HOST_BUDGET_FRACTION),
    )
    per_round = int(per_checkpoint_bytes) * max(1, int(n_parallel))
    affordable = budget // per_round
    if affordable >= default:
        return default
    return min(default, max(CTX_CHECKPOINTS_MIN_USEFUL, int(affordable)))


def effective_ctx_checkpoints(
    args: Optional[Iterable[str]],
    requested: Optional[int],
    *,
    supports_flag: bool,
    per_checkpoint_bytes: int = 0,
    n_parallel: int = 1,
    total_host_bytes: Optional[int] = None,
    upstream_default: Optional[int] = None,
    inherited: Optional[int] = None,
) -> int:
    """Resolve the child count: extras, field, an inherited env value, then the budget.

    ``inherited`` is llama.cpp's own LLAMA_ARG_CTX_CHECKPOINTS, which it applies before
    argv, so argv beats it and it beats the build default.
    """
    if not supports_flag:
        return 0
    override = resolve_ctx_checkpoints(args, requested)
    if override:
        return override
    if parse_ctx_checkpoints_override(args) == 0 or requested == 0:
        return 0
    if inherited is not None:
        return inherited
    return ctx_checkpoints_within_host_budget(
        per_checkpoint_bytes, n_parallel, total_host_bytes, upstream_default = upstream_default
    )


def resolve_requested_ctx(args: Optional[Iterable[str]], fallback_n_ctx: int) -> int:
    """Return the context size load_model should treat as requested. Single source of truth for
    load_model's ctx-override conditional so tests don't reimplement and assert against their own
    logic."""
    override = parse_ctx_override(args)
    return override if override is not None else fallback_n_ctx


def matches_explicit_ctx_override(args: Optional[Iterable[str]], n_ctx: Any) -> bool:
    """Whether a pass-through ``-c``/``--ctx-size`` matches the context the caller is already sending
    as a first-class field.

    Context is the one first-class field whose load-time value is a VRAM-fit TARGET, so a matching
    flag is not a stale shadow but the user's standing decision to run past the estimated threshold.
    Both strippers ask here rather than mirroring the test, so auto-switch and /load inheritance
    cannot drift.

    False for anything unconfirmable, which is the pre-existing strip: no flag, a malformed one, or
    a non-positive/non-int ``n_ctx``. That last case is real -- override rows are coerced on write
    but returned verbatim on read, so a row from an older build can hold any JSON type and must not
    raise here.
    """
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
        return False
    try:
        return parse_ctx_override(args) == n_ctx
    except ValueError:
        return False


def _last_flag_value(
    args: Optional[Iterable[str]],
    flags: frozenset[str],
    *,
    preserve_raw: bool = False,
    validate_value: Optional[Callable[[str], object]] = None,
) -> Optional[str]:
    """Return the last-wins string value among ``flags`` in extras, or None. Handles both
    ``--flag=value`` and ``--flag value`` forms and raises if a matched flag has no (or an empty)
    value. Shared by the single-knob last-wins parsers (cache type, split mode)."""
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

        raw_value = str(raw_value)
        if not raw_value.strip():
            raise ValueError(f"llama-server flag '{flag}' requires a non-empty value")
        if validate_value is not None:
            validate_value(raw_value)
        override = raw_value if preserve_raw else raw_value.strip()

    return override


def parse_cache_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins cache type if extras pass cache flags. Mirrors parse_ctx_override but
    for cache type. Recognises both -ctk (key) and -ctv (value); when both appear, returns the
    last-wins value, treating key and value cache flags as the same setting because Unsloth's KV
    estimate has a single cache_type_kv knob."""
    return _last_flag_value(args, _CACHE_FLAGS)


def _validate_reasoning_budget_value(raw_value: str) -> int:
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError("llama-server --reasoning-budget requires an integer value") from exc
    if value < -1:
        raise ValueError("llama-server --reasoning-budget requires a value of at least -1")
    if value > _REASONING_BUDGET_MAX:
        raise ValueError(
            f"llama-server --reasoning-budget requires a value of at most {_REASONING_BUDGET_MAX}"
        )
    return value


def parse_reasoning_budget_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``--reasoning-budget`` value."""
    raw_value = _last_flag_value(
        args, _REASONING_BUDGET_FLAGS, validate_value = _validate_reasoning_budget_value
    )
    return None if raw_value is None else int(raw_value)


def parse_reasoning_budget_message_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last user-supplied ``--reasoning-budget-message`` value."""
    value = _last_flag_value(
        args,
        _REASONING_BUDGET_MESSAGE_FLAGS,
        preserve_raw = True,
        validate_value = validate_reasoning_budget_message,
    )
    return value


def resolve_reasoning_budget(args: Optional[Iterable[str]], fallback: int) -> int:
    override = parse_reasoning_budget_override(args)
    return override if override is not None else fallback


def resolve_reasoning_budget_message(args: Optional[Iterable[str]], fallback: str) -> str:
    override = parse_reasoning_budget_message_override(args)
    return override if override is not None else fallback


def resolve_reasoning_budget_with_env(
    args: Optional[Iterable[str]],
    fallback: int,
    env: Optional[Mapping[str, str]] = None,
) -> int:
    """Resolve CLI/first-class intent, then inherit llama.cpp's env default."""
    override = parse_reasoning_budget_override(args)
    if override is not None:
        return override
    if fallback != -1:
        return fallback
    raw_value = (env if env is not None else os.environ).get("LLAMA_ARG_THINK_BUDGET")
    if raw_value is None:
        return fallback
    return _validate_reasoning_budget_value(raw_value)


def resolve_reasoning_budget_message_with_env(
    args: Optional[Iterable[str]],
    fallback: str,
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Resolve CLI/first-class intent, then inherit llama.cpp's env default."""
    override = parse_reasoning_budget_message_override(args)
    if override is not None:
        return override
    if fallback:
        return fallback
    return (env if env is not None else os.environ).get("LLAMA_ARG_THINK_BUDGET_MESSAGE", "")


def parse_gpu_layers_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied GPU layer count from extras. Manual GPU memory mode strips
    llama.cpp offload flags because the first-class load fields own them, so callers use this
    parser first to preserve an explicit ``-ngl`` / ``--gpu-layers`` / ``--n-gpu-layers`` value
    when translating the extras into those fields."""
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


def _as_emitted(value: float) -> float:
    """``value`` as the manual launcher will write it, which is ``f"{x:g}"``: six significant
    digits, so the text the child parses is not always the number validated here."""
    return float(f"{value:g}")


def _as_float32(value: float) -> float:
    """``value`` as llama.cpp would hold it: overflow becomes inf rather than raising.

    ``struct.pack`` raises OverflowError where the C cast it stands in for saturates, so the
    caller would have to guard every call site instead of just testing isfinite once.
    """
    try:
        return struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


# Largest share llama.cpp's float array can hold; anything above is out_of_range to std::stof.
_FLOAT32_MAX = struct.unpack("=f", struct.pack("=f", 3.4028234663852886e38))[0]

# FLT_MIN: libstdc++ stof throws out_of_range on subnormal results too.
_FLOAT32_MIN_NORMAL = struct.unpack("=f", struct.pack("=f", 1.1754943508222875e-38))[0]


def parse_tensor_split_override(
    args: Optional[Iterable[str]], *, reserialized: bool = False
) -> Optional[list[float]]:
    """Return the last user-supplied ``-ts`` / ``--tensor-split`` ratios from extras.

    Manual GPU memory with ``gpu_layers >= 0`` strips ``--tensor-split`` because the first-class
    ``tensor_split`` field owns it. Callers must promote the last-wins extras value into that
    field first, or an asymmetric MoE split such as ``-ts 2.2,1`` is discarded and llama-server
    falls back to a near-even layer count (#11330).

    Delimiters match llama.cpp's ``[,/]+`` (``-ts 3/1`` is the same as ``-ts 3,1``). Degenerate
    values raise so ``validate_extra_args`` can refuse them as a 400 rather than stripping them
    silently.

    Bounds are llama.cpp's, not Python's: ``std::stof`` (common/arg.cpp) throws
    ``std::out_of_range`` above FLT_MAX and on any subnormal, and the shares are prefix-summed
    into a float array (llama-model.cpp), so a value this parser would take as a finite double
    can still abort the server at startup. Every share is rounded to float32 BEFORE it joins the
    total, because that is the order llama.cpp adds them in.

    ``reserialized`` is which text the child will parse. Manual mode promotes the ratio into the
    first-class field and the launcher writes it back out with ``f"{x:g}"``, six significant
    digits, so there the emitted string is judged. Everywhere else ``-ts`` is pass-through and
    llama-server reads the user's own text, so judging a rounded version would refuse input that
    runs: ``-ts 1.1754943508222874e-38,1`` is fine as typed and subnormal once re-serialized.
    """
    raw_value = _last_flag_value(args, _TENSOR_SPLIT_FLAGS)
    if raw_value is None:
        return None
    try:
        parts = [float(p) for p in re.split(r"[,/]+", raw_value) if p.strip()]
    except ValueError as exc:
        raise ValueError(
            "llama-server --tensor-split requires a comma- or slash-separated list of numbers"
        ) from exc
    if not parts:
        raise ValueError(
            "llama-server --tensor-split requires a comma- or slash-separated list of numbers"
        )
    if any((not math.isfinite(v)) or v < 0 for v in parts):
        raise ValueError("llama-server --tensor-split entries must be finite and non-negative")
    if sum(parts) <= 0:
        raise ValueError("llama-server --tensor-split must have a positive total")
    running = 0.0
    for part in parts:
        # The share as the child will hold it (six-digit rendering when re-serialized).
        share = _as_float32(_as_emitted(part) if reserialized else part)
        if not math.isfinite(share):
            raise ValueError(
                "llama-server --tensor-split entries must fit in a 32-bit float "
                f"(at most {_FLOAT32_MAX:g})"
            )
        if part != 0 and share < _FLOAT32_MIN_NORMAL:
            raise ValueError(
                "llama-server --tensor-split entries must be 0 or at least "
                f"{_FLOAT32_MIN_NORMAL:g}: a smaller share is a subnormal float and "
                "std::stof refuses it"
            )
        # llama.cpp prefix-sums in float32, so accumulate the same way.
        running = _as_float32(running + share)
        if not math.isfinite(running):
            raise ValueError("llama-server --tensor-split adds up past the 32-bit float range")
    return parts


def check_batch_floor(args: Optional[Iterable[str]], n_parallel: int) -> None:
    """Raise when a pass-through --batch-size would abort llama-server. The launcher raises the
    value it emits itself to ``max(slots, 2)``, with the measurements recorded beside that code:
    ``-b 1`` aborts at any slot count, and a batch below ``--parallel`` aborts too. Extras are
    appended AFTER that flag and win the last-wins parse, so a small value here is not a smaller
    batch, it is a server that dies during startup, and by then the previous model has been
    unloaded. Only the shapes that are certainly wrong: an unreadable value is left to
    llama-server, which names it better than a guess here would."""
    raw_value = _last_flag_value(args, _BATCH_FLAGS)
    if raw_value is None:
        return
    try:
        value = int(raw_value)
    except ValueError:
        return
    floor = max(2, int(n_parallel or 1))
    if value < floor:
        raise ValueError(
            f"llama-server aborts on --batch-size {value}: it needs at least {floor} "
            f"for the {max(1, int(n_parallel or 1))} parallel slot(s) this load serves"
        )


def fit_is_enabled_in(args: Optional[Iterable[str]]) -> bool:
    """Whether the last ``--fit`` in extras turns the fitter ON. Only ``--fit on`` hands placement
    back to llama.cpp; ``--fit off`` disables it and so cannot move weights to the CPU. Upstream
    requires a value and rejects anything that is neither truthy nor falsey, so an absent or
    unreadable value is not an enable."""
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    return raw_value is not None and raw_value.strip().lower() in _ENV_TRUE_VALUES


def fit_is_effectively_on(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the fitter actually runs, over the WHOLE argv and the env twin. ``fit_is_enabled_in``
    answers for the extras alone; this answers for the child. llama.cpp defaults the fitter ON
    and applies the env before argv, so only an explicit "off" turns it off, and an unreadable
    value keeps it on."""
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT")
    if raw_value is None:
        return True
    return str(raw_value).strip().lower() not in _ENV_FALSE_VALUES


def fit_ctx_in(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Return the last --fit-ctx, falling back to the supplied environment.

    Unset or invalid values return None. Preserve negatives: llama.cpp stores them
    unsigned, disabling context reduction.
    """
    raw_value = _last_flag_value(args, _FIT_CTX_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT_CTX")
    if raw_value is None:
        return None
    try:
        return int(str(raw_value).strip())
    except ValueError:
        return None


def fit_target_margin_in(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[float]:
    """The per-device margin an effective ``--fit-target`` asks the fitter to keep.

    ``-fitt/--fit-target`` takes a list of MiB values, one per device, and a single value is
    broadcast across all of them (common/arg.cpp; default 1024, ``fit_params_target`` in
    common/common.h). The fitter refuses to allocate into that margin and spills the rest to host
    RAM instead (``targets.push_back(dmds_full[id].free - margins[id])``, common/fit.cpp), so a
    load-mode fit that credits VRAM has to price it.

    Returns the LARGEST value in the list, because the fit charges one margin to every device it
    credits and understating would claim a fit that is not there. ``None`` when nothing readable is
    set, which leaves the caller on llama.cpp's own default rather than on a guess. Last-wins over
    argv, and the env twin only when argv sets nothing, the same precedence
    ``fit_is_effectively_on`` uses.
    """
    raw_value = _last_flag_value(args, _FIT_TARGET_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT_TARGET")
    if raw_value is None:
        return None
    values: list[float] = []
    # Upstream splits on both ',' and '/' (common/arg.cpp).
    for part in str(raw_value).replace("/", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            return None
    return max(values) if values else None


def split_policy_starves_devices(
    args: Optional[Iterable[str]],
    n_credited: int,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """True when the effective split leaves fewer devices holding weights than ``n_credited``.

    ``--split-mode`` and ``--tensor-split`` are pass-through under auto-select
    (``_SPLIT_SHADOWING_FLAGS`` is stripped only when the Tensor Parallelism toggle owns the split),
    so both reach the child appended after Unsloth's own placement flags, and either can quietly
    shrink the pool a pooled VRAM credit was priced for. ``--split-mode none`` is
    ``LLAMA_SPLIT_MODE_NONE`` (common/arg.cpp), which puts the whole model on ``--main-gpu`` alone,
    while ``row``/``layer``/``tensor`` all keep every device. ``--tensor-split`` is a per-device
    proportion list, and upstream zero-fills every device past the end of the list, so a short list
    starves the tail just as an explicit ``0`` starves its own device.

    Value-aware on purpose: a restatement that keeps all devices is not an override, and voiding on
    it would abstain from a fit that is really there.
    """
    if n_credited <= 1:
        return False
    mode = _last_flag_value(args, _SPLIT_MODE_FLAGS)
    if mode is None and env:
        mode = env.get("LLAMA_ARG_SPLIT_MODE")
    if str(mode or "").strip().lower() == "none":
        return True
    raw_split = _last_flag_value(args, _TENSOR_SPLIT_FLAGS)
    if raw_split is None and env:
        raw_split = env.get("LLAMA_ARG_TENSOR_SPLIT")
    if raw_split is None:
        return False
    holding = 0
    for part in str(raw_split).replace("/", ",").split(",")[:n_credited]:
        part = part.strip()
        if not part:
            continue
        try:
            if float(part) > 0.0:
                holding += 1
        except ValueError:
            return False
    return holding < n_credited


def parse_cache_override_per_axis(
    args: Optional[Iterable[str]],
) -> tuple[Optional[str], Optional[str]]:
    """Last-wins --cache-type-k / --cache-type-v values kept apart, as (k, v). parse_cache_override
    collapses both axes to one last-wins value; this keeps them separate so an asymmetric K/V can
    be budgeted by its heavier axis."""
    return (
        _last_flag_value(args, _CACHE_TYPE_K_FLAGS),
        _last_flag_value(args, _CACHE_TYPE_V_FLAGS),
    )


def resolve_cache_type_kv(
    args: Optional[Iterable[str]], fallback_cache_type_kv: Optional[str]
) -> Optional[str]:
    """Return the cache type load_model should treat as requested. Single source of truth for
    ``load_model``'s cache override conditional."""
    override = parse_cache_override(args)
    return override if override is not None else fallback_cache_type_kv


def parse_split_mode_override(args: Optional[Iterable[str]]) -> Optional[str]:
    """Return the last-wins ``--split-mode`` / ``-sm`` value from extras. Mirrors
    parse_cache_override for the multi-GPU split mode; returns the raw mode string (``tensor`` /
    ``row`` / ``none`` / ``layer``), or None when extras don't set it."""
    return _last_flag_value(args, _SPLIT_MODE_FLAGS)


def resolve_tensor_parallel(args: Optional[Iterable[str]], fallback_tensor_parallel: bool) -> bool:
    """Return the tensor-parallel state load_model should treat as requested. A user-supplied
    ``--split-mode`` in extras last-wins-overrides the toggle, so reconcile it back into the
    boolean: any explicit split mode means tensor-parallel is on iff that mode is ``tensor``.
    Falls back to the toggle value when extras don't set it."""
    override = parse_split_mode_override(args)
    if override is None:
        return fallback_tensor_parallel
    return override.strip().lower() == "tensor"


def _env_split_mode_is_tensor(env: Optional[Mapping[str, str]] = None) -> bool:
    """True when the inherited LLAMA_ARG_SPLIT_MODE env selects tensor. Unsloth emits --split-mode
    only on its tensor branch, so a tensor env on the layer path would run the child
    tensor-parallel unbudgeted; this flips the budget to tensor. Only tensor is heavier, so other
    modes are ignored."""
    raw = (os.environ if env is None else env).get("LLAMA_ARG_SPLIT_MODE")
    return bool(raw) and raw.strip().lower() == "tensor"


def _effective_tensor_parallel(
    extra_args: Optional[Iterable[str]],
    tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Tensor-parallel decision including the inherited LLAMA_ARG_SPLIT_MODE env.
    resolve_tensor_parallel (extras + toggle), flipped on when extras set no split mode but the
    child inherits a tensor split env. Shared by load_model (which budgets and launches it) and
    the tensor-fallback wrapper (so an env-only tensor crash still retries layer split)."""
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
    """Whether a duplicate load request matches a loaded server's tensor state. Env-only tensor mode
    is a launch hint load_model may downgrade to layer split (capacity/buffer), scrubbing the
    child env. So only let an inherited tensor env raise a match against a server that *actually*
    launched tensor; on a downgraded (layer) server the env is ignored, and an identical request
    would downgrade the same way, avoiding an endless reload of a healthy server."""
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
    """True when pass-through args opt out of vision mmproj loading. llama-server parses
    --mmproj-auto / --no-mmproj / --no-mmproj-auto as one boolean with last-wins semantics;
    mirror that here."""
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


def extra_args_image_max_tokens(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Return the effective positive ``--image-max-tokens`` value."""
    found: Optional[int] = None
    raw_env = (os.environ if env is None else env).get("LLAMA_ARG_IMAGE_MAX_TOKENS")
    if raw_env:
        try:
            parsed_env = int(str(raw_env).strip())
        except (TypeError, ValueError):
            parsed_env = 0
        if parsed_env > 0:
            found = parsed_env
    tokens = [str(a) for a in (args or ())]
    for index, raw in enumerate(tokens):
        if raw.startswith("--image-max-tokens="):
            value = raw.partition("=")[2]
        elif raw == "--image-max-tokens" and index + 1 < len(tokens):
            value = tokens[index + 1]
        else:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            found = parsed
    return found


def extra_args_mmproj_auto(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Return whether llama-server will discover an adjacent projector."""
    source_env = os.environ if env is None else env
    enabled = False
    if source_env.get("LLAMA_ARG_NO_MMPROJ_AUTO") is not None:
        enabled = False
    else:
        raw = source_env.get("LLAMA_ARG_MMPROJ_AUTO")
        if raw is not None:
            enabled = str(raw).strip().lower() in _ENV_TRUE_VALUES
    if not args:
        return enabled
    for raw in args:
        flag = _flag_name(str(raw))
        if flag in _MMPROJ_ENABLE_FLAGS:
            enabled = True
        elif flag in _MMPROJ_DISABLE_FLAGS:
            enabled = False
    return enabled


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
    strip_reasoning_budget: bool = False,
    strip_reasoning_budget_message: bool = False,
    strip_mlock: bool = False,
    strip_no_mmap: bool = False,
    strip_load_mode_aliases: bool = False,
    strip_load_mode: bool = False,
    strip_batch: bool = False,
    strip_ubatch: bool = False,
    strip_ctx_checkpoints: bool = False,
    strip_cache_ram: bool = False,
    strip_spec_draft_cache: bool = False,
) -> list[str]:
    """Strip flags that shadow first-class Unsloth settings.

    Used when inheriting a previous load's ``llama_extra_args`` so an inherited `-c 4096` can't
    override the current `max_seq_length` (same for cache / spec / template / split-mode). Each
    ``strip_*`` toggle controls one group; the route only strips groups whose first-class field the
    caller actually supplied.

    ``strip_split_mode`` removes both ``--split-mode`` and the coupled ``--tensor-split`` (the
    Tensor Parallelism toggle owns the whole split). ``strip_tensor_split`` removes
    ``--tensor-split`` *alone*, so manual mode can replace an inherited per-GPU ratio while leaving
    the user's ``--split-mode`` row/none/layer choice intact. ``strip_device`` is enabled when
    ``gpu_ids`` owns placement.

    ``strip_mlock`` / ``strip_no_mmap`` are enabled by the Model Memory settings so a
    RAM-reservation flag cannot survive a load the user asked to keep RAM-free. ``strip_no_mmap``
    covers every spelling of mode `none`, so the negative DirectIO forms go with it. All boolean:
    only the token is dropped.
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
    if strip_reasoning_budget:
        shadowing |= _REASONING_BUDGET_FLAGS
    if strip_reasoning_budget_message:
        shadowing |= _REASONING_BUDGET_MESSAGE_FLAGS
    if strip_mlock:
        shadowing |= _MLOCK_FLAGS
    if strip_no_mmap:
        shadowing |= _RAM_RESERVING_FLAGS
    if strip_load_mode_aliases:
        shadowing |= _LOAD_MODE_ALIAS_FLAGS
    if strip_load_mode:
        shadowing |= _LOAD_MODE_FLAGS
    if strip_batch:
        shadowing |= _BATCH_FLAGS
    if strip_ubatch:
        shadowing |= _UBATCH_FLAGS
    if strip_ctx_checkpoints:
        shadowing |= _CTX_CHECKPOINTS_FLAGS
    if strip_cache_ram:
        shadowing |= _CACHE_RAM_FLAGS
    if strip_spec_draft_cache:
        shadowing |= _SPEC_DRAFT_CACHE_FLAGS

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
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out


def strip_split_mode_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Remove the split-mode group (``--split-mode`` / ``-sm`` and the coupled ``--tensor-split`` /
    ``-ts``) from ``args``, keeping every other shadow flag. Preserves a None/empty input so the
    inherit-vs-explicit-empty distinction survives. Used where tensor mode is being forced off
    (downgrade / fallback)."""
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


def strip_context_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Remove the context group (``-c`` / ``--ctx-size``) from ``args``, keeping every other shadow
    flag. Preserves a None/empty input so the inherit-vs-explicit-empty distinction survives.
    Used by the Metal zero-context floor, where a trailing ``-c 0`` would last-wins override the
    floor and pin the native length again."""
    if not args:
        return args
    return strip_shadowing_flags(
        args,
        strip_context = True,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
    )


MANAGED_DIO_FLAGS: tuple[str, ...] = ("--load-mode", "dio")


def no_reserve_requires_dio(*, supports_load_mode: bool, gpu_offload_confirmed: bool) -> bool:
    """Whether "Don't reserve system RAM" owes this launch ``--load-mode dio``.

    Windows keeps the whole GGUF mapping resident after a full offload, because
    ``unmap_fragment`` is a no-op there (#9033), so the setting only means
    anything on that platform once the offload is confirmed and the build
    understands the flag.

    Placement only. Whether a loader choice further down the chain then overrides
    the pair is not asked here: ``resolve_launch_load_mode`` answers that by
    resolving the argv the policy really produces.
    """
    return sys.platform == "win32" and supports_load_mode and gpu_offload_confirmed


def resolve_launch_load_mode(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool,
    weights_in_host_memory: bool,
    gpu_offload_confirmed: bool,
    requested_load_mode: Optional[str],
    env: Optional[Mapping[str, str]],
    settings: tuple[bool, bool],
) -> tuple[bool, bool]:
    """``(policy_emitted_dio, effective_dio)`` under this ``(keep_resident, no_ram_reserve)``.

    Runs the REAL policy chain and resolves the argv it produces, rather than
    assembling a hypothetical by hand. The launch calls it for each settings pair
    it needs an answer about, so those answers cannot disagree about one launch.

    Asking by hand is what went wrong before: the env view and the surviving
    extras are scrubbed and stripped BY the settings, so a hypothetical built
    from the live ones silently answered for the wrong toggle.

    The two halves are DIFFERENT questions and conflating them is the other way
    to get this wrong. ``effective_dio`` is what the child runs, which a user's
    own ``dio`` satisfies on its own. ``policy_emitted_dio`` is whether THIS
    policy put the pair there, which is what "did the policy change this launch"
    and "would a relaunch add something" actually turn on.
    """
    managed, extras = apply_model_memory_policy(
        extra_args,
        supports_load_mode = supports_load_mode,
        weights_in_host_memory = weights_in_host_memory,
        gpu_offload_confirmed = gpu_offload_confirmed,
        env = env,
        settings = settings,
    )
    selected, extras = apply_load_mode_policy(
        extras,
        supports_load_mode = supports_load_mode,
        weights_in_host_memory = weights_in_host_memory,
        requested_load_mode = requested_load_mode,
        settings = settings,
    )
    return (
        tuple(managed) == MANAGED_DIO_FLAGS,
        resolve_effective_direct_io([*managed, *selected, *extras], env),
    )


def apply_model_memory_policy(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool = False,
    weights_in_host_memory: bool = True,
    gpu_offload_confirmed: bool = False,
    env: Optional[Mapping[str, str]] = None,
    settings: Optional[tuple[bool, bool]] = None,
) -> tuple[list[str], list[str]]:
    """Resolve the Model Memory settings into llama-server flags, returning ``(managed_flags,
    extras)``: what Unsloth emits itself, and the user's extras with any vetoed flag removed.

    "Keep model in GPU memory" page-locks the weights (``--load-mode mmap+mlock``, or the deprecated
    ``--mlock``) but ONLY when ``weights_in_host_memory``. mlock pins a whole mapping in host RAM,
    so for a model fully offloaded to a discrete GPU it would hold a second full copy of the weights
    in system RAM without doing anything for VRAM residency; there, residency is carried by the
    idle-unload veto alone. Every other load-mode-bearing flag is stripped from the emitted extras,
    because a trailing one resets the whole mode and would drop the mlock.

    "Don't reserve system RAM" drops ``--mlock`` / ``--no-mmap``, leaving the
    default mmap path except on Windows with confirmed full GPU offload, where
    supported builds use DirectIO to avoid retaining the resident file mapping.
    With both off nothing is stripped, so a hand-typed flag still applies.

    ``gpu_offload_confirmed`` is that confirmation, and the caller has to
    establish it POSITIVELY. ``not weights_in_host_memory`` is not the same
    thing: that predicate exists to skip a page-lock, so it errs towards True
    for a device it did not probe and towards False for an offload the build
    cannot perform, and neither direction is safe for choosing a loader.

    ``env`` is the child's environment AFTER ``scrub_memory_env``. An inherited
    loader choice that survives the scrub is a non-reserving one the settings
    disclaim, and argv beats the environment in llama.cpp, so the managed
    DirectIO stands aside for it the way the fit's own mode does.

    ``settings`` is a ``(keep_resident, no_ram_reserve)`` the caller already read
    as one coherent snapshot. A launch that has to consult the toggles itself,
    to decide whether a placement answer is worth probing for, must hand that
    same pair down: re-reading here would let a save landing in between gate the
    probe on one value and the flags on another.

    The per-model Mmap/Mlock control is resolved separately, by
    ``apply_load_mode_policy``, which runs after this and defers to it.
    """
    if settings is None:
        try:
            from utils.model_memory_settings import get_model_memory_settings
        except Exception:
            return [], list(extra_args or [])

        # One snapshot for both decisions, so a concurrent save cannot split them.
        settings = get_model_memory_settings()
    keep_resident, no_ram_reserve = settings
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
    if (
        no_ram_reserve
        and no_reserve_requires_dio(
            supports_load_mode = supports_load_mode,
            gpu_offload_confirmed = gpu_offload_confirmed,
        )
        and not memory_env_selects_load_mode(env)
    ):
        # Windows cannot partially unmap the GGUF after offload, so stream (dio).
        managed.extend(MANAGED_DIO_FLAGS)
    if keep_resident and not no_ram_reserve and weights_in_host_memory:
        # mmap+mlock matches what --mlock meant alongside the default mmap.
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


def apply_load_mode_policy(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool = False,
    weights_in_host_memory: bool = True,
    requested_load_mode: Optional[str] = None,
    settings: Optional[tuple[bool, bool]] = None,
) -> tuple[list[str], list[str]]:
    """Resolve the per-model Mmap/Mlock control into llama-server flags, returning ``(managed_flags,
    extras)`` like ``apply_model_memory_policy``, and meant to run straight after it on the extras
    that call returned.

    The Model Memory settings win, which is what the Run settings panel tells the user, so changing
    the order without changing ``loadModeOverrideNotice`` makes that note wrong. "Keep model in GPU
    memory" owns the mode outright while it applies; "Don't reserve system RAM" vetoes the values
    holding a full host copy (``none``, ``mlock``, ``mmap+mlock``) and leaves ``mmap`` and ``dio``
    alone.

    ``auto`` emits nothing: it IS llama.cpp's default, so pinning it would freeze what a later build
    may redefine. An unknown value is dropped rather than passed through, since llama-server exits
    on one.
    """
    tokens = list(extra_args or [])
    mode = _normalize_load_mode_value(requested_load_mode)
    if not mode:
        return [], tokens
    if settings is None:
        try:
            from utils.model_memory_settings import get_model_memory_settings
            settings = get_model_memory_settings()
        except Exception:
            settings = (False, False)
    keep_resident, no_ram_reserve = settings
    if keep_resident and not no_ram_reserve and weights_in_host_memory:
        logger.info(
            "Model Memory: 'Keep model in GPU memory' owns the load mode; "
            "ignoring the requested %r.",
            mode,
        )
        return [], tokens
    if no_ram_reserve and mode in _LOAD_MODE_MLOCK_VALUES | _LOAD_MODE_RESERVING_VALUES:
        logger.info(
            "Model Memory: 'Don't reserve system RAM' drops the requested load mode %r.",
            mode,
        )
        return [], tokens
    if not supports_load_mode:
        legacy = _LEGACY_LOAD_MODE_FLAGS.get(mode)
        if not legacy:
            logger.info("llama-server has no --load-mode; skipping the requested %r mode.", mode)
            return [], tokens
        return list(legacy), tokens
    # Emitted before the extras so a flag typed for this load wins (last-wins).
    return ["--load-mode", mode], tokens


def _normalize_load_mode_value(value: Optional[str]) -> str:
    """Canonical --load-mode, or "" for "no opinion" (unset, auto, unknown)."""
    mode = (value or "").strip().lower()
    if mode in {"", "auto"}:
        return ""
    if mode not in _LOAD_MODE_VALUES:
        logger.warning("Ignoring unknown load mode %r", value)
        return ""
    return mode


def _strip_reserving_load_modes(tokens: list[str]) -> list[str]:
    """Drop only ``--load-mode`` values that lock or reserve host RAM. No-reserve vetoes the
    reservation, not the loader: ``mmap`` and ``dio`` hold no full host copy, so a DirectIO
    preset survives instead of silently falling back to mmap. Unknown values are left alone
    rather than rewritten."""
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


def model_memory_owns_placement(settings: Optional[tuple[bool, bool]] = None) -> bool:
    """True when either toggle is on, so the child env must be scrubbed.

    ``settings`` answers for a GIVEN ``(keep_resident, no_ram_reserve)`` instead of
    the live one, which is what lets the launch ask what it WOULD scrub under a
    setting that is not currently on.
    """
    if settings is not None:
        return settings[0] or settings[1]
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return False
    return get_keep_resident() or get_no_ram_reserve()


def _env_var_locks_or_reserves(name: str, value: str) -> bool:
    """Whether this inherited var, as set, locks or reserves host RAM. Mirrors the argv rule: the
    settings own the RESERVATION, not the loader, so a DirectIO or mmap choice made through the
    environment survives the same way ``--load-mode dio`` does. An unrecognised value is left
    alone."""
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        return normalized in _ENV_TRUE_VALUES
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        return True
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        return normalized in _ENV_FALSE_VALUES
    if name == "LLAMA_ARG_LOAD_MODE":
        return normalized in _LOAD_MODE_MLOCK_VALUES or normalized in _LOAD_MODE_RESERVING_VALUES
    return False


# LLAMA_ARG_* twins of denied flags; llama.cpp reads env before argv. Not a security boundary.
DENIED_ENV_VARS: tuple[str, ...] = (
    "LLAMA_ARG_TOOLS",
    "LLAMA_ARG_TOOLS_RUNTIME",
    "LLAMA_ARG_AGENT",
    "LLAMA_ARG_MCP_SERVERS_CONFIG",
    "LLAMA_ARG_MCP_SERVERS_JSON",
    "LLAMA_ARG_CORS_ORIGINS",
    "LLAMA_ARG_CORS_HEADERS",
    "LLAMA_ARG_CORS_METHODS",
    "LLAMA_ARG_CORS_CREDENTIALS",
    "LLAMA_ARG_MEDIA_PATH",
    # Failure classification reads llama-server's output; a redirect hides the cause.
    "LLAMA_ARG_LOG_FILE",
    "LLAMA_ARG_LOG_DISABLE",
    # Moves /health too, so every load would time out.
    "LLAMA_ARG_API_PREFIX",
    # Unsloth sends the child no auth header, so an inherited key makes it refuse every request.
    "LLAMA_API_KEY",
    "LLAMA_ARG_API_KEY",
    "LLAMA_ARG_API_KEY_FILE",
    # With both, the child listens on https while Unsloth probes http: every load times out.
    "LLAMA_ARG_SSL_KEY_FILE",
    "LLAMA_ARG_SSL_CERT_FILE",
    # Remaining twins of denied flags, from the bundled --help "(env: NAME)" entries.
    "LLAMA_ARG_MODEL",
    "LLAMA_ARG_MODEL_URL",
    "LLAMA_ARG_DOCKER_REPO",
    "LLAMA_ARG_HF_REPO",
    "LLAMA_ARG_HF_FILE",
    "LLAMA_ARG_ALIAS",
    "LLAMA_ARG_HOST",
    "LLAMA_ARG_PORT",
    "LLAMA_ARG_REUSE_PORT",
    "LLAMA_ARG_N_PARALLEL",
    "LLAMA_ARG_POOLING",
    "LLAMA_ARG_EMBEDDINGS",
    "LLAMA_ARG_RERANKING",
    "LLAMA_ARG_UI",
    "LLAMA_ARG_UI_CONFIG",
    "LLAMA_ARG_UI_CONFIG_FILE",
    "LLAMA_ARG_UI_MCP_PROXY",
    "LLAMA_ARG_STATIC_PATH",
    # LLAMA_ARG_MMPROJ / _URL stay allowed: _launch_has_mmproj reads them as launch inputs.
    "LLAMA_ARG_MODELS_DIR",
    "LLAMA_ARG_MODELS_PRESET",
    "LLAMA_ARG_MODELS_MAX",
    "LLAMA_ARG_MODELS_AUTOLOAD",
)

# For the drift test; names do not always match (LLAMA_ARG_STATIC_PATH is --path).
DENIED_ENV_TWIN_FLAGS: dict[str, str] = {
    "LLAMA_ARG_STATIC_PATH": "--path",
    "LLAMA_API_KEY": "--api-key",
    "LLAMA_ARG_API_KEY": "--api-key",
    "LLAMA_ARG_N_PARALLEL": "--parallel",
    "LLAMA_ARG_EMBEDDINGS": "--embeddings",
    "LLAMA_ARG_RERANKING": "--reranking",
    "LLAMA_ARG_UI_CONFIG_FILE": "--ui-config-file",
    "LLAMA_ARG_MODELS_AUTOLOAD": "--models-autoload",
}


def scrub_denied_env(env: dict) -> list[str]:
    """Drop inherited ``LLAMA_ARG_*`` twins of denied flags. Returns the names removed."""
    removed = [name for name in DENIED_ENV_VARS if name in env]
    for name in removed:
        env.pop(name, None)
    return removed


def extra_args_select_load_mode(extra_args: Optional[Iterable[str]]) -> bool:
    """Whether the pass-through block already picks a loader mode itself.

    The argv twin of ``memory_env_selects_load_mode``, and it answers for both spellings: the
    ``--load-mode`` enum, and the flags it replaced (``--no-mmap`` / ``--mmap`` / the direct-IO
    pair), which are what this launch emits on a build predating the enum
    (``_LEGACY_LOAD_MODE_FLAGS``) and so are what a user's own flag would collide with there.

    Presence, not value: a fit-derived mode stands aside for any pick the user made rather than
    trying to rank the two. Their tokens are appended after the managed block and llama.cpp is
    last-wins, so theirs governs either way.
    """
    for raw in extra_args or ():
        if _flag_name(str(raw)) in _LOAD_MODE_FLAGS | _LOAD_MODE_ALIAS_FLAGS:
            return True
    return False


def memory_env_selects_load_mode(env: Optional[Mapping[str, str]]) -> bool:
    """Whether the inherited environment already picks a loader mode.

    llama.cpp applies the ``LLAMA_ARG_*`` twins BEFORE argv (common/arg.cpp runs its environment
    loop and only then "handle command line arguments"), and both assign the same
    ``params.load_mode``, so a managed ``--load-mode`` emitted here beats an inherited choice
    without the user typing anything. A mode DERIVED from a fit has to stand aside for that choice,
    the same way it stands aside for the per-model pick; a hand-typed flag is argv and still wins by
    last-arg.

    Per variable, matching what upstream really does with each value: a no-value option
    (``--mlock``) runs its handler only for a truthy one (``opt.handler_void && is_truthy(value)``),
    a negative alias (``LLAMA_ARG_NO_MMAP`` / ``LLAMA_ARG_NO_DIO``) counts by PRESENCE whatever it
    says (``get_value_from_env`` forces "0" when it exists), and the rest assign a mode for any
    value they parse.
    """
    if not env:
        return False
    for name in MEMORY_ENV_VARS:
        if name not in env:
            continue
        value = str(env.get(name) or "").strip()
        if name == "LLAMA_ARG_MLOCK":
            if value.lower() in _ENV_TRUE_VALUES:
                return True
            continue
        if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
            return True
        if value:
            return True
    return False


def scrub_memory_env(env: dict, settings: Optional[tuple[bool, bool]] = None) -> list[str]:
    """Drop inherited memory placement the settings override.

    Returns the names removed, for logging. A no-op with both toggles off, so an
    existing LLAMA_ARG_MLOCK deployment keeps working untouched. Only the values
    that actually lock or reserve go: an inherited ``LLAMA_ARG_DIO=1`` is a
    loader choice, not a reservation, and no-reserve has no quarrel with it.
    """
    if not model_memory_owns_placement(settings):
        return []
    removed = [
        name
        for name in MEMORY_ENV_VARS
        if name in env and _env_var_locks_or_reserves(name, env[name])
    ]
    for name in removed:
        env.pop(name, None)
    return removed


# Pageable twin of each buffer-allocating mode; upstream sets use_mmap only for mmap modes.
_PAGEABLE_LOAD_MODE: dict[str, Optional[str]] = {"none": None, "mlock": "mmap+mlock"}
# Already-mapping modes: stripped only when a LATER reserving selector shadowed their lock,
# since keeping them would hand the lock back.
_SHADOWED_LOCK_LOAD_MODE = frozenset({"mmap+mlock"})


def _pageable_mode_replacement(
    normalized: str, drop_shadowed_mlock: bool
) -> tuple[bool, Optional[str]]:
    """``(rewrite, replacement)`` for one ``--load-mode`` value, argv or env alike. ``replacement``
    None removes the selector, leaving llama.cpp's default mapping."""
    if normalized in _PAGEABLE_LOAD_MODE:
        return True, None if drop_shadowed_mlock else _PAGEABLE_LOAD_MODE[normalized]
    if normalized in _SHADOWED_LOCK_LOAD_MODE:
        return drop_shadowed_mlock, None
    return False, None


def _pageable_env_value(
    name: str,
    value: str,
    drop_shadowed_mlock: bool = False,
) -> tuple[bool, Optional[str]]:
    """``(rewrite, new_value)`` for an inherited var that disables mmap.

    ``rewrite`` False leaves the var alone. ``new_value`` None means remove it; a string replaces
    it, which is how a locked mode keeps its lock. ``LLAMA_ARG_MLOCK`` is otherwise left alone: on
    its own it sets the lock bit over the default mapping and holds no unmapped copy.

    ``drop_shadowed_mlock`` says the launch is NOT locked before the rewrite, because a later
    selector reset the mlock bit -- llama.cpp resolves these last-wins, so ``LLAMA_ARG_MLOCK=1``
    beside ``LLAMA_ARG_NO_MMAP`` leaves the child unlocked. There is then no lock to carry, so the
    var goes and a ``mlock`` mode is dropped rather than promoted to ``mmap+mlock``.
    """
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        # Only when already shadowed: resurrecting it would page-lock the oversized mapping
        return drop_shadowed_mlock and normalized in _ENV_TRUE_VALUES, None
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        # Presence alone selects mode "none", whatever the value says.
        return True, None
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        return normalized in _ENV_FALSE_VALUES, None
    if name == "LLAMA_ARG_LOAD_MODE":
        return _pageable_mode_replacement(normalized, drop_shadowed_mlock)
    return False, None


def force_pageable_load(
    argv: Optional[Iterable[str]], env: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    """Rewrite a launch that would hold a full unmapped host copy into a pageable one.

    Returns ``(argv, overridden)``, naming the argv tokens and env vars that were dropped or
    rewritten, for the log line and the warning. Empty means the launch was already pageable and
    ``argv`` comes back unchanged.

    Modes ``none`` and ``mlock`` read the weights into a buffer llama.cpp allocates (``use_mmap`` is
    set for ``mmap``/``mmap+mlock``/``auto`` and nothing else), so a model larger than free RAM
    cannot load at all rather than paging in slowly. Both sides of llama.cpp's env-then-argv
    resolution are rewritten, since the environment supplies the default the argv only overrides
    when it names the same option.

    A lock that is EFFECTIVE is preserved rather than dropped: ``mlock`` becomes ``mmap+mlock`` and
    ``--no-mmap --mlock`` keeps its lock through the strip, so "keep this in RAM" still holds, over
    a mapping the kernel can fall back on. ``none`` has no lock to keep and needs no replacement
    flag at all, mmap being the default, so a build predating ``--load-mode`` is handed nothing it
    cannot parse.

    A SHADOWED lock is not resurrected. These options resolve last-wins, so ``--mlock --no-mmap``
    (and the ``LLAMA_ARG_`` twins, where the negative alias is read after the affirmative one) runs
    unlocked and unmapped: the reserving selector already cleared the lock bit, which is what
    ``resolve_effective_memory_state`` reports. Dropping only the selector and leaving the earlier
    ``--mlock`` standing would hand the child ``mmap+mlock`` and page-lock the whole oversized
    mapping into the RAM this override exists to keep pageable, worse than the load it fixes. So the
    pre-rewrite state decides, not the tokens that happen to be present.
    """
    tokens = [str(a) for a in (argv or [])]
    _mlock_now, _reserves_now = resolve_effective_memory_state(tokens, env)
    drop_shadowed_mlock = _reserves_now and not _mlock_now
    overridden: list[str] = []
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        flag = _flag_name(token)
        if flag in _RAM_RESERVING_FLAGS:
            overridden.append(token)
            i += 1
            continue
        if drop_shadowed_mlock and flag in _MLOCK_FLAGS:
            overridden.append(token)
            i += 1
            continue
        if flag in _LOAD_MODE_FLAGS:
            if "=" in token:
                value, step = token.split("=", 1)[1], 1
            elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
                value, step = tokens[i + 1], 2
            else:
                value, step = "", 1
            normalized = value.strip().lower()
            rewrite_mode, replacement = _pageable_mode_replacement(normalized, drop_shadowed_mlock)
            if rewrite_mode:
                overridden.append(" ".join(tokens[i : i + step]))
                if replacement is not None:
                    out.extend([tokens[i].split("=", 1)[0], replacement])
                i += step
                continue
            out.extend(tokens[i : i + step])
            i += step
            continue
        out.append(token)
        i += 1
    if env is not None:
        for name in MEMORY_ENV_VARS:
            if name not in env:
                continue
            rewrite, new_value = _pageable_env_value(name, str(env[name]), drop_shadowed_mlock)
            if not rewrite:
                continue
            if new_value is None:
                env.pop(name, None)
            else:
                env[name] = new_value
            overridden.append(name)
    return out, overridden


# Mirrors llama_cpp's _LLAMA_ARG_TRUE/FALSE_VALUES; duplicated to avoid an import cycle.
_ENV_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_ENV_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})

# Mirrored by LOAD_MODES in per-model-config.ts.
_LOAD_MODE_VALUES = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})
# Pre-enum spellings; plain mmap and dio have none, so they are skipped.
_LEGACY_LOAD_MODE_FLAGS: dict[str, list[str]] = {
    "none": ["--no-mmap"],
    "mlock": ["--no-mmap", "--mlock"],
    "mmap+mlock": ["--mlock"],
}
_LOAD_MODE_MLOCK_VALUES = frozenset({"mlock", "mmap+mlock"})
_LOAD_MODE_RESERVING_VALUES = frozenset({"none", "mlock"})


def resolve_effective_memory_state(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[bool, bool]:
    """``(mlock, reserves_ram)`` the child will actually run with.

    Mirrors llama.cpp: env supplies defaults, argv overrides last-wins. Used to
    compare a running process against the current settings, so the reload hint
    reflects the launched state rather than only what Unsloth emitted.
    """
    mlock, reserves_ram, _direct_io = resolve_effective_load_state(argv, env)
    return mlock, reserves_ram


def resolve_effective_direct_io(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the child will actually run DirectIO.

    ``mmap`` and ``dio`` both hold no full host copy, so they are the same
    ``(mlock, reserves_ram)`` and the reload comparator cannot tell them apart.
    Where the Windows no-reserve policy owes a launch dio, that difference is
    the whole decision, so it is resolved from the same parse rather than a
    second one that could drift from it.
    """
    return resolve_effective_load_state(argv, env)[2]


def resolve_effective_load_state(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[bool, bool, bool]:
    """``(mlock, reserves_ram, direct_io)``. Every branch that resolves the mode
    assigns all three, so the DirectIO bit cannot fall out of step with the pair."""
    env = env or {}
    mlock = False
    reserves_ram = False
    direct_io = False
    # Env vars run the flag's handler in registration order, so a later one overwrites the mode.
    if str(env.get("LLAMA_ARG_MLOCK", "")).strip().lower() in _ENV_TRUE_VALUES:
        mlock, direct_io = True, False
    # LLAMA_ARG_NO_<NAME> present forces false whatever the value, and beats the affirmative var.
    _mmap_env = "0" if "LLAMA_ARG_NO_MMAP" in env else str(env.get("LLAMA_ARG_MMAP", ""))
    _mmap_env = _mmap_env.strip().lower()
    if _mmap_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram, direct_io = False, False, False
    elif _mmap_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram, direct_io = False, True, False
    _dio_env = "0" if "LLAMA_ARG_NO_DIO" in env else str(env.get("LLAMA_ARG_DIO", ""))
    _dio_env = _dio_env.strip().lower()
    if _dio_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram, direct_io = False, False, True
    elif _dio_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram, direct_io = False, True, False
    _mode_env = str(env.get("LLAMA_ARG_LOAD_MODE", "")).strip().lower()
    if _mode_env:
        mlock = _mode_env in _LOAD_MODE_MLOCK_VALUES
        reserves_ram = _mode_env in _LOAD_MODE_RESERVING_VALUES
        direct_io = _mode_env == "dio"

    tokens = [str(a) for a in (argv or [])]
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        flag = _flag_name(tok)
        if flag is None:
            i += 1
            continue
        if flag in _MLOCK_FLAGS:
            mlock, direct_io = True, False
            i += 1
        elif flag in _NO_MMAP_FLAGS:
            # --no-mmap resets the whole mode, clearing an earlier --mlock.
            mlock = False
            reserves_ram = True
            direct_io = False
            i += 1
        elif flag in _DIO_ON_FLAGS:
            mlock = False
            reserves_ram = False
            direct_io = True
            i += 1
        elif flag in _DIO_OFF_FLAGS:
            # Not plain mmap: upstream maps these to mode `none` (full host buffer).
            mlock = False
            reserves_ram = True
            direct_io = False
            i += 1
        elif flag == "--mmap":
            mlock = False
            reserves_ram = False
            direct_io = False
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
                direct_io = value == "dio"
            i += step
        else:
            i += 1
    return mlock, reserves_ram, direct_io


def memory_state_satisfies_settings(
    state: Optional[tuple[bool, bool]],
    policy_active: bool = False,
    mlock_applicable: bool = True,
    direct_io: Optional[bool] = None,
    dio_applicable: bool = False,
    dio_managed: bool = False,
) -> bool:
    """True when a launched ``(mlock, reserves_ram)`` matches the settings.

    Shared by the duplicate-load comparator (so toggling a setting forces a real relaunch instead of
    returning already-loaded) and the settings route (so the reload hint agrees with it).

    ``state`` is None for a process this policy does not govern, such as the diffusion runner, which
    has no load-mode of its own; nothing about it can contradict the settings, so it always matches.

    ``policy_active`` says the launch differed from an unmanaged one, because a flag was emitted, a
    requested one suppressed, or an inherited env var scrubbed. With both toggles off the policy no
    longer applies, so any of those has to be undone on the next launch, while a launch it never
    touched is left alone.

    ``mlock_applicable`` is False when the launch is fully offloaded to a
    discrete GPU, where page-locking host RAM buys nothing and is deliberately
    not emitted. Residency there is the idle-unload veto, which needs no
    relaunch, so demanding mlock would ask for a reload that can never satisfy
    the check.

    ``direct_io`` is whether the launch actually streams, and ``dio_applicable``
    whether ``no_reserve_requires_dio`` held for its platform, build and
    placement. Default mmap and DirectIO are the same ``(mlock, reserves_ram)``,
    so without these a Windows full-offload process launched on the default
    mapping reports as satisfying a no-reserve that would now emit dio, and both
    the reload hint and the duplicate-load fast path leave it running. None means
    a caller that does not track it, which never forces a reload.
    """
    if state is None:
        return True
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return True
    mlock, reserves_ram = state
    if get_no_ram_reserve():
        # mlock_applicable only excuses a MISSING lock; a live reservation still has to go, wherever the weights are.
        if mlock or reserves_ram:
            return False
        return not (dio_applicable and direct_io is False)
    if get_keep_resident():
        # Managed dio must go when no-reserve does; `dio_managed` not `policy_active` (env scrub sets it).
        if direct_io and dio_applicable and dio_managed:
            return False
        return mlock or not mlock_applicable
    return not policy_active
