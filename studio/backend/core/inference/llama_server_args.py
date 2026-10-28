# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Rejects only Unsloth-managed flags (model, auth, network, slots); user flags come last and win."""

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


# --batch-size / --ubatch-size range, mirrored by N_BATCH_MIN/MAX in per-model-config.ts
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
# Keep rollback available; zero forces a full prompt re-ingest after divergence.
CTX_CHECKPOINTS_MIN_USEFUL = 2

# Single source of slot-count aliases for the denial, its hint and the single-sequence retry.
_PARALLEL_FLAGS: frozenset[str] = frozenset({"-np", "--parallel", "--n-parallel"})

# Each group = every alias of one denied flag; extend when llama.cpp adds an alias.
_DENYLIST_GROUPS: tuple[frozenset[str], ...] = (
    # Parallel slots are owned by --parallel and n_parallel; a pass-through desyncs slot bookkeeping.
    _PARALLEL_FLAGS,
    # Model identity comes from LoadRequest; a second -m would load a model Unsloth did not record.
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
    # Networking: Unsloth binds + proxies; retargeting orphans the proxy.
    frozenset({"--host"}),
    frozenset({"--port"}),
    frozenset({"--path"}),
    frozenset({"--api-prefix"}),
    frozenset({"--reuse-port"}),
    # Auth / TLS: Unsloth terminates auth; upstream --api-key / TLS shadows Unsloth's key and breaks the proxy hop
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
    # Server-mode flips: --embedding is set from the GGUF pooling type at load, not by hand.
    frozenset({"--embedding", "--embeddings"}),
    frozenset({"--rerank", "--reranking"}),
    # Pooling decides whether the embedding launch is safe; an override could pick NONE/RANK.
    frozenset({"--pooling"}),
    # llama-server's --tools would silently stack on Unsloth's --enable-tools / --disable-tools policy.
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
    # Serves local files over the child's HTTP surface.
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
        # llama-server runs <dir>/ffmpeg to decode a video.
        "--video-ffmpeg-dir",
        # Not a path: llama-server sends the model's tensors to these hosts.
        "--rpc",
        # Not a path either: llama-server downloads these Hub drafters with the installation's HF token.
        "-hfd",
        "-hfrd",
        "--hf-repo-draft",
        "--spec-draft-hf",
        "-hffd",
        "--hf-file-draft",
    }
)

# Built-in presets (--*-default, older --fim-qwen-7b-spec) fetch a Hub repo with the HF token; --spec-default stays open.
_PRESET_FLAG_RE = re.compile(r"^--[a-z0-9.-]+-default$|^--fim-[a-z0-9.-]+-spec$")
_NON_PRESET_DEFAULT_FLAGS: frozenset[str] = frozenset({"--spec-default"})


def _is_preset_flag(flag: Optional[str]) -> bool:
    return (
        flag is not None
        and flag not in _NON_PRESET_DEFAULT_FLAGS
        and _PRESET_FLAG_RE.match(flag.lower()) is not None
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
        elif _is_preset_flag(flag):
            found.append((flag, ""))
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

# Shape bounds, not security (the denylist is); fail here naming the limit rather than at execve.
MAX_EXTRA_ARG_TOKENS = 256
MAX_EXTRA_ARGS_BYTES = 32 * 1024
# CreateProcess caps the whole command line at 32767 chars; leave room for Unsloth's own args.
MAX_EXTRA_ARGS_BYTES_WINDOWS = 24 * 1024


# CreateProcess takes one command string; extras must fit the limit minus this reserve.
WINDOWS_COMMAND_LIMIT = 32767
WINDOWS_COMMAND_RESERVE = 8192


def windows_command_length(args: list) -> int:
    """Windows command-line length as list2cmdline builds it; escapes make a plain sum wrong."""
    import subprocess
    return len(subprocess.list2cmdline([str(a) for a in args]))


def max_extra_args_bytes() -> int:
    """The size cap for this platform."""
    return MAX_EXTRA_ARGS_BYTES_WINDOWS if sys.platform == "win32" else MAX_EXTRA_ARGS_BYTES


def _flag_name(token: str) -> Optional[str]:
    """Flag name for a token, or None; folds --key=value, underscores and attached -np8 to a base name."""
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
    """True only for '=' or an attached short value; a name change alone (--ctx_size) does not count."""
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
    # Count of following tokens the last flag may claim as values; a switch claims none.
    pending_values = 0
    # Values still owed to a two-value flag, tracked apart because it is the one arity this module knows for certain
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
            # Users capping predictions with --parallel 1 get no pointer to n_parallel, so name the knob here.
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
            # _flag_name strips before lookup, so a quoted "--top-k " passed the denylist
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
            # Its own value when attached, otherwise the tokens that follow.
            attached = _value_is_attached(token, flag)
            if pending_two_value > 0:
                raise ValueError(f"llama-server flag '{two_value_flag}' takes two values")
            # An attached value is only ONE of the two; START= still owes END.
            if flag in _TWO_VALUE_FLAGS:
                pending_values = 1 if attached else 2
                pending_two_value = pending_values
            elif flag in _OPTIONAL_SECOND_VALUE_FLAGS:
                # Allowed, not owed: pending_two_value stays 0, so nothing insists on the second token
                pending_values = 1 if attached else 2
                pending_two_value = 0
            else:
                pending_values = 0 if attached else 1
                pending_two_value = 0
            two_value_flag = flag
        out.append(token)
    if pending_two_value > 0:
        # Only START without END is checkable; an ordinary flag's arity is unknown, so llama-server decides.
        raise ValueError(f"llama-server flag '{two_value_flag}' takes two values")
    if sys.platform == "win32":
        # After the per-token walk, because this is a property of the whole list.
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
    """Drops managed flags from carried-over args together with their values, so none is orphaned."""
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
            # Padded flags are refused with their value, else the value becomes a bare positional arg.
            dropped.append(flag)
            skip_next = _takes_next(index, token, flag)
            continue
        if (
            flag is not None
            and _takes_next(index, token, flag)
            and (_has_control_characters(tokens[index + 1]) or not _is_spawnable(tokens[index + 1]))
        ):
            # Log dropped flags, not just skip them; a silently vanished flag is harder to explain later.
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
# inherited copies of these shadow n_batch / n_ubatch, stripped only when the field is set
_BATCH_FLAGS: frozenset[str] = frozenset({"-b", "--batch-size"})
_UBATCH_FLAGS: frozenset[str] = frozenset({"-ub", "--ubatch-size"})
# Stripped only when its field is set; --swa-checkpoints is the old name of --ctx-checkpoints.
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
# The fitter's context floor, also never stripped; see fit_ctx_in.
_FIT_CTX_FLAGS: frozenset[str] = frozenset({"-fitc", "--fit-ctx"})
_LAYER_OFFLOAD_FLAGS: frozenset[str] = _GPU_LAYER_FLAGS | _FIT_FLAGS
_MOE_OFFLOAD_FLAGS: frozenset[str] = frozenset({"-ncmoe", "--n-cpu-moe", "-cmoe", "--cpu-moe"})
_OFFLOAD_SHADOWING_FLAGS: frozenset[str] = _LAYER_OFFLOAD_FLAGS | _MOE_OFFLOAD_FLAGS

# Full-model RAM reservations; stripped only when a Model Memory toggle vetoes them.
_MLOCK_FLAGS: frozenset[str] = frozenset({"--mlock", "-mlock"})
# Modern spelling of both, as an enum value. Takes a value, so NOT boolean.
_LOAD_MODE_FLAGS: frozenset[str] = frozenset({"--load-mode", "-lm"})
_NO_MMAP_FLAGS: frozenset[str] = frozenset({"--no-mmap", "-no-mmap"})
# Deprecated selectors reset the WHOLE mode (dropping mlock). Negative spellings map to `none`,
# so no-reserve must veto those too.
_DIO_ON_FLAGS: frozenset[str] = frozenset({"--direct-io", "-dio"})
_DIO_OFF_FLAGS: frozenset[str] = frozenset({"--no-direct-io", "-ndio"})
_DIO_FLAGS: frozenset[str] = _DIO_ON_FLAGS | _DIO_OFF_FLAGS
_LOAD_MODE_ALIAS_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | frozenset({"--mmap"}) | _DIO_FLAGS
# Every spelling that asks for a full-model host buffer.
_RAM_RESERVING_FLAGS: frozenset[str] = _NO_MMAP_FLAGS | _DIO_OFF_FLAGS
# llama.cpp reads these before argv, so inherited values survive token stripping.
MEMORY_ENV_VARS: tuple[str, ...] = (
    "LLAMA_ARG_MLOCK",
    "LLAMA_ARG_MMAP",
    "LLAMA_ARG_LOAD_MODE",
    "LLAMA_ARG_DIO",
    # Legacy negative aliases, honoured by PRESENCE whatever the value.
    "LLAMA_ARG_NO_MMAP",
    "LLAMA_ARG_NO_DIO",
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
    """Last user --ctx-checkpoints value, or None; extras win over the field, so sizing prices that."""
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
    """Caps per-slot checkpoints to the host budget and the default; unknown sizes keep the default."""
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
    """Child count: extras, then field, then env LLAMA_ARG_CTX_CHECKPOINTS, then the budget."""
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
    """True when a pass-through -c/--ctx-size equals n_ctx, which is a choice to run past the VRAM fit."""
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
    """Last-wins cache type from -ctk or -ctv; both are one setting, as the KV estimate has one knob."""
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
    """Last user-supplied -ngl value from extras, read before Manual mode strips the flags."""
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
    """Rounds to float32 as llama.cpp holds it; overflow becomes inf, not an OverflowError."""
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
    """Last -ts ratios, checked against llama.cpp's float32 limits; bad shares raise, giving a 400."""
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
    """Raises when a pass-through --batch-size is below 2 or below --parallel; llama-server dies at startup after the old model unloads; unreadable left to it."""
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
    """Only a truthy last --fit in extras enables the fitter; absent or unreadable values do not."""
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    return raw_value is not None and raw_value.strip().lower() in _ENV_TRUE_VALUES


def fit_is_effectively_on(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the child runs the fitter; llama.cpp defaults it on and only an explicit off stops it."""
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT")
    if raw_value is None:
        return True
    return str(raw_value).strip().lower() not in _ENV_FALSE_VALUES


def fit_ctx_in(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Negative values are kept: llama.cpp stores them unsigned, where they disable context reduction."""
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
    """Largest --fit-target margin: the fit charges one margin to every credited device; None if unset."""
    raw_value = _last_flag_value(args, _FIT_TARGET_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT_TARGET")
    if raw_value is None:
        return None
    values: list[float] = []
    # Upstream splits on comma and slash (common/arg.cpp), so 4096/4096 is two devices, not one token that would wrongly abstain.
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
    """True when --split-mode none or a short --tensor-split leaves fewer devices than credited; a restatement keeping all devices is not an override."""
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
    """An explicit --split-mode in extras overrides the toggle; tensor-parallel is on only for tensor."""
    override = parse_split_mode_override(args)
    if override is None:
        return fallback_tensor_parallel
    return override.strip().lower() == "tensor"


def _env_split_mode_is_tensor(env: Optional[Mapping[str, str]] = None) -> bool:
    """An inherited LLAMA_ARG_SPLIT_MODE=tensor is budgeted as tensor, or the child runs it unbudgeted."""
    raw = (os.environ if env is None else env).get("LLAMA_ARG_SPLIT_MODE")
    return bool(raw) and raw.strip().lower() == "tensor"


def _effective_tensor_parallel(
    extra_args: Optional[Iterable[str]],
    tensor_parallel: bool,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """Tensor decision with inherited LLAMA_ARG_SPLIT_MODE=tensor when extras set no mode; shared with the fallback so an env-only tensor crash still retries."""
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
    """An inherited tensor env matches only a server that really launched tensor, so no reload loop."""
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


def _extra_args_positive_int(
    args: Optional[Iterable[str]], flag: str, env_name: str, env: Optional[Mapping[str, str]]
) -> Optional[int]:
    found: Optional[int] = None
    raw_env = (os.environ if env is None else env).get(env_name)
    if raw_env:
        try:
            parsed_env = int(str(raw_env).strip())
        except (TypeError, ValueError):
            parsed_env = 0
        if parsed_env > 0:
            found = parsed_env
    tokens = [str(a) for a in (args or ())]
    for index, raw in enumerate(tokens):
        if raw.startswith(flag + "="):
            value = raw.partition("=")[2]
        elif raw == flag and index + 1 < len(tokens):
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


def extra_args_image_max_tokens(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Return the effective positive ``--image-max-tokens`` value."""
    return _extra_args_positive_int(args, "--image-max-tokens", "LLAMA_ARG_IMAGE_MAX_TOKENS", env)


def extra_args_image_min_tokens(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[int]:
    """Return the effective positive ``--image-min-tokens`` value."""
    return _extra_args_positive_int(args, "--image-min-tokens", "LLAMA_ARG_IMAGE_MIN_TOKENS", env)


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
    """Drops inherited flags shadowing first-class settings so an old -c cannot override max_seq_length; --tensor-split can be stripped alone."""
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
        # Drop the flag; also consume the next token unless it's boolean, already inline (`-c=4096`)
        if flag in _BOOLEAN_SHADOWING_FLAGS or "=" in tok:
            i += 1
        elif i + 1 < n and _flag_name(tokens[i + 1]) is None:
            i += 2
        else:
            i += 1
    return out


def strip_split_mode_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Removes --split-mode / -sm and coupled --tensor-split / -ts; None or empty input stays unchanged so inherit vs explicit-empty survives."""
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
    """Removes -c/--ctx-size only; None/empty input stays unchanged so inherit vs explicit-empty survives; trailing -c 0 breaks the Metal floor."""
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


# Named managed-block tokens let the launch remove them once their placement stops holding.
MANAGED_DIO_FLAGS: tuple[str, ...] = ("--load-mode", "dio")


def no_reserve_requires_dio(
    *, supports_load_mode: Optional[bool], gpu_offload_confirmed: bool
) -> bool:
    """Windows keeps a full offload mapping resident, so DirectIO is owed only on confirmed offload and on a build supporting --load-mode."""
    return sys.platform == "win32" and bool(supports_load_mode) and gpu_offload_confirmed


def load_mode_support(server_caps: Mapping[str, object]) -> Optional[bool]:
    """--load-mode support from a --help probe; None when the probe failed or listed nothing."""
    if server_caps.get("supports_load_mode"):
        return True
    if server_caps.get("help_probe_ok") and server_caps.get("flags"):
        return False
    return None


def keep_resident_lock_flags(supports_load_mode: Optional[bool]) -> list[str]:
    """Page-lock spelling for this build; none when unknown, since newer builds exit on --mlock."""
    if supports_load_mode is None:
        return []
    # mmap+mlock, not bare mlock, matches what --mlock meant with default mmap.
    return ["--load-mode", "mmap+mlock"] if supports_load_mode else ["--mlock"]


def keep_resident_lock_applicable(host_resident: bool, supports_load_mode: Optional[bool]) -> bool:
    """Whether a missing lock is owed; an unknown build gets none, so a reload must not ask for one."""
    return bool(host_resident) and supports_load_mode is not None


def resolve_launch_load_mode(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: Optional[bool],
    weights_in_host_memory: bool,
    gpu_offload_confirmed: bool,
    requested_load_mode: Optional[str],
    env: Optional[Mapping[str, str]],
    settings: tuple[bool, bool],
) -> tuple[bool, bool]:
    """Runs the real policy chain, not a hand-built argv; effective_dio is what the child runs, policy_emitted_dio whether the policy added it."""
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
    supports_load_mode: Optional[bool] = False,
    weights_in_host_memory: bool = True,
    gpu_offload_confirmed: bool = False,
    env: Optional[Mapping[str, str]] = None,
    settings: Optional[tuple[bool, bool]] = None,
) -> tuple[list[str], list[str]]:
    """Mlock only when weights sit in host RAM; offload must be positively confirmed, never inferred; pass the same settings pair down, never re-read."""
    if settings is None:
        try:
            from utils.model_memory_settings import get_model_memory_settings
        except Exception:
            # Settings unavailable (bare unit-test import): behave as before.
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
        # An inherited non-reserving choice survives the scrub; argv beats env, so the pair stands aside.
        and not memory_env_selects_load_mode(env)
    ):
        # On confirmed full offload, Windows streams via dio since the GGUF cannot be partially unmapped; per-model modes still win later.
        managed.extend(MANAGED_DIO_FLAGS)
    lock = keep_resident_lock_flags(supports_load_mode)
    if keep_resident and not no_ram_reserve and weights_in_host_memory and lock:
        managed.extend(lock)
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
    supports_load_mode: Optional[bool] = False,
    weights_in_host_memory: bool = True,
    requested_load_mode: Optional[str] = None,
    settings: Optional[tuple[bool, bool]] = None,
) -> tuple[list[str], list[str]]:
    """Model Memory settings win, so run after apply_model_memory_policy; auto emits nothing; unknown values dropped since llama-server exits on them."""
    tokens = list(extra_args or [])
    mode = _normalize_load_mode_value(requested_load_mode)
    if not mode:
        return [], tokens
    if settings is None:
        try:
            from utils.model_memory_settings import get_model_memory_settings
            settings = get_model_memory_settings()
        except Exception:
            # Settings unavailable (bare unit-test import): nothing to defer to.
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
    if supports_load_mode is None:
        # Newer builds exit on the legacy flags, so an unknown build gets none.
        logger.info(
            "llama-server --help probe was inconclusive; skipping the requested %r mode.", mode
        )
        return [], tokens
    if not supports_load_mode:
        # A build predating the enum understands the spellings it replaced, and only for the values that had one.
        legacy = _LEGACY_LOAD_MODE_FLAGS.get(mode)
        if not legacy:
            logger.info("llama-server has no --load-mode; skipping the requested %r mode.", mode)
            return [], tokens
        return list(legacy), tokens
    # Emitted before the extras and strips nothing, so a typed flag wins last-wins; the route drops any inherited copy first.
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
    """Drops only the --load-mode values that reserve host RAM; mmap and dio hold no full copy and stay."""
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
    """True when either toggle is on, so the child env is scrubbed; a given pair replaces the live one."""
    if settings is not None:
        return settings[0] or settings[1]
    try:
        from utils.model_memory_settings import get_keep_resident, get_no_ram_reserve
    except Exception:
        return False
    return get_keep_resident() or get_no_ram_reserve()


def _env_var_locks_or_reserves(name: str, value: str) -> bool:
    """Mirrors the argv rule: settings own the reservation, not the loader, so inherited dio survives."""
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        return normalized in _ENV_TRUE_VALUES
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        return True
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        # Falsy selects "none"; truthy selects mmap / dio, neither of which holds a full copy.
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
    # API prefix moves every endpoint, /health included, so an inherited one turns every load into a timeout.
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
    # Web UI and MCP proxy (upstream: not for untrusted use) and the directory the child serves.
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
    """Any user --load-mode or legacy flag counts by presence, so a fit-derived mode stands aside."""
    for raw in extra_args or ():
        if _flag_name(str(raw)) in _LOAD_MODE_FLAGS | _LOAD_MODE_ALIAS_FLAGS:
            return True
    return False


def memory_env_selects_load_mode(env: Optional[Mapping[str, str]]) -> bool:
    """Inherited LLAMA_ARG_* env picks a mode a fit-derived mode must yield to: negatives count by presence, MLOCK only if truthy."""
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
    """Removes only inherited vars that lock or reserve RAM; no-op with both toggles off; LLAMA_ARG_DIO=1 is a loader and stays."""
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
        # Falsy selects "none"; truthy selects mmap / dio, neither of which holds a full copy.
        return normalized in _ENV_FALSE_VALUES, None
    if name == "LLAMA_ARG_LOAD_MODE":
        return _pageable_mode_replacement(normalized, drop_shadowed_mlock)
    return False, None


def force_pageable_load(
    argv: Optional[Iterable[str]], env: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    """Rewrites none/mlock to pageable: effective locks become mmap+mlock, shadowed locks stay dropped; decided by pre-rewrite state, not tokens."""
    tokens = [str(a) for a in (argv or [])]
    # Only RAM-reserving launches are rewritten; pageable dio or mmap keeps every token, mlock included.
    _mlock_now, _reserves_now = resolve_effective_memory_state(tokens, env)
    drop_shadowed_mlock = _reserves_now and not _mlock_now
    overridden: list[str] = []
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        flag = _flag_name(token)
        # --no-mmap / --no-direct-io are deprecated spellings of none; valueless, so drop just the token.
        if flag in _RAM_RESERVING_FLAGS:
            overridden.append(token)
            i += 1
            continue
        # A lock a later selector already cleared; named in overridden so the log describes the rewrite.
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
            # After --no-mmap, mlock is unlocked, so mmap+mlock would add a lock; mmap+mlock --no-mmap locks.
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

# Every --load-mode value llama-server documents; unknown ones dropped since llama-server exits on them. Mirrored by LOAD_MODES in per-model-config.ts.
_LOAD_MODE_VALUES = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})
# Pre-enum spellings; plain mmap and dio have none, so they are skipped.
_LEGACY_LOAD_MODE_FLAGS: dict[str, list[str]] = {
    "none": ["--no-mmap"],
    "mlock": ["--no-mmap", "--mlock"],
    "mmap+mlock": ["--mlock"],
}
_LOAD_MODE_MLOCK_VALUES = frozenset({"mlock", "mmap+mlock"})
# Modes reading weights into a full host buffer; dio streams and mmap maps, so neither reserves RAM.
_LOAD_MODE_RESERVING_VALUES = frozenset({"none", "mlock"})


def resolve_effective_memory_state(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> tuple[bool, bool]:
    """What the child runs with: env supplies defaults, argv overrides last-wins, as llama.cpp does."""
    mlock, reserves_ram, _direct_io = resolve_effective_load_state(argv, env)
    return mlock, reserves_ram


def resolve_effective_direct_io(
    argv: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the child streams via DirectIO: mmap and dio have the same (mlock, reserves_ram) pair."""
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
        # Either mode it can mean maps or buffers the weights; neither streams.
        mlock, direct_io = True, False
    # LLAMA_ARG_NO_<NAME> present forces false whatever the value, and beats the affirmative var.
    _mmap_env = "0" if "LLAMA_ARG_NO_MMAP" in env else str(env.get("LLAMA_ARG_MMAP", ""))
    _mmap_env = _mmap_env.strip().lower()
    if _mmap_env in _ENV_TRUE_VALUES:
        mlock, reserves_ram, direct_io = False, False, False
    elif _mmap_env in _ENV_FALSE_VALUES:
        mlock, reserves_ram, direct_io = False, True, False
    # LLAMA_ARG_DIO likewise: on selects DirectIO, off selects "none".
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
            # Only the mlock bit matters: mlock and mmap+mlock both replace an earlier DirectIO mode.
            mlock, direct_io = True, False
            i += 1
        elif flag in _NO_MMAP_FLAGS:
            # --no-mmap resets the whole mode, clearing an earlier --mlock.
            mlock = False
            reserves_ram = True
            direct_io = False
            i += 1
        elif flag in _DIO_ON_FLAGS:
            # Deprecated load-mode selector resets the mode, dropping mlock; DirectIO holds no full host copy.
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
    """Launch satisfies settings; mlock not needed on full GPU offload; Windows full offload needs dio reload; None never forces a reload."""
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


_NGRAM_MOD_LEGACY_FLAGS = {
    "--spec-ngram-mod-n-match": "--spec-ngram-size-n",
    "--spec-ngram-mod-n-min": "--draft-min",
    "--spec-ngram-mod-n-max": "--draft-max",
}


def translate_ngram_mod_args(
    args: Iterable[str],
    flavor: Optional[str],
    *,
    chain_with_mtp: bool = False,
) -> list[str]:
    """Respell ``--spec-ngram-mod-n-*`` for a legacy build (else it refuses to start); with MTP drop min/max (MTP owns the range)."""
    args = [str(a) for a in args]
    if flavor != "legacy":
        return args
    out: list[str] = []
    i = 0
    while i < len(args):
        _flag, eq, value = args[i].partition("=")
        legacy = _NGRAM_MOD_LEGACY_FLAGS.get(_flag_name(args[i]) or "")
        # A following flag is not this flag's value; consuming it would orphan that flag's own value.
        if legacy is None or (
            not eq and (i + 1 >= len(args) or _flag_name(args[i + 1]) is not None)
        ):
            out.append(args[i])
            i += 1
            continue
        if not eq:
            value = args[i + 1]
        i += 1 if eq else 2
        if chain_with_mtp and legacy != "--spec-ngram-size-n":
            continue
        out.extend([legacy, value])
    return out
