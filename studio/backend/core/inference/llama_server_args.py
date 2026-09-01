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

import logging
import os
import sys
from typing import Any, Iterable, Mapping, Optional

logger = logging.getLogger(__name__)

# Valid llama-server --parallel range, shared with LoadRequest.n_parallel.
# Mirrored by callers that cannot import this: run.py and unsloth_cli/commands/
# studio.py (_PARALLEL_MIN/MAX), per-model-config.ts (N_PARALLEL_MIN/MAX);
# test_parallel_slots_per_load.py pins them together.
PARALLEL_MIN = 1
PARALLEL_MAX = 64

# --batch-size / --ubatch-size range, mirrored by N_BATCH_MIN/MAX in per-model-config.ts
BATCH_MIN = 1
BATCH_MAX = 65536

# Sanity bounds, not upstream ones: a stray keystroke fails here rather than in the
# child. --cache-ram floors at -1 ("no limit"); 0 disables the cache. Mirrored by
# CTX_CHECKPOINTS_MAX / CACHE_RAM_MAX in per-model-config.ts.
CTX_CHECKPOINTS_MAX = 256
CACHE_RAM_MAX_MIB = 1024 * 1024

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
    # Pooling decides whether the managed embedding launch is safe. A pass-through
    # override appended after --embedding could switch it to NONE or RANK.
    frozenset({"--pooling"}),
    # llama-server's own built-in tools flag would silently stack on top of
    # Unsloth's --enable-tools / --disable-tools policy resolver.
    frozenset({"--tools"}),
    # --agent is --tools by another name: upstream documents it as "enable CORS
    # proxy and ALL built-in tools", and that set includes exec_shell_command.
    # Denying --tools while allowing this left the same capability one alias away.
    frozenset({"-ag", "--agent", "-no-ag", "--no-agent"}),
    # Where those tools run: docker:/podman: spins up a container, ssh:<target>
    # runs them on another host entirely.
    frozenset({"--tools-runtime"}),
    # MCP servers are tools from a config file or an inline JSON blob; upstream
    # says "do not enable in untrusted environments" for both.
    frozenset({"--mcp-servers-config"}),
    frozenset({"--mcp-servers-json"}),
    # CORS: Unsloth terminates browser access at its own origin, so widening the
    # child's would hand a page past the boundary the proxy exists to hold.
    frozenset({"--cors-origins"}),
    frozenset({"--cors-headers"}),
    frozenset({"--cors-methods"}),
    frozenset({"--cors-credentials", "--no-cors-credentials"}),
    # Serves local files over the child's HTTP surface.
    frozenset({"--media-path"}),
    # Startup output is how _classify_llama_start_failure tells a bad GGUF from an
    # OOM from a rejected flag; redirecting or silencing it makes every failure
    # the same opaque one.
    frozenset({"--log-file"}),
    frozenset({"--log-disable"}),
    # Slot-state dir: Unsloth owns it for KV persistence across idle unload. Endpoint
    # exposure (--slots, --props) is deliberately NOT denied alongside it: Unsloth
    # reads GET /props and never /slots, so either is the user's own call.
    frozenset({"--slot-save-path"}),
    # These print and exit instead of serving, so the load would "succeed" with no
    # server behind it and only time out later.
    frozenset({"-h", "--help", "--usage"}),
    frozenset({"--version"}),
    frozenset({"--list-devices"}),
    frozenset({"-cl", "--cache-list"}),
    frozenset({"--completion-bash"}),
)

_DENYLIST: frozenset[str] = frozenset().union(*_DENYLIST_GROUPS)

# Flags that take TWO values rather than one. Scanned out of `llama-server --help`:
# every other option is `--flag VALUE` or a switch, and this list exists so the
# positional check below does not refuse a legitimate second value.
_TWO_VALUE_FLAGS: frozenset[str] = frozenset({"--control-vector-layer-range"})

# Flags that take a second value on SOME builds. Today's llama.cpp writes the scale
# into the value ("--lora-scaled FNAME:SCALE"), and older ones took it as a separate
# token ("--lora-scaled FNAME SCALE"); both spellings are already handled in
# _sidecar_weight_files. So the second token is allowed but never required: demanding
# it would refuse the current syntax, and refusing it broke a list that loaded before
# the positional check existed.
_OPTIONAL_SECOND_VALUE_FLAGS: frozenset[str] = frozenset(
    {"--lora-scaled", "--control-vector-scaled"}
)

# Shape bounds. Not a security boundary -- the denylist is -- but a pasted file or a
# runaway generator should fail here, naming the limit, rather than at execve or in
# llama-server's own parser. Generous enough that a grammar or a JSON schema fits.
MAX_EXTRA_ARG_TOKENS = 256
MAX_EXTRA_ARGS_BYTES = 32 * 1024
# Windows passes CreateProcess ONE string for the whole command line, capped at 32767
# characters, and the model path, Unsloth's own flags and the quoting subprocess adds
# all come out of the same budget. So the extras get a smaller share there: accepting
# the full 32 KiB would pass every check here and then fail inside Popen, after the
# load had already begun switching models.
MAX_EXTRA_ARGS_BYTES_WINDOWS = 24 * 1024


# CreateProcess takes the whole command line as ONE string, capped here. The rest of
# the command (the binary, the model path, Unsloth's own flags) has to fit too, so the
# extras are checked against the limit minus this reserve.
WINDOWS_COMMAND_LIMIT = 32767
WINDOWS_COMMAND_RESERVE = 8192


def windows_command_length(args: list) -> int:
    """Characters ``subprocess`` would put on a Windows command line for ``args``.

    list2cmdline is the exact serializer Popen uses there, and it is not a sum of
    lengths: a value needing quotes has its backslashes doubled, so an escape-heavy
    grammar can nearly double. Measuring it is the only honest check.
    """
    import subprocess
    return len(subprocess.list2cmdline([str(a) for a in args]))


def max_extra_args_bytes() -> int:
    """The size cap for this platform."""
    return MAX_EXTRA_ARGS_BYTES_WINDOWS if sys.platform == "win32" else MAX_EXTRA_ARGS_BYTES


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


def _value_is_attached(token: str, flag: str) -> bool:
    """Whether this token carries its own value, rather than expecting the next one.

    Not "the name changed": _flag_name also folds llama.cpp's underscore spelling
    (--ctx_size is --ctx-size to it, and the binary takes both), so comparing the
    normalised name against the raw token read "--ctx_size 4096" as attached and then
    refused the 4096 as a bare value. Only "=" and an attached short like -np8 are
    values in the same token.
    """
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
    """Validate user-supplied llama-server args. Returns a flat list ready to
    extend the llama-server command; raises ``ValueError`` naming the
    offending flag on the first managed token."""
    if not args:
        return []
    out: list[str] = []
    total_bytes = 0
    # How many following tokens the flag just seen may still claim as values. A
    # switch claims none, so the next bare token has no owner.
    pending_values = 0
    # Values still owed to a two-value flag, tracked apart because it is the one
    # arity this module knows for certain.
    pending_two_value = 0
    two_value_flag = ""
    for raw in args:
        token = str(raw)
        if len(out) >= MAX_EXTRA_ARG_TOKENS:
            raise ValueError(
                f"too many extra llama-server args (limit {MAX_EXTRA_ARG_TOKENS} tokens)"
            )
        # A grammar or JSON schema is a legitimately long single token, so the cap
        # is on the whole list rather than per token.
        # Strictly, unlike the sizing below: JSON and the browser can both carry an
        # unpaired surrogate, which survives every check here and then makes
        # subprocess.Popen raise while it encodes argv, long after the load has begun
        # switching models. Refused at the boundary, where it is still a 400.
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
        # execve rejects a NUL outright; the rest would reach the child's parser as
        # invisible characters and be blamed on the flag they are attached to.
        if _has_control_characters(token):
            raise ValueError("extra llama-server args cannot contain control characters")
        flag = _flag_name(token)
        if flag is not None and flag in _DENYLIST:
            message = (
                f"llama-server flag '{flag}' is managed by Unsloth Studio "
                f"and cannot be passed as an extra arg"
            )
            # Why (#9510): users reaching for `--parallel 1` to cap concurrent predictions on a
            # local model hit this refusal with no pointer to the supported knob; name it.
            if flag in {"-np", "--parallel", "--n-parallel"}:
                message += "; set n_parallel on the load request (parallel decode slots) instead"
            raise ValueError(message)
        if flag is None:
            # A token belonging to no flag. Today's llama-server answers "invalid
            # argument" and refuses to start, which is a failed load rather than a
            # 400, and a build that did accept a positional would read it as the
            # model path: that is the one thing the -m / --model denial exists to
            # prevent, and it would sidestep the native-path lease as well.
            if pending_values <= 0:
                raise ValueError(
                    "extra llama-server args cannot contain a bare value "
                    f"('{token[:64]}'); every value must follow its flag"
                )
            pending_values -= 1
            if pending_two_value > 0:
                pending_two_value -= 1
        elif token != token.strip():
            # _flag_name strips before it looks anything up, so a quoted "--top-k "
            # passed the denylist and the arity walk as --top-k and then went to the
            # child with the space still on it. llama.cpp looks the whole token up,
            # so it answers "error: invalid argument: --top-k" (measured on b10342),
            # naming a flag that looks correct in the log. Only flag-shaped tokens:
            # a VALUE may legitimately end in whitespace, a chat template or a
            # grammar being the obvious ones.
            raise ValueError(
                f"llama-server does not accept the spaces around '{token[:64]}': "
                f"write it as '{flag}'"
            )
        elif "=" in token:
            # llama.cpp looks the WHOLE token up in its option map, folding only the
            # underscore spelling, so "--top-k=20" is not "--top-k" with a value: it
            # is an argument it has never heard of. Measured on b10342 and b10360,
            # where --top-k=20, --ctx-size=4096 and --flash-attn=on each exit with
            # "error: invalid argument". Accepting the GNU spelling here meant the
            # switch tore down the resident model and the child then refused to
            # start, so it is refused while it is still a 400 with somewhere to go.
            # Splitting it here would be a guess: for a switch the value is not one,
            # and this module cannot know an ordinary flag's arity.
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
            # An attached value is ONE of the two, not the whole option:
            # "--control-vector-layer-range=1" still owes its END, and
            # llama-server exits on the incomplete option.
            if flag in _TWO_VALUE_FLAGS:
                pending_values = 1 if attached else 2
                pending_two_value = pending_values
            elif flag in _OPTIONAL_SECOND_VALUE_FLAGS:
                # Allowed, not owed: pending_two_value stays 0, so nothing here
                # insists on the second token.
                pending_values = 1 if attached else 2
                pending_two_value = 0
            else:
                pending_values = 0 if attached else 1
                pending_two_value = 0
            two_value_flag = flag
        out.append(token)
    if pending_two_value > 0:
        # Only this shape is checkable: an ordinary flag's arity is unknown here, so
        # a list ending in one is left to llama-server. START without END is a launch
        # that fails on the command line rather than a request that fails here.
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
    return out


def drop_managed_flags(args: Optional[Iterable[str]]) -> tuple[list[str], list[str]]:
    """Split stored args into what still loads and the flag names removed.

    For the paths that CARRY OVER an existing value rather than receive a new one.
    The denylist grows (``--agent`` and the MCP flags were added once a text box
    made them one paste away), so an override saved by an older build can hold a
    name that is refused today. Refusing there punishes a user for a decision made
    later: the load, or the save of an unrelated setting, fails naming a flag they
    may not remember writing. Dropping is the same judgement applied quietly.

    A flag takes its value with it, or ``--log-file /var/log/x`` would leave a bare
    ``/var/log/x`` behind, which llama.cpp reads as a positional model path. The
    bounds and the control-character rule are enforced by re-validating what is
    left, so the result is always something ``validate_extra_args`` accepts.
    """
    tokens = [str(raw) for raw in (args or [])]

    def _takes_next(
        index: int,
        token: str,
        flag: str,
        source: list = None,
    ) -> bool:
        """True when the token's value is the NEXT token rather than its own.

        ``source`` defaults to the input list; the trimming loop passes the list it
        is shortening, where "the next token" means the one just removed.
        """
        if _value_is_attached(token, flag):
            return False
        seq = tokens if source is None else source
        if source is not None:
            # Called on the last surviving token, whose value was the token just shed.
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
        # A control character never reached the child as anything but noise, and a
        # NUL never reached it at all (execve refuses). A poisoned VALUE takes its
        # flag with it for the same reason a denied flag takes its value: a flag
        # left expecting one would eat the next token and change what that means.
        if _has_control_characters(token) or not _is_spawnable(token):
            # A placeholder either way: this list is joined into a warning log, and
            # the unusable characters are in the token itself, so echoing its name
            # would rewrite whatever is reading that log just as echoing its value
            # would.
            dropped.append("<flag>" if flag is not None else "<value>")
            if flag is not None:
                # Its value goes too, exactly as a denied flag's does: an orphan left
                # behind is a bare positional, which llama-server reads as the model
                # path.
                skip_next = _takes_next(index, token, flag)
            elif kept:
                owner = _flag_name(kept[-1])
                if owner is not None:
                    dropped.append(owner)
                    kept.pop()
            continue
        if flag is not None and "=" in token:
            # An attached value llama-server refuses outright, whatever the flag. Dropped
            # here with the denied names rather than left to the trimming loop below:
            # that loop sheds the TAIL, so one legacy "--top-k=20" in the middle would
            # cost every flag written after it. Nothing to skip, the value is in the
            # token. After the control-character check, so a poisoned name is still
            # logged as a placeholder rather than echoed.
            dropped.append(flag)
            continue
        if flag is not None and token != token.strip():
            # Refused for the same reason, and its value goes with it: the padding is
            # part of the token llama.cpp looks up, so the flag never arrives and the
            # value it was written for would be left as a bare positional.
            dropped.append(flag)
            skip_next = _takes_next(index, token, flag)
            continue
        if (
            flag is not None
            and _takes_next(index, token, flag)
            and (_has_control_characters(tokens[index + 1]) or not _is_spawnable(tokens[index + 1]))
        ):
            # Recorded, not just skipped: the value is about to be dropped for its
            # control characters, and a flag that vanished without a word in the log
            # is the harder half of that to explain afterwards.
            dropped.append(flag)
            continue
        kept.append(token)

    while kept:
        try:
            return validate_extra_args(kept), dropped
        except ValueError:
            # Only the bounds can still fail here, and they are about length, so the
            # tail is the right thing to shed. A flag whose value has just gone with
            # it goes too: `['--grammar', <33 KiB>]` trimmed to `['--grammar']` is
            # syntactically valid to this validator, which knows the arity of only a
            # few flags, and llama-server then refuses the launch over a flag with
            # no value.
            # Names, not values: this list goes into a log line, and the token that
            # broke the bound is by definition enormous.
            dropped.append(_flag_name(kept[-1]) or "<value>")
            kept = kept[:-1]
            last_flag = _flag_name(kept[-1]) if kept else None
            if last_flag is not None and _takes_next(len(kept) - 1, kept[-1], last_flag, kept):
                dropped.append(last_flag)
                kept = kept[:-1]
            # A two-value flag loses the whole option rather than half of it: one
            # value left behind is a command llama-server refuses at startup, which
            # is the failure this trimming exists to avoid.
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
    """Every denied flag, sorted, for a UI that wants to explain a rejection before
    the request is made. The validator stays the authority; this is only a mirror."""
    return sorted(_DENYLIST)


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
        # stripped. The per-drafter tuning knobs (-ngld, --spec-draft-device) are
        # deliberately NOT stripped: the VRAM budget reads them via the same parsers
        # the child honors, so they stay consistent on inherit, and stripping them
        # would silently move a CPU-offloaded drafter back onto the GPU. The draft
        # cache dtype is in that group too, and has its own toggle used only when
        # spec_draft_cache_type is set, the same rule the batch pair follows.
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
# inherited copies of these shadow n_batch / n_ubatch, stripped only when the field is set
_BATCH_FLAGS: frozenset[str] = frozenset({"-b", "--batch-size"})
_UBATCH_FLAGS: frozenset[str] = frozenset({"-ub", "--ubatch-size"})
# Same rule for the tuning group: stripped only when its field is supplied.
# --swa-checkpoints is upstream's older spelling of --ctx-checkpoints.
_CTX_CHECKPOINTS_FLAGS: frozenset[str] = frozenset(
    {"-ctxcp", "--ctx-checkpoints", "--swa-checkpoints"}
)
_CACHE_RAM_FLAGS: frozenset[str] = frozenset({"-cram", "--cache-ram"})
# One group: the control sets a single dtype, so an inherited pair that split K
# from V has to go whole.
_SPEC_DRAFT_CACHE_K_FLAGS: frozenset[str] = frozenset(
    {"-ctkd", "--cache-type-k-draft", "--spec-draft-type-k"}
)
_SPEC_DRAFT_CACHE_V_FLAGS: frozenset[str] = frozenset(
    {"-ctvd", "--cache-type-v-draft", "--spec-draft-type-v"}
)
_SPEC_DRAFT_CACHE_FLAGS: frozenset[str] = _SPEC_DRAFT_CACHE_K_FLAGS | _SPEC_DRAFT_CACHE_V_FLAGS
_FIT_FLAGS: frozenset[str] = frozenset({"-fit", "--fit"})
# The fitter's per-device margin. Never stripped (llama.cpp is last-wins), so a
# pass-through value is what the child really keeps free; see fit_target_margin_in.
_FIT_TARGET_FLAGS: frozenset[str] = frozenset({"-fitt", "--fit-target"})
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


def parse_ctx_checkpoints_override(args: Optional[Iterable[str]]) -> Optional[int]:
    """Return the last user-supplied ``--ctx-checkpoints`` value, or None.

    The control emits its flag before the extras, so a copy typed for this load
    last-wins at launch. Sizing has to price that value, not the field, or a
    ``--ctx-checkpoints 256`` in the extras allocates 256 per-slot snapshots
    against a fit that budgeted the field's count.
    """
    value = _last_flag_value(args, _CTX_CHECKPOINTS_FLAGS)
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        # Malformed extras are refused at the boundary; sizing must not raise here.
        return None
    return max(0, parsed)


def resolve_ctx_checkpoints(args: Optional[Iterable[str]], requested: Optional[int]) -> int:
    """The checkpoint count the launch will actually run: extras beat the field."""
    override = parse_ctx_checkpoints_override(args)
    return int(override if override is not None else (requested or 0))


def resolve_requested_ctx(args: Optional[Iterable[str]], fallback_n_ctx: int) -> int:
    """Return the context size load_model should treat as requested.

    Single source of truth for load_model's ctx-override conditional so
    tests don't reimplement and assert against their own logic.
    """
    override = parse_ctx_override(args)
    return override if override is not None else fallback_n_ctx


def matches_explicit_ctx_override(args: Optional[Iterable[str]], n_ctx: Any) -> bool:
    """Whether a pass-through ``-c``/``--ctx-size`` matches the context the caller
    is already sending as a first-class field.

    Context is the one first-class field whose load-time value is a VRAM-fit
    TARGET, so a matching flag is not a stale shadow but the user's standing
    decision to run past the estimated threshold. Both strippers ask here rather
    than mirroring the test, so auto-switch and /load inheritance cannot drift.

    False for anything unconfirmable, which is the pre-existing strip: no flag, a
    malformed one, or a non-positive/non-int ``n_ctx``. That last case is real --
    override rows are coerced on write but returned verbatim on read, so a row
    from an older build can hold any JSON type and must not raise here.
    """
    if isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0:
        return False
    try:
        return parse_ctx_override(args) == n_ctx
    except ValueError:
        # Malformed extras are refused at the boundary; this must not raise here.
        return False


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


def check_batch_floor(args: Optional[Iterable[str]], n_parallel: int) -> None:
    """Raise when a pass-through --batch-size would abort llama-server.

    The launcher raises the value it emits itself to ``max(slots, 2)``, with the
    measurements recorded beside that code: ``-b 1`` aborts at any slot count, and a
    batch below ``--parallel`` aborts too. Extras are appended AFTER that flag and win
    the last-wins parse, so a small value here is not a smaller batch, it is a server
    that dies during startup, and by then the previous model has been unloaded.

    Only the shapes that are certainly wrong: an unreadable value is left to
    llama-server, which names it better than a guess here would.
    """
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
    """Whether the last ``--fit`` in extras turns the fitter ON.

    Only ``--fit on`` hands placement back to llama.cpp; ``--fit off`` disables
    it and so cannot move weights to the CPU. Upstream requires a value and
    rejects anything that is neither truthy nor falsey, so an absent or
    unreadable value is not an enable.
    """
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    return raw_value is not None and raw_value.strip().lower() in _ENV_TRUE_VALUES


def fit_is_effectively_on(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> bool:
    """Whether the fitter actually runs, over the WHOLE argv and the env twin.

    ``fit_is_enabled_in`` answers for the extras alone; this answers for the
    child. llama.cpp defaults the fitter ON and applies the env before argv, so
    only an explicit "off" turns it off, and an unreadable value keeps it on.
    """
    raw_value = _last_flag_value(args, _FIT_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT")
    if raw_value is None:
        return True
    return str(raw_value).strip().lower() not in _ENV_FALSE_VALUES


def fit_target_margin_in(
    args: Optional[Iterable[str]], env: Optional[Mapping[str, str]] = None
) -> Optional[float]:
    """The per-device margin an effective ``--fit-target`` asks the fitter to keep.

    ``-fitt/--fit-target`` takes a list of MiB values, one per device, and a
    single value is broadcast across all of them (common/arg.cpp; default 1024,
    ``fit_params_target`` in common/common.h). The fitter refuses
    to allocate into that margin and spills the rest to host RAM instead
    (``targets.push_back(dmds_full[id].free - margins[id])``, common/fit.cpp), so
    a load-mode fit that credits VRAM has to price it.

    Returns the LARGEST value in the list, because the fit charges one margin to
    every device it credits and understating would claim a fit that is not there.
    ``None`` when nothing readable is set, which leaves the caller on llama.cpp's
    own default rather than on a guess. Last-wins over argv, and the env twin only
    when argv sets nothing, the same precedence ``fit_is_effectively_on`` uses.
    """
    raw_value = _last_flag_value(args, _FIT_TARGET_FLAGS)
    if raw_value is None and env:
        raw_value = env.get("LLAMA_ARG_FIT_TARGET")
    if raw_value is None:
        return None
    values: list[float] = []
    # Upstream splits on both "," and "/" (common/arg.cpp), so "4096/4096" is a
    # well-formed two-device margin; reading it as one token would wrongly abstain.
    for part in str(raw_value).replace("/", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            # Upstream rejects the whole list, so abstain rather than price part of it.
            return None
    return max(values) if values else None


def split_policy_starves_devices(
    args: Optional[Iterable[str]],
    n_credited: int,
    env: Optional[Mapping[str, str]] = None,
) -> bool:
    """True when the effective split leaves fewer devices holding weights than ``n_credited``.

    ``--split-mode`` and ``--tensor-split`` are pass-through under auto-select
    (``_SPLIT_SHADOWING_FLAGS`` is stripped only when the Tensor Parallelism toggle
    owns the split), so both reach the child appended after Unsloth's own placement
    flags. Either can quietly shrink the pool a pooled VRAM credit was priced for:

    * ``--split-mode none`` is ``LLAMA_SPLIT_MODE_NONE`` (common/arg.cpp), which
      puts the whole model on ``--main-gpu`` alone. ``row``/``layer``/``tensor``
      all keep every device, so only ``none`` starves.
    * ``--tensor-split`` is a per-device proportion list, and upstream zero-fills
      every device past the end of the list (common/arg.cpp), so a short list
      starves the tail just as an explicit ``0`` starves its own device.

    Value-aware on purpose: a restatement that keeps all devices is not an
    override, and voiding on it would abstain from a fit that is really there.
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
    # Upstream splits on both "," and "/" (common/arg.cpp).
    for part in str(raw_split).replace("/", ",").split(",")[:n_credited]:
        part = part.strip()
        if not part:
            continue
        try:
            if float(part) > 0.0:
                holding += 1
        except ValueError:
            # Upstream throws on this and the child never starts: nothing to misprice.
            return False
    return holding < n_credited


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
    strip_batch: bool = False,
    strip_ubatch: bool = False,
    strip_ctx_checkpoints: bool = False,
    strip_cache_ram: bool = False,
    strip_spec_draft_cache: bool = False,
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


def strip_context_only(args: Optional[Iterable[str]]) -> Optional[list[str]]:
    """Remove the context group (``-c`` / ``--ctx-size``) from ``args``, keeping
    every other shadow flag. Preserves a None/empty input so the
    inherit-vs-explicit-empty distinction survives. Used by the Metal
    zero-context floor, where a trailing ``-c 0`` would last-wins override the
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

    The per-model Mmap/Mlock control is resolved separately, by
    ``apply_load_mode_policy``, which runs after this and defers to it.
    """
    try:
        from utils.model_memory_settings import get_model_memory_settings
    except Exception:
        # Settings unavailable (bare unit-test import): behave as before.
        return [], list(extra_args or [])

    # One snapshot for both decisions: read separately, a save landing between
    # them strips for one setting and locks for the other, so a saved --mlock
    # could survive a committed no-reserve.
    keep_resident, no_ram_reserve = get_model_memory_settings()
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
    if keep_resident and not no_ram_reserve and weights_in_host_memory:
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


def apply_load_mode_policy(
    extra_args: Optional[Iterable[str]],
    *,
    supports_load_mode: bool = False,
    weights_in_host_memory: bool = True,
    requested_load_mode: Optional[str] = None,
) -> tuple[list[str], list[str]]:
    """Resolve the per-model Mmap/Mlock control into llama-server flags.

    Returns ``(managed_flags, extras)``, like ``apply_model_memory_policy``, and
    is meant to run straight after it on the extras that call returned.

    The Model Memory settings win, which is what the Run settings panel tells the
    user, so changing the order without changing ``loadModeOverrideNotice`` makes
    that note wrong. "Keep model in GPU memory" owns the mode outright while it
    applies; "Don't reserve system RAM" vetoes the values holding a full host copy
    (``none``, ``mlock``, ``mmap+mlock``) and leaves ``mmap`` and ``dio`` alone.

    ``auto`` emits nothing: it IS llama.cpp's default, so pinning it would freeze
    what a later build may redefine. An unknown value is dropped rather than passed
    through, since llama-server exits on one.
    """
    tokens = list(extra_args or [])
    mode = _normalize_load_mode_value(requested_load_mode)
    if not mode:
        return [], tokens
    try:
        from utils.model_memory_settings import get_model_memory_settings
        keep_resident, no_ram_reserve = get_model_memory_settings()
    except Exception:
        # Settings unavailable (bare unit-test import): nothing to defer to.
        keep_resident, no_ram_reserve = False, False
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
        # A build predating the enum understands the spellings it replaced, and
        # only for the values that had one.
        legacy = _LEGACY_LOAD_MODE_FLAGS.get(mode)
        if not legacy:
            logger.info("llama-server has no --load-mode; skipping the requested %r mode.", mode)
            return [], tokens
        return list(legacy), tokens
    # Emitted BEFORE the extras and stripping nothing, like every other control
    # here: a flag typed for THIS load is appended after and last-wins, which is
    # what the panel's diagnostics promise. An INHERITED copy is a different
    # thing, and the route drops that one before it ever reaches here.
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


def _env_var_locks_or_reserves(name: str, value: str) -> bool:
    """Whether this inherited var, as set, locks or reserves host RAM.

    Mirrors the argv rule: the settings own the RESERVATION, not the loader, so
    a DirectIO or mmap choice made through the environment survives the same way
    ``--load-mode dio`` does. An unrecognised value is left alone.
    """
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        return normalized in _ENV_TRUE_VALUES
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        # Presence alone selects mode "none", which is a full host buffer.
        return True
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        # Falsy selects "none"; truthy selects mmap / dio, neither of which
        # holds a full copy.
        return normalized in _ENV_FALSE_VALUES
    if name == "LLAMA_ARG_LOAD_MODE":
        return normalized in _LOAD_MODE_MLOCK_VALUES or normalized in _LOAD_MODE_RESERVING_VALUES
    return False


# The LLAMA_ARG_* twins of flags the denylist refuses. llama.cpp reads these before
# argv, so a name refused in extra args is still reachable through the environment
# Unsloth's own process inherits. Anyone who can set that environment can already do
# worse, so this is not the boundary -- it just stops a denied flag arriving by the
# back door and leaving no trace in the recorded command.
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
    # The twins of --log-file and --log-disable. Unsloth classifies a failed start by
    # reading llama-server's own output, so an inherited redirect leaves every
    # failure looking like the same opaque one; and unlike the flags, Unsloth emits
    # nothing later that would override these. LLAMA_ARG_LOG_DISABLE has no twin in
    # today's builds, and is listed so it cannot arrive as one.
    "LLAMA_ARG_LOG_FILE",
    "LLAMA_ARG_LOG_DISABLE",
    # --api-prefix moves every endpoint, including the /health Unsloth waits on, so an
    # inherited one turns every load into a timeout.
    "LLAMA_ARG_API_PREFIX",
    # --api-key and its file. Unsloth terminates auth itself and sends the child no
    # Authorization header, so an inherited key makes the healthy child refuse every
    # request. The bundled build reads LLAMA_API_KEY for the flag and
    # LLAMA_ARG_API_KEY_FILE for the file; the third spelling is listed because the
    # name has moved between releases and none of them is ours to honour.
    "LLAMA_API_KEY",
    "LLAMA_ARG_API_KEY",
    "LLAMA_ARG_API_KEY_FILE",
    # The twins of --ssl-key-file and --ssl-cert-file. Given both, llama-server
    # listens on https, while Unsloth probes /health and proxies over http against
    # the port it launched: the child comes up healthy and every load times out.
    # Measured on b10360, where an inherited pair turns "listening on
    # http://127.0.0.1:PORT" into "listening on https://...".
    "LLAMA_ARG_SSL_KEY_FILE",
    "LLAMA_ARG_SSL_CERT_FILE",
    # The rest of the twins its --help documents for a denied flag, enumerated from
    # the bundled b10342 help rather than picked one at a time: every "(env: NAME)"
    # whose option this module refuses. Unsloth emits most of these itself and argv
    # wins over the environment, so removing them changes nothing in the ordinary
    # case; they are here for the paths where it does not, and so a flag denied in
    # the box is not reachable through the environment instead. The mapping below
    # records which flag each one belongs to, since the name does not always say
    # (LLAMA_ARG_STATIC_PATH is --path).
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
    # The web UI and its MCP proxy, which upstream marks as not for untrusted
    # environments, and the directory the child serves files from.
    "LLAMA_ARG_UI",
    "LLAMA_ARG_UI_CONFIG",
    "LLAMA_ARG_UI_CONFIG_FILE",
    "LLAMA_ARG_UI_MCP_PROXY",
    "LLAMA_ARG_STATIC_PATH",
    # Deliberately absent: LLAMA_ARG_MMPROJ and LLAMA_ARG_MMPROJ_URL. --mmproj is
    # refused in the box because Unsloth resolves the projector itself, but the
    # environment twin is an INPUT here: _launch_has_mmproj reads both to know the
    # launch has a projector at all, which is what keeps the vision and audio state
    # of a model loaded through an inherited one. Only the paravirtual CPU recovery
    # drops them, where an unpinned projector is the corrupt path it is undoing.
    # The pooling twins are absent for the opposite reason: load_model already pops
    # LLAMA_ARG_POOLING / _RERANKING / _EMBEDDINGS itself, next to where it decides
    # what the GGUF header says.
    # The multi-model server mode: a child holding its own model directory, preset
    # and autoload policy is not the single model Unsloth launched and accounts for.
    "LLAMA_ARG_MODELS_DIR",
    "LLAMA_ARG_MODELS_PRESET",
    "LLAMA_ARG_MODELS_MAX",
    "LLAMA_ARG_MODELS_AUTOLOAD",
)

# Which flag each twin belongs to, for the drift test. Derived by name for most of
# them, but not all: LLAMA_ARG_STATIC_PATH is --path, LLAMA_API_KEY is --api-key, and
# a rename upstream would otherwise leave a variable here guarding nothing.
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

    The argv twin of ``memory_env_selects_load_mode``, and it answers for both
    spellings: the ``--load-mode`` enum, and the flags it replaced
    (``--no-mmap`` / ``--mmap`` / the direct-IO pair), which are what this launch
    emits on a build predating the enum (``_LEGACY_LOAD_MODE_FLAGS``) and so are
    what a user's own flag would collide with there.

    Presence, not value: a fit-derived mode stands aside for any pick the user made
    rather than trying to rank the two. Their tokens are appended after the managed
    block and llama.cpp is last-wins, so theirs governs either way.
    """
    for raw in extra_args or ():
        if _flag_name(str(raw)) in _LOAD_MODE_FLAGS | _LOAD_MODE_ALIAS_FLAGS:
            return True
    return False


def memory_env_selects_load_mode(env: Optional[Mapping[str, str]]) -> bool:
    """Whether the inherited environment already picks a loader mode.

    llama.cpp applies the ``LLAMA_ARG_*`` twins BEFORE argv (common/arg.cpp runs
    its environment loop and only then "handle command line arguments"), and both
    assign the same ``params.load_mode``, so a managed ``--load-mode`` emitted here
    beats an inherited choice without the user typing anything. A mode DERIVED from
    a fit has to stand aside for that choice, the same way it stands aside for the
    per-model pick; a hand-typed flag is argv and still wins by last-arg.

    Per variable, matching what upstream really does with each value: a no-value
    option (``--mlock``) runs its handler only for a truthy one (``opt.handler_void
    && is_truthy(value)``), a negative alias (``LLAMA_ARG_NO_MMAP`` /
    ``LLAMA_ARG_NO_DIO``) counts by PRESENCE whatever it says (``get_value_from_env``
    forces "0" when it exists), and the rest assign a mode for any value they parse.
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


def scrub_memory_env(env: dict) -> list[str]:
    """Drop inherited memory placement the settings override.

    Returns the names removed, for logging. A no-op with both toggles off, so an
    existing LLAMA_ARG_MLOCK deployment keeps working untouched. Only the values
    that actually lock or reserve go: an inherited ``LLAMA_ARG_DIO=1`` is a
    loader choice, not a reservation, and no-reserve has no quarrel with it.
    """
    if not model_memory_owns_placement():
        return []
    removed = [
        name
        for name in MEMORY_ENV_VARS
        if name in env and _env_var_locks_or_reserves(name, env[name])
    ]
    for name in removed:
        env.pop(name, None)
    return removed


# The pageable twin of each mode that reads the weights into a buffer it allocates.
# Upstream sets use_mmap for mmap / mmap+mlock / auto only (llama-model-loader.cpp),
# so `mlock` is a full host copy that is also locked, and `mmap+mlock` is the same
# lock over a mapping. `none` has no lock to preserve, so it goes back to the default.
_PAGEABLE_LOAD_MODE: dict[str, Optional[str]] = {"none": None, "mlock": "mmap+mlock"}
# Modes that already map, so the rewrite has no unmapped copy of its own to fix and
# leaves them alone. It reaches them only when a LATER reserving selector shadowed the
# lock (``--load-mode mmap+mlock --no-mmap`` runs unlocked and unmapped, last-wins), and
# there dropping only the selector hands the child back the lock it had lost -- over the
# full-size mapping the override just restored, which is the one outcome it exists to
# prevent. So they are stripped in that state and in no other.
_SHADOWED_LOCK_LOAD_MODE = frozenset({"mmap+mlock"})


def _pageable_mode_replacement(
    normalized: str, drop_shadowed_mlock: bool
) -> tuple[bool, Optional[str]]:
    """``(rewrite, replacement)`` for one ``--load-mode`` value, argv or env alike.

    ``replacement`` None removes the selector, leaving llama.cpp's default mapping.
    """
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

    ``rewrite`` False leaves the var alone. ``new_value`` None means remove it; a
    string replaces it, which is how a locked mode keeps its lock. ``LLAMA_ARG_MLOCK``
    is otherwise left alone: on its own it sets the lock bit over the default mapping
    and holds no unmapped copy.

    ``drop_shadowed_mlock`` says the launch is NOT locked before the rewrite, because
    a later selector reset the mlock bit -- llama.cpp resolves these last-wins, so
    ``LLAMA_ARG_MLOCK=1`` beside ``LLAMA_ARG_NO_MMAP`` leaves the child unlocked. There
    is then no lock to carry, so the var goes and a ``mlock`` mode is dropped rather
    than promoted to ``mmap+mlock``.
    """
    normalized = value.strip().lower()
    if name == "LLAMA_ARG_MLOCK":
        # Only when it is already shadowed: resurrecting it would page-lock the
        # oversized mapping into the RAM this override exists to keep pageable.
        return drop_shadowed_mlock and normalized in _ENV_TRUE_VALUES, None
    if name in {"LLAMA_ARG_NO_MMAP", "LLAMA_ARG_NO_DIO"}:
        # Presence alone selects mode "none", whatever the value says.
        return True, None
    if name in {"LLAMA_ARG_MMAP", "LLAMA_ARG_DIO"}:
        # Falsy selects "none"; truthy selects mmap / dio, neither of which
        # holds a full copy.
        return normalized in _ENV_FALSE_VALUES, None
    if name == "LLAMA_ARG_LOAD_MODE":
        return _pageable_mode_replacement(normalized, drop_shadowed_mlock)
    return False, None


def force_pageable_load(
    argv: Optional[Iterable[str]], env: Optional[dict] = None
) -> tuple[list[str], list[str]]:
    """Rewrite a launch that would hold a full unmapped host copy into a pageable one.

    Returns ``(argv, overridden)``, naming the argv tokens and env vars that were
    dropped or rewritten, for the log line and the warning. Empty means the launch was
    already pageable and ``argv`` comes back unchanged.

    Modes ``none`` and ``mlock`` read the weights into a buffer llama.cpp allocates
    (``use_mmap`` is set for ``mmap``/``mmap+mlock``/``auto`` and nothing else), so a
    model larger than free RAM cannot load at all rather than paging in slowly. Both
    sides of llama.cpp's env-then-argv resolution are rewritten, since the environment
    supplies the default the argv only overrides when it names the same option.

    A lock that is EFFECTIVE is preserved rather than dropped: ``mlock`` becomes
    ``mmap+mlock`` and ``--no-mmap --mlock`` keeps its lock through the strip, so "keep
    this in RAM" still holds -- over a mapping the kernel can fall back on. ``none`` has
    no lock to keep and needs no replacement flag at all, mmap being the default, so a
    build predating ``--load-mode`` is handed nothing it cannot parse.

    A SHADOWED lock is not resurrected. These options resolve last-wins, so
    ``--mlock --no-mmap`` (and the ``LLAMA_ARG_`` twins, where the negative alias is
    read after the affirmative one) runs unlocked and unmapped: the reserving selector
    already cleared the lock bit, which is what ``resolve_effective_memory_state``
    reports. Dropping only the selector and leaving the earlier ``--mlock`` standing
    would hand the child ``mmap+mlock`` and page-lock the whole oversized mapping into
    the RAM this override exists to keep pageable -- worse than the load it fixes. So
    the pre-rewrite state decides, not the tokens that happen to be present.
    """
    tokens = [str(a) for a in (argv or [])]
    # What the child runs TODAY, across env and argv in llama.cpp's own resolution
    # order. Only a launch that reserves RAM is rewritten at all, so a pageable one
    # (``dio``, plain ``mmap``) keeps every token it was given, mlock included.
    _mlock_now, _reserves_now = resolve_effective_memory_state(tokens, env)
    drop_shadowed_mlock = _reserves_now and not _mlock_now
    overridden: list[str] = []
    out: list[str] = []
    i, n = 0, len(tokens)
    while i < n:
        token = tokens[i]
        flag = _flag_name(token)
        # --no-mmap / --no-direct-io: upstream's deprecated spellings for "none".
        # Valueless, so the token goes and nothing follows it out.
        if flag in _RAM_RESERVING_FLAGS:
            overridden.append(token)
            i += 1
            continue
        # A lock a later selector already cleared. Valueless like the above, and
        # named in `overridden` so the log line describes the whole rewrite.
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
            # `--load-mode mlock --no-mmap` is unlocked by the time the child parses
            # it, so mmap+mlock would ADD a lock; `mmap+mlock --no-mmap` is the same
            # shape one spelling further on, and there the selector itself is the lock.
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


# Mirrors llama_cpp's _LLAMA_ARG_TRUE/FALSE_VALUES; duplicated so this module
# stays dependency-free (llama_cpp imports from here, not the other way).
_ENV_TRUE_VALUES = frozenset({"on", "enabled", "true", "1"})
_ENV_FALSE_VALUES = frozenset({"off", "disabled", "false", "0"})

# Every --load-mode value llama-server documents, so an unknown one is dropped
# here rather than exiting the child. Mirrored by LOAD_MODES in per-model-config.ts.
_LOAD_MODE_VALUES = frozenset({"auto", "none", "mmap", "mlock", "mmap+mlock", "dio"})
# What each mode meant before the enum existed, for a build that predates it.
# "auto" is the default and needs no flag; "mmap+mlock" is what a bare --mlock
# asked for alongside the default mmap. There is no pre-enum spelling for plain
# "mmap" or for "dio", so those are skipped rather than approximated.
_LEGACY_LOAD_MODE_FLAGS: dict[str, list[str]] = {
    "none": ["--no-mmap"],
    "mlock": ["--no-mmap", "--mlock"],
    "mmap+mlock": ["--mlock"],
}
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
