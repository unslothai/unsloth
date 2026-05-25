# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the llama-server pass-through args validator.

The validator is the boundary between user CLI/HTTP input and the
llama-server subprocess. These tests pin denylist behaviour so it
doesn't quietly regress when new managed flags are added.
"""

from __future__ import annotations

import re

import pytest

from core.inference.llama_server_args import (
    is_managed_flag,
    strip_shadowing_flags,
    validate_extra_args,
)


# ── Pass-through (allowed) ───────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        # Sampling
        ["--top-k", "20"],
        ["--top-p", "0.9", "--min-p", "0.05"],
        ["--seed", "-1"],  # negative value, not a flag
        ["--temp", "0.0"],
        ["--repeat-penalty", "1.05"],
        ["--mirostat", "2", "--mirostat-lr", "0.1"],
        ["--xtc-probability", "0.05", "--xtc-threshold", "0.1"],
        ["--dry-multiplier", "0.5"],
        # Tier-2 knobs that map to LoadRequest fields
        ["--cache-type-k", "q8_0"],
        ["--cache-type-v", "q8_0"],
        ["--chat-template-file", "/tmp/tpl.jinja"],
        ["--chat-template-kwargs", '{"reasoning_effort":"high"}'],
        ["--spec-type", "ngram-mod"],
        ["--spec-default"],
        # MTP path (llama.cpp #22673).
        ["--spec-type", "draft-mtp"],
        ["--spec-type", "draft-mtp", "--spec-draft-n-max", "6"],
        [
            "--spec-type",
            "ngram-mod,draft-mtp",
            "--spec-draft-n-max",
            "3",
            "--spec-ngram-mod-n-match",
            "24",
            "--spec-ngram-mod-n-min",
            "48",
            "--spec-ngram-mod-n-max",
            "64",
        ],
        # Reasoning controls
        ["--reasoning-format", "deepseek"],
        ["-rea", "auto"],
        # Soft-managed: user-supplied flags last-wins-override Studio's
        # auto-set version. --parallel / -np / --n-parallel are NOT
        # here -- they're hard-denied (KV-cache + slot count would
        # desync). Use `unsloth studio run --parallel N` instead.
        ["-c", "131072"],
        ["--ctx-size", "8192"],
        ["--flash-attn", "off"],
        ["-fa", "on"],
        ["--no-context-shift"],
        ["--context-shift"],
        ["--jinja"],
        ["--no-jinja"],
        ["-ngl", "-1"],
        ["--gpu-layers", "32"],
        ["-t", "16"],
        ["--threads", "32"],
        ["-fit", "off"],
        ["--fit", "on"],
        ["--fit-ctx", "8192"],
    ],
)
def test_pass_through_allowed(args):
    assert validate_extra_args(args) == args


def test_none_returns_empty_list():
    assert validate_extra_args(None) == []


def test_empty_list_returns_empty_list():
    assert validate_extra_args([]) == []


def test_value_with_equals_form_passes_through():
    assert validate_extra_args(["--top-k=20"]) == ["--top-k=20"]


def test_non_flag_token_passes_through():
    # Bare positionals are passed through; llama-server can reject them.
    assert validate_extra_args(["foo"]) == ["foo"]


# ── Denylist (rejected) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "denied",
    [
        # Parallel slots -- owned by the typer --parallel flag.
        "-np",
        "--parallel",
        "--n-parallel",
        # Model identity (every alias; bumping llama.cpp must keep
        # every form rejected, not just the long).
        "-m",
        "--model",
        "-mu",
        "--model-url",
        "-dr",
        "--docker-repo",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
        "-hfv",
        "-hfrv",
        "--hf-repo-v",
        "-hffv",
        "--hf-file-v",
        "-hft",
        "--hf-token",
        "-mm",
        "--mmproj",
        "-mmu",
        "--mmproj-url",
        # Networking (Studio binds + proxies)
        "--host",
        "--port",
        "--path",
        "--api-prefix",
        "--reuse-port",
        # Auth / TLS
        "--api-key",
        "--api-key-file",
        "--ssl-key-file",
        "--ssl-cert-file",
        # Single-model server (legacy --webui + current --ui group)
        "--webui",
        "--no-webui",
        "--ui",
        "--no-ui",
        "--ui-config",
        "--ui-config-file",
        "--ui-mcp-proxy",
        "--no-ui-mcp-proxy",
        "--models-dir",
        "--models-preset",
        "--models-max",
        "--models-autoload",
        "--no-models-autoload",
    ],
)
def test_denylist_rejects_all_aliases(denied):
    with pytest.raises(ValueError, match = denied):
        validate_extra_args([denied, "value"])


@pytest.mark.parametrize(
    "args,offending",
    [
        # Pass-through --parallel would last-wins-override the real
        # slot count while Studio's KV-cache fit + llama_parallel_slots
        # stay at the typer value -- plan vs. process disagree.
        (["--parallel", "8"], "--parallel"),
        (["--parallel=8"], "--parallel"),
        (["--n-parallel", "16"], "--n-parallel"),
        (["--n-parallel=16"], "--n-parallel"),
        (["-np", "32"], "-np"),
        # Attached short form: Click clusters it CLI-side; HTTP /load
        # with `["-np8"]` must still resolve to managed.
        (["-np8"], "-np"),
        (["-np64"], "-np"),
        # Out-of-range values that would bypass the typer 1..64 guard.
        (["--parallel", "999"], "--parallel"),
        (["-np", "0"], "-np"),
        (["-np999"], "-np"),
        # Signed attached forms; `-np-1` must not slip past.
        (["-np-1"], "-np"),
        (["-np+1"], "-np"),
    ],
)
def test_parallel_flags_are_managed(args, offending):
    with pytest.raises(ValueError, match = re.escape(offending)):
        validate_extra_args(args)


def test_denylist_rejects_equals_form():
    with pytest.raises(ValueError, match = "--port"):
        validate_extra_args(["--port=9000"])


@pytest.mark.parametrize(
    "padded",
    [" --parallel", "--parallel ", "\t--parallel", "  -np", "-np \n", "-np\t"],
)
def test_denylist_rejects_whitespace_padded_forms(padded):
    # `_flag_name` trims whitespace before lookup; otherwise a trailing
    # space could slip a managed flag past the boundary.
    with pytest.raises(ValueError, match = "parallel|np"):
        validate_extra_args([padded, "8"])


@pytest.mark.parametrize(
    "attached",
    ["-np8x", "-np-1foo", "-np+1bar", "-np9zzz"],
)
def test_denylist_rejects_np_with_digit_prefix_and_junk(attached):
    # Backend `_flag_name` must classify the same forms the CLI
    # rewriter expands, else HTTP /load could smuggle `-np8x` through.
    with pytest.raises(ValueError, match = "np"):
        validate_extra_args([attached])


def test_denylist_rejects_short_form_when_long_is_denied():
    # `-m` is the short form of --model; rejecting only the long
    # form would leave a trivial bypass.
    with pytest.raises(ValueError, match = "-m"):
        validate_extra_args(["-m", "/some/other/path.gguf"])


def test_denylist_message_names_offending_flag():
    with pytest.raises(ValueError) as excinfo:
        validate_extra_args(["--top-k", "20", "--api-key", "secret"])
    assert "--api-key" in str(excinfo.value)


def test_first_denied_flag_short_circuits():
    # Validation stops at the first denied flag; the message names it.
    with pytest.raises(ValueError, match = "--port"):
        validate_extra_args(["--port", "1", "--host", "x"])


# ── Numeric values that look flag-ish ─────────────────────────────────


@pytest.mark.parametrize("value", ["-1", "-0.5", "-42", "-.5"])
def test_negative_number_value_is_not_flag(value):
    # `--seed -1`: the -1 is a value, not a flag.
    assert validate_extra_args(["--seed", value]) == ["--seed", value]


# ── is_managed_flag helper ───────────────────────────────────────────


def test_is_managed_flag_true_for_denied():
    assert is_managed_flag("--port") is True
    assert is_managed_flag("--api-key") is True
    assert is_managed_flag("-m") is True
    assert is_managed_flag("--model") is True
    # Parallel slots owned by the typer --parallel flag.
    assert is_managed_flag("--parallel") is True
    assert is_managed_flag("--n-parallel") is True
    assert is_managed_flag("-np") is True
    # Normalised forms must classify like the canonical token so
    # is_managed_flag filtering stays in sync with validate_extra_args.
    assert is_managed_flag("-np8") is True
    assert is_managed_flag("--parallel=8") is True
    assert is_managed_flag("--port=9000") is True


def test_is_managed_flag_false_for_pass_through():
    assert is_managed_flag("--top-k") is False
    assert is_managed_flag("--cache-type-k") is False
    assert is_managed_flag("--chat-template-file") is False
    # Soft-managed flags pass through (last-wins override)
    assert is_managed_flag("-c") is False
    assert is_managed_flag("--ctx-size") is False
    assert is_managed_flag("--flash-attn") is False
    assert is_managed_flag("-ngl") is False
    assert is_managed_flag("--threads") is False


# ── strip_shadowing_flags ─────────────────────────────────────────────


def test_strip_shadowing_flags_drops_context_when_requested():
    out = strip_shadowing_flags(
        ["-c", "4096", "--top-k", "20"],
        strip_context = True,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
    )
    assert out == ["--top-k", "20"]


def test_strip_shadowing_flags_keeps_context_when_not_requested():
    out = strip_shadowing_flags(
        ["-c", "4096", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
    )
    assert out == ["-c", "4096", "--top-k", "20"]


def test_strip_shadowing_flags_keeps_chat_template_when_template_disabled():
    # No chat_template_override supplied; inherited
    # --chat-template-file must survive.
    out = strip_shadowing_flags(
        ["--chat-template-file", "/tmp/custom.jinja", "--top-k", "20"],
        strip_context = True,
        strip_cache = True,
        strip_spec = True,
        strip_template = False,
    )
    assert out == ["--chat-template-file", "/tmp/custom.jinja", "--top-k", "20"]


def test_strip_shadowing_flags_drops_template_when_requested():
    out = strip_shadowing_flags(
        ["--chat-template-file", "/tmp/custom.jinja", "--top-k", "20"],
        strip_template = True,
    )
    assert out == ["--top-k", "20"]


def test_strip_shadowing_flags_keeps_cache_when_cache_disabled():
    out = strip_shadowing_flags(
        ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--top-k", "20"],
        strip_cache = False,
    )
    assert out == [
        "--cache-type-k",
        "q8_0",
        "--cache-type-v",
        "q8_0",
        "--top-k",
        "20",
    ]


def test_strip_shadowing_flags_keeps_spec_when_spec_disabled():
    out = strip_shadowing_flags(
        ["--spec-type", "ngram-mod", "--draft-min", "48", "--top-k", "20"],
        strip_spec = False,
    )
    assert out == [
        "--spec-type",
        "ngram-mod",
        "--draft-min",
        "48",
        "--top-k",
        "20",
    ]


def test_strip_shadowing_flags_drops_mtp_flags_when_requested():
    # MTP / draft-mtp flags must drop when speculative_type re-applies.
    out = strip_shadowing_flags(
        [
            "--spec-type",
            "draft-mtp",
            "--spec-draft-n-max",
            "6",
            "--spec-ngram-mod-n-match",
            "24",
            "--spec-ngram-mod-n-min",
            "48",
            "--spec-ngram-mod-n-max",
            "6",
            "--top-k",
            "20",
        ],
        strip_spec = True,
    )
    assert out == ["--top-k", "20"]


def test_is_managed_flag_false_for_mtp_pass_through():
    assert is_managed_flag("--spec-draft-n-max") is False
    assert is_managed_flag("--spec-ngram-mod-n-match") is False
    assert is_managed_flag("--spec-ngram-mod-n-min") is False
    assert is_managed_flag("--spec-ngram-mod-n-max") is False


def test_strip_shadowing_flags_boolean_does_not_consume_next_token():
    # `--spec-default` is boolean; drop just the flag, keep the next token.
    out = strip_shadowing_flags(["--spec-default", "ngram-mod"], strip_spec = True)
    assert out == ["ngram-mod"]


def test_strip_shadowing_flags_jinja_boolean_preserves_positional():
    out = strip_shadowing_flags(["--jinja", "trailing-positional"], strip_template = True)
    assert out == ["trailing-positional"]


def test_strip_shadowing_flags_no_jinja_boolean_preserves_positional():
    out = strip_shadowing_flags(
        ["--no-jinja", "trailing-positional"], strip_template = True
    )
    assert out == ["trailing-positional"]


def test_strip_shadowing_flags_equals_form_drops_only_the_flag():
    out = strip_shadowing_flags(["--ctx-size=4096", "--seed", "-1"], strip_context = True)
    assert out == ["--seed", "-1"]


def test_strip_shadowing_flags_handles_none_input():
    assert strip_shadowing_flags(None) == []


def test_strip_shadowing_flags_handles_empty_input():
    assert strip_shadowing_flags([]) == []


def test_strip_shadowing_flags_defaults_strip_everything():
    # The route's already-loaded comparator calls with no kwargs to
    # detect ANY shadowing flag in stored extras.
    out = strip_shadowing_flags(
        ["-c", "4096", "--cache-type-k", "q8_0", "--spec-default", "--jinja"]
    )
    assert out == []
