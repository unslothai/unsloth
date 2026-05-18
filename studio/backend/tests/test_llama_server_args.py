# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the llama-server pass-through args validator.

The validator is the security boundary between user-supplied CLI / HTTP
input and the llama-server subprocess command. These tests pin the
denylist behavior so the boundary doesn't quietly regress when new
managed flags are added.
"""

from __future__ import annotations

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
        # Reasoning controls
        ["--reasoning-format", "deepseek"],
        ["-rea", "auto"],
        # Soft-managed flags the user may want to override on the CLI;
        # llama.cpp's last-wins parsing means these win over Studio's
        # auto-set version.
        ["-c", "131072"],
        ["--ctx-size", "8192"],
        ["--parallel", "1"],
        ["-np", "8"],
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
    # A bare positional value (not preceded by a flag) is preserved
    # verbatim. llama-server may reject it, but that's not our job.
    assert validate_extra_args(["foo"]) == ["foo"]


# ── Denylist (rejected) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "denied",
    [
        # Model identity
        "-m",
        "--model",
        "-hf",
        "-hfr",
        "--hf-repo",
        "-hff",
        "--hf-file",
        "-hft",
        "--hf-token",
        "-mm",
        "--mmproj",
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
        # Single-model server
        "--webui",
        "--no-webui",
        "--models-dir",
        "--models-max",
    ],
)
def test_denylist_rejects_all_aliases(denied):
    with pytest.raises(ValueError, match = denied):
        validate_extra_args([denied, "value"])


def test_denylist_rejects_equals_form():
    with pytest.raises(ValueError, match = "--port"):
        validate_extra_args(["--port=9000"])


def test_denylist_rejects_short_form_when_long_is_denied():
    # -m is the short form of the hard-denied --model; rejecting only
    # the long form would leave a trivial bypass.
    with pytest.raises(ValueError, match = "-m"):
        validate_extra_args(["-m", "/some/other/path.gguf"])


def test_denylist_message_names_offending_flag():
    with pytest.raises(ValueError) as excinfo:
        validate_extra_args(["--top-k", "20", "--api-key", "secret"])
    assert "--api-key" in str(excinfo.value)


def test_first_denied_flag_short_circuits():
    # Validation stops at the first denied flag; later denied flags
    # in the same call don't matter for behaviour, but the message
    # should name the first one we hit.
    with pytest.raises(ValueError, match = "--port"):
        validate_extra_args(["--port", "1", "--host", "x"])


# ── Numeric values that look flag-ish ─────────────────────────────────


@pytest.mark.parametrize("value", ["-1", "-0.5", "-42", "-.5"])
def test_negative_number_value_is_not_flag(value):
    # ``--seed -1`` is a value, not a flag. Validator must not try
    # to look up "-1" in the denylist.
    assert validate_extra_args(["--seed", value]) == ["--seed", value]


# ── is_managed_flag helper ───────────────────────────────────────────


def test_is_managed_flag_true_for_denied():
    assert is_managed_flag("--port") is True
    assert is_managed_flag("--api-key") is True
    assert is_managed_flag("-m") is True
    assert is_managed_flag("--model") is True


def test_is_managed_flag_false_for_pass_through():
    assert is_managed_flag("--top-k") is False
    assert is_managed_flag("--cache-type-k") is False
    assert is_managed_flag("--chat-template-file") is False
    # Soft-managed flags pass through (last-wins override)
    assert is_managed_flag("-c") is False
    assert is_managed_flag("--ctx-size") is False
    assert is_managed_flag("--parallel") is False
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
    # Caller did not supply chat_template_override; the inherited
    # --chat-template-file must survive the strip.
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


def test_strip_shadowing_flags_boolean_does_not_consume_next_token():
    # --spec-default is a boolean shadowing flag; the value-skipping
    # heuristic must skip just the flag, not the following positional.
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
    # The route's already-loaded comparator calls strip_shadowing_flags
    # with no kwargs to detect ANY shadowing flag in stored extras.
    out = strip_shadowing_flags(
        ["-c", "4096", "--cache-type-k", "q8_0", "--spec-default", "--jinja"]
    )
    assert out == []
