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
    validate_extra_args,
)


# ── Pass-through (allowed) ───────────────────────────────────────────


@pytest.mark.parametrize(
    "args",
    [
        ["--top-k", "20"],
        ["--top-p", "0.9", "--min-p", "0.05"],
        ["--seed", "-1"],  # negative value, not a flag
        ["--temp", "0.0"],
        ["--repeat-penalty", "1.05"],
        ["--mirostat", "2", "--mirostat-lr", "0.1"],
        ["--cache-type-k", "q8_0"],  # tier-2: allowed pass-through
        ["--cache-type-v", "q8_0"],
        ["--chat-template-file", "/tmp/tpl.jinja"],
        ["--spec-type", "ngram-mod"],
        ["--xtc-probability", "0.05", "--xtc-threshold", "0.1"],
        ["--dry-multiplier", "0.5"],
        ["--reasoning-format", "deepseek"],
        ["-rea", "auto"],
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
        # Networking
        "--host",
        "--port",
        "--path",
        "--api-prefix",
        # Auth / TLS
        "--api-key",
        "--api-key-file",
        "--ssl-key-file",
        "--ssl-cert-file",
        # Forced perf / context
        "-c",
        "--ctx-size",
        "-np",
        "--parallel",
        "-fa",
        "--flash-attn",
        "--no-context-shift",
        "--context-shift",
        "--jinja",
        "--no-jinja",
        # GPU placement
        "-fit",
        "--fit",
        "-ngl",
        "--gpu-layers",
        "--n-gpu-layers",
        "-t",
        "--threads",
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


def test_denylist_rejects_short_form_when_long_is_managed():
    # -c is the short form of the managed --ctx-size; rejecting only
    # the long form would leave a trivial bypass.
    with pytest.raises(ValueError, match = "-c"):
        validate_extra_args(["-c", "8192"])


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


def test_is_managed_flag_true_for_managed():
    assert is_managed_flag("--port") is True
    assert is_managed_flag("-c") is True
    assert is_managed_flag("--ctx-size") is True


def test_is_managed_flag_false_for_pass_through():
    assert is_managed_flag("--top-k") is False
    assert is_managed_flag("--cache-type-k") is False
    assert is_managed_flag("--chat-template-file") is False
