# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Unit tests for the llama-server pass-through args validator.

The validator is the boundary between user CLI/HTTP input and the
llama-server subprocess. These tests pin denylist behaviour so it doesn't
regress when new managed flags are added.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

# Load llama_server_args.py directly to avoid dragging in the full backend
# chain via core/inference/__init__.py. The validator is dependency-free.
_LSA_PATH = Path(__file__).resolve().parent.parent / "core" / "inference" / "llama_server_args.py"
_spec = importlib.util.spec_from_file_location("_lsa_test_only", _LSA_PATH)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)
is_managed_flag = _lsa.is_managed_flag
parse_cache_override = _lsa.parse_cache_override
parse_cache_override_per_axis = _lsa.parse_cache_override_per_axis
parse_ctx_override = _lsa.parse_ctx_override
parse_split_mode_override = _lsa.parse_split_mode_override
resolve_cache_type_kv = _lsa.resolve_cache_type_kv
resolve_tensor_parallel = _lsa.resolve_tensor_parallel
strip_shadowing_flags = _lsa.strip_shadowing_flags
strip_split_mode_only = _lsa.strip_split_mode_only
extra_args_disable_mmproj = _lsa.extra_args_disable_mmproj
validate_extra_args = _lsa.validate_extra_args


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
        # Soft-managed: user flags last-wins over Unsloth's auto-set version.
        # --parallel / -np / --n-parallel are hard-denied (KV-cache + slot
        # count would desync); use `unsloth studio run --parallel N` instead.
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
        # Memory placement flags (soft-managed; shadowed on inherit)
        ["--mlock"],
        ["--no-mmap", "--mlock"],
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
        # Model identity (every alias; bumping llama.cpp must keep every
        # form rejected, not just the long one).
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
        # Networking (Unsloth binds + proxies)
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
        # Server-mode flips: --embedding / --rerank restrict llama-server to
        # those endpoints and break Unsloth's chat hop.
        "--embedding",
        "--embeddings",
        "--rerank",
        "--reranking",
        # llama-server's own --tools clashes with Unsloth's tool policy.
        "--tools",
        # Slot-state dir: Studio owns it for KV persistence across idle unload.
        "--slot-save-path",
    ],
)
def test_denylist_rejects_all_aliases(denied):
    with pytest.raises(ValueError, match = denied):
        validate_extra_args([denied, "value"])


@pytest.mark.parametrize(
    "args,offending",
    [
        # Pass-through --parallel would last-wins-override the real slot
        # count while Unsloth's KV-cache fit + llama_parallel_slots stay at
        # the typer value -- plan vs. process disagree.
        (["--parallel", "8"], "--parallel"),
        (["--parallel=8"], "--parallel"),
        (["--n-parallel", "16"], "--n-parallel"),
        (["--n-parallel=16"], "--n-parallel"),
        (["-np", "32"], "-np"),
        # Attached short form: Click clusters it CLI-side; HTTP /load with
        # `["-np8"]` must still resolve to managed.
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


def test_slot_save_path_is_managed_in_all_forms():
    for args in (["--slot-save-path", "/tmp/x"], ["--slot-save-path=/tmp/x"], ["--slot-save-path"]):
        with pytest.raises(ValueError, match = "--slot-save-path"):
            validate_extra_args(args)
    assert is_managed_flag("--slot-save-path") is True
    assert is_managed_flag("--slot-save-path=/tmp/x") is True
    # --slots (read-only diagnostics endpoint) stays a user choice.
    assert is_managed_flag("--slots") is False


@pytest.mark.parametrize(
    "padded",
    [" --parallel", "--parallel ", "\t--parallel", "  -np", "-np \n", "-np\t"],
)
def test_denylist_rejects_whitespace_padded_forms(padded):
    # `_flag_name` trims whitespace before lookup; else a trailing space
    # could slip a managed flag past the boundary.
    with pytest.raises(ValueError, match = "parallel|np"):
        validate_extra_args([padded, "8"])


@pytest.mark.parametrize(
    "attached",
    ["-np8x", "-np-1foo", "-np+1bar", "-np9zzz"],
)
def test_denylist_rejects_np_with_digit_prefix_and_junk(attached):
    # Backend `_flag_name` must classify the same forms the CLI rewriter
    # expands, else HTTP /load could smuggle `-np8x` through.
    with pytest.raises(ValueError, match = "np"):
        validate_extra_args([attached])


def test_denylist_rejects_short_form_when_long_is_denied():
    # `-m` is the short form of --model; rejecting only the long form
    # would leave a trivial bypass.
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
    # Memory placement flags are pass-through (shadowed on inherit only).
    assert is_managed_flag("--mlock") is False
    assert is_managed_flag("--no-mmap") is False


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
    assert out == ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--top-k", "20"]


def test_strip_shadowing_flags_keeps_spec_when_spec_disabled():
    out = strip_shadowing_flags(
        ["--spec-type", "ngram-mod", "--draft-min", "48", "--top-k", "20"],
        strip_spec = False,
    )
    assert out == ["--spec-type", "ngram-mod", "--draft-min", "48", "--top-k", "20"]


def test_strip_shadowing_flags_keeps_device_by_default():
    # --device is pass-through by default (users may pin when Unsloth auto-selects).
    out = strip_shadowing_flags(
        ["--device", "Vulkan1", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
        strip_memory_mode = False,
    )
    assert out == ["--device", "Vulkan1", "--top-k", "20"]


def test_strip_shadowing_flags_drops_device_when_requested():
    # strip_device drops --device/-dev + value when explicit gpu_ids owns placement.
    for flag in ("--device", "-dev"):
        out = strip_shadowing_flags(
            [flag, "Vulkan1", "--top-k", "20"],
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
            strip_memory_mode = False,
            strip_device = True,
        )
        assert out == ["--top-k", "20"], flag


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


# ── parse_ctx_override ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "args,expected",
    [
        (None, None),
        ([], None),
        (["--top-k", "20"], None),
        (["--ctx-size", "128000"], 128000),
        (["--ctx-size=128000"], 128000),
        (["-c", "128000"], 128000),
        (["-c=128000"], 128000),
        (["-c", "4096", "--ctx-size", "128000"], 128000),
    ],
)
def test_parse_ctx_override(args, expected):
    assert parse_ctx_override(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["--ctx-size"],
        ["--ctx-size", "--top-k"],
        ["--ctx-size", "abc"],
        ["--ctx-size=abc"],
        ["-c", "-1"],
    ],
)
def test_parse_ctx_override_rejects_malformed_values(args):
    with pytest.raises(ValueError, match = "ctx-size|'-c'"):
        parse_ctx_override(args)


def test_validate_extra_args_rejects_malformed_ctx_override():
    with pytest.raises(ValueError, match = "ctx-size"):
        validate_extra_args(["--ctx-size", "abc"])


# ── parse_cache_override ─────────────────────────────────────────────


@pytest.mark.parametrize(
    "args,expected",
    [
        (None, None),
        ([], None),
        (["--top-k", "20"], None),
        (["--cache-type-k", "q8_0"], "q8_0"),
        (["-ctk", "q4_0"], "q4_0"),
        (["-ctv", "q4_0"], "q4_0"),
        (["--cache-type-k=q4_0"], "q4_0"),
        (["-ctk", "f16", "-ctk", "q8_0"], "q8_0"),
    ],
)
def test_parse_cache_override(args, expected):
    assert parse_cache_override(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["-ctk"],
        ["-ctk", "-c", "4096"],
    ],
)
def test_parse_cache_override_rejects_malformed_values(args):
    with pytest.raises(ValueError, match = "cache-type|'-ctk'"):
        parse_cache_override(args)


@pytest.mark.parametrize(
    "args, expected",
    [
        (["--cache-type-k", "f32", "--cache-type-v", "f16"], ("f32", "f16")),
        (["-ctk", "q8_0", "-ctv", "q4_0"], ("q8_0", "q4_0")),
        (["--cache-type-k=f32"], ("f32", None)),
        (["--cache-type-v", "f16"], (None, "f16")),
        (["-c", "4096"], (None, None)),
        (None, (None, None)),
        # Last-wins is kept per axis.
        (["-ctk", "f16", "-ctk", "f32"], ("f32", None)),
    ],
)
def test_parse_cache_override_per_axis(args, expected):
    # Unlike parse_cache_override (collapses both axes to one last-wins value),
    # this keeps K and V apart so an asymmetric cache can be budgeted per axis.
    assert parse_cache_override_per_axis(args) == expected


def test_resolve_cache_type_kv_uses_override_when_present():
    assert resolve_cache_type_kv(["--cache-type-k", "q8_0"], "f16") == "q8_0"


def test_resolve_cache_type_kv_uses_fallback_without_override():
    assert resolve_cache_type_kv(["--top-k", "20"], "f16") == "f16"


def test_strip_shadowing_flags_boolean_does_not_consume_next_token():
    # `--spec-default` is boolean; drop just the flag, keep the next token.
    out = strip_shadowing_flags(["--spec-default", "ngram-mod"], strip_spec = True)
    assert out == ["ngram-mod"]


def test_strip_shadowing_flags_jinja_boolean_preserves_positional():
    out = strip_shadowing_flags(["--jinja", "trailing-positional"], strip_template = True)
    assert out == ["trailing-positional"]


def test_strip_shadowing_flags_no_jinja_boolean_preserves_positional():
    out = strip_shadowing_flags(["--no-jinja", "trailing-positional"], strip_template = True)
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


# ── --split-mode (Tensor Parallelism toggle) ─────────────────────────
# Soft-shadowed exactly like --cache-type-*: pass-through allowed (keeps
# the row/none/layer modes the boolean toggle doesn't expose), stripped
# on inherit, and reconciled back into the round-tripped tensor_parallel
# state.


@pytest.mark.parametrize(
    "args",
    [
        ["--split-mode", "tensor"],
        ["--split-mode", "row"],
        ["--split-mode", "none"],
        ["--split-mode", "layer"],
        ["-sm", "tensor"],
        ["--split-mode=row"],
        ["-sm=tensor"],
    ],
)
def test_split_mode_passes_through(args):
    # Not denylisted -- a user keeps row/none/layer via extras.
    assert validate_extra_args(args) == args


def test_split_mode_is_not_managed():
    assert is_managed_flag("--split-mode") is False
    assert is_managed_flag("-sm") is False


@pytest.mark.parametrize(
    "args,expected",
    [
        (None, None),
        ([], None),
        (["--top-k", "20"], None),
        (["--split-mode", "tensor"], "tensor"),
        (["--split-mode", "row"], "row"),
        (["-sm", "none"], "none"),
        (["--split-mode=layer"], "layer"),
        (["-sm=tensor"], "tensor"),
        # last-wins when supplied twice
        (["-sm", "row", "--split-mode", "tensor"], "tensor"),
    ],
)
def test_parse_split_mode_override(args, expected):
    assert parse_split_mode_override(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["--split-mode"],
        ["-sm"],
        ["--split-mode", "-c", "4096"],  # next token is a flag, not a value
    ],
)
def test_parse_split_mode_override_rejects_malformed_values(args):
    with pytest.raises(ValueError, match = "split-mode|'-sm'"):
        parse_split_mode_override(args)


def test_validate_extra_args_rejects_malformed_split_mode():
    # Validation catches a value-less --split-mode at the boundary,
    # mirroring the early --ctx-size / --cache-type checks.
    with pytest.raises(ValueError, match = "split-mode"):
        validate_extra_args(["--split-mode"])


@pytest.mark.parametrize(
    "args,fallback,expected",
    [
        # No override -> fall back to the toggle value, both directions.
        (["--top-k", "20"], True, True),
        (["--top-k", "20"], False, False),
        (None, True, True),
        ([], False, False),
        # Explicit override wins: tensor -> on, anything else -> off,
        # regardless of the toggle fallback.
        (["--split-mode", "tensor"], False, True),
        (["-sm", "tensor"], False, True),
        (["--split-mode", "row"], True, False),
        (["--split-mode", "none"], True, False),
        (["--split-mode", "layer"], True, False),
        (["--split-mode=tensor"], False, True),
        # Case-insensitive on the mode string.
        (["--split-mode", "TENSOR"], False, True),
        # last-wins across multiple --split-mode flags.
        (["-sm", "tensor", "--split-mode", "row"], True, False),
    ],
)
def test_resolve_tensor_parallel(args, fallback, expected):
    assert resolve_tensor_parallel(args, fallback) is expected


def test_strip_shadowing_flags_drops_split_mode_when_requested():
    out = strip_shadowing_flags(
        ["--split-mode", "row", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = True,
    )
    assert out == ["--top-k", "20"]


def test_extra_args_disable_mmproj_detects_flag():
    assert extra_args_disable_mmproj(["--no-mmproj"]) is True
    assert extra_args_disable_mmproj(["--threads", "12", "--no-mmproj"]) is True
    assert extra_args_disable_mmproj(["--no-mmproj-auto"]) is True


def test_extra_args_disable_mmproj_false_when_absent():
    assert extra_args_disable_mmproj(None) is False
    assert extra_args_disable_mmproj(["--threads", "12"]) is False


def test_extra_args_disable_mmproj_last_wins():
    assert extra_args_disable_mmproj(["--no-mmproj", "--mmproj-auto"]) is False
    assert extra_args_disable_mmproj(["--mmproj-auto", "--no-mmproj-auto"]) is True


def test_strip_shadowing_flags_drops_model_draft_with_spec():
    # --model-draft (and aliases) are Unsloth-managed since the separate
    # MTP drafter support: an inherited copy must not last-wins-override
    # the auto-detected drafter.
    out = strip_shadowing_flags(
        ["--model-draft", "/old/mtp.gguf", "-md", "/old2.gguf", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = True,
        strip_template = False,
    )
    assert out == ["--top-k", "20"]


@pytest.mark.parametrize(
    "selector",
    [
        ["--spec-draft-hf", "org/repo"],
        ["-hfd", "org/repo"],
        ["-hfrd", "org/repo"],
        ["--hf-repo-draft", "org/repo"],
        ["--spec-draft-hf=org/repo"],
    ],
)
def test_strip_shadowing_flags_drops_hf_drafter_selectors_with_spec(selector):
    # HF drafter selectors must reset on inherit like local --model-draft, or a
    # stale inherited HF drafter last-wins over Unsloth's re-derived spec choice.
    out = strip_shadowing_flags(
        selector + ["--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = True,
        strip_template = False,
    )
    assert out == ["--top-k", "20"]


def test_strip_shadowing_flags_keeps_draft_tuning_with_spec():
    # Per-drafter tuning knobs are deliberately preserved: the VRAM budget reads
    # them via the same parsers the child honors (so they stay consistent on
    # inherit), and stripping --spec-draft-ngl would move a CPU drafter to GPU.
    keep = [
        "--spec-draft-type-k",
        "q4_0",
        "--spec-draft-type-v",
        "q4_0",
        "--spec-draft-ngl",
        "0",
        "--spec-draft-device",
        "cpu",
    ]
    out = strip_shadowing_flags(
        list(keep),
        strip_context = False,
        strip_cache = False,
        strip_spec = True,
        strip_template = False,
    )
    assert out == keep


def test_strip_shadowing_flags_keeps_split_mode_when_not_requested():
    # No tensor_parallel field supplied on the Apply -> an inherited
    # --split-mode survives (mirrors the chat-template keep behavior).
    out = strip_shadowing_flags(
        ["--split-mode", "row", "--top-k", "20"],
        strip_context = True,
        strip_cache = True,
        strip_spec = True,
        strip_template = True,
        strip_split_mode = False,
    )
    assert out == ["--split-mode", "row", "--top-k", "20"]


def test_strip_shadowing_flags_drops_split_mode_short_alias_and_equals():
    assert strip_shadowing_flags(["-sm", "tensor", "--top-k", "20"], strip_split_mode = True) == [
        "--top-k",
        "20",
    ]
    assert strip_shadowing_flags(["--split-mode=row", "--seed", "-1"], strip_split_mode = True) == [
        "--seed",
        "-1",
    ]


def test_strip_shadowing_flags_defaults_strip_split_mode_too():
    # The route's already-loaded comparator (no kwargs) must see a stored
    # --split-mode as a shadowing flag so it forces a reload.
    assert strip_shadowing_flags(["--split-mode", "tensor"]) == []


def test_strip_offload_is_opt_in_and_covers_moe():
    base = dict(
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
    )
    # Default: offload (incl. MoE) flags are NOT stripped.
    assert strip_shadowing_flags(["--n-cpu-moe", "8", "--top-k", "20"], **base) == [
        "--n-cpu-moe",
        "8",
        "--top-k",
        "20",
    ]
    # Opt-in strips layer AND MoE offload flags (value-aware), keeps the rest.
    assert strip_shadowing_flags(
        ["--n-cpu-moe", "8", "--gpu-layers", "33", "--fit", "off", "--top-k", "20"],
        **base,
        strip_offload = True,
    ) == ["--top-k", "20"]
    # Boolean --cpu-moe drops the flag only, not the following value.
    assert strip_shadowing_flags(["--cpu-moe", "--seed", "-1"], **base, strip_offload = True) == [
        "--seed",
        "-1",
    ]


@pytest.mark.parametrize(
    "args",
    [
        ["--split-mode", "tensor", "-c", "4096"],
        ["-sm", "tensor", "-c", "4096"],
        ["--split-mode=tensor", "-c", "4096"],
        ["-sm=tensor", "-c", "4096"],
    ],
)
def test_strip_split_mode_only_keeps_other_shadow_flags(args):
    # Every --split-mode form (long/short, space/=) is dropped; -c survives.
    assert strip_split_mode_only(args) == ["-c", "4096"]


def test_strip_split_mode_only_preserves_none_and_empty():
    # None means "inherit"; [] means "explicit empty" -- both must round-trip.
    assert strip_split_mode_only(None) is None
    assert strip_split_mode_only([]) == []


def test_strip_shadowing_flags_drops_tensor_split_with_split_mode():
    # --tensor-split is coupled to the split mode: stripped together so a stale
    # ratio can't override Unsloth's computed tensor split. Other flags survive.
    out = strip_shadowing_flags(
        ["--split-mode", "row", "--tensor-split", "1,1", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = True,
    )
    assert out == ["--top-k", "20"]


def test_strip_shadowing_flags_keeps_tensor_split_when_not_requested():
    # strip_split_mode=False keeps the whole split group (mode + ratios).
    assert strip_shadowing_flags(
        ["--tensor-split", "1,1", "--top-k", "20"], strip_split_mode = False
    ) == ["--tensor-split", "1,1", "--top-k", "20"]


def test_strip_split_mode_only_drops_tensor_split_too():
    # Downgrade / layer fallback must drop the coupled --tensor-split (all forms).
    assert strip_split_mode_only(
        ["--split-mode", "tensor", "--tensor-split", "1,1", "-c", "4096"]
    ) == ["-c", "4096"]
    assert strip_split_mode_only(["-sm=tensor", "-ts=3,1"]) == []


def test_strip_tensor_split_alone_preserves_split_mode():
    # Manual mode emits its own --tensor-split, so an inherited ratio is dropped
    # -- but the user's --split-mode row/none/layer choice (which the manual
    # ratio toggle can't express) must survive. strip_tensor_split removes only
    # the ratio, unlike strip_split_mode which removes the whole group.
    out = strip_shadowing_flags(
        ["--split-mode", "row", "--tensor-split", "1,1", "--top-k", "20"],
        strip_context = False,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
        strip_split_mode = False,
        strip_tensor_split = True,
    )
    assert out == ["--split-mode", "row", "--top-k", "20"]


def test_strip_shadowing_flags_keeps_model_draft_without_spec():
    out = strip_shadowing_flags(
        ["--model-draft", "/custom/mtp.gguf"],
        strip_context = True,
        strip_cache = False,
        strip_spec = False,
        strip_template = False,
    )
    assert out == ["--model-draft", "/custom/mtp.gguf"]


# ── Memory mode shadowing (#7164) ───────────────────────────────────────────


def test_strip_shadowing_flags_drops_memory_mode_when_requested():
    out = strip_shadowing_flags(
        [
            "--load-mode",
            "dio",
            "--mlock",
            "--no-mmap",
            "--mmap",
            "--direct-io",
            "--top-k",
            "20",
        ],
        strip_memory_mode = True,
    )
    assert out == ["--top-k", "20"]


def test_strip_shadowing_flags_keeps_memory_mode_when_not_requested():
    out = strip_shadowing_flags(
        ["--mlock", "--no-mmap", "--mmap", "--top-k", "20"],
        strip_memory_mode = False,
    )
    assert out == ["--mlock", "--no-mmap", "--mmap", "--top-k", "20"]


def test_strip_shadowing_flags_default_keeps_memory_mode():
    # Memory-mode stripping is opt-in (like offload/tensor_split/device): the default
    # must PRESERVE inherited --mlock/--mmap/--no-mmap so an Apply that omits
    # gguf_memory_mode doesn't silently drop a user's pass-through memory flag (#7188).
    assert strip_shadowing_flags(["--mlock", "--no-mmap", "--mmap"]) == [
        "--mlock",
        "--no-mmap",
        "--mmap",
    ]


def test_strip_split_mode_only_keeps_memory_mode():
    # Tensor->layer downgrade must not strip the user's memory mode choice.
    assert strip_split_mode_only(["--mlock", "--no-mmap", "--mmap", "-sm", "tensor"]) == [
        "--mlock",
        "--no-mmap",
        "--mmap",
    ]


def test_strip_shadowing_flags_drops_inverse_mmap_flag():
    # An inherited --mmap must not override Unsloth's resident-mode --no-mmap (#7164).
    out = strip_shadowing_flags(["--mmap"], strip_memory_mode = True)
    assert out == []


@pytest.mark.parametrize("flag", ["--mlock", "--no-mmap", "--mmap"])
def test_strip_memory_mode_valueless_preserves_next_token(flag):
    # The memory-mode flags take no value; stripping must not consume the next token.
    out = strip_shadowing_flags([flag, "--top-k", "40"], strip_memory_mode = True)
    assert out == ["--top-k", "40"]


def test_strip_memory_mode_kept_when_field_not_supplied():
    # Pass-through preserved when the user didn't change gguf_memory_mode.
    out = strip_shadowing_flags(["--mmap"], strip_memory_mode = False)
    assert out == ["--mmap"]
