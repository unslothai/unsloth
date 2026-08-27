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
parse_gpu_layers_override = _lsa.parse_gpu_layers_override
parse_split_mode_override = _lsa.parse_split_mode_override
resolve_cache_type_kv = _lsa.resolve_cache_type_kv
resolve_tensor_parallel = _lsa.resolve_tensor_parallel
strip_shadowing_flags = _lsa.strip_shadowing_flags
strip_split_mode_only = _lsa.strip_split_mode_only
strip_context_only = _lsa.strip_context_only
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
        # --parallel / -np / --n-parallel are hard-denied; use Parallel Slots.
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


def test_the_attached_value_form_is_refused():
    # llama.cpp looks the whole token up in its option map and folds only the
    # underscore spelling, so "--top-k=20" is an argument it has never heard of.
    # Measured on b10342 and b10360: "error: invalid argument: --top-k=20", and the
    # same for --ctx-size=4096 and --flash-attn=on. Accepting it here meant the
    # switch tore down the resident model and the child then refused to start.
    with pytest.raises(ValueError, match = "two separate arguments"):
        validate_extra_args(["--top-k=20"])
    # The detached spelling is what it takes, and the underscore one still folds.
    assert validate_extra_args(["--top-k", "20"]) == ["--top-k", "20"]
    assert validate_extra_args(["--ctx_size", "4096"]) == ["--ctx_size", "4096"]
    # A managed name is still named as managed: that message says which control
    # owns it, which is the more useful of the two.
    with pytest.raises(ValueError, match = "managed by Unsloth Studio"):
        validate_extra_args(["--parallel=8"])
    # An "=" inside a VALUE is untouched: it is the value's own syntax.
    assert validate_extra_args(["--override-kv", "a=int:2"]) == ["--override-kv", "a=int:2"]


def test_managed_long_flag_underscore_alias_is_rejected():
    with pytest.raises(ValueError, match = "slot-save-path"):
        validate_extra_args(["--slot_save_path", "/tmp/slots"])


def test_a_bare_positional_is_rejected():
    # This used to pass through on the grounds that llama-server can reject it, and
    # it does: "error: invalid argument: foo", so the launch fails instead of the
    # request. Refused here now that a textbox can produce one, because a build that
    # DID accept a positional would read it as the model path, which is exactly what
    # denying -m / --model prevents.
    with pytest.raises(ValueError, match = "bare value"):
        validate_extra_args(["foo"])
    # A value that follows its flag is untouched.
    assert validate_extra_args(["--numa", "distribute"]) == ["--numa", "distribute"]


# ── Denylist (rejected) ──────────────────────────────────────────────


@pytest.mark.parametrize(
    "denied",
    [
        # Parallel slots -- owned by typer --parallel and LoadRequest.n_parallel.
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
        "--pooling",
        # llama-server's own --tools clashes with Unsloth's tool policy.
        "--tools",
        # --agent is --tools by another name ("enable CORS proxy and ALL built-in
        # tools", which includes exec_shell_command), and --tools-runtime says where
        # those tools run -- a container, or another host over ssh.
        "-ag",
        "--agent",
        "-no-ag",
        "--no-agent",
        "--tools-runtime",
        # MCP servers are the same capability from a file or an inline blob.
        "--mcp-servers-config",
        "--mcp-servers-json",
        # Unsloth terminates browser access at its own origin.
        "--cors-origins",
        "--cors-headers",
        "--cors-methods",
        "--cors-credentials",
        "--no-cors-credentials",
        "--media-path",
        # Startup output is how a bad GGUF is told from an OOM from a rejected flag.
        "--log-file",
        "--log-disable",
        # Slot-state dir: Unsloth owns it for KV persistence across idle unload.
        "--slot-save-path",
        # These print and exit instead of serving.
        "-h",
        "--help",
        "--usage",
        "--version",
        "--list-devices",
        "-cl",
        "--cache-list",
        "--completion-bash",
        # Aliases of already-denied UI flags; upstream ships both spellings.
        "--webui-config",
        "--webui-config-file",
        "--webui-mcp-proxy",
        "--no-webui-mcp-proxy",
    ],
)
def test_denylist_rejects_all_aliases(denied):
    with pytest.raises(ValueError, match = denied):
        validate_extra_args([denied, "value"])


@pytest.mark.parametrize(
    "args,offending",
    [
        # Pass-through --parallel would last-wins-override the real slot count
        # while the KV-cache fit and slot bookkeeping stay at the resolved value.
        (["--parallel", "8"], "--parallel"),
        (["--parallel=8"], "--parallel"),
        (["--n-parallel", "16"], "--n-parallel"),
        (["--n-parallel=16"], "--n-parallel"),
        (["-np", "32"], "-np"),
        # Attached short form: Click clusters it CLI-side; HTTP /load with
        # `["-np8"]` must still resolve to managed.
        (["-np8"], "-np"),
        (["-np64"], "-np"),
        # Out-of-range values that would bypass the PARALLEL_MIN/MAX bounds.
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
    # Endpoint exposure stays a user choice: Unsloth reads GET /props and never
    # /slots, so neither flag can strand it.
    assert is_managed_flag("--slots") is False
    assert is_managed_flag("--no-slots") is False
    assert is_managed_flag("--props") is False


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
    # Parallel slots owned by typer --parallel and LoadRequest.n_parallel.
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
    )
    assert out == ["--device", "Vulkan1", "--top-k", "20"]


def test_strip_shadowing_flags_drops_device_when_requested():
    # strip_device drops device placement flags when gpu_ids owns placement.
    for flag in ("--device", "-dev", "--main-gpu", "-mg"):
        out = strip_shadowing_flags(
            [flag, "Vulkan1", "--top-k", "20"],
            strip_context = False,
            strip_cache = False,
            strip_spec = False,
            strip_template = False,
            strip_split_mode = False,
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


# ── parse_gpu_layers_override ────────────────────────────────────────


@pytest.mark.parametrize(
    "args,expected",
    [
        (None, None),
        ([], None),
        (["--top-k", "20"], None),
        (["--gpu-layers", "20"], 20),
        (["--gpu-layers=20"], 20),
        (["--n-gpu-layers", "0"], 0),
        (["-ngl", "-1"], -1),
        (["-ngl", "12", "--gpu-layers", "20"], 20),
    ],
)
def test_parse_gpu_layers_override(args, expected):
    assert parse_gpu_layers_override(args) == expected


@pytest.mark.parametrize(
    "args",
    [
        ["--gpu-layers"],
        ["--gpu-layers", "--top-k"],
        ["--gpu-layers", "abc"],
        ["--gpu-layers=-2"],
    ],
)
def test_parse_gpu_layers_override_rejects_malformed_values(args):
    with pytest.raises(ValueError, match = "gpu-layers|GPU layers"):
        parse_gpu_layers_override(args)


def test_validate_extra_args_rejects_malformed_gpu_layers_override():
    with pytest.raises(ValueError, match = "GPU layers"):
        validate_extra_args(["-ngl", "abc"])


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
    ],
)
def test_split_mode_passes_through(args):
    # Not denylisted -- a user keeps row/none/layer via extras.
    assert validate_extra_args(args) == args


@pytest.mark.parametrize("args", [["--split-mode=row"], ["-sm=tensor"]])
def test_the_attached_split_mode_spelling_is_refused(args):
    # The parsers below still read the attached form, since they also run over
    # Unsloth's own emitted flags; the boundary is where the user's spelling of it
    # is turned back, while the message can still reach them.
    with pytest.raises(ValueError, match = "two separate arguments"):
        validate_extra_args(args)


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


@pytest.mark.parametrize(
    "args",
    [
        ["-c", "0", "--top-k", "20"],
        ["--ctx-size", "0", "--top-k", "20"],
        ["-c=0", "--top-k", "20"],
        ["--ctx-size=0", "--top-k", "20"],
    ],
)
def test_strip_context_only_drops_every_context_form(args):
    # Every -c / --ctx-size form (long/short, space/=) goes; the rest survives.
    assert strip_context_only(args) == ["--top-k", "20"]


def test_strip_context_only_keeps_other_shadow_flags():
    # Only the context group: the cache/spec/template/split flags are untouched.
    assert strip_context_only(["-c", "0", "--split-mode", "row", "--cache-type-k", "q8_0"]) == [
        "--split-mode",
        "row",
        "--cache-type-k",
        "q8_0",
    ]


def test_strip_context_only_preserves_none_and_empty():
    # None means "inherit"; [] means "explicit empty" -- both must round-trip.
    assert strip_context_only(None) is None
    assert strip_context_only([]) == []


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


# --- shape bounds -----------------------------------------------------------
# Not the security boundary (the denylist is), just a floor under what reaches
# execve, so a pasted file fails here naming the limit instead of in the child.


def test_token_count_is_capped():
    with pytest.raises(ValueError, match = "too many"):
        validate_extra_args(["--verbose"] * (_lsa.MAX_EXTRA_ARG_TOKENS + 1))
    # The cap itself still passes, so the limit is inclusive as stated.
    assert len(validate_extra_args(["--verbose"] * _lsa.MAX_EXTRA_ARG_TOKENS)) == (
        _lsa.MAX_EXTRA_ARG_TOKENS
    )


def test_total_size_is_capped():
    with pytest.raises(ValueError, match = "too large"):
        validate_extra_args(["--grammar", "x" * (_lsa.MAX_EXTRA_ARGS_BYTES + 1)])


def test_a_long_single_token_is_allowed_under_the_total():
    # A grammar or JSON schema is legitimately one long token, so the cap is on the
    # list rather than per token.
    schema = "x" * (_lsa.MAX_EXTRA_ARGS_BYTES // 2)
    assert validate_extra_args(["--grammar", schema]) == ["--grammar", schema]


def test_the_size_cap_counts_bytes_not_characters():
    # Astral-plane characters are 4 bytes each; a character-counted cap would let
    # through four times the argv this claims to bound.
    big = "\U0001f600" * (_lsa.MAX_EXTRA_ARGS_BYTES // 4)
    with pytest.raises(ValueError, match = "too large"):
        validate_extra_args(["--grammar", big])


@pytest.mark.parametrize("token", ["a\x00b", "a\x07b", "\x1b[31m"])
def test_control_characters_are_rejected(token):
    with pytest.raises(ValueError, match = "control characters"):
        validate_extra_args([token])


@pytest.mark.parametrize("token", ["line\nbreak", "tab\there"])
def test_tab_and_newline_survive(token):
    # A chat template or grammar passed inline carries both. As its flag's value,
    # which is where such a string actually arrives: a bare one is refused now.
    assert validate_extra_args(["--grammar", token]) == ["--grammar", token]


# --- environment twins ------------------------------------------------------


def test_denied_env_twins_are_scrubbed():
    env = {
        "LLAMA_ARG_AGENT": "1",
        "LLAMA_ARG_TOOLS": "all",
        "LLAMA_ARG_MCP_SERVERS_JSON": "{}",
        "PATH": "/usr/bin",
    }
    removed = _lsa.scrub_denied_env(env)
    assert set(removed) == {"LLAMA_ARG_AGENT", "LLAMA_ARG_TOOLS", "LLAMA_ARG_MCP_SERVERS_JSON"}
    # Only the twins go.
    assert env == {"PATH": "/usr/bin"}


def test_scrubbing_is_a_no_op_without_the_twins():
    env = {"PATH": "/usr/bin", "LLAMA_ARG_MLOCK": "1"}
    assert _lsa.scrub_denied_env(env) == []
    assert env == {"PATH": "/usr/bin", "LLAMA_ARG_MLOCK": "1"}


def test_every_denied_env_var_names_a_denied_flag():
    # The twins are only worth scrubbing while the flag itself is refused; this
    # catches a group being dropped from the denylist and leaving a live back door.
    #
    # Both prefixes, because llama.cpp uses both: --api-key reads LLAMA_API_KEY while
    # --api-key-file reads LLAMA_ARG_API_KEY_FILE, and which one a flag gets has
    # changed between releases.
    for name in _lsa.DENIED_ENV_VARS:
        stem = name.removeprefix("LLAMA_ARG_").removeprefix("LLAMA_")
        flag = _lsa.DENIED_ENV_TWIN_FLAGS.get(name) or "--" + stem.lower().replace("_", "-")
        assert is_managed_flag(flag), f"{name} has no denied flag ({flag})"


def test_every_denied_flag_with_a_twin_in_the_help_is_scrubbed():
    # The list was enumerated from the bundled b10342 --help rather than guessed:
    # every "(env: NAME)" whose option this module refuses. Recorded here as the
    # pairs that mattered, so a name dropped from the denylist, or a twin dropped
    # from the scrub, is a red test rather than a back door found later.
    #
    # llama.cpp applies the environment BEFORE argv, so the ones Unsloth always emits
    # are overridden anyway; the rest are the reason this exists.
    for env_var, flag in (
        ("LLAMA_ARG_UI_MCP_PROXY", "--ui-mcp-proxy"),
        ("LLAMA_ARG_UI", "--ui"),
        ("LLAMA_ARG_STATIC_PATH", "--path"),
        ("LLAMA_ARG_MODELS_DIR", "--models-dir"),
        ("LLAMA_ARG_MODELS_AUTOLOAD", "--models-autoload"),
        ("LLAMA_ARG_EMBEDDINGS", "--embeddings"),
        ("LLAMA_ARG_RERANKING", "--reranking"),
        ("LLAMA_ARG_MODEL", "--model"),
        ("LLAMA_ARG_HF_REPO", "--hf-repo"),
        ("LLAMA_ARG_HOST", "--host"),
        ("LLAMA_ARG_PORT", "--port"),
        ("LLAMA_ARG_N_PARALLEL", "--parallel"),
        ("LLAMA_ARG_SSL_KEY_FILE", "--ssl-key-file"),
        ("LLAMA_ARG_SSL_CERT_FILE", "--ssl-cert-file"),
    ):
        assert is_managed_flag(flag), flag
        assert env_var in _lsa.DENIED_ENV_VARS, env_var
    env = {name: "1" for name in _lsa.DENIED_ENV_VARS}
    env["PATH"] = "/usr/bin"
    # The projector twins are an INPUT here, not a back door: _launch_has_mmproj reads
    # both to know the launch has a projector at all, which is what keeps the vision
    # and audio state of a model loaded through an inherited one. Scrubbing them
    # globally cleared that state (test_gpu_init_crash_message caught it), and only
    # the paravirtual CPU recovery drops them, where an unpinned projector is the
    # corrupt path it is undoing.
    for kept in ("LLAMA_ARG_MMPROJ", "LLAMA_ARG_MMPROJ_URL"):
        assert kept not in _lsa.DENIED_ENV_VARS, kept
    # HF_TOKEN is deliberately not here: it is the standard Hugging Face credential
    # Unsloth's own downloads use, not a llama-server behaviour switch, and the child
    # is always given a local -m path rather than a repo to fetch.
    assert "HF_TOKEN" not in _lsa.DENIED_ENV_VARS
    _lsa.scrub_denied_env(env)
    assert env == {"PATH": "/usr/bin"}


def test_the_projector_env_twins_survive_the_scrub():
    # --mmproj is refused in the box because Unsloth resolves the projector itself,
    # but the environment twin is an INPUT: _launch_has_mmproj reads both names to
    # know the launch has a projector at all, which is what keeps the vision and
    # audio state of a model loaded through an inherited one. Scrubbing them made
    # every such load report itself as text-only.
    env = {
        "LLAMA_ARG_MMPROJ": "/models/mmproj-F16.gguf",
        "LLAMA_ARG_MMPROJ_URL": "https://example.invalid/mmproj-F16.gguf",
        "LLAMA_ARG_AGENT": "1",
    }
    removed = _lsa.scrub_denied_env(env)
    assert removed == ["LLAMA_ARG_AGENT"]
    assert env == {
        "LLAMA_ARG_MMPROJ": "/models/mmproj-F16.gguf",
        "LLAMA_ARG_MMPROJ_URL": "https://example.invalid/mmproj-F16.gguf",
    }
    # Only the paravirtual CPU recovery drops them, where an unpinned projector is
    # the corrupt path it is undoing, and it does that itself.
    assert "LLAMA_ARG_MMPROJ" not in _lsa.DENIED_ENV_VARS
    assert "LLAMA_ARG_MMPROJ_URL" not in _lsa.DENIED_ENV_VARS


# ------------------------------------- an inherited loader mode is a real choice


@pytest.mark.parametrize(
    "env,expected",
    [
        ({}, False),
        (None, False),
        # The enum itself is handler_string, so any value assigns the mode.
        ({"LLAMA_ARG_LOAD_MODE": "mmap"}, True),
        ({"LLAMA_ARG_LOAD_MODE": "dio"}, True),
        # Set but empty selects nothing; upstream would reject it, not default.
        ({"LLAMA_ARG_LOAD_MODE": "  "}, False),
        # --mlock is handler_void: only a truthy value assigns anything.
        ({"LLAMA_ARG_MLOCK": "1"}, True),
        ({"LLAMA_ARG_MLOCK": "0"}, False),
        # The deprecated boolean twins assign the whole mode either way.
        ({"LLAMA_ARG_MMAP": "on"}, True),
        ({"LLAMA_ARG_MMAP": "off"}, True),
        ({"LLAMA_ARG_DIO": "1"}, True),
        # Negative aliases count by PRESENCE: get_value_from_env forces "0".
        ({"LLAMA_ARG_NO_MMAP": "0"}, True),
        ({"LLAMA_ARG_NO_DIO": ""}, True),
        ({"LLAMA_ARG_FIT": "off", "LLAMA_ARG_DEVICE": "none"}, False),
    ],
)
def test_memory_env_selects_load_mode(env, expected):
    assert _lsa.memory_env_selects_load_mode(env) is expected


# --- the shared "is this ctx flag the user's opt-in?" test ----------------------
# Both stripping paths (model_override_load_kwargs on the API auto-switch, and
# _resolve_inherited_extra_args on /load) ask this one function, so a value that
# survives one reload survives the other.


def test_matching_ctx_override_confirms_only_an_exact_positive_int():
    assert _lsa.matches_explicit_ctx_override(["--ctx-size", "100352"], 100352)
    assert _lsa.matches_explicit_ctx_override(["-c", "100352"], 100352)
    # llama.cpp folds the underscore spelling, and so does the matcher.
    assert _lsa.matches_explicit_ctx_override(["--ctx_size", "100352"], 100352)
    # Last-wins, exactly as the launch parses it.
    assert _lsa.matches_explicit_ctx_override(
        ["--ctx-size", "8192", "--ctx-size", "100352"], 100352
    )
    assert not _lsa.matches_explicit_ctx_override(
        ["--ctx-size", "100352", "--ctx-size", "8192"], 100352
    )
    # A different value is a stale shadow, not an opt-in.
    assert not _lsa.matches_explicit_ctx_override(["--ctx-size", "8192"], 100352)
    assert not _lsa.matches_explicit_ctx_override(["--top-k", "40"], 100352)
    assert not _lsa.matches_explicit_ctx_override(None, 100352)


def test_matching_ctx_override_is_total_over_stored_junk():
    # Override rows are coerced on write but returned verbatim on read, so this is
    # reached with whatever JSON an older build, the API or a hand edit left behind.
    # None of it may raise inside a load, and none of it counts as confirmed.
    for n_ctx in (None, 0, -1, "100352", "", "x", 100352.0, True, False, [100352], {"v": 1}):
        assert not _lsa.matches_explicit_ctx_override(["--ctx-size", "100352"], n_ctx), n_ctx
    # A malformed flag raises in parse_ctx_override; the matcher answers False.
    assert not _lsa.matches_explicit_ctx_override(["--ctx-size", "--top-k"], 100352)


# ── The pageable override never resurrects a shadowed lock ──────────────────
# force_pageable_load rewrites an oversized non-mmap launch so the weights page in
# from disk instead of being allocated whole in host RAM. llama.cpp resolves these
# options last-wins, so `--mlock --no-mmap` runs UNLOCKED and unmapped: the strip has
# to read the EFFECTIVE state, not the tokens. Dropping only the selector and leaving
# the earlier --mlock standing hands the child mmap+mlock and page-locks the whole
# oversized mapping into the RAM the override exists to keep pageable.


def _rewritten_state(argv, env = None):
    """``((mlock, reserves_ram), argv, env)`` after the pageable rewrite."""
    env = dict(env or {})
    out, overridden = _lsa.force_pageable_load(list(argv), env)
    return _lsa.resolve_effective_memory_state(out, env), out, env, overridden


@pytest.mark.parametrize(
    "argv",
    [
        ["--mlock", "--no-mmap"],
        ["--mlock", "--no-direct-io"],
        ["--mlock", "--load-mode", "none"],
        ["--load-mode", "mlock", "--no-mmap"],
        ["--load-mode=mlock", "--no-mmap"],
        # The mapped spelling of the lock. It holds no unmapped copy of its own, so
        # the rewrite has no reason to touch it -- until a later reserving selector
        # shadows it, where leaving it standing is what re-locks the mapping.
        ["--load-mode", "mmap+mlock", "--no-mmap"],
        ["--load-mode=mmap+mlock", "--no-mmap"],
        ["--load-mode", "mmap+mlock", "--no-direct-io"],
    ],
    ids = [
        "no-mmap",
        "no-dio",
        "load-mode-none",
        "load-mode-mlock",
        "load-mode-mlock-equals",
        "load-mode-mmap-mlock",
        "load-mode-mmap-mlock-equals",
        "load-mode-mmap-mlock-no-dio",
    ],
)
def test_a_shadowed_lock_is_not_resurrected_by_the_pageable_rewrite(argv):
    # The pre-rewrite child is already unlocked, so there is no lock to carry.
    assert _lsa.resolve_effective_memory_state(argv) == (False, True)

    (mlock, reserves), out, _env, overridden = _rewritten_state(argv)

    assert overridden, f"the unmapped launch was not rewritten at all: {out}"
    assert not reserves, f"the child still holds a full unmapped copy: {out}"
    assert not mlock, f"the rewrite page-locked the oversized mapping: {out}"


@pytest.mark.parametrize(
    "argv, expect_tokens",
    [
        (["--no-mmap", "--mlock"], ["--mlock"]),
        (["--load-mode", "mlock"], ["--load-mode", "mmap+mlock"]),
        (["--load-mode=mlock"], ["--load-mode", "mmap+mlock"]),
    ],
    ids = ["no-mmap-then-mlock", "load-mode-mlock", "load-mode-mlock-equals"],
)
def test_an_effective_lock_survives_the_pageable_rewrite(argv, expect_tokens):
    """The control. "Keep this in RAM" is a real request when nothing shadowed it, so
    it is carried onto a mapping (mmap+mlock) rather than discarded."""
    assert _lsa.resolve_effective_memory_state(argv) == (True, True)

    (mlock, reserves), out, _env, overridden = _rewritten_state(argv)

    assert overridden, f"the unmapped launch was not rewritten at all: {out}"
    assert (mlock, reserves) == (True, False), f"the lock was dropped or kept unmapped: {out}"
    assert out == expect_tokens, out


def test_the_env_twin_of_a_shadowed_lock_goes_too():
    """llama.cpp reads LLAMA_ARG_* before argv and the negative alias after the
    affirmative one, so LLAMA_ARG_MLOCK=1 beside LLAMA_ARG_NO_MMAP is shadowed exactly
    as the argv pair is. Leaving the var behind locks the child through the environment
    with nothing in the argv to show it."""
    env = {"LLAMA_ARG_MLOCK": "1", "LLAMA_ARG_NO_MMAP": "1"}
    assert _lsa.resolve_effective_memory_state([], env) == (False, True)

    (mlock, reserves), _out, out_env, overridden = _rewritten_state([], env)

    assert "LLAMA_ARG_NO_MMAP" not in out_env
    assert "LLAMA_ARG_MLOCK" not in out_env, out_env
    assert "LLAMA_ARG_MLOCK" in overridden, overridden
    assert (mlock, reserves) == (False, False)


def test_the_env_mapped_lock_mode_shadowed_by_an_argv_selector_goes_too():
    """The env half of the mapped spelling. llama.cpp reads LLAMA_ARG_* before argv,
    so an inherited LLAMA_ARG_LOAD_MODE=mmap+mlock followed by --no-mmap on the command
    line is shadowed exactly as the all-argv pair is, and leaving the var behind locks
    the restored mapping through the environment with nothing in the argv to show it."""
    env = {"LLAMA_ARG_LOAD_MODE": "mmap+mlock"}
    assert _lsa.resolve_effective_memory_state(["--no-mmap"], env) == (False, True)

    (mlock, reserves), _out, out_env, overridden = _rewritten_state(["--no-mmap"], env)

    assert "LLAMA_ARG_LOAD_MODE" not in out_env, out_env
    assert overridden, "the unmapped launch was not rewritten at all"
    assert (mlock, reserves) == (
        False,
        False,
    ), f"the rewrite page-locked the oversized mapping: {out_env}"


def test_an_unshadowed_mapped_lock_is_left_entirely_alone():
    """The control that keeps the strip above scoped. ``mmap+mlock`` on its own already
    maps, so it holds no full unmapped copy and is not this override's business: it must
    come back untouched and, with nothing rewritten, report no override at all. Without
    that, a launch that was always pageable would be reported as having been remapped."""
    argv = ["--load-mode", "mmap+mlock"]
    assert _lsa.resolve_effective_memory_state(argv) == (True, False)

    (mlock, reserves), out, _env, overridden = _rewritten_state(argv)

    assert out == argv, out
    assert not overridden, f"an already-mapped launch was reported as overridden: {overridden}"
    assert (mlock, reserves) == (True, False)


def test_the_env_mlock_mode_that_is_not_shadowed_keeps_its_lock():
    """The control for the env half: LLAMA_ARG_LOAD_MODE=mlock with nothing after it
    really is locked, so it becomes the mapped twin instead of being dropped."""
    env = {"LLAMA_ARG_LOAD_MODE": "mlock"}
    assert _lsa.resolve_effective_memory_state([], env) == (True, True)

    (mlock, reserves), _out, out_env, _overridden = _rewritten_state([], env)

    assert out_env["LLAMA_ARG_LOAD_MODE"] == "mmap+mlock"
    assert (mlock, reserves) == (True, False)


def test_an_argv_lock_that_overrides_an_inherited_selector_survives():
    """The cross case, and the reason the rewrite cannot read either side alone: argv
    is resolved after the environment, so --mlock beats an inherited LLAMA_ARG_NO_MMAP
    and the lock is the user's live request."""
    env = {"LLAMA_ARG_NO_MMAP": "1"}
    argv = ["--mlock"]
    assert _lsa.resolve_effective_memory_state(argv, env) == (True, True)

    (mlock, reserves), out, out_env, _overridden = _rewritten_state(argv, env)

    assert "LLAMA_ARG_NO_MMAP" not in out_env
    assert out == ["--mlock"], out
    assert (mlock, reserves) == (True, False)


def test_a_pageable_launch_keeps_every_token_including_its_lock():
    """Nothing here reserves RAM, so the override does not apply and the launch comes
    back byte for byte -- the mlock strip is scoped to a rewrite, not to any --mlock."""
    argv = ["--mlock", "--load-mode", "dio"]
    assert _lsa.resolve_effective_memory_state(argv) == (False, False)

    out, overridden = _lsa.force_pageable_load(list(argv), {})

    assert out == argv and overridden == []
