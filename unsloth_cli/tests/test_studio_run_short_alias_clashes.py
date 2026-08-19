# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for short-alias clashes with llama-server flags.

`unsloth studio run` passes unknown flags through to llama-server.
Pre-cleanup it exposed 1-char shorts ``-m`` / ``-f`` plus ``-hfr``;
Click clustered llama-server tokens against them (``-fa`` -> ``-f a``,
``-mg 0`` -> ``-m g``, ``-fitt 1024`` -> ``-f itt``, ...), silently
breaking ~11 pass-through flags.

The cleanup drops ``-m``, ``-f``, ``-hfr``. The 2-char ``-hf`` stays
(documented; multi-char shorts don't cluster). Long forms remain.
``studio_default`` keeps ``-f`` because it has no pass-through.

See ``test_studio_run_parallel_flag.py`` for ``--parallel`` /
``-np`` coverage and re-exec forwarding.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _studio_mod():
    from unsloth_cli.commands import studio as _s
    return _s


def _decls_for(param_name):
    sig = inspect.signature(_studio_mod().run)
    opt = sig.parameters[param_name].default
    return set(getattr(opt, "param_decls", []) or [])


# Surface checks: removed shorts must not reappear.


def test_model_short_aliases_removed():
    """`-m` / `-hfr` removed from --model; `-hf` kept (multi-char,
    doesn't cluster)."""
    decls = _decls_for("model")
    assert "-m" not in decls, "`-m` re-added; brings back `-mg`/`-md` clustering"
    assert "-hfr" not in decls, "`-hfr` was re-added; remove it"
    assert "--model" in decls
    assert "--hf-repo" in decls
    assert "-hf" in decls, "`-hf` is documented and must keep working"


def test_frontend_short_alias_removed_from_run():
    """`-f` must not be on `run` (eats `-fa`/`-fit`/`-fitt`/`-fitc`)."""
    decls = _decls_for("frontend")
    assert "-f" not in decls, "`-f` re-added on run(); brings back `-fa` clustering"
    assert "--frontend" in decls


def test_studio_default_keeps_dash_f():
    """`studio_default` keeps `-f`: no pass-through tail to clash with."""
    sig = inspect.signature(_studio_mod().studio_default)
    opt = sig.parameters["frontend"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "-f" in decls


# Behaviour checks: llama-server shorts must reach the child verbatim.


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_capture(monkeypatch):
    studio_mod = _studio_mod()
    captured = []
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_bin = Path("/fake/studio/venv/unsloth_studio/bin/unsloth")
    monkeypatch.setattr(studio_mod, "_studio_venv_python", lambda: fake_bin.parent / "python")
    real_is_file = Path.is_file
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda self: True if str(self) == str(fake_bin) else real_is_file(self),
    )
    from unsloth_cli import _tool_policy as _tp

    monkeypatch.setattr(
        _tp,
        "resolve_tool_policy",
        lambda host, flag, yes, silent: False if flag is None else bool(flag),
    )
    monkeypatch.setattr(sys, "platform", "linux")

    def fake_execvp(file, argv):
        captured.append(list(argv))
        raise _ExecCaptured(argv)

    monkeypatch.setattr(studio_mod.os, "execvp", fake_execvp)
    return captured


def _invoke(monkeypatch, args):
    import typer as _typer

    studio_mod = _studio_mod()
    captured = _install_capture(monkeypatch)
    app = _typer.Typer()
    app.command(
        context_settings = {
            "allow_extra_args": True,
            "ignore_unknown_options": True,
        },
    )(studio_mod.run)
    CliRunner().invoke(app, args, catch_exceptions = True)
    return captured


# (short_flag, value, llama-server long name). All were silently
# mis-parsed pre-cleanup.
_PREVIOUSLY_BROKEN = [
    ("-fa", None, "--flash-attn"),
    ("-fit", None, "--fit"),
    ("-fitt", "1024", "--fit-target"),
    ("-fitc", "4096", "--fit-ctx"),
    ("-mg", "0", "--main-gpu"),
    ("-md", "/path/draft.gguf", "--spec-draft-model"),
    ("-hff", "Q4_K_M.gguf", "--hf-file"),
    ("-cmoe", None, "--cpu-moe"),
    ("-cram", "16384", "--cache-ram"),
    ("-sm", "row", "--split-mode"),
    ("-ncmoe", "8", "--n-cpu-moe"),
]


@pytest.mark.parametrize("flag,value,llama_long_name", _PREVIOUSLY_BROKEN)
def test_previously_broken_short_flag_now_passes_through(monkeypatch, flag, value, llama_long_name):
    """Each of these was eaten by typer pre-cleanup; must pass through verbatim now."""
    extras = [flag] if value is None else [flag, value]
    captured = _invoke(monkeypatch, ["--model", "X"] + extras)
    assert len(captured) == 1, f"parent did not re-exec for {extras}"
    argv = captured[0]
    assert flag in argv, (
        f"llama-server short flag {flag!r} ({llama_long_name}) was eaten "
        f"by typer; child argv = {argv}"
    )
    if value is not None:
        idx = argv.index(flag)
        assert (
            idx + 1 < len(argv) and argv[idx + 1] == value
        ), f"value for {flag!r} was lost or moved; argv = {argv}"


def test_dash_hf_documented_alias_still_works(monkeypatch):
    """`-hf` must keep working: multi-char shorts don't cluster in Click."""
    captured = _invoke(
        monkeypatch,
        ["-hf", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"],
    )
    assert len(captured) == 1
    argv = captured[0]
    # `_split_repo_variant` peels the `:variant` suffix before re-exec.
    assert argv[argv.index("--model") + 1] == ("unsloth/gemma-4-26B-A4B-it-GGUF"), argv
    assert argv[argv.index("--gguf-variant") + 1] == "UD-Q4_K_XL", argv


# Legacy `-m` / `-hfr` / `-f` were typer aliases pre-PR. The
# preprocessor promotes EXACT matches back to their typer params and
# leaves clustered tokens (`-mg`, `-fa`, ...) in the pass-through tail.


@pytest.mark.parametrize(
    "legacy_args,expected_model",
    [
        (["-m", "unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-m=unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-hfr", "unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-hfr=unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
    ],
)
def test_legacy_model_aliases_still_promote_to_model(monkeypatch, legacy_args, expected_model):
    """Pre-PR `-m X` / `-hfr X` set --model X; preprocessor preserves that."""
    captured = _invoke(monkeypatch, legacy_args)
    assert len(captured) == 1, f"parent did not re-exec for {legacy_args}"
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == expected_model, argv
    # Promoted alias must not also leak into the pass-through tail.
    for alias in ("-m", "-hfr"):
        if alias in legacy_args:
            assert alias not in argv, f"legacy {alias} leaked into child argv: {argv}"


def test_legacy_frontend_alias_still_promotes_to_frontend(monkeypatch):
    """Pre-PR `-f dist` set --frontend dist; preprocessor preserves it."""
    captured = _invoke(monkeypatch, ["--model", "X", "-f", "/tmp/dist"])
    assert len(captured) == 1
    argv = captured[0]
    # Compare via Path: on Windows str(Path("/tmp/dist")) = "\tmp\dist".
    assert Path(argv[argv.index("--frontend") + 1]) == Path("/tmp/dist"), argv
    assert "-f" not in argv, f"-f leaked into child argv: {argv}"


def test_legacy_model_alias_conflicts_with_long_form(monkeypatch):
    """`--model X` plus `-m Y` is ambiguous; must error pre-re-exec."""
    captured = _invoke(monkeypatch, ["--model", "X", "-m", "Y"])
    assert len(captured) == 0, f"expected error before re-exec, got launch with argv = {captured}"


def test_clustered_tokens_are_not_promoted(monkeypatch):
    """`-mg` / `-fa` / `-fitt` are llama-server flags and must survive
    in the tail even though they start with `-m` / `-f`."""
    captured = _invoke(
        monkeypatch,
        ["--model", "X", "-mg", "0", "-fa", "-fitt", "1024"],
    )
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == "X", argv
    for flag in ("-mg", "-fa", "-fitt"):
        assert flag in argv, f"{flag!r} was promoted instead of passed through: {argv}"


def test_legacy_m_with_repo_variant_syntax(monkeypatch):
    """`-m repo:variant` must round-trip through preprocessor +
    _split_repo_variant into --model + --gguf-variant."""
    captured = _invoke(
        monkeypatch,
        ["-m", "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL"],
    )
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF", argv
    assert argv[argv.index("--gguf-variant") + 1] == "UD-Q4_K_XL", argv


def test_missing_model_after_preprocessor_errors(monkeypatch):
    """Neither --model nor a legacy alias → clean exit(2) before re-exec."""
    captured = _invoke(monkeypatch, ["--parallel", "8"])
    assert len(captured) == 0, f"expected exit before re-exec, got launch with argv = {captured}"


def test_legacy_m_inline_value_form(monkeypatch):
    """`-m=foo` is promoted like `-m foo`."""
    captured = _invoke(monkeypatch, ["-m=unsloth/Qwen3-1.7B-GGUF"])
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF", argv


# Unit tests for _consume_legacy_short_aliases.


def test_consume_helper_exact_match_space_form():
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["-m", "FOO", "--top-k", "20"],
        ("-m",),
        None,
        "--model",
    )
    assert value == "FOO"
    assert remaining == ["--top-k", "20"]


def test_consume_helper_exact_match_inline_form():
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["-m=FOO", "--top-k", "20"],
        ("-m",),
        None,
        "--model",
    )
    assert value == "FOO"
    assert remaining == ["--top-k", "20"]


def test_consume_helper_leaves_clusters_alone():
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["-mg", "0", "-md", "/x"],
        ("-m",),
        None,
        "--model",
    )
    assert value is None
    assert remaining == ["-mg", "0", "-md", "/x"]


def test_consume_helper_value_already_set_raises():
    helper = _studio_mod()._consume_legacy_short_aliases
    import typer as _typer
    with pytest.raises(_typer.BadParameter):
        helper(["-m", "Y"], ("-m",), "X", "--model")


def test_consume_helper_missing_value_raises():
    helper = _studio_mod()._consume_legacy_short_aliases
    import typer as _typer
    with pytest.raises(_typer.BadParameter):
        helper(["-m"], ("-m",), None, "--model")


def test_consume_helper_multiple_aliases_in_group():
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["-hfr", "FOO", "--top-k", "20"],
        ("-m", "-hfr"),
        None,
        "--model",
    )
    assert value == "FOO"
    assert remaining == ["--top-k", "20"]


def test_consume_helper_preserves_value_when_no_match():
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["--top-k", "20"],
        ("-m",),
        "PRESET",
        "--model",
    )
    assert value == "PRESET"
    assert remaining == ["--top-k", "20"]


# `-p` is typer short for --port, so Click clusters `-np8` as `-n -p 8`
# (port=8, parallel dropped). The rewrite splits to `-np 8` pre-parse.


def test_expand_np_rewrites_attached_form(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "studio", "run", "--model", "X", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "studio", "run", "--model", "X", "-np", "8"]


@pytest.mark.parametrize("value", ["1", "8", "64", "999"])
def test_expand_np_rewrites_all_digit_values(monkeypatch, value):
    monkeypatch.setattr(sys, "argv", ["unsloth", "studio", "run", f"-np{value}"])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "studio", "run", "-np", value]


def test_expand_np_leaves_space_form_alone(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", "-np", "8"])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np", "8"]


def test_expand_np_leaves_equals_form_alone(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", "-np=8"])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np=8"]


def test_expand_np_leaves_non_digit_suffix_alone(monkeypatch):
    # `-npfoo` isn't a numeric attached value; let typer reject it.
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", "-npfoo"])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-npfoo"]


def test_expand_np_leaves_bare_np_alone(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", "-np"])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np"]


def test_expand_np_handles_multiple_occurrences(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "run", "-np8", "-np16"],
    )
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np", "8", "-np", "16"]


@pytest.mark.parametrize("attached,expected", [("-np-1", "-1"), ("-np+1", "+1")])
def test_expand_np_handles_signed_attached_forms(monkeypatch, attached, expected):
    """Signed `-np-1` / `-np+1` must split too, else Click sets port=-1."""
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", attached])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np", expected]


@pytest.mark.parametrize(
    "attached,expected_suffix",
    [("-np8x", "8x"), ("-np-1foo", "-1foo"), ("-np9bar", "9bar")],
)
def test_expand_np_rewrites_numeric_prefix_even_with_junk(monkeypatch, attached, expected_suffix):
    """`-np8x` would surface as a baffling --port error; rewriting to
    `-np 8x` makes typer report against `-np` where it was typed."""
    monkeypatch.setattr(sys, "argv", ["unsloth", "run", attached])
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "-np", expected_suffix]


def test_consume_helper_rejects_empty_inline_value():
    """`-m=` must error, not silently become --model ''."""
    import typer as _typer

    helper = _studio_mod()._consume_legacy_short_aliases
    with pytest.raises(_typer.BadParameter, match = "non-empty"):
        helper(["-m="], ("-m",), None, "--model")


# Gate isolation: importing unsloth_cli from a third-party script must
# leave its sys.argv intact. Pins the narrow basename set.


@pytest.mark.parametrize(
    "third_party_argv0",
    [
        "/home/user/myproj/cli.py",
        "cli.py",
        "/usr/bin/some-tool",
        "pytest",
        "/opt/wrapper/launch.py",
        "unsloth-cli",
        "unsloth-cli.py",
    ],
)
def test_third_party_importers_do_not_trigger_np_rewrite(monkeypatch, third_party_argv0):
    """Only the `unsloth` / `unsloth.exe` console-script may run the
    canonicaliser; third-party scripts must keep their argv intact."""
    import os as _os
    import importlib

    starting_argv = [third_party_argv0, "subcmd", "-np8", "--input", "foo"]
    monkeypatch.setattr(sys, "argv", list(starting_argv))
    # Force a fresh import so the import-time gate actually runs.
    monkeypatch.delitem(sys.modules, "unsloth_cli", raising = False)
    importlib.import_module("unsloth_cli")
    assert sys.argv == starting_argv, (
        f"third-party argv[0]={third_party_argv0!r} triggered the "
        f"unsloth -np canonicaliser; sys.argv was mutated to {sys.argv}"
    )
    _ = _os  # silence unused-import linters when monkeypatch lazy-binds


def test_attached_np8_no_longer_silently_sets_port(monkeypatch):
    """After the gate runs, `-np8` produces --parallel=8 (not --port=8)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "studio", "run", "--model", "X", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    captured = _invoke(monkeypatch, sys.argv[2:])  # drop "unsloth studio"
    assert len(captured) == 1, "parent did not re-exec"
    argv = captured[0]
    assert argv[argv.index("--parallel") + 1] == "8", argv
    assert argv[argv.index("--port") + 1] == "8888", argv


def test_expand_np_stops_at_double_dash(monkeypatch):
    """Tokens after `--` are positional; `-np8` stays raw."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "run", "--model", "X", "--", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "--model", "X", "--", "-np8"]


def test_consume_helper_stops_at_double_dash():
    """Alias promotion must not reach past `--`."""
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["--top-k", "20", "--", "-m", "FOO"],
        ("-m",),
        None,
        "--model",
    )
    assert value is None
    assert remaining == ["--top-k", "20", "--", "-m", "FOO"]


def test_consume_helper_rejects_long_flag_as_value():
    """`-m --flash-attn` errors; `--xxx` is unambiguously a flag."""
    import typer as _typer

    helper = _studio_mod()._consume_legacy_short_aliases
    with pytest.raises(_typer.BadParameter, match = "--flash-attn"):
        helper(["-m", "--flash-attn"], ("-m",), None, "--model")


def test_consume_helper_allows_bare_dash_as_value():
    """Lone `-` is a stdin/path sentinel, not a flag."""
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(["-m", "-", "--top-k", "20"], ("-m",), None, "--model")
    assert value == "-"
    assert remaining == ["--top-k", "20"]


def test_consume_helper_allows_short_dash_value():
    """`-foo` may be a path or a leading-dash model name; only `--long`
    tokens are rejected as values."""
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(["-m", "-foo", "--top-k", "20"], ("-m",), None, "--model")
    assert value == "-foo"
    assert remaining == ["--top-k", "20"]
