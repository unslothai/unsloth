# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for `unsloth studio run` short-alias clashes with
llama-server short flags.

`unsloth studio run` runs with ``ignore_unknown_options=True`` +
``allow_extra_args=True`` so unknown flags pass through to llama-server
(e.g. ``-ngl 32``, ``-c 8192``, ``--top-k 20``).

Before the cleanup it exposed 1-character shorts ``-m`` (--model) and
``-f`` (--frontend) plus ``-hfr`` (--hf-repo). Click's short-option
clustering then silently mis-parsed multi-char llama-server tokens:
``-fa`` -> ``-f a`` (frontend=a), ``-mg 0`` -> ``-m g`` + stray ``0``,
``-fitt 1024`` -> ``-f itt`` + stray ``1024``, etc. The docstring
promise ("any flag this command does not recognize is forwarded
verbatim") was silently violated for ~11 llama-server short flags.

The cleanup removes the colliding 1-char shorts ``-m`` and ``-f`` and
the redundant ``-hfr``. The 2-char ``-hf`` is kept (documented in
basics/api/README.md; Click treats multi-char shorts atomically so it
does not cluster). Long forms ``--model``, ``--hf-repo``, ``--frontend``
remain. ``studio_default`` keeps ``-f`` because it has no pass-through.
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


# Surface-level checks: removed shorts must not reappear.


def test_model_short_aliases_removed():
    """`-m` and `-hfr` were removed from --model. `-hf` is kept
    (documented) and is safe because Click treats multi-char shorts
    atomically (no clustering of `-hff`, `-hfv`, `-hffv`, `-hft`)."""
    decls = _decls_for("model")
    assert "-m" not in decls, (
        "`-m` was re-added; re-introduces Click clustering "
        "(`-mg` -> `-m g`, `-md` -> `-m d`, ...)"
    )
    assert "-hfr" not in decls, "`-hfr` was re-added; remove it"
    assert "--model" in decls
    assert "--hf-repo" in decls
    assert "-hf" in decls, "`-hf` is documented and must keep working"


def test_frontend_short_alias_removed_from_run():
    """`-f` must not be a typer alias on `studio run` (cluster-eats
    `-fa`, `-fit`, `-fitt`, `-fitc`)."""
    decls = _decls_for("frontend")
    assert "-f" not in decls, (
        "`-f` was re-added on run(); re-introduces Click clustering "
        "(`-fa` -> `-f a`, ...)"
    )
    assert "--frontend" in decls


def test_studio_default_keeps_dash_f():
    """`studio_default` keeps `-f` because it has no pass-through args."""
    sig = inspect.signature(_studio_mod().studio_default)
    opt = sig.parameters["frontend"].default
    decls = set(getattr(opt, "param_decls", []) or [])
    assert "-f" in decls


# Behaviour-level checks: each llama-server short flag must survive
# typer parsing and land verbatim in the re-exec'd child argv.


class _ExecCaptured(SystemExit):
    def __init__(self, argv):
        super().__init__(0)
        self.argv = list(argv)


def _install_capture(monkeypatch):
    studio_mod = _studio_mod()
    captured = []
    monkeypatch.setattr(sys, "prefix", "/nonexistent/outer/venv")
    fake_bin = Path("/fake/studio/venv/unsloth_studio/bin/unsloth")
    monkeypatch.setattr(
        studio_mod, "_studio_venv_python", lambda: fake_bin.parent / "python"
    )
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


# Each entry is (short_flag, value, llama-server long name). All of
# these were silently mis-parsed before the cleanup.
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
def test_previously_broken_short_flag_now_passes_through(
    monkeypatch,
    flag,
    value,
    llama_long_name,
):
    """Each of these was silently mis-parsed before the short-alias
    cleanup. They must now pass through to the re-exec'd child verbatim."""
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
    """`-hf` is documented in basics/api/README.md and must keep working.
    Multi-char shorts don't cluster in Click."""
    captured = _invoke(
        monkeypatch,
        ["-hf", "unsloth/gemma-4-26B-A4B-it-GGUF:UD-Q4_K_XL"],
    )
    assert len(captured) == 1
    argv = captured[0]
    # _split_repo_variant in the parent strips the variant suffix
    # before re-exec, so the child sees --model + --gguf-variant.
    assert argv[argv.index("--model") + 1] == ("unsloth/gemma-4-26B-A4B-it-GGUF"), argv
    assert argv[argv.index("--gguf-variant") + 1] == "UD-Q4_K_XL", argv


# Legacy-alias backwards compatibility. -m / -hfr / -f were typer aliases
# pre-PR. Dropping them from typer would break any script using the exact
# tokens, so an in-function preprocessor promotes EXACT matches back into
# their typer parameters while leaving clustered tokens (`-mg`, `-fa`, ...)
# in the llama-server pass-through tail.


@pytest.mark.parametrize(
    "legacy_args,expected_model",
    [
        (["-m", "unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-m=unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-hfr", "unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
        (["-hfr=unsloth/Qwen3-1.7B-GGUF"], "unsloth/Qwen3-1.7B-GGUF"),
    ],
)
def test_legacy_model_aliases_still_promote_to_model(
    monkeypatch,
    legacy_args,
    expected_model,
):
    """Pre-PR `-m X` / `-hfr X` set --model X. The preprocessor must
    keep those scripts working."""
    captured = _invoke(monkeypatch, legacy_args)
    assert len(captured) == 1, f"parent did not re-exec for {legacy_args}"
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == expected_model, argv
    # The legacy alias must not also leak into the pass-through tail.
    for alias in ("-m", "-hfr"):
        if alias in legacy_args:
            assert alias not in argv, f"legacy {alias} leaked into child argv: {argv}"


def test_legacy_frontend_alias_still_promotes_to_frontend(monkeypatch):
    """Pre-PR `-f dist` set --frontend dist. Preprocessor preserves it."""
    captured = _invoke(monkeypatch, ["--model", "X", "-f", "/tmp/dist"])
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--frontend") + 1] == "/tmp/dist", argv
    assert "-f" not in argv, f"-f leaked into child argv: {argv}"


def test_legacy_model_alias_conflicts_with_long_form(monkeypatch):
    """Passing both --model X and -m Y should error -- ambiguous intent."""
    captured = _invoke(monkeypatch, ["--model", "X", "-m", "Y"])
    # The preprocessor raises typer.BadParameter before reaching execvp.
    assert (
        len(captured) == 0
    ), f"expected error before re-exec, got launch with argv = {captured}"


def test_clustered_tokens_are_not_promoted(monkeypatch):
    """`-mg 0`, `-fa`, `-fitt 1024` are llama-server flags, not legacy
    aliases. They must pass through to llama-server even though they
    start with `-m` / `-f`."""
    captured = _invoke(
        monkeypatch,
        ["--model", "X", "-mg", "0", "-fa", "-fitt", "1024"],
    )
    assert len(captured) == 1
    argv = captured[0]
    # Studio --model came from --model X, not from `-mg` cluster.
    assert argv[argv.index("--model") + 1] == "X", argv
    # All three llama-server tokens survive verbatim in the tail.
    for flag in ("-mg", "-fa", "-fitt"):
        assert flag in argv, f"{flag!r} was promoted instead of passed through: {argv}"


def test_legacy_m_with_repo_variant_syntax(monkeypatch):
    """`-m unsloth/foo:UD-Q4_K_XL` must round-trip through the preprocessor
    AND _split_repo_variant so the child sees --model + --gguf-variant."""
    captured = _invoke(
        monkeypatch,
        ["-m", "unsloth/Qwen3-1.7B-GGUF:UD-Q4_K_XL"],
    )
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF", argv
    assert argv[argv.index("--gguf-variant") + 1] == "UD-Q4_K_XL", argv


def test_missing_model_after_preprocessor_errors(monkeypatch):
    """If neither --model nor any legacy alias is supplied, the
    preprocessor's required-check must trigger a clean exit(2)."""
    captured = _invoke(monkeypatch, ["--parallel", "8"])
    # No execvp because the missing-model check raises typer.Exit(2)
    # before reaching the re-exec branch.
    assert (
        len(captured) == 0
    ), f"expected exit before re-exec, got launch with argv = {captured}"


def test_legacy_m_inline_value_form(monkeypatch):
    """`-m=foo` inline form must be promoted just like `-m foo`."""
    captured = _invoke(monkeypatch, ["-m=unsloth/Qwen3-1.7B-GGUF"])
    assert len(captured) == 1
    argv = captured[0]
    assert argv[argv.index("--model") + 1] == "unsloth/Qwen3-1.7B-GGUF", argv


# Direct unit tests for the _consume_legacy_short_aliases helper.


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


# _expand_attached_np_short: Click clusters `-np8` as `-n -p 8` because
# `-p` is the typer short for `--port`. This silently sets port=8 and
# drops the parallel value. The rewrite splits the attached form into
# the space-separated form before Click parses.


def test_expand_np_rewrites_attached_form(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "studio", "run", "--model", "X", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == [
        "unsloth",
        "studio",
        "run",
        "--model",
        "X",
        "-np",
        "8",
    ]


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
    # `-npfoo` is not a valid attached value -- leave for typer to reject.
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


def test_attached_np8_no_longer_silently_sets_port(monkeypatch):
    """Behavioural pin: after _expand_attached_np_short runs in the
    entry-point gate, `-np8` produces --parallel=8 (not --port=8)."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "studio", "run", "--model", "X", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    captured = _invoke(monkeypatch, sys.argv[2:])  # skip "unsloth studio"
    assert len(captured) == 1, "parent did not re-exec"
    argv = captured[0]
    assert argv[argv.index("--parallel") + 1] == "8", argv
    # Port must stay at the typer default (not silently rewritten to 8).
    assert argv[argv.index("--port") + 1] == "8888", argv


def test_expand_np_stops_at_double_dash(monkeypatch):
    """`--` ends option processing; post-`--` `-np8` must stay raw."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["unsloth", "run", "--model", "X", "--", "-np8"],
    )
    _studio_mod()._expand_attached_np_short()
    assert sys.argv == ["unsloth", "run", "--model", "X", "--", "-np8"]


def test_consume_helper_stops_at_double_dash():
    """Legacy alias promotion must not consume tokens past `--`."""
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(
        ["--top-k", "20", "--", "-m", "FOO"],
        ("-m",),
        None,
        "--model",
    )
    assert value is None
    assert remaining == ["--top-k", "20", "--", "-m", "FOO"]


def test_consume_helper_rejects_flag_as_value():
    """`-m -fa` should error: -fa is a flag, not a model name."""
    import typer as _typer

    helper = _studio_mod()._consume_legacy_short_aliases
    with pytest.raises(_typer.BadParameter, match = "-fa"):
        helper(["-m", "-fa"], ("-m",), None, "--model")


def test_consume_helper_allows_bare_dash_as_value():
    """A lone `-` is a valid stdin/path sentinel, not a flag."""
    helper = _studio_mod()._consume_legacy_short_aliases
    value, remaining = helper(["-m", "-", "--top-k", "20"], ("-m",), None, "--model")
    assert value == "-"
    assert remaining == ["--top-k", "20"]
