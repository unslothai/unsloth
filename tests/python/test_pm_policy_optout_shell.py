# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""UNSLOTH_RESPECT_PM_POLICY: install.sh's predicate against install_python_stack.py's.

The opt-out is read at four entry points and every one of them has to answer the same way,
because a variable that means "on" to the shell half of an install and "off" to the Python
half leaves the operator with neither the relaxations nor the refusal they asked for.

Nothing here retypes a predicate. The sh function is lifted out of install.sh by regex and
executed by a real /bin/sh, so what is compared is the shipped text and not a paraphrase of
it; the Python side is imported. The run_install_cmd `case` statement is lifted and executed
the same way, so the default (no opt-out) arm is pinned by what it actually does to argv,
not by a reviewer's memory of what it should.

Run: python -m pytest -q tests/python/test_pm_policy_optout_shell.py
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_DIR = REPO_ROOT / "studio"
sys.path.insert(0, str(STUDIO_DIR))

import install_python_stack as ips  # noqa: E402

INSTALL_SH = REPO_ROOT / "install.sh"
INSTALL_SH_SOURCE = INSTALL_SH.read_text(encoding = "utf-8")

# A deliberately small PATH: the predicate may only lean on tools every POSIX host has.
# If a future edit reaches for a coreutil that is not in /bin or /usr/bin, these run under
# an environment that will say so instead of quietly borrowing the developer's PATH.
MINIMAL_PATH = "/usr/bin:/bin"

HAVE_SH = shutil.which("sh") is not None
requires_sh = pytest.mark.skipif(not HAVE_SH, reason = "needs a POSIX /bin/sh")


# ---------------------------------------------------------------------------
# Lifting the real text out of install.sh
# ---------------------------------------------------------------------------

def _extract_sh_function(name: str) -> str:
    """The shipped text of a top-level `name() { ... }` in install.sh.

    Top-level only, hence the line-anchored closing brace: install.sh indents every nested
    block, so the first column-zero `}` is this function's and no inner one can end it early.
    """
    match = re.search(
        r"^" + re.escape(name) + r"\(\)\s*\{\n.*?^\}\n",
        INSTALL_SH_SOURCE,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, f"{name}() not found in install.sh -- the test's regex has drifted"
    return match.group(0)


RESPECT_FN_SH = _extract_sh_function("_respect_pm_policy")
RUN_INSTALL_CMD_SH = _extract_sh_function("run_install_cmd")

# The index-var scrub, lifted out of run_install_cmd. Anchored at four-space indentation
# because that is the one `case` at the top level of the function body.
_CASE_MATCH = re.search(
    r"^    case \" \$\* \" in\n.*?^    esac\n",
    RUN_INSTALL_CMD_SH,
    re.MULTILINE | re.DOTALL,
)
assert _CASE_MATCH is not None, "run_install_cmd's `case \" $* \" in` block not found"
DEFAULT_INDEX_CASE_SH = _CASE_MATCH.group(0)


def test_lifted_text_is_the_real_thing():
    """Guard the lift itself: an empty or truncated extraction must not read as a pass."""
    assert "UNSLOTH_RESPECT_PM_POLICY" in RESPECT_FN_SH
    assert RESPECT_FN_SH.count("\n") >= 4
    assert RESPECT_FN_SH.rstrip().endswith("}")
    assert "--default-index" in DEFAULT_INDEX_CASE_SH
    assert DEFAULT_INDEX_CASE_SH.count("set -- env") == 2, (
        "expected exactly two arms (opt-out and default) in run_install_cmd's scrub"
    )
    assert "_respect_pm_policy" in DEFAULT_INDEX_CASE_SH, (
        "the scrub no longer consults the opt-out predicate"
    )


# ---------------------------------------------------------------------------
# The shared value matrix
# ---------------------------------------------------------------------------

# None means "variable not set at all", which is a different input from "set to empty".
MATRIX: tuple[object, ...] = (
    None,
    "",
    # The allowlist, and the near misses that must stay off.
    "1", "0", "true", "TRUE", "True", "TrUe", "yes", "YES", "on", "ON", "On",
    "false", "no", "off", "garbage", "2", "-1", "enabled", "onn", "ye",
    # 't' and 'y' are deliberately NOT members, though setup.ps1's neighbouring boolish
    # predicates take them. A copy-paste between the two would widen the opt-out silently.
    "t", "T", "y", "Y", "n", "f",
    # Leading / trailing / tab / newline padding. An exported value picks these up from a
    # heredoc, a CI matrix cell or a copy-paste, so they are not hypothetical.
    " 1", "1 ", " 1 ", "  yes  ", "\ttrue", "true\t", "\t on \t", "\n1", "true\n",
    "on\n", "\non\n", "\r\n1", "1\r\n", "1\r", "\v1", "\f1",
    # INTERNAL whitespace. An earlier revision of the shell side expanded the variable
    # unquoted, which let the shell word-split and collapse the inside of a value; "t rue"
    # and "o n" then matched the allowlist in sh while Python rejected them, and "1 1"
    # became two words. Quoted now -- these cases exist to keep it that way.
    "t rue", "o n", "1 1", "y es", "tr ue", "1  1", "t\true", "o\nn", "y e s",
    # UNICODE whitespace. str.strip() and .NET Trim() remove these; POSIX sh cannot
    # portably, so all three trim an explicit ASCII set instead and these read as
    # unrecognised, hence OFF, everywhere. Before that they were ON to Python and
    # PowerShell and OFF to sh, so the shell phase relaxed a policy the Python phase
    # withheld. A pasted non-breaking space is the realistic one.
    "\xa01", "1\xa0", "\xa0true\xa0", "\u20021", "1\u3000", "\u200a1", "\u20281",
)


def _label(value: object) -> str:
    return "<unset>" if value is None else repr(value)


def _env_for(value: object) -> dict[str, str]:
    env = {"PATH": MINIMAL_PATH}
    if value is not None:
        env["UNSLOTH_RESPECT_PM_POLICY"] = value  # type: ignore[assignment]
    return env


def _sh_predicate_status(value: object) -> int:
    """Exit status of install.sh's OWN _respect_pm_policy() under a real /bin/sh."""
    script = RESPECT_FN_SH + "\n_respect_pm_policy\n"
    proc = subprocess.run(
        ["sh", "-c", script],
        env = _env_for(value),
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.stderr == "", f"sh predicate wrote to stderr for {_label(value)}: {proc.stderr!r}"
    return proc.returncode


def _python_predicate_status(value: object) -> int:
    """Exit-status form of ips._respect_pm_policy(), so the two are directly comparable."""
    saved = os.environ.get(ips._POLICY_OPT_OUT_ENV)
    try:
        if value is None:
            os.environ.pop(ips._POLICY_OPT_OUT_ENV, None)
        else:
            os.environ[ips._POLICY_OPT_OUT_ENV] = value  # type: ignore[assignment]
        return 0 if ips._respect_pm_policy() else 1
    finally:
        if saved is None:
            os.environ.pop(ips._POLICY_OPT_OUT_ENV, None)
        else:
            os.environ[ips._POLICY_OPT_OUT_ENV] = saved


ALLOWLIST = ("1", "true", "yes", "on")

# The set all three trim. NOT str.strip(), which also removes Unicode whitespace that POSIX
# sh cannot portably match; see _respect_pm_policy().
ASCII_WHITESPACE = " \t\n\r\v\f"


def _sh_semantics(value: object) -> int:
    """What a correct reading of the shipped sh text predicts, modelled independently.

    `case "$(printf '%s' "$VAR" | tr ... | tr ... | sed ...)"` -- the value is quoted
    throughout, so nothing is word-split or glob-expanded. The first tr maps every other
    whitespace character to a space (sed cannot trim across a newline), the second lowers
    case, and the sed trims both ends: together, an ASCII-only .strip().lower(). The trim is
    at the ENDS only, so "t rue" is the true spelling on neither side, and it is ASCII only,
    so a Unicode-padded value is unrecognised rather than trimmed.
    """
    raw = "" if value is None else value
    assert isinstance(raw, str)
    return 0 if raw.strip(ASCII_WHITESPACE).lower() in ALLOWLIST else 1


def _python_semantics(value: object) -> int:
    raw = "" if value is None else value
    assert isinstance(raw, str)
    return 0 if raw.strip(ASCII_WHITESPACE).lower() in ALLOWLIST else 1


@requires_sh
@pytest.mark.parametrize("value", MATRIX, ids = [_label(v) for v in MATRIX])
def test_sh_predicate_matches_its_own_documented_semantics(value):
    """The shipped sh text does exactly what reading it says it does.

    Run first and separately from the cross-language comparison: if the shell ever starts
    collapsing internal whitespace again, this is the assertion that names the cause instead
    of just reporting a mismatch with Python.
    """
    assert _sh_predicate_status(value) == _sh_semantics(value), (
        f"install.sh's _respect_pm_policy() did not behave as its text reads for {_label(value)}"
    )


@pytest.mark.parametrize("value", MATRIX, ids = [_label(v) for v in MATRIX])
def test_python_predicate_matches_its_own_documented_semantics(value):
    assert _python_predicate_status(value) == _python_semantics(value), (
        f"ips._respect_pm_policy() did not behave as its text reads for {_label(value)}"
    )


@requires_sh
@pytest.mark.parametrize(
    "value",
    MATRIX,
    ids = [_label(v) for v in MATRIX],
)
def test_sh_and_python_agree_value_by_value(value):
    """Exit status compared value-by-value, for every input the two are meant to share.

    This includes every internal-whitespace case: "t rue", "o n", "1 1" must read as OFF on
    both sides. They are the regression that a quoted expansion buys.
    """
    sh_status = _sh_predicate_status(value)
    py_status = _python_predicate_status(value)
    assert sh_status == py_status, (
        f"install.sh and install_python_stack.py disagree for {_label(value)}: "
        f"sh exit {sh_status}, Python exit {py_status}"
    )


@requires_sh
def test_internal_whitespace_is_off_on_both_sides():
    """Explicit, non-parametrised restatement of the historical bug, so it reads in a log."""
    for value in ("t rue", "o n", "1 1", "y es", "tr ue", "1  1", "t\true", "y e s"):
        assert _sh_predicate_status(value) == 1, f"sh accepted internal-whitespace {value!r}"
        assert _python_predicate_status(value) == 1, f"Python accepted internal-whitespace {value!r}"


@requires_sh
def test_the_two_predicates_never_disagree():
    """No value in the matrix may be ON to one entry point and OFF to another.

    The whole promise is that one variable means one thing for a whole install. A padded
    value that the shell read as off while Python read it as on would apply the shell's
    relaxations to the pinned torch install and withhold Python's from everything after --
    the operator getting neither the opt-out they asked for nor the default they did not.
    That is worse than having no gate, so it is asserted over the ENTIRE matrix rather than
    over a recorded allowance: an exception list is how the disagreement got there once.
    """
    disagreed = {
        value: (_sh_predicate_status(value), _python_predicate_status(value))
        for value in MATRIX
        if value is not None
        and _sh_predicate_status(value) != _python_predicate_status(value)
    }
    assert not disagreed, (
        "install.sh and install_python_stack.py disagree on the opt-out for:\n"
        + "\n".join(
            f"  {_label(v)}: sh={'on' if sh == 0 else 'off'}, python={'on' if py == 0 else 'off'}"
            for v, (sh, py) in sorted(disagreed.items())
        )
    )


# ---------------------------------------------------------------------------
# run_install_cmd's --default-index scrub: the arm the opt-out must NOT have changed
# ---------------------------------------------------------------------------

ADDITIVE_SCRUB = ("UV_DEFAULT_INDEX", "UV_INDEX_URL", "UV_INDEX", "UV_EXTRA_INDEX_URL", "UV_TORCH_BACKEND")
POLICY_BEARING_SCRUB = ("UV_FIND_LINKS", "UV_CONFIG_FILE")


def _case_arms() -> tuple[str, str]:
    """(opt-out arm, default arm) -- the two `set -- env ...` lines, in source order."""
    lines = [ln.strip() for ln in DEFAULT_INDEX_CASE_SH.splitlines() if "set -- env" in ln]
    assert len(lines) == 2, f"expected 2 scrub arms, got {len(lines)}: {lines!r}"
    # The if/else is written opt-out first; assert that rather than assume it, or a swapped
    # pair would silently reassign every expectation below.
    body = DEFAULT_INDEX_CASE_SH
    assert body.index("if _respect_pm_policy; then") < body.index(lines[0])
    opt_out_at = body.index(lines[0])
    else_at = body.index("else")
    assert opt_out_at < else_at, "the opt-out arm is no longer the `then` branch"
    return lines[0], lines[1]


def test_default_arm_of_run_install_cmd_is_unchanged():
    """The pre-existing behaviour, quoted flag for flag. The opt-out is additive or it is a bug."""
    _, default_arm = _case_arms()
    flags = re.findall(r"-u (\w+)", default_arm)
    assert flags == list(ADDITIVE_SCRUB) + list(POLICY_BEARING_SCRUB), (
        f"default arm's -u flags changed: {flags!r}"
    )
    assert len(flags) == 7, f"default arm must carry seven -u flags, carries {len(flags)}"
    assert "UV_NO_CONFIG=1" in default_arm, "default arm no longer forces UV_NO_CONFIG=1"


def test_opt_out_arm_carries_only_the_additive_five():
    opt_out_arm, _default = _case_arms()
    flags = re.findall(r"-u (\w+)", opt_out_arm)
    assert flags == list(ADDITIVE_SCRUB), f"opt-out arm's -u flags changed: {flags!r}"
    assert len(flags) == 5, f"opt-out arm must carry exactly five -u flags, carries {len(flags)}"
    for kept in POLICY_BEARING_SCRUB:
        assert kept not in opt_out_arm, f"opt-out arm still scrubs {kept}"
    assert "UV_NO_CONFIG" not in opt_out_arm, "opt-out arm still forces UV_NO_CONFIG=1"


@requires_sh
@pytest.mark.parametrize(
    ("policy_value", "expect_policy_bearing_scrubbed"),
    [(None, True), ("", True), ("0", True), ("garbage", True), ("1", False), ("TRUE", False), ("on", False)],
    ids = ["unset", "empty", "0", "garbage", "1", "TRUE", "on"],
)
def test_run_install_cmd_case_executes_as_written(policy_value, expect_policy_bearing_scrubbed):
    """Run the lifted `case` under sh and read the argv it produces.

    The textual assertions above can only see what the file says; this sees what sh does with
    it -- including that the arms are reachable at all and that the guard is the right way up.
    """
    script = (
        RESPECT_FN_SH
        + "\n_scrub_argv() {\n"
        + "    shift\n"
        + DEFAULT_INDEX_CASE_SH
        + "    for _a in \"$@\"; do printf '%s\\n' \"$_a\"; done\n"
        + "}\n"
        + "_scrub_argv 'install torch' uv pip install --default-index https://example.invalid/simple torch\n"
    )
    proc = subprocess.run(
        ["sh", "-c", script],
        env = _env_for(policy_value),
        capture_output = True,
        text = True,
        timeout = 60,
    )
    assert proc.returncode == 0, f"lifted case failed under sh: {proc.stderr!r}"
    argv = proc.stdout.splitlines()
    assert argv[0] == "env", f"the scrub did not prepend env: {argv!r}"
    unset = [argv[i + 1] for i, tok in enumerate(argv) if tok == "-u"]
    for name in ADDITIVE_SCRUB:
        assert name in unset, f"{name} must be scrubbed on every arm; argv={argv!r}"
    for name in POLICY_BEARING_SCRUB:
        assert (name in unset) is expect_policy_bearing_scrubbed, (
            f"{name} scrub={name in unset}, expected {expect_policy_bearing_scrubbed}; argv={argv!r}"
        )
    assert ("UV_NO_CONFIG=1" in argv) is expect_policy_bearing_scrubbed, (
        f"UV_NO_CONFIG=1 presence wrong for {_label(policy_value)}; argv={argv!r}"
    )
    # The command itself must survive intact on both arms.
    assert argv[-4:] == ["install", "--default-index", "https://example.invalid/simple", "torch"], (
        f"the scrub mangled the command: {argv!r}"
    )


@requires_sh
def test_non_default_index_commands_are_left_alone():
    """No --default-index means no scrub at all, on either arm. The opt-out must not widen it."""
    script = (
        RESPECT_FN_SH
        + "\n_scrub_argv() {\n    shift\n"
        + DEFAULT_INDEX_CASE_SH
        + "    for _a in \"$@\"; do printf '%s\\n' \"$_a\"; done\n}\n"
        + "_scrub_argv 'install x' uv pip install --index-url https://example.invalid/simple torch\n"
    )
    for value in (None, "1", "0"):
        proc = subprocess.run(
            ["sh", "-c", script], env = _env_for(value),
            capture_output = True, text = True, timeout = 60,
        )
        assert proc.returncode == 0, proc.stderr
        argv = proc.stdout.splitlines()
        assert argv[0] == "uv", f"untouched command was rewritten for {_label(value)}: {argv!r}"
        assert "-u" not in argv


def test_python_side_withholds_sdist_exemption_under_opt_out():
    """The Python predicate is load-bearing, not decorative: one caller pinned end to end."""
    assert _python_predicate_status("1") == 0
    saved = os.environ.get(ips._POLICY_OPT_OUT_ENV)
    try:
        os.environ.pop(ips._POLICY_OPT_OUT_ENV, None)
        assert ips._sdist_only_build_args("argbind") == ["--no-binary", "argbind"]
        os.environ[ips._POLICY_OPT_OUT_ENV] = "yes"
        assert ips._sdist_only_build_args("argbind") == []
        os.environ[ips._POLICY_OPT_OUT_ENV] = "garbage"
        assert ips._sdist_only_build_args("argbind") == ["--no-binary", "argbind"]
    finally:
        if saved is None:
            os.environ.pop(ips._POLICY_OPT_OUT_ENV, None)
        else:
            os.environ[ips._POLICY_OPT_OUT_ENV] = saved
