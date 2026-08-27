# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""install.sh honours UNSLOTH_RESPECT_PM_POLICY, and reads it the way Python does.

The pinned torch install in install.sh runs before install_python_stack.py exists in the
process tree, so the Python opt-out cannot reach it: without the gate an operator who set
the variable would still have their uv.toml bypassed for that install. The predicate is
lifted out of install.sh and executed by a real /bin/sh, then compared value for value
against _respect_pm_policy(). A variable that means one thing to the shell and another to
Python would be worse than no gate at all, and nothing else in the tree would catch it.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
INSTALL_SH = REPO / "install.sh"

sys.path.insert(0, str(REPO / "studio"))
import install_python_stack as ips  # noqa: E402


def _shell_predicate() -> str:
    """The real _respect_pm_policy() out of install.sh, or fail loudly."""
    text = INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(r"^_respect_pm_policy\(\) \{.*?^\}", text, re.MULTILINE | re.DOTALL)
    assert match, "install.sh no longer defines _respect_pm_policy()"
    return match.group(0)


# A value the shell and Python disagree on is the failure this file exists to catch, so
# the set spans the false list, its casing, surrounding space, and ordinary true values.
@pytest.mark.parametrize(
    "value",
    [
        "",
        "0",
        "false",
        "FALSE",
        "False",
        "no",
        "NO",
        " no ",
        " 0 ",
        "\t0\t",
        "1",
        "yes",
        "on",
        "true",
        "2",
        "anything",
        # Internal whitespace: the shell trimmed with `tr -d`, which deletes every space
        # rather than the ends, so `f alse` collapsed to the false spelling here while
        # Python read it as an ordinary non-false value and turned the opt-out ON.
        "f alse",
        "n o",
        "fal se",
        "0 0",
        " no thing ",
        "a b",
        # A lone separator strips to empty on both sides, which is the false spelling.
        " ",
        "\t",
        "\n",
        " \t ",
    ],
)
@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh")
def test_the_shell_and_python_agree_on_the_opt_out(value, monkeypatch):
    script = f"{_shell_predicate()}\nif _respect_pm_policy; then echo ON; else echo OFF; fi\n"
    result = subprocess.run(
        ["sh", "-c", script],
        capture_output = True,
        text = True,
        timeout = 60,
        env = {"UNSLOTH_RESPECT_PM_POLICY": value, "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    shell_says_on = result.stdout.strip() == "ON"

    monkeypatch.setenv("UNSLOTH_RESPECT_PM_POLICY", value)
    assert shell_says_on == ips._respect_pm_policy(), (
        f"install.sh and install_python_stack.py disagree on "
        f"UNSLOTH_RESPECT_PM_POLICY={value!r}: shell={'ON' if shell_says_on else 'OFF'}, "
        f"python={'ON' if ips._respect_pm_policy() else 'OFF'}"
    )


@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh")
def test_an_unset_variable_is_off_on_both_sides(monkeypatch):
    script = f"{_shell_predicate()}\nif _respect_pm_policy; then echo ON; else echo OFF; fi\n"
    result = subprocess.run(
        ["sh", "-c", script],
        capture_output = True,
        text = True,
        timeout = 60,
        env = {"PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "OFF"
    monkeypatch.delenv("UNSLOTH_RESPECT_PM_POLICY", raising = False)
    assert ips._respect_pm_policy() is False


def test_the_pinned_branch_is_gated_on_the_opt_out():
    """Both arms exist, and only the default one discards the operator's uv config."""
    text = INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(
        r'\*" --default-index "\*\)(.*?)^\s*esac',
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "install.sh no longer scrubs the env for --default-index"
    branch = match.group(1)
    assert "_respect_pm_policy" in branch, (
        "the pinned install in install.sh runs before install_python_stack.py, so without "
        "this gate the opt-out cannot reach it"
    )
    opt_out, _, default = branch.partition("else")
    # The additive mirror variables go in BOTH arms: the pin is itself a provenance
    # control, and #6898 is an inherited mirror pulling CPU torch over the CUDA build.
    for arm, name in ((opt_out, "opt-out"), (default, "default")):
        assert "-u UV_INDEX_URL" in arm, f"the {name} arm stopped scrubbing the mirror"
    assert (
        "-u UV_CONFIG_FILE" not in opt_out
    ), "the opt-out arm must leave the operator's uv.toml in place"
    assert (
        "UV_NO_CONFIG=1" not in opt_out
    ), "forcing UV_NO_CONFIG=1 would discard the very policy the opt-out promises to keep"
    assert (
        "-u UV_CONFIG_FILE" in default and "UV_NO_CONFIG=1" in default
    ), "the default path must not move: it is what fixes #6898 and #8530"


def test_the_opt_out_arm_keeps_the_uv_wheelhouse():
    """uv's no-index lives only in uv.toml, so UV_FIND_LINKS cannot be dropped.

    Measured on the pinned uv 0.12.1, a uv.toml `[pip] no-index = true` plus
    UV_FIND_LINKS installs from the wheelhouse and the same command without it fails
    with "index lookups were disabled and no additional package locations were
    provided". install.sh runs its pinned torch install before the Python installer,
    so the opt-out has to keep it here too.
    """
    text = INSTALL_SH.read_text(encoding = "utf-8")
    match = re.search(
        r'\*" --default-index "\*\)(.*?)^\s*esac',
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "install.sh no longer scrubs the env for --default-index"
    opt_out, _, default = match.group(1).partition("else")
    assert (
        "-u UV_FIND_LINKS" not in opt_out
    ), "the opt-out arm must leave the operator's wheelhouse in place"
    assert "-u UV_FIND_LINKS" in default, "the default path must not move: it is what fixes #6898"


def _run_case_arm(value: "str | None") -> str:
    """Execute install.sh's real --default-index arm and return the argv it builds.

    String-matching the branch is not enough: a gate rewritten to something that can
    never be true still contains the words the structural test looks for, so it stays
    green while the opt-out silently stops working. This runs the thing.
    """
    text = INSTALL_SH.read_text(encoding = "utf-8")
    predicate = _shell_predicate()
    start = text.index("run_install_cmd() {")
    body = text[start : text.index("\n}", start)]
    case_start = body.index('case " $* " in')
    case_end = body.index("esac", case_start) + 4
    script = (
        predicate
        + "\nbuild() {\n"
        + body[case_start:case_end]
        + '\nprintf "%s\\n" "$*"\n}\nbuild "$@"\n'
    )
    env = {"PATH": "/usr/bin:/bin"}
    if value is not None:
        env["UNSLOTH_RESPECT_PM_POLICY"] = value
    result = subprocess.run(
        [
            "sh",
            "-c",
            script,
            "sh",
            "uv",
            "pip",
            "install",
            "--default-index",
            "https://download.pytorch.org/whl/cu124",
            "torch",
        ],
        capture_output = True,
        text = True,
        timeout = 60,
        env = env,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh")
def test_the_gate_actually_selects_the_arm_when_the_opt_out_is_set():
    """Executed, not pattern-matched. Fails if the gate becomes unreachable."""
    opted = _run_case_arm("1")
    assert "-u UV_CONFIG_FILE" not in opted, opted
    assert "UV_NO_CONFIG=1" not in opted, opted
    assert "-u UV_FIND_LINKS" not in opted, opted
    # The additive mirror still goes, in both arms.
    assert "-u UV_INDEX_URL" in opted, opted


@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh")
@pytest.mark.parametrize("value", [None, "", "0", "false", "no"])
def test_the_default_arm_is_selected_for_every_off_spelling(value):
    """The default path must be reached whenever the operator has not opted in."""
    built = _run_case_arm(value)
    assert "-u UV_CONFIG_FILE" in built, (value, built)
    assert "UV_NO_CONFIG=1" in built, (value, built)
    assert "-u UV_FIND_LINKS" in built, (value, built)


@pytest.mark.skipif(shutil.which("sh") is None, reason = "no POSIX sh")
def test_an_unset_opt_out_spawns_no_helper_processes():
    """The default path must not pay subprocesses to decide it is the default path.

    `tr` and `sed` were run for every pinned install, including the overwhelming
    majority where the variable is not set at all.
    """
    predicate = _shell_predicate()
    # Poison both helpers: if the unset path reaches them, the shell reports failure.
    script = (
        "tr() { echo 'HELPER-RAN' >&2; return 1; }\n"
        "sed() { echo 'HELPER-RAN' >&2; return 1; }\n"
        + predicate
        + "\nif _respect_pm_policy; then echo ON; else echo OFF; fi\n"
    )
    result = subprocess.run(
        ["sh", "-c", script],
        capture_output = True,
        text = True,
        timeout = 60,
        env = {"PATH": "/usr/bin:/bin"},
    )
    assert result.stdout.strip() == "OFF", result.stdout
    assert (
        "HELPER-RAN" not in result.stderr
    ), "the unset path still spawns tr/sed; it should decide before doing any work"
