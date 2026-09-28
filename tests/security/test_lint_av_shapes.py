# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""scripts/lint_av_shapes.py: every rule fires, the repository is clean against its baseline,
and each false positive that reached a user's machine is caught at the commit that shipped it.

Fixture text is joined from fragments, like the lint's own, so this file does not carry the
shapes it checks for.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "lint_av_shapes.py"

_spec = importlib.util.spec_from_file_location("lint_av_shapes", SCRIPT)
L = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = L
_spec.loader.exec_module(L)

_J = "".join


def _rules(relative: str, text: str) -> dict:
    found: dict = {}
    for finding in L.scan_text(relative, text):
        found.setdefault(finding.rule, set()).add(finding.severity)
    return found


def test_the_self_test_passes():
    assert L.self_test() == 0


def test_the_repository_is_clean_against_its_baseline():
    result = subprocess.run(
        [sys.executable, str(SCRIPT)], cwd = REPO, capture_output = True, text = True, timeout = 300
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_every_error_group_in_the_baseline_has_a_reason():
    document = json.loads(
        (REPO / "scripts" / "av_shapes_baseline.json").read_text(encoding = "utf-8")
    )
    bare = [
        (g["file"], g["rule"])
        for g in document["groups"]
        if g.get("severity") == "error" and (not g.get("reason") or g["reason"] == "REVIEW ME")
    ]
    assert not bare, bare


def test_a_new_line_fails_and_says_what_to_do_instead(tmp_path):
    probe = tmp_path / "probe.ps1"
    probe.write_text("$b = $m." + _J(("Define", "PInvoke", "Method")) + "('x')\n", encoding = "utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--paths", str(probe), "--json", str(tmp_path / "out.json")],
        cwd = REPO, capture_output = True, text = True, timeout = 120,
    )  # fmt: skip
    assert result.returncode == 1, result.stdout
    assert "[AV003] error" in result.stdout
    assert "why:" in result.stdout and "fix:" in result.stdout
    report = json.loads((tmp_path / "out.json").read_text(encoding = "utf-8"))
    assert [entry["rule"] for entry in report] == ["AV003"]


def test_a_warning_never_fails_the_build(tmp_path):
    probe = tmp_path / "probe.ps1"
    probe.write_text("$sb = [scriptblock]::Create($text)\n", encoding = "utf-8")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--paths", str(probe)],
        cwd = REPO,
        capture_output = True,
        text = True,
    )
    assert result.returncode == 0, result.stdout
    assert "[AV015] warn" in result.stdout


def test_inline_allow_needs_the_rule_and_a_reason_and_is_ignored_in_shipped_installers():
    line = "$sb = [scriptblock]::Create($text)"
    assert "AV015" not in _rules(
        "t.ps1", line + "  # lint-allow: AV015 loads a function body under test"
    )
    assert "AV015" in _rules("t.ps1", line + "  # lint-allow: AV015")
    assert "AV015" in _rules(
        "t.ps1", line + "  # lint-allow: AV003 loads a function body under test"
    )
    assert "AV015" in _rules(
        "install.ps1", line + "  # lint-allow: AV015 loads a function body under test"
    )


def test_the_documented_one_liner_is_not_flagged_but_a_third_party_pipe_is():
    ours = "Write-Host 'irm https://unsloth.ai/install.ps1 | " + _J(("i", "ex")) + "'"
    theirs = "irm https://example.invalid/install.ps1 | " + _J(("i", "ex"))
    assert "AV005" not in _rules("t.ps1", ours)
    assert _rules("t.ps1", theirs).get("AV005") == {"error"}


def test_one_credential_marker_warns_and_two_fail():
    base = "import os, requests\nhome = os.environ['HOME']\nrequests.get(u)\n"
    one = base + "p = '" + _J(("id_", "rsa")) + "'\n"
    two = one + "q = '" + _J(("wallet", ".dat")) + "'\n"
    assert _rules("s.py", one).get("AV008") == {"warn"}
    assert _rules("s.py", two).get("AV008") == {"error"}
    assert "AV008" not in _rules("s.py", two.replace("requests.get(u)", "pass"))


def _at(revision: str, path: str) -> str:
    result = subprocess.run(["git", "show", f"{revision}:{path}"], cwd = REPO, capture_output = True)
    if result.returncode != 0:
        pytest.skip(f"{revision}:{path} is not in this clone's history (shallow checkout)")
    return result.stdout.decode("utf-8", "replace")


# (commit, file, rule the shipped version must raise); the version after the fix must raise no error.
INCIDENTS = [
    ("99c392887^", "99c392887", "scripts/scan_packages.py", "AV008"),
    ("bd612c863^", "bd612c863", "install.ps1", "AV003"),
    ("268b771b9", None, "tests/python/test_windows_setup_output_encoding.py", "AV001"),
    ("268b771b9", None, "tests/studio/install/test_keep_install_backcompat_9979.py", "AV009"),
    ("268b771b9", None, "tests/security/test_release_desktop_signing_simulation.py", "AV009"),
    ("268b771b9", None, "tests/studio/test_install_rollback_lifecycle.ps1", "AV010"),
]


@pytest.mark.parametrize("before, after, path, rule", INCIDENTS, ids = [i[2] for i in INCIDENTS])
def test_each_shipped_incident_is_caught_and_its_fix_is_not(before, after, path, rule):
    assert rule in _rules(path, _at(before, path))
    if after is not None:
        errors = {r for r, sev in _rules(path, _at(after, path)).items() if "error" in sev}
        assert not errors, errors
