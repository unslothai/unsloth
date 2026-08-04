# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Guards the Colab oracle drift tripwire in scripts/notebook_validator.py.

History: `colab-diff --strict` is the daily cron's escalation of the advisory
PR-time check, and it had two defects that cancelled each other out badly.

  * notebooks-ci.yml ran `refresh-colab` (which overwrites the committed pip
    snapshot in place) BEFORE the diff, so the pip leg compared upstream
    against a copy of itself and could never report drift -- the one oracle a
    rule actually reads.
  * `refresh-colab` only knew how to fetch pip-freeze, so the apt-list and
    os-info snapshots had no acknowledgement path at all and drifted until
    --strict failed on Ubuntu security bumps that nothing can consult.

Net effect: the cron was permanently red for apt churn while blind to 233
entries of real pip drift. These tests pin both halves of the fix.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import notebook_validator as nv  # noqa: E402

PIP = "torch==2.10.0\naccelerate==1.13.0\n"
APT = "curl/jammy,now 7.81.0-1ubuntu1.24 amd64 [installed]\n"
OS_INFO = "R version 4.5.3\n"

UPSTREAM = {
    "pip-freeze.gpu.txt": PIP,
    "apt-list-gpu.txt": APT,
    "os-info-gpu.txt": OS_INFO,
}


@pytest.fixture
def oracle(tmp_path, monkeypatch):
    """A snapshot dir that matches upstream, plus a stubbed fetch. Mutate the
    returned dict to make upstream differ from what is committed on disk."""
    upstream = dict(UPSTREAM)
    for upstream_name, snapshot_name in nv.COLAB_ORACLE_FILES.items():
        (tmp_path / snapshot_name).write_text(UPSTREAM[upstream_name], encoding = "utf-8")

    def fake_urlopen(url, timeout = None):
        name = url.rsplit("/", 1)[-1]
        return io.BytesIO(upstream[name].encode("utf-8"))

    monkeypatch.setattr(nv.urllib.request, "urlopen", fake_urlopen)
    return upstream, tmp_path


def _diff(snapshot_dir, strict):
    return nv.cmd_colab_diff(argparse.Namespace(snapshot_dir = str(snapshot_dir), strict = strict))


def test_no_drift_is_clean(oracle):
    _, snapshot_dir = oracle
    assert _diff(snapshot_dir, strict = True) == 0


def test_pip_drift_fails_strict(oracle):
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = "torch==2.10.0\naccelerate==1.14.0\n"
    assert _diff(snapshot_dir, strict = True) == 1


def test_pip_drift_is_advisory_without_strict(oracle):
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = "torch==2.10.0\naccelerate==1.14.0\n"
    assert _diff(snapshot_dir, strict = False) == 0


@pytest.mark.parametrize(
    "name, drifted",
    [
        ("apt-list-gpu.txt", "curl/jammy,now 7.81.0-1ubuntu1.25 amd64 [installed]\n"),
        ("os-info-gpu.txt", "R version 4.6.0\n"),
    ],
)
def test_non_rule_oracles_never_fail_strict(oracle, capsys, name, drifted):
    """An Ubuntu security bump or an R release must not turn the cron red:
    nothing resolves a rule against these two files."""
    upstream, snapshot_dir = oracle
    upstream[name] = drifted
    assert _diff(snapshot_dir, strict = True) == 0
    # Reported, just not fatal -- the signal is the point, the failure was not.
    out = capsys.readouterr().out
    assert "CHANGED" in out
    assert "::notice::" in out


def test_strict_oracle_is_the_one_lint_pins_against(oracle):
    """--colab-pin is fed COLAB_FALLBACK_FILE, and that is the snapshot whose
    drift is fatal. If these ever diverge the tripwire is guarding the wrong
    file again."""
    assert nv.COLAB_ORACLE_FILES[nv.COLAB_STRICT_ORACLE] == nv.COLAB_FALLBACK_FILE.name


def test_refresh_all_writes_every_snapshot(oracle, tmp_path):
    """The acknowledgement path: one command has to be able to clear a drift
    report, or the snapshots rot until --strict fires on them."""
    upstream, _ = oracle
    for key in upstream:
        upstream[key] = upstream[key].replace("2.10.0", "2.11.0").replace("4.5.3", "4.6.0")
        upstream[key] = upstream[key].replace("1ubuntu1.24", "1ubuntu1.25")
    out_dir = tmp_path / "fresh"
    rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    assert rc == 0
    for upstream_name, snapshot_name in nv.COLAB_ORACLE_FILES.items():
        assert (out_dir / snapshot_name).read_text(encoding = "utf-8") == upstream[upstream_name]
    assert _diff(out_dir, strict = True) == 0


def test_refresh_without_all_still_writes_only_pip(oracle, tmp_path):
    """Back-compat: notebooks-ci.yml still calls the single-file form to feed
    `lint --colab-pin` live data."""
    _, _ = oracle
    # A dir of its own: the fixture pre-seeds tmp_path with all three snapshots.
    dest = tmp_path / "pip_only"
    out = dest / "just_pip.txt"
    rc = nv.cmd_refresh_colab(argparse.Namespace(all = False, snapshot_dir = str(dest), out = str(out)))
    assert rc == 0
    assert out.read_text(encoding = "utf-8") == PIP
    assert sorted(p.name for p in dest.iterdir()) == ["just_pip.txt"]


def test_workflow_diffs_before_it_refreshes():
    """The ordering bug itself: refresh-colab overwrites the committed pip
    snapshot, so a diff placed after it compares upstream with upstream."""
    wf = (REPO_ROOT / ".github/workflows/notebooks-ci.yml").read_text(encoding = "utf-8")
    for job in ("static", "static-with-pypi"):
        rest = wf.split(f"\n  {job}:", 1)[1]
        # Up to the next job key, which is the next line indented exactly two.
        nxt = re.search(r"^  [A-Za-z0-9_-]+:$", rest, re.M)
        body = rest[: nxt.start()] if nxt else rest

        # Match the invocations, not the prose: both steps are described in
        # comments that name the other subcommand.
        def _at(sub):
            m = re.search(rf"notebook_validator\.py {sub}\b", body)
            return m.start() if m else -1

        diff_at, refresh_at = _at("colab-diff"), _at("refresh-colab")
        assert diff_at != -1, f"{job} lost its colab-diff step"
        assert refresh_at != -1, f"{job} lost its refresh-colab step"
        assert diff_at < refresh_at, (
            f"{job} refreshes the snapshot before diffing it, which makes the "
            "pip leg of colab-diff structurally unable to report drift"
        )
