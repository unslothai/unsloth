# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Guards colab-diff --strict; refresh-colab must not overwrite the pip snapshot before the diff."""

from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.error
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import notebook_validator as nv  # noqa: E402

# A real pin file: a stub missing the R-INST seed packages tests the refusal, not the drift.
PIP = (
    "torch==2.10.0\ntorchcodec==0.10.0\npeft==0.19.0\ntorchao==0.16.0\n"
    "transformers==5.1.0\ntokenizers==0.23.0\naccelerate==1.13.0\n"
)
APT = "curl/jammy,now 7.81.0-1ubuntu1.24 amd64 [installed]\n"
# The Python line is rule-bearing under COLAB_STRICT_ORACLE_KEYS.
OS_INFO = "Python 3.13.15\nR version 4.5.3\n"

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


def _diff(
    snapshot_dir,
    strict,
    full = False,
):
    return nv.cmd_colab_diff(
        argparse.Namespace(snapshot_dir = str(snapshot_dir), strict = strict, full = full)
    )


def test_no_drift_is_clean(oracle):
    _, snapshot_dir = oracle
    assert _diff(snapshot_dir, strict = True) == 0


def test_pip_drift_fails_strict(oracle):
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = PIP.replace("accelerate==1.13.0", "accelerate==1.14.0")
    assert _diff(snapshot_dir, strict = True) == 1


def test_pip_drift_is_advisory_without_strict(oracle):
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = PIP.replace("accelerate==1.13.0", "accelerate==1.14.0")
    assert _diff(snapshot_dir, strict = False) == 0


@pytest.mark.parametrize(
    "name, drifted",
    [
        ("apt-list-gpu.txt", "curl/jammy,now 7.81.0-1ubuntu1.25 amd64 [installed]\n"),
        # Only the R release changes, which no rule consults.
        ("os-info-gpu.txt", "Python 3.13.15\nR version 4.6.0\n"),
    ],
)
def test_non_rule_oracles_never_fail_strict(oracle, capsys, name, drifted):
    """An Ubuntu security bump or an R release must not turn the cron red:
    nothing resolves a rule against these two files."""
    upstream, snapshot_dir = oracle
    upstream[name] = drifted
    assert _diff(snapshot_dir, strict = True) == 0
    out = capsys.readouterr().out
    assert "CHANGED" in out
    assert "::notice::" in out


def test_a_capped_listing_says_how_to_see_the_rest(oracle, capsys):
    """A capped listing must say how to see the rest, as every pip-oracle package is rule-bearing."""
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = "".join(f"pkg{i}=={i}.0\n" for i in range(200)) + PIP

    assert _diff(snapshot_dir, strict = False) == 0

    capped = capsys.readouterr().out
    assert "more new entries (--full to list them)" in capped, capped
    assert capped.count("  NEW      ") == 50, capped


def test_full_lists_every_drifted_entry(oracle, capsys):
    upstream, snapshot_dir = oracle
    upstream["pip-freeze.gpu.txt"] = "".join(f"pkg{i}=={i}.0\n" for i in range(200)) + PIP

    assert _diff(snapshot_dir, strict = False, full = True) == 0

    full = capsys.readouterr().out
    assert "more new entries" not in full, full
    assert full.count("  NEW      ") == 200, full
    assert "NEW      pkg199==199.0" in full, full


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
        # Up to the next job key, the next line indented exactly two.
        nxt = re.search(r"^  [A-Za-z0-9_-]+:$", rest, re.M)
        body = rest[: nxt.start()] if nxt else rest

        # Match invocations, not prose: comments name the other subcommand.
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


def test_refresh_all_is_atomic(oracle, tmp_path, monkeypatch):
    """A failed refresh leaves no mixed snapshot set, since a partial pip write would silence --strict."""
    upstream, snapshot_dir = oracle
    for key in upstream:
        upstream[key] = "REFRESHED\n"

    def flaky(url, timeout = None):
        name = url.rsplit("/", 1)[-1]
        if name != nv.COLAB_STRICT_ORACLE:
            raise urllib.error.URLError("network down")
        return io.BytesIO(upstream[name].encode("utf-8"))

    monkeypatch.setattr(nv.urllib.request, "urlopen", flaky)
    rc = nv.cmd_refresh_colab(
        argparse.Namespace(all = True, snapshot_dir = str(snapshot_dir), out = None)
    )
    assert rc == 2
    for upstream_name, snapshot_name in nv.COLAB_ORACLE_FILES.items():
        assert (snapshot_dir / snapshot_name).read_text(encoding = "utf-8") == UPSTREAM[
            upstream_name
        ], f"{snapshot_name} was overwritten even though the refresh failed"


def test_refresh_all_writes_nothing_into_a_fresh_dir_on_failure(oracle, tmp_path, monkeypatch):
    """Same guarantee when the destination did not exist yet: no half-populated
    directory is left behind for a later --strict to read as clean."""
    upstream, _ = oracle

    def dead(url, timeout = None):
        raise urllib.error.URLError("network down")

    monkeypatch.setattr(nv.urllib.request, "urlopen", dead)
    dest = tmp_path / "never_created"
    assert nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(dest), out = None)) == 2
    assert not dest.exists()


def test_cron_lint_survives_a_strict_drift_failure():
    """The strict step exits 1 on a Colab rotation, which is exactly when the
    live-PyPI pass is worth having. Without `if: always()` on the steps after
    it, the job would only ever lint on the days nothing drifted."""
    wf = (REPO_ROOT / ".github/workflows/notebooks-ci.yml").read_text(encoding = "utf-8")
    rest = wf.split("\n  static-with-pypi:", 1)[1]
    nxt = re.search(r"^  [A-Za-z0-9_-]+:$", rest, re.M)
    body = rest[: nxt.start()] if nxt else rest
    strict_at = body.index("--strict")
    for step in ("Refresh Colab oracle", "Lint with live PyPI metadata"):
        at = body.index(f"- name: {step}")
        assert at > strict_at, f"{step} must stay after the strict gate"
        blk = body[at:]
        end = blk.find("\n      - name:", 1)
        blk = blk[:end] if end != -1 else blk
        # The directive on its own line: prose in the step's comment mentions `if: always()`.
        assert re.search(
            r"^\s*if: always\(\)\s*$", blk, re.M
        ), f"{step} would be skipped when strict drift fires"


def test_a_missing_strict_snapshot_fails_strict(oracle):
    """A missing strict snapshot must fail --strict; the rules read it, so absence is not no drift."""
    _, snapshot_dir = oracle
    (snapshot_dir / nv.COLAB_ORACLE_FILES["os-info-gpu.txt"]).unlink()
    assert _diff(snapshot_dir, strict = True) == 1


def test_a_missing_pip_snapshot_fails_strict(oracle):
    """A missing pip snapshot must fail --strict; the oracle fixture hands back one directory per test."""
    _, snapshot_dir = oracle
    (snapshot_dir / nv.COLAB_ORACLE_FILES["pip-freeze.gpu.txt"]).unlink()
    assert _diff(snapshot_dir, strict = True) == 1


def test_a_missing_advisory_snapshot_stays_advisory(oracle):
    """apt-list carries no rule-bearing key, so its absence must not redden the cron."""
    _, snapshot_dir = oracle
    (snapshot_dir / nv.COLAB_ORACLE_FILES["apt-list-gpu.txt"]).unlink()
    assert _diff(snapshot_dir, strict = True) == 0


def test_a_strict_key_absent_from_both_oracles_fails_strict(oracle):
    """Present in BOTH, not merely equal in both.

    An upstream format change acknowledged into the snapshot leaves the two parses identical and
    empty of the key, so the no-drift return fired while `_colab_python_version` answered None and
    marker evaluation silently replayed every requirement."""
    upstream, snapshot_dir = oracle
    without_python = "R version 4.5.3\n"
    upstream["os-info-gpu.txt"] = without_python
    (snapshot_dir / nv.COLAB_ORACLE_FILES["os-info-gpu.txt"]).write_text(
        without_python, encoding = "utf-8"
    )
    assert _diff(snapshot_dir, strict = True) == 1


def test_an_unreadable_strict_value_fails_strict(oracle):
    """A strict value its consumer cannot parse, such as `Python version 3.14`, must fail --strict."""
    upstream, snapshot_dir = oracle
    reformatted = "Python version 3.14\nR version 4.5.3\n"
    upstream["os-info-gpu.txt"] = reformatted
    (snapshot_dir / nv.COLAB_ORACLE_FILES["os-info-gpu.txt"]).write_text(
        reformatted, encoding = "utf-8"
    )
    assert _diff(snapshot_dir, strict = True) == 1


def test_an_advisory_oracle_that_will_not_fetch_does_not_fail_the_refresh(oracle, tmp_path, capsys):
    """An unfetchable advisory oracle such as apt-list must not block refreshing the other snapshots."""
    upstream, _ = oracle
    real = nv.urllib.request.urlopen

    def flaky(url, timeout = None):
        if url.endswith("apt-list-gpu.txt"):
            raise urllib.error.URLError("boom")
        return real(url, timeout = timeout)

    nv.urllib.request.urlopen = flaky
    try:
        out_dir = tmp_path / "partial"
        rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    finally:
        nv.urllib.request.urlopen = real
    assert rc == 0
    assert sorted(p.name for p in out_dir.iterdir()) == sorted(
        [
            nv.COLAB_ORACLE_FILES["pip-freeze.gpu.txt"],
            nv.COLAB_ORACLE_FILES["os-info-gpu.txt"],
        ]
    )
    assert "skipping apt-list-gpu.txt" in capsys.readouterr().out


def test_a_rule_bearing_oracle_that_will_not_fetch_still_fails_the_refresh(oracle, tmp_path):
    """pip is what --colab-pin resolves against, so its absence is fatal and writes nothing."""
    upstream, _ = oracle

    def dead(url, timeout = None):
        raise urllib.error.URLError("boom")

    real = nv.urllib.request.urlopen
    nv.urllib.request.urlopen = dead
    try:
        out_dir = tmp_path / "none"
        rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    finally:
        nv.urllib.request.urlopen = real
    assert rc == 2
    assert not out_dir.exists()


@pytest.mark.parametrize(
    "name, payload",
    [
        # An upstream rotation to a format _parse_os_lines cannot read the Python line out of.
        ("os-info-gpu.txt", '{"python": "3.13.15"}\n'),
        # The Python line present but holding something _marker_environment cannot parse.
        ("os-info-gpu.txt", "Python (unknown)\n"),
        # An empty pin file resolves every R-INST rule against nothing.
        ("pip-freeze.gpu.txt", "\n"),
    ],
)
def test_a_refresh_never_acknowledges_a_payload_the_rules_cannot_read(
    oracle, tmp_path, name, payload
):
    """Writing it would leave upstream and snapshot equally empty of the key, and colab-diff
    compares the two parses, so the strict tripwire would go quiet on the very rotation it
    exists to catch."""
    upstream, _ = oracle
    upstream[name] = payload
    out_dir = tmp_path / "rotated"
    rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    assert rc == 2
    assert not out_dir.exists()


def test_a_failed_write_restores_the_whole_snapshot_set(oracle, tmp_path, monkeypatch, capsys):
    """A failed refresh must restore the whole set, not leave new packages beside a stale Python."""
    upstream, snapshot_dir = oracle
    committed = {
        name: (snapshot_dir / name).read_bytes() for name in nv.COLAB_ORACLE_FILES.values()
    }
    for key in upstream:
        upstream[key] = upstream[key].replace("2.10.0", "2.11.0").replace("3.13.15", "3.14.1")

    real_write = nv._atomic_write_bytes
    calls: list[str] = []

    def flaky(path, data):
        calls.append(path.name)
        if len(calls) > 1 and path.name != calls[0]:
            raise OSError("no space left on device")
        return real_write(path, data)

    monkeypatch.setattr(nv, "_atomic_write_bytes", flaky)
    rc = nv.cmd_refresh_colab(
        argparse.Namespace(all = True, snapshot_dir = str(snapshot_dir), out = None)
    )
    assert rc == 2
    monkeypatch.setattr(nv, "_atomic_write_bytes", real_write)
    for name, data in committed.items():
        assert (snapshot_dir / name).read_bytes() == data, name
    assert "restored" in capsys.readouterr().err


def test_a_write_failure_removes_a_file_that_was_not_there_before(oracle, tmp_path, monkeypatch):
    """Nothing to restore means nothing left behind: a fresh directory stays empty of the
    half-written generation rather than keeping the one file that landed."""
    upstream, _ = oracle
    out_dir = tmp_path / "fresh"

    real_write = nv._atomic_write_bytes
    calls: list[str] = []

    def flaky(path, data):
        calls.append(path.name)
        if len(calls) > 1:
            raise OSError("no space left on device")
        return real_write(path, data)

    monkeypatch.setattr(nv, "_atomic_write_bytes", flaky)
    rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    assert rc == 2
    assert list(out_dir.iterdir()) == []


# Spelled out: reading the module in a decorator fails collection when the symbol moves.
SEED_PACKAGES = ("torch", "torchcodec", "peft", "torchao", "transformers", "tokenizers")


def test_the_seed_list_is_the_one_the_rules_use():
    assert set(SEED_PACKAGES) == set(nv._COLAB_PIP_REQUIRED)


@pytest.mark.parametrize("dropped", SEED_PACKAGES)
def test_a_pin_file_missing_a_seed_package_is_not_acknowledged(oracle, tmp_path, dropped):
    """A pin file missing a seed package must be rejected, or the lint silently skips its rules."""
    upstream, _ = oracle
    upstream["pip-freeze.gpu.txt"] = "\n".join(
        line for line in PIP.splitlines() if not line.startswith(f"{dropped}==")
    )
    out_dir = tmp_path / f"missing_{dropped}"
    rc = nv.cmd_refresh_colab(argparse.Namespace(all = True, snapshot_dir = str(out_dir), out = None))
    assert rc == 2
    assert not out_dir.exists()


def test_a_rollback_survives_a_filesystem_that_is_still_full(oracle, tmp_path, monkeypatch):
    """Rollback must survive a still-full disk, since rewriting old bytes can fail and mix generations."""
    upstream, snapshot_dir = oracle
    committed = {
        name: (snapshot_dir / name).read_bytes() for name in nv.COLAB_ORACLE_FILES.values()
    }
    for key in upstream:
        upstream[key] = upstream[key].replace("2.10.0", "2.11.0").replace("3.13.15", "3.14.1")

    real_write = nv._atomic_write_bytes
    calls: list[str] = []

    def full_disk(path, data):
        calls.append(path.name)
        if len(calls) > 1:
            raise OSError("no space left on device")  # every write from here on, rollback too
        return real_write(path, data)

    monkeypatch.setattr(nv, "_atomic_write_bytes", full_disk)
    rc = nv.cmd_refresh_colab(
        argparse.Namespace(all = True, snapshot_dir = str(snapshot_dir), out = None)
    )
    assert rc == 2
    monkeypatch.setattr(nv, "_atomic_write_bytes", real_write)
    for name, data in committed.items():
        assert (snapshot_dir / name).read_bytes() == data, name
    assert not [p for p in snapshot_dir.iterdir() if p.name.endswith(".rollback")]


def test_a_dry_run_install_does_not_undo_a_removal():
    """A `pip install --dry-run` changes nothing, so it must not undo an earlier uninstall."""
    assert nv._removed_by_cell(
        "!pip uninstall -y tokenizers; pip install --dry-run tokenizers", "tokenizers"
    )
    assert not nv._removed_by_cell(
        "!pip uninstall -y tokenizers; pip install tokenizers", "tokenizers"
    )


def test_an_unfetchable_rule_bearing_oracle_fails_strict(oracle, capsys):
    """An unfetchable rule-bearing oracle must fail --strict; a check that never ran is not a pass."""
    upstream, snapshot_dir = oracle
    real = nv.urllib.request.urlopen

    def flaky(url, timeout = None):
        if url.endswith("os-info-gpu.txt"):
            raise urllib.error.URLError("boom")
        return real(url, timeout = timeout)

    nv.urllib.request.urlopen = flaky
    try:
        assert _diff(snapshot_dir, strict = True) == 1
    finally:
        nv.urllib.request.urlopen = real
    assert "::error::" in capsys.readouterr().out


def test_an_unfetchable_advisory_oracle_stays_a_warning(oracle, capsys):
    """apt-list carries no rule-bearing key, so its absence must not redden the cron."""
    upstream, snapshot_dir = oracle
    real = nv.urllib.request.urlopen

    def flaky(url, timeout = None):
        if url.endswith("apt-list-gpu.txt"):
            raise urllib.error.URLError("boom")
        return real(url, timeout = timeout)

    nv.urllib.request.urlopen = flaky
    try:
        assert _diff(snapshot_dir, strict = True) == 0
    finally:
        nv.urllib.request.urlopen = real
    assert "::warning::" in capsys.readouterr().out
