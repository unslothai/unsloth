"""Regression tests for scripts/lint_workflow_triggers.py, guarding GHSA-g7cv-rxg3-hmpx vectors."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "lint_workflow_triggers.py"


def _run(workflows_dir: Path, require_host: bool = False) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(SCRIPT), "--workflows-dir", str(workflows_dir)]
    if require_host:
        cmd.append("--require-host")
    return subprocess.run(cmd, capture_output = True, text = True)


def test_lint_passes_on_current_workflows():
    """The live `.github/workflows/` tree must pass the lint."""
    live = REPO_ROOT / ".github" / "workflows"
    proc = _run(live)
    assert (
        proc.returncode == 0
    ), f"live tree failed lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_lint_rejects_pull_request_target(tmp_path):
    """Synthetic PR_TARGET trigger must produce rc=1 with a named finding."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "bad.yml").write_text(
        "name: bad\n"
        "on:\n"
        "  pull_request_target:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo evil\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "BANNED trigger 'pull_request_target'" in proc.stderr
    assert "GHSA-g7cv-rxg3-hmpx" in proc.stderr


def test_lint_rejects_pull_request_target_in_yaml_extension(tmp_path):
    """GitHub Actions also loads `.yaml`; the lint must not stop at `.yml`."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "bad.yaml").write_text(
        "name: bad\n"
        "on:\n"
        "  pull_request_target:\n"
        "    branches: [main]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo evil\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "bad.yaml" in proc.stderr
    assert "BANNED trigger 'pull_request_target'" in proc.stderr


def _host_workflow(filter_key: str | None) -> str:
    """A workflow that runs the lint, optionally with a path filter."""
    flt = f"    {filter_key}:\n      - 'studio/**'\n" if filter_key else ""
    return (
        "name: host\n"
        "on:\n"
        "  pull_request:\n" + flt + "jobs:\n"
        "  lint:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: python3 scripts/lint_workflow_triggers.py\n"
    )


@pytest.mark.parametrize("filter_key", ["paths", "paths-ignore"])
def test_lint_rejects_path_filter_on_its_own_host(tmp_path, filter_key):
    """A host with a path filter can be skipped by the PR that adds it.

    `pull_request` resolves the workflow file from the PR merge ref, so the
    filter takes effect for its own PR and the gate never runs on the change
    it exists to review. Both filter keys are the same bypass.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(_host_workflow(filter_key))
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert filter_key in proc.stderr
    assert "host.yml" in proc.stderr


def test_lint_accepts_unfiltered_host(tmp_path):
    """An unfiltered `pull_request` host satisfies the wiring requirement."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(_host_workflow(None))
    proc = _run(wf, require_host = True)
    assert proc.returncode == 0, proc.stderr


def test_lint_rejects_missing_host(tmp_path):
    """Deleting the gate's workflow must fail the gate, not silently pass."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "unrelated.yml").write_text(
        "name: unrelated\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo hi\n"
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


def test_comment_mention_is_not_a_host(tmp_path):
    """Naming the script in a comment must not count as running it."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "mentions.yml").write_text(
        "name: mentions\n"
        "on:\n"
        "  pull_request:\n"
        "    paths:\n"
        "      - 'studio/**'\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      # scripts/lint_workflow_triggers.py needs PyYAML\n"
        "      - run: echo hi\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, proc.stderr


def test_workflow_trigger_lint_host_exists_and_is_unfiltered():
    """End to end on the live tree, with the host requirement forced on."""
    proc = _run(REPO_ROOT / ".github" / "workflows", require_host = True)
    assert (
        proc.returncode == 0
    ), f"live tree failed lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def test_workflow_changes_require_code_owner_review():
    """CODEOWNERS must cover `.github/workflows/`.

    The lint cannot stop a PR that disables the lint's own host workflow, so
    the merge-time control is owner review on workflow changes.
    """
    owners = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding = "utf-8")
    rules = [
        line.split("#", 1)[0].strip()
        for line in owners.splitlines()
        if line.split("#", 1)[0].strip()
    ]
    for pattern in ("/.github/workflows/", "/.github/CODEOWNERS"):
        assert any(
            r.split()[0] == pattern and len(r.split()) > 1 for r in rules
        ), f"CODEOWNERS has no owner rule for {pattern}"


def test_lint_rejects_unjustified_workflow_run(tmp_path):
    """`workflow_run` requires an explicit allow-comment in the YAML."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "RESTRICTED trigger 'workflow_run'" in proc.stderr


def test_lint_allows_justified_workflow_run(tmp_path):
    """With the allow-comment, workflow_run is permitted."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "chained.yml").write_text(
        "# lint:workflow_triggers-allow-workflow_run -- justified by ticket #1234\n"
        "name: chained\n"
        "on:\n"
        "  workflow_run:\n"
        "    workflows: ['CI']\n"
        "    types: [completed]\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: echo elevated\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"justified workflow_run rejected:\n{proc.stderr}"


def test_lint_rejects_shared_cache_key_between_pr_and_publish(tmp_path):
    """A cache key declared in both a PR-triggered workflow and the
    publish workflow is the TanStack cache-poisoning vector."""
    wf = tmp_path / "wf"
    wf.mkdir()
    # PR-triggered: writes a cache the publish job will also restore.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    # Publish workflow with the IDENTICAL cache key (the attack pattern).
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: shared-cache-v1\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "cache-key" in proc.stderr.lower() or "cache key" in proc.stderr.lower()
    assert "shared-cache-v1" in proc.stderr
