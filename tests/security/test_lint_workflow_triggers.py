"""Regression tests for scripts/lint_workflow_triggers.py, guarding GHSA-g7cv-rxg3-hmpx vectors."""

from __future__ import annotations

import fnmatch
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


def _host_workflow(restriction: str = "", step_extra: str = "") -> str:
    """A workflow that runs the lint, optionally narrowed or non-blocking."""
    return (
        "name: host\n"
        "on:\n"
        "  pull_request:\n" + restriction + "jobs:\n"
        "  lint:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: python3 scripts/lint_workflow_triggers.py\n" + step_extra
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("paths", "      - 'studio/**'\n"),
        ("paths-ignore", "      - 'studio/**'\n"),
        ("branches", "      - some-other-branch\n"),
        ("branches-ignore", "      - main\n"),
        ("types", "      - closed\n"),
    ],
)
def test_lint_rejects_a_narrowed_host(tmp_path, key, value):
    """A host narrowed any way can be skipped by the PR that narrows it.

    `pull_request` resolves the workflow file from the PR merge ref, so the
    restriction takes effect for its own PR and the gate never runs on the
    change it exists to review. `paths` is only the most obvious key: a
    branch or event-type restriction skips ordinary PRs just as well.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(_host_workflow(f"    {key}:\n{value}"))
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert key in proc.stderr
    assert "host.yml" in proc.stderr


@pytest.mark.parametrize("where", ["step", "job"])
@pytest.mark.parametrize(
    "key, value, expected",
    [
        ("continue-on-error", "true", "continue-on-error"),
        ("if", "${{ false }}", "'if:' condition"),
    ],
)
def test_lint_rejects_a_host_that_cannot_fail(tmp_path, where, key, value, expected):
    """A host that runs but cannot fail, or is skipped, is not a gate.

    `continue-on-error` makes findings advisory; a false `if:` skips the step
    entirely. Either way the workflow goes green with the lint never enforcing
    anything.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    if where == "step":
        body = _host_workflow(step_extra = f"        {key}: {value}\n")
    else:
        body = _host_workflow().replace(
            "    runs-on: ubuntu-latest\n",
            f"    runs-on: ubuntu-latest\n    {key}: {value}\n",
        )
    (wf / "host.yml").write_text(body)
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert expected in proc.stderr


def test_lint_accepts_unfiltered_host(tmp_path):
    """A bare `pull_request:` host that can fail satisfies the requirement."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(_host_workflow())
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


@pytest.mark.parametrize(
    "mention",
    [
        "      # scripts/lint_workflow_triggers.py needs PyYAML\n",
        "      # - run: python3 scripts/lint_workflow_triggers.py\n",
    ],
    ids = ["prose", "commented-run-step"],
)
def test_commented_mention_is_not_a_host(tmp_path, mention):
    """A mention that executes nothing must not register as a host.

    The commented-out `run:` line is the dangerous one: it would let a repo
    with the real host workflow deleted still satisfy `--require-host`.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "mentions.yml").write_text(
        "name: mentions\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n" + mention +
        "      - run: echo hi\n"
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


def test_workflow_trigger_lint_host_exists_and_is_unfiltered():
    """End to end on the live tree, with the host requirement forced on."""
    proc = _run(REPO_ROOT / ".github" / "workflows", require_host = True)
    assert (
        proc.returncode == 0
    ), f"live tree failed lint:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _codeowners_rules(text: str) -> list[tuple[str, list[str]]]:
    """Rules in file order, INCLUDING ownerless ones.

    A pattern with no owners is valid CODEOWNERS and clears ownership for what
    it matches, so skipping those lines would miss a silent carve-out.
    """
    rules = []
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            rules.append((fields[0], fields[1:]))
    return rules


def _pattern_matches(pattern: str, path: str) -> bool:
    """Approximate GitHub's CODEOWNERS matching, erring toward matching.

    A directory pattern owns everything beneath it, so every directory prefix
    of the path is a candidate. Patterns are globbed rather than compared
    literally: `**/workflows/` is a valid rule that a substring search misses.
    An unanchored pattern may start at any depth.
    """
    if pattern == "*":
        return True
    anchored = pattern.startswith("/")
    is_dir = pattern.endswith("/")
    body = pattern.strip("/")
    segments = path.split("/")

    candidates = ["/".join(segments[:i]) for i in range(1, len(segments))]
    if not is_dir:
        candidates.append(path)
    if not anchored:
        candidates += [
            "/".join(segments[j:i])
            for i in range(1, len(segments) + 1)
            for j in range(1, i)
        ]
    return any(fnmatch.fnmatch(c, body) for c in candidates)


def _effective_owners(text: str, path: str) -> list[str]:
    """Owners GitHub would require, i.e. the LAST matching rule wins."""
    owners: list[str] = []
    for pattern, people in _codeowners_rules(text):
        if _pattern_matches(pattern, path):
            owners = people
    return owners


CODEOWNERS_PROBES = (
    ".github/workflows/workflow-trigger-lint.yml",
    ".github/CODEOWNERS",
)


def test_workflow_changes_require_code_owner_review():
    """Every workflow must keep an EFFECTIVE code owner.

    The lint cannot stop a PR that disables the lint's own host workflow, so
    owner review is the merge-time control. GitHub applies only the last
    matching pattern, so checking that a rule exists somewhere is not enough:
    a later rule, broad or narrow, silently takes over. Any workflow can hand
    a fork PR the base repo's secrets, so every one of them is checked, not
    just the lint host. Delegating a workflow to another maintainer is fine;
    leaving one unowned is not.
    """
    text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding = "utf-8")
    workflows = sorted(
        p.relative_to(REPO_ROOT).as_posix()
        for p in (REPO_ROOT / ".github" / "workflows").iterdir()
        if p.suffix in (".yml", ".yaml")
    )
    assert workflows, "no workflow files found"
    for probe in workflows:
        assert _effective_owners(text, probe), (
            f"CODEOWNERS leaves {probe} with no effective owner; a later "
            "pattern overrode the .github/workflows/ rule."
        )
    for probe in CODEOWNERS_PROBES:
        owners = _effective_owners(text, probe)
        assert "@danielhanchen" in owners, (
            f"CODEOWNERS gives {probe} effective owners {owners or '(none)'}; "
            "a later pattern overrode the workflow rule."
        )


@pytest.mark.parametrize(
    "override, expected",
    [
        ("* @someone-else", ["@someone-else"]),
        ("/.github/ @someone-else", ["@someone-else"]),
        (f"/{CODEOWNERS_PROBES[0]} @someone-else", ["@someone-else"]),
        # A pattern with no owners is valid, and clears ownership.
        (f"/{CODEOWNERS_PROBES[0]}", []),
        # Globbed and unanchored directory patterns are valid rules too.
        ("**/workflows/ @someone-else", ["@someone-else"]),
        ("workflows/ @someone-else", ["@someone-else"]),
        (".github/*/ @someone-else", ["@someone-else"]),
    ],
    ids = [
        "catch-all",
        "parent-dir",
        "narrower-file",
        "ownerless",
        "globbed-dir",
        "unanchored-dir",
        "wildcard-segment",
    ],
)
def test_codeowners_guard_catches_a_later_rule(override, expected):
    """The guard must fail whichever way a trailing rule takes precedence."""
    text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding = "utf-8")
    owners = _effective_owners(f"{text}\n{override}\n", CODEOWNERS_PROBES[0])
    assert owners == expected, f"{override!r} should win, got {owners}"


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
