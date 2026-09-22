"""Regression tests for scripts/lint_workflow_triggers.py, guarding GHSA-g7cv-rxg3-hmpx vectors."""

from __future__ import annotations

import re
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
    """A host narrowed any way is skipped by the PR that narrows it.

    The merge ref carries the restriction, so it applies to that PR. A branch
    or event-type filter skips ordinary PRs as well as `paths` does.
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
    """A host that cannot fail, or is skipped, is not a gate.

    `continue-on-error` makes findings advisory; a false `if:` skips the step.
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


@pytest.mark.parametrize(
    "command, expected",
    [
        ("python3 scripts/lint_workflow_triggers.py || true", "chained"),
        ("python3 scripts/lint_workflow_triggers.py ; true", "chained"),
        ("python3 scripts/lint_workflow_triggers.py | tee lint.log", "chained"),
        ("python3 scripts/lint_workflow_triggers.py &", "chained"),
        (
            "set +e\n          python3 scripts/lint_workflow_triggers.py",
            "other shell besides the lint command",
        ),
        (
            "python3 scripts/lint_workflow_triggers.py --workflows-dir /tmp/empty",
            "--workflows-dir",
        ),
        ("python3 scripts/lint_workflow_triggers.py --no-require-host", "--no-require-host"),
        ("python3 scripts/lint_workflow_triggers.py --workflows-d /tmp/empty", "--workflows-d"),
        ("python3 scripts/lint_workflow_triggers.py --help", "--help"),
    ],
    ids = [
        "or-true",
        "semi-true",
        "pipe-tee",
        "background",
        "extra-shell",
        "elsewhere-dir",
        "self-check-off",
        "abbreviated-flag",
        "help",
    ],
)
def test_lint_rejects_a_defanged_invocation(tmp_path, command, expected):
    """Running the script is not enough; it has to be able to gate.

    A pipeline or `|| true` detaches the step's status from the lint's, and an
    argument can redirect it, disable its wiring check, or exit early.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py",
            f"run: |\n          {command}",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert expected in proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        "python3 -c 'pass' scripts/lint_workflow_triggers.py",
        "python3 -m json.tool scripts/lint_workflow_triggers.py",
        "echo scripts/lint_workflow_triggers.py",
        # A decoy with the right basename but not this repository's script.
        "python3 /tmp/lint_workflow_triggers.py",
    ],
    ids = ["dash-c", "dash-m", "echo", "decoy-path"],
)
def test_lint_does_not_count_a_non_running_command_as_a_host(tmp_path, command):
    """None of these execute the repository's lint, so none is a host."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py", f"run: {command}"
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


@pytest.mark.parametrize(
    "body",
    [
        # A decoy under any prefix, not just a bare /tmp path.
        "python3 /tmp/scripts/lint_workflow_triggers.py",
        # Defining a function is not calling it.
        "never_called() {\n            python3 scripts/lint_workflow_triggers.py\n          }",
        # A here-document is data, not a command.
        "cat <<'EOF'\n          python3 scripts/lint_workflow_triggers.py\n          EOF",
    ],
    ids = ["prefixed-decoy", "uncalled-function", "heredoc"],
)
def test_lint_rejects_an_unexecuted_lint_command(tmp_path, body):
    """Text that looks like the invocation but never runs it is not a host."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py",
            f"run: |\n          {body}",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


@pytest.mark.parametrize(
    "where",
    ["step", "job-defaults", "workflow-defaults"],
)
def test_lint_rejects_a_custom_shell(tmp_path, where):
    """A shell template can wrap the command and drop its exit status."""
    evil = "bash -c '\"{0}\" || true'"
    body = _host_workflow()
    if where == "step":
        body = body.replace(
            "      - run: python3 scripts/lint_workflow_triggers.py\n",
            f"      - run: python3 scripts/lint_workflow_triggers.py\n        shell: {evil}\n",
        )
    elif where == "job-defaults":
        body = body.replace(
            "    runs-on: ubuntu-latest\n",
            f"    runs-on: ubuntu-latest\n    defaults:\n      run:\n        shell: {evil}\n",
        )
    else:
        body = body.replace("jobs:\n", f"defaults:\n  run:\n    shell: {evil}\njobs:\n")
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(body)
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "shell" in proc.stderr


def test_lint_accepts_an_explicit_plain_shell(tmp_path):
    """`shell: bash` is ordinary and must keep working."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "      - run: python3 scripts/lint_workflow_triggers.py\n",
            "      - run: python3 scripts/lint_workflow_triggers.py\n        shell: bash\n",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        # A fake interpreter that merely contains "python" in its name.
        "/tmp/fakepython scripts/lint_workflow_triggers.py",
        # Flags that make python print and exit before running the file.
        "python3 --version scripts/lint_workflow_triggers.py",
        "python3 -V scripts/lint_workflow_triggers.py",
        "python3 --help scripts/lint_workflow_triggers.py",
    ],
    ids = ["fake-interpreter", "version-long", "version-short", "help-before-path"],
)
def test_lint_rejects_a_non_running_interpreter(tmp_path, command):
    """The interpreter must be a python that actually executes the file."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py", f"run: {command}"
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


@pytest.mark.parametrize("where", ["step", "job-defaults", "workflow-defaults"])
def test_lint_rejects_a_changed_working_directory(tmp_path, where):
    """`working-directory` resolves the command to a different file."""
    body = _host_workflow()
    if where == "step":
        body = body.replace(
            "      - run: python3 scripts/lint_workflow_triggers.py\n",
            "      - run: python3 scripts/lint_workflow_triggers.py\n"
            "        working-directory: /tmp\n",
        )
    elif where == "job-defaults":
        body = body.replace(
            "    runs-on: ubuntu-latest\n",
            "    runs-on: ubuntu-latest\n"
            "    defaults:\n      run:\n        working-directory: /tmp\n",
        )
    else:
        body = body.replace("jobs:\n", "defaults:\n  run:\n    working-directory: /tmp\njobs:\n")
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(body)
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "working-directory" in proc.stderr


@pytest.mark.parametrize("suffix", [".yml", ".yaml"])
def test_publish_cache_key_collision_found_under_both_suffixes(tmp_path, suffix):
    """Scanning `.yaml` is pointless if publisher classification misses it."""
    (wf := tmp_path / "wf").mkdir()
    (wf / f"release-desktop{suffix}").write_text(
        "name: release\n"
        "on:\n"
        "  push:\n"
        "    branches: [main]\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          key: shared-cache-key\n"
    )
    (wf / "pr.yml").write_text(
        "name: pr\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          key: shared-cache-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "cache key" in proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        # -i drops into the REPL after the script;
        # on EOF the interpreter exits 0 even though the lint called sys.exit(1).
        "python3 -i scripts/lint_workflow_triggers.py",
        # Anything outside the allowlist fails closed.
        "python3 -d scripts/lint_workflow_triggers.py",
        "python3 -uB scripts/lint_workflow_triggers.py",
    ],
    ids = ["interactive", "unknown-flag", "combined-flag"],
)
def test_lint_rejects_flags_outside_the_allowlist(tmp_path, command):
    """Only flags that leave run-this-file-and-return-its-status intact count."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py", f"run: {command}"
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


@pytest.mark.parametrize("flag", ["-u", "-E", "-s", "-B", "-q", "-O"])
def test_lint_accepts_allowlisted_flags(tmp_path, flag):
    """The allowlist must not reject ordinary interpreter flags."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py",
            f"run: python3 {flag} scripts/lint_workflow_triggers.py",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("scope", ["step", "job", "workflow"])
@pytest.mark.parametrize("key", ["BASH_ENV", "PATH"])
def test_lint_rejects_execution_redirecting_env(tmp_path, scope, key):
    """`BASH_ENV` runs before the step script; `PATH` picks the interpreter."""
    body = _host_workflow()
    entry = f"env:\n  {key}: /tmp/x\n"
    if scope == "step":
        body = body.replace(
            "      - run: python3 scripts/lint_workflow_triggers.py\n",
            "      - run: python3 scripts/lint_workflow_triggers.py\n"
            f"        env:\n          {key}: /tmp/x\n",
        )
    elif scope == "job":
        body = body.replace(
            "    runs-on: ubuntu-latest\n",
            f"    runs-on: ubuntu-latest\n    env:\n      {key}: /tmp/x\n",
        )
    else:
        body = body.replace("jobs:\n", entry + "jobs:\n")
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(body)
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert key in proc.stderr


@pytest.mark.parametrize("value", ["false", "true", "[opened]", "'yes'"])
def test_lint_rejects_a_non_mapping_pull_request_value(tmp_path, value):
    """GitHub will not load such a workflow, so it cannot be the gate."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace("  pull_request:\n", f"  pull_request: {value}\n")
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "not a valid event configuration" in proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        # A repo-root `./python3` can be added by the PR itself.
        "./python3 scripts/lint_workflow_triggers.py",
        "bin/python3 scripts/lint_workflow_triggers.py",
        # A substitution in an option value runs before python does.
        'python3 -W "$(touch pwned)" scripts/lint_workflow_triggers.py',
    ],
    ids = ["relative-interpreter", "repo-path-interpreter", "expansion-in-value"],
)
def test_lint_rejects_pr_controlled_interpreters(tmp_path, command):
    """The interpreter and its option values must not come from the checkout."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py", f"run: {command}"
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


@pytest.mark.parametrize("interpreter", ["python3", "python", "/usr/bin/python3"])
def test_lint_accepts_trusted_interpreters(tmp_path, interpreter):
    """A bare command or a system path stays acceptable."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py",
            f"run: {interpreter} scripts/lint_workflow_triggers.py",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("key", ["PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"])
def test_lint_rejects_python_startup_env(tmp_path, key):
    """`sitecustomize.py` on PYTHONPATH runs before the lint and can exit 0."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "      - run: python3 scripts/lint_workflow_triggers.py\n",
            "      - run: python3 scripts/lint_workflow_triggers.py\n"
            f"        env:\n          {key}: ./pr-controlled\n",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert key in proc.stderr


def test_lint_rejects_expansion_in_the_interpreter_token(tmp_path):
    """Bash substitutes before the trusted-path test can mean anything."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py",
            'run: |\n          "/usr/$(printf bin)/python3" ' "scripts/lint_workflow_triggers.py",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "does not cover every PR" in proc.stderr


def test_lint_rejects_a_containerized_host(tmp_path):
    """A PR-selected image controls the shell and environment."""
    (wf := tmp_path / "wf").mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "    runs-on: ubuntu-latest\n",
            "    runs-on: ubuntu-latest\n    container: alpine:latest\n",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "container" in proc.stderr


def test_lint_rejects_a_host_job_with_needs(tmp_path):
    """A skipped prerequisite skips the lint job without failing the run."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "  lint:\n    runs-on: ubuntu-latest\n",
            "  setup:\n"
            "    runs-on: ubuntu-latest\n"
            "    if: ${{ false }}\n"
            "    steps:\n"
            "      - run: echo hi\n"
            "  lint:\n"
            "    runs-on: ubuntu-latest\n"
            "    needs: setup\n",
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 1
    assert "needs:" in proc.stderr


@pytest.mark.parametrize(
    "command",
    [
        "python3 scripts/lint_workflow_triggers.py",
        "python3 -u scripts/lint_workflow_triggers.py",
        "python scripts/lint_workflow_triggers.py",
        "python3 -X utf8 scripts/lint_workflow_triggers.py",
    ],
    ids = ["plain", "dash-u", "python", "dash-X-with-value"],
)
def test_lint_accepts_ordinary_invocations(tmp_path, command):
    """Tightening host detection must not reject normal ways to run it."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "host.yml").write_text(
        _host_workflow().replace(
            "run: python3 scripts/lint_workflow_triggers.py", f"run: {command}"
        )
    )
    proc = _run(wf, require_host = True)
    assert proc.returncode == 0, proc.stderr


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

    The commented-out `run:` is the dangerous one: it would satisfy
    `--require-host` with the real workflow deleted.
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
        "    steps:\n" + mention + "      - run: echo hi\n"
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


def _pattern_regex(pattern: str) -> re.Pattern:
    """CODEOWNERS globbing: `*` stops at `/`, `**` crosses directories.

    `fnmatch` gets both halves wrong here. Its `*` consumes `/`, so
    `/.github/*` would wrongly claim nested workflow files and fail a valid
    CODEOWNERS change; and it cannot express `**/` matching zero directories,
    so `/.github/**/workflows/` would wrongly miss one.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out.append("(?:[^/]+/)*")
            i += 3
        elif pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pattern[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"\Z")


def _is_valid_owner(token: str) -> bool:
    """GitHub only requests review from an @user, an @org/team, or an email."""
    if token.startswith("@"):
        return len(token) > 1
    local, _, domain = token.partition("@")
    return bool(local and "." in domain)


def _pattern_matches(pattern: str, path: str) -> bool:
    """Approximate GitHub's CODEOWNERS matching.

    A pattern that NAMES a directory owns everything beneath it, so directory
    prefixes of the path are candidates. A pattern with a wildcard in it does
    not: GitHub documents `docs/*` as matching `docs/getting-started.md` but
    not `docs/build-app/troubleshooting.md`.

    Only a pattern with no internal separator floats to any depth, gitignore
    style. A leading slash anchors, and so does an internal one, so
    `workflows/lint.yml` is root-relative and does NOT match
    `.github/workflows/lint.yml`; a bare `workflows/` still matches at any
    depth.
    """
    if pattern == "*":
        return True
    is_dir = pattern.endswith("/")
    body = pattern.strip("/")
    anchored = pattern.startswith("/") or "/" in body
    rx = _pattern_regex(body)
    segments = path.split("/")

    def matches(candidate: str) -> bool:
        parts = candidate.split("/")
        starts = [0] if anchored else range(len(parts))
        return any(rx.match("/".join(parts[j:])) for j in starts)

    prefixes = ["/".join(segments[:i]) for i in range(1, len(segments))]
    if is_dir:
        return any(matches(p) for p in prefixes)
    if matches(path):
        return True
    # A plain path may name a directory, and then owns everything under it.
    return not any(ch in body for ch in "*?") and any(matches(p) for p in prefixes)


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
        owners = [o for o in _effective_owners(text, probe) if _is_valid_owner(o)]
        assert owners, (
            f"CODEOWNERS leaves {probe} with no effective owner GitHub could "
            "request review from; a later pattern overrode the "
            ".github/workflows/ rule."
        )
    for probe in CODEOWNERS_PROBES:
        owners = _effective_owners(text, probe)
        assert "@danielhanchen" in owners, (
            f"CODEOWNERS gives {probe} effective owners {owners or '(none)'}; "
            "a later pattern overrode the workflow rule."
        )


@pytest.mark.parametrize(
    "pattern, path, matches",
    [
        # `*` stops at a directory boundary, like GitHub's `docs/*` example.
        ("/.github/*", ".github/workflows/lint.yml", False),
        ("/.github/*", ".github/CODEOWNERS", True),
        # `**` crosses them, and `**/` may match zero.
        ("/.github/**", ".github/workflows/lint.yml", True),
        ("/.github/**/workflows/", ".github/workflows/lint.yml", True),
        ("/.github/**/workflows/", ".github/a/b/workflows/lint.yml", True),
        # A plain path naming a directory owns everything beneath it.
        ("/.github/workflows/", ".github/workflows/lint.yml", True),
        ("/scripts", "scripts/data/x.txt", True),
        # Unanchored patterns may start at any depth.
        ("workflows/", ".github/workflows/lint.yml", True),
        ("**/workflows/", ".github/workflows/lint.yml", True),
        # An internal slash anchors at the root, gitignore style, so this names a top-level workflows/ and not the one
        # under .github/.
        ("workflows/lint.yml", ".github/workflows/lint.yml", False),
        ("workflows/lint.yml", "workflows/lint.yml", True),
        ("/unsloth/", ".github/workflows/lint.yml", False),
        ("/unsloth", "unsloth_zoo/x.py", False),
    ],
)
def test_codeowners_pattern_semantics(pattern, path, matches):
    """The matcher must model GitHub, in both directions.

    Under-matching hides a rule that steals ownership; over-matching fails a
    valid CODEOWNERS change that never touched the workflows.
    """
    assert _pattern_matches(pattern, path) is matches


@pytest.mark.parametrize(
    "token, valid",
    [
        ("@danielhanchen", True),
        ("@unslothai/maintainers", True),
        ("danielhanchen@gmail.com", True),
        ("not-an-owner", False),
        ("@", False),
    ],
)
def test_owner_token_validity(token, valid):
    """An unusable owner token leaves a path effectively unowned.

    GitHub cannot request review from a bare word, so counting it as an owner
    would let a trailing rule quietly disown a workflow.
    """
    assert _is_valid_owner(token) is valid


def test_invalid_owner_does_not_count_as_ownership():
    """The realistic mistake: a later rule naming a non-owner."""
    text = (REPO_ROOT / ".github" / "CODEOWNERS").read_text(encoding = "utf-8")
    probe = CODEOWNERS_PROBES[0]
    owners = _effective_owners(f"{text}\n/{probe} not-an-owner\n", probe)
    assert owners == ["not-an-owner"]
    assert not [o for o in owners if _is_valid_owner(o)]


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
        # `**/` may match zero directories.
        ("/.github/**/workflows/ @someone-else", ["@someone-else"]),
    ],
    ids = [
        "catch-all",
        "parent-dir",
        "narrower-file",
        "ownerless",
        "globbed-dir",
        "unanchored-dir",
        "wildcard-segment",
        "double-star-zero-dirs",
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


def test_lint_rejects_a_publish_restore_keys_prefix_over_a_pr_namespace(tmp_path):
    """A prefix restore reaches the same cache an equal key would, and was invisible.

    `restore-keys` restores the newest entry whose key merely STARTS WITH the prefix, so
    a publish workflow can adopt an entry a pull request wrote without the two keys ever
    being equal. The exact-key check next to this one compares whole strings, so it never
    saw this route.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
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
        "          key: pip-v2-${{ runner.os }}-abc\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: pip-v2-exact-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"prefix restore accepted:\n{proc.stdout}\n{proc.stderr}"
    assert "restore-keys" in proc.stderr
    assert "'pip-'" in proc.stderr


def test_a_partitioned_publish_prefix_is_accepted(tmp_path):
    """The fix must actually pass, or the rule above is just a ban on restore-keys."""
    wf = tmp_path / "wf"
    wf.mkdir()
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
        "          key: pip-v2-${{ runner.os }}-abc\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: node_modules\n"
        "          key: pip-publish-only-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-publish-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, f"partitioned prefix rejected:\n{proc.stderr}"


def test_lint_sees_cache_keys_declared_in_composite_actions(tmp_path):
    """The keys that matter mostly live in .github/actions, which the lint used to skip.

    A PR-triggered workflow here delegates its key to a composite action, so the workflow
    text carries only `${{ steps.x.outputs.key }}`. Before composite actions were read,
    the PR-side namespace was effectively empty and a publish prefix could not be
    compared against anything.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: pip-v2-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ steps.pip-cache.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pip-v2-publish-${{ runner.os }}\n"
        "          restore-keys: |\n"
        "            pip-v2-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite action's pip-v2- key was not seen, so the publish prefix matched "
        f"nothing:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v2-" in proc.stderr


def test_a_restore_keys_entry_that_opens_with_an_unexpandable_expression_is_refused(tmp_path):
    """A prefix whose leading expression cannot be expanded decides nothing.

    `${{ runner.os }}-` is NOT this case: it takes exactly Linux, Windows and macOS, so it
    expands into three decidable prefixes. `${{ matrix.flavour }}-` is unbounded, so
    whether it reaches a pull-request namespace cannot be answered here.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: k-${{ matrix.flavour }}\n"
        "          restore-keys: |\n"
        "            ${{ matrix.flavour }}-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1
    assert "not decidable" in proc.stderr


def _publish_with_restore_keys(key: str, prefixes: str) -> str:
    return (
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {key}\n"
        "          restore-keys: |\n" + prefixes
    )


def _pr_workflow(key: str) -> str:
    return (
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        f"          key: {key}\n"
    )


def test_a_publish_prefix_longer_than_the_pr_literal_head_is_caught(tmp_path):
    """One-directional prefix comparison missed this, and it is the common shape.

    A PR key `pip-v2-${{ runner.os }}-abc` has the literal head `pip-v2-`. A publish
    prefix `pip-v2-Linux-` is LONGER than that head, so `head.startswith(prefix)` is
    False and the pairing was accepted, while the runtime key `pip-v2-Linux-abc` does
    start with the prefix and would be restored. restore-keys matching is left-anchored
    and exact, per GitHub's dependency-caching reference, so either string being a prefix
    of the other means they can meet.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-v2-pub-${{ runner.os }}", "            pip-v2-Linux-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"longer publish prefix accepted:\n{proc.stdout}\n{proc.stderr}"
    assert "pip-v2-Linux-" in proc.stderr


def test_an_expression_led_pr_key_is_expanded_not_dropped(tmp_path):
    """Dropping these was a gate bypass, not a gap in coverage.

    `${{ runner.os }}-shared-abc` has no literal head, so it used to be filtered out of
    the comparison entirely and a publish prefix `Linux-shared-` passed. runner.os takes
    exactly Linux, Windows and macOS, so expanding it makes the case decidable.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("${{ runner.os }}-shared-abc"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pub-${{ runner.os }}", "            Linux-shared-\n")
    )
    proc = _run(wf)
    assert (
        proc.returncode == 1
    ), f"expression-led PR key silently dropped:\n{proc.stdout}\n{proc.stderr}"
    assert "Linux-shared-" in proc.stderr


def test_a_composite_that_builds_its_key_in_shell_is_read(tmp_path):
    """The real composites name their namespace in shell, not in `key:`.

    pip-cache-restore sets `prefix="pip-v2-${name}-..."` in a run step and exposes it as
    an output, so its YAML `key:` is only `${{ steps.probe.outputs.key }}`. A check that
    read YAML alone learned nothing about the very actions it was added to cover, and the
    earlier regression test hid that by putting a literal key in the workflow instead.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "outputs:\n"
        "  key:\n"
        "    value: ${{ steps.probe.outputs.key }}\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        prefix="pip-v2-${name}-${{ runner.os }}-py${pyver}-"\n'
        '        echo "key=${prefix}${hash}" >> "$GITHUB_OUTPUT"\n'
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-v2-pub", "            pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite's shell-built pip-v2- namespace was not seen:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v2-" in proc.stderr


def test_a_publish_only_composite_is_not_treated_as_a_pr_namespace(tmp_path):
    """Scanning every action made a publish-only cache reject its own prefix.

    That is a false failure on a safe configuration, and a security lint that fails on
    correct code is one someone eventually switches off. Only actions reachable from a
    pull-request-triggered workflow count.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "release-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: release cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: release-only-${{ runner.os }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("pip-v2-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/release-cache\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: release-only-pub\n"
        "          restore-keys: |\n"
        "            release-only-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a publish-only composite was treated as a PR namespace:\n" f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_restore_keys_prefix_after_a_blank_line_is_still_read(tmp_path):
    """A blank line inside the block scalar used to truncate the list silently.

    A YAML block scalar runs until the indentation drops, blank lines included, and
    actions/cache reads the value as a newline-delimited list and skips empty entries. So
    `safe-`, a blank line, then `shared-` really does offer `shared-` at runtime, while
    the reader stopped at the blank line and never compared it. That is a bypass anyone
    can reach by formatting a long restore-keys block for readability, and the dropped
    entries are exactly the ones furthest from the eye.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "release-only-${{ runner.os }}",
            "            release-only-\n\n            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a prefix after a blank line in the block was dropped, so the collision was "
        f"never compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_cache_key_in_a_local_reusable_workflow_is_seen(tmp_path):
    """`uses: ./.github/workflows/x.yml` names the file, not a directory with action.yml.

    Probing only for `action.yml` beneath the reference found nothing, so a reusable
    workflow called from a pull request declared keys that stayed outside the comparison
    entirely: reachable from a pull request in fact, invisible to the check. The
    composite-action case was already covered, which is what made this one easy to miss.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: reuse-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("reuse-v1-pub-${{ runner.os }}", "            reuse-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the reusable workflow's reuse-v1- key was not seen, so the publish prefix "
        f"matched nothing:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "reuse-v1-" in proc.stderr


def test_a_composite_key_equal_to_a_publish_key_is_caught(tmp_path):
    """An EQUAL key, not a prefix, and declared in a composite action rather than a workflow.

    Composite keys reached the prefix comparison but not the exact one, so a publish
    workflow sharing a literal key with a PR-reachable action and carrying no
    restore-keys at all passed. That is the original cache-poisoning shape, and it was
    the one route through this check with nothing watching it.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "shared-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: shared cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: wheels-shared-key\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/shared-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: wheels-shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a composite action's literal key equal to the publish key was accepted:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "wheels-shared-key" in proc.stderr


def test_a_shell_built_namespace_is_narrowed_by_the_inputs_callers_pass(tmp_path):
    """Recording a namespace more broadly than the real one is a false rejection.

    The pip cache builds `prefix="pip-${name}-..."`, so reading the shell alone records
    the bare head `pip-`, which then collides with any publish prefix beginning `pip-`
    including a properly partitioned `pip-release-` that no pull request can write.
    Substituting the `name:` values callers actually pass gives `pip-mlx-`, which does
    not.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        name="${{ inputs.name }}"\n'
        '        prefix="pip-${name}-${{ runner.os }}-"\n'
        '        echo "key=${prefix}abc" >> "$GITHUB_OUTPUT"\n'
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-release-pub-${{ runner.os }}", "            pip-release-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`pip-release-` cannot be written by a pull request whose only namespace is "
        f"`pip-mlx-`, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # And the narrowed namespace still has teeth: a publish prefix over `pip-mlx-` fails.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pip-mlx-pub-${{ runner.os }}", "            pip-mlx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a publish prefix over the PR's real `pip-mlx-` namespace must still be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-mlx-" in proc.stderr


def test_a_publish_composite_that_restores_a_pr_namespace_is_caught(tmp_path):
    """The publish side delegates to local actions too, and that half went unread.

    Only the top-level publish workflow was parsed for `restore-keys`, so a publish
    workflow whose composite owns the `actions/cache/restore` contributed no prefixes at
    all. A pull request writing `shared-*` against a publish-only composite restoring
    `shared-` therefore passed, and a comparison that collects nothing on one side
    reports success rather than admitting it looked at nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "publish-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: publish cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: shared-publish-${{ runner.os }}\n"
        "        restore-keys: |\n"
        "          shared-\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/publish-cache\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish composite's `shared-` fallback was never collected, so the "
        f"collision was not compared:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_narrowing_keeps_the_broad_head_when_a_caller_is_dynamic(tmp_path):
    """Substituting only the literal call sites discards the dynamic one's namespace.

    With one caller passing `name: mlx` and another `name: ${{ matrix.cache_name }}`, the
    literal set is non-empty, so narrowing replaced the broad `pip-v2-` head with
    `pip-v2-mlx-` alone. The matrix caller can still expand to `shared`, write
    `pip-v2-shared-abc`, and a publish `restore-keys: pip-v2-shared-` would pass. The
    narrowing I added to remove a false REJECTION had therefore opened a false
    ACCEPTANCE, which is the worse of the two.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pip-cache-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        name="${{ inputs.name }}"\n'
        '        prefix="pip-v2-${name}-"\n'
        '        echo "key=${prefix}abc" >> "$GITHUB_OUTPUT"\n'
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  literal:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: mlx\n"
        "  dynamic:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        cache_name: [shared, other]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: ${{ matrix.cache_name }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "pip-v2-shared-pub-${{ runner.os }}", "            pip-v2-shared-\n"
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a dynamic call site's namespace was discarded by the narrowing, so a publish "
        f"prefix over it passed:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v2-" in proc.stderr


def test_a_folded_restore_keys_block_is_one_prefix_not_several(tmp_path):
    """Folding joins the lines with spaces, so the runtime fallback is a single string.

    `restore-keys: >` over `safe-only-` and `shared-` reaches actions/cache as
    `safe-only- shared-`, which cannot restore a `shared-` key: there is no fallback
    named `shared-` at all. Reading each physical line as its own prefix invented one,
    and rejecting a configuration over an invented fallback is how a security lint earns
    the reputation that gets it switched off.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: safe-only-${{ runner.os }}\n"
        "          restore-keys: >\n"
        "            safe-only-\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a folded block is one space-joined prefix and cannot reach the `shared-` "
        f"namespace, so this must pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # The literal form of the same block really does offer `shared-`, and must fail.
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "safe-only-${{ runner.os }}",
            "            safe-only-\n            shared-\n",
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a literal block DOES offer `shared-` as a separate fallback and must be "
        f"rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_quoted_restore_keys_field_is_read(tmp_path):
    """`"restore-keys": |` is valid YAML and offers the same fallback.

    The lexical reader matched only the bare token, so this spelling produced no
    prefixes at all and the collision was accepted. One of several spellings that had to
    be added one at a time before the reader was replaced with the parser, which resolves
    all of them to the same mapping key.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        '          "key": pub-${{ runner.os }}\n'
        '          "restore-keys": |\n'
        "            shared-\n"
    )
    proc = _run(wf)
    assert (
        proc.returncode == 1
    ), f"a quoted restore-keys field was not read:\n{proc.stdout}\n{proc.stderr}"
    assert "shared-" in proc.stderr


def test_a_restore_keys_sequence_is_read(tmp_path):
    """`restore-keys: [a-, shared-]` is the sequence form, which the line reader never saw."""
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-${{ runner.os }}-abc"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: pub-${{ runner.os }}\n"
        "          restore-keys: [safe-, shared-]\n"
    )
    proc = _run(wf)
    assert (
        proc.returncode == 1
    ), f"a sequence-form restore-keys was not read:\n{proc.stdout}\n{proc.stderr}"
    assert "shared-" in proc.stderr


def test_a_flow_style_local_uses_is_followed(tmp_path):
    """`- {uses: ./.github/actions/x}` is the same step mapping in flow style.

    The traversal matched `uses:` lexically, so a flow-style or quoted-key call was never
    followed and the namespace that action declares stayed outside the comparison.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "shared-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: shared cache\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: flow-v1-${{ runner.os }}-abc\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - {uses: ./.github/actions/shared-cache}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("flow-v1-pub-${{ runner.os }}", "            flow-v1-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a flow-style local `uses` was not followed, so the action's namespace was "
        f"invisible:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "flow-v1-" in proc.stderr


def test_a_caller_supplied_key_is_resolved_not_dismissed(tmp_path):
    """`key: ${{ inputs.cache_key }}` is caller-supplied, which is not delegation.

    `steps.*` genuinely delegates: the real key is built in a composite's shell and is
    collected from there. `inputs.*` is different, because the value comes from the
    CALLER, so a pull request passing `cache_key: shared-abc` writes the `shared-`
    namespace. Treating the two alike dropped this key silently and a publish `shared-`
    fallback passed.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-abc\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub-${{ runner.os }}", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"a caller-supplied cache key was dismissed as delegation:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-" in proc.stderr


def test_a_key_delegated_to_a_step_output_is_still_accepted(tmp_path):
    """The other half of the rule above, and the reason it is not simply stricter.

    Every live caller of this repository's pip-cache-save passes
    `key: ${{ steps.pip-cache.outputs.key }}`, whose real namespace was already collected
    from the restoring action's shell. Reporting that as undecidable failed the live tree,
    which is the false-failure shape that gets a security check switched off, so
    resolved-with-no-literals has to mean delegation rather than doubt.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "cache-save"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: cache save\n"
        "inputs:\n"
        "  key:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache/save@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=own-v1-abc' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: ./.github/actions/cache-save\n"
        "        with:\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("unrelated-pub-${{ runner.os }}", "            unrelated-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a key delegated to a step output, against an unrelated publish namespace, "
        f"must pass:\n{proc.stdout}\n{proc.stderr}"
    )


def test_inputs_are_collected_through_a_wrapper_action(tmp_path):
    """Call sites live in composites too, not only in workflow files.

    A cache action reached through a wrapper gets its inputs from that wrapper. Reading
    only the top-level workflows meant that call site was invisible, so if the workflow
    ALSO called the action directly with a literal, every input looked resolved and the
    narrowing dropped the namespace the wrapper passes.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    inner = root / "actions" / "pip-cache-restore"
    wrapper = root / "actions" / "setup-wrapper"
    wf.mkdir(parents = True)
    inner.mkdir(parents = True)
    wrapper.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: pip cache restore\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        '        name="${{ inputs.name }}"\n'
        '        prefix="pip-v3-${name}-"\n'
        '        echo "key=${prefix}abc" >> "$GITHUB_OUTPUT"\n'
    )
    (wrapper / "action.yml").write_text(
        "name: setup wrapper\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: ./.github/actions/pip-cache-restore\n"
        "      with:\n"
        "        name: wrapped\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  direct:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pip-cache-restore\n"
        "        with:\n"
        "          name: direct\n"
        "  viawrapper:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/setup-wrapper\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys(
            "pip-v3-wrapped-pub-${{ runner.os }}", "            pip-v3-wrapped-\n"
        )
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the wrapper's `name: wrapped` call site was not collected, so the narrowing "
        f"dropped that namespace:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pip-v3-" in proc.stderr


def test_an_omission_before_the_first_literal_is_counted(tmp_path):
    """Counting omissions in the same pass made the answer depend on call-site order.

    A site that omitted an input had no bucket yet, so the omission went unrecorded, and
    a later site supplying a literal made the input look fully resolved. Verified before
    fixing: a composite called first with no `name` and then with `name: safe` reported
    `({'safe'}, True)`, so the namespace narrowed to `safe` and the action's real default
    namespace was left undefended.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pipc"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pipc\n"
        "inputs:\n"
        "  name:\n"
        "    default: shared\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"pipx-${name}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    # The omitting call site comes FIRST, which is the ordering that used to be lost.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  defaulted:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "  named:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pipc\n"
        "        with:\n"
        "          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("pipx-pub-${{ runner.os }}", "            pipx-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the first call site omits `name`, so the namespace is not fully resolved and "
        f"the broad `pipx-` head must still be defended:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "pipx-" in proc.stderr


def test_a_prefix_that_opens_with_a_variable_is_recovered(tmp_path):
    """`prefix="${name}-pip-..."` put its literal part after the variable.

    The head pattern requires an alphanumeric start, so this composite contributed no
    head at all, while the `steps.*` key reading its output was dismissed as delegation.
    A publish `restore-keys: shared-pip-` then had nothing to be compared against, which
    is the failure mode where a check reports success having looked at nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "varfirst"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: varfirst\n"
        "inputs:\n"
        "  name:\n"
        "    required: true\n"
        "outputs:\n"
        "  key:\n"
        "    value: ${{ steps.probe.outputs.key }}\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - id: probe\n"
        "      shell: bash\n"
        "      run: |\n"
        "        name=\"${{ inputs.name }}\"\n"
        "        prefix=\"${name}-pip-${{ runner.os }}-\"\n"
        "        echo \"key=${prefix}abc\" >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: vf\n"
        "        uses: ./.github/actions/varfirst\n"
        "        with:\n"
        "          name: shared\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ steps.vf.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pip-pub-${{ runner.os }}", "            shared-pip-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the namespace is `shared-pip-` once `name` is substituted, so the publish "
        f"fallback over it must be rejected:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-pip-" in proc.stderr


def test_an_input_backed_key_is_resolved_before_the_exact_comparison(tmp_path):
    """The plainest shape of all: an equal key, with no `restore-keys` anywhere.

    Input resolution reached the prefix comparison through `pr_heads` and stopped there,
    so a pull request writing `shared-key` directly, against a dispatch workflow calling
    a reusable workflow whose key is `${{ inputs.cache_key }}` with
    `cache_key: shared-key`, compared a literal against an unexpanded expression and
    matched nothing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    wf.mkdir(parents = True)
    (wf / "shared-build.yml").write_text(
        "name: shared-build\n"
        "on:\n"
        "  workflow_call:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        required: true\n"
        "        type: string\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-key"))
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  call:\n"
        "    uses: ./.github/workflows/shared-build.yml\n"
        "    with:\n"
        "      cache_key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish side's key resolves to `shared-key`, which the PR writes directly, "
        f"so this exact collision must be caught:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-key" in proc.stderr


def test_a_longer_fallback_over_a_complete_pr_key_is_accepted(tmp_path):
    """The reverse-prefix direction only holds for a head cut short by an expression.

    A PR key that is exactly `shared` is saved as `shared`, and
    `shared`.startswith(`shared-long`) is false, so a publish fallback `shared-long`
    cannot restore it. Allowing the reverse unconditionally rejected every longer
    fallback that merely shared an opening with a complete key, which is a false failure
    on a correct configuration.
    """
    wf = tmp_path / "wf"
    wf.mkdir()
    (wf / "pr-build.yml").write_text(_pr_workflow("shared"))
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-long-pub", "            shared-long-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared-long-` cannot restore a key that is exactly `shared`, so this must "
        f"pass:\n{proc.stdout}\n{proc.stderr}"
    )

    # The truncated case still has teeth: the same fallback over an expression-completed
    # key CAN meet it at runtime.
    (wf / "pr-build.yml").write_text(_pr_workflow("shared-long-${{ runner.os }}-abc"))
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the runtime key here really does start with `shared-long-`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def _lint_module():
    """The lint script loaded as a module, for testing its predicates directly.

    The rest of this file drives the script as a subprocess, which is the right way to
    test the tool but cannot reach a single function.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_lint_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_truncation_predicate_reads_the_key():
    """The rule above is only as good as this predicate, so it is tested directly."""
    lint = _lint_module()
    _is_truncated = lint._is_truncated
    _prefix_compatible = lint._prefix_compatible
    # `runner.os` is expanded before the head is taken, so a key containing only that
    # expression ends up with a COMPLETE head and is not truncated. It is the expressions
    # this check cannot expand that cut a head short.
    cases = [
        ("shared", False),
        ("shared-long-abc", False),
        ("pip-v2-${{ runner.os }}-abc", False),
        ("${{ runner.os }}-shared-abc", False),
        ("pip-${{ hashFiles('x') }}", True),
        ("pip-${{ matrix.flavour }}-abc", True),
    ]
    for key, expected in cases:
        assert _is_truncated(key) is expected, f"_is_truncated({key!r})"

    # A head cut short by an expression may be reached by a LONGER publish prefix; a
    # complete key may not.
    assert _prefix_compatible("pip-v2-", "pip-v2-Linux-", True) is True
    assert _prefix_compatible("shared", "shared-long", False) is False
    # Either way, a prefix the head already starts with is always compatible.
    assert _prefix_compatible("shared-long-abc", "shared-", False) is True


def test_a_declared_input_default_is_part_of_the_namespace(tmp_path):
    """Actions applies a declared default when the caller omits the input.

    A composite declaring `cache_key` with default `shared-key` writes that namespace on
    a bare invocation. Reading only the call sites left the key as an unexpanded
    expression, so the exact comparison matched nothing, and with no `restore-keys` in
    play the undecidable-prefix path never reported it either. A silent pass.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "defaulted-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: defaulted cache\n"
        "inputs:\n"
        "  cache_key:\n"
        "    default: shared-key\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/defaulted-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the bare invocation writes the default namespace `shared-key`, which the "
        f"publish workflow uses exactly:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "shared-key" in proc.stderr


def test_a_declared_default_does_not_settle_an_explicit_dynamic_value(tmp_path):
    """A default applies to an OMISSION. It says nothing about an explicit override.

    Recording a declared default as blanket resolution -- the first fix for the omission
    case -- meant a second caller passing `key: ${{ matrix.cache_key }}` was marked
    resolved on the strength of a default it had overridden. Its namespace is unknown, so
    the publish prefix `shared-` cannot be shown not to reach it, and dropping it turned
    the fix for one false failure into a silent bypass.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "defaulted-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: defaulted cache\n"
        "inputs:\n"
        "  cache_key:\n"
        "    default: unrelated-default\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    strategy:\n"
        "      matrix:\n"
        "        cache_key: [a, b]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/defaulted-cache\n"
        "        with:\n"
        "          cache_key: ${{ matrix.cache_key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key\n"
        "          restore-keys: |\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix caller overrode the default with a value the check cannot expand, "
        f"so the PR namespace is undecided and the publish prefix cannot be "
        f"cleared:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_shell_key_that_never_leaves_the_step_is_not_a_namespace(tmp_path):
    """A shell variable becomes a cache key by being written to `$GITHUB_OUTPUT`.

    Reading every assignment named `key` or `prefix` regardless meant an unrelated
    `key="shared-${RANDOM}"` -- a temp-file name, in a workflow with no cache at all --
    registered `shared-` as a pull-request cache namespace and failed the publish
    workflow's legitimate `restore-keys: shared-`. A false failure on a correct tree is
    how a guard gets exempted, so the recovered value has to be tied to something that
    can actually become a key.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n"
        "  pull_request:\n"
        "jobs:\n"
        "  build:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - run: |\n"
        '          key="shared-${RANDOM}"\n'
        '          echo hello > "/tmp/$key"\n'
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "jobs:\n"
        "  publish:\n"
        "    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n"
        "          path: wheels\n"
        "          key: shared-key-v1\n"
        "          restore-keys: |\n"
        "            shared-\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the pull request workflow caches nothing and the assignment never reaches "
        f"$GITHUB_OUTPUT, so `shared-` is not a namespace it can "
        f"write:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_input_embedded_in_a_key_is_expanded(tmp_path):
    """`key: prefix-${{ inputs.name }}` with `name: shared` runs as `prefix-shared`.

    Only a key that was NOTHING but one expression got expanded, so the commonest
    spelling -- an input with a literal prefix in front of it -- kept its raw text
    through the exact comparison and matched no publish key. With no `restore-keys` on
    the publish side the prefix pass never looked either, leaving the plainest exact
    collision unguarded.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "embedded"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: embedded\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n"
        "  using: composite\n"
        "  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n"
        "        path: wheels\n"
        "        key: prefix-${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/embedded\n"
        "        with:\n          name: shared\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-shared\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the composite writes `prefix-shared`, which the publish workflow restores by "
        f"that exact name:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "prefix-shared" in proc.stderr


def test_runner_os_is_expanded_before_the_exact_comparison(tmp_path):
    """`shared-${{ runner.os }}` and `shared-Linux` are the same key on a Linux runner.

    The exact comparison was a plain string test, so two keys that are equal at run time
    but differ textually never met. `_prefix_candidates` already knew the three values
    `runner.os` takes, but only the fallback-prefix pass used them, so a publish workflow
    with no `restore-keys` never benefited.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ runner.os }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-Linux\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"both jobs write `shared-Linux` on a Linux runner:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_an_unresolvable_pr_key_is_reported_against_an_exact_publish_key(tmp_path):
    """Fail closed when the PR key's value cannot be settled and could equal a publish key.

    Unresolved keys were reported only from inside the restore-prefix pass, so a publish
    workflow that uses an exact key and no `restore-keys` at all had the question never
    asked: a matrix-supplied `shared-${{ matrix.tag }}` may well produce `shared-key`,
    and the lint exited 0.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        tag: [a, b]\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.tag }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix value is unknown and `shared-key` is one of the keys it could "
        f"produce:\n{proc.stdout}\n{proc.stderr}"
    )

    # ... and a publish key the PR key's fixed head cannot lead to stays accepted, or
    # every unresolved key in the tree would reject every publish key in it.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: wheels-publish-only\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"a key headed `shared-` cannot become `wheels-publish-only` however its tail "
        f"resolves:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_targets_sharing_an_input_name_keep_their_own_namespaces(tmp_path):
    """An input name does not identify a namespace; the definition it belongs to does.

    Merging every reachable target's inputs by field name handed one composite's values
    to another, so a publish key equal to a value only the NON-caching composite ever
    receives was rejected. A guard that fails a correct configuration is one that gets
    deleted, so this direction matters as much as the bypasses.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    caching = root / "actions" / "caching"
    unrelated = root / "actions" / "unrelated"
    wf.mkdir(parents = True)
    caching.mkdir(parents = True)
    unrelated.mkdir(parents = True)
    (caching / "action.yml").write_text(
        "name: caching\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    # Same input NAME, no cache anywhere in it.
    (unrelated / "action.yml").write_text(
        "name: unrelated\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.cache_key }}\n"
        "      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/caching\n"
        "        with:\n          cache_key: safe-key\n"
        "      - uses: ./.github/actions/unrelated\n"
        "        with:\n          cache_key: publish-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: publish-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`publish-key` only ever reaches the composite that caches nothing, so no PR "
        f"cache writes it:\n{proc.stdout}\n{proc.stderr}"
    )

    # The caching composite receiving it IS a collision, so the narrowing kept its teeth.
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/caching\n"
        "        with:\n          cache_key: publish-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"now the caching composite is the one given `publish-key`:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_reusable_workflow_is_named_by_its_file_not_its_directory():
    """`uses: ./.github/workflows/reuse.yml` names a FILE. An action names a directory.

    Taking the parent directory for both made every reusable workflow a target called
    `workflows`, which no call site mentions, so the values its callers pass were never
    recovered and a key built from one of them had no namespace at all.
    """
    lint = _lint_module()
    # The FULL local reference, which is what a `uses:` line writes. A basename was
    # enough to tell a workflow from an action, and not enough to tell two actions
    # apart: `.github/actions/a/cache` and `.github/actions/b/cache` both end in
    # `cache`, so their call sites pooled and one action's value was invented for the
    # other.
    assert (
        lint._target_name(Path(".github/workflows/reuse.yml"))
        == ".github/workflows/reuse.yml"
    )
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yml"))
        == ".github/actions/pip-cache"
    )
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yaml"))
        == ".github/actions/pip-cache"
    )
    # An action is still named by its directory and a reusable workflow by its file,
    # which is the distinction this started from.
    assert lint._target_name(Path(".github/actions/a/cache/action.yml")) != (
        lint._target_name(Path(".github/actions/b/cache/action.yml"))
    )


def test_an_unresolvable_publish_key_is_reported_too(tmp_path):
    """Failing closed on one side only leaves half this check's own rule unenforced.

    A publish composite called with `cache_key: ${{ matrix.cache_key }}` may well
    produce a literal a pull request also writes, and the publish branch skipped every
    key that still contained an expression. The PR side had already been made to fail
    closed, which made the asymmetry easy to miss: the rule looked enforced.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pub-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pub cache\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        cache_key: [x, y]\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-cache\n"
        "        with:\n          cache_key: ${{ matrix.cache_key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the matrix could produce `shared-key`, which the pull request writes:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_two_identically_spelled_unresolved_keys_collide(tmp_path):
    """`shared-${{ hashFiles('lock') }}` on both sides is one key at run time.

    Both sides were dropped for still containing an expression, so the most direct
    collision there is -- the same key, written the same way, in both workflows -- was
    invisible. A pull request that leaves the lockfile untouched writes exactly the entry
    the publish run restores.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    body = (
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ hashFiles('lock') }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n" + body
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n" + body
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"identical keys resolve identically:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "identically" in proc.stderr


def test_a_publish_key_is_expanded_with_its_own_targets_inputs(tmp_path):
    """The publish side kept using the merged namespace, so its scoping changed nothing.

    `publish_by_target` was computed and never read. Two publish-reachable actions
    sharing an input name therefore had every key expanded with both their values, and a
    cache composite given `publish-key` was also expanded to an unrelated action's
    `safe-key` and reported as colliding with a pull request cache of that name.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    caching = root / "actions" / "pub-caching"
    unrelated = root / "actions" / "pub-unrelated"
    wf.mkdir(parents = True)
    caching.mkdir(parents = True)
    unrelated.mkdir(parents = True)
    (caching / "action.yml").write_text(
        "name: pub caching\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n        key: ${{ inputs.cache_key }}\n"
    )
    (unrelated / "action.yml").write_text(
        "name: pub unrelated\n"
        "inputs:\n  cache_key:\n    description: k\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.cache_key }}\n      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: safe-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-caching\n"
        "        with:\n          cache_key: publish-key\n"
        "      - uses: ./.github/actions/pub-unrelated\n"
        "        with:\n          cache_key: safe-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`safe-key` only ever reaches the action that caches nothing:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_composite_key_is_not_counted_a_second_time_without_its_inputs(tmp_path):
    """A duplicate entry with an empty namespace made a decided key look undecided.

    Composite YAML keys were added once with their target's inputs and once more with no
    namespace at all. The second copy expanded to nothing, counted as unresolved, and the
    fail-closed rule then rejected an unrelated publish key on the strength of the
    duplicate -- even though every caller passes a literal and the key can only be one
    thing.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "decided"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: decided\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/decided\n"
        "        with:\n          name: safe\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-other\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"every caller passes `name: safe`, so the PR key can only be `prefix-safe` and "
        f"`prefix-other` is a different namespace:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_differently_spelled_unresolved_keys_are_paired(tmp_path):
    """Both sides unresolved, spelled differently, is the case the earlier fixes missed.

    The identical-text rule only reaches keys written the same way, and an unresolved
    publish key was compared against RESOLVED PR keys alone before continuing, so
    `shared-${{ matrix.pr_part }}` and `shared-${{ matrix.pub_part }}` were never
    paired even though both can become `shared-x`. Each earlier fix covered one
    unresolved side at a time.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pr_part: [a, b]\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.pr_part }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pub_part: [x, y]\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: shared-${{ matrix.pub_part }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"both keys are headed `shared-` and neither tail is known, so they can be the "
        f"same entry:\n{proc.stdout}\n{proc.stderr}"
    )

    # Incompatible heads stay accepted, or every unresolved key in a tree would reject
    # every other one.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    strategy:\n      matrix:\n        pub_part: [x, y]\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: wheels-only-${{ matrix.pub_part }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared-` and `wheels-only-` cannot become each other:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_delegated_key_whose_producer_was_not_read_stays_undecided(tmp_path):
    """Delegation settles a key only when the producer's namespace was recovered.

    `_shell_built_key_prefixes` reads two narrow spellings. A workflow emitting its key
    with `printf 'key=%s\\n'` matches neither, so nothing was recorded -- and dismissing
    the key as "delegated" turned an unread producer into a clean bill of health, letting
    a publish `restore-keys: shared-` through.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: printf 'key=%s\\n' \"shared-$GITHUB_SHA\" >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the producer's spelling was not recognised, so the key's namespace is "
        f"unknown and `shared-` cannot be cleared:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_literal_producer_output_is_read_as_a_key(tmp_path):
    """A producer emitting a fully literal key is resolved, not unrecognised.

    Requiring a dynamic HEAD as the evidence that a producer was understood failed the
    opposite case: `echo 'key=own-v1-abc'` has no tail to assemble, so the head
    extractor finds nothing while the key is completely known. The value is an exact
    cache key, so it joins the comparison rather than only vouching for the step.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=own-v1-abc' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: unrelated-v1\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the key is known to be `own-v1-abc`, which is not `unrelated-v1`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )

    # And the literal really is compared, rather than merely vouching for the producer.
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: own-v1-abc\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish workflow restores exactly the key the probe writes:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_side_delegated_key_is_resolved_from_its_own_shell(tmp_path):
    """Shell heads were collected from PR-reachable documents only.

    So a publish workflow that emits `key=shared-key` and restores
    `${{ steps.probe.outputs.key }}` had that key dismissed as delegated with nothing
    recovered to compare it against, and a pull request writing the literal `shared-key`
    passed an exact collision.
    """
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=shared-key' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the publish workflow's own shell says the key is `shared-key`, which the "
        f"pull request writes:\n{proc.stdout}\n{proc.stderr}"
    )


def test_two_actions_sharing_a_directory_name_keep_their_call_sites(tmp_path):
    """`a/cache` and `b/cache` are different actions, however they end.

    Matching call sites on the last path component pooled them, so a value passed to one
    was attributed to the other and a publish key no pull request writes was rejected.
    """
    root = tmp_path / ".github"
    wf = root / "workflows"
    first = root / "actions" / "a" / "cache"
    second = root / "actions" / "b" / "cache"
    wf.mkdir(parents = True)
    first.mkdir(parents = True)
    second.mkdir(parents = True)
    (first / "action.yml").write_text(
        "name: a cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (second / "action.yml").write_text(
        "name: b cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - run: echo ${{ inputs.name }}\n      shell: bash\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/a/cache\n"
        "        with:\n          name: safe\n"
        "      - uses: ./.github/actions/b/cache\n"
        "        with:\n          name: shared\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-shared\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"`shared` only ever reaches b/cache, which caches nothing, so no PR cache "
        f"writes `prefix-shared`:\n{proc.stdout}\n{proc.stderr}"
    )
