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
    """A narrowed host skips the PR that narrows it, since the merge ref carries the restriction."""
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
    """A host with continue-on-error or a false if: is no gate; the lint must reject it."""
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
    """A pipeline or || true can detach the lint's exit status, so the invocation must be rejected."""
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
        "/tmp/fakepython scripts/lint_workflow_triggers.py",
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
    """A commented-out run must not count as a host, or --require-host passes with no workflow."""
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
    """Keep ownerless rules too: a pattern with no owners still clears ownership for what it matches."""
    rules = []
    for line in text.splitlines():
        fields = line.split("#", 1)[0].split()
        if fields:
            rules.append((fields[0], fields[1:]))
    return rules


def _pattern_regex(pattern: str) -> re.Pattern:
    """CODEOWNERS globs: * stops at '/' and ** crosses directories, which fnmatch gets wrong."""
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
    """A leading or inner slash anchors a CODEOWNERS pattern to root; a bare name floats to any depth."""
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
    """Each workflow needs an effective code owner, since GitHub applies only the last matching rule."""
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
        ("/.github/workflows/", ".github/workflows/lint.yml", True),
        ("/scripts", "scripts/data/x.txt", True),
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
    """Match GitHub both ways: under-matching hides a stolen owner, over-matching fails valid changes."""
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
    """A bare word is not a usable owner: counting it would let a trailing rule quietly disown a path."""
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
        (f"/{CODEOWNERS_PROBES[0]}", []),
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
    """A restore-keys prefix can adopt a PR-written entry, so it is checked like an exact key."""
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
    """Cache keys live mostly in .github/actions, so the lint must read composite actions too."""
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
    """A restore-keys prefix that opens with an unbounded expression cannot be decided, so it is refused."""
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
    """A prefix collision runs both ways: either key being a prefix of the other can meet."""
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
    """An expression-led PR key must be expanded, not dropped: runner.os takes three known values."""
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
    """Some composites build the key in shell, not in key:, so their shell namespace must be read."""
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
    """Only PR-reachable actions form a PR namespace; publish-only caches must not be rejected."""
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
    """A blank line in restore-keys does not end the list; actions/cache still restores later prefixes."""
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
    """A local reusable workflow is named by its file; read its cache keys from that file."""
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
    """A publish key equal to a composite-declared key must be caught, not just prefix matches."""
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
    """Shell-built namespaces are narrowed by the name values callers pass, or a bare head over-rejects."""
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
    """Publish composites that restore a PR namespace are read too, not only top-level workflows."""
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
    """A dynamic caller keeps the broad head, since literal-only narrowing would drop its namespace."""
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
    """A folded restore-keys block is one space-joined string, not one prefix per physical line."""
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
    """A quoted 'restore-keys' key is the same YAML mapping key, so the parser must read it."""
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
    """Flow-style and quoted-key 'uses' steps must be followed, not only the lexical block form."""
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
    """An inputs.* key comes from the caller, not a delegated step; resolve it, do not dismiss it."""
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
    """A key delegated to a step output is accepted when its producer's namespace was already collected."""
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
    """Call sites in wrapper composites count too, since a wrapped cache action gets its inputs there."""
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
    """An omitted input must be counted even when a later call site supplies a literal."""
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
        '        name="${{ inputs.name }}"\n'
        '        prefix="pipx-${name}-"\n'
        '        echo "key=${prefix}abc" >> "$GITHUB_OUTPUT"\n'
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
    """Recover the literal tail of a prefix that opens with a variable, rather than dropping it."""
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
        '        name="${{ inputs.name }}"\n'
        '        prefix="${name}-pip-${{ runner.os }}-"\n'
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
    """An input-backed key must be resolved before the exact comparison, or it matches nothing."""
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
    """The reverse-prefix match applies only to a head cut short by an expression, not to complete keys."""
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

    (wf / "pr-build.yml").write_text(_pr_workflow("shared-long-${{ runner.os }}-abc"))
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the runtime key here really does start with `shared-long-`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def _lint_module():
    """Loads the lint script as a module so its predicates can be tested directly, not via subprocess."""
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

    assert _prefix_compatible("pip-v2-", "pip-v2-Linux-", True) is True
    assert _prefix_compatible("shared", "shared-long", False) is False
    assert _prefix_compatible("shared-long-abc", "shared-", False) is True


def test_a_declared_input_default_is_part_of_the_namespace(tmp_path):
    """A declared default for an omitted input is part of the namespace the action writes."""
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
    """A declared default covers only an omitted input, never an explicit dynamic value such as a matrix."""
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
    """Only a shell key written to $GITHUB_OUTPUT can become a cache namespace; temp names cannot."""
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
    """Expand an input embedded after a literal prefix, not only a key that is a lone expression."""
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
    """Expand runner.os before the exact comparison; shared-Linux and shared-${{ runner.os }} match."""
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
        f"both jobs write `shared-Linux` on a Linux runner:\n{proc.stdout}\n" f"{proc.stderr}"
    )


def test_an_unresolvable_pr_key_is_reported_against_an_exact_publish_key(tmp_path):
    """An unresolvable PR key must fail closed against an exact publish key, not only against prefixes."""
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
    """Same-named inputs on different actions need separate namespaces, or a correct config fails."""
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
    """A reusable workflow is named by its file, not its parent directory, which would collapse them all."""
    lint = _lint_module()
    # The FULL local reference: basenames cannot tell .github/actions/a/cache from b/cache.
    assert lint._target_name(Path(".github/workflows/reuse.yml")) == ".github/workflows/reuse.yml"
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yml"))
        == ".github/actions/pip-cache"
    )
    assert (
        lint._target_name(Path(".github/actions/pip-cache/action.yaml"))
        == ".github/actions/pip-cache"
    )
    assert lint._target_name(Path(".github/actions/a/cache/action.yml")) != (
        lint._target_name(Path(".github/actions/b/cache/action.yml"))
    )


def test_an_unresolvable_publish_key_is_reported_too(tmp_path):
    """An unresolvable publish key must be reported too, not skipped; it may equal a PR literal."""
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
    """Identically spelled unresolved keys on both sides can collide at run time, so compare them."""
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
    assert (
        proc.returncode == 1
    ), f"identical keys resolve identically:\n{proc.stdout}\n{proc.stderr}"
    assert "identically" in proc.stderr


def test_a_publish_key_is_expanded_with_its_own_targets_inputs(tmp_path):
    """Expand a publish key with its own target's inputs, not the merged set of all targets."""
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
    """A composite key must not be added twice, once without its inputs, or it looks unresolved."""
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
    """Unresolved keys spelled differently must still be paired when both can become the same value."""
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
        f"`shared-` and `wheels-only-` cannot become each other:\n{proc.stdout}\n" f"{proc.stderr}"
    )


def test_a_delegated_key_whose_producer_was_not_read_stays_undecided(tmp_path):
    """A delegated key stays undecided if its producer was not read, such as a printf 'key=%s' emitter."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        '        run: printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
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
    """A literal producer output is an exact key, so it joins the comparison rather than being skipped."""
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
    """Shell heads must also be collected from publish workflows, not only PR-reachable documents."""
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
    """a/cache and b/cache are different actions; match call sites by full path, not the last component."""
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


def test_one_readable_producer_does_not_vouch_for_an_unreadable_one(tmp_path):
    """Resolution belongs to each producing step; a readable step must not vouch for an unreadable one."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: safe\n"
        "        run: echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        "      - id: make\n"
        '        run: printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the key in use comes from the step that could NOT be read, whatever the "
        f"other step spells correctly:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_producer_that_declares_its_key_inline_is_readable():
    """A step's or local action's inline key: is its output, so the producer is readable."""
    lint = _lint_module()
    # Identities are (document, job, step id): a step id is unique only within its job.
    here = ("wf.yml", "build")
    producers = {("wf.yml", "build", "probe"): {"key"}}
    assert lint._delegation_is_read("${{ steps.probe.outputs.key }}", producers, here) is True
    assert (
        lint._delegation_is_read(
            "${{ steps.probe.outputs.key }}", {("wf.yml", "build", "probe"): set()}, here
        )
        is False
    )
    # Per OUTPUT, not per step: a step may write several, and recovering one says
    # nothing about the others.
    assert lint._delegation_is_read("${{ steps.probe.outputs.danger }}", producers, here) is False
    # The SAME id in a different job is a different step and vouches for nothing.
    assert (
        lint._delegation_is_read(
            "${{ steps.probe.outputs.key }}", {("wf.yml", "other", "probe"): {"key"}}, here
        )
        is False
    )
    assert (
        lint._delegation_is_read(
            "${{ steps.probe.outputs.key }}", {("z.yml", "build", "probe"): {"key"}}, here
        )
        is False
    )
    assert lint._delegation_is_read("${{ steps.other.outputs.key }}", producers, here) is False
    assert lint._delegation_is_read("${{ needs.build.outputs.key }}", producers, here) is False


def test_a_readable_namesake_in_another_job_vouches_for_nothing(tmp_path):
    """Step ids are unique only within a job; identify producers by document, job and step id."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n"
        "  first:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        '        run: printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.probe.outputs.key }}\n"
        "  second:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: probe\n"
        "        run: echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the caching job's own `probe` could not be read, whatever the other job's "
        f"step of the same name spells:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_inline_key_input_does_not_certify_an_unrelated_output(tmp_path):
    """A with: key input proves nothing about the action's outputs; only the output it publishes counts."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "sneaky"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: sneaky\n"
        "inputs:\n  key:\n    description: unrelated\n"
        "outputs:\n"
        "  key:\n"
        "    description: the real cache key\n"
        "    value: ${{ steps.inner.outputs.key }}\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - id: inner\n"
        "      shell: bash\n"
        '      run: printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: maker\n"
        "        uses: ./.github/actions/sneaky\n"
        "        with:\n          key: safe-key\n"
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.maker.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the action's published key comes from a command that could not be read; the "
        f"`key` input it happens to accept says nothing about it:\n{proc.stdout}\n"
        f"{proc.stderr}"
    )


def test_a_top_level_publish_input_is_not_resolved_by_a_child_targets_value(tmp_path):
    """A dispatch workflow's own inputs are chosen by the dispatcher, so no child value may resolve them."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "unrelated"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: unrelated\n"
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
        "        with:\n          path: wheels\n          key: shared-key\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n"
        "  workflow_dispatch:\n"
        "    inputs:\n"
        "      cache_key:\n"
        "        description: chosen by whoever dispatches\n"
        "        type: string\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/unrelated\n"
        "        with:\n          cache_key: safe-key\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ inputs.cache_key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the dispatch input can be given `shared-key`, and the unrelated action's "
        f"`safe-key` says nothing about it:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_action_used_from_a_checkout_subdirectory_is_reachable(tmp_path):
    """Checkout-subdirectory refs like ./unsloth/.github/actions/x name a reachable action."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "nested-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: nested cache\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: shared-inner\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        # checked out under `unsloth/`, so the reference carries that prefix
        "      - uses: ./unsloth/.github/actions/nested-cache\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: shared-inner\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the prefixed reference names the same action, whose key the publish workflow "
        f"restores exactly:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_commented_out_output_does_not_certify_a_producer(tmp_path):
    """A commented line executes nothing, so it must not certify an unreadable producer as readable."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        "          # echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        '          printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the only line that runs is the printf, which this check cannot read:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_an_unquoted_scalar_key_is_compared(tmp_path):
    """An unquoted key: 123 is an int to YAML and a cache key to Actions, so it must be compared."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: 123\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: 123\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, f"both workflows use the same key:\n{proc.stdout}\n{proc.stderr}"


def test_a_wrapper_forwarding_its_own_input_is_resolved(tmp_path):
    """A forwarded ${{ inputs.name }} is whatever the wrapper's callers pass, so it must be resolved."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    inner = root / "actions" / "inner-cache"
    wrapper = root / "actions" / "wrapper"
    wf.mkdir(parents = True)
    inner.mkdir(parents = True)
    wrapper.mkdir(parents = True)
    (inner / "action.yml").write_text(
        "name: inner cache\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache@v4\n"
        "      with:\n        path: wheels\n        key: prefix-${{ inputs.name }}\n"
    )
    (wrapper / "action.yml").write_text(
        "name: wrapper\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: ./.github/actions/inner-cache\n"
        "      with:\n        name: ${{ inputs.name }}\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/wrapper\n"
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
        f"every caller passes `name: safe`, so the only PR key is `prefix-safe`:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )

    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: prefix-safe\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"`prefix-safe` is exactly what the wrapper produces:\n{proc.stdout}\n" f"{proc.stderr}"
    )


def test_one_readable_output_does_not_certify_another_from_the_same_step(tmp_path):
    """Readability belongs to each output, so a recognisable key must not certify an unreadable one."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        "          echo 'key=safe-key' >> \"$GITHUB_OUTPUT\"\n"
        '          printf \'danger=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.danger }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the cache uses `danger`, which was written by the form this check cannot "
        f"read:\n{proc.stdout}\n{proc.stderr}"
    )


def test_an_unrelated_assignment_does_not_certify_the_output(tmp_path):
    """A recognisable assignment elsewhere in the body is not evidence about the output being read."""
    wf = tmp_path / ".github" / "workflows"
    wf.mkdir(parents = True)
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: |\n"
        '          safe_key="safe-${RANDOM}"\n'
        '          printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/save@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    (wf / "release-desktop.yml").write_text(
        _publish_with_restore_keys("shared-pub", "            shared-\n")
    )
    proc = _run(wf)
    assert (
        proc.returncode == 1
    ), f"the only line writing an output is the printf:\n{proc.stdout}\n{proc.stderr}"


def test_two_publish_jobs_sharing_a_raw_key_keep_their_own_scopes(tmp_path):
    """(path, key) is not unique, so each publish job must keep its own scope, not the first job's."""
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
        "jobs:\n"
        "  first:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        "        run: echo 'key=unrelated-v1' >> \"$GITHUB_OUTPUT\"\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
        "  second:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - id: make\n"
        '        run: printf \'key=%s\\n\' "shared-$GITHUB_SHA" >> "$GITHUB_OUTPUT"\n'
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n"
        "          key: ${{ steps.make.outputs.key }}\n"
    )
    proc = _run(wf)
    assert proc.returncode == 1, (
        f"the second job's producer could not be read, and the first job's readable "
        f"namesake says nothing about it:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_key_passed_to_a_non_cache_action_is_not_a_cache_namespace(tmp_path):
    """Not every field named key is a cache: a key passed to a non-cache action is not a namespace."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "signer"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: signer\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: some/signing-action@v1\n"
        "      with:\n        key: release-key\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/signer\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache/restore@v4\n"
        "        with:\n          path: wheels\n          key: release-key\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the pull request path caches nothing; `key` there is a signing key:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_a_call_site_reached_through_a_checkout_prefix_is_matched(tmp_path):
    """Call-site matching must strip the runtime prefix too, or literal inputs are never recovered."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "prefixed-cache"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: prefixed cache\n"
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
        "      - uses: ./repo/.github/actions/prefixed-cache\n"
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
        f"the prefixed call site passes `name: safe`, so the only PR key is "
        f"`prefix-safe`:\n{proc.stdout}\n{proc.stderr}"
    )


def test_a_publish_restore_prefix_is_expanded_with_its_own_inputs(tmp_path):
    """A restore-keys fallback must expand its inputs, or it reduces to a far broader prefix."""
    root = tmp_path / ".github"
    wf = root / "workflows"
    action = root / "actions" / "pub-restore"
    wf.mkdir(parents = True)
    action.mkdir(parents = True)
    (action / "action.yml").write_text(
        "name: pub restore\n"
        "inputs:\n  name:\n    description: n\n"
        "runs:\n  using: composite\n  steps:\n"
        "    - uses: actions/cache/restore@v4\n"
        "      with:\n        path: wheels\n"
        "        key: prefix-${{ inputs.name }}-exact\n"
        "        restore-keys: |\n"
        "          prefix-${{ inputs.name }}-\n"
    )
    (wf / "pr-build.yml").write_text(
        "name: pr-build\n"
        "on:\n  pull_request:\n"
        "jobs:\n  build:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: actions/cache@v4\n"
        "        with:\n          path: wheels\n          key: prefix-pr-exact\n"
    )
    (wf / "release-desktop.yml").write_text(
        "name: release-desktop\n"
        "on:\n  workflow_dispatch:\n"
        "jobs:\n  publish:\n    runs-on: ubuntu-latest\n"
        "    steps:\n"
        "      - uses: ./.github/actions/pub-restore\n"
        "        with:\n          name: publish\n"
    )
    proc = _run(wf)
    assert proc.returncode == 0, (
        f"the runtime fallback is `prefix-publish-`, which cannot reach "
        f"`prefix-pr-exact`:\n{proc.stdout}\n{proc.stderr}"
    )
