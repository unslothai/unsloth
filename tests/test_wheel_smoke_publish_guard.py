# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the "Publishing path uploads the wheel only" guard.

The guard lives inside a heredoc in .github/workflows/wheel-smoke.yml, so these
tests extract the shipped script text rather than a copy of it, and run it
against synthetic build.sh files. Two ways a guard like this silently passes a
release that publishes an sdist:

  - it inspects only the first argument, so `dist/*.whl dist/*.tar.gz` passes;
  - the workflow never runs at all, because build.sh is not in the path filters.

Both are covered below.
"""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "wheel-smoke.yml"
LINT = REPO / ".github" / "workflows" / "workflow-trigger-lint.yml"
STEP_NAME = "Publishing path uploads the wheel only"


def _workflow():
    return yaml.safe_load(WORKFLOW.read_text())


def _on_block(wf):
    # PyYAML parses a bare `on:` key as the boolean True.
    return wf.get("on", wf.get(True))


def _guard_source():
    """The Python between `python - <<'PY'` and the closing `PY`."""
    for step in _workflow()["jobs"]["wheel"]["steps"]:
        if step.get("name") == STEP_NAME:
            body = step["run"]
            break
    else:
        pytest.fail(f"workflow has no step named {STEP_NAME!r}")

    lines = body.splitlines()
    start = next(i for i, l in enumerate(lines) if l.strip().endswith("<<'PY'"))
    end = next(i for i, l in enumerate(lines[start + 1 :], start + 1) if l.strip() == "PY")
    return "\n".join(lines[start + 1 : end])


def _run_guard(tmp_path, build_sh_body):
    (tmp_path / "build.sh").write_text(build_sh_body)
    script = tmp_path / "_guard.py"
    script.write_text(_guard_source())
    return subprocess.run(
        [sys.executable, str(script)], cwd = tmp_path, capture_output = True, text = True
    )


PROLOGUE = "#!/bin/bash\nset -euo pipefail\npython -m build\n"


@pytest.mark.parametrize(
    "upload_line",
    [
        "python -m twine upload dist/*.whl",
        "twine upload dist/*.whl",
        # Flags that consume a value must not be mistaken for artifacts.
        "python -m twine upload -r pypi dist/*.whl",
        "python -m twine upload --repository-url https://upload.pypi.org/legacy/ dist/*.whl",
        "python -m twine upload --non-interactive dist/*.whl",
        "python -m twine upload dist/a-1-py3-none-any.whl dist/b-1-py3-none-any.whl",
        # -s is store_true, so the wheel after it is an artifact, not its value.
        "python -m twine upload -s dist/*.whl",
        # These two do consume a value, and the value is not an artifact.
        "python -m twine upload --sign-with gpg2 dist/*.whl",
        "python -m twine upload -i me@example.com dist/*.whl",
    ],
)
def test_wheel_only_uploads_pass(tmp_path, upload_line):
    r = _run_guard(tmp_path, PROLOGUE + upload_line + "\n")
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize(
    "upload_line",
    [
        # The regression #10419 exists to catch.
        "python -m twine upload dist/*",
        # The bypass: wheel first, sdist second. A first-argument-only check
        # passes this while PyPI receives the ~86MB sdist.
        "python -m twine upload dist/*.whl dist/*.tar.gz",
        "python -m twine upload dist/*.tar.gz dist/*.whl",
        "python -m twine upload -r pypi dist/*.whl dist/unsloth-1.0.tar.gz",
        "python -m twine upload dist/*.zip",
        # Signing on with the sdist listed first. Treating -s as value-taking
        # swallowed the tarball, leaving only the wheel visible, so the guard
        # passed while twine uploaded both.
        "python -m twine upload -s dist/*.tar.gz dist/*.whl",
    ],
)
def test_non_wheel_uploads_fail(tmp_path, upload_line):
    r = _run_guard(tmp_path, PROLOGUE + upload_line + "\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "non-wheel" in r.stdout


def test_missing_upload_line_fails(tmp_path):
    r = _run_guard(tmp_path, PROLOGUE)
    assert r.returncode == 1
    assert "no twine upload line" in r.stdout


@pytest.mark.parametrize(
    "no_op",
    [
        ": twine upload dist/*.whl",
        "true twine upload dist/*.whl",
        "echo twine upload dist/*.whl",
    ],
)
def test_a_shell_no_op_is_not_an_upload(tmp_path, no_op):
    """A line that names twine but never runs it must not satisfy the check.

    Unanchored, `: twine upload dist/*.whl` registered a wheel target, so a
    build.sh whose real upload had been deleted still reported PASS. That is
    precisely what the missing-upload check exists to catch.
    """
    r = _run_guard(tmp_path, PROLOGUE + no_op + "\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no twine upload line" in r.stdout


@pytest.mark.parametrize(
    "heredoc",
    [
        "cat <<'USAGE'\ntwine upload dist/*.whl\nUSAGE\n",
        "cat <<USAGE\ntwine upload dist/*.whl\nUSAGE\n",
        "cat <<-USAGE\n\ttwine upload dist/*.whl\n\tUSAGE\n",
    ],
)
def test_heredoc_text_is_not_an_upload(tmp_path, heredoc):
    """Usage text naming the command is text, not the release path."""
    r = _run_guard(tmp_path, PROLOGUE + heredoc)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no twine upload line" in r.stdout


def test_a_heredoc_does_not_hide_the_real_upload(tmp_path):
    """Skipping heredoc bodies must not skip the invocation that follows one."""
    body = PROLOGUE + "cat <<'USAGE'\ntwine upload dist/*.tar.gz\nUSAGE\n"
    body += "python -m twine upload dist/*.whl\n"
    r = _run_guard(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize(
    "invocation",
    [
        "twine upload dist/*.whl",
        "python -m twine upload dist/*.whl",
        "python3 -m twine upload dist/*.whl",
        "    python -m twine upload dist/*.whl",
    ],
)
def test_real_invocation_forms_are_still_recognised(tmp_path, invocation):
    """Anchoring must not stop the guard seeing how build.sh actually calls it."""
    r = _run_guard(tmp_path, PROLOGUE + invocation + "\n")
    assert r.returncode == 0, r.stdout + r.stderr


def test_commented_out_upload_is_not_an_artifact(tmp_path):
    """A comment mentioning the bad glob must not be read as a real upload."""
    body = PROLOGUE + "# never do: twine upload dist/*\npython -m twine upload dist/*.whl\n"
    r = _run_guard(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr


def test_real_build_sh_passes_the_guard(tmp_path):
    r = _run_guard(tmp_path, (REPO / "build.sh").read_text())
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("event", ["pull_request", "push"])
def test_build_sh_is_in_the_path_filters(event):
    """The guard reads build.sh, so build.sh must trigger the workflow.

    GitHub skips a `paths`-filtered workflow entirely when no changed file
    matches, so a PR touching only build.sh would otherwise never run this.
    """
    paths = _on_block(_workflow())[event]["paths"]
    assert "build.sh" in paths, f"{event} paths filter omits build.sh: {paths}"


# --------------------------------------------------------------------------
# The tests above only bite if something actually collects THIS module.
#
# `pyproject.toml` sets testpaths = ["tests/security"], and pytest uses that
# list "when no specific directories, files or test ids are given in the
# command line" -- so a bare `pytest` never reaches this file, and every
# invocation that would has to name it. Meanwhile the change these tests exist
# to reject (reintroducing the first-argument-only parser) edits
# `wheel-smoke.yml` and nothing else, and GitHub only runs a `paths`-filtered
# workflow when "at least one path matches a pattern in the paths filter".
#
# wheel-smoke.yml does trigger on its own YAML, but that run only executes the
# guard against the current, single-target build.sh: a broken parser still says
# PASS. So the regression is caught only by a job that both collects this
# module and starts on a workflow-only diff. workflow-trigger-lint.yml is the
# one job in the repo with no paths filter, which makes it the only candidate.
# tests/studio/test_workflow_guards_run_unfiltered.py enforces this rule for
# tests/studio; this module lives in tests/, so it asserts it for itself.
# --------------------------------------------------------------------------


def _lint_doc():
    return yaml.safe_load(LINT.read_text(encoding = "utf-8"))


def test_this_module_runs_in_the_unfiltered_guard_job():
    """Named explicitly, because testpaths means nothing collects it by accident."""
    runs = "\n".join(
        str(step.get("run", "")) for step in _lint_doc()["jobs"]["workflow-trigger-lint"]["steps"]
    )
    assert Path(__file__).name in runs, (
        f"workflow-trigger-lint does not name {Path(__file__).name}. It is the only job "
        f"with no paths filter, so on a PR that edits only wheel-smoke.yml -- exactly the "
        f"change these tests exist to reject -- nothing else collects this module, and the "
        f"regression merges green."
    )


def test_the_job_that_runs_this_module_has_no_paths_filter():
    """The premise. A filter here and this module stops seeing workflow-only PRs."""
    doc = _lint_doc()
    on = _on_block(doc)
    for trigger in ("pull_request", "push"):
        config = on.get(trigger)
        if not isinstance(config, dict):
            continue
        assert not config.get("paths") and not config.get("paths-ignore"), (
            f"workflow-trigger-lint now filters its {trigger} trigger on paths, so it no "
            f"longer runs on every workflow-only PR."
        )
