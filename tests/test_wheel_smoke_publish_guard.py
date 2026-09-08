# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Regression tests for the "Publishing path uploads the wheel only" guard.

The guard text is extracted from .github/workflows/wheel-smoke.yml rather than
copied, then run against synthetic build.sh files.
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
        "python -m twine upload -s dist/*.whl",
        # twine's credential env vars put an assignment in front of the command.
        "TWINE_USERNAME=__token__ python -m twine upload dist/*.whl",
        "TWINE_USERNAME=__token__ TWINE_PASSWORD=x twine upload dist/*.whl",
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
        "python -m twine upload dist/*",
        # Wheel first, sdist second: a first-argument-only check passes this.
        "python -m twine upload dist/*.whl dist/*.tar.gz",
        "python -m twine upload dist/*.tar.gz dist/*.whl",
        "python -m twine upload -r pypi dist/*.whl dist/unsloth-1.0.tar.gz",
        "python -m twine upload dist/*.zip",
        # -s is store_true, so the sdist after it is an artifact, not its value.
        "python -m twine upload -s dist/*.tar.gz dist/*.whl",
        "TWINE_REPOSITORY_URL=https://example.invalid python -m twine upload dist/*",
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
        # Stripping the assignment prefix must stop at the real command.
        "X=1 : twine upload dist/*.whl",
    ],
)
def test_a_shell_no_op_is_not_an_upload(tmp_path, no_op):
    """A line that names twine but never runs it must not satisfy the check."""
    r = _run_guard(tmp_path, PROLOGUE + no_op + "\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no twine upload line" in r.stdout


@pytest.mark.parametrize(
    "heredoc",
    [
        "cat <<'USAGE'\ntwine upload dist/*.whl\nUSAGE\n",
        "cat <<USAGE\ntwine upload dist/*.whl\nUSAGE\n",
        "cat <<-USAGE\n\ttwine upload dist/*.whl\n\tUSAGE\n",
        # The delimiter is a shell word, not an identifier.
        "cat <<'PUBLISH-USAGE'\ntwine upload dist/*.whl\nPUBLISH-USAGE\n",
        "cat <<PUBLISH-USAGE\ntwine upload dist/*.whl\nPUBLISH-USAGE\n",
        "cat <<'EOF.TXT'\ntwine upload dist/*.whl\nEOF.TXT\n",
        "cat <<'END OF HELP'\ntwine upload dist/*.whl\nEND OF HELP\n",
        "cat <<\\USAGE\ntwine upload dist/*.whl\nUSAGE\n",
    ],
)
def test_heredoc_text_is_not_an_upload(tmp_path, heredoc):
    """Usage text naming the command is text, not the release path."""
    r = _run_guard(tmp_path, PROLOGUE + heredoc)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "no twine upload line" in r.stdout


@pytest.mark.parametrize("delimiter", ["USAGE", "'PUBLISH-USAGE'"])
def test_a_heredoc_does_not_hide_the_real_upload(tmp_path, delimiter):
    """Skipping heredoc bodies must not skip the invocation that follows one."""
    word = delimiter.strip("'")
    body = PROLOGUE + f"cat <<{delimiter}\ntwine upload dist/*.tar.gz\n{word}\n"
    body += "python -m twine upload dist/*.whl\n"
    r = _run_guard(tmp_path, body)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("here_string", ['cat <<< "usage"', "cat <<<'publish-usage'"])
def test_a_here_string_does_not_open_a_heredoc(tmp_path, here_string):
    """`<<<` contains `<<`; reading it as a heredoc swallowed the real upload."""
    r = _run_guard(tmp_path, PROLOGUE + here_string + "\npython -m twine upload dist/*.whl\n")
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
    """GitHub skips the workflow entirely on a PR that touches only build.sh."""
    paths = _on_block(_workflow())[event]["paths"]
    assert "build.sh" in paths, f"{event} paths filter omits build.sh: {paths}"


# testpaths = ["tests/security"] means a bare `pytest` never collects this module,
# and a parser regression edits only wheel-smoke.yml. workflow-trigger-lint.yml is
# the only job with no paths filter, so it is the only one that can catch that PR.


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
        # Presence first: `continue` on a missing key let the trigger be deleted
        # outright and still pass.
        assert trigger in on, (
            f"workflow-trigger-lint no longer runs on {trigger}, so this module stops "
            f"being collected for that event."
        )
        config = on.get(trigger)
        # A bare `pull_request:` parses as None and filters nothing, which is fine.
        if not isinstance(config, dict):
            continue
        assert not config.get("paths") and not config.get("paths-ignore"), (
            f"workflow-trigger-lint now filters its {trigger} trigger on paths, so it no "
            f"longer runs on every workflow-only PR."
        )


@pytest.mark.parametrize(
    ("body", "rc"),
    [
        # << closes on an exact delimiter, so an indented copy stays body.
        ("cat <<'USAGE'\n    USAGE\ntwine upload dist/*.whl\nUSAGE\n", 1),
        # <<- strips leading tabs, so a tab-indented delimiter does close it.
        ("cat <<-USAGE\n\tUSAGE\nUSAGE\npython -m twine upload dist/*.whl\n", 0),
    ],
    ids = ["exact-delimiter-only", "dash-strips-tabs"],
)
def test_heredoc_terminators_follow_bash_rules(tmp_path, body, rc):
    assert _run_guard(tmp_path, PROLOGUE + body).returncode == rc


@pytest.mark.parametrize(
    "upload_line",
    [
        "python -m twine upload dist/*.whl >twine.log",
        "python -m twine upload dist/*.whl > twine.log",
        "python -m twine upload dist/*.whl 2> err.log",
        "python -m twine upload dist/*.whl 2>&1 | tee log",
        "python -m twine upload dist/*.whl >>twine.log",
    ],
)
def test_redirections_are_not_artifacts(tmp_path, upload_line):
    """`>twine.log` is shell syntax. Counting it failed a valid release line."""
    r = _run_guard(tmp_path, PROLOGUE + upload_line + "\n")
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_redirection_does_not_hide_a_non_wheel(tmp_path):
    """Dropping redirections must not drop the artifact check with them."""
    r = _run_guard(tmp_path, PROLOGUE + "python -m twine upload dist/*.tar.gz >log\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "non-wheel" in r.stdout


@pytest.mark.parametrize(
    "upload_line",
    [
        "python -m twine upload dist/*.whl && python -m twine upload dist/*.tar.gz",
        "python -m twine upload dist/*.whl ; twine upload dist/*.tar.gz",
        "python -m twine upload dist/*.whl || twine upload dist/*.tar.gz",
    ],
)
def test_a_second_chained_upload_is_still_inspected(tmp_path, upload_line):
    """Every command on the line counts, not just the first."""
    r = _run_guard(tmp_path, PROLOGUE + upload_line + "\n")
    assert r.returncode == 1, r.stdout + r.stderr
    assert "non-wheel" in r.stdout


@pytest.mark.parametrize(
    "upload_line",
    [
        "python -m twine upload dist/*.whl && echo done",
        "python -m twine upload dist/*.whl | tee upload.log",
    ],
)
def test_a_chained_non_upload_is_not_an_artifact(tmp_path, upload_line):
    """Segmenting must not turn the other side of an operator into artifacts."""
    r = _run_guard(tmp_path, PROLOGUE + upload_line + "\n")
    assert r.returncode == 0, r.stdout + r.stderr


def test_an_assignment_prefix_does_not_hide_a_second_upload(tmp_path):
    """The visible wheel-only line must not cover for an assignment-prefixed one."""
    body = PROLOGUE + "python -m twine upload dist/*.whl\n"
    body += "TWINE_REPOSITORY_URL=https://example.invalid twine upload dist/*.tar.gz\n"
    r = _run_guard(tmp_path, body)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "non-wheel" in r.stdout
