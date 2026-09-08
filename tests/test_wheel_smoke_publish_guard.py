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
