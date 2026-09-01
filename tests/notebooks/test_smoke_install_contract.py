# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team.
"""Pins the two contracts the notebooks-ci smoke job kept breaking silently.

The job installs a Colab-shaped venv on a CPU runner and then runs one
notebook's install cell. It had never once reached the import check it exists
for: 872 matrix legs across 109 scheduled runs and 112 days, every one either
cancelled at the 25 minute cap or failed. Two independent causes, and each is
the kind that reads as green in review.

  * The runner pinned Python 3.12 while the committed Colab snapshot beside it
    had been refreshed to a 3.13 image. `audioop-lts` is a backport of the
    stdlib module 3.13 removed, so it carries Requires-Python >=3.13 and could
    never resolve on 3.12. That failed the bulk install in 8 seconds on every
    run and dropped the job into a 682-pin one-at-a-time fallback that spent
    the whole cap. Nothing tied the two files together, so the rotation moved
    one and not the other.

  * The workflow rebuilt the converted script's filename in shell rather than
    asking the converter, and the copy was wrong for every row of the matrix.

Both are cheap to assert and impossible to notice otherwise, because a leg that
exceeds `timeout-minutes` is scored `cancelled`, and `cancelled` outranks
`failure` in GitHub's run rollup.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[2]
WORKFLOW = REPO / ".github" / "workflows" / "notebooks-ci.yml"
MAPPING = REPO / "scripts" / "data" / "colab_to_cpu_pin.json"
FREEZE = REPO / "scripts" / "data" / "colab_pip_freeze.gpu.txt"

sys.path.insert(0, str(REPO / "scripts"))
from notebook_to_python import converted_filename  # noqa: E402

JOB = "smoke-install"


def _job() -> dict:
    doc = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))
    job = doc["jobs"].get(JOB)
    assert job, f"{JOB} is gone from {WORKFLOW.name}; this file checks nothing"
    return job


def _notebooks() -> list[str]:
    nbs = _job()["strategy"]["matrix"]["notebook"]
    assert nbs, "the smoke matrix is empty; this guard checks nothing"
    return nbs


def _mapping() -> dict:
    return json.loads(MAPPING.read_text(encoding = "utf-8"))


# --- the interpreter the snapshot was taken on ---------------------------------------


def test_the_snapshot_records_the_interpreter_it_was_captured_on():
    assert re.fullmatch(r"3\.\d+", _mapping().get("python_version", "")), (
        "colab_to_cpu_pin.json must record python_version, the interpreter the freeze "
        "beside it came from. Without it nothing connects a Colab rotation to the "
        "runner pin, which is how the job came to install a 3.13 environment on 3.12."
    )


def test_the_smoke_job_runs_the_interpreter_the_snapshot_names():
    want = _mapping()["python_version"]
    pins = [
        str((s.get("with") or {}).get("python-version"))
        for s in _job()["steps"]
        if "setup-python" in str(s.get("uses", ""))
    ]
    assert pins, f"{JOB} does not pin an interpreter at all"
    assert set(pins) == {want}, (
        f"{JOB} pins Python {pins} but the Colab snapshot was captured on {want}. A pin "
        f"carrying a Requires-Python floor above the runner cannot resolve, and one "
        f"such pin fails the whole bulk install."
    )


def test_the_freeze_resolves_against_the_interpreter_the_snapshot_names():
    """Anything with an explicit floor above the pinned interpreter is unresolvable.

    `audioop-lts` is the live example and the reason this file exists. It only exists
    for 3.13+, so its presence in the freeze is itself evidence of the interpreter the
    image was running.
    """
    want = _mapping()["python_version"]
    names = {
        m.group(1).lower()
        for line in FREEZE.read_text(encoding = "utf-8").splitlines()
        if (m := re.match(r"^([A-Za-z0-9._-]+)\s*==", line.strip()))
    }
    if "audioop-lts" in names:
        assert want == "3.13" or tuple(map(int, want.split("."))) >= (3, 13), (
            f"the freeze pins audioop-lts, which requires Python >= 3.13, but "
            f"python_version says {want}"
        )


# --- the converted script's name ------------------------------------------------------


@pytest.mark.parametrize("notebook", _notebooks())
def test_every_matrix_notebook_maps_to_one_converted_script(notebook):
    name = converted_filename(Path(notebook).name)
    assert name.endswith(".py") and not name.endswith("_.py"), (
        f"{notebook} converts to {name!r}. A trailing underscore is the signature of "
        f"rebuilding the name in shell, where basename's newline becomes one."
    )


def test_converted_names_do_not_collide_across_the_matrix():
    """The job takes 'the one .py in the output directory', so a collision hides a leg."""
    seen: dict[str, list[str]] = {}
    for nb in _notebooks():
        seen.setdefault(converted_filename(Path(nb).name), []).append(nb)
    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not clashes, f"these matrix notebooks convert to the same filename: {clashes}"


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("Gemma3_(4B)-Vision.ipynb", "Gemma3_4B_Vision.py"),
        ("Whisper.ipynb", "Whisper.py"),
        # A dot survives. The shell copy mapped it to `_`, so even ignoring the
        # trailing underscore it named a different file for these two.
        ("Llama3.1_(8B)-GRPO.ipynb", "Llama3.1_8B_GRPO.py"),
        ("gpt-oss-(20B)-Fine-tuning.ipynb", "gpt_oss_20B_Fine_tuning.py"),
    ],
)
def test_the_naming_rule_itself(filename, expected):
    assert converted_filename(filename) == expected


def _shell(job) -> str:
    """Every `run:` body in the job with comment lines removed.

    Comments are stripped because the steps quote the old broken pipeline to explain
    why it went, and a rule about what the shell DOES must not read prose as code.
    """
    lines = []
    for step in job["steps"]:
        for line in str(step.get("run", "")).splitlines():
            if not line.lstrip().startswith("#"):
                lines.append(line)
    return "\n".join(lines)


def test_the_workflow_asks_the_converter_instead_of_rebuilding_the_name():
    """Anti-regression: the second spelling of the rule must stay gone.

    Structural checks above cannot see a reintroduced `tr` pipeline, because they read
    the matrix rather than the step body.
    """
    body = _shell(_job())
    assert "tr -c '[:alnum:]_'" not in body, (
        "the smoke job is rebuilding the converted filename in shell again. Call "
        "scripts/notebook_to_python.py on the one notebook and take the file it wrote."
    )
    assert "notebook_to_python.py" in body, (
        "the smoke job should convert its own matrix notebook with the converter "
        "directly, so the name it looks for is the name that was written"
    )


# --- the install itself ---------------------------------------------------------------


def test_the_seed_install_refuses_source_builds():
    """Nine pins in this snapshot are sdist-only and need system libraries the runner
    lacks. Without --only-binary pip spends 20-90s per package on a doomed build."""
    body = _shell(_job())
    installs = [ln for ln in body.splitlines() if re.search(r"\bpip install\b", ln)]
    offenders = [
        ln.strip() for ln in installs if "--upgrade pip" not in ln and "--only-binary" not in ln
    ]
    assert not offenders, "these seed installs allow source builds:\n  " + "\n  ".join(offenders)


def test_the_known_unbuildable_pins_are_skipped():
    """Each of these failed a source build in the 2026-08-31 run, and any one of them
    is enough to fail a single bulk resolve for the whole set."""
    skip = set(_mapping()["skip"])
    # name -> the system dependency whose absence killed its build in that run.
    system_bound = {
        "cyipopt": "ipopt",
        "dbus-python": "dbus-1",
        "dlib": "cmake",
        "gdal": "gdal-config",
        "pycairo": "cairo",
        "pygobject": "girepository",
        "python-apt": "apt",
        "rpy2": "R_HOME",
    }
    missing = sorted(set(system_bound) - skip)
    assert not missing, (
        f"these pins cannot build on ubuntu-latest on any interpreter and only cost "
        f"build time, so they belong in the skip list: {missing}"
    )
