"""The Backend CI ruff-unsloth install must outlast a minutes-long burst of 504s from GitHub releases.

The step is run as written, with ``pip`` and ``sleep`` replaced by fakes on ``PATH``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "studio-backend-ci.yml"
STEP = "Install the pinned ruff-unsloth (formatter fixed-point guard)"

pytestmark = pytest.mark.skipif(
    shutil.which("bash") is None or os.name == "nt", reason = "needs bash"
)


def _step_script():
    jobs = yaml.safe_load(WORKFLOW.read_text(encoding = "utf-8"))["jobs"]
    runs = [
        s["run"] for job in jobs.values() for s in job.get("steps", []) if s.get("name") == STEP
    ]
    assert len(runs) == 1, f"expected one {STEP!r} step, found {len(runs)}"
    return runs[0]


def _run(tmp_path, pip_failures):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    fakes = {
        # Stands in for the pin reader; the step only needs a non-empty requirements file.
        "python": "#!/bin/sh\necho 'ruff-unsloth @ https://example.invalid/ruff.whl'\n",
        "pip": (
            "#!/bin/sh\n"
            f'n=$(cat "{state}/pip" 2>/dev/null || echo 0); n=$((n + 1)); echo $n > "{state}/pip"\n'
            f'[ "$n" -gt {pip_failures} ]\n'
        ),
        "sleep": f'#!/bin/sh\necho "$1" >> "{state}/sleep"\n',
    }
    for name, body in fakes.items():
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    env = dict(
        os.environ, PATH = f"{bin_dir}{os.pathsep}{os.environ['PATH']}", RUNNER_TEMP = str(tmp_path)
    )
    proc = subprocess.run(
        ["bash", "-e", "-c", _step_script()], env = env, capture_output = True, text = True, timeout = 60
    )
    sleeps = (
        [int(x) for x in (state / "sleep").read_text().split()]
        if (state / "sleep").exists()
        else []
    )
    return proc, sleeps


def test_a_five_minute_burst_of_failures_still_installs(tmp_path):
    proc, sleeps = _run(tmp_path, pip_failures = 5)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sum(sleeps) >= 300, f"backed off only {sum(sleeps)}s across {sleeps}"


def test_a_lasting_outage_still_fails_the_job(tmp_path):
    proc, sleeps = _run(tmp_path, pip_failures = 100)
    assert proc.returncode != 0
    assert "ruff-unsloth install failed" in proc.stdout + proc.stderr
    assert sum(sleeps) < 600, f"waited {sum(sleeps)}s before giving up"


def test_a_healthy_download_does_not_wait(tmp_path):
    proc, sleeps = _run(tmp_path, pip_failures = 0)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert sleeps == []
