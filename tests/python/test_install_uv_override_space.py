"""Run the install.sh UV_OVERRIDE space-safety shell test (issue #6503) under
pytest, so the auto-discovered CPU test job executes it. The dedicated
`Shell installer tests` CI job runs a fixed script list that this is not part
of, so without this wrapper the regression would only be covered locally via
tests/run_all.sh.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SHELL_TEST = REPO_ROOT / "tests" / "sh" / "test_install_uv_override_space.sh"


@pytest.mark.skipif(sys.platform == "win32", reason = "POSIX shell installer test")
@pytest.mark.skipif(shutil.which("bash") is None, reason = "bash not available")
def test_install_uv_override_space_shell():
    assert SHELL_TEST.is_file(), f"missing shell test: {SHELL_TEST}"
    proc = subprocess.run(
        ["bash", str(SHELL_TEST)],
        capture_output = True,
        text = True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ALL PASSED" in proc.stdout, proc.stdout + proc.stderr
