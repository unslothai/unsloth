# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""A missing optional dependency must skip the test module, not the session.

`tests/utils/os_utils.py` exits the process when a package is absent. The TTS
test modules call those helpers at import time, so under pytest the SystemExit
escaped collection and took the whole run down with INTERNALERROR / exit code 3
instead of skipping one file. Standalone script callers keep the exit.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.utils.os_utils import require_package, require_python_package

ABSENT_PACKAGE = "unsloth-nonexistent-system-package"
ABSENT_EXECUTABLE = "unsloth-nonexistent-executable"
ABSENT_MODULE = "unsloth_nonexistent_python_package"


def test_require_package_skips_rather_than_exiting():
    with pytest.raises(pytest.skip.Exception):
        require_package(ABSENT_PACKAGE, ABSENT_EXECUTABLE)


def test_require_python_package_skips_rather_than_exiting():
    with pytest.raises(pytest.skip.Exception):
        require_python_package(ABSENT_MODULE)


def test_present_package_still_returns_normally():
    require_python_package("pytest")
    require_package("python", sys.executable)


@pytest.mark.parametrize(
    "helper, args",
    [
        ("require_package", f"{ABSENT_PACKAGE!r}, {ABSENT_EXECUTABLE!r}"),
        ("require_python_package", f"{ABSENT_MODULE!r}"),
    ],
)
def test_missing_dep_at_import_time_does_not_kill_the_session(tmp_path, helper, args):
    """The real regression: the helper called at module scope, in its own session."""
    module = tmp_path / f"test_absent_{helper}.py"
    module.write_text(
        f"import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"from tests.utils.os_utils import {helper}\n"
        f"{helper}({args})\n"
        f"\n"
        f"def test_unreachable():\n"
        f"    raise AssertionError('module-level skip should have stopped this')\n",
        encoding = "utf-8",
    )

    process = subprocess.run(
        [sys.executable, "-m", "pytest", str(module), "-q", "-p", "no:cacheprovider"],
        capture_output = True,
        text = True,
        cwd = str(tmp_path),
    )
    output = process.stdout + process.stderr

    assert "INTERNALERROR" not in output, output
    assert "SystemExit" not in output, output
    assert "skipped" in output, output
    # 5 is "no tests ran", which a module-level skip legitimately produces
    assert process.returncode in (0, 5), f"exit code {process.returncode}\n{output}"
