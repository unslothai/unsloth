# SPDX-License-Identifier: AGPL-3.0-only
"""A missing optional dependency must skip one module, not kill the session.

`require_package` / `require_python_package` run at import time, so their old
`sys.exit(1)` aborted the whole pytest run instead of skipping four files.
"""

import subprocess
import sys
import textwrap

import pytest

from tests.utils.os_utils import require_python_package


_ABSENT = "unsloth_definitely_not_a_real_package_xyz"


def _repo_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parents[1]


def test_a_missing_package_skips_the_module_under_pytest():
    # Anything other than Skipped here means the session-wide abort is back.
    with pytest.raises(pytest.skip.Exception):
        require_python_package(_ABSENT)


def test_an_installed_package_is_a_no_op():
    require_python_package("sys", import_name = "sys")


def test_the_standalone_script_path_still_exits():
    # Direct runs must still exit(1), so use a subprocess with pytest genuinely
    # absent from sys.modules rather than faking it.
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_repo_root())!r})
        for name in list(sys.modules):
            if name == "pytest" or name.startswith("pytest."):
                del sys.modules[name]
        from tests.utils.os_utils import require_python_package
        require_python_package({_ABSENT!r})
        print("NO EXIT")
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output = True,
        text = True,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "NO EXIT" not in result.stdout
