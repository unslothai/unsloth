# SPDX-License-Identifier: AGPL-3.0-only
"""A missing optional dependency must skip one module, not kill the session."""

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
    with pytest.raises(pytest.skip.Exception):
        require_python_package(_ABSENT)


def test_an_installed_package_is_a_no_op():
    require_python_package("sys", import_name = "sys")


def test_the_standalone_script_path_still_exits():
    # A subprocess, so pytest is genuinely absent from sys.modules, not faked.
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
