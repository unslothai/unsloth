# SPDX-License-Identifier: AGPL-3.0-only
"""A missing optional dependency must skip one module, not kill the session."""

import subprocess
import sys
import textwrap

import pytest

from tests.utils.os_utils import require_package, require_python_package


_ABSENT = "unsloth_definitely_not_a_real_package_xyz"
_ABSENT_SYSTEM = "unsloth-nonexistent-system-package"
_ABSENT_EXECUTABLE = "unsloth-nonexistent-executable"


def _repo_root():
    import pathlib
    return pathlib.Path(__file__).resolve().parents[1]


def test_a_missing_package_skips_the_module_under_pytest():
    with pytest.raises(pytest.skip.Exception):
        require_python_package(_ABSENT)


def test_a_missing_system_package_skips_the_module_too():
    with pytest.raises(pytest.skip.Exception):
        require_package(_ABSENT_SYSTEM, _ABSENT_EXECUTABLE)


def test_an_installed_package_is_a_no_op():
    require_python_package("sys", import_name = "sys")
    require_package("python", sys.executable)


@pytest.mark.parametrize(
    "helper, args",
    [
        ("require_package", f"{_ABSENT_SYSTEM!r}, {_ABSENT_EXECUTABLE!r}"),
        ("require_python_package", f"{_ABSENT!r}"),
    ],
)
def test_the_call_at_module_scope_does_not_kill_the_session(tmp_path, helper, args):
    """The real regression: the helper at import time, in its own pytest session."""
    module = tmp_path / f"test_absent_{helper}.py"
    module.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_repo_root())!r})
        from tests.utils.os_utils import {helper}
        {helper}({args})

        def test_unreachable():
            raise AssertionError("the module-level skip should have stopped this")
    """), encoding = "utf-8")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(module), "-q", "-p", "no:cacheprovider"],
        capture_output = True, text = True, cwd = str(tmp_path),
    )
    output = result.stdout + result.stderr
    assert "INTERNALERROR" not in output, output
    assert "SystemExit" not in output, output
    assert "skipped" in output, output
    # 5 is "no tests ran", which a module-level skip legitimately produces.
    assert result.returncode in (0, 5), f"exit code {result.returncode}\n{output}"


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
