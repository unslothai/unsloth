# SPDX-License-Identifier: AGPL-3.0-only
"""A missing optional dependency must skip one module, not kill the session.

`require_package` / `require_python_package` are called at module import time by
the four tests/saving/text_to_speech_models files. They used to `sys.exit(1)`
when a package was absent, and under pytest that lands during collection: pytest
turns SystemExit there into an INTERNALERROR and aborts the whole run. A host
without xcodec2 therefore executed zero tests across the entire suite.
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
    # pytest.skip raises Skipped, which is what allows the rest of the session
    # to carry on. Anything else here means the abort is back.
    with pytest.raises(pytest.skip.Exception):
        require_python_package(_ABSENT)


def test_an_installed_package_is_a_no_op():
    require_python_package("sys", import_name = "sys")


def test_the_standalone_script_path_still_exits():
    # These files are also meant to be runnable directly, so outside pytest the
    # historical `sys.exit(1)` must survive. Run it in a subprocess with pytest
    # genuinely absent from sys.modules rather than faking it.
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
