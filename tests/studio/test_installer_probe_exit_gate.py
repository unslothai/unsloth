# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A probe that exits unsuccessfully must not have its output believed.

`python -I` still runs the venv's own site-packages hooks, so a printing .pth or
sitecustomize can emit a POSTVER= line of its own. The last-sentinel extraction
covers the case where the probe survives to print the authoritative line after it;
when the interpreter dies first, the hook's line is the last one standing, and the
only correct read is no answer at all. This drives the extracted
_bounded_pkg_probe through both exits and holds it to that.
"""

import re
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason = "exercises the bash installer")

REPO = Path(__file__).resolve().parents[2]
SETUP_SH = REPO / "studio" / "setup.sh"

FUNC = re.compile(r"^_bounded_pkg_probe\(\) \{\n.*?\n\}$", re.S | re.M)


def _run_probe(tmp_path: Path, fake_python: str) -> str:
    m = FUNC.search(SETUP_SH.read_text(encoding = "utf-8"))
    assert m, "_bounded_pkg_probe is gone from setup.sh"
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents = True)
    py = venv_bin / "python"
    py.write_text("#!/usr/bin/env bash\n" + fake_python, encoding = "utf-8")
    py.chmod(py.stat().st_mode | stat.S_IXUSR)
    script = tmp_path / "probe.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f'VENV_DIR="{tmp_path / "venv"}"\n'
        "_PKG_PROBE_PY='pass'\n"
        "_PKG_NAME=unsloth\n" + m.group(0) + "\n_bounded_pkg_probe\n",
        encoding = "utf-8",
    )
    result = subprocess.run(
        ["bash", str(script)], capture_output = True, text = True, timeout = 60, check = True
    )
    return result.stdout


def test_nonzero_exit_discards_the_hook_line(tmp_path: Path) -> None:
    out = _run_probe(tmp_path, "echo POSTVER=9.9.9\nexit 1\n")
    assert out.strip() == "", (
        "the probe died before its own sentinel, so the POSTVER= line it left "
        "behind belongs to a site hook and must not be read as an answer"
    )


def test_clean_exit_yields_the_last_sentinel(tmp_path: Path) -> None:
    out = _run_probe(tmp_path, "echo POSTVER=9.9.9\necho POSTVER=1.2.3\nexit 0\n")
    assert out.strip() == "1.2.3"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
