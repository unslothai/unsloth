# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An empty PYTHONPATH component is an import location, not padding.

`propagate_torchao_fix_to_subprocesses` prepends its directory to PYTHONPATH
for every descendant process, and rebuilding that value with
`[p for p in current.split(os.pathsep) if p]` would drop empty components.
`export PYTHONPATH="$PYTHONPATH:/opt/mylib"` leaves one on the very common
machine where PYTHONPATH was unset, and CPython reads it as the cwd: 3.11+
absolutises every component in Modules/getpath.py (abspath("") is the cwd),
and 3.10 puts the literal "" on sys.path, which site.removeduppaths() then
makes absolute. Same outcome, different layer.

A SET-BUT-EMPTY PYTHONPATH is the opposite case: CPython ignores it entirely,
so turning "" into a lone "" component would ADD the cwd to every descendant.
The rebuild keeps that special case.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth import import_fixes as IF  # noqa: E402


def _stage(monkeypatch, tmp_path, pythonpath):
    """Drive the real function with its gate forced open.

    The gate returns None on a healthy torch/torchao pair, so without this
    nothing below would run any of the code under test. `find_spec("torchao")`
    is the one part not faked, so skip rather than pass when it is absent.
    """
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("no torchao here; the function returns before PYTHONPATH")
    monkeypatch.setattr(
        IF, "importlib_version", lambda name: "0.18.0" if name == "torchao" else "0"
    )
    monkeypatch.setattr(IF, "_torch_really_has", lambda F, name: False)
    # Keep the generated sitecustomize in tmp_path, so no staged directory is left behind in the real temp dir.
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    # monkeypatch.setenv/delenv restores PYTHONPATH at teardown even though the function writes os.environ directly, so
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising = False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)

    directory = IF.propagate_torchao_fix_to_subprocesses()
    assert directory is not None, "gate did not fire; the test proves nothing"
    # Makes the exact-string assertions below sound.
    assert os.pathsep not in directory, directory
    return directory, os.environ.get("PYTHONPATH", "")


@pytest.mark.parametrize(
    "before",
    [
        os.pathsep
        + "/opt/lib",  # PYTHONPATH=$PYTHONPATH:/opt/lib, unset PYTHONPATH=/opt/lib:$PYTHONPATH, unset
        "/opt/lib" + os.pathsep,
        "/opt/a" + os.pathsep + os.pathsep + "/opt/b",
        os.pathsep,
    ],
)
def test_empty_components_survive(monkeypatch, tmp_path, before):
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert after == directory + os.pathsep + before, after


@pytest.mark.parametrize("before", [None, ""])
def test_an_absent_or_empty_pythonpath_does_not_gain_the_cwd(monkeypatch, tmp_path, before):
    """CPython ignores a set-but-empty PYTHONPATH, so "" must not be split
    into a lone "" component."""
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert after == directory, after


def test_it_is_still_idempotent(monkeypatch, tmp_path):
    """Two calls must not stack the directory, nor eat the empty components
    the first preserved."""
    before = os.pathsep + "/opt/lib"
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert IF.propagate_torchao_fix_to_subprocesses() == directory
    assert os.environ["PYTHONPATH"] == after


# ---- the premise and the consequence, with real child processes -----------
def _probe_tree(tmp_path):
    """cwddir holds a module reachable ONLY through the cwd.

    The child runs as a script in scriptdir, so sys.path[0] is scriptdir, never
    the cwd. `only_in_cwd` is importable iff a PYTHONPATH component is the cwd.
    """
    cwddir = tmp_path / "cwddir"
    libdir = tmp_path / "libdir"
    scriptdir = tmp_path / "scriptdir"
    for d in (cwddir, libdir, scriptdir):
        d.mkdir(exist_ok = True)
    (cwddir / "only_in_cwd.py").write_text("MARKER = 1\n", encoding = "utf-8")
    (scriptdir / "probe.py").write_text(
        "import importlib.util as u\nprint(u.find_spec('only_in_cwd') is not None)\n",
        encoding = "utf-8",
    )
    return cwddir, libdir, scriptdir


def _cwd_is_importable(cwddir, scriptdir, pythonpath):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    out = subprocess.run(
        [sys.executable, str(scriptdir / "probe.py")],
        cwd = str(cwddir),
        env = env,
        capture_output = True,
        text = True,
        timeout = 300,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "True"


def test_the_cwd_really_is_importable_from_an_empty_component(tmp_path):
    """The premise, against this interpreter, with both controls."""
    cwddir, libdir, scriptdir = _probe_tree(tmp_path)
    assert _cwd_is_importable(cwddir, scriptdir, None) is False
    assert _cwd_is_importable(cwddir, scriptdir, str(libdir)) is False
    assert _cwd_is_importable(cwddir, scriptdir, os.pathsep + str(libdir)) is True


def test_the_rewritten_pythonpath_still_reaches_the_cwd(monkeypatch, tmp_path):
    """The consequence: what a descendant process actually sees afterwards."""
    cwddir, libdir, scriptdir = _probe_tree(tmp_path)
    before = os.pathsep + str(libdir)
    assert _cwd_is_importable(cwddir, scriptdir, before) is True
    _directory, after = _stage(monkeypatch, tmp_path, before)
    assert _cwd_is_importable(cwddir, scriptdir, after) is True, after


def test_a_set_but_empty_pythonpath_still_reaches_nothing_new(monkeypatch, tmp_path):
    """And the mirror image: no cwd before, no cwd after."""
    cwddir, _libdir, scriptdir = _probe_tree(tmp_path)
    assert _cwd_is_importable(cwddir, scriptdir, "") is False
    _directory, after = _stage(monkeypatch, tmp_path, "")
    assert _cwd_is_importable(cwddir, scriptdir, after) is False, after


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
