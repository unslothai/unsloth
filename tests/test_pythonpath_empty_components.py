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
and hands the result to every descendant process. Rebuilding that value with
`[p for p in current.split(os.pathsep) if p]` drops the empty components, and
an empty component is what

    export PYTHONPATH="$PYTHONPATH:/opt/mylib"

produces on the very common machine where PYTHONPATH was not previously set.
CPython reads it as the current working directory:

  * 3.11+ absolutises it in the interpreter itself. Modules/getpath.py at tag
    v3.13.12 lines 666-668 (identically at v3.11.14 / v3.12.12 lines 658-660):

        if use_environment and ENV_PYTHONPATH:
            for p in ENV_PYTHONPATH.split(DELIM):
                pythonpath.append(abspath(p))

    abspath("") is the cwd, so "" becomes a real directory on sys.path. Since
    v3.11.0 this file is used on Windows too: PC/getpathp.c was deleted by
    "bpo-45582: Port getpath[p].c to Python (GH-29041)".
  * 3.10 has no such loop. Modules/getpath.c v3.10.19 lines 1375-1379 copies
    $PYTHONPATH in verbatim (PC/getpathp.c lines 840-847 does the same on
    Windows) and Python/pathconfig.c lines 272-307 splits it without touching
    the components, so a literal "" reaches sys.path -- which the import
    system already resolves against the cwd -- and Lib/site.py
    removeduppaths() (lines 129-145, called from main() line 604) rewrites it
    to the absolute cwd. Same outcome, different layer.

A SET-BUT-EMPTY PYTHONPATH is the opposite case and must stay opposite:
CPython ignores it entirely (`if use_environment and ENV_PYTHONPATH` above,
and before that Python/initconfig.c config_get_env_dup, v3.13.12 lines
1352-1382, which NULLs the value when `var[0] == '\\0'`). Turning "" into a
lone "" component would ADD the cwd to every descendant, which is a new bug in
the other direction, so the rebuild keeps that one special case.
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

    The gate is genuinely conditional -- it returns None on a healthy
    torch/torchao pair -- so without this nothing below would execute a single
    line of the code under test. `find_spec("torchao")` is the one part not
    faked, so skip rather than silently pass when torchao is absent.
    """
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("no torchao here; the function returns before PYTHONPATH")
    monkeypatch.setattr(IF, "importlib_version",
                        lambda name: "0.18.0" if name == "torchao" else "0")
    monkeypatch.setattr(IF, "_torch_really_has", lambda F, name: False)
    # Keeps the generated sitecustomize inside tmp_path instead of the real
    # temp dir, so a test run leaves no staged directory behind.
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    # monkeypatch.setenv/delenv already restores PYTHONPATH at teardown, which
    # is why there is no autouse restore fixture here: the function writes
    # os.environ directly and monkeypatch recorded the pre-test value.
    if pythonpath is None:
        monkeypatch.delenv("PYTHONPATH", raising = False)
    else:
        monkeypatch.setenv("PYTHONPATH", pythonpath)

    directory = IF.propagate_torchao_fix_to_subprocesses()
    assert directory is not None, "gate did not fire; the test proves nothing"
    # Makes the exact-string assertions below sound.
    assert os.pathsep not in directory, directory
    return directory, os.environ.get("PYTHONPATH", "")


@pytest.mark.parametrize("before", [
    os.pathsep + "/opt/lib",                       # PYTHONPATH=$PYTHONPATH:/opt/lib, unset
    "/opt/lib" + os.pathsep,                       # PYTHONPATH=/opt/lib:$PYTHONPATH, unset
    "/opt/a" + os.pathsep + os.pathsep + "/opt/b", # interior empty
    os.pathsep,                                    # separator only
])
def test_empty_components_survive(monkeypatch, tmp_path, before):
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert after == directory + os.pathsep + before, after


@pytest.mark.parametrize("before", [None, ""])
def test_an_absent_or_empty_pythonpath_does_not_gain_the_cwd(
        monkeypatch, tmp_path, before):
    """The hazard in the other direction. CPython ignores a set-but-empty
    PYTHONPATH, so "" must not be split into a lone "" component."""
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert after == directory, after


def test_it_is_still_idempotent(monkeypatch, tmp_path):
    """Two calls must not stack the directory, and the second must not eat the
    empty components the first preserved."""
    before = os.pathsep + "/opt/lib"
    directory, after = _stage(monkeypatch, tmp_path, before)
    assert IF.propagate_torchao_fix_to_subprocesses() == directory
    assert os.environ["PYTHONPATH"] == after


# ---- the premise and the consequence, with real child processes -----------

def _probe_tree(tmp_path):
    """cwddir holds a module reachable ONLY through the cwd.

    The child is run as a script in scriptdir, so sys.path[0] is scriptdir and
    never the cwd. `only_in_cwd` is therefore importable if and only if some
    PYTHONPATH component resolves to the cwd.
    """
    cwddir = tmp_path / "cwddir"
    libdir = tmp_path / "libdir"
    scriptdir = tmp_path / "scriptdir"
    for d in (cwddir, libdir, scriptdir):
        d.mkdir(exist_ok = True)
    (cwddir / "only_in_cwd.py").write_text("MARKER = 1\n", encoding = "utf-8")
    (scriptdir / "probe.py").write_text(
        "import importlib.util as u\n"
        "print(u.find_spec('only_in_cwd') is not None)\n", encoding = "utf-8")
    return cwddir, libdir, scriptdir


def _cwd_is_importable(cwddir, scriptdir, pythonpath):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    out = subprocess.run([sys.executable, str(scriptdir / "probe.py")],
                         cwd = str(cwddir), env = env, capture_output = True,
                         text = True, timeout = 300)
    assert out.returncode == 0, out.stderr
    return out.stdout.strip() == "True"


def test_the_cwd_really_is_importable_from_an_empty_component(tmp_path):
    """The premise, against this interpreter, with both controls."""
    cwddir, libdir, scriptdir = _probe_tree(tmp_path)
    assert _cwd_is_importable(cwddir, scriptdir, None) is False
    assert _cwd_is_importable(cwddir, scriptdir, str(libdir)) is False
    assert _cwd_is_importable(
        cwddir, scriptdir, os.pathsep + str(libdir)) is True


def test_the_rewritten_pythonpath_still_reaches_the_cwd(monkeypatch, tmp_path):
    """The consequence: what a descendant process actually sees afterwards."""
    cwddir, libdir, scriptdir = _probe_tree(tmp_path)
    before = os.pathsep + str(libdir)
    assert _cwd_is_importable(cwddir, scriptdir, before) is True
    _directory, after = _stage(monkeypatch, tmp_path, before)
    assert _cwd_is_importable(cwddir, scriptdir, after) is True, after


def test_a_set_but_empty_pythonpath_still_reaches_nothing_new(
        monkeypatch, tmp_path):
    """And the mirror image: no cwd before, no cwd after."""
    cwddir, _libdir, scriptdir = _probe_tree(tmp_path)
    assert _cwd_is_importable(cwddir, scriptdir, "") is False
    _directory, after = _stage(monkeypatch, tmp_path, "")
    assert _cwd_is_importable(cwddir, scriptdir, after) is False, after


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
