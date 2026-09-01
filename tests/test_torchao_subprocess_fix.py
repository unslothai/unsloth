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

"""The torchao fix has to reach processes unsloth does not launch.

`fix_torchao_torch_symbol_skew` patches `torch.nn.functional` in the current
interpreter, but vLLM inspects model architectures in a SEPARATE process that
imports torch and torchao itself, so `fast_inference = True` still dies with
the same ImportError, shown only as "Model architectures [...] failed to be
inspected". Observed on Colab AFTER the in-process fix reported success.

`sitecustomize` is the hook that reaches such a process: `site` imports it at
interpreter startup off PYTHONPATH, which subprocesses inherit. A `.pth` would
work too, but only inside a real site directory, which a library should not be
writing into.

The hazard dominating these tests is shadowing: `sitecustomize` is a single
global name that other things legitimately install (this machine has one at
/etc/unslothai/python/), so replacing it would silently disable them.
"""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unsloth import import_fixes as IF  # noqa: E402


def test_it_is_valid_python():
    """A syntax error breaks every subprocess on the machine, which is far
    worse than the bug being fixed."""
    compile(IF._subprocess_sitecustomize_source(), "<sitecustomize>", "exec")


def test_it_chains_instead_of_shadowing():
    src = IF._subprocess_sitecustomize_source()
    assert "_chain_to_the_real_sitecustomize" in src
    assert 'PathFinder.find_spec("sitecustomize"' in src


def test_the_chain_runs_before_the_fix():
    src = IF._subprocess_sitecustomize_source()
    assert src.index("_chain_to_the_real_sitecustomize()") < src.index("sys.meta_path.insert")


def test_it_never_raises_at_interpreter_startup():
    src = IF._subprocess_sitecustomize_source()
    tail = src[src.index("_chain_to_the_real_sitecustomize()") :]
    assert "except Exception:" in tail


def test_it_is_version_gated():
    src = IF._subprocess_sitecustomize_source()
    assert "(0, 18)" in src, "torchao < 0.18 guards its own import"


def test_it_does_not_overwrite_existing_torch_symbols():
    assert "if hasattr(F, name):" in IF._subprocess_sitecustomize_source()


def test_the_child_never_imports_unsloth():
    """Importing unsloth at the start of every subprocess would pay the full
    import cost each time and could recurse back through
    propagate_torchao_fix_to_subprocesses, so the logic is inlined."""
    src = IF._subprocess_sitecustomize_source()
    assert "import unsloth" not in src
    assert "from unsloth" not in src


def test_it_only_imports_the_stdlib_and_torch():
    """Anything heavier makes interpreter startup slower for every process."""
    import ast

    tree = ast.parse(IF._subprocess_sitecustomize_source())
    allowed = {
        "os",
        "sys",
        "importlib",
        "importlib.util",
        "importlib.machinery",
        "importlib.metadata",
        "torch",
        "torch.nn.functional",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in allowed, f"unexpected import: {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in allowed, f"unexpected import: {node.module}"


@pytest.fixture(autouse = True)
def _restore_pythonpath():
    before = os.environ.get("PYTHONPATH")
    yield
    if before is None:
        os.environ.pop("PYTHONPATH", None)
    else:
        os.environ["PYTHONPATH"] = before


def _torchao_is_broken_here() -> bool:
    try:
        import importlib.metadata as md
        from packaging.version import Version
        if Version(md.version("torchao")) < Version("0.18.0"):
            return False
    except Exception:
        return False
    import torch.nn.functional as F

    # `_torch_really_has`, not `hasattr`:
    return not all(IF._torch_really_has(F, n) for n in IF._TORCHAO_TORCH_SYMBOLS)


def test_it_is_a_noop_on_a_healthy_environment():
    """Nothing is written and PYTHONPATH is untouched unless the fix is
    actually needed."""
    if _torchao_is_broken_here():
        pytest.skip("this environment genuinely needs the fix")
    before = os.environ.get("PYTHONPATH")
    assert IF.propagate_torchao_fix_to_subprocesses() is None
    assert os.environ.get("PYTHONPATH") == before


def test_it_uses_the_platform_path_separator():
    """Windows uses ';'. A hardcoded ':' would corrupt PYTHONPATH there."""
    import inspect

    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    assert "os.pathsep" in src
    assert 'split(":")' not in src


def test_it_writes_atomically():
    """A truncated sitecustomize read by a concurrent subprocess is a
    SyntaxError at startup."""
    import inspect

    src = inspect.getsource(IF._write_hook_atomically)
    assert "os.replace" in src


def test_the_temporary_file_cannot_be_pre_empted():
    """A predictable temporary name can be pre-created as a symlink by anyone
    who could write into the directory before it was tightened."""
    import inspect

    src = inspect.getsource(IF._write_hook_atomically)
    assert "O_EXCL" in src
    assert "O_NOFOLLOW" in src
    assert "os.urandom" in src
    assert "getpid" not in src


def test_it_is_idempotent_on_pythonpath():
    import inspect
    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    assert "if directory not in parts:" in src


def test_the_directory_is_private_to_this_user():
    """The temp dir is shared and everything on PYTHONPATH runs in every
    subprocess, so a fixed name there is code execution for whoever creates it
    first."""
    directory = IF._subprocess_fix_directory()
    if hasattr(os, "getuid"):
        assert directory.endswith("-%d" % os.getuid()), directory
        info = os.lstat(directory)
        assert info.st_uid == os.getuid()
        assert oct(info.st_mode & 0o777) == oct(0o700), oct(info.st_mode)


def test_it_refuses_a_directory_owned_by_someone_else(monkeypatch, tmp_path):
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership only")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    hostile = tmp_path / ("unsloth_subprocess_import_fix-%d" % os.getuid())
    hostile.mkdir()

    real_lstat = os.lstat

    class _NotOurs:
        st_mode = real_lstat(str(hostile)).st_mode
        st_uid = os.getuid() + 1

    monkeypatch.setattr(
        os, "lstat", lambda p: _NotOurs() if str(p) == str(hostile) else real_lstat(p)
    )
    with pytest.raises(RuntimeError, match = "owned by another user"):
        IF._subprocess_fix_directory()


@pytest.mark.parametrize("mode", [0o777, 0o770, 0o707, 0o702, 0o720])
def test_an_existing_loose_directory_is_tightened(monkeypatch, tmp_path, mode):
    """`os.makedirs(mode = 0o700, exist_ok = True)` does NOT re-apply the mode
    to an existing directory, and one left group- or world-writable is code
    execution in every subprocess started after `import unsloth`."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX permissions only")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    loose = tmp_path / ("unsloth_subprocess_import_fix-%d" % os.getuid())
    loose.mkdir()
    os.chmod(loose, mode)
    assert os.lstat(loose).st_mode & 0o022, "fixture did not take"

    directory = IF._subprocess_fix_directory()

    assert directory == str(loose)
    after = os.lstat(directory).st_mode & 0o777
    assert not after & 0o022, "still writable by group/other: %04o" % after
    assert oct(after) == oct(0o700), oct(after)


def test_a_directory_that_cannot_be_tightened_is_refused(monkeypatch, tmp_path):
    """Some network and FUSE mounts accept chmod and change nothing; handing
    back a world-writable PYTHONPATH entry anyway is the whole bug."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX permissions only")
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    loose = tmp_path / ("unsloth_subprocess_import_fix-%d" % os.getuid())
    loose.mkdir()
    os.chmod(loose, 0o777)
    monkeypatch.setattr(os, "chmod", lambda *a, **k: None)

    with pytest.raises(RuntimeError, match = "group- or world-writable"):
        IF._subprocess_fix_directory()


def test_the_refusal_does_not_propagate_as_a_crash(monkeypatch):
    """A hostile directory must degrade to "no subprocess fix", not kill the
    import."""
    import inspect

    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    i = src.index("_subprocess_fix_directory()")
    assert "try:" in src[:i]
    assert "except Exception as exception:" in src[i:]


@pytest.fixture
def staged(tmp_path):
    """The generated sitecustomize on a PYTHONPATH dir, plus a fake torch and
    a fake torchao 0.18 that reproduces the real import error."""
    site = tmp_path / "hook"
    site.mkdir()
    (site / "sitecustomize.py").write_text(IF._subprocess_sitecustomize_source(), encoding = "utf-8")

    fake = tmp_path / "fake"
    (fake / "torch" / "nn").mkdir(parents = True)
    (fake / "torch" / "__init__.py").write_text(
        "from . import nn\n__version__='2.9.0'\n", encoding = "utf-8"
    )
    (fake / "torch" / "nn" / "__init__.py").write_text(
        "from . import functional\n", encoding = "utf-8"
    )
    (fake / "torch" / "nn" / "functional.py").write_text(
        "def linear(*a, **k):\n    return None\n", encoding = "utf-8"
    )
    (fake / "torchao").mkdir()
    (fake / "torchao" / "__init__.py").write_text(
        "from torch.nn.functional import ScalingType, scaled_grouped_mm\n__version__='0.18.0'\n",
        encoding = "utf-8",
    )
    d = fake / "torchao-0.18.0.dist-info"
    d.mkdir()
    (d / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torchao\nVersion: 0.18.0\n", encoding = "utf-8"
    )
    return site, fake


@pytest.fixture(scope = "session")
def bare_interpreter(tmp_path_factory):
    """An interpreter whose site-packages is empty.

    PYTHONPATH can shadow a module but cannot un-install one: the child still
    finds the real torchao's dist-info, which is what the hook reads. On a
    machine that has torchao, "absent" is only expressible as a bare venv.
    """
    venv = tmp_path_factory.mktemp("bare") / "venv"
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", "--without-pip", str(venv)],
            check = True,
            capture_output = True,
            timeout = 300,
        )
    except Exception as exception:
        pytest.skip(f"cannot build a venv here ({exception})")
    exe = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not exe.is_file():
        pytest.skip("venv produced no interpreter")
    return str(exe)


def _child(
    code: str,
    path_entries,
    timeout = 300,
    executable = None,
):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in path_entries)
    return subprocess.run(
        [executable or sys.executable, "-c", textwrap.dedent(code)],
        capture_output = True,
        text = True,
        timeout = timeout,
        env = env,
    )


def test_the_broken_import_really_fails_without_the_hook(staged):
    """Guards the premise; without this the test below proves nothing."""
    _site, fake = staged
    p = _child("import torchao", [fake])
    assert p.returncode != 0
    assert "cannot import name 'ScalingType'" in p.stderr, p.stderr


def test_the_hook_fixes_a_process_we_never_launched(staged):
    site, fake = staged
    p = _child("import torchao; print('OK', torchao.__version__)", [site, fake])
    assert "OK 0.18.0" in p.stdout, p.stdout + p.stderr


def test_it_does_not_import_torch_until_torchao_is(staged):
    """Most Python descendants never use torchao, and importing torch there
    costs seconds of startup and torch's memory."""
    site, fake = staged
    p = _child("import sys; print('TORCH', 'torch' in sys.modules)", [site, fake])
    assert "TORCH False" in p.stdout, p.stdout + p.stderr
    p = _child("import torchao, sys; print('TORCH', 'torch' in sys.modules)", [site, fake])
    assert "TORCH True" in p.stdout, p.stdout + p.stderr


def test_it_also_reaches_a_grandchild(staged):
    """vLLM's inspector is not necessarily a direct child."""
    site, fake = staged
    p = _child(
        "import subprocess, sys;"
        "r = subprocess.run([sys.executable, '-c',"
        "'import torchao; print(\"GRANDCHILD OK\")'],"
        "capture_output=True, text=True);"
        "print(r.stdout, r.stderr)",
        [site, fake],
    )
    assert "GRANDCHILD OK" in p.stdout, p.stdout + p.stderr


def test_the_placeholder_still_refuses_to_be_used(staged):
    site, fake = staged
    p = _child(
        """
        import torchao
        from torch.nn.functional import ScalingType
        try:
            ScalingType.DYNAMIC
            print("NO ERROR")
        except RuntimeError as e:
            print("RAISED", "does not exist in this torch" in str(e))
    """,
        [site, fake],
    )
    assert "RAISED True" in p.stdout, p.stdout + p.stderr


def test_an_existing_sitecustomize_still_runs(staged, tmp_path):
    """The shadowing hazard is real: this machine already has a sitecustomize
    at /etc/unslothai/python/, and ours must chain to it."""
    site, fake = staged
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("print('OTHER SITECUSTOMIZE RAN')\n", encoding = "utf-8")
    p = _child("import torchao; print('OK')", [site, other, fake])
    assert "OTHER SITECUSTOMIZE RAN" in p.stdout, p.stdout + p.stderr
    assert "OK" in p.stdout


def test_a_package_form_existing_sitecustomize_still_runs(staged, tmp_path):
    """CPython imports the package form exactly like the file form, so probing
    for a `sitecustomize.py` would silently drop it in every subprocess."""
    site, fake = staged
    other = tmp_path / "other"
    (other / "sitecustomize").mkdir(parents = True)
    (other / "sitecustomize" / "__init__.py").write_text(
        "from . import extra\nprint('PACKAGE SITECUSTOMIZE RAN', extra.NAME)\n",
        encoding = "utf-8",
    )
    (other / "sitecustomize" / "extra.py").write_text("NAME = 'submodule'\n", encoding = "utf-8")
    p = _child("import torchao; print('OK')", [site, other, fake])
    assert "PACKAGE SITECUSTOMIZE RAN submodule" in p.stdout, p.stdout + p.stderr
    assert "OK" in p.stdout


def test_a_pyc_only_existing_sitecustomize_still_runs(staged, tmp_path):
    """Same argument for a shipped `sitecustomize.pyc` with no source."""
    import py_compile

    site, fake = staged
    source = tmp_path / "src"
    source.mkdir()
    written = source / "sitecustomize.py"
    written.write_text("print('PYC SITECUSTOMIZE RAN')\n", encoding = "utf-8")
    other = tmp_path / "other_pyc"
    other.mkdir()
    py_compile.compile(str(written), cfile = str(other / "sitecustomize.pyc"), doraise = True)
    p = _child("import torchao; print('OK')", [site, other, fake])
    assert "PYC SITECUSTOMIZE RAN" in p.stdout, p.stdout + p.stderr
    assert "OK" in p.stdout


def test_a_broken_package_form_sitecustomize_does_not_kill_the_process(staged, tmp_path):
    site, fake = staged
    other = tmp_path / "other"
    (other / "sitecustomize").mkdir(parents = True)
    (other / "sitecustomize" / "__init__.py").write_text(
        "raise RuntimeError('boom')\n", encoding = "utf-8"
    )
    p = _child("import torchao; print('STILL OK')", [site, other, fake])
    assert "STILL OK" in p.stdout, p.stdout + p.stderr


def test_a_broken_existing_sitecustomize_does_not_kill_the_process(staged, tmp_path):
    """Chaining must not turn somebody else's bug into a startup crash."""
    site, fake = staged
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("raise RuntimeError('boom')\n", encoding = "utf-8")
    p = _child("import torchao; print('STILL OK')", [site, other, fake])
    assert "STILL OK" in p.stdout, p.stdout + p.stderr


def test_it_does_nothing_when_torchao_is_absent(staged, tmp_path, bare_interpreter):
    """No torchao, no patching: torch.nn.functional must be untouched."""
    site, _fake = staged
    bare = tmp_path / "bare"
    (bare / "torch" / "nn").mkdir(parents = True)
    (bare / "torch" / "__init__.py").write_text(
        "from . import nn\n__version__='2.9.0'\n", encoding = "utf-8"
    )
    (bare / "torch" / "nn" / "__init__.py").write_text(
        "from . import functional\n", encoding = "utf-8"
    )
    (bare / "torch" / "nn" / "functional.py").write_text(
        "def linear(*a, **k):\n    return None\n", encoding = "utf-8"
    )
    p = _child(
        """
        import torch.nn.functional as F
        print("PATCHED", hasattr(F, "ScalingType"))
    """,
        [site, bare],
        executable = bare_interpreter,
    )
    assert "PATCHED False" in p.stdout, p.stdout + p.stderr


def test_the_hook_does_not_disturb_ordinary_startup(staged):
    site, fake = staged
    p = _child("print('HELLO')", [site, fake])
    assert p.returncode == 0
    assert "HELLO" in p.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ---- the in-process fix must not disable this one -------------------------
def test_the_in_process_fix_does_not_disable_the_subprocess_fix(monkeypatch, tmp_path):
    """_gpu_init.py runs fix_torchao_torch_symbol_skew() immediately before
    this one, so a gate asking only `hasattr` would read its placeholders as a
    healthy torch and stage nothing, in exactly the environments vLLM's
    inspector child needs it."""
    import torch.nn.functional as F

    if all(IF._torch_really_has(F, n) for n in IF._TORCHAO_TORCH_SYMBOLS):
        pytest.skip("this torch provides every symbol; nothing to place")

    # conftest.py imports unsloth, so on an affected environment the placeholders are already installed and the call
    for name in IF._TORCHAO_TORCH_SYMBOLS:
        if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
            delattr(F, name)

    monkeypatch.setattr(
        IF, "importlib_version", lambda name: "0.18.0" if name == "torchao" else "0"
    )
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("PYTHONPATH", raising = False)
    try:
        assert IF.fix_torchao_torch_symbol_skew() is True
        directory = IF.propagate_torchao_fix_to_subprocesses()
        assert (
            directory is not None
        ), "staged nothing: the in-process placeholders defeated the gate"
        assert os.path.isfile(os.path.join(directory, "sitecustomize.py"))
        assert directory in os.environ["PYTHONPATH"].split(os.pathsep)
    finally:
        for name in IF._TORCHAO_TORCH_SYMBOLS:
            if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
                delattr(F, name)


def test_a_placeholder_does_not_count_as_a_real_torch_symbol():
    """The distinction the gate above turns on."""
    import torch.nn.functional as F

    placeholder = IF._make_torch_symbol_placeholder("ScalingType", "detail")
    assert IF._torch_really_has(F, "scaled_dot_product_attention") is True
    assert (
        IF._torch_really_has(type("_F", (), {"ScalingType": placeholder}), "ScalingType") is False
    )
    assert IF._torch_really_has(type("_F", (), {}), "ScalingType") is False


# a hook file planted before the directory was tightened ---------------
def _plant(directory, kind, source):
    """A `sitecustomize.py` as it could survive a once-writable directory."""
    target = directory / "sitecustomize.py"
    if kind == "symlink":
        elsewhere = directory.parent / "planted_elsewhere.py"
        elsewhere.write_text(source, encoding = "utf-8")
        os.symlink(elsewhere, target)
        return elsewhere
    target.write_text(source, encoding = "utf-8")
    if kind == "group_writable":
        os.chmod(target, 0o666)
    return target


@pytest.mark.parametrize("kind", ["symlink", "group_writable"])
def test_a_planted_hook_is_not_trusted(tmp_path, kind):
    """Contents equal to ours are what an attacker arranges to skip the
    rewrite; only a private regular file of ours makes them evidence."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership/permissions only")
    directory = tmp_path / "dir"
    directory.mkdir()
    _plant(directory, kind, IF._subprocess_sitecustomize_source())
    assert IF._existing_hook_is_trustworthy(str(directory / "sitecustomize.py")) is False


def test_a_foreign_owned_hook_is_not_trusted(tmp_path, monkeypatch):
    """A file created while the directory was world-writable stays theirs to
    rewrite after the directory is tightened."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership only")
    target = tmp_path / "sitecustomize.py"
    target.write_text("x = 1\n", encoding = "utf-8")
    real_lstat = os.lstat
    theirs = os.stat_result(
        tuple(real_lstat(str(target)))[:4] + (os.getuid() + 1,) + tuple(real_lstat(str(target)))[5:]
    )
    monkeypatch.setattr(os, "lstat", lambda p: theirs if str(p) == str(target) else real_lstat(p))
    assert IF._existing_hook_is_trustworthy(str(target)) is False


def test_our_own_hook_file_is_trusted(tmp_path):
    """The fast path must survive, or concurrent runs fight over the file."""
    target = tmp_path / "sitecustomize.py"
    target.write_text(IF._subprocess_sitecustomize_source(), encoding = "utf-8")
    os.chmod(target, 0o600)
    assert IF._existing_hook_is_trustworthy(str(target)) is True
    assert IF._existing_hook_is_trustworthy(str(tmp_path / "absent.py")) is True


@pytest.mark.parametrize("kind", ["symlink", "group_writable"])
def test_a_planted_hook_is_replaced_even_when_it_matches(monkeypatch, tmp_path, kind):
    """End to end: a once-world-writable directory already holds a hook whose
    contents match ours. Tightening the directory does not revoke write access
    to that file, so leaving it is code execution in every subprocess."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX ownership/permissions only")
    import torch.nn.functional as F

    for name in IF._TORCHAO_TORCH_SYMBOLS:
        if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
            delattr(F, name)
    if all(IF._torch_really_has(F, n) for n in IF._TORCHAO_TORCH_SYMBOLS):
        pytest.skip("this torch provides every symbol; nothing to stage")

    monkeypatch.setattr(
        IF, "importlib_version", lambda name: "0.18.0" if name == "torchao" else "0"
    )
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("PYTHONPATH", raising = False)

    loose = tmp_path / ("unsloth_subprocess_import_fix-%d" % os.getuid())
    loose.mkdir()
    os.chmod(loose, 0o777)
    planted = _plant(loose, kind, IF._subprocess_sitecustomize_source())

    try:
        directory = IF.propagate_torchao_fix_to_subprocesses()
        assert directory == str(loose)
        target = loose / "sitecustomize.py"
        info = os.lstat(target)
        import stat as _stat

        assert _stat.S_ISREG(info.st_mode), "still a symlink to a file we do not own"
        assert not _stat.S_IMODE(info.st_mode) & 0o022, oct(info.st_mode)
        assert target.read_text(encoding = "utf-8") == IF._subprocess_sitecustomize_source()
        if kind == "symlink":
            # Rewriting the planted file must no longer reach any child.
            planted.write_text("raise SystemExit('hijacked')\n", encoding = "utf-8")
            assert target.read_text(encoding = "utf-8") != planted.read_text(encoding = "utf-8")
    finally:
        for name in IF._TORCHAO_TORCH_SYMBOLS:
            if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
                delattr(F, name)


def test_a_pre_created_temporary_symlink_is_not_followed(monkeypatch, tmp_path):
    """The other half of a once-world-writable directory: the temporary file
    the hook is staged through. A predictable name lets someone leave a
    symlink behind, and os.replace would install that link as the hook."""
    if not hasattr(os, "getuid"):
        pytest.skip("POSIX symlinks only")
    import stat as _stat

    import torch.nn.functional as F

    for name in IF._TORCHAO_TORCH_SYMBOLS:
        if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
            delattr(F, name)
    if all(IF._torch_really_has(F, n) for n in IF._TORCHAO_TORCH_SYMBOLS):
        pytest.skip("this torch provides every symbol; nothing to stage")

    monkeypatch.setattr(
        IF, "importlib_version", lambda name: "0.18.0" if name == "torchao" else "0"
    )
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.delenv("PYTHONPATH", raising = False)

    loose = tmp_path / ("unsloth_subprocess_import_fix-%d" % os.getuid())
    loose.mkdir()
    os.chmod(loose, 0o777)
    theirs = tmp_path / "planted_elsewhere.py"
    theirs.write_text("# theirs\n", encoding = "utf-8")
    os.symlink(theirs, loose / ("sitecustomize.py.%d.tmp" % os.getpid()))

    try:
        assert IF.propagate_torchao_fix_to_subprocesses() == str(loose)
        target = loose / "sitecustomize.py"
        info = os.lstat(target)
        assert _stat.S_ISREG(info.st_mode), "installed the planted symlink as the hook"
        assert not _stat.S_IMODE(info.st_mode) & 0o022, oct(info.st_mode)
        assert target.read_text(encoding = "utf-8") == IF._subprocess_sitecustomize_source()
        # Their file was never opened, so rewriting it reaches nothing.
        assert theirs.read_text(encoding = "utf-8") == "# theirs\n"
    finally:
        for name in IF._TORCHAO_TORCH_SYMBOLS:
            if getattr(getattr(F, name, None), "__unsloth_placeholder__", False):
                delattr(F, name)


def test_the_staging_file_is_private_and_leaves_nothing_behind(tmp_path):
    directory = tmp_path / "dir"
    directory.mkdir(mode = 0o700)
    target = directory / "sitecustomize.py"
    IF._write_hook_atomically(str(target), "MARK = 1\n")
    assert target.read_text(encoding = "utf-8") == "MARK = 1\n"
    assert [p.name for p in directory.iterdir()] == ["sitecustomize.py"]
    if hasattr(os, "getuid"):
        import stat as _stat
        assert oct(_stat.S_IMODE(os.lstat(target).st_mode)) == oct(0o600)


# the chained sitecustomize keeps its own name -------------------------
def test_a_chained_package_stays_importable(staged, tmp_path):
    """Restoring our module under `sitecustomize` would hide the real one from
    `import sitecustomize` and break the relative imports its own callbacks
    perform after startup."""
    site, fake = staged
    other = tmp_path / "other"
    (other / "sitecustomize").mkdir(parents = True)
    (other / "sitecustomize" / "__init__.py").write_text(
        textwrap.dedent(
            """
            import atexit
            NAME = 'the_real_package'
            def _later():
                from . import extra          # delayed, as installed hooks do
                print('LATER OK', extra.NAME)
            atexit.register(_later)
            """
        ),
        encoding = "utf-8",
    )
    (other / "sitecustomize" / "extra.py").write_text("NAME = 'submodule'\n", encoding = "utf-8")
    p = _child(
        """
        import sitecustomize, torchao
        print('NAME', getattr(sitecustomize, 'NAME', '<<shadowed by ours>>'))
        print('PACKAGE', hasattr(sitecustomize, '__path__'))
        """,
        [site, other, fake],
    )
    assert "NAME the_real_package" in p.stdout, p.stdout + p.stderr
    assert "PACKAGE True" in p.stdout, p.stdout + p.stderr
    assert "LATER OK submodule" in p.stdout, p.stdout + p.stderr
    assert "Exception ignored in atexit" not in p.stderr, p.stderr


def test_a_broken_chained_module_does_not_keep_our_name(staged, tmp_path):
    """The handover only happens on success: one that raised half-way through
    must not be left behind under the name."""
    site, fake = staged
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        "MARK = 'half-initialised'\nraise RuntimeError('boom')\n", encoding = "utf-8"
    )
    p = _child(
        """
        import sys, torchao
        print('IN MODULES', 'sitecustomize' in sys.modules)
        print('MARK', getattr(sys.modules.get('sitecustomize'), 'MARK', '<<none>>'))
        print('STILL OK')
        """,
        [site, other, fake],
    )
    assert "STILL OK" in p.stdout, p.stdout + p.stderr
    assert "MARK <<none>>" in p.stdout, p.stdout + p.stderr


_COUNT_HOOKS = (
    "import sys;"
    "print('HOOKS', sum(1 for h in sys.meta_path"
    " if type(h).__name__ == '_TorchaoImportHook'))"
)


def test_a_symlink_alias_of_our_own_directory_is_not_chained(staged, tmp_path):
    """A string compare treats an alias as somebody else's sitecustomize, so
    the two spellings load and execute one another until the stack runs out.
    Every unwound level then installs another finder."""
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks only")
    site, _fake = staged
    alias = tmp_path / "alias"
    try:
        os.symlink(site, alias, target_is_directory = True)
    except (OSError, NotImplementedError) as exception:
        pytest.skip(f"cannot create a directory symlink here ({exception})")

    p = _child(_COUNT_HOOKS, [alias, site])
    assert "HOOKS 1" in p.stdout, p.stdout + p.stderr
    assert "RecursionError" not in p.stderr, p.stderr


def test_a_symlinked_copy_of_the_hook_file_is_not_chained(staged, tmp_path):
    """Same hazard one level down: the directory differs, but the
    `sitecustomize.py` inside it is this very file."""
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks only")
    site, _fake = staged
    other = tmp_path / "other"
    other.mkdir()
    try:
        os.symlink(site / "sitecustomize.py", other / "sitecustomize.py")
    except (OSError, NotImplementedError) as exception:
        pytest.skip(f"cannot create a symlink here ({exception})")

    p = _child(_COUNT_HOOKS, [site, other])
    assert "HOOKS 1" in p.stdout, p.stdout + p.stderr
    assert "RecursionError" not in p.stderr, p.stderr


def test_an_aliased_directory_still_chains_to_a_real_sitecustomize(staged, tmp_path):
    """The canonical compare must reject only ourselves, not everyone else."""
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks only")
    site, fake = staged
    alias = tmp_path / "alias"
    try:
        os.symlink(site, alias, target_is_directory = True)
    except (OSError, NotImplementedError) as exception:
        pytest.skip(f"cannot create a directory symlink here ({exception})")
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text("print('OTHER SITECUSTOMIZE RAN')\n", encoding = "utf-8")

    p = _child("import torchao; print('OK')", [alias, site, other, fake])
    assert "OTHER SITECUSTOMIZE RAN" in p.stdout, p.stdout + p.stderr
    assert "OK" in p.stdout, p.stdout + p.stderr
