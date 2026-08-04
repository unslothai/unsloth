"""The torchao fix has to reach processes unsloth does not launch.

`fix_torchao_torch_symbol_skew` patches `torch.nn.functional` in the current
interpreter. vLLM inspects model architectures in a SEPARATE process, which
imports torch and torchao itself and sees none of that, so with
`fast_inference = True` the run still dies:

    ERROR registry.py:781] Error in inspecting model architecture 'Qwen3ForCausalLM'
    ERROR registry.py:781] ImportError: cannot import name 'ScalingType' from 'torch.nn.functional'

and the user is shown only "Model architectures ['Qwen3ForCausalLM'] failed to
be inspected", which names neither torchao nor torch. Observed on Colab AFTER
the in-process fix had reported success, so the parent looked healthy right up
to the failure.

`sitecustomize` is the hook that reaches such a process: `site` imports it at
interpreter startup, it is found on PYTHONPATH, and subprocesses inherit
PYTHONPATH. A `.pth` file would work too but only inside a real site
directory, which a library should not be writing into.

The hazard that dominates these tests is **shadowing**. `sitecustomize` is a
single global name and other things legitimately install one -- the machine
this was developed on has one at /etc/unslothai/python/sitecustomize.py -- so
replacing it would silently disable whatever it does for every subprocess
unsloth spawns.
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


# ---- the generated sitecustomize -----------------------------------------

def test_it_is_valid_python():
    """It is imported at the start of EVERY subprocess. A syntax error here
    breaks all of them, which is far worse than the bug being fixed."""
    compile(IF._subprocess_sitecustomize_source(), "<sitecustomize>", "exec")


def test_it_chains_instead_of_shadowing():
    src = IF._subprocess_sitecustomize_source()
    assert "_chain_to_the_real_sitecustomize" in src
    assert "sitecustomize.py" in src


def test_the_chain_runs_before_the_fix():
    src = IF._subprocess_sitecustomize_source()
    assert src.index("_chain_to_the_real_sitecustomize()") < src.index("_apply()")


def test_it_never_raises_at_interpreter_startup():
    src = IF._subprocess_sitecustomize_source()
    tail = src[src.index("_chain_to_the_real_sitecustomize()"):]
    assert "except Exception:" in tail


def test_it_is_version_gated():
    src = IF._subprocess_sitecustomize_source()
    assert "(0, 18)" in src, "torchao < 0.18 guards its own import"


def test_it_does_not_overwrite_existing_torch_symbols():
    assert "if hasattr(F, name):" in IF._subprocess_sitecustomize_source()


def test_the_child_never_imports_unsloth():
    """This runs at the START of every subprocess on the machine. Importing
    unsloth there would pay the full import cost each time and could recurse
    back through propagate_torchao_fix_to_subprocesses itself. The logic is
    inlined for that reason."""
    src = IF._subprocess_sitecustomize_source()
    assert "import unsloth" not in src
    assert "from unsloth" not in src


def test_it_only_imports_the_stdlib_and_torch():
    """Anything heavier makes interpreter startup slower for every process."""
    import ast
    tree = ast.parse(IF._subprocess_sitecustomize_source())
    allowed = {"os", "sys", "importlib", "importlib.util",
               "importlib.metadata", "torch", "torch.nn.functional"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name in allowed, f"unexpected import: {a.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module in allowed, f"unexpected import: {node.module}"


# ---- staging ---------------------------------------------------------------

@pytest.fixture(autouse=True)
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
    return not all(hasattr(F, n) for n in IF._TORCHAO_TORCH_SYMBOLS)


def test_it_is_a_noop_on_a_healthy_environment():
    """Nothing is written and PYTHONPATH is untouched unless the fix is
    actually needed. This environment has torchao 0.17, which is healthy."""
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
    """Concurrent runs must never let a subprocess read a truncated
    sitecustomize -- that would be a SyntaxError at startup."""
    import inspect
    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    assert "os.replace" in src


def test_it_is_idempotent_on_pythonpath():
    import inspect
    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    assert "if directory not in parts:" in src


# ---- the directory it writes into -----------------------------------------

def test_the_directory_is_private_to_this_user():
    """The temp dir is shared, and everything on PYTHONPATH runs in every
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

    monkeypatch.setattr(os, "lstat", lambda p: _NotOurs() if str(p) == str(hostile)
                        else real_lstat(p))
    with pytest.raises(RuntimeError, match="owned by another user"):
        IF._subprocess_fix_directory()


def test_the_refusal_does_not_propagate_as_a_crash(monkeypatch):
    """`propagate_...` wraps the staging in try/except and warns. A hostile
    directory must degrade to "no subprocess fix", not kill the import."""
    import inspect
    src = inspect.getsource(IF.propagate_torchao_fix_to_subprocesses)
    i = src.index("_subprocess_fix_directory()")
    assert "try:" in src[:i]
    assert "except Exception as exception:" in src[i:]


# ---- behaviour, with real interpreters ------------------------------------

@pytest.fixture
def staged(tmp_path):
    """The generated sitecustomize on a PYTHONPATH dir, plus a fake torch and
    a fake torchao 0.18 that reproduces the real import error."""
    site = tmp_path / "hook"
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        IF._subprocess_sitecustomize_source(), encoding="utf-8")

    fake = tmp_path / "fake"
    (fake / "torch" / "nn").mkdir(parents=True)
    (fake / "torch" / "__init__.py").write_text(
        "from . import nn\n__version__='2.9.0'\n", encoding="utf-8")
    (fake / "torch" / "nn" / "__init__.py").write_text(
        "from . import functional\n", encoding="utf-8")
    (fake / "torch" / "nn" / "functional.py").write_text(
        "def linear(*a, **k):\n    return None\n", encoding="utf-8")
    (fake / "torchao").mkdir()
    (fake / "torchao" / "__init__.py").write_text(
        "from torch.nn.functional import ScalingType, scaled_grouped_mm\n"
        "__version__='0.18.0'\n", encoding="utf-8")
    d = fake / "torchao-0.18.0.dist-info"
    d.mkdir()
    (d / "METADATA").write_text(
        "Metadata-Version: 2.1\nName: torchao\nVersion: 0.18.0\n",
        encoding="utf-8")
    return site, fake


def _child(code: str, path_entries, timeout=300):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in path_entries)
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                          capture_output=True, text=True, timeout=timeout,
                          env=env)


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


def test_it_also_reaches_a_grandchild(staged):
    """vLLM's inspector is not necessarily a direct child."""
    site, fake = staged
    p = _child(
        "import subprocess, sys;"
        "r = subprocess.run([sys.executable, '-c',"
        "'import torchao; print(\"GRANDCHILD OK\")'],"
        "capture_output=True, text=True);"
        "print(r.stdout, r.stderr)", [site, fake])
    assert "GRANDCHILD OK" in p.stdout, p.stdout + p.stderr


def test_the_placeholder_still_refuses_to_be_used(staged):
    site, fake = staged
    p = _child('''
        import torchao
        from torch.nn.functional import ScalingType
        try:
            ScalingType.DYNAMIC
            print("NO ERROR")
        except RuntimeError as e:
            print("RAISED", "does not exist in this torch" in str(e))
    ''', [site, fake])
    assert "RAISED True" in p.stdout, p.stdout + p.stderr


def test_an_existing_sitecustomize_still_runs(staged, tmp_path):
    """The shadowing hazard, which is real: this machine already has a
    sitecustomize at /etc/unslothai/python/. Ours must chain to it."""
    site, fake = staged
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        "print('OTHER SITECUSTOMIZE RAN')\n", encoding="utf-8")
    p = _child("import torchao; print('OK')", [site, other, fake])
    assert "OTHER SITECUSTOMIZE RAN" in p.stdout, p.stdout + p.stderr
    assert "OK" in p.stdout


def test_a_broken_existing_sitecustomize_does_not_kill_the_process(staged,
                                                                   tmp_path):
    """Chaining must not turn somebody else's bug into a startup crash for
    every subprocess."""
    site, fake = staged
    other = tmp_path / "other"
    other.mkdir()
    (other / "sitecustomize.py").write_text(
        "raise RuntimeError('boom')\n", encoding="utf-8")
    p = _child("import torchao; print('STILL OK')", [site, other, fake])
    assert "STILL OK" in p.stdout, p.stdout + p.stderr


def test_it_does_nothing_when_torchao_is_absent(staged, tmp_path):
    """No torchao, no patching: torch.nn.functional must be untouched."""
    site, _fake = staged
    bare = tmp_path / "bare"
    (bare / "torch" / "nn").mkdir(parents=True)
    (bare / "torch" / "__init__.py").write_text(
        "from . import nn\n__version__='2.9.0'\n", encoding="utf-8")
    (bare / "torch" / "nn" / "__init__.py").write_text(
        "from . import functional\n", encoding="utf-8")
    (bare / "torch" / "nn" / "functional.py").write_text(
        "def linear(*a, **k):\n    return None\n", encoding="utf-8")
    p = _child('''
        import torch.nn.functional as F
        print("PATCHED", hasattr(F, "ScalingType"))
    ''', [site, bare])
    assert "PATCHED False" in p.stdout, p.stdout + p.stderr


def test_the_hook_does_not_disturb_ordinary_startup(staged):
    site, fake = staged
    p = _child("print('HELLO')", [site, fake])
    assert p.returncode == 0
    assert "HELLO" in p.stdout


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
