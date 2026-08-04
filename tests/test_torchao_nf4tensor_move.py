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

"""torchao 0.18.0 moved nf4tensor; torchtune still imports the old path.

torchao 0.18.0 (2026-08-03) relocated `torchao/dtypes/nf4tensor.py` to
`torchao/quantization/quantize_/workflows/nf4/nf4_tensor.py`. torchtune imports
the old path and xcodec2 imports torchtune, so every Llasa TTS notebook died
one cell after a green install with

    ModuleNotFoundError: No module named 'torchao.dtypes.nf4tensor'

Pinning torchao below 0.18 in those notebooks fixes those notebooks. Aliasing
the module in `import_fixes.py` fixes anyone who imports the old path, on any
notebook, and lets them keep the new torchao.

Built like the vLLM tokenizer stub beside it: a meta path finder APPENDED after
the real finders, so an older torchao that still ships the module always wins,
and resolution is lazy so `import unsloth` pays nothing for it.

The layouts here are real package trees written to a tmp_path and imported in a
SUBPROCESS. Stubbing `sys.modules` would test the stub; this tests the import
machinery, which is where an appended-versus-inserted finder actually matters.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _make_torchao(root: Path, *, old: bool, new: bool):
    """A minimal torchao package with either layout, or neither."""
    pkg = root / "torchao"
    (pkg / "dtypes").mkdir(parents=True, exist_ok=True)
    (pkg / "__init__.py").write_text("__version__ = '0.18.0'\n", encoding="utf-8")
    (pkg / "dtypes" / "__init__.py").write_text("", encoding="utf-8")
    if old:
        (pkg / "dtypes" / "nf4tensor.py").write_text(
            "WHICH = 'old'\ndef to_nf4(x): return x\n", encoding="utf-8")
    if new:
        d = pkg / "quantization" / "quantize_" / "workflows" / "nf4"
        d.mkdir(parents=True, exist_ok=True)
        for part in (pkg / "quantization", pkg / "quantization" / "quantize_",
                     pkg / "quantization" / "quantize_" / "workflows", d):
            (part / "__init__.py").write_text("", encoding="utf-8")
        (d / "nf4_tensor.py").write_text(
            "WHICH = 'new'\ndef to_nf4(x): return x\n", encoding="utf-8")
    return pkg


def _run(root: Path, body: str):
    """Import the fix in a subprocess with `root` first on sys.path.

    import_fixes is loaded by file path, not as `unsloth.import_fixes`, so the
    test never triggers unsloth's full GPU init just to reach one function.
    """
    script = textwrap.dedent(f"""
        import sys, importlib.util
        sys.path.insert(0, {str(root)!r})
        spec = importlib.util.spec_from_file_location(
            "_if", {str(ROOT / "unsloth" / "import_fixes.py")!r})
        _if = importlib.util.module_from_spec(spec)
        sys.modules["_if"] = _if
        spec.loader.exec_module(_if)
    """) + textwrap.dedent(body)
    return subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=180)


def test_the_new_layout_is_reachable_under_the_old_name(tmp_path):
    """The actual fix: torchao 0.18, torchtune's import works anyway."""
    _make_torchao(tmp_path, old=False, new=True)
    r = _run(tmp_path, """
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("RESOLVED", m.WHICH, hasattr(m, "to_nf4"))
    """)
    assert "RESOLVED new True" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_without_the_fix_that_import_fails(tmp_path):
    """The premise. Without it the test above could pass on any torchao."""
    _make_torchao(tmp_path, old=False, new=True)
    r = _run(tmp_path, """
        try:
            import torchao.dtypes.nf4tensor
            print("UNEXPECTEDLY OK")
        except ModuleNotFoundError as e:
            print("EXPECTED FAILURE", e)
    """)
    assert "EXPECTED FAILURE" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_an_older_torchao_still_gets_its_own_module(tmp_path):
    """Appended, not inserted at 0. On torchao < 0.18 the real module must win,
    or the fix would silently swap out code that was working."""
    _make_torchao(tmp_path, old=True, new=True)
    r = _run(tmp_path, """
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("RESOLVED", m.WHICH)
    """)
    assert "RESOLVED old" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_neither_layout_still_raises(tmp_path):
    """A genuinely missing module must not be masked by an empty alias -- that
    would turn a clear ModuleNotFoundError into an AttributeError later."""
    _make_torchao(tmp_path, old=False, new=False)
    r = _run(tmp_path, """
        _if.fix_torchao_nf4tensor_move()
        try:
            import torchao.dtypes.nf4tensor
            print("UNEXPECTEDLY OK")
        except ModuleNotFoundError:
            print("STILL RAISES")
    """)
    assert "STILL RAISES" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_is_idempotent(tmp_path):
    """`import unsloth` twice, or a re-import, must not stack finders."""
    _make_torchao(tmp_path, old=False, new=True)
    r = _run(tmp_path, """
        import sys
        before = len(sys.meta_path)
        for _ in range(5):
            _if.fix_torchao_nf4tensor_move()
        print("ADDED", len(sys.meta_path) - before)
    """)
    assert "ADDED 1" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_no_torchao_means_no_finder(tmp_path):
    """Nothing installed: do not append a finder that can never fire.

    site-packages is pruned from sys.path first. An empty tmp_path is not
    enough -- this venv really has torchao, so the first version of this test
    asserted no finder while a finder was correctly being added, and it was the
    test that was wrong.
    """
    r = _run(tmp_path, """
        import sys
        sys.path = [p for p in sys.path
                    if "site-packages" not in p and "dist-packages" not in p]
        import importlib
        importlib.invalidate_caches()
        assert importlib.util.find_spec("torchao") is None, "torchao still visible"
        before = len(sys.meta_path)
        _if.fix_torchao_nf4tensor_move()
        print("ADDED", len(sys.meta_path) - before)
    """)
    assert "ADDED 0" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_does_not_import_torchao_eagerly(tmp_path):
    """Lazy by construction. Calling the fix must not drag torchao into every
    `import unsloth`, which on some builds is seconds and a CUDA probe."""
    _make_torchao(tmp_path, old=False, new=True)
    r = _run(tmp_path, """
        import sys
        _if.fix_torchao_nf4tensor_move()
        print("LOADED", "torchao.dtypes.nf4tensor" in sys.modules)
    """)
    assert "LOADED False" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_is_wired_into_the_init_sequence():
    """A fix nobody calls is not a fix."""
    src = (ROOT / "unsloth" / "_gpu_init.py").read_text(encoding="utf-8")
    assert "fix_torchao_nf4tensor_move," in src, "not imported"
    assert "fix_torchao_nf4tensor_move()" in src, "not called"


def test_the_real_environment_is_left_alone():
    """This machine has torchao 0.17, which still ships the old path. The fix
    must be a no-op there rather than redirecting a working import."""
    import importlib.util
    if importlib.util.find_spec("torchao") is None:
        pytest.skip("torchao not installed")
    try:
        has_old = importlib.util.find_spec("torchao.dtypes.nf4tensor") is not None
    except Exception:
        has_old = False
    if not has_old:
        pytest.skip("this torchao already uses the new layout")
    import importlib
    spec = importlib.util.spec_from_file_location(
        "_if_real", ROOT / "unsloth" / "import_fixes.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_if_real"] = mod
    spec.loader.exec_module(mod)
    mod.fix_torchao_nf4tensor_move()
    import torchao.dtypes.nf4tensor as m
    assert "quantize_" not in getattr(m, "__file__", ""), m.__file__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_relocated_module_keeps_its_own_specification(tmp_path):
    """create_module() returns the module torchao actually ships, and
    module_from_spec then overwrites that shared object's __spec__ with the
    alias's (importlib/_bootstrap.py assigns __spec__ unconditionally). Left
    alone, find_spec reports the old name for the new module and reload runs
    the alias loader's no-op exec_module instead of the file."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(tmp_path, """
        import importlib.util
        NEW = "torchao.quantization.quantize_.workflows.nf4.nf4_tensor"
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("SPEC", m.__spec__.name)
        print("FINDSPEC", importlib.util.find_spec(NEW).name)
        print("SAME", sys.modules[NEW] is m)
    """)
    new = "torchao.quantization.quantize_.workflows.nf4.nf4_tensor"
    assert f"SPEC {new}" in r.stdout, (r.stdout, r.stderr[-2000:])
    assert f"FINDSPEC {new}" in r.stdout, (r.stdout, r.stderr[-2000:])
    assert "SAME True" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_the_mlx_branch_installs_the_alias_too():
    """_gpu_init.py is the only other caller and the MLX branch never reaches
    it, so on Apple Silicon xcodec2 would still die on the old path. Source
    level, because the branch only runs when mlx is importable."""
    import ast
    src = (ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    branch = [n for n in ast.walk(ast.parse(src))
              if isinstance(n, ast.If)
              and any(getattr(x, "id", None) == "_IS_MLX"
                      for x in ast.walk(n.test))]
    assert branch, "the _IS_MLX branch moved; this test needs updating"
    body = ast.get_source_segment(src, branch[0])
    assert "fix_torchao_nf4tensor_move" in body
    assert "fix_torchao_torch_symbol_skew" in body
