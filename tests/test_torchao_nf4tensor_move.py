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

"""torchao 0.18.0 moved nf4tensor; torchtune still imports the old path.

torchao 0.18.0 relocated `torchao/dtypes/nf4tensor.py` under
`quantization/quantize_/workflows/nf4/`. torchtune imports the old path and
xcodec2 imports torchtune, so every Llasa TTS notebook died one cell after a
green install with ModuleNotFoundError. Pinning torchao below 0.18 fixes those
notebooks; aliasing the module fixes anyone importing the old path, anywhere,
on the new torchao.

Built like the vLLM tokenizer stub beside it: a meta path finder APPENDED after
the real ones, so an older torchao that still ships the module wins, and
resolution is lazy so `import unsloth` pays nothing.

The layouts are real package trees written to tmp_path and imported in a
SUBPROCESS. Stubbing `sys.modules` would test the stub; this tests the import
machinery, where appended-versus-inserted actually matters.
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
    (pkg / "dtypes").mkdir(parents = True, exist_ok = True)
    (pkg / "__init__.py").write_text("__version__ = '0.18.0'\n", encoding = "utf-8")
    (pkg / "dtypes" / "__init__.py").write_text("", encoding = "utf-8")
    if old:
        (pkg / "dtypes" / "nf4tensor.py").write_text(
            "WHICH = 'old'\ndef to_nf4(x): return x\n", encoding = "utf-8"
        )
    if new:
        d = pkg / "quantization" / "quantize_" / "workflows" / "nf4"
        d.mkdir(parents = True, exist_ok = True)
        for part in (
            pkg / "quantization",
            pkg / "quantization" / "quantize_",
            pkg / "quantization" / "quantize_" / "workflows",
            d,
        ):
            (part / "__init__.py").write_text("", encoding = "utf-8")
        (d / "nf4_tensor.py").write_text(
            "WHICH = 'new'\ndef to_nf4(x): return x\n", encoding = "utf-8"
        )
    return pkg


def _run(root: Path, body: str):
    """Import the fix in a subprocess with `root` first on sys.path.

    Loaded by file path, not as `unsloth.import_fixes`, so one function does
    not trigger unsloth's full GPU init.
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
    return subprocess.run(
        [sys.executable, "-c", script], capture_output = True, text = True, timeout = 180
    )


def test_the_new_layout_is_reachable_under_the_old_name(tmp_path):
    """The actual fix: torchao 0.18, torchtune's import works anyway."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(
        tmp_path,
        """
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("RESOLVED", m.WHICH, hasattr(m, "to_nf4"))
    """,
    )
    assert "RESOLVED new True" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_without_the_fix_that_import_fails(tmp_path):
    """The premise. Without it the test above could pass on any torchao."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(
        tmp_path,
        """
        try:
            import torchao.dtypes.nf4tensor
            print("UNEXPECTEDLY OK")
        except ModuleNotFoundError as e:
            print("EXPECTED FAILURE", e)
    """,
    )
    assert "EXPECTED FAILURE" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_an_older_torchao_still_gets_its_own_module(tmp_path):
    """Appended, not inserted at 0: on torchao < 0.18 the real module must
    win, or the fix silently swaps out working code."""
    _make_torchao(tmp_path, old = True, new = True)
    r = _run(
        tmp_path,
        """
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("RESOLVED", m.WHICH)
    """,
    )
    assert "RESOLVED old" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_neither_layout_still_raises(tmp_path):
    """An empty alias would turn a clear ModuleNotFoundError into an
    AttributeError later."""
    _make_torchao(tmp_path, old = False, new = False)
    r = _run(
        tmp_path,
        """
        _if.fix_torchao_nf4tensor_move()
        try:
            import torchao.dtypes.nf4tensor
            print("UNEXPECTEDLY OK")
        except ModuleNotFoundError:
            print("STILL RAISES")
    """,
    )
    assert "STILL RAISES" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_is_idempotent(tmp_path):
    """`import unsloth` twice, or a re-import, must not stack finders."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(
        tmp_path,
        """
        import sys
        before = len(sys.meta_path)
        for _ in range(5):
            _if.fix_torchao_nf4tensor_move()
        print("ADDED", len(sys.meta_path) - before)
    """,
    )
    assert "ADDED 1" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_no_torchao_means_no_finder(tmp_path):
    """Nothing installed: do not append a finder that can never fire.

    site-packages is pruned from sys.path first, because an empty tmp_path is
    not enough when this venv really has torchao.
    """
    r = _run(
        tmp_path,
        """
        import sys
        sys.path = [p for p in sys.path
                    if "site-packages" not in p and "dist-packages" not in p]
        import importlib
        importlib.invalidate_caches()
        assert importlib.util.find_spec("torchao") is None, "torchao still visible"
        before = len(sys.meta_path)
        _if.fix_torchao_nf4tensor_move()
        print("ADDED", len(sys.meta_path) - before)
    """,
    )
    assert "ADDED 0" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_does_not_import_torchao_eagerly(tmp_path):
    """Calling the fix must not drag torchao into every `import unsloth`,
    which on some builds is seconds and a CUDA probe."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(
        tmp_path,
        """
        import sys
        _if.fix_torchao_nf4tensor_move()
        print("LOADED", "torchao.dtypes.nf4tensor" in sys.modules)
    """,
    )
    assert "LOADED False" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_it_is_wired_into_the_init_sequence():
    """A fix nobody calls is not a fix."""
    src = (ROOT / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    assert "fix_torchao_nf4tensor_move," in src, "not imported"
    assert "fix_torchao_nf4tensor_move()" in src, "not called"


def test_the_real_environment_is_left_alone():
    """On a torchao that still ships the old path the fix must be a no-op
    rather than redirecting a working import."""
    # In a child: once anything in this session has imported unsloth the alias is registered, so an in-process check
    p = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import importlib.util, os, sys
            if importlib.util.find_spec("torchao") is None:
                print("SKIP torchao not installed"); raise SystemExit
            import torchao
            old = os.path.join(os.path.dirname(torchao.__file__),
                               "dtypes", "nf4tensor.py")
            if not os.path.isfile(old):
                print("SKIP this torchao already uses the new layout"); raise SystemExit
            spec = importlib.util.spec_from_file_location("_if_real", sys.argv[1])
            mod = importlib.util.module_from_spec(spec)
            sys.modules["_if_real"] = mod
            spec.loader.exec_module(mod)
            mod.fix_torchao_nf4tensor_move()
            import torchao.dtypes.nf4tensor as m
            print("FILE", getattr(m, "__file__", ""))
        """),
            str(ROOT / "unsloth" / "import_fixes.py"),
        ],
        capture_output = True,
        text = True,
        timeout = 600,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    if "SKIP " in p.stdout:
        pytest.skip(p.stdout.split("SKIP ", 1)[1].strip())
    line = [l for l in p.stdout.splitlines() if l.startswith("FILE ")]
    assert line, p.stdout + p.stderr
    assert "quantize_" not in line[0], line[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


def test_the_relocated_module_keeps_its_own_specification(tmp_path):
    """create_module() returns the module torchao ships, and module_from_spec
    then overwrites that shared object's __spec__ with the alias's. Left alone,
    find_spec reports the old name for the new module and reload runs the alias
    loader's no-op exec_module instead of the file."""
    _make_torchao(tmp_path, old = False, new = True)
    r = _run(
        tmp_path,
        """
        import importlib.util
        NEW = "torchao.quantization.quantize_.workflows.nf4.nf4_tensor"
        _if.fix_torchao_nf4tensor_move()
        import torchao.dtypes.nf4tensor as m
        print("SPEC", m.__spec__.name)
        print("FINDSPEC", importlib.util.find_spec(NEW).name)
        print("SAME", sys.modules[NEW] is m)
    """,
    )
    new = "torchao.quantization.quantize_.workflows.nf4.nf4_tensor"
    assert f"SPEC {new}" in r.stdout, (r.stdout, r.stderr[-2000:])
    assert f"FINDSPEC {new}" in r.stdout, (r.stdout, r.stderr[-2000:])
    assert "SAME True" in r.stdout, (r.stdout, r.stderr[-2000:])


def test_the_mlx_branch_installs_the_alias_too():
    """The MLX branch never reaches _gpu_init.py, the only other caller, so
    Apple Silicon xcodec2 would still die on the old path. Checked at source
    level because the branch only runs when mlx is importable."""
    import ast

    src = (ROOT / "unsloth" / "__init__.py").read_text(encoding = "utf-8")
    branch = [
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.If)
        and any(getattr(x, "id", None) == "_IS_MLX" for x in ast.walk(n.test))
    ]
    assert branch, "the _IS_MLX branch moved; this test needs updating"
    body = ast.get_source_segment(src, branch[0])
    assert "fix_torchao_nf4tensor_move" in body
    assert "fix_torchao_torch_symbol_skew" in body
