"""An old torchao must not end LoRA creation that never touches torchao.

`peft.import_utils.is_torchao_available` returns False when torchao is
absent, but raises when it is installed and older than peft's minimum:

    ImportError: Found an incompatible version of torchao. Found version
    0.10.0, but only versions above 0.16.0 are supported

peft calls it from `dispatch_torchao` for every LoRA layer it builds, so one
stale optional dependency ends `get_peft_model` outright.
FunctionGemma_(270M)-LMStudio dies this way on Kaggle, whose preinstalled
torchao is 0.10.0. Its sibling notebook survives on the same kernel only
because it happens to run `pip install --upgrade "torchao>=0.16.0"` first --
so the difference between pass and fail is one install line, not the model.

"Installed but unusable" is much closer to "not installed" than to "fatal".
Any other ImportError still propagates, because a torchao that fails to
import for a different reason is a real problem.
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture
def peft_env(monkeypatch):
    """A fake peft: import_utils plus a consumer that imported the name."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("peft")}

    def build(raiser):
        import_utils = types.ModuleType("peft.import_utils")
        import_utils.is_torchao_available = raiser
        consumer = types.ModuleType("peft.tuners.lora.torchao")
        # `from peft.import_utils import is_torchao_available` binds the
        # ORIGINAL here, which is what actually gets called.
        consumer.is_torchao_available = raiser
        pkg = types.ModuleType("peft")
        pkg.__path__ = []
        pkg.import_utils = import_utils
        for name, mod in (("peft", pkg),
                          ("peft.import_utils", import_utils),
                          ("peft.tuners.lora.torchao", consumer)):
            monkeypatch.setitem(sys.modules, name, mod)
        return import_utils, consumer

    yield build
    for k in [k for k in sys.modules if k.startswith("peft")]:
        if k not in saved:
            sys.modules.pop(k, None)


def _fix():
    """Load the function without importing unsloth (which needs a GPU)."""
    import ast
    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if (isinstance(node, ast.FunctionDef)
                and node.name == "fix_peft_stale_torchao_import_error"):
            ns = {"functools": __import__("functools"), "sys": sys,
                  "logger": types.SimpleNamespace(warning=lambda *a, **k: None)}
            exec(ast.get_source_segment(src, node), ns)
            return ns["fix_peft_stale_torchao_import_error"]
    raise AssertionError("fix_peft_stale_torchao_import_error not found")


FIX = _fix()

STALE = ImportError("Found an incompatible version of torchao. Found version "
                    "0.10.0, but only versions above 0.16.0 are supported")


def _raiser(exc):
    def is_torchao_available():
        raise exc
    return is_torchao_available


# ---- the bug --------------------------------------------------------------

def test_stale_torchao_becomes_false(peft_env):
    iu, _ = peft_env(_raiser(STALE))
    assert FIX() is True
    assert iu.is_torchao_available() is False


def test_the_module_that_actually_calls_it_is_patched(peft_env):
    # dispatch_torchao lives here and holds its own reference; patching only
    # import_utils would leave the real call site still raising.
    _, consumer = peft_env(_raiser(STALE))
    FIX()
    assert consumer.is_torchao_available() is False


def test_warning_is_emitted_once(peft_env):
    iu, _ = peft_env(_raiser(STALE))
    seen = []
    import ast
    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "fix_peft_stale_torchao_import_error")
    ns = {"functools": __import__("functools"), "sys": sys,
          "logger": types.SimpleNamespace(warning=seen.append)}
    exec(ast.get_source_segment(src, node), ns)
    ns["fix_peft_stale_torchao_import_error"]()
    for _ in range(5):
        iu.is_torchao_available()
    assert len(seen) == 1, "one stale dependency, one message"
    assert "torchao" in seen[0]
    assert "upgrade" in seen[0].lower()


# ---- what must still fail -------------------------------------------------

def test_an_unrelated_import_error_still_raises(peft_env):
    iu, _ = peft_env(_raiser(ImportError("libcudart.so.12: cannot open "
                                         "shared object file")))
    FIX()
    with pytest.raises(ImportError):
        iu.is_torchao_available()


def test_a_non_import_error_still_raises(peft_env):
    iu, _ = peft_env(_raiser(RuntimeError("torchao exploded")))
    FIX()
    with pytest.raises(RuntimeError):
        iu.is_torchao_available()


# ---- what must not change -------------------------------------------------

def test_a_working_torchao_still_answers_true(peft_env):
    iu, _ = peft_env(lambda: True)
    FIX()
    assert iu.is_torchao_available() is True


def test_absent_torchao_still_answers_false(peft_env):
    iu, _ = peft_env(lambda: False)
    FIX()
    assert iu.is_torchao_available() is False


def test_no_peft_is_not_an_error(monkeypatch):
    for k in [k for k in sys.modules if k.startswith("peft")]:
        monkeypatch.delitem(sys.modules, k, raising=False)
    monkeypatch.setattr(sys, "path", [p for p in sys.path])
    import builtins
    real = builtins.__import__

    def no_peft(name, *a, **k):
        if name.startswith("peft"):
            raise ModuleNotFoundError("No module named 'peft'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_peft)
    assert FIX() is None


def test_applying_twice_is_a_no_op(peft_env):
    iu, _ = peft_env(_raiser(STALE))
    assert FIX() is True
    first = iu.is_torchao_available
    assert FIX() is False, "already patched"
    assert iu.is_torchao_available is first, "must not stack wrappers"


def test_metadata_survives(peft_env):
    iu, _ = peft_env(_raiser(STALE))
    FIX()
    assert iu.is_torchao_available.__name__ == "is_torchao_available"


# ---- wiring ---------------------------------------------------------------

def test_called_from_gpu_init():
    src = (REPO_ROOT / "unsloth" / "_gpu_init.py").read_text(encoding="utf-8")
    assert "fix_peft_stale_torchao_import_error,\n" in src, "not imported"
    assert "\nfix_peft_stale_torchao_import_error()\n" in src, "not called"
    assert "\ndel fix_peft_stale_torchao_import_error\n" in src, "not cleaned up"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
