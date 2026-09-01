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

"""An old torchao must not end LoRA creation that never touches torchao.

`peft.import_utils.is_torchao_available` returns False when torchao is absent
but raises when it is installed and older than peft's minimum, and
`dispatch_torchao` calls it for every LoRA layer, so one stale optional
dependency ends `get_peft_model`. FunctionGemma_(270M)-LMStudio dies this way
on Kaggle, whose preinstalled torchao is 0.10.0; its sibling notebook survives
the same kernel only because it upgrades torchao first.

"Installed but unusable" is closer to "not installed" than to "fatal". Any
other ImportError still propagates, including ones whose message also says
"torchao" (missing submodule, unloadable extension), which is why the version
complaint is matched rather than the word.
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
        # `from peft.import_utils import ...` binds the ORIGINAL here, and this is the copy that actually gets called.
        consumer.is_torchao_available = raiser
        pkg = types.ModuleType("peft")
        pkg.__path__ = []
        pkg.import_utils = import_utils
        for name, mod in (
            ("peft", pkg),
            ("peft.import_utils", import_utils),
            ("peft.tuners.lora.torchao", consumer),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        return import_utils, consumer

    yield build
    for k in [k for k in sys.modules if k.startswith("peft")]:
        if k not in saved:
            sys.modules.pop(k, None)


_WANTED = ("fix_peft_stale_torchao_import_error", "_TORCHAO_STALE_VERSION_ERROR")


def _fix(warning = None):
    """Load the function without importing unsloth (which needs a GPU).

    The module-level regex it consults must come along, or the wrapper
    NameErrors on the first suppressed ImportError.
    """
    import ast
    import re

    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    ns = {
        "functools": __import__("functools"),
        "sys": sys,
        "re": re,
        "logger": types.SimpleNamespace(
            warning = warning if warning is not None else (lambda *a, **k: None),
        ),
    }
    for node in tree.body:
        name = None
        if isinstance(node, ast.FunctionDef):
            name = node.name
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
        if name in _WANTED:
            exec(ast.get_source_segment(src, node), ns)
    for name in _WANTED:
        assert name in ns, f"{name} not found in import_fixes.py"
    return ns["fix_peft_stale_torchao_import_error"]


FIX = _fix()

STALE = ImportError(
    "Found an incompatible version of torchao. Found version "
    "0.10.0, but only versions above 0.16.0 are supported"
)


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
    # dispatch_torchao holds its own reference;
    _, consumer = peft_env(_raiser(STALE))
    FIX()
    assert consumer.is_torchao_available() is False


def test_warning_is_emitted_once(peft_env):
    iu, _ = peft_env(_raiser(STALE))
    seen = []
    _fix(warning = seen.append)()
    for _ in range(5):
        iu.is_torchao_available()
    assert len(seen) == 1, "one stale dependency, one message"
    assert "torchao" in seen[0]
    assert "upgrade" in seen[0].lower()


# ---- what must still fail -------------------------------------------------
def test_an_unrelated_import_error_still_raises(peft_env):
    iu, _ = peft_env(_raiser(ImportError("libcudart.so.12: cannot open shared object file")))
    FIX()
    with pytest.raises(ImportError):
        iu.is_torchao_available()


@pytest.mark.parametrize(
    "message",
    [
        # Half-installed torchao: says "torchao", is not a version complaint, and calling it "unavailable" would hide a
        "No module named 'torchao.quantization'",
        "cannot import name 'quantize_' from 'torchao'",
        # An extension built against a different torch/CUDA.
        "libtorchao_ops_cuda.so: cannot open shared object file: No such file or directory",
        "/site-packages/torchao/_C.so: undefined symbol: _ZN3c105ErrorC1E",
    ],
)
def test_a_broken_torchao_still_raises_even_though_it_says_torchao(peft_env, message):
    iu, _ = peft_env(_raiser(ImportError(message)))
    FIX()
    with pytest.raises(ImportError):
        iu.is_torchao_available()


@pytest.mark.parametrize(
    "message",
    [
        # peft's current wording.
        "Found an incompatible version of torchao. Found version 0.10.0, "
        "but only versions above 0.16.0 are supported",
        # Rewordings that must keep being read as "too old", not "broken".
        "torchao 0.10.0 is installed but only versions above 0.16.0 are supported",
        "This requires torchao>=0.16.0",
    ],
)
def test_every_spelling_of_the_version_complaint_is_swallowed(peft_env, message):
    iu, _ = peft_env(_raiser(ImportError(message)))
    FIX()
    assert iu.is_torchao_available() is False


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
        monkeypatch.delitem(sys.modules, k, raising = False)
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
    src = (REPO_ROOT / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    assert "fix_peft_stale_torchao_import_error,\n" in src, "not imported"
    assert "\nfix_peft_stale_torchao_import_error()\n" in src, "not called"
    assert "\ndel fix_peft_stale_torchao_import_error\n" in src, "not cleaned up"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
