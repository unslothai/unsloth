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

"""A class torchao deleted must not end LoRA creation that never touches torchao.

peft's LoRA dispatch table runs every dispatcher in order and keeps the first
module one returns, so a dispatcher that RAISES ends `get_peft_model` for
models it does not apply to. `dispatch_torchao` imports `AffineQuantizedTensor`
and `LinearActivationQuantizedTensor` at call time purely to run one
`isinstance` check, and torchao 0.18.0 deleted the second one outright; peft
0.18.1 still imports it. 4-bit runs never notice, because `dispatch_bnb_4bit`
matches and returns first; plain 16-bit LoRA falls through to the torchao
dispatcher and dies at step 0.

Declining (None) for every weight would swap a loud bug for a quiet one:
`AffineQuantizedTensor` is still importable, so a weight of that class would
silently get an ordinary LoRA layer instead of `TorchaoLoraLinear`. The
dispatcher therefore keeps doing its isinstance check against whichever of the
two classes this torchao ships, and only weights of the removed class go
unmatched. A torchao that is BROKEN rather than newer must still raise, which
is why the two class names are matched rather than the word "torchao".
"""

import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


_WANTED = (
    "fix_peft_torchao_missing_tensor_subclass",
    "_guard_peft_torchao_dispatcher",
    "_peft_torchao_tensor_subclasses",
    "_PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS",
    "_PEFT_TORCHAO_TENSOR_SUBCLASSES",
)


def _fix(warning = None):
    """Load the fix without importing unsloth (which needs a GPU)."""
    import ast
    import functools
    import importlib
    import inspect
    import re

    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    tree = ast.parse(src)
    ns = {
        "functools": functools,
        "importlib": importlib,
        "inspect": inspect,
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
    return ns["fix_peft_torchao_missing_tensor_subclass"]


FIX = _fix()

MISSING = ImportError(
    "cannot import name 'LinearActivationQuantizedTensor' from "
    "'torchao.quantization' (/site-packages/torchao/quantization/__init__.py)"
)


class _BlockTorchao:
    """A meta path finder that makes torchao look absent rather than merely reduced."""

    def find_module(
        self,
        fullname,
        path = None,
    ):
        return None

    def find_spec(
        self,
        fullname,
        path = None,
        target = None,
    ):
        if fullname == "torchao" or fullname.startswith("torchao."):
            raise ModuleNotFoundError("No module named 'torchao'", name = "torchao")
        return None


@pytest.fixture
def fake_torchao(monkeypatch):
    """Stand in for torchao with a chosen subset of the two tensor subclasses present.

    torchao 0.18.0 keeps `AffineQuantizedTensor` only as an empty stub for
    backwards compatibility, so a real instance of it cannot be built on that
    version; these classes stand in for the real ones peft's isinstance check
    was written against.
    """

    def build(affine = True, linear_activation = False):
        for name in [k for k in sys.modules if k == "torchao" or k.startswith("torchao.")]:
            monkeypatch.delitem(sys.modules, name, raising = False)
        classes = {}
        pkg = types.ModuleType("torchao")
        pkg.__path__ = []
        dtypes = types.ModuleType("torchao.dtypes")
        quantization = types.ModuleType("torchao.quantization")
        if affine:
            classes["AffineQuantizedTensor"] = type("AffineQuantizedTensor", (), {})
            dtypes.AffineQuantizedTensor = classes["AffineQuantizedTensor"]
        if linear_activation:
            classes["LinearActivationQuantizedTensor"] = type(
                "LinearActivationQuantizedTensor",
                (),
                {},
            )
            quantization.LinearActivationQuantizedTensor = classes[
                "LinearActivationQuantizedTensor"
            ]
        pkg.dtypes = dtypes
        pkg.quantization = quantization
        for name, mod in (
            ("torchao", pkg),
            ("torchao.dtypes", dtypes),
            ("torchao.quantization", quantization),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        return classes

    def absent():
        for name in [k for k in sys.modules if k == "torchao" or k.startswith("torchao.")]:
            monkeypatch.delitem(sys.modules, name, raising = False)
        blocker = _BlockTorchao()
        monkeypatch.setattr(sys, "meta_path", [blocker] + list(sys.meta_path))

    build.absent = absent
    return build


class _FakeBaseTunerLayer:
    def get_base_layer(self):
        return self.base_layer


class _FakeTorchaoLoraLinear:
    """Stands in for peft's TorchaoLoraLinear so construction is observable."""

    def __init__(self, target, adapter_name, **kwargs):
        self.target = target
        self.adapter_name = adapter_name
        self.kwargs = kwargs


@pytest.fixture
def peft_env(monkeypatch, fake_torchao):
    """A fake peft: the module defining dispatch_torchao plus the one that imported it."""
    saved = {k: v for k, v in sys.modules.items() if k.startswith("peft")}

    def build(dispatcher, torchao_available = True):
        definer = types.ModuleType("peft.tuners.lora.torchao")
        definer.dispatch_torchao = dispatcher
        # The degraded path reaches these three exactly where upstream's dispatcher does.
        definer.TorchaoLoraLinear = _FakeTorchaoLoraLinear
        tuners_utils = types.ModuleType("peft.tuners.tuners_utils")
        tuners_utils.BaseTunerLayer = _FakeBaseTunerLayer
        import_utils = types.ModuleType("peft.import_utils")
        import_utils.is_torchao_available = lambda: torchao_available
        # `from .torchao import dispatch_torchao` binds the ORIGINAL here, and model.py is where
        # the dispatch list is actually built, so this copy is the one that really runs.
        caller = types.ModuleType("peft.tuners.lora.model")
        caller.dispatch_torchao = dispatcher
        pkg = types.ModuleType("peft")
        pkg.__path__ = []
        for name, mod in (
            ("peft", pkg),
            ("peft.import_utils", import_utils),
            ("peft.tuners", types.ModuleType("peft.tuners")),
            ("peft.tuners.tuners_utils", tuners_utils),
            ("peft.tuners.lora", types.ModuleType("peft.tuners.lora")),
            ("peft.tuners.lora.torchao", definer),
            ("peft.tuners.lora.model", caller),
        ):
            monkeypatch.setitem(sys.modules, name, mod)
        return definer, caller

    yield build
    for k in [k for k in sys.modules if k.startswith("peft")]:
        if k not in saved:
            sys.modules.pop(k, None)


def _raiser(exc):
    def dispatch_torchao(
        target,
        adapter_name,
        lora_config = None,
        **kwargs,
    ):
        raise exc

    return dispatch_torchao


class _Layer:
    def __init__(self, weight):
        self.weight = weight


# ---- the bug --------------------------------------------------------------


def test_a_deleted_tensor_subclass_no_longer_ends_dispatch(peft_env, fake_torchao):
    fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    assert FIX() is True
    # A weight of neither class: what plain 16-bit LoRA has, and the case that used to raise.
    assert definer.dispatch_torchao(_Layer("plain"), "default") is None


def test_the_module_that_actually_dispatches_is_patched(peft_env, fake_torchao):
    # model.py holds its own reference; patching the defining module alone would leave the real
    # call site raising.
    fake_torchao()
    _, caller = peft_env(_raiser(MISSING))
    FIX()
    assert caller.dispatch_torchao(_Layer("plain"), "default") is None


def test_both_copies_share_one_wrapper_so_it_warns_once(peft_env, fake_torchao):
    seen = []
    fake_torchao()
    definer, caller = peft_env(_raiser(MISSING))
    _fix(warning = seen.append)()
    for _ in range(5):
        definer.dispatch_torchao(_Layer("plain"), "default")
        caller.dispatch_torchao(_Layer("plain"), "default")
    assert len(seen) == 1, "one missing class, one message"
    assert "LinearActivationQuantizedTensor" in seen[0]
    assert "torchao" in seen[0]


def test_the_warning_does_not_claim_the_other_class_is_gone_too(peft_env, fake_torchao):
    # The message must not say torchao weights cannot exist: AffineQuantizedTensor is still there
    # and is still matched, so only the removed subclass is unreachable.
    seen = []
    fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    _fix(warning = seen.append)()
    definer.dispatch_torchao(_Layer("plain"), "default")
    assert len(seen) == 1
    assert "AffineQuantizedTensor" in seen[0], "must say which class is still handled"
    assert "cannot exist" not in seen[0]


@pytest.mark.parametrize(
    "message",
    [
        # peft 0.18.1's wording on torchao 0.18.0.
        "cannot import name 'LinearActivationQuantizedTensor' from 'torchao.quantization'",
        # Code reaching past the package for the same class.
        "No module named 'torchao.quantization.linear_activation_quantized_tensor'",
    ],
)
def test_every_spelling_of_the_missing_class_is_handled(peft_env, fake_torchao, message):
    fake_torchao()
    definer, _ = peft_env(_raiser(ImportError(message)))
    FIX()
    assert definer.dispatch_torchao(_Layer("plain"), "default") is None


# ---- the class that is still there must still be matched ------------------


def test_an_affine_quantized_weight_still_gets_the_torchao_lora_layer(peft_env, fake_torchao):
    """The regression this guards: returning None would give a real torchao weight plain LoRA."""
    classes = fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    target = _Layer(classes["AffineQuantizedTensor"]())
    built = definer.dispatch_torchao(target, "default", lora_config = "config", r = 8)
    assert isinstance(built, _FakeTorchaoLoraLinear), "must not fall through to dispatch_default"
    assert built.target is target
    assert built.adapter_name == "default"
    assert built.kwargs == {"r": 8}, "lora_config is a named parameter, not a layer kwarg"


def test_the_third_parameter_is_read_by_position_not_by_name(peft_env, fake_torchao):
    # peft 0.19 renamed dispatch_torchao's third parameter from lora_config to config (and moved
    # to a single TorchAOBaseTensor import, so only peft <= 0.18 reaches this path at all). Read
    # by position so a rename cannot leak the config through as a layer kwarg.
    classes = fake_torchao(affine = True, linear_activation = False)

    def dispatch_torchao(
        target,
        adapter_name,
        config = None,
        **kwargs,
    ):
        raise MISSING

    definer, _ = peft_env(dispatch_torchao)
    FIX()
    target = _Layer(classes["AffineQuantizedTensor"]())
    built = definer.dispatch_torchao(target, "default", config = "config", r = 8)
    assert isinstance(built, _FakeTorchaoLoraLinear)
    assert built.kwargs == {"r": 8}


def test_a_weight_of_neither_class_still_declines(peft_env, fake_torchao):
    fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    assert definer.dispatch_torchao(_Layer(object()), "default") is None


def test_the_mirror_case_matches_the_other_class(peft_env, fake_torchao):
    # If torchao ever drops AffineQuantizedTensor instead, the surviving class must still match.
    classes = fake_torchao(affine = False, linear_activation = True)
    gone = ImportError("cannot import name 'AffineQuantizedTensor' from 'torchao.dtypes'")
    definer, _ = peft_env(_raiser(gone))
    FIX()
    target = _Layer(classes["LinearActivationQuantizedTensor"]())
    assert isinstance(definer.dispatch_torchao(target, "default"), _FakeTorchaoLoraLinear)


def test_neither_class_present_declines(peft_env, fake_torchao):
    fake_torchao(affine = False, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    assert definer.dispatch_torchao(_Layer("plain"), "default") is None


def test_a_base_tuner_layer_target_is_unwrapped_like_upstream(peft_env, fake_torchao):
    classes = fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    target = _FakeBaseTunerLayer()
    target.base_layer = _Layer(classes["AffineQuantizedTensor"]())
    assert isinstance(definer.dispatch_torchao(target, "default"), _FakeTorchaoLoraLinear)


def test_a_weightless_target_declines_like_upstream(peft_env, fake_torchao):
    fake_torchao()
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    assert definer.dispatch_torchao(object(), "default") is None


def test_is_torchao_available_is_still_honoured(peft_env, fake_torchao):
    classes = fake_torchao(affine = True, linear_activation = False)
    definer, _ = peft_env(_raiser(MISSING), torchao_available = False)
    FIX()
    target = _Layer(classes["AffineQuantizedTensor"]())
    assert definer.dispatch_torchao(target, "default") is None


# ---- what must still fail -------------------------------------------------


@pytest.mark.parametrize(
    "message",
    [
        # Half-installed torchao: calling this "no torchao weights here" would hide a broken install.
        "No module named 'torchao.quantization'",
        "No module named 'torchao'",
        # An extension built against a different torch/CUDA.
        "libtorchao_ops_cuda.so: cannot open shared object file: No such file or directory",
        "/site-packages/torchao/_C.so: undefined symbol: _ZN3c105ErrorC1E",
    ],
)
def test_a_broken_torchao_still_raises(peft_env, fake_torchao, message):
    fake_torchao()
    definer, _ = peft_env(_raiser(ImportError(message)))
    FIX()
    with pytest.raises(ImportError):
        definer.dispatch_torchao(_Layer("plain"), "default")


def test_a_torchao_that_vanishes_under_us_still_raises(peft_env, fake_torchao):
    # The dispatcher blamed the missing class, but the class lookup finds no torchao at all: that
    # is a broken install, not a newer one, and must not be reported as "no torchao weight here".
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    fake_torchao.absent()
    with pytest.raises(ImportError):
        definer.dispatch_torchao(_Layer("plain"), "default")


def test_a_non_import_error_still_raises(peft_env, fake_torchao):
    fake_torchao()
    definer, _ = peft_env(_raiser(RuntimeError("torchao exploded")))
    FIX()
    with pytest.raises(RuntimeError):
        definer.dispatch_torchao(_Layer("plain"), "default")


# ---- what must not change -------------------------------------------------


def test_a_working_dispatcher_still_returns_its_module(peft_env, fake_torchao):
    sentinel = object()
    fake_torchao(affine = True, linear_activation = True)

    def dispatch_torchao(target, adapter_name, **kwargs):
        return sentinel if target == "quantized" else None

    definer, _ = peft_env(dispatch_torchao)
    FIX()
    assert definer.dispatch_torchao("quantized", "default") is sentinel
    assert definer.dispatch_torchao("plain", "default") is None


def test_arguments_reach_the_original_untouched(peft_env, fake_torchao):
    seen = {}
    fake_torchao(affine = True, linear_activation = True)

    def dispatch_torchao(target, adapter_name, **kwargs):
        seen.update(target = target, adapter_name = adapter_name, kwargs = kwargs)
        return None

    definer, _ = peft_env(dispatch_torchao)
    FIX()
    definer.dispatch_torchao("target", "default", lora_config = "config", r = 8)
    assert seen == {
        "target": "target",
        "adapter_name": "default",
        "kwargs": {"lora_config": "config", "r": 8},
    }


def test_no_peft_is_not_an_error(monkeypatch):
    for k in [k for k in sys.modules if k.startswith("peft")]:
        monkeypatch.delitem(sys.modules, k, raising = False)
    import builtins

    real = builtins.__import__

    def no_peft(name, *a, **k):
        if name.startswith("peft"):
            raise ModuleNotFoundError("No module named 'peft'")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_peft)
    assert FIX() is None


def test_a_peft_without_the_dispatcher_is_not_an_error(monkeypatch):
    # Renamed, moved or removed upstream: nothing to wrap, and nothing to crash over.
    saved = {k: v for k, v in sys.modules.items() if k.startswith("peft")}
    for k in saved:
        monkeypatch.delitem(sys.modules, k, raising = False)
    pkg = types.ModuleType("peft")
    pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "peft", pkg)
    assert FIX() is False


def test_applying_twice_is_a_no_op(peft_env, fake_torchao):
    fake_torchao()
    definer, caller = peft_env(_raiser(MISSING))
    assert FIX() is True
    first = definer.dispatch_torchao
    assert FIX() is False, "already patched"
    assert definer.dispatch_torchao is first, "must not stack wrappers"
    assert caller.dispatch_torchao is first


def test_metadata_survives(peft_env, fake_torchao):
    fake_torchao()
    definer, _ = peft_env(_raiser(MISSING))
    FIX()
    assert definer.dispatch_torchao.__name__ == "dispatch_torchao"


# ---- against the peft and torchao that are actually installed -------------


def test_real_plain_lora_survives_this_torchao():
    """The user-visible failure: a plain 16-bit LoRA layer on a torchao without the class."""
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from peft import LoraConfig, get_peft_model

    model = torch.nn.Sequential()
    model.add_module("q_proj", torch.nn.Linear(8, 8, bias = False))

    assert FIX() in (True, False)
    peft_model = get_peft_model(model, LoraConfig(r = 4, target_modules = ["q_proj"]))
    layer = peft_model.base_model.model.q_proj
    assert "default" in layer.lora_A
    assert layer(torch.randn(2, 8)).shape == (2, 8)


def test_the_real_torchao_class_lookup_agrees_with_the_imports_peft_does():
    """Against whatever torchao is installed, the helper must mirror peft's two imports."""
    pytest.importorskip("torchao")
    ns = {}
    import ast
    import functools
    import importlib
    import inspect
    import re

    src = (REPO_ROOT / "unsloth" / "import_fixes.py").read_text(encoding = "utf-8")
    ns = {
        "functools": functools,
        "importlib": importlib,
        "inspect": inspect,
        "sys": sys,
        "re": re,
        "logger": types.SimpleNamespace(warning = lambda *a, **k: None),
    }
    for node in ast.parse(src).body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
        if name in _WANTED:
            exec(ast.get_source_segment(src, node), ns)

    classes, missing = ns["_peft_torchao_tensor_subclasses"]()
    expected = []
    for module_name, class_name in ns["_PEFT_TORCHAO_TENSOR_SUBCLASSES"]:
        try:
            getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError):
            continue
        expected.append(class_name)
    assert [cls.__name__ for cls in classes] == expected
    assert len(classes) + len(missing) == len(ns["_PEFT_TORCHAO_TENSOR_SUBCLASSES"])


# ---- wiring ---------------------------------------------------------------


def test_called_from_gpu_init():
    src = (REPO_ROOT / "unsloth" / "_gpu_init.py").read_text(encoding = "utf-8")
    assert "fix_peft_torchao_missing_tensor_subclass,\n" in src, "not imported"
    assert "\nfix_peft_torchao_missing_tensor_subclass()\n" in src, "not called"
    assert "\ndel fix_peft_torchao_missing_tensor_subclass\n" in src, "not cleaned up"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
