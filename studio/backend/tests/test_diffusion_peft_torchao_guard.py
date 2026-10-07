# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Studio's copy of the peft ``dispatch_torchao`` guard: the diffusion process never imports unsloth,
so without it peft <= 0.18 plus torchao 0.18 raises on every diffusion LoRA load."""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path

import pytest

import core.inference.diffusion_torchao_patches as patches

_REPO_ROOT = Path(__file__).resolve().parents[3]
_IMPORT_FIXES = _REPO_ROOT / "unsloth" / "import_fixes.py"
_PATCH_MODULE = Path(patches.__file__)

_REMOVED = "cannot import name 'LinearActivationQuantizedTensor' from 'torchao.quantization'"

# The Studio copy differs from unsloth's only in who the warning says is speaking.
_WORDING = (("Unsloth Studio: ", "Unsloth: "), ("Studio now runs", "Unsloth now runs"))


def _function_dump(path: Path, name: str) -> str:
    source = path.read_text(encoding = "utf-8")
    for studio, unsloth in _WORDING:
        source = source.replace(studio, unsloth)
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            if node.body and isinstance(node.body[0], ast.Expr):
                if isinstance(getattr(node.body[0], "value", None), ast.Constant):
                    node.body = node.body[1:]
            return ast.dump(node)
    raise AssertionError(f"{path} has no top-level def {name}")


@pytest.mark.parametrize(
    "name", ("_peft_torchao_tensor_subclasses", "_guard_peft_torchao_dispatcher")
)
def test_studio_copy_matches_unsloth_import_fixes(name):
    assert _function_dump(_PATCH_MODULE, name) == _function_dump(
        _IMPORT_FIXES, name
    ), f"{name} drifted from unsloth/import_fixes.py; port the change to both copies."


def _assigned_value_dump(path: Path, name: str) -> str:
    for node in ast.parse(path.read_text(encoding = "utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return ast.dump(node.value)
    raise AssertionError(f"{path} has no top-level {name}")


def test_studio_and_unsloth_match_the_same_torchao_removals():
    """The matcher is shared state too: when unsloth patched first, its matcher is the one that runs."""
    name = "_PEFT_TORCHAO_MISSING_TENSOR_SUBCLASS"
    assert _assigned_value_dump(_PATCH_MODULE, name) == _assigned_value_dump(_IMPORT_FIXES, name)


@pytest.fixture(autouse = True)
def _restore_peft_dispatchers(monkeypatch):
    """The patcher wraps every loaded peft copy, real ones included. Put them back afterwards so a
    later test of unsloth's own copy in this worker still finds an unpatched dispatcher."""
    for name, module in tuple(sys.modules.items()):
        if name.startswith("peft") and module is not None and hasattr(module, "dispatch_torchao"):
            monkeypatch.setattr(module, "dispatch_torchao", module.dispatch_torchao)
    yield


class _AffineQuantizedTensor:
    pass


class _TorchaoLoraLinear:
    def __init__(self, target, adapter_name, **kwargs):
        self.target, self.adapter_name, self.kwargs = target, adapter_name, kwargs


@pytest.fixture
def fake_peft(monkeypatch):
    """A torchao that lost LinearActivationQuantizedTensor, and a peft whose dispatcher imports it."""
    dtypes = types.ModuleType("torchao.dtypes")
    dtypes.AffineQuantizedTensor = _AffineQuantizedTensor
    quantization = types.ModuleType("torchao.quantization")
    torchao = types.ModuleType("torchao")
    torchao.dtypes, torchao.quantization = dtypes, quantization
    for name, module in (
        ("torchao", torchao),
        ("torchao.dtypes", dtypes),
        ("torchao.quantization", quantization),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    lora_torchao = types.ModuleType("peft.tuners.lora.torchao")
    namespace = {
        "is_torchao_available": lambda: True,
        "TorchaoLoraLinear": _TorchaoLoraLinear,
        "BaseTunerLayer": type("BaseTunerLayer", (), {}),
    }
    exec(
        "def dispatch_torchao(target, adapter_name, lora_config, **kwargs):\n"
        f"    raise ImportError({_REMOVED!r})\n",
        namespace,
    )
    lora_torchao.__dict__.update(namespace)
    lora_model = types.ModuleType("peft.tuners.lora.model")
    lora_model.dispatch_torchao = lora_torchao.dispatch_torchao
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.torchao", lora_torchao)
    monkeypatch.setitem(sys.modules, "peft.tuners.lora.model", lora_model)
    return lora_torchao, lora_model


def test_a_plain_linear_gets_no_torchao_layer_instead_of_an_import_error(fake_peft):
    lora_torchao, lora_model = fake_peft
    original = lora_torchao.dispatch_torchao
    with pytest.raises(ImportError):
        original(types.SimpleNamespace(weight = object()), "default", None)

    assert patches._patch_peft_torchao_dispatchers() is True
    # Both bindings point at one wrapper, so the list peft actually walks is the patched one.
    assert lora_model.dispatch_torchao is lora_torchao.dispatch_torchao is not original
    assert (
        lora_model.dispatch_torchao(types.SimpleNamespace(weight = object()), "default", None) is None
    )
    # Idempotent: a second pass finds nothing left to wrap.
    assert patches._patch_peft_torchao_dispatchers() is False


def test_a_weight_of_a_surviving_subclass_still_gets_the_torchao_layer(fake_peft):
    _, lora_model = fake_peft
    patches._patch_peft_torchao_dispatchers()
    layer = types.SimpleNamespace(weight = _AffineQuantizedTensor())
    built = lora_model.dispatch_torchao(layer, "default", None, r = 4)
    assert isinstance(built, _TorchaoLoraLinear)
    assert built.target is layer and built.adapter_name == "default" and built.kwargs == {"r": 4}


def test_an_unrelated_import_error_still_surfaces(monkeypatch, fake_peft):
    lora_torchao, lora_model = fake_peft

    def broken(target, adapter_name, lora_config, **kwargs):
        raise ImportError("libtorchao_ops.so: cannot open shared object file")

    monkeypatch.setattr(lora_torchao, "dispatch_torchao", broken)
    monkeypatch.setattr(lora_model, "dispatch_torchao", broken)
    patches._patch_peft_torchao_dispatchers()
    with pytest.raises(ImportError, match = "shared object"):
        lora_model.dispatch_torchao(types.SimpleNamespace(weight = object()), "default", None)


def test_a_torchao_without_the_dtypes_package_gets_a_plain_lora_layer(monkeypatch, fake_peft):
    """torchao main (after 0.18) deleted the whole ``torchao.dtypes`` package, so peft <= 0.18's
    first import inside ``dispatch_torchao`` is a ModuleNotFoundError naming the package, not the
    class. That is the same removal, and every LoRA target must still fall through to peft's
    ordinary layer instead of failing the load."""
    lora_torchao, lora_model = fake_peft
    monkeypatch.delitem(
        sys.modules, "torchao.dtypes"
    )  # the fake torchao has no __path__: the import fails
    monkeypatch.delattr(sys.modules["torchao"], "dtypes")
    monkeypatch.setattr(
        sys.modules["torchao"], "__spec__", importlib.machinery.ModuleSpec("torchao", None)
    )

    # peft <= 0.18's first torchao import, defined in the fake peft module so the guard reads its globals.
    namespace = dict(lora_torchao.__dict__)
    exec(
        "def dispatch_torchao(target, adapter_name, lora_config, **kwargs):\n"
        "    from torchao.dtypes import AffineQuantizedTensor\n",
        namespace,
    )
    removed = namespace["dispatch_torchao"]
    monkeypatch.setattr(lora_torchao, "dispatch_torchao", removed)
    monkeypatch.setattr(lora_model, "dispatch_torchao", removed)
    with pytest.raises(ModuleNotFoundError, match = "torchao.dtypes"):
        removed(types.SimpleNamespace(weight = object()), "default", None)
    patches._patch_peft_torchao_dispatchers()
    assert (
        lora_model.dispatch_torchao(types.SimpleNamespace(weight = object()), "default", None) is None
    )


@pytest.mark.parametrize(
    "message",
    (
        "No module named 'torchao'",
        "No module named 'torchao.dtypes.uintx'",
        "No module named 'torchao.dtypesx'",
    ),
)
def test_other_missing_modules_still_surface(monkeypatch, fake_peft, message):
    lora_torchao, lora_model = fake_peft

    def broken(target, adapter_name, lora_config, **kwargs):
        raise ModuleNotFoundError(message)

    monkeypatch.setattr(lora_torchao, "dispatch_torchao", broken)
    monkeypatch.setattr(lora_model, "dispatch_torchao", broken)
    patches._patch_peft_torchao_dispatchers()
    with pytest.raises(ModuleNotFoundError):
        lora_model.dispatch_torchao(types.SimpleNamespace(weight = object()), "default", None)


def test_the_guard_waits_for_peft_instead_of_importing_it(monkeypatch):
    monkeypatch.delitem(sys.modules, patches._PEFT_LORA_DISPATCH_MODULE, raising = False)
    monkeypatch.setattr(
        sys,
        "meta_path",
        [f for f in sys.meta_path if not getattr(f, patches._PEFT_TORCHAO_GUARD_SENTINEL, False)],
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a, **k: object() if name == "peft" else None
    )
    assert patches.install_peft_torchao_dispatch_guard() is True
    assert patches._PEFT_LORA_DISPATCH_MODULE not in sys.modules
    finders = [f for f in sys.meta_path if getattr(f, patches._PEFT_TORCHAO_GUARD_SENTINEL, False)]
    assert len(finders) == 1 and sys.meta_path[0] is finders[0]
    assert patches.install_peft_torchao_dispatch_guard() is False  # no second finder


def test_every_diffusion_entry_point_installs_the_guard(monkeypatch):
    """The int_mm installer is what diffusion.py, video.py, the transformer quantiser and the
    diffusion trainer already call at import, so the guard rides on it regardless of its switch."""
    calls = []
    monkeypatch.setattr(patches, "install_peft_torchao_dispatch_guard", lambda: calls.append(1))
    monkeypatch.setenv(patches._TORCHAO_INT_MM_ENV, "0")
    patches.install_torchao_int_mm_patch()
    assert calls == [1]


def test_real_peft_lora_injection_survives_the_installed_torchao():
    peft = pytest.importorskip("peft")
    pytest.importorskip("torchao")
    torch = pytest.importorskip("torch")
    patches.install_peft_torchao_dispatch_guard()
    importlib.import_module(patches._PEFT_LORA_DISPATCH_MODULE)
    net = torch.nn.Sequential(torch.nn.Linear(8, 8))
    peft.inject_adapter_in_model(peft.LoraConfig(r = 2, target_modules = ["0"]), net)
    assert hasattr(net[0], "lora_A")
