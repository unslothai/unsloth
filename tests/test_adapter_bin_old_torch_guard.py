# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""torch < 2.6 runs code from a .bin even with weights_only=True (CVE-2025-32434): adapter /
checkpoint .bin reads must be refused there, before torch.load, and stay unchanged on >= 2.6."""

import ast
import importlib.util
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_import_fixes():
    # By path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", _REPO_ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import_fixes = _load_import_fixes()


def _function(relpath, name, package):
    """Exec one top-level function out of a module that cannot be imported without a GPU."""
    source = (_REPO_ROOT / relpath).read_text(encoding = "utf-8")
    node = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    namespace = {
        "torch": torch,
        "os": __import__("os"),
        "__package__": package,
        "__name__": f"{package}._under_test",
    }
    exec(compile(ast.Module([node], []), relpath, "exec"), namespace)
    return namespace[name]


@pytest.fixture
def stub_unsloth(monkeypatch):
    for name in ("unsloth", "unsloth.models"):
        if name not in sys.modules:
            module = types.ModuleType(name)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setitem(sys.modules, "unsloth.import_fixes", import_fixes)


@pytest.fixture
def bin_file(tmp_path):
    path = tmp_path / "adapter_model.bin"
    torch.save({"base_model.model.lm_head.lora_B.weight": torch.zeros(4, 2)}, path)
    return str(path)


@pytest.fixture
def load_spy(monkeypatch):
    calls = []
    real = torch.load

    def spy(*args, **kwargs):
        calls.append(args[0] if args else kwargs.get("f"))
        return real(*args, **kwargs)

    monkeypatch.setattr(torch, "load", spy)
    return calls


SITES = [
    ("unsloth/models/loader.py", "_adapter_file_shapes", "unsloth.models", lambda f, p: f(p)),
    ("unsloth/models/loader_utils.py", "_bin_dtypes", "unsloth.models", lambda f, p: f(p)),
]


@pytest.mark.parametrize("relpath,name,package,call", SITES, ids = [s[1] for s in SITES])
def test_unsloth_bin_reads_refused_on_old_torch(
    monkeypatch, stub_unsloth, bin_file, load_spy, relpath, name, package, call
):
    fn = _function(relpath, name, package)
    monkeypatch.setattr(torch, "__version__", "2.5.1+cu124")
    with pytest.raises(RuntimeError, match = "CVE-2025-32434"):
        call(fn, bin_file)
    assert load_spy == [], "torch.load ran on torch < 2.6"


@pytest.mark.parametrize("relpath,name,package,call", SITES, ids = [s[1] for s in SITES])
def test_unsloth_bin_reads_unchanged_on_new_torch(
    monkeypatch, stub_unsloth, bin_file, relpath, name, package, call
):
    fn = _function(relpath, name, package)
    monkeypatch.setattr(torch, "__version__", "2.6.0+cu124")
    assert call(fn, bin_file)


def test_adapter_fits_text_model_source_guards_bin():
    # _adapter_fits_text_model swallows errors, so check its .bin branch is guarded at the source.
    source = (_REPO_ROOT / "unsloth/models/_utils.py").read_text(encoding = "utf-8")
    start = source.index("def _adapter_fits_text_model(")
    body = source[start : source.index("\ndef ", start + 1)]
    load = body.index("torch.load(path")
    assert "refuse_pickle_checkpoint_on_old_torch(path)" in body[:load]


def _peft_bin_dir(tmp_path):
    peft = pytest.importorskip("peft")
    folder = tmp_path / "lora"
    folder.mkdir()
    torch.save(
        {"base_model.model.x.lora_A.weight": torch.zeros(2, 2)}, folder / "adapter_model.bin"
    )
    return peft, str(folder)


@pytest.fixture
def restore_peft_torch_load():
    peft_sl = pytest.importorskip("peft.utils.save_and_load")
    original = peft_sl.torch_load
    yield
    peft_sl.torch_load = original


def test_peft_adapter_bin_refused_on_old_torch(
    monkeypatch, tmp_path, load_spy, restore_peft_torch_load
):
    peft, folder = _peft_bin_dir(tmp_path)
    monkeypatch.setattr(torch, "__version__", "2.5.1")
    patch = getattr(import_fixes, "patch_peft_bin_load_old_torch", lambda: None)
    patch()
    patch()  # idempotent
    from peft.utils import load_peft_weights

    with pytest.raises(RuntimeError, match = "adapter_model.safetensors"):
        load_peft_weights(folder, device = "cpu")
    assert load_spy == [], "PEFT torch.load ran on torch < 2.6"


def test_peft_untouched_on_new_torch(monkeypatch, tmp_path, restore_peft_torch_load):
    peft, folder = _peft_bin_dir(tmp_path)
    import peft.utils.save_and_load as peft_sl

    before = peft_sl.torch_load
    monkeypatch.setattr(torch, "__version__", "2.6.0")
    getattr(import_fixes, "patch_peft_bin_load_old_torch", lambda: None)()
    assert peft_sl.torch_load is before
    from peft.utils import load_peft_weights

    assert "base_model.model.x.lora_A.weight" in load_peft_weights(folder, device = "cpu")


def test_peft_guard_installed_after_peft_import_shims():
    # The guard returns silently when peft.utils.save_and_load cannot import, so it must run after the shims that
    # make PEFT importable on older transformers.
    import pathlib

    calls = [
        line.strip()
        for line in (pathlib.Path(__file__).resolve().parents[1] / "unsloth" / "_gpu_init.py")
        .read_text()
        .splitlines()
        if line.strip().endswith("()") and not line.startswith((" ", "\t", "#"))
    ]
    guard = calls.index("patch_peft_bin_load_old_torch()")
    for shim in (
        "fix_peft_transformers_tensor_parallel_import_compat()",
        "fix_peft_transformers_weight_conversion_import()",
    ):
        assert calls.index(shim) < guard, shim
