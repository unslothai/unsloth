# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Forces weights_only=True on torch.load in the two .pt2 export loader modules (CVE-2026-4538)."""

import importlib
import importlib.util
import io
import pathlib
import pickle
import sys
import types

import pytest

torch = pytest.importorskip("torch")

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SERIALIZE = "torch._export.serde.serialize"
_PACKAGE = "torch.export.pt2_archive._package"


def _load_import_fixes():
    # By path: importing the unsloth package needs an accelerator.
    spec = importlib.util.spec_from_file_location(
        "unsloth_import_fixes_under_test", _REPO_ROOT / "unsloth" / "import_fixes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


import_fixes = _load_import_fixes()


class NotATensor:
    pass


def _payload(obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    return buffer.getvalue()


def _load_from(module_name):
    """torch.load(weights_only=False) issued from inside `module_name`."""
    namespace = dict(vars(importlib.import_module(module_name)))
    namespace.update(torch = torch, io = io)
    code = (lambda raw: torch.load(io.BytesIO(raw), weights_only = False)).__code__
    return types.FunctionType(code, namespace)


@pytest.fixture
def guarded(monkeypatch):
    import torch._dynamo  # noqa: F401  # `import unsloth` always loads dynamo

    monkeypatch.delenv("UNSLOTH_ALLOW_UNSAFE_PT2_LOAD", raising = False)
    original_load = torch.load
    package = sys.modules.get(_PACKAGE)
    original_pickle = getattr(package, "pickle", None)
    original_meta_path = list(sys.meta_path)
    import_fixes.patch_torch_export_pt2_unsafe_load()
    yield
    torch.load = original_load
    sys.meta_path[:] = original_meta_path
    package = sys.modules.get(_PACKAGE)
    if package is not None and original_pickle is not None:
        package.pickle = original_pickle
    elif package is not None and hasattr(
        getattr(package, "pickle", None), "_unsloth_pt2_real_pickle"
    ):
        package.pickle = package.pickle._unsloth_pt2_real_pickle


def _unguarded_load():
    load = torch.load
    while getattr(load, "_unsloth_pt2_guard", False):
        load = load.__wrapped__
    return load


def test_unpatched_torch_unpickles_from_export_loader(monkeypatch):
    monkeypatch.setattr(torch, "load", _unguarded_load())
    loaded = _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))
    assert type(loaded["x"]).__name__ == "NotATensor"


def test_export_loader_non_tensor_payload_blocked(guarded):
    with pytest.raises(pickle.UnpicklingError, match = "UNSLOTH_ALLOW_UNSAFE_PT2_LOAD=1"):
        _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))


def test_export_loader_tensor_payload_still_loads(guarded):
    loaded = _load_from(_SERIALIZE)(_payload({"w": torch.ones(2)}))
    assert torch.equal(loaded["w"], torch.ones(2))


def test_opt_out_restores_torch_behaviour(guarded, monkeypatch):
    monkeypatch.setenv("UNSLOTH_ALLOW_UNSAFE_PT2_LOAD", "1")
    loaded = _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))
    assert type(loaded["x"]).__name__ == "NotATensor"


def test_unrelated_callers_unaffected(guarded):
    loaded = torch.load(io.BytesIO(_payload({"x": NotATensor()})), weights_only = False)
    assert type(loaded["x"]).__name__ == "NotATensor"


def test_patch_is_idempotent(guarded):
    first = torch.load
    package = sys.modules.get(_PACKAGE)
    first_pickle = getattr(package, "pickle", None)
    import_fixes.patch_torch_export_pt2_unsafe_load()
    assert torch.load is first
    assert getattr(package, "pickle", None) is first_pickle


def test_opaque_pickle_loads_blocked(guarded):
    package = sys.modules.get(_PACKAGE)
    if package is None or not hasattr(package, "pickle"):
        pytest.skip("this torch has no pt2_archive._package pickle use")
    assert package.pickle.dumps({"a": 1})
    with pytest.raises(pickle.UnpicklingError, match = "CVE-2026-4538"):
        package.pickle.loads(pickle.dumps({"a": 1}))


def _rng_style_wrapper(inner, flag):
    # Stands in for patch_unsafe_trainer_rng_load's torch.load shim, which lives in unsloth.*.
    namespace = {"__name__": "unsloth.import_fixes", "inner": inner}
    exec("def wrapper(*args, **kwargs):\n    return inner(*args, **kwargs)\n", namespace)
    wrapper = namespace["wrapper"]
    wrapper._unsloth_rng_guard = True
    wrapper._unsloth_rng_flag = flag
    return wrapper


def test_rng_guard_composes_in_either_order(guarded):
    # The rng guard (CVE-2026-1839) keys its idempotency on markers on torch.load.
    flag = object()
    guarded_load = torch.load
    torch.load = _rng_style_wrapper(guarded_load, flag)
    with pytest.raises(pickle.UnpicklingError):
        _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))
    torch.load = _rng_style_wrapper(_unguarded_load(), flag)
    import_fixes.patch_torch_export_pt2_unsafe_load()
    assert torch.load._unsloth_pt2_guard and torch.load._unsloth_rng_flag is flag
    with pytest.raises(pickle.UnpicklingError):
        _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))


def test_export_roundtrip_and_compile_unchanged(guarded, tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3)
            self.register_buffer("scale", torch.full((3,), 2.0))

        def forward(self, x):
            return self.linear(x) * self.scale + 1

    model, x = Model(), torch.randn(2, 4)
    path = tmp_path / "model.pt2"
    torch.export.save(torch.export.export(model, (x,)), str(path))
    reloaded = torch.export.load(str(path))
    assert torch.allclose(reloaded.module()(x), model(x))
    compiled = torch.compile(lambda a: torch.sin(a) * 2 + 1, backend = "eager")
    assert torch.allclose(compiled(x), torch.sin(x) * 2 + 1)


def test_outer_torch_load_wrapper_does_not_hide_export_caller(guarded):
    # Another library wrapping torch.load after Unsloth adds a frame between loader and guard.
    inner = torch.load

    def outer(*args, **kwargs):
        return inner(*args, **kwargs)

    torch.load = outer
    with pytest.raises(pickle.UnpicklingError, match = "UNSLOTH_ALLOW_UNSAFE_PT2_LOAD=1"):
        _load_from(_SERIALIZE)(_payload({"x": NotATensor()}))


def test_package_imported_after_patch_is_hardened(guarded):
    if importlib.util.find_spec("torch.export.pt2_archive") is None:
        pytest.skip("this torch has no pt2_archive package")
    parent = importlib.import_module("torch.export.pt2_archive")
    saved_module = sys.modules.pop(_PACKAGE, None)
    saved_attr = parent.__dict__.pop("_package", None)
    try:
        import_fixes.patch_torch_export_pt2_unsafe_load()
        module = importlib.import_module(_PACKAGE)
        if not isinstance(getattr(module, "pickle", None), types.ModuleType):
            pytest.skip("this torch's _package does not use pickle")
        assert isinstance(module.pickle, import_fixes._Pt2PickleModule)
        with pytest.raises(pickle.UnpicklingError):
            module.pickle.loads(pickle.dumps(1))
    finally:
        if saved_module is not None:
            sys.modules[_PACKAGE] = saved_module
        if saved_attr is not None:
            parent._package = saved_attr
