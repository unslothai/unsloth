import importlib
import importlib.util
import sys
from pathlib import Path

import pytest


IMPORT_FIXES_PATH = Path(__file__).resolve().parents[1] / "unsloth" / "import_fixes.py"


def _load_import_fixes_module(module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, IMPORT_FIXES_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_causal_conv1d_package(tmp_path: Path, import_error_message: str):
    package_path = tmp_path / "causal_conv1d"
    package_path.mkdir()
    (package_path / "__init__.py").write_text(
        f"raise ImportError({import_error_message!r})\n",
        encoding = "utf-8",
    )


@pytest.fixture(autouse = True)
def _restore_import_state():
    original_find_spec = importlib.util.find_spec
    original_meta_path = list(sys.meta_path)
    original_sys_path = list(sys.path)
    original_causal_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "causal_conv1d" or name.startswith("causal_conv1d.")
    }

    yield

    importlib.util.find_spec = original_find_spec
    sys.meta_path[:] = original_meta_path
    sys.path[:] = original_sys_path

    for name in list(sys.modules):
        if name == "causal_conv1d" or name.startswith("causal_conv1d."):
            sys.modules.pop(name, None)

    for name, module in original_causal_modules.items():
        sys.modules[name] = module


def test_disable_broken_causal_conv1d_blocks_imports_and_masks_find_spec(tmp_path):
    _create_causal_conv1d_package(
        tmp_path,
        "causal_conv1d_cuda.so: undefined symbol: _ZN3c103hip28c10_hip_check_implementationEiPKcS2_ib",
    )
    sys.path.insert(0, str(tmp_path))

    import_fixes = _load_import_fixes_module("import_fixes_test_broken")
    import_fixes.disable_broken_causal_conv1d()

    assert import_fixes.CAUSAL_CONV1D_BROKEN is True
    assert importlib.util.find_spec("causal_conv1d") is None
    assert importlib.util.find_spec("causal_conv1d.causal_conv1d_interface") is None

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("causal_conv1d")

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("causal_conv1d.cpp_functions")

    with pytest.raises(ModuleNotFoundError):
        exec("from causal_conv1d import *", {})


def test_disable_broken_causal_conv1d_ignores_non_symbol_errors(tmp_path):
    _create_causal_conv1d_package(
        tmp_path, "causal_conv1d import failed for another reason"
    )
    sys.path.insert(0, str(tmp_path))

    import_fixes = _load_import_fixes_module("import_fixes_test_not_broken")
    import_fixes.disable_broken_causal_conv1d()

    assert import_fixes.CAUSAL_CONV1D_BROKEN is False
    assert importlib.util.find_spec("causal_conv1d") is not None
    assert not any(
        getattr(finder, "_unsloth_causal_conv1d_blocker", False)
        for finder in sys.meta_path
    )
