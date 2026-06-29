import importlib.machinery
import importlib.util
from pathlib import Path
import sys
import types

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
IMPORT_FIXES = REPO_ROOT / "unsloth" / "import_fixes.py"


def _load_import_fixes():
    spec = importlib.util.spec_from_file_location("_unsloth_import_fixes_peft_tp", IMPORT_FIXES)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_module(name, *, is_package = False, attrs = None):
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader = None, is_package = is_package)
    if is_package:
        module.__path__ = []
    if attrs:
        for k, v in attrs.items():
            setattr(module, k, v)
    sys.modules[name] = module
    return module


@pytest.fixture(autouse = True)
def _restore_import_fixtures():
    keep = {
        "transformers",
        "transformers.integrations",
        "transformers.integrations.tensor_parallel",
        "peft",
        "peft.utils",
        "peft.utils.save_and_load",
    }
    snapshot = {name: sys.modules.get(name) for name in keep}
    yield
    for name, value in snapshot.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _install_fake_transformers_tensor_parallel(existing):
    transformers = _install_fake_module("transformers", is_package = True)
    integrations = _install_fake_module("transformers.integrations", is_package = True)
    setattr(transformers, "integrations", integrations)

    tp = _install_fake_module(
        "transformers.integrations.tensor_parallel",
        attrs = existing,
    )
    setattr(integrations, "tensor_parallel", tp)
    return tp


def test_missing_tensor_parallel_symbol_import_succeeds_after_fix(monkeypatch):
    module = _load_import_fixes()

    tp_mod = _install_fake_transformers_tensor_parallel(
        {
            "ColwiseParallel": object,
            "RowwiseParallel": object,
        }
    )

    monkeypatch.setattr(
        module,
        "_extract_peft_tensor_parallel_imported_symbols",
        lambda: ("ALL_PARALLEL_STYLES", "ColwiseParallel", "EmbeddingParallel", "RowwiseParallel"),
    )

    assert module.fix_peft_transformers_tensor_parallel_import_compat() is True

    import transformers.integrations.tensor_parallel as patched
    from transformers.integrations.tensor_parallel import EmbeddingParallel

    assert patched == sys.modules["transformers.integrations.tensor_parallel"]
    assert EmbeddingParallel is getattr(patched, "EmbeddingParallel")
    assert getattr(EmbeddingParallel, "__unsloth_stub__", False)


def test_existing_embedding_parallel_is_not_replaced(monkeypatch):
    module = _load_import_fixes()

    class RealEmbeddingParallel:
        pass

    tp_mod = _install_fake_transformers_tensor_parallel(
        {
            "EmbeddingParallel": RealEmbeddingParallel,
            "ColwiseParallel": object,
        }
    )

    monkeypatch.setattr(
        module,
        "_extract_peft_tensor_parallel_imported_symbols",
        lambda: ("ALL_PARALLEL_STYLES", "ColwiseParallel", "EmbeddingParallel", "RowwiseParallel"),
    )
    assert module.fix_peft_transformers_tensor_parallel_import_compat() is True

    assert tp_mod.EmbeddingParallel is RealEmbeddingParallel
    assert not getattr(tp_mod.EmbeddingParallel, "__unsloth_stub__", False)


def test_missing_tensor_parallel_module_is_not_created(monkeypatch):
    module = _load_import_fixes()
    previous_tp_module = sys.modules.pop("transformers.integrations.tensor_parallel", None)

    original_find_spec = module.importlib.util.find_spec
    def fake_find_spec(name):
        if name == "transformers.integrations.tensor_parallel":
            return None
        return original_find_spec(name)

    monkeypatch.setattr(module.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        module,
        "_extract_peft_tensor_parallel_imported_symbols",
        lambda: ("EmbeddingParallel",),
    )

    assert module.fix_peft_transformers_tensor_parallel_import_compat() is None

    try:
        spec = importlib.util.find_spec("transformers.integrations.tensor_parallel")
    except ModuleNotFoundError as exc:
        assert exc.name == "transformers"
    else:
        assert spec is None
    assert "transformers.integrations.tensor_parallel" not in sys.modules


def test_placeholder_raises_on_real_use(monkeypatch):
    module = _load_import_fixes()

    tp_mod = _install_fake_transformers_tensor_parallel({})
    monkeypatch.setattr(
        module,
        "_extract_peft_tensor_parallel_imported_symbols",
        lambda: ("EmbeddingParallel",),
    )
    assert module.fix_peft_transformers_tensor_parallel_import_compat() is True

    with pytest.raises(NotImplementedError, match = "EmbeddingParallel"):
        tp_mod.EmbeddingParallel()

    assert getattr(tp_mod.EmbeddingParallel, "__unsloth_stub__", False)
