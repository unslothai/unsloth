import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest
import torch
from safetensors.torch import save_file


REPO_ROOT = Path(__file__).resolve().parents[1]
LOADER_UTILS = REPO_ROOT / "unsloth" / "models" / "loader_utils.py"
LOADER = REPO_ROOT / "unsloth" / "models" / "loader.py"


def _install_stub_package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module
    return module


@pytest.fixture(autouse = True)
def _restore_loader_utils_modules():
    keep = {
        "unsloth",
        "unsloth.models",
        "unsloth.models.loader_utils",
        "unsloth.device_type",
        "unsloth.models.mapper",
        "unsloth.models._utils",
        "unsloth_zoo",
        "unsloth_zoo.utils",
    }
    snapshot = {name: sys.modules.get(name) for name in keep}
    yield
    for name, value in snapshot.items():
        if value is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = value


def _install_loader_utils_stubs():
    _install_stub_package("unsloth", REPO_ROOT / "unsloth")
    _install_stub_package("unsloth.models", REPO_ROOT / "unsloth" / "models")
    _install_stub_package("unsloth_zoo", REPO_ROOT)

    device_type = types.ModuleType("unsloth.device_type")
    device_type.DEVICE_TYPE_TORCH = "cuda"
    sys.modules["unsloth.device_type"] = device_type

    mapper = types.ModuleType("unsloth.models.mapper")
    mapper.INT_TO_FLOAT_MAPPER = {}
    mapper.FLOAT_TO_INT_MAPPER = {}
    mapper.MAP_TO_UNSLOTH_16bit = {}
    mapper.FLOAT_TO_FP8_BLOCK_MAPPER = {}
    mapper.FLOAT_TO_FP8_ROW_MAPPER = {}
    sys.modules["unsloth.models.mapper"] = mapper

    model_utils = types.ModuleType("unsloth.models._utils")
    model_utils.TorchAOConfig = type("TorchAOConfig", (), {})
    sys.modules["unsloth.models._utils"] = model_utils

    zoo_utils = types.ModuleType("unsloth_zoo.utils")
    zoo_utils.Version = lambda value: value
    zoo_utils.get_quant_type = lambda config: None
    sys.modules["unsloth_zoo.utils"] = zoo_utils


def _load_loader_utils():
    _install_loader_utils_stubs()
    sys.modules.pop("unsloth.models.loader_utils", None)
    spec = importlib.util.spec_from_file_location("unsloth.models.loader_utils", LOADER_UTILS)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _make_checkpoint(path, tensors):
    full_path = os.path.join(path, "model-00001-of-00001.safetensors")
    save_file(tensors, full_path)
    return path


def _make_bin_checkpoint(path, tensors):
    full_path = os.path.join(path, "pytorch_model-00001-of-00001.bin")
    torch.save(tensors, full_path)
    return path


class _Fp8Owner(torch.nn.Module):
    def __init__(
        self,
        weight,
        quant_method = "fp8",
        weight_scale = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.quant_method = quant_method
        if weight_scale is not None:
            self.weight_scale = weight_scale


def test_restores_missing_fp8_weight_scale_inv():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16))

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([1.25], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([1.25], dtype = torch.float16))
    assert "fp8.weight_scale_inv" in model.state_dict()
    assert torch.equal(
        model.state_dict()["fp8.weight_scale_inv"],
        torch.tensor([1.25], dtype = torch.float16),
    )
    assert not hasattr(model.fp8, "weight_scale")


def test_does_not_attach_scale_inv_to_non_fp8_module():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.linear = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16), quant_method = "int8")

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"linear.weight_scale_inv": torch.tensor([0.5], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 0
    assert skipped == 1
    assert not hasattr(model.linear, "weight_scale_inv")


def test_preserves_existing_scale_state():
    loader_utils = _load_loader_utils()
    weight_scale = torch.tensor([3.0], dtype = torch.float16)
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(
        weight = torch.randn(4, 4, dtype = torch.float16),
        weight_scale = weight_scale,
    )
    existing_inv = torch.tensor([5.0], dtype = torch.float16)
    model.fp8.register_buffer("weight_scale_inv", existing_inv)
    weight_scale_id = id(model.fp8.weight_scale)
    weight_scale_inv_id = id(model.fp8.weight_scale_inv)

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([1.0], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert id(model.fp8.weight_scale) == weight_scale_id
    assert torch.equal(model.fp8.weight_scale, weight_scale)
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([1.0], dtype = torch.float16))
    assert "fp8.weight_scale_inv" in model.state_dict()
    assert torch.equal(
        model.state_dict()["fp8.weight_scale_inv"],
        torch.tensor([1.0], dtype = torch.float16),
    )


class _PackedFp8Owner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.ones(4, 4, dtype = torch.int8)
        self.quant_method = "fp8"


def test_preserves_checkpoint_scale_dtype_for_packed_fp8_weights():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([1.25], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert model.fp8.weight_scale_inv.dtype == torch.float32
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([1.25], dtype = torch.float32))


def test_restores_missing_fp8_weight_scale_inv_from_bin_checkpoint():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_bin_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([2.5], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([2.5], dtype = torch.float32))


def test_loader_detects_direct_or_requested_fp8_restore_paths():
    tree = ast.parse(LOADER.read_text())
    hit_count = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.BoolOp)
            and isinstance(test.op, ast.Or)
            and len(test.values) == 2
            and isinstance(test.values[0], ast.Name)
            and test.values[0].id == "restore_fp8_scales"
            and isinstance(test.values[1], ast.Call)
            and isinstance(test.values[1].func, ast.Name)
            and test.values[1].func.id == "_model_has_real_fp8_modules"
        ):
            continue

        call_names = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                call_names.add(child.func.id)
        if {
            "_restore_missing_fp8_weight_scale_inv",
            "_tag_model_with_fp8_torchao_config",
        }.issubset(call_names):
            hit_count += 1

    assert hit_count == 2


def test_loader_restores_fp8_scales_with_base_revision_for_peft():
    tree = ast.parse(LOADER.read_text())
    hit_count = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (
            not isinstance(node.func, ast.Name)
            or node.func.id != "_restore_missing_fp8_weight_scale_inv"
        ):
            continue

        revision_kw = next((kw for kw in node.keywords if kw.arg == "revision"), None)
        if revision_kw is None:
            continue
        value = revision_kw.value
        if not (
            isinstance(value, ast.IfExp)
            and isinstance(value.test, ast.UnaryOp)
            and isinstance(value.test.op, ast.Not)
            and isinstance(value.test.operand, ast.Name)
            and value.test.operand.id == "is_peft"
            and isinstance(value.body, ast.Name)
            and value.body.id == "revision"
            and isinstance(value.orelse, ast.Constant)
            and value.orelse.value is None
        ):
            continue
        hit_count += 1

    assert hit_count == 2


def test_fp8_probe_skips_missing_optional_kernel_module():
    loader_utils = _load_loader_utils()
    module = torch.nn.Linear(4, 4, bias = False)

    assert loader_utils._is_real_fp8_owner(module) is False


def test_model_has_real_fp8_modules_detects_nested_fp8_layers():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    assert loader_utils._model_has_real_fp8_modules(model) is True


def test_fp8_probe_uses_absolute_optional_import_and_narrow_missing_module_guard():
    tree = ast.parse(LOADER_UTILS.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "_is_real_fp8_owner":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Try):
                continue

            imports = [stmt for stmt in child.body if isinstance(stmt, ast.ImportFrom)]
            handlers = child.handlers
            if not imports or not handlers:
                continue

            import_stmt = imports[0]
            handler = handlers[0]
            assert import_stmt.module == "unsloth.kernels.fp8"
            assert isinstance(handler.type, ast.Name)
            assert handler.type.id == "ModuleNotFoundError"
            return

    pytest.fail("_is_real_fp8_owner no longer protects the optional fp8 kernel import.")
