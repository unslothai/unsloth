import ast
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types

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


class _Fp8Owner(torch.nn.Module):
    def __init__(self, weight, quant_method = "fp8", weight_scale = None):
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
    model.fp8.weight_scale_inv = existing_inv
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

    assert restored == 0
    assert skipped == 1
    assert id(model.fp8.weight_scale_inv) == weight_scale_inv_id
    assert id(model.fp8.weight_scale) == weight_scale_id
    assert torch.equal(model.fp8.weight_scale, weight_scale)
    assert torch.equal(model.fp8.weight_scale_inv, existing_inv)


def test_loader_calls_restore_only_for_fp8_loads():
    tree = ast.parse(LOADER.read_text())
    hit_count = 0

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "load_in_fp8"
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.NotEq)
            and len(test.comparators) == 1
            and isinstance(test.comparators[0], ast.Constant)
            and test.comparators[0].value is False
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
