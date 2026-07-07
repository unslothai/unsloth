import ast
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types

import pytest
import safetensors.torch
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
    model_utils._apply_text_only_key_mapping = lambda *args, **kwargs: None
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


def _make_checkpoint(
    path,
    tensors,
    filename = "model.safetensors",
):
    full_path = os.path.join(path, filename)
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


def test_skips_bin_checkpoint_without_torch_load(monkeypatch):
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_bin_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([2.5], dtype = torch.float32)},
        )
        monkeypatch.setattr(
            loader_utils.torch,
            "load",
            lambda *args, **kwargs: pytest.fail("torch.load should not run"),
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 0
    assert skipped == 0
    assert not hasattr(model.fp8, "weight_scale_inv")


def test_variant_limits_scale_restore_to_selected_safetensors():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([1.0], dtype = torch.float32)},
            filename = "model.safetensors",
        )
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([3.0], dtype = torch.float32)},
            filename = "model.fp8.safetensors",
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
            variant = "fp8",
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([3.0], dtype = torch.float32))


def test_subfolder_limits_scale_restore_to_selected_safetensors():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([1.0], dtype = torch.float32)},
        )
        subfolder = Path(checkpoint_dir) / "nested"
        subfolder.mkdir()
        _make_checkpoint(
            subfolder,
            {"fp8.weight_scale_inv": torch.tensor([4.0], dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
            subfolder = "nested",
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([4.0], dtype = torch.float32))


def test_safetensors_index_opens_only_scale_shards(monkeypatch):
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()
    opened = []

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([5.0], dtype = torch.float32)},
            filename = "scale.safetensors",
        )
        _make_checkpoint(
            checkpoint_dir,
            {"other.weight": torch.tensor([9.0], dtype = torch.float32)},
            filename = "other.safetensors",
        )
        index = {
            "weight_map": {
                "fp8.weight_scale_inv": "scale.safetensors",
                "other.weight": "other.safetensors",
            }
        }
        (Path(checkpoint_dir) / "model.safetensors.index.json").write_text(json.dumps(index))

        def recording_safe_open(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_safe_open(path, *args, **kwargs)

        original_safe_open = safetensors.torch.safe_open
        monkeypatch.setattr(safetensors.torch, "safe_open", recording_safe_open)
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert opened == ["scale.safetensors"]


def test_variant_safetensors_index_uses_transformers_name(monkeypatch):
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()
    opened = []

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.tensor([6.0], dtype = torch.float32)},
            filename = "scale.fp8.safetensors",
        )
        index = {"weight_map": {"fp8.weight_scale_inv": "scale.fp8.safetensors"}}
        (Path(checkpoint_dir) / "model.safetensors.index.fp8.json").write_text(json.dumps(index))

        def recording_safe_open(path, *args, **kwargs):
            opened.append(Path(path).name)
            return original_safe_open(path, *args, **kwargs)

        original_safe_open = safetensors.torch.safe_open
        monkeypatch.setattr(safetensors.torch, "safe_open", recording_safe_open)
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
            variant = "fp8",
        )

    assert restored == 1
    assert skipped == 0
    assert opened == ["scale.fp8.safetensors"]
    assert torch.equal(model.fp8.weight_scale_inv, torch.tensor([6.0], dtype = torch.float32))


def test_skips_meta_device_scale_restore():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(
        weight = torch.empty(4, 4, device = "meta"), weight_scale = torch.empty(1, device = "meta")
    )

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
    assert not hasattr(model.fp8, "weight_scale_inv")


def test_skips_shape_incompatible_local_scale():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(
        weight = torch.randn(4, 4, dtype = torch.float16),
        weight_scale = torch.ones(2, dtype = torch.float16),
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.ones(4, dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 0
    assert skipped == 1
    assert not hasattr(model.fp8, "weight_scale_inv")


def test_reshapes_same_numel_local_scale_placeholder():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(
        weight = torch.randn(4, 4, dtype = torch.float16),
        weight_scale = torch.ones((1, 4), dtype = torch.float16),
    )

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.arange(4, dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert model.fp8.weight_scale_inv.shape == (1, 4)
    assert torch.equal(
        model.fp8.weight_scale_inv, torch.arange(4, dtype = torch.float16).reshape(1, 4)
    )


def test_restores_multielement_scale_without_local_placeholder():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16))

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.ones((2, 1), dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale_inv, torch.ones((2, 1), dtype = torch.float16))


def test_skips_oversized_scale_without_local_placeholder():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16))

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale_inv": torch.ones((8, 1), dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 0
    assert skipped == 1
    assert not hasattr(model.fp8, "weight_scale_inv")


def test_restores_fbgemm_weight_scale_tensor():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16))

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {"fp8.weight_scale": torch.full((2, 1), 2.0, dtype = torch.float32)},
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 1
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale, torch.full((2, 1), 2.0, dtype = torch.float16))


def test_restore_accepts_cache_download_selection_kwargs():
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
            cache_dir = checkpoint_dir,
            force_download = False,
        )

    assert restored == 1
    assert skipped == 0


class _Fp8Expert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj_scale_inv = torch.ones(2, dtype = torch.float16)
        self.up_proj_scale = torch.ones(2, dtype = torch.float16)
        self.gate_up_proj_scale_inv = torch.ones(2, dtype = torch.float16)
        self.down_proj_scale_inv = torch.ones(2, dtype = torch.float16)
        self.gate_up_proj_scale = torch.ones(2, dtype = torch.float16)
        self.down_proj_scale = torch.ones(2, dtype = torch.float16)


def test_restores_fp8_expert_scale_tensors():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.expert = _Fp8Expert()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {
                "expert.gate_up_proj_scale_inv": torch.full((2,), 2.0, dtype = torch.float32),
                "expert.down_proj_scale_inv": torch.full((2,), 3.0, dtype = torch.float32),
            },
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 2
    assert skipped == 0
    assert torch.equal(
        model.expert.gate_up_proj_scale_inv, torch.full((2,), 2.0, dtype = torch.float16)
    )
    assert torch.equal(model.expert.down_proj_scale_inv, torch.full((2,), 3.0, dtype = torch.float16))


def test_restores_fbgemm_expert_scale_tensors():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.expert = _Fp8Expert()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {
                "expert.gate_up_proj_scale": torch.full((2,), 4.0, dtype = torch.float32),
                "expert.down_proj_scale": torch.full((2,), 5.0, dtype = torch.float32),
            },
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 2
    assert skipped == 0
    assert torch.equal(model.expert.gate_up_proj_scale, torch.full((2,), 4.0, dtype = torch.float16))
    assert torch.equal(model.expert.down_proj_scale, torch.full((2,), 5.0, dtype = torch.float16))


def test_restores_fp8_scale_alias_tensors():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _Fp8Owner(weight = torch.randn(4, 4, dtype = torch.float16))
    model.expert = _Fp8Expert()

    with tempfile.TemporaryDirectory() as checkpoint_dir:
        _make_checkpoint(
            checkpoint_dir,
            {
                "fp8.scale": torch.full((2, 1), 6.0, dtype = torch.float32),
                "expert.up_proj_scale": torch.full((2,), 7.0, dtype = torch.float32),
                "expert.up_proj_scale_inv": torch.full((2,), 8.0, dtype = torch.float32),
            },
        )
        restored, skipped = loader_utils._restore_missing_fp8_weight_scale_inv(
            model,
            model_name = checkpoint_dir,
            local_files_only = True,
        )

    assert restored == 3
    assert skipped == 0
    assert torch.equal(model.fp8.weight_scale, torch.full((2, 1), 6.0, dtype = torch.float16))
    assert torch.equal(model.expert.up_proj_scale, torch.full((2,), 7.0, dtype = torch.float16))
    assert torch.equal(model.expert.up_proj_scale_inv, torch.full((2,), 8.0, dtype = torch.float16))


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


def test_loader_passes_selected_artifact_knobs_to_fp8_restore():
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
        keyword_names = {kw.arg for kw in node.keywords}
        if {
            "subfolder",
            "variant",
            "use_safetensors",
            "cache_dir",
            "force_download",
        }.issubset(keyword_names):
            hit_count += 1

    assert hit_count == 2


def test_fp8_owner_probe_tolerates_optional_kernel_import_errors():
    tree = ast.parse(LOADER_UTILS.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        if not any(
            isinstance(child, ast.ImportFrom) and child.module == "unsloth.kernels.fp8"
            for child in node.body
        ):
            continue
        assert any(
            handler.type is not None
            and isinstance(handler.type, ast.Name)
            and handler.type.id == "Exception"
            for handler in node.handlers
        )
        return

    raise AssertionError("FP8 owner probe import guard not found")


def test_fastmodel_peft_base_mapping_forwards_load_in_fp8():
    tree = ast.parse(LOADER.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not isinstance(node.test, ast.Name) or node.test.id != "is_peft":
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            if not any(
                isinstance(target, ast.Name) and target.id == "model_name"
                for target in child.targets
            ):
                continue
            value = child.value
            if not isinstance(value, ast.Call):
                continue
            if not isinstance(value.func, ast.Name) or value.func.id != "get_model_name":
                continue
            keyword_names = {kw.arg for kw in value.keywords}
            if {"load_in_4bit", "load_in_fp8"}.issubset(keyword_names):
                return

    pytest.fail("FastModel PEFT base mapping must forward load_in_fp8 to get_model_name.")


def test_offline_fp8_cache_uses_safe_serialization():
    tree = ast.parse(LOADER_UTILS.read_text())

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "save_pretrained":
            continue
        for kw in node.keywords:
            if kw.arg == "safe_serialization" and isinstance(kw.value, ast.Constant):
                assert kw.value.value is True
                return

    pytest.fail("_offline_quantize_to_fp8 must save safetensors caches.")


def test_offline_fp8_cache_regenerates_bin_only_cache(monkeypatch):
    loader_utils = _load_loader_utils()

    class DummyConfig:
        architectures = ["DummyForCausalLM"]

    class DummyModel:
        def save_pretrained(
            self,
            path,
            safe_serialization = False,
        ):
            assert safe_serialization is True
            os.makedirs(path, exist_ok = True)
            Path(path, "model.safetensors").write_text("safe")

    class DummyTokenizer:
        def save_pretrained(self, path):
            Path(path, "tokenizer.json").write_text("{}")

    class DummyAutoModel:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyModel()

    class DummyTokenizerLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyTokenizer()

    import transformers

    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", lambda *args, **kwargs: DummyConfig()
    )
    monkeypatch.setattr(transformers, "AutoModelForCausalLM", DummyAutoModel)
    monkeypatch.setattr(transformers, "AutoModelForImageTextToText", DummyAutoModel)
    monkeypatch.setattr(transformers, "AutoTokenizer", DummyTokenizerLoader)
    monkeypatch.setattr(transformers, "AutoProcessor", DummyTokenizerLoader)
    monkeypatch.setattr(transformers, "TorchAoConfig", lambda config: config)
    monkeypatch.setattr(loader_utils, "_get_torchao_fp8_config", lambda mode: {"mode": mode})
    monkeypatch.setattr(loader_utils.torch.cuda, "empty_cache", lambda: None)

    with tempfile.TemporaryDirectory() as temp_dir:
        monkeypatch.setattr(loader_utils.tempfile, "gettempdir", lambda: temp_dir)
        stale_cache = Path(temp_dir) / "model-fp8-row"
        stale_cache.mkdir()
        Path(stale_cache, "pytorch_model.bin").write_text("old")

        result = loader_utils._offline_quantize_to_fp8("org/model", "row")
        assert result == str(stale_cache)
        assert not Path(result, "pytorch_model.bin").exists()
        assert Path(result, "model.safetensors").exists()


def test_offline_fp8_cache_forces_safetensors_loads():
    tree = ast.parse(LOADER.read_text())
    hit_lines = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        if not any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_offline_quantize_to_fp8"
            for child in ast.walk(node)
        ):
            continue
        for child in ast.walk(node):
            if not isinstance(child, ast.Assign):
                continue
            for target in child.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "kwargs"
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "use_safetensors"
                    and isinstance(child.value, ast.Constant)
                    and child.value.value is True
                ):
                    hit_lines.add(child.lineno)

    assert len(hit_lines) == 2


def test_fp8_probe_skips_missing_optional_kernel_module():
    loader_utils = _load_loader_utils()
    module = torch.nn.Linear(4, 4, bias = False)

    assert loader_utils._is_real_fp8_owner(module) is False


def test_model_has_real_fp8_modules_detects_nested_fp8_layers():
    loader_utils = _load_loader_utils()
    model = torch.nn.Module()
    model.fp8 = _PackedFp8Owner()

    assert loader_utils._model_has_real_fp8_modules(model) is True


def test_fp8_probe_uses_absolute_optional_import_and_broad_nonfatal_guard():
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
            assert handler.type.id == "Exception"
            return

    pytest.fail("_is_real_fp8_owner no longer protects the optional fp8 kernel import.")
