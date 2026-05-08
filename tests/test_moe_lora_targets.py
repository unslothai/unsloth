import ast
import re
from pathlib import Path
from typing import List, Optional

import torch


def _load_moe_target_helpers():
    utils_path = (
        Path(__file__).resolve().parents[1] / "unsloth" / "models" / "_utils.py"
    )
    tree = ast.parse(utils_path.read_text())
    names = {
        "is_moe_model",
        "_resolve_moe_parameter_name",
        "get_moe_target_parameters",
        "_MOE_BROAD_MLP_TARGETS",
    }
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in names for target in node.targets
        ):
            body.append(node)
    module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Optional": Optional, "List": List, "re": re}
    exec(compile(module, str(utils_path), "exec"), namespace)
    return namespace


_HELPERS = _load_moe_target_helpers()
get_moe_target_parameters = _HELPERS["get_moe_target_parameters"]


class _Config:
    num_experts = 2


class _Experts(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = torch.nn.Parameter(torch.randn(2, 4, 4))
        self.down_proj = torch.nn.Parameter(torch.randn(2, 4, 4))


class _SharedExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = torch.nn.Linear(4, 4)
        self.up_proj = torch.nn.Linear(4, 4)
        self.down_proj = torch.nn.Linear(4, 4)


class _Mlp(torch.nn.Module):
    def __init__(self, with_shared_expert=True, shared_expert_name="shared_expert"):
        super().__init__()
        self.experts = _Experts()
        if with_shared_expert:
            setattr(self, shared_expert_name, _SharedExpert())


class _Block(torch.nn.Module):
    def __init__(self, with_shared_expert=True, shared_expert_name="shared_expert"):
        super().__init__()
        self.mlp = _Mlp(
            with_shared_expert=with_shared_expert,
            shared_expert_name=shared_expert_name,
        )


class _MoeModel(torch.nn.Module):
    def __init__(self, with_shared_expert=True, shared_expert_name="shared_expert"):
        super().__init__()
        self.config = _Config()
        self.layers = torch.nn.ModuleList(
            [
                _Block(
                    with_shared_expert=with_shared_expert,
                    shared_expert_name=shared_expert_name,
                )
            ]
        )


def test_bare_mlp_targets_add_routed_parameters_without_mutating_target_modules():
    model = _MoeModel()
    target_modules = ["gate_proj", "up_proj", "down_proj", "gate_up_proj"]

    target_parameters = get_moe_target_parameters(model, target_modules)

    assert target_parameters == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    assert target_modules == [
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
    ]


def test_target_modules_are_not_mutated_when_shared_expert_modules_do_not_exist():
    model = _MoeModel(with_shared_expert=False)
    target_modules = ["gate_proj", "up_proj", "down_proj"]

    get_moe_target_parameters(model, target_modules)

    assert target_modules == ["gate_proj", "up_proj", "down_proj"]


def test_target_modules_are_not_mutated_when_plural_shared_experts_exist():
    model = _MoeModel(shared_expert_name="shared_experts")
    target_modules = ["gate_proj", "up_proj", "down_proj"]

    get_moe_target_parameters(model, target_modules)

    assert target_modules == [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]


def test_qualified_targets_do_not_expand_to_routed_moe_parameters():
    model = _MoeModel()

    assert get_moe_target_parameters(model, ["shared_expert.gate_proj"]) is None
    assert get_moe_target_parameters(model, ["mlp.experts.down_proj"]) is None
    assert get_moe_target_parameters(model, "model.layers.0.mlp.gate_proj") is None


def test_bare_mlp_targets_expand_to_routed_moe_parameters():
    model = _MoeModel()

    assert get_moe_target_parameters(
        model, ["gate_proj", "up_proj", "down_proj", "gate_up_proj"]
    ) == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]


def test_empty_target_modules_keeps_explicit_target_parameters_path_available():
    model = _MoeModel()

    assert get_moe_target_parameters(model, []) is None


def test_none_target_modules_does_not_infer_all_moe_parameters():
    model = _MoeModel()

    assert get_moe_target_parameters(model, None) is None
