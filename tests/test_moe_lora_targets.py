import ast
import re
from pathlib import Path
from typing import List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
UTILS_PATH = REPO_ROOT / "unsloth" / "models" / "_utils.py"


def _load_moe_helpers():
    tree = ast.parse(UTILS_PATH.read_text())
    names = {
        "is_moe_model",
        "_resolve_moe_parameter_name",
        "get_moe_target_parameters",
    }
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body = body, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {"Optional": Optional, "List": List, "re": re}
    exec(compile(module, str(UTILS_PATH), "exec"), namespace)
    return namespace


get_moe_target_parameters = _load_moe_helpers()["get_moe_target_parameters"]


class _Config:
    model_type = "mixtral"
    num_local_experts = 8


class _UnfusedMixtralModel:
    config = _Config()

    def named_parameters(self):
        for name in (
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.mlp.experts.0.w2.weight",
            "model.layers.0.mlp.experts.0.w3.weight",
        ):
            yield name, None


class _FusedMoeModel:
    config = _Config()

    def named_parameters(self):
        for name in (
            "model.layers.0.mlp.experts.gate_up_proj",
            "model.layers.0.mlp.experts.down_proj",
        ):
            yield name, None


def test_issue_5403_w_targets_do_not_create_fused_target_parameters():
    assert (
        get_moe_target_parameters(
            _UnfusedMixtralModel(),
            ["q_proj", "k_proj", "v_proj", "o_proj", "w1", "w2", "w3"],
        )
        is None
    )


def test_broad_fused_mlp_targets_still_create_fused_target_parameters():
    assert get_moe_target_parameters(
        _FusedMoeModel(), ["gate_proj", "up_proj", "down_proj"]
    ) == ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
