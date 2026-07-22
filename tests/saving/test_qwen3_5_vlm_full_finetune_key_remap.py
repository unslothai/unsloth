import ast
import types
from pathlib import Path

import torch


def _load_qwen3_5_vlm_save_helpers():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    helpers = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "_is_qwen3_5_vlm",
            "_qwen3_5_vlm_state_dict_for_save",
        }
    ]
    module = ast.Module(body = helpers, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


def _qwen3_5_vlm_model():
    return types.SimpleNamespace(
        config = types.SimpleNamespace(
            architectures = ["Qwen3_5ForConditionalGeneration"],
            model_type = "qwen3_5",
            vision_config = types.SimpleNamespace(),
        )
    )


def test_qwen3_5_vlm_state_dict_uses_hf_checkpoint_namespace():
    helpers = _load_qwen3_5_vlm_save_helpers()
    state_dict = {
        "language_model.model.embed_tokens.weight": torch.ones(2, 2),
        "language_model.model.layers.0.input_layernorm.weight": torch.ones(2),
        "language_model.lm_head.weight": torch.ones(2, 2),
        "visual.blocks.0.norm1.weight": torch.ones(2),
        "other.weight": torch.ones(2),
    }

    remapped = helpers["_qwen3_5_vlm_state_dict_for_save"](state_dict)

    assert "model.language_model.embed_tokens.weight" in remapped
    assert "model.language_model.layers.0.input_layernorm.weight" in remapped
    assert "lm_head.weight" in remapped
    assert "model.visual.blocks.0.norm1.weight" in remapped
    assert "other.weight" in remapped
    assert "language_model.model.embed_tokens.weight" not in remapped
    assert "language_model.lm_head.weight" not in remapped
    assert "visual.blocks.0.norm1.weight" not in remapped


def test_qwen3_5_vlm_detection_requires_vision_config():
    helpers = _load_qwen3_5_vlm_save_helpers()
    assert helpers["_is_qwen3_5_vlm"](_qwen3_5_vlm_model())

    model = types.SimpleNamespace(
        config = types.SimpleNamespace(
            architectures = ["Qwen3_5ForCausalLM"],
            model_type = "qwen3_5_text",
        )
    )

    assert not helpers["_is_qwen3_5_vlm"](model)
