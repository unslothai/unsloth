import ast
import json
import re
import types
from pathlib import Path

import torch


def _load_qwen3_5_vlm_save_helpers():
    source = Path(__file__).parents[2] / "unsloth" / "save.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    names = {
        "_QWEN3_5_GGUF_ARCHITECTURES",
        "_QWEN3_5_GGUF_MODEL_TYPES",
        "_as_list",
        "_is_qwen3_5_config_dict",
        "_qwen3_5_num_hidden_layers",
        "_qwen3_5_existing_mtp_layers",
        "_qwen3_5_tensor_names_from_save_directory",
        "_qwen3_5_infer_mtp_layers_from_tensor_names",
        "_ensure_qwen3_5_mtp_config_for_gguf",
        "_is_qwen3_5_vlm",
        "_qwen3_5_vlm_state_dict_for_save",
    }
    helpers = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    module = ast.Module(body = helpers, type_ignores = [])
    ast.fix_missing_locations(module)
    namespace = {
        "json": json,
        "Path": Path,
        "re": re,
        "torch": torch,
        "logger": types.SimpleNamespace(warning_once = lambda *_args, **_kwargs: None),
    }
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


def test_qwen3_5_gguf_save_restores_missing_mtp_config(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    index = {
        "metadata": {},
        "weight_map": {
            "model.layers.31.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
            "model.layers.32.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    updated = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 1
    assert updated["text_config"]["mtp_num_hidden_layers"] == 1


def test_qwen3_5_gguf_save_reads_pytorch_shard_index(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    index = {
        "metadata": {},
        "weight_map": {
            "model.layers.32.mlp.down_proj.weight": "pytorch_model-00001-of-00001.bin",
        },
    }
    (tmp_path / "pytorch_model.bin.index.json").write_text(json.dumps(index), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    updated = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 1
    assert updated["text_config"]["mtp_num_hidden_layers"] == 1


def test_qwen3_5_gguf_save_reads_variant_pytorch_shard_index(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    for index_name in ("pytorch_model.bin.index.fp16.json", "pytorch_model.fp16.bin.index.json"):
        save_dir = tmp_path / index_name
        save_dir.mkdir()
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 32,
            },
        }
        (save_dir / "config.json").write_text(json.dumps(config), encoding = "utf-8")
        index = {
            "metadata": {},
            "weight_map": {
                "model.layers.32.mlp.down_proj.weight": "pytorch_model-00001-of-00001.fp16.bin",
            },
        }
        (save_dir / index_name).write_text(json.dumps(index), encoding = "utf-8")

        helpers["_ensure_qwen3_5_mtp_config_for_gguf"](save_dir)

        updated = json.loads((save_dir / "config.json").read_text(encoding = "utf-8"))
        assert updated["mtp_num_hidden_layers"] == 1
        assert updated["text_config"]["mtp_num_hidden_layers"] == 1


def test_qwen3_5_gguf_save_reads_unsharded_pytorch_checkpoint(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    for checkpoint_name in ("pytorch_model.bin", "pytorch_model.fp16.bin"):
        save_dir = tmp_path / checkpoint_name
        save_dir.mkdir()
        config = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 32,
            },
        }
        (save_dir / "config.json").write_text(json.dumps(config), encoding = "utf-8")
        state_dict = {
            "model.layers.32.mlp.down_proj.weight": torch.ones(1),
        }
        torch.save(state_dict, save_dir / checkpoint_name)

        helpers["_ensure_qwen3_5_mtp_config_for_gguf"](save_dir)

        updated = json.loads((save_dir / "config.json").read_text(encoding = "utf-8"))
        assert updated["mtp_num_hidden_layers"] == 1
        assert updated["text_config"]["mtp_num_hidden_layers"] == 1


def test_qwen3_5_gguf_save_recognizes_vlm_prefixed_mtp_tensors(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    index = {
        "metadata": {},
        "weight_map": {
            "language_model.mtp.layers.0.mlp.down_proj.weight": "model.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    updated = json.loads((tmp_path / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 1
    assert updated["text_config"]["mtp_num_hidden_layers"] == 1


def test_qwen3_5_gguf_save_fills_missing_mtp_config_copy(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    root_only = tmp_path / "root_only"
    root_only.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (root_only / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](root_only)

    updated = json.loads((root_only / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 1
    assert updated["text_config"]["mtp_num_hidden_layers"] == 1

    text_only = tmp_path / "text_only"
    text_only.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "mtp_num_hidden_layers": 1,
        },
    }
    (text_only / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](text_only)

    updated = json.loads((text_only / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 1
    assert updated["text_config"]["mtp_num_hidden_layers"] == 1

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "mtp_num_hidden_layers": 1,
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "mtp_num_hidden_layers": 2,
        },
    }
    (mismatched / "config.json").write_text(json.dumps(config), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](mismatched)

    updated = json.loads((mismatched / "config.json").read_text(encoding = "utf-8"))
    assert updated["mtp_num_hidden_layers"] == 2
    assert updated["text_config"]["mtp_num_hidden_layers"] == 2


def test_qwen3_5_gguf_save_leaves_non_qwen_config_unchanged(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "num_hidden_layers": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    index = {
        "metadata": {},
        "weight_map": {
            "model.layers.32.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    assert json.loads((tmp_path / "config.json").read_text(encoding = "utf-8")) == config


def test_qwen3_5_gguf_save_ignores_malformed_safetensors_index(tmp_path):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    (tmp_path / "model.safetensors.index.json").write_text("{", encoding = "utf-8")

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    assert json.loads((tmp_path / "config.json").read_text(encoding = "utf-8")) == config


def test_qwen3_5_gguf_save_handles_config_write_failure(tmp_path, monkeypatch):
    helpers = _load_qwen3_5_vlm_save_helpers()
    config = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding = "utf-8")
    index = {
        "metadata": {},
        "weight_map": {
            "model.layers.32.mlp.down_proj.weight": "model.safetensors-00001-of-00001.safetensors",
        },
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index), encoding = "utf-8")

    original_dump = json.dump

    def fail_config_dump(value, file, *args, **kwargs):
        if Path(file.name).name in {"config.json", ".config.json.tmp"}:
            file.write("{")
            raise OSError("disk full")
        return original_dump(value, file, *args, **kwargs)

    monkeypatch.setattr(json, "dump", fail_config_dump)

    helpers["_ensure_qwen3_5_mtp_config_for_gguf"](tmp_path)

    assert json.loads((tmp_path / "config.json").read_text(encoding = "utf-8")) == config
    assert not (tmp_path / ".config.json.tmp").exists()
