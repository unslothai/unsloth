# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""FastSentenceTransformer.get_peft_model on the fast encoder path must hand every LoRA option to LoraConfig, and refuse aLoRA."""

from __future__ import annotations

import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("sentence_transformers")
pytest.importorskip("peft")

MODELS = Path(__file__).resolve().parents[1] / "unsloth" / "models"
PACKAGE = "_unsloth_st_fast_encoder_stub"


def _function(
    path,
    name,
    class_name = None,
):
    tree = ast.parse(path.read_text(encoding = "utf-8"), filename = str(path))
    body = tree.body
    if class_name is not None:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == class_name).body
    node = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    return node


def _exec(node, namespace):
    module = ast.Module(body = [node], type_ignores = [])
    exec(compile(module, node.name, "exec"), namespace)
    return namespace[node.name]


@pytest.fixture
def get_peft_model(monkeypatch):
    # The real function and the real reject_alora; only what they import from unsloth is stubbed.
    reject_alora = _exec(_function(MODELS / "_utils.py", "reject_alora"), {})
    utils = types.ModuleType(f"{PACKAGE}._utils")
    utils.reject_alora = reject_alora
    utils._mark_unsloth_disable_data_parallel = lambda model: None
    llama = types.ModuleType(f"{PACKAGE}.llama")
    llama._unfreeze_modules_to_save = lambda model: None
    package = types.ModuleType(PACKAGE)
    package.__path__ = []
    for module in (package, utils, llama):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    class FastSentenceTransformer:
        @staticmethod
        def _patch_transformer_module_save_config(module, config):
            pass

        @staticmethod
        def _estimate_compile_threshold(model):
            return 0

    namespace = {
        "__package__": PACKAGE,
        "inspect": inspect,
        "FastSentenceTransformer": FastSentenceTransformer,
        "_install_fsdp2_collective_st_save": lambda cls: None,
    }
    node = _function(
        MODELS / "sentence_transformer.py", "get_peft_model", "FastSentenceTransformer"
    )
    return _exec(node, namespace)


@pytest.fixture
def fast_encoder(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.models import Pooling, Transformer
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "red", "square"]),
        encoding = "utf-8",
    )
    config = BertConfig(
        vocab_size = 8,
        hidden_size = 16,
        num_hidden_layers = 2,
        num_attention_heads = 2,
        intermediate_size = 32,
        max_position_embeddings = 64,
    )
    model_dir = tmp_path / "model"
    BertModel(config).save_pretrained(model_dir)
    BertTokenizerFast(vocab_file = str(vocab)).save_pretrained(model_dir)
    model = SentenceTransformer(
        modules = [Transformer(str(model_dir)), Pooling(16)],
        device = "cpu",
    )
    model._unsloth_fast_encoder = True
    return model


def test_exclude_modules_and_lora_bias_reach_the_adapter(get_peft_model, fast_encoder):
    model = get_peft_model(
        fast_encoder,
        r = 4,
        lora_alpha = 8,
        target_modules = ["query", "key", "value"],
        exclude_modules = r".*layer\.0\..*",
        lora_bias = True,
        use_gradient_checkpointing = False,
    )
    peft_model = model[0].auto_model
    config = peft_model.peft_config["default"]
    assert config.task_type == "FEATURE_EXTRACTION"
    assert config.lora_bias is True
    names = [name for name, _ in peft_model.named_parameters() if ".lora_A." in name]
    assert names and all(".layer.1." in name for name in names), names


def test_activated_lora_is_refused_instead_of_trained_as_plain_lora(get_peft_model, fast_encoder):
    with pytest.raises(NotImplementedError, match = "Activated LoRA"):
        get_peft_model(
            fast_encoder,
            r = 4,
            lora_alpha = 8,
            target_modules = ["query", "key", "value"],
            alora_invocation_tokens = [5, 6],
            use_gradient_checkpointing = False,
        )
    assert not hasattr(fast_encoder[0].auto_model, "peft_config")
