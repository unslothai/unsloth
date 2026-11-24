# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A sentence-transformers model's own sentence_bert_config.json must not choose from_pretrained
arguments that run its code: `weights_only: false` unpickled the folder's pytorch_model.bin with remote
code off. The caller's model_kwargs and plain configs keep working."""

import json
import os

import pytest

st = pytest.importorskip("sentence_transformers")
torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from utils.security import st_config_args


class _Marker:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (open, (self.path, "w"))


def _model(
    folder,
    model_args,
    marker = None,
):
    os.makedirs(folder)
    cfg = transformers.BertConfig(
        vocab_size = 32,
        hidden_size = 8,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 16,
        max_position_embeddings = 16,
    )
    transformers.BertModel(cfg).save_pretrained(folder, safe_serialization = False)
    if marker is not None:
        # No dtype in config.json: dtype "auto" then reads it from the pickle.
        with open(os.path.join(folder, "config.json")) as f:
            config = json.load(f)
        config.pop("dtype", None)
        config.pop("torch_dtype", None)
        with open(os.path.join(folder, "config.json"), "w") as f:
            json.dump(config, f)
        torch.save({"payload": _Marker(marker)}, os.path.join(folder, "pytorch_model.bin"))
    with open(os.path.join(folder, "vocab.txt"), "w") as f:
        f.write(
            "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [f"w{i}" for i in range(27)])
        )
    with open(os.path.join(folder, "sentence_bert_config.json"), "w") as f:
        json.dump({"max_seq_length": 16, "model_args": model_args}, f)
    os.makedirs(os.path.join(folder, "1_Pooling"))
    with open(os.path.join(folder, "1_Pooling", "config.json"), "w") as f:
        json.dump({"word_embedding_dimension": 8, "pooling_mode_mean_tokens": True}, f)
    with open(os.path.join(folder, "modules.json"), "w") as f:
        json.dump(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ],
            f,
        )
    return folder


def test_repo_set_weights_only_false_does_not_unpickle(tmp_path):
    marker = str(tmp_path / "pickle-ran")
    folder = _model(
        str(tmp_path / "attack"),
        {"weights_only": False, "use_safetensors": False, "dtype": "auto"},
        marker,
    )
    st_config_args.install()
    try:
        st.SentenceTransformer(folder, device = "cpu", trust_remote_code = False)
    except Exception:
        pass  # weights_only=True refuses the pickle; only whether it ran matters
    assert not os.path.exists(marker), "the model's own config turned torch.load's weights_only off"


def test_install_is_idempotent_and_plain_models_load(tmp_path):
    st_config_args.install()
    st_config_args.install()
    loader = st.models.Transformer.load_config
    assert getattr(loader.__func__, "_unsloth_scrubs_load_args")
    assert not getattr(loader.__func__.__wrapped__, "_unsloth_scrubs_load_args", False)
    folder = _model(str(tmp_path / "plain"), {"attn_implementation": "eager"})
    model = st.SentenceTransformer(folder, device = "cpu")
    assert model.encode(["w1 w2"]).shape == (1, 8)


def test_scrub_keeps_safe_args_and_drops_code_paths():
    cfg = {
        "model_args": {
            "weights_only": False,
            "use_safetensors": False,
            "attn_implementation": "kernels-community/flash-attn",
            "allow_all_kernels": True,
            "dtype": "float32",
        },
        "config_args": {"attn_implementation": "sdpa", "use_safetensors": True},
        "tokenizer_name_or_path": "/root/secret",
    }
    out = st_config_args.scrub_load_args(cfg)
    assert out["model_args"] == {"dtype": "float32"}
    assert out["config_args"] == {"attn_implementation": "sdpa", "use_safetensors": True}
    assert "tokenizer_name_or_path" not in out
    assert st_config_args.scrub_load_args({"tokenizer_name_or_path": "tok"}) == {
        "tokenizer_name_or_path": "tok"
    }
