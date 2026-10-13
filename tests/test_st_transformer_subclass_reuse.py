# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A modules.json "type" that subclasses sentence-transformers' Transformer (tencent/WeMM-Embedding
ships `modeling_st_wemm.WeMMTransformer`) must wrap the model Unsloth loaded, not load a second,
unpatched copy through its own `load()`. unslothai/unsloth#13566.

Offline and CPU-only: a tiny random BERT saved to a temp dir.
"""

import json
import sys

import pytest

pytest.importorskip("sentence_transformers")

from unsloth import FastSentenceTransformer  # noqa: E402

PKG = "unsloth_st_subclass_pkg"


@pytest.fixture
def subclass_model(tmp_path, monkeypatch):
    from transformers import BertConfig, BertModel, BertTokenizerFast

    vocab = tmp_path / "vocab.txt"
    vocab.write_text(
        "\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "red", "square"]),
        encoding = "utf-8",
    )
    tokenizer = BertTokenizerFast(vocab_file = str(vocab))
    config = BertConfig(
        vocab_size = 8,
        hidden_size = 16,
        num_hidden_layers = 1,
        num_attention_heads = 2,
        intermediate_size = 32,
        max_position_embeddings = 64,
    )
    model = BertModel(config).eval()

    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    pkg = tmp_path / "site" / PKG
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text(
        "from sentence_transformers.models import Transformer\n"
        "class RepoTransformer(Transformer):\n"
        "    def repo_method(self):\n"
        "        return 'repo'\n"
        "class NotATransformer:\n"
        "    @classmethod\n"
        "    def load(cls, path, **kwargs):\n"
        "        return cls()\n",
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    def write_modules(type_ref):
        (model_dir / "modules.json").write_text(
            json.dumps([{"idx": 0, "name": "0", "path": "", "type": type_ref}]),
            encoding = "utf-8",
        )

    yield model_dir, model, tokenizer, write_modules

    for name in [m for m in sys.modules if m.split(".")[0] == PKG]:
        del sys.modules[name]


def _load(model_dir, model, tokenizer):
    modules, no_modules_json = FastSentenceTransformer._load_modules(
        str(model_dir),
        None,
        model,
        tokenizer,
        32,
        "mean",
        trust_remote_code = True,
    )
    assert no_modules_json is False
    return modules["0"]


def test_a_transformer_subclass_wraps_the_loaded_model(subclass_model):
    model_dir, model, tokenizer, write_modules = subclass_model
    write_modules(f"{PKG}.RepoTransformer")

    module = _load(model_dir, model, tokenizer)

    assert type(module).__name__ == "RepoTransformer"
    assert module.repo_method() == "repo"
    assert module.auto_model is model
    assert module.max_seq_length == 32


def test_the_stock_transformer_still_wraps_the_loaded_model(subclass_model):
    from sentence_transformers.models import Transformer

    model_dir, model, tokenizer, write_modules = subclass_model
    write_modules("sentence_transformers.models.Transformer")

    module = _load(model_dir, model, tokenizer)

    assert type(module) is Transformer
    assert module.auto_model is model


def test_a_custom_module_that_is_not_a_transformer_keeps_its_own_load(subclass_model):
    model_dir, model, tokenizer, write_modules = subclass_model
    write_modules(f"{PKG}.NotATransformer")

    module = _load(model_dir, model, tokenizer)

    assert type(module).__name__ == "NotATransformer"
