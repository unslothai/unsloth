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

"""A modules.json "type" is untrusted input, not an import target.

`_module_path` fetches modules.json with `hf_hub_download`, so publishing a Hub repo is
enough; no local access required. See `_resolve_module_class` for the gate being tested.

Offline and CPU-only: the marker package is inert and only appends to a file, which is how
these tests observe whether an import happened at all.
"""

import json
import os
import sys

import pytest

from unsloth import FastSentenceTransformer


MARKER = "unsloth_st_gate_marker"


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    """A model folder whose single modules.json entry names the marker package."""
    witness = tmp_path / "witness.txt"

    pkg = tmp_path / "site" / MARKER
    pkg.mkdir(parents = True)
    (pkg / "__init__.py").write_text(
        f"open({str(witness)!r}, 'a').write('imported\\n')\n"
        "class Thing:\n"
        "    @staticmethod\n"
        "    def load(path, **kwargs):\n"
        "        return Thing()\n",
        encoding = "utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path / "site"))

    model = tmp_path / "model"
    (model / "0_Sub").mkdir(parents = True)
    (model / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Sub", "type": f"{MARKER}.Thing"}]),
        encoding = "utf-8",
    )

    yield model, witness

    for name in [m for m in sys.modules if m.split(".")[0] == MARKER]:
        del sys.modules[name]


def _load_modules(model, trust_remote_code):
    return FastSentenceTransformer._load_modules(
        str(model),
        None,
        None,
        None,
        64,
        "mean",
        trust_remote_code = trust_remote_code,
    )


def test_an_untrusted_module_type_is_never_imported(model_dir):
    """The refusal must land BEFORE the import, so the witness must not exist."""
    model, witness = model_dir

    with pytest.raises(ValueError, match = "not part of Sentence Transformers"):
        _load_modules(model, False)

    assert not witness.exists(), "modules.json type was imported without consent"


def test_the_refusal_fails_the_load_rather_than_skipping_the_module(model_dir):
    """Fail closed, rather than degrading to "load the other modules and carry on"."""
    model, witness = model_dir
    (model / "1_Pooling").mkdir()
    (model / "modules.json").write_text(
        json.dumps(
            [
                {"idx": 0, "name": "0", "path": "0_Sub", "type": f"{MARKER}.Thing"},
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
            ]
        ),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError):
        _load_modules(model, False)

    assert not witness.exists()


def test_consent_still_loads_a_custom_module_type(model_dir):
    model, witness = model_dir

    modules, no_modules_json = _load_modules(model, True)

    assert witness.exists(), "consented module type was not imported"
    assert no_modules_json is False
    assert [type(m).__name__ for m in modules.values()] == ["Thing"]


def test_a_non_string_module_type_is_refused(model_dir):
    model, witness = model_dir
    (model / "modules.json").write_text(
        json.dumps([{"idx": 0, "name": "0", "path": "0_Sub", "type": {"bad": 1}}]),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError):
        _load_modules(model, False)

    assert not witness.exists()


@pytest.mark.parametrize(
    "class_ref",
    [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.Pooling",
        "sentence_transformers.models.Normalize",
        "sentence_transformers.models.Dense",
    ],
)
def test_the_stock_module_types_resolve_without_consent(class_ref, tmp_path):
    """These are every type real embedders ship: bge, MiniLM, e5, gte, bge-m3 and
    Qwen3-Embedding use the first three, embeddinggemma-300m adds the two Dense entries."""
    module_class = FastSentenceTransformer._resolve_module_class(class_ref, str(tmp_path), False)

    assert module_class.__module__.startswith("sentence_transformers")


@pytest.mark.parametrize(
    "class_ref",
    [
        "sentence_transformers.models.Transformer",
        "sentence_transformers.models.transformer.Transformer",
        "sentence_transformers.base.modules.transformer.Transformer",
    ],
)
def test_the_fast_encoder_path_still_recognises_the_transformer_refs(class_ref):
    """These three refs name ST's Transformer across the 5.x/6.x layouts."""
    assert FastSentenceTransformer._is_transformer_module_ref(class_ref) is True


def test_an_untrusted_type_is_not_probed_as_a_transformer_ref(model_dir):
    """It used to import the ref just to compare it against ST's Transformer, which is the same
    execution by another door. It is now a pure string test."""
    model, witness = model_dir

    assert FastSentenceTransformer._is_transformer_module_ref(f"{MARKER}.Thing") is False
    assert not witness.exists()


def test_a_module_config_may_not_smuggle_a_class_path_past_the_gate(tmp_path, monkeypatch):
    """Simulate sentence-transformers < 6, whose Dense loader imports AND CALLS
    activation_function ungated, by hiding import_module_class."""
    import sentence_transformers.util as st_util
    from sentence_transformers.models import Dense

    monkeypatch.delattr(st_util, "import_module_class", raising = False)

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps({"in_features": 8, "out_features": 8, "activation_function": f"{MARKER}.Thing"}),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "does not gate it"):
        FastSentenceTransformer._check_module_config_class_refs(
            str(load_path), "sentence_transformers.models.Dense", "some/repo", False, Dense
        )


@pytest.mark.parametrize(
    "module_name, config_name, config",
    [
        # Each module names its own config file, so a check reading only "config.json" skips the two
        # modules whose loaders resolve a dotted path out of it, which are the reason it exists.
        ("Router", "router_config.json", {"types": {"query": MARKER + ".Thing"}}),
        ("WordEmbeddings", "wordembedding_config.json", {"tokenizer_class": MARKER + ".Thing"}),
    ],
)
def test_an_in_namespace_type_cannot_smuggle_a_path_via_its_own_config_file(
    module_name, config_name, config, model_dir, monkeypatch
):
    """Through _load_modules, not the helper: the type is an allowed sentence_transformers.* class,
    so the refusal has to come from reading the config file that module's loader reads."""
    import sentence_transformers.util as st_util
    from sentence_transformers import models as st_models

    monkeypatch.delattr(st_util, "import_module_class", raising = False)
    if getattr(st_models, module_name, None) is None:
        pytest.skip(f"installed sentence-transformers has no {module_name}")

    model, witness = model_dir
    (model / f"1_{module_name}").mkdir()
    (model / f"1_{module_name}" / config_name).write_text(json.dumps(config), encoding = "utf-8")
    (model / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": f"1_{module_name}",
                    "type": f"sentence_transformers.models.{module_name}",
                }
            ]
        ),
        encoding = "utf-8",
    )

    with pytest.raises(ValueError, match = "does not gate it"):
        _load_modules(model, False)

    assert not witness.exists()


def test_a_real_dense_config_is_untouched_by_that_check(tmp_path, monkeypatch):
    """embeddinggemma-300m's Dense modules declare torch.nn.modules.linear.Identity, exactly what
    ST >= 6 allows unconsented."""
    import sentence_transformers.util as st_util
    from sentence_transformers.models import Dense

    monkeypatch.delattr(st_util, "import_module_class", raising = False)

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps(
            {
                "in_features": 768,
                "out_features": 3072,
                "bias": False,
                "activation_function": "torch.nn.modules.linear.Identity",
            }
        ),
        encoding = "utf-8",
    )

    FastSentenceTransformer._check_module_config_class_refs(
        str(load_path),
        "sentence_transformers.models.Dense",
        "unsloth/embeddinggemma-300m",
        False,
        Dense,
    )


def test_the_installed_sentence_transformers_gate_is_not_second_guessed(tmp_path):
    """ST >= 6 gates its own config refs, so the check is a no-op rather than a diverging copy."""
    import sentence_transformers.util as st_util
    from sentence_transformers.models import Dense

    if not hasattr(st_util, "import_module_class"):
        pytest.skip("installed sentence-transformers has no gate of its own")

    load_path = tmp_path / "2_Dense"
    load_path.mkdir()
    (load_path / "config.json").write_text(
        json.dumps({"activation_function": f"{MARKER}.Thing"}), encoding = "utf-8"
    )

    FastSentenceTransformer._check_module_config_class_refs(
        str(load_path), "sentence_transformers.models.Dense", "some/repo", False, Dense
    )
