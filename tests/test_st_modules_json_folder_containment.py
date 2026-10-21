# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""modules.json module folders must stay inside the model folder."""

import json

import pytest

pytest.importorskip("sentence_transformers")

from unsloth import FastSentenceTransformer  # noqa: E402


def _model(tmp_path, modules):
    model = tmp_path / "model"
    model.mkdir()
    (model / "modules.json").write_text(json.dumps(modules), encoding = "utf-8")
    return str(model)


@pytest.mark.parametrize(
    "path,type_",
    [
        ("../victim", "sentence_transformers.models.Transformer"),
        ("/abs/victim", "sentence_transformers.models.Router"),
        ("sub/../../victim", "sentence_transformers.models.Pooling"),
        ("D:victim", "sentence_transformers.models.Pooling"),
    ],
    ids = ["transformer_dotdot", "router_absolute", "pooling_nested_dotdot", "drive_relative"],
)
def test_module_folder_outside_the_model_is_refused(tmp_path, path, type_):
    model = _model(tmp_path, [{"idx": 0, "name": "0", "path": path, "type": type_}])
    with pytest.raises(ValueError, match = "outside the model"):
        FastSentenceTransformer._check_modules_json_types(model, None, False)


def test_stock_layout_still_passes(tmp_path):
    model = _model(
        tmp_path,
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ],
    )
    (tmp_path / "model" / "1_Pooling").mkdir()
    FastSentenceTransformer._check_modules_json_types(model, None, False)


def test_a_module_folder_that_walks_back_inside_passes(tmp_path):
    model = _model(
        tmp_path,
        [
            {"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
            {
                "idx": 1,
                "name": "1",
                "path": "modules/../1_Pooling",
                "type": "sentence_transformers.models.Pooling",
            },
        ],
    )
    (tmp_path / "model" / "1_Pooling").mkdir()
    FastSentenceTransformer._check_modules_json_types(model, None, False)
