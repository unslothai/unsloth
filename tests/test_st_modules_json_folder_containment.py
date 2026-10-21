# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen, Michael Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
    ],
    ids = ["transformer_dotdot", "router_absolute", "pooling_nested_dotdot"],
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
