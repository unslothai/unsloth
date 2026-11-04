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

import ast
import os
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

UTILS = Path(__file__).resolve().parents[2] / "unsloth" / "models" / "_utils.py"


def _offload_to_disk():
    tree = ast.parse(UTILS.read_text(encoding = "utf-8"))
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "offload_to_disk"
    )
    namespace = {"os": os, "torch": torch, "pickle": pickle}
    exec(compile(ast.Module(body = [fn], type_ignores = []), str(UTILS), "exec"), namespace)
    return namespace["offload_to_disk"]


@pytest.mark.parametrize("absolute", [False, True])
def test_offloaded_embeddings_stay_under_temporary_location(tmp_path, absolute):
    name_or_path = str(tmp_path / "local-model") if absolute else "unsloth/Qwen3-0.6B"
    model = SimpleNamespace(config = SimpleNamespace(_name_or_path = name_or_path))
    rank_dirs = [tmp_path / "buffers_rank0", tmp_path / "buffers_rank1"]
    saved = [
        _offload_to_disk()(torch.zeros(4, 4), model, "input_embeddings", str(location))
        for location in rank_dirs
    ]
    files = [Path(w._offloaded_file_location) for w in saved]
    # DDP ranks pass their own temporary_location; an absolute model path must not collapse them.
    assert files[0] != files[1]
    for location, file in zip(rank_dirs, files):
        assert location in file.parents
