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

"""16bit loads and merges take a row's first name, so it must not be an unpublished ``unsloth/`` repo."""

import os

import pytest

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")

# key -> (unpublished name, upstream)
ROWS = {
    "unsloth/mistral-7b-instruct-v0.1-bnb-4bit": (
        "unsloth/mistral-7b-instruct-v0.1",
        "mistralai/Mistral-7B-Instruct-v0.1",
    ),
    "unsloth/Hermes-3-Llama-3.1-70B-bnb-4bit": (
        "unsloth/Hermes-3-Llama-3.1-70B",
        "NousResearch/Hermes-3-Llama-3.1-70B",
    ),
    "unsloth/Qwen2-VL-72B-bnb-4bit": (
        "unsloth/Qwen2-VL-72B",
        "Qwen/Qwen2-VL-72B",
    ),
    "unsloth/Llama-3.1-Tulu-3-70B-bnb-4bit": (
        "unsloth/Llama-3.1-Tulu-3-70B",
        "allenai/Llama-3.1-Tulu-3-70B",
    ),
    "unsloth/OpenThinker-7B-unsloth-bnb-4bit": (
        "unsloth/OpenThinker-7B",
        "open-thoughts/OpenThinker-7B",
    ),
}


def _load_mappers():
    with open(MAPPER_PATH, encoding = "utf-8") as f:
        source = f.read()
    namespace = {}
    exec(compile(source, MAPPER_PATH, "exec"), namespace)
    return namespace


@pytest.mark.parametrize("key", sorted(ROWS))
def test_row_resolves_16bit_to_upstream(key):
    namespace = _load_mappers()
    unpublished, upstream = ROWS[key]

    assert namespace["INT_TO_FLOAT_MAPPER"][key] == upstream
    assert namespace["INT_TO_FLOAT_MAPPER"][key.lower()] == upstream
    assert upstream.lower() not in namespace["MAP_TO_UNSLOTH_16bit"]
    assert namespace["FLOAT_TO_INT_MAPPER"][upstream] == key


def test_unpublished_names_are_gone():
    with open(MAPPER_PATH, encoding = "utf-8") as f:
        source = f.read()
    for unpublished, _ in ROWS.values():
        assert f'"{unpublished}",' not in source, unpublished
