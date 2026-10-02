"""Rows in ``unsloth/models/mapper.py`` must not resolve a 16bit load or merge to an
``unsloth/`` repo that was never published.

``get_model_name(..., load_in_4bit = False)`` returns the first name of a row, and
``MAP_TO_UNSLOTH_16bit`` sends the upstream name there too, so a dead first name breaks
16bit loading and ``save_pretrained_merged`` (``could not be read locally or on Hugging
Face``) even though the row already lists a working upstream. Same shape as the Apertus
70B row: drop the unpublished name and keep the upstream.

``mapper.py`` has no imports, so it is exec'd directly without importing ``unsloth``.
"""

import os

import pytest

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")

# key -> (unpublished 16bit name, upstream the row should resolve to)
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
    # 4bit loads of the upstream name still reach the row.
    assert namespace["FLOAT_TO_INT_MAPPER"][upstream] == key


def test_unpublished_names_are_gone():
    with open(MAPPER_PATH, encoding = "utf-8") as f:
        source = f.read()
    for unpublished, _ in ROWS.values():
        assert f'"{unpublished}",' not in source, unpublished
