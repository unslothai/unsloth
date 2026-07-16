"""Regression test for the duplicate ``unsloth/gemma-2b-bnb-4bit`` key in
``unsloth/models/mapper.py``.

The 4bit instruction-tuned Gemma 2B entry was accidentally keyed with the base
model's repo name, so ``__INT_TO_FLOAT_MAPPER`` held two identical
``unsloth/gemma-2b-bnb-4bit`` keys. Python keeps only the last value for a
duplicate literal key, so the base 4bit repo resolved to the *instruct* model,
the base model lost its reverse (4x-faster) mapping, and
``unsloth/gemma-2b-it-bnb-4bit`` was never registered at all.

``mapper.py`` has no imports, so we exec it directly and inspect the built
mappers without importing ``unsloth`` (which requires a GPU).
"""

import os

MAPPER_PATH = os.path.join(os.path.dirname(__file__), os.pardir, "unsloth", "models", "mapper.py")


def _load_mappers():
    with open(MAPPER_PATH) as f:
        source = f.read()
    namespace = {}
    exec(compile(source, MAPPER_PATH, "exec"), namespace)
    return namespace


def test_gemma_2b_base_and_instruct_4bit_are_distinct():
    namespace = _load_mappers()
    int_to_float = namespace["INT_TO_FLOAT_MAPPER"]
    float_to_int = namespace["FLOAT_TO_INT_MAPPER"]

    # The base 4bit repo must resolve to the base model, not the instruct one.
    assert int_to_float["unsloth/gemma-2b-bnb-4bit"] == "unsloth/gemma-2b"

    # The instruct 4bit repo must be registered and resolve to the instruct model.
    assert "unsloth/gemma-2b-it-bnb-4bit" in int_to_float
    assert int_to_float["unsloth/gemma-2b-it-bnb-4bit"] == "unsloth/gemma-2b-it"

    # The base model must reverse-map back to the base 4bit repo.
    assert float_to_int["unsloth/gemma-2b"] == "unsloth/gemma-2b-bnb-4bit"
    assert float_to_int["google/gemma-2b"] == "unsloth/gemma-2b-bnb-4bit"

    # The instruct model must reverse-map to the instruct 4bit repo.
    assert float_to_int["unsloth/gemma-2b-it"] == "unsloth/gemma-2b-it-bnb-4bit"
    assert float_to_int["google/gemma-2b-it"] == "unsloth/gemma-2b-it-bnb-4bit"
