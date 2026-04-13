"""Test mapper behavior with edge-case model name inputs."""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))


def _load_mappers():
    g = {}
    exec(
        open(
            os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")
        ).read(),
        g,
    )
    return g


def test_empty_string_not_in_mappers():
    g = _load_mappers()
    assert "" not in g["INT_TO_FLOAT_MAPPER"]
    assert "" not in g["FLOAT_TO_INT_MAPPER"]
    assert "" not in g["MAP_TO_UNSLOTH_16bit"]


def test_none_key_not_in_mappers():
    g = _load_mappers()
    assert None not in g["INT_TO_FLOAT_MAPPER"]
    assert None not in g["FLOAT_TO_INT_MAPPER"]
    assert None not in g["MAP_TO_UNSLOTH_16bit"]


def test_trailing_slash_not_matched():
    """Model names with trailing slashes should not match."""
    g = _load_mappers()
    assert g["FLOAT_TO_INT_MAPPER"].get("google/gemma-4-e2b-it/") is None
    assert g["MAP_TO_UNSLOTH_16bit"].get("google/gemma-4-26b-a4b-it/") is None


def test_whitespace_model_name_not_matched():
    g = _load_mappers()
    assert g["FLOAT_TO_INT_MAPPER"].get(" google/gemma-4-e2b-it") is None
    assert g["FLOAT_TO_INT_MAPPER"].get("google/gemma-4-e2b-it ") is None


def test_partial_prefix_not_matched():
    """A substring of a valid model name should not accidentally match."""
    g = _load_mappers()
    FTI = g["FLOAT_TO_INT_MAPPER"]
    assert FTI.get("google/gemma-4") is None
    assert FTI.get("unsloth/gemma-4") is None
    assert FTI.get("gemma-4-E2B-it") is None
