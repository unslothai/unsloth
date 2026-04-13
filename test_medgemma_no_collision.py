"""Test that medgemma models (containing 'gemma-4' substring via 'medgemma-4b') are not misrouted."""

import os, sys

sys.path.insert(0, os.path.dirname(__file__))

MAPPINGS_FILE = os.path.join(
    os.path.dirname(__file__),
    "studio",
    "backend",
    "utils",
    "datasets",
    "model_mappings.py",
)


def _load():
    g = {}
    exec(open(MAPPINGS_FILE).read(), g)
    return g


def _load_mappers():
    g = {}
    exec(
        open(
            os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")
        ).read(),
        g,
    )
    return g


def test_medgemma_routes_to_gemma3_not_gemma4():
    """medgemma-4b-it should route to gemma-3, not gemma-4."""
    g = _load()
    M2T = g["MODEL_TO_TEMPLATE_MAPPER"]
    assert M2T.get("unsloth/medgemma-4b-it") == "gemma-3"
    assert M2T.get("google/medgemma-4b-it") == "gemma-3"


def test_medgemma_not_routed_to_gemma4_bnb():
    """medgemma bnb-4bit should point to medgemma, not gemma-4 bnb-4bit."""
    g = _load_mappers()
    FTI = g["FLOAT_TO_INT_MAPPER"]
    result = FTI.get("google/medgemma-4b-it")
    if result is not None:
        assert (
            "medgemma" in result
        ), f"medgemma-4b-it routed away from medgemma: {result}"


def test_medgemma_bnb4bit_maps_to_gemma3():
    """medgemma bnb-4bit should map to gemma-3 template, not gemma-4."""
    g = _load()
    M2T = g["MODEL_TO_TEMPLATE_MAPPER"]
    result = M2T.get("unsloth/medgemma-4b-it-unsloth-bnb-4bit")
    assert result == "gemma-3", f"Expected 'gemma-3', got {result!r}"
