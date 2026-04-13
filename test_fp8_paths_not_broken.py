"""Test that new Gemma-4/LFM entries don't break existing FP8 mapper paths."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def _load_mappers():
    g = {}
    exec(open(os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")).read(), g)
    return g


def test_fp8_block_mapper_not_empty():
    g = _load_mappers()
    assert len(g["FLOAT_TO_FP8_BLOCK_MAPPER"]) > 0


def test_fp8_row_mapper_not_empty():
    g = _load_mappers()
    assert len(g["FLOAT_TO_FP8_ROW_MAPPER"]) > 0


def test_gemma4_not_in_fp8_mappers():
    """Gemma-4 doesn't have FP8 variants, so should not appear in FP8 mappers."""
    g = _load_mappers()
    for key in g["FLOAT_TO_FP8_BLOCK_MAPPER"]:
        assert "gemma-4" not in key, f"Gemma-4 key {key} in FP8 BLOCK mapper"
    for key in g["FLOAT_TO_FP8_ROW_MAPPER"]:
        assert "gemma-4" not in key, f"Gemma-4 key {key} in FP8 ROW mapper"


def test_existing_fp8_entries_intact():
    """Existing FP8 entries (e.g., Llama-3.1) should still work."""
    g = _load_mappers()
    FP8B = g["FLOAT_TO_FP8_BLOCK_MAPPER"]
    # Llama-3.1-8B has FP8 entries
    llama_fp8 = [k for k in FP8B if "llama-3.1-8b" in k.lower()]
    assert len(llama_fp8) > 0, "Llama-3.1-8B FP8 entries missing"
