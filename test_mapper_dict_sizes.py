"""Test that mapper dicts have expected sizes and no entries were accidentally lost."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def _load_mappers():
    g = {}
    exec(open(os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")).read(), g)
    return g


def test_int_to_float_has_gemma4_entries():
    g = _load_mappers()
    ITF = g["INT_TO_FLOAT_MAPPER"]
    gemma4 = [k for k in ITF if "gemma-4" in k]
    # 3 bnb-4bit entries x 2 (original + lowercased) = 6
    assert len(gemma4) >= 6, f"Expected at least 6 Gemma-4 ITF entries, got {len(gemma4)}"


def test_float_to_int_has_gemma4_entries():
    g = _load_mappers()
    FTI = g["FLOAT_TO_INT_MAPPER"]
    gemma4 = [k for k in FTI if "gemma-4" in k]
    # Each bnb-4bit entry adds 2 float names x 2 (original + lowercased) = 12
    assert len(gemma4) >= 12, f"Expected at least 12 Gemma-4 FTI entries, got {len(gemma4)}"


def test_map_to_unsloth_16bit_has_new_entries():
    g = _load_mappers()
    M16 = g["MAP_TO_UNSLOTH_16bit"]
    # Our fix added: 26B-A4B-it, E2B, E4B, 26B-A4B, 31B, LFM2.5 = 6 entries x 2 case = 12
    # Plus the loop-derived instruct entries
    gemma4_16 = [k for k in M16 if "gemma-4" in k]
    assert len(gemma4_16) >= 10, f"Expected at least 10 Gemma-4 M16 entries, got {len(gemma4_16)}"


def test_lfm_entries_present():
    g = _load_mappers()
    ITF = g["INT_TO_FLOAT_MAPPER"]
    M16 = g["MAP_TO_UNSLOTH_16bit"]
    lfm_itf = [k for k in ITF if "lfm" in k.lower()]
    lfm_m16 = [k for k in M16 if "lfm" in k.lower()]
    assert len(lfm_itf) >= 2, f"Expected at least 2 LFM ITF entries, got {len(lfm_itf)}"
    assert len(lfm_m16) >= 2, f"Expected at least 2 LFM M16 entries, got {len(lfm_m16)}"


def test_total_mapper_size_not_shrunk():
    """Total mapper sizes should not have decreased from adding new entries."""
    g = _load_mappers()
    # These are minimum thresholds based on the pre-existing + new entries
    assert len(g["INT_TO_FLOAT_MAPPER"]) > 200
    assert len(g["FLOAT_TO_INT_MAPPER"]) > 200
    assert len(g["MAP_TO_UNSLOTH_16bit"]) > 100
