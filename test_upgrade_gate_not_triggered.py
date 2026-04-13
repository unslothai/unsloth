"""Test that new mapper entries don't trigger the false 'upgrade Unsloth' gate."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def _load_mappers():
    g = {}
    exec(open(os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")).read(), g)
    return g


def _simulate_get_model_name(model_name, load_in_4bit, ITF, FTI, M16):
    """Simulate __get_model_name and return (result, would_check_upgrade)."""
    lower = model_name.lower()
    if not load_in_4bit and lower in ITF:
        return ITF[lower], False
    elif not load_in_4bit and lower in M16:
        return M16[lower], False
    elif load_in_4bit and lower in FTI:
        if lower.endswith("-bnb-4bit"):
            return model_name, False
        return FTI[lower], False
    return None, True  # None triggers the upgrade check


def test_gemma4_instruct_4bit_no_upgrade_check():
    g = _load_mappers()
    result, upgrade = _simulate_get_model_name(
        "google/gemma-4-E2B-it", True,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert not upgrade, f"Would trigger upgrade check! result={result}"
    assert result is not None


def test_gemma4_instruct_16bit_no_upgrade_check():
    g = _load_mappers()
    result, upgrade = _simulate_get_model_name(
        "google/gemma-4-E2B-it", False,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert not upgrade, f"Would trigger upgrade check! result={result}"


def test_lfm2_4bit_no_upgrade_check():
    g = _load_mappers()
    result, upgrade = _simulate_get_model_name(
        "LiquidAI/LFM2-1.2B", True,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert not upgrade, f"Would trigger upgrade check! result={result}"


def test_lfm25_16bit_no_upgrade_check():
    g = _load_mappers()
    result, upgrade = _simulate_get_model_name(
        "LiquidAI/LFM2.5-1.2B-Instruct", False,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert not upgrade, f"Would trigger upgrade check! result={result}"


def test_gemma4_base_16bit_no_upgrade_check():
    """Base models with MAP_TO_UNSLOTH_16bit redirect should not trigger upgrade."""
    g = _load_mappers()
    result, upgrade = _simulate_get_model_name(
        "google/gemma-4-E2B", False,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert not upgrade, f"Would trigger upgrade check! result={result}"
