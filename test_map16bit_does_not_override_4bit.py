"""Test that MAP_TO_UNSLOTH_16bit doesn't interfere with the load_in_4bit=True path."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))


def _load_mappers():
    g = {}
    exec(open(os.path.join(os.path.dirname(__file__), "unsloth", "models", "mapper.py")).read(), g)
    return g


def _simulate_loader(model_name, load_in_4bit, ITF, FTI, M16):
    """Return (result, which_path) matching loader_utils.__get_model_name priority."""
    lower = model_name.lower()
    if not load_in_4bit and lower in ITF:
        return ITF[lower], "int_to_float"
    elif not load_in_4bit and lower in M16:
        return M16[lower], "map_to_16bit"
    elif load_in_4bit and lower in FTI:
        if lower.endswith("-bnb-4bit"):
            return model_name, "bnb_passthrough"
        return FTI[lower], "float_to_int"
    return None, "fallthrough"


def test_4bit_path_takes_precedence_over_16bit():
    """When load_in_4bit=True and model is in FLOAT_TO_INT, the 4-bit path wins."""
    g = _load_mappers()
    result, path = _simulate_loader(
        "google/gemma-4-E2B-it", True,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert path == "float_to_int", f"Expected float_to_int, got {path}"
    assert "bnb-4bit" in result


def test_16bit_path_used_when_4bit_false():
    """When load_in_4bit=False, MAP_TO_UNSLOTH_16bit is used for models not in ITF."""
    g = _load_mappers()
    result, path = _simulate_loader(
        "google/gemma-4-26B-A4B-it", False,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert path == "map_to_16bit", f"Expected map_to_16bit, got {path}"
    assert result == "unsloth/gemma-4-26B-A4B-it"


def test_base_model_4bit_falls_through():
    """Base models without bnb-4bit entries should fall through when load_in_4bit=True."""
    g = _load_mappers()
    result, path = _simulate_loader(
        "google/gemma-4-E2B", True,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert path == "fallthrough", f"Expected fallthrough, got {path} (result={result})"


def test_lfm25_16bit_uses_map_to_16bit():
    g = _load_mappers()
    result, path = _simulate_loader(
        "LiquidAI/LFM2.5-1.2B-Instruct", False,
        g["INT_TO_FLOAT_MAPPER"], g["FLOAT_TO_INT_MAPPER"], g["MAP_TO_UNSLOTH_16bit"]
    )
    assert path == "map_to_16bit"
    assert result == "unsloth/LFM2.5-1.2B-Instruct"
