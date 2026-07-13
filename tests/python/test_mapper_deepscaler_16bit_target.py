import ast
from pathlib import Path


def _load_int_to_float_mapper():
    # Extract the __INT_TO_FLOAT_MAPPER literal from mapper.py without importing
    # unsloth (importing unsloth needs unsloth_zoo / a GPU). The dict only holds
    # string and tuple literals, so ast.literal_eval is enough.
    source = Path(__file__).parents[2] / "unsloth" / "models" / "mapper.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", "") == "__INT_TO_FLOAT_MAPPER" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("__INT_TO_FLOAT_MAPPER not found in mapper.py")


def _base_name(model_name):
    return (
        model_name.split("/")[-1].lower().replace("-unsloth-bnb-4bit", "").replace("-bnb-4bit", "")
    )


def _sixteen_bit_target(values):
    # A mapper value is either a flat tuple of targets or a dict keyed by bit
    # width ("8"/"16"). Return the first 16bit target in both shapes, or None.
    if isinstance(values, dict):
        values = values.get("16")
    if isinstance(values, tuple) and values:
        return values[0]
    return None


def test_deepscaler_dynamic_quant_maps_to_its_own_16bit_model():
    # The dynamic 4bit key must resolve to the unsloth 16bit mirror of the SAME
    # model, not to an unrelated one (DeepHermes-3-Llama-3-8B). values[0] is
    # fanned out into INT_TO_FLOAT_MAPPER / MAP_TO_UNSLOTH_16bit / FLOAT_TO_INT_
    # MAPPER, so a wrong value silently mis-routes every DeepScaleR load.
    mapper = _load_int_to_float_mapper()
    key = "unsloth/DeepScaleR-1.5B-Preview-unsloth-bnb-4bit"
    assert mapper[key][0] == "unsloth/DeepScaleR-1.5B-Preview"


def test_unsloth_dynamic_quant_values_point_at_matching_16bit_model():
    # General invariant for every "-unsloth-bnb-4bit" entry whose 16bit target is
    # an unsloth mirror: the base model name of the 16bit target must match the
    # key's. This covers both the flat-tuple entries and the dict-structured
    # ones (Llama-3.x, Qwen3, ...) that key their targets by bit width.
    mapper = _load_int_to_float_mapper()
    mismatched = []
    for key, values in mapper.items():
        if not key.endswith("-unsloth-bnb-4bit"):
            continue
        target = _sixteen_bit_target(values)
        if target and target.startswith("unsloth/") and _base_name(target) != _base_name(key):
            mismatched.append((key, target))
    assert mismatched == []
