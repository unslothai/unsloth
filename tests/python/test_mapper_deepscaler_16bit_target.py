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
        model_name.split("/")[-1]
        .lower()
        .replace("-unsloth-bnb-4bit", "")
        .replace("-bnb-4bit", "")
    )


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
    # an unsloth mirror: the base model name of values[0] must match the key's.
    mapper = _load_int_to_float_mapper()
    mismatched = [
        (key, values[0])
        for key, values in mapper.items()
        if key.endswith("-unsloth-bnb-4bit")
        and isinstance(values, tuple)
        and values
        and values[0].startswith("unsloth/")
        and _base_name(values[0]) != _base_name(key)
    ]
    assert mismatched == []
