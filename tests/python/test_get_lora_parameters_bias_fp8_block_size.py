import ast
from pathlib import Path


_HELPERS = ("_packed_base", "_is_packed_state")


def _load_function(name):
    # Extract without importing unsloth, which needs a GPU.
    source = Path(__file__).parents[2] / "unsloth" / "kernels" / "utils.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name in (name, *_HELPERS)
    ]
    assert [f.name for f in funcs].count(name) == 1, (name, funcs)
    namespace = {"getattr": getattr, "_FP8_WEIGHT_DTYPES": ()}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


class _Obj:
    pass


def _make_disabled_block_fp8_proj(block_size):
    weight = _Obj()
    weight.quant_state = _Obj()
    base_layer = _Obj()
    base_layer.weight = weight
    base_layer.quant_method = "fp8"
    base_layer.block_size = block_size
    base_layer.bias = None
    proj = _Obj()
    proj.base_layer = base_layer
    proj.merged = True
    proj.disable_adapters = True
    return proj, weight.quant_state


def test_bias_variant_propagates_fp8_block_size_on_disabled_path():
    # fp8 kernels read weight_scale.block_size, so the real block size must survive this path.
    get_lora_parameters_bias = _load_function("get_lora_parameters_bias")

    proj, weight_scale = _make_disabled_block_fp8_proj([64, 128])
    get_lora_parameters_bias(proj)

    assert getattr(weight_scale, "block_size", [128, 128]) == [64, 128]


def _make_decompressed_merged_proj():
    weight = _Obj()
    weight.dtype = "bfloat16"
    base_layer = _Obj()
    base_layer.weight = weight
    base_layer.quant_method = "fp8"
    base_layer.block_size = [128, 128]
    base_layer.bias = None
    proj = _Obj()
    proj.base_layer = base_layer
    proj.merged = True
    proj.disable_adapters = True
    return proj


def test_bias_variant_keeps_none_quant_state_for_decompressed_layer():
    # No quant state here; fast_linear_forward needs W_quant None to fall back to matmul.
    get_lora_parameters_bias = _load_function("get_lora_parameters_bias")

    W, W_quant = get_lora_parameters_bias(_make_decompressed_merged_proj())[:2]

    assert W_quant is None
    assert getattr(W, "block_size", None) == [128, 128]
