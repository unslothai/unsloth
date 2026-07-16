import ast
from pathlib import Path


def _load_function(name):
    # Extract a function from kernels/utils.py without importing unsloth (which
    # needs a GPU / torch / bitsandbytes). The functions exercised here only use
    # getattr and the _FP8_WEIGHT_DTYPES name on the paths under test.
    source = Path(__file__).parents[2] / "unsloth" / "kernels" / "utils.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(funcs) == 1, (name, funcs)
    namespace = {"getattr": getattr, "_FP8_WEIGHT_DTYPES": ()}
    module = ast.Module(body = funcs, type_ignores = [])
    ast.fix_missing_locations(module)
    exec(compile(module, str(source), "exec"), namespace)
    return namespace[name]


class _Obj:
    pass


def _make_disabled_block_fp8_proj(block_size):
    # A merged/disabled projection whose base layer is a block-fp8 weight that
    # ships a non-default block size on its checkpoint.
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
    # Downstream fp8 kernels read getattr(weight_scale, "block_size", [128, 128]),
    # so the checkpoint's real block size must survive the merged/disabled path,
    # exactly as it does for the non-bias sibling get_lora_parameters.
    get_lora_parameters_bias = _load_function("get_lora_parameters_bias")

    proj, weight_scale = _make_disabled_block_fp8_proj([64, 128])
    get_lora_parameters_bias(proj)

    assert getattr(weight_scale, "block_size", [128, 128]) == [64, 128]
