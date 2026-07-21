import ast
from pathlib import Path


def _load_function(name):
    # Extract a function from kernels/utils.py without importing unsloth (which
    # needs a GPU / torch / bitsandbytes). get_lora_parameters only uses getattr,
    # hasattr and the _FP8_WEIGHT_DTYPES name on the paths under test.
    source = Path(__file__).parents[2] / "unsloth" / "kernels" / "utils.py"
    tree = ast.parse(source.read_text(encoding = "utf-8"))
    funcs = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(funcs) == 1, (name, funcs)
    namespace = {"getattr": getattr, "hasattr": hasattr, "_FP8_WEIGHT_DTYPES": ()}
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
    proj = _Obj()
    proj.base_layer = base_layer
    proj.merged = True
    proj.disable_adapters = True
    return proj, weight.quant_state


def test_propagates_fp8_block_size_on_disabled_path():
    # get_lora_parameters already sets block_size before its early return; downstream
    # fp8 kernels read getattr(weight_scale, "block_size", [128, 128]), so the
    # checkpoint's real block size must survive the merged/disabled path.
    get_lora_parameters = _load_function("get_lora_parameters")

    proj, weight_scale = _make_disabled_block_fp8_proj([64, 128])
    get_lora_parameters(proj)

    assert getattr(weight_scale, "block_size", [128, 128]) == [64, 128]


def _make_decompressed_merged_proj():
    # A merged compressed-tensors layer that was decompressed back to bf16. It keeps
    # quant_method == "fp8" from the checkpoint metadata, but the live weight is bf16
    # so there is no quant state to attach a block size to.
    weight = _Obj()
    weight.dtype = "bfloat16"
    base_layer = _Obj()
    base_layer.weight = weight
    base_layer.quant_method = "fp8"
    base_layer.block_size = [128, 128]
    proj = _Obj()
    proj.base_layer = base_layer
    proj.merged = True
    proj.disable_adapters = True
    return proj


def test_keeps_none_quant_state_for_decompressed_layer():
    # Mirrors the get_lora_parameters_bias guard: with no quant state, assigning
    # W_quant.block_size must not assume one is present, or it raises AttributeError
    # on None. fast_lora relies on getting W_quant None back to fall back to a plain
    # matmul, so this path must stay crash-free.
    get_lora_parameters = _load_function("get_lora_parameters")

    W, W_quant = get_lora_parameters(_make_decompressed_merged_proj())[:2]

    assert W_quant is None
    assert getattr(W, "block_size", None) == [128, 128]
