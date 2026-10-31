"""Block Swap decides on the one-card map before `exclude_no_placement_params` keeps Qwen3.8-Flash-Next's
n-gram table on CPU: on the expanded map both `offload_layers = "auto"` and a Count saw two devices and bailed."""

import ast, os, types

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(HERE, "unsloth", "models")


def _source(name):
    return open(os.path.join(MODELS, name), encoding = "utf-8").read()


def _lines(tree, predicate):
    lines = [n.lineno for n in ast.walk(tree) if predicate(n)]
    assert lines
    return lines


def _calls(name):
    return lambda n: isinstance(n, ast.Call) and getattr(n.func, "id", None) == name


def test_no_placement_exclusion_runs_after_every_block_swap_decision():
    tree = ast.parse(_source("vision.py"))
    (exclude,) = _lines(tree, _calls("exclude_no_placement_params"))
    for decision in ("refuse_block_swap_load", "resolve_auto_block_swap"):
        assert max(_lines(tree, _calls(decision))) < exclude, decision
    pins = [
        n.lineno
        for n in ast.walk(tree)
        if isinstance(n, ast.Assign)
        and getattr(n.targets[0], "id", None) == "device_map"
        and isinstance(n.value, ast.Dict)
        and any(getattr(v, "id", None) == "_block_swap_device" for v in n.value.values)
    ]
    assert pins and max(pins) < exclude


def _block_swap_load_device():
    src = _source("_utils.py")
    node = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.FunctionDef) and n.name == "block_swap_load_device"
    )

    class Device:
        def __init__(
            self,
            kind,
            index = None,
        ):
            if isinstance(kind, str) and ":" in kind:
                kind, index = kind.split(":")[0], int(kind.split(":")[1])
            self.type, self.index = kind, index

    cuda = types.SimpleNamespace(
        is_available = lambda: True, current_device = lambda: 0, device_count = lambda: 1
    )
    ns = {"DEVICE_TYPE_TORCH": "cuda", "torch": types.SimpleNamespace(cuda = cuda, device = Device)}
    exec(ast.get_source_segment(src, node), ns)
    return ns["block_swap_load_device"]


def test_the_expanded_map_is_why_block_swap_bailed():
    device = _block_swap_load_device()
    assert device({"": 0}) == 0
    assert device("sequential") == 0
    # What exclude_no_placement_params makes of {"": 0} for Qwen4Exp: the table's module on CPU, its siblings on 0.
    expanded = {"model.layers": 0, "model.ple.ple_embedding.ngram_embedding": "cpu", "lm_head": 0}
    assert device(expanded) is None
