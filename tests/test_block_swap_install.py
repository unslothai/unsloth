"""Block swap helpers from _utils.py, extracted with ast to avoid importing torch's CUDA stack."""

import ast, os
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = ("_check_block_swap", "install_block_swap", "trim_config_for_block_swap")


def _load(
    *,
    moe = False,
    integrated = False,
    zoo = True,
):
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    nodes = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    assert set(nodes) == set(NAMES), f"missing from _utils.py: {set(NAMES) - set(nodes)}"

    calls = []

    class FakeSwap:
        def __init__(self, layers, n, depth):
            calls.append((layers, n, depth))

    ns = {
        "is_moe_model": lambda m: moe,
        "is_integrated_unified_memory_gpu": lambda: integrated,
        "BlockSwap": FakeSwap if zoo else None,
        "build_host_layers": object() if zoo else None,
        "find_decoder_layers": lambda m: m.layers,
    }
    for name in NAMES:
        exec(ast.get_source_segment(src, nodes[name]), ns)
    return ns, calls


class _Layers(list):
    pass


class _Model:
    def __init__(self, vllm = None):
        self.layers = _Layers(["L0", "L1", "L2"])
        if vllm is not None:
            self.vllm_engine = vllm


def test_zero_is_a_no_op():
    ns, calls = _load()
    m = _Model()
    assert ns["install_block_swap"](m, 0) is None
    assert ns["install_block_swap"](m, None) is None
    assert calls == [] and not hasattr(m, "_unsloth_block_swap")


def test_happy_path_installs_and_records():
    ns, calls = _load()
    m = _Model()
    sw = ns["install_block_swap"](m, 2, prefetch_depth = 3)
    assert calls == [(m.layers, 2, 3)]
    assert m._unsloth_block_swap is sw
    assert m.layers._unsloth_block_swap is sw


def test_refuses_vllm():
    ns, calls = _load()
    with pytest.raises(ValueError, match = "fast_inference"):
        ns["install_block_swap"](_Model(vllm = object()), 2)
    assert calls == []


def test_refuses_moe():
    ns, calls = _load(moe = True)
    with pytest.raises(ValueError, match = "MoE"):
        ns["install_block_swap"](_Model(), 2)
    assert calls == []


def test_refuses_unified_memory():
    ns, calls = _load(integrated = True)
    with pytest.raises(ValueError, match = "unified-memory"):
        ns["install_block_swap"](_Model(), 2)
    assert calls == []


def test_refuses_without_gradient_checkpointing():
    ns, calls = _load()
    for off in (False, None):
        with pytest.raises(ValueError, match = "use_gradient_checkpointing"):
            ns["install_block_swap"](_Model(), 2, use_gradient_checkpointing = off)
    assert calls == []


def test_old_zoo_is_an_import_error():
    ns, calls = _load(zoo = False)
    with pytest.raises(ImportError, match = "unsloth_zoo"):
        ns["install_block_swap"](_Model(), 2)
    assert calls == []


def test_refusal_order_checks_cheap_things_first():
    ns, _ = _load(moe = True)
    with pytest.raises(ValueError, match = "fast_inference"):
        ns["install_block_swap"](_Model(vllm = object()), 2)


def test_swapper_from_load_is_reused():
    ns, calls = _load()
    m = _Model()
    m._unsloth_block_swap = existing = object()
    assert ns["install_block_swap"](m, 0) is existing
    assert ns["install_block_swap"](m, 8) is existing
    assert calls == []
    with pytest.raises(ValueError, match = "use_gradient_checkpointing"):
        ns["install_block_swap"](m, 0, use_gradient_checkpointing = False)


class _Config:
    def __init__(self, n = 8, **extra):
        self.num_hidden_layers = n
        self.layer_types = ["full_attention"] * n
        self.other_list = [1, 2, 3]
        self.__dict__.update(extra)


def test_trim_off_at_zero():
    ns, _ = _load()
    cfg = _Config()
    assert ns["trim_config_for_block_swap"](cfg, 0) is None
    assert cfg.num_hidden_layers == 8 and len(cfg.layer_types) == 8


def test_trim_shortens_per_layer_lists_and_returns_originals():
    ns, _ = _load()
    cfg = _Config()
    saved = ns["trim_config_for_block_swap"](cfg, 5)
    assert cfg.num_hidden_layers == 3 and cfg.layer_types == ["full_attention"] * 3
    assert cfg.other_list == [1, 2, 3], "lists not sized to the layer count are untouched"
    assert saved == {"num_hidden_layers": 8, "layer_types": ["full_attention"] * 8}


def test_trim_keeps_at_least_one_layer_on_the_card():
    ns, _ = _load()
    cfg = _Config()
    ns["trim_config_for_block_swap"](cfg, 100)
    assert cfg.num_hidden_layers == 1


def test_trim_refuses_what_install_refuses():
    ns, _ = _load(moe = True)
    with pytest.raises(ValueError, match = "MoE"):
        ns["trim_config_for_block_swap"](_Config(), 4)
    ns, _ = _load(zoo = False)
    with pytest.raises(ImportError, match = "unsloth_zoo"):
        ns["trim_config_for_block_swap"](_Config(), 4)
