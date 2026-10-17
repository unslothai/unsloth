"""block_swap_shaped from _utils.py, extracted with ast: PEFT must see an evicted MoE expert weight's real shape."""

import ast, contextlib, os, types
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = ("block_swap_shaped", "_block_swap_shaped_fallback")


def _load():
    src = open(UTILS, encoding = "utf-8").read()
    nodes = {n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    assert set(nodes) == set(NAMES)
    ns = {"contextlib": contextlib}
    for name in NAMES:
        exec(ast.get_source_segment(src, nodes[name]), ns)
    ns["_block_swap_shaped_fallback"] = contextlib.contextmanager(ns["_block_swap_shaped_fallback"])
    return ns


def _evicted_swapper(shape = (4, 16, 8)):
    """A zoo-2026.10.6-style swapper: no shaped(), one evicted block holding an expert weight."""
    w = torch.nn.Parameter(torch.randn(shape), requires_grad = False)
    empty = torch.empty(0)
    w.data = empty
    block = types.SimpleNamespace(resident = False, params = [w], empties = [empty], sig = ((shape, w.dtype, "cpu", 0),))
    return types.SimpleNamespace(blocks = [block]), w


def test_older_zoo_still_gets_the_real_shape_while_peft_wraps():
    ns = _load()
    swapper, w = _evicted_swapper()
    with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = swapper)):
        assert w.shape == (4, 16, 8)
        assert w.untyped_storage().nbytes() == w.element_size()
    assert w.numel() == 0


def test_zoo_shaped_is_used_when_present():
    ns = _load()
    entered = []

    @contextlib.contextmanager
    def shaped():
        entered.append(True)
        yield

    model = types.SimpleNamespace(_unsloth_block_swap = types.SimpleNamespace(shaped = shaped))
    with ns["block_swap_shaped"](model):
        pass
    assert entered == [True]


def test_no_swap_and_unknown_swappers_are_no_ops():
    ns = _load()
    with ns["block_swap_shaped"](types.SimpleNamespace()):
        pass
    with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = object())):
        pass


def test_data_dependent_inits_never_see_the_placeholder():
    ns = _load()
    for init in ("pissa", "pissa_niter_4", "olora", "loftq", "eva", "corda"):
        swapper, w = _evicted_swapper()
        with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = swapper), init):
            assert w.numel() == 0, init
    swapper, w = _evicted_swapper()
    with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = swapper), "gaussian"):
        assert w.shape == (4, 16, 8)
