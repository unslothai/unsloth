"""Tests install_block_swap in _utils.py: off at 0, the three refusals name their
reason, a missing unsloth_zoo module is an ImportError, and the happy path hands
the decoder layers to BlockSwap and records the swapper on the model. Extracted
with ast so nothing has to import torch's CUDA stack."""

import ast, os
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")


def _load(
    *,
    moe = False,
    integrated = False,
    zoo = True,
):
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    for node in mod.body:
        if isinstance(node, ast.FunctionDef) and node.name == "install_block_swap":
            break
    else:
        raise AssertionError("install_block_swap not found in _utils.py")

    calls = []

    class FakeSwap:
        def __init__(self, layers, n, depth):
            calls.append((layers, n, depth))

    ns = {
        "is_moe_model": lambda m: moe,
        "is_integrated_unified_memory_gpu": lambda: integrated,
        "BlockSwap": FakeSwap if zoo else None,
        "find_decoder_layers": lambda m: m.layers,
    }
    exec(ast.get_source_segment(src, node), ns)
    return ns["install_block_swap"], calls


class _Model:
    def __init__(self, vllm = None):
        self.layers = ["L0", "L1", "L2"]
        if vllm is not None:
            self.vllm_engine = vllm


def test_zero_is_a_no_op():
    install, calls = _load()
    m = _Model()
    assert install(m, 0) is None
    assert install(m, None) is None
    assert calls == [] and not hasattr(m, "_unsloth_block_swap")


def test_happy_path_installs_and_records():
    install, calls = _load()
    m = _Model()
    sw = install(m, 2, prefetch_depth = 3)
    assert calls == [(m.layers, 2, 3)]
    assert m._unsloth_block_swap is sw


def test_refuses_vllm():
    install, calls = _load()
    with pytest.raises(ValueError, match = "fast_inference"):
        install(_Model(vllm = object()), 2)
    assert calls == []


def test_refuses_moe():
    install, calls = _load(moe = True)
    with pytest.raises(ValueError, match = "MoE"):
        install(_Model(), 2)
    assert calls == []


def test_refuses_unified_memory():
    install, calls = _load(integrated = True)
    with pytest.raises(ValueError, match = "unified-memory"):
        install(_Model(), 2)
    assert calls == []


def test_old_zoo_is_an_import_error():
    install, calls = _load(zoo = False)
    with pytest.raises(ImportError, match = "unsloth_zoo"):
        install(_Model(), 2)
    assert calls == []


def test_refusal_order_checks_cheap_things_first():
    # vLLM is checked before MoE so a fast_inference user gets that reason,
    # not a MoE one, when both apply.
    install, _ = _load(moe = True)
    with pytest.raises(ValueError, match = "fast_inference"):
        install(_Model(vllm = object()), 2)
