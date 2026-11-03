# SPDX-License-Identifier: AGPL-3.0-only
import ast, contextlib, os, types
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = ("block_swap_shaped", "_block_swap_shaped_fallback")


def _load():
    src = open(UTILS, encoding = "utf-8").read()
    nodes = {
        n.name: n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in NAMES
    }
    assert set(nodes) == set(NAMES)
    ns = {"contextlib": contextlib}
    for name in NAMES:
        exec(ast.get_source_segment(src, nodes[name]), ns)
    ns["_block_swap_shaped_fallback"] = contextlib.contextmanager(ns["_block_swap_shaped_fallback"])
    return ns


def _evicted_swapper(shape = (4, 16, 8)):
    w = torch.nn.Parameter(torch.randn(shape), requires_grad = False)
    empty = torch.empty(0)
    w.data = empty
    block = types.SimpleNamespace(
        resident = False, params = [w], empties = [empty], sig = ((shape, w.dtype, "cpu", 0),)
    )
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
    for init in ("pissa", "pissa_niter_4", "olora", "loftq", "corda", "orthogonal"):
        swapper, w = _evicted_swapper()
        with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = swapper), init):
            assert w.numel() == 0, init
    for init in ("gaussian", "eva"):
        swapper, w = _evicted_swapper()
        with ns["block_swap_shaped"](types.SimpleNamespace(_unsloth_block_swap = swapper), init):
            assert w.shape == (4, 16, 8), init


def test_fallback_leaves_a_block_fetched_inside_the_context():
    w = torch.nn.Parameter(torch.empty(0), requires_grad = False)
    blk = types.SimpleNamespace(resident = False, params = [w], empties = [w.data], sig = [((4, 8),)])
    with _load()["_block_swap_shaped_fallback"](types.SimpleNamespace(blocks = [blk])):
        w.data, blk.resident = torch.ones(4, 8), True
    assert w.shape == (4, 8)


def test_dora_rejects_evicted_weights_before_initializing_magnitudes():
    import inspect
    import pytest
    from peft import LoraConfig, get_peft_model

    model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias = False))
    model.requires_grad_(False)
    swapper, weight = _evicted_swapper((4, 8))
    model[0].weight = weight
    model._unsloth_block_swap = swapper
    config = LoraConfig(r = 2, target_modules = ["0"], use_dora = True)
    shaped = _load()["block_swap_shaped"]
    options = (
        {"use_dora": config.use_dora} if "use_dora" in inspect.signature(shaped).parameters else {}
    )
    with pytest.raises(ValueError, match = "DoRA"):
        with shaped(model, **options):
            get_peft_model(model, config)
    assert weight.numel() == 0


def test_dora_without_block_swap_keeps_real_magnitudes():
    from peft import LoraConfig, get_peft_model
    for resident_swap in (False, True):
        model = torch.nn.Sequential(torch.nn.Linear(8, 4, bias = False))
        expected = model[0].weight.detach().norm(dim = 1)
        if resident_swap:
            model._unsloth_block_swap = types.SimpleNamespace(
                blocks = [types.SimpleNamespace(resident = True)]
            )
        config = LoraConfig(r = 2, target_modules = ["0"], use_dora = True)
        with _load()["block_swap_shaped"](model, use_dora = config.use_dora):
            model = get_peft_model(model, config)
        magnitude = model.base_model.model[0].lora_magnitude_vector["default"].weight
        torch.testing.assert_close(magnitude, expected)


def test_modules_to_save_rejects_copying_evicted_weights():
    import inspect
    import pytest
    from peft import LoraConfig, get_peft_model

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias = False), torch.nn.Linear(4, 4, bias = False)
    )
    model.requires_grad_(False)
    swapper, weight = _evicted_swapper((4, 8))
    model[0].weight = weight
    model._unsloth_block_swap = swapper
    config = LoraConfig(r = 2, target_modules = ["1"], modules_to_save = ["0"])
    shaped = _load()["block_swap_shaped"]
    options = (
        {"modules_to_save": config.modules_to_save}
        if "modules_to_save" in inspect.signature(shaped).parameters
        else {}
    )
    with pytest.raises(ValueError, match = "modules_to_save"):
        with shaped(model, **options):
            get_peft_model(model, config)
    assert weight.numel() == 0


def test_modules_to_save_preserves_resident_modules_beside_evicted_weights():
    from peft import LoraConfig, get_peft_model

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias = False), torch.nn.Linear(4, 4, bias = False)
    )
    expected = model[1].weight.detach().clone()
    swapper, weight = _evicted_swapper((4, 8))
    model[0].weight = weight
    model._unsloth_block_swap = swapper
    config = LoraConfig(r = 2, target_modules = ["0"], modules_to_save = ["1"])
    with _load()["block_swap_shaped"](model, modules_to_save = config.modules_to_save):
        model = get_peft_model(model, config)
    saved = model.base_model.model[1].modules_to_save["default"].weight
    torch.testing.assert_close(saved, expected)
