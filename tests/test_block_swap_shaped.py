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


def test_dora_targets_can_stay_resident_beside_evicted_blocks():
    import inspect
    from peft import LoraConfig, get_peft_model

    model = torch.nn.Sequential(
        torch.nn.Linear(8, 4, bias = False), torch.nn.Linear(4, 4, bias = False)
    )
    expected = model[1].weight.detach().norm(dim = 1)
    swapper, weight = _evicted_swapper((4, 8))
    model[0].weight = weight
    model._unsloth_block_swap = swapper
    config = LoraConfig(r = 2, target_modules = ["1"], use_dora = True)
    shaped = _load()["block_swap_shaped"]
    options = (
        {"lora_config": config} if "lora_config" in inspect.signature(shaped).parameters else {}
    )
    with shaped(model, use_dora = True, **options):
        model = get_peft_model(model, config)
    torch.testing.assert_close(
        model.base_model.model[1].lora_magnitude_vector["default"].weight, expected
    )


def test_kasa_never_truncates_placeholder_weights():
    import inspect
    import pytest
    import peft

    if not hasattr(peft, "KasaConfig"):
        pytest.skip("Installed PEFT has no KaSA")
    model = torch.nn.Sequential(torch.nn.Linear(8, 6, bias = False))
    model.requires_grad_(False)
    swapper, weight = _evicted_swapper((6, 8))
    model[0].weight = weight
    model._unsloth_block_swap = swapper
    config = peft.LoraConfig(r = 2, target_modules = ["0"], kasa_config = peft.KasaConfig())
    shaped = _load()["block_swap_shaped"]
    options = (
        {"lora_config": config} if "lora_config" in inspect.signature(shaped).parameters else {}
    )
    with pytest.raises(ValueError, match = "KaSA"):
        with shaped(model, **options):
            peft.get_peft_model(model, config)
    assert weight.numel() == 0


def test_both_adapter_reload_paths_wrap_evicted_experts(tmp_path):
    from peft import LoraConfig, PeftModel, get_peft_model

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = torch.nn.Parameter(torch.randn(4, 16, 8))

    config = LoraConfig(
        r = 2, target_modules = [], target_parameters = ["0.experts"], modules_to_save = ["1"]
    )
    original = torch.nn.Sequential(Experts(), torch.nn.Linear(8, 8, bias = False))
    expected = original[1].weight.detach().clone()
    saved = get_peft_model(original, config)
    saved.save_pretrained(tmp_path)
    source = open(os.path.join(HERE, "unsloth", "models", "loader.py"), encoding = "utf-8").read()
    contexts = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.With)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and isinstance(child.func.value, ast.Name)
            and child.func.value.id == "PeftModel"
            and child.func.attr == "from_pretrained"
            for statement in node.body
            for child in ast.walk(statement)
        )
    ]
    assert len(contexts) == 2
    for node in contexts:
        model = torch.nn.Sequential(Experts(), torch.nn.Linear(8, 8, bias = False))
        swapper, weight = _evicted_swapper()
        model[0].experts = weight
        saved_swapper, saved_weight = _evicted_swapper((8, 8))
        model[1].weight = saved_weight
        swapper.blocks.extend(saved_swapper.blocks)
        model._unsloth_block_swap = swapper
        ns = dict(
            _load(),
            model = model,
            PeftModel = PeftModel,
            contextlib = contextlib,
            fast_pissa = False,
            old_model_name = str(tmp_path),
            token = None,
            revision = None,
            local_files_only = True,
            trust_remote_code = False,
            peft_load_kwargs = {"config": config},
            load_config = config,
        )
        exec(compile(ast.Module(body = [node], type_ignores = []), "loader_reload", "exec"), ns)
        assert isinstance(ns["model"], PeftModel)
        assert weight.numel() == 0
        assert saved_weight.numel() == 0
        torch.testing.assert_close(
            ns["model"].base_model.model[1].modules_to_save["default"].weight, expected
        )
