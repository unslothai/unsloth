# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block swap helpers from _utils.py, extracted with ast to avoid importing torch's CUDA stack."""

import ast, os, types
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = ("_check_block_swap", "install_block_swap", "trim_config_for_block_swap")


def _load(
    *,
    moe = False,
    integrated = False,
    zoo = True,
    cuda = True,
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
        "torch": types.SimpleNamespace(cuda = types.SimpleNamespace(is_available = lambda: cuda)),
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
    def __init__(
        self,
        n = 8,
        **extra,
    ):
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


def _peft_signature_and_calls(path):
    mod = ast.parse(open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read())
    for cls in (n for n in mod.body if isinstance(n, ast.ClassDef)):
        for fn in cls.body:
            if isinstance(fn, ast.FunctionDef) and fn.name == "get_peft_model":
                return fn
    raise AssertionError(f"get_peft_model not found in {path}")


@pytest.mark.parametrize("path", ["llama.py", "vision.py"])
def test_block_swap_layers_is_last_so_positional_callers_keep_their_slots(path):
    fn = _peft_signature_and_calls(path)
    assert fn.args.args[-1].arg == "block_swap_layers"
    assert fn.args.kwarg is not None


def test_new_model_route_forwards_block_swap_layers():
    fn = _peft_signature_and_calls("llama.py")
    forwarded = [
        call
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "get_peft_model"
        and getattr(call.func.value, "id", None) == "FastBaseModel"
    ]
    assert forwarded, "llama get_peft_model no longer delegates to FastBaseModel"
    for call in forwarded:
        assert "block_swap_layers" in {k.arg for k in call.keywords}


@pytest.mark.parametrize("path", ["llama.py", "gemma.py", "gemma2.py"])
def test_fast_decode_loops_fetch_swapped_layers(path):
    # Decode loops read layer weights without calling the layer, so the swap hooks never fire.
    src = open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read()
    assert "block_swap.enter(idx)" in src and "block_swap.leave(idx)" in src


def _load_key_filter(monkeypatch, has_method = True):
    import re, sys, types

    mod = ast.parse(open(UTILS, encoding = "utf-8").read())
    keep = [
        n
        for n in mod.body
        if (isinstance(n, ast.FunctionDef) and n.name == "skip_swapped_checkpoint_keys")
        or (
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", None) == "_SWAPPED_LAYER_KEY" for t in n.targets)
        )
    ]
    assert len(keep) == 2
    seen = []

    class PreTrainedModel:
        pass

    if has_method:

        def _get_key_renaming_mapping(
            self,
            checkpoint_keys,
            key_mapping = None,
        ):
            seen.append(list(checkpoint_keys))
            return {k: k for k in checkpoint_keys}

        PreTrainedModel._get_key_renaming_mapping = _get_key_renaming_mapping
    fake = types.ModuleType("transformers.modeling_utils")
    fake.PreTrainedModel = PreTrainedModel
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake)
    ns = {"re": re}
    exec(compile(ast.Module(body = keep, type_ignores = []), UTILS, "exec"), ns)
    return ns["skip_swapped_checkpoint_keys"], PreTrainedModel, seen


def test_swapped_tail_keys_are_hidden_from_a_4x_load_then_restored(monkeypatch):
    skip, cls, seen = _load_key_filter(monkeypatch)
    original = cls.__dict__["_get_key_renaming_mapping"]
    undo = skip({"num_hidden_layers": 4}, 2)
    keys = [
        "model.embed_tokens.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.layers.2.mlp.down_proj.weight",
        "model.layers.3.self_attn.q_proj.weight.absmax",
        "lm_head.weight",
    ]
    cls()._get_key_renaming_mapping(keys, key_mapping = None)
    assert seen[-1] == [
        "model.embed_tokens.weight",
        "model.layers.1.mlp.down_proj.weight",
        "lm_head.weight",
    ]
    undo()
    assert cls.__dict__["_get_key_renaming_mapping"] is original


def test_key_filter_is_inert_when_off_or_on_5x(monkeypatch):
    skip, cls, _ = _load_key_filter(monkeypatch)
    original = cls.__dict__["_get_key_renaming_mapping"]
    skip(None, 2)()
    assert cls.__dict__["_get_key_renaming_mapping"] is original
    skip, cls, _ = _load_key_filter(monkeypatch, has_method = False)
    skip({"num_hidden_layers": 4}, 2)()
    assert "_get_key_renaming_mapping" not in cls.__dict__


def test_checkpoint_tensors_reads_the_requested_variant(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")
    mod = ast.parse(open(UTILS, encoding = "utf-8").read())
    fn = next(
        n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "_checkpoint_tensors"
    )
    ns = {"os": os}
    exec(compile(ast.Module(body = [fn], type_ignores = []), UTILS, "exec"), ns)
    safetensors_torch.save_file({"w": torch.zeros(2)}, str(tmp_path / "model.safetensors"))
    safetensors_torch.save_file({"w": torch.ones(2)}, str(tmp_path / "model.fp16.safetensors"))
    tensors, _ = ns["_checkpoint_tensors"](str(tmp_path), variant = "fp16")
    assert torch.equal(tensors["w"](), torch.ones(2))
    tensors, _ = ns["_checkpoint_tensors"](str(tmp_path))
    assert torch.equal(tensors["w"](), torch.zeros(2))


def test_non_safetensors_formats_are_refused_before_the_prefix_loads():
    mod = ast.parse(
        open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    )
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    refusal = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Raise) and "needs a safetensors checkpoint" in ast.unparse(n)
    ]
    trim = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "trim_config_for_block_swap"
    ]
    assert refusal and trim and refusal[0] < trim[0]


def test_refuses_without_a_cuda_or_rocm_gpu():
    ns, calls = _load(cuda = False)
    with pytest.raises(ValueError, match = "CUDA or ROCm"):
        ns["install_block_swap"](_Model(), 4)
    assert not calls


def test_prewrapped_peft_and_quantized_checkpoints_are_covered():
    src = open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    mod = ast.parse(src)
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    peft = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_peft_model"
    )
    installs = [
        n
        for n in ast.walk(peft)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "install_block_swap"
    ]
    assert len(installs) >= 2, "the pre-wrapped PEFT return path must install the swap too"
    load = next(
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained"
    )
    assert any(
        isinstance(n, ast.Raise) and "_ckpt_quant_method" in ast.unparse(n) for n in ast.walk(load)
    )
