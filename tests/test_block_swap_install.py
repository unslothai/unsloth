# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Block swap helpers from _utils.py, extracted with ast to avoid importing torch's CUDA stack."""

import ast, os, types
import pytest

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UTILS = os.path.join(HERE, "unsloth", "models", "_utils.py")
NAMES = (
    "_check_block_swap",
    "_new_block_swap",
    "refuse_block_swap_load",
    "_training_reserve_bytes",
    "_auto_block_swap_indices",
    "install_block_swap",
    "trim_config_for_block_swap",
)


def _load(
    *,
    moe = False,
    integrated = False,
    zoo = True,
    cuda = True,
    auto_pick = ([], 0),
):
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    nodes = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in NAMES}
    assert set(nodes) == set(NAMES), f"missing from _utils.py: {set(NAMES) - set(nodes)}"

    calls = []

    class FakeSwap:
        def __init__(
            self,
            layers,
            n,
            depth,
            placement = "tail",
        ):
            calls.append((layers, n, depth, placement))

    ns = {
        "is_moe_model": lambda m: moe,
        "is_integrated_unified_memory_gpu": lambda: integrated,
        "BlockSwap": FakeSwap if zoo else None,
        "build_host_layers": object() if zoo else None,
        "find_decoder_layers": lambda m: m.layers,
        "auto_swap_indices": lambda layers, reserve, depth: auto_pick,
        "_offload_embedding_for_room": lambda model: False,
        "estimate_training_reserve_bytes": lambda config, seq_len, extra_bytes = 0: 2**30,
        "torch": types.SimpleNamespace(cuda = types.SimpleNamespace(is_available = lambda: cuda)),
    }
    for name in NAMES:
        exec(ast.get_source_segment(src, nodes[name]), ns)
    return ns, calls


class _Layers(list):
    pass


class _Model:
    config = None

    def __init__(self, vllm = None):
        self.layers = _Layers(["L0", "L1", "L2"])
        if vllm is not None:
            self.vllm_engine = vllm

    def parameters(self):
        return []


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
    assert calls == [(m.layers, 2, 3, "spread")]
    assert m._unsloth_block_swap is sw
    assert m.layers._unsloth_block_swap is sw


def test_refuses_vllm():
    ns, calls = _load()
    with pytest.raises(ValueError, match = "fast_inference"):
        ns["install_block_swap"](_Model(vllm = object()), 2)
    assert calls == []


def test_moe_installs_with_a_notice(capsys):
    ns, calls = _load(moe = True)
    ns["install_block_swap"](_Model(), 2)
    assert len(calls) == 1 and "MoE" in capsys.readouterr().out


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
    ns, _ = _load(integrated = True)
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
    ns, _ = _load(integrated = True)
    with pytest.raises(ValueError, match = "unified-memory"):
        ns["trim_config_for_block_swap"](_Config(), 4)
    ns, _ = _load(zoo = False)
    with pytest.raises(ImportError, match = "unsloth_zoo"):
        ns["trim_config_for_block_swap"](_Config(), 4)


def _refusals(fn, needle):
    """Lines of `if <needle ...>: block_swap_layers = refuse_block_swap_load(...)` in `fn`."""
    return [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.If)
        and needle in ast.unparse(n.test) + ast.unparse(n.body)
        and "refuse_block_swap_load" in ast.unparse(n.body)
    ]


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
    assert [a.arg for a in fn.args.args[-2:]] == ["block_swap_layers", "checkpoint_skip_layers"]
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
        assert {"block_swap_layers", "checkpoint_skip_layers"} <= {k.arg for k in call.keywords}


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
    refusal = _refusals(fn, "needs a safetensors checkpoint")
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
    assert _refusals(load, "_ckpt_quant_method")


def test_every_custom_decode_loop_is_served():
    # A FastXModel with its own model-level decode loop must call block_swap.enter/leave.
    models = os.path.join(HERE, "unsloth", "models")
    for name in sorted(os.listdir(models)):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(models, name), encoding = "utf-8").read()
        loops = [
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.FunctionDef)
            and n.name.endswith("fast_forward_inference")
            and "Attention" not in n.name
        ]
        if loops:
            assert (
                "block_swap" in src
            ), f"{name} has a decode loop that never fetches swapped blocks"


def test_a_caller_quantization_config_is_refused_before_loading():
    mod = ast.parse(
        open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    )
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    raises = _refusals(fn, "quantization_config")
    trim = [
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "trim_config_for_block_swap"
    ]
    assert raises and trim and raises[0] < trim[0]


def test_every_unsupported_load_mode_is_refused_before_trimming():
    mod = ast.parse(
        open(os.path.join(HERE, "unsloth", "models", "llama.py"), encoding = "utf-8").read()
    )
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == "FastLlamaModel")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_pretrained")
    trim = min(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "trim_config_for_block_swap"
    )
    for needle in (
        "load_in_8bit",
        "quantization_config",
        "state_dict",
        "_ckpt_quant_method",
        "safetensors",
    ):
        hits = _refusals(fn, needle)
        assert hits and min(hits) < trim, needle


def test_older_zoo_without_placement_still_installs():
    ns, calls = _load()

    class OldSwap:
        def __init__(self, layers, n, depth):
            calls.append((layers, n, depth))

    ns["BlockSwap"] = OldSwap
    m = _Model()
    ns["install_block_swap"](m, 2)
    assert calls == [(m.layers, 2, 2)]


def test_auto_swaps_nothing_when_the_card_has_room(capsys):
    ns, calls = _load(auto_pick = ([], 0))
    assert ns["install_block_swap"](_Model(), "auto") is None
    assert calls == [] and "swaps nothing" in capsys.readouterr().out


def test_auto_swaps_the_layers_the_shortfall_needs():
    ns, calls = _load(auto_pick = ([0, 2], 0))
    m = _Model()
    ns["install_block_swap"](m, "auto")
    assert calls == [(m.layers, [0, 2], 2, "spread")]


def test_auto_without_a_gpu_is_a_quiet_no_op():
    ns, calls = _load(cuda = False)
    assert ns["install_block_swap"](_Model(), "auto") is None
    assert calls == []


def test_load_refusals_fall_back_for_auto_and_raise_for_a_count(capsys):
    ns, _ = _load()
    assert ns["refuse_block_swap_load"]("auto", "does not take a state_dict.") == 0
    assert "loads every layer onto the GPU" in capsys.readouterr().out
    with pytest.raises(ValueError, match = "state_dict"):
        ns["refuse_block_swap_load"](4, "does not take a state_dict.")


def test_auto_moves_the_embedding_before_any_layer():
    ns, calls = _load()
    picks = iter([([0, 2], 0), ([], 0)])
    ns["auto_swap_indices"] = lambda layers, reserve, depth: next(picks)
    moved = []
    ns["_offload_embedding_for_room"] = lambda model: moved.append(model) or True
    m = _Model()
    assert ns["install_block_swap"](m, "auto") is None
    assert moved == [m] and calls == []


def _load_host_helpers(device_count = 1, cuda = True):
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    names = ("block_swap_load_device", "begin_block_swap_load")
    nodes = {n.name: n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name in names}
    import contextlib, torch

    seen = []
    ns = {
        "DEVICE_TYPE_TORCH": "cuda",
        "contextlib": contextlib,
        "load_layers_to_host": lambda n, **kw: seen.append((n, kw))
        or contextlib.nullcontext("loading"),
        "torch": types.SimpleNamespace(
            device = torch.device,
            cuda = types.SimpleNamespace(
                is_available = lambda: cuda,
                device_count = lambda: device_count,
                current_device = lambda: 0,
            ),
        ),
    }
    for name in names:
        exec(ast.get_source_segment(src, nodes[name]), ns)
    return ns, seen


@pytest.mark.parametrize(
    "device_map, count, expected",
    [
        ({"": 0}, 2, 0),
        ({"": "cuda:1"}, 2, 1),
        ("cuda:1", 2, 1),
        ("sequential", 1, 0),
        ("sequential", 2, None),  # a strategy would split across the two cards
        ({"a": 0, "b": 1}, 2, None),
        ({"": "cpu"}, 1, None),
        (None, 1, 0),
    ],
)
def test_host_load_needs_one_card(device_map, count, expected):
    ns, _ = _load_host_helpers(device_count = count)
    assert ns["block_swap_load_device"](device_map) == expected


def test_host_load_without_cuda_has_no_card():
    ns, _ = _load_host_helpers(cuda = False)
    assert ns["block_swap_load_device"]({"": 0}) is None


def test_host_load_runs_for_layers_or_streamed_embeddings_only():
    ns, seen = _load_host_helpers()
    with ns["begin_block_swap_load"](0, {"": 0}) as state:
        assert state is None
    with ns["begin_block_swap_load"](0, {"": 0}, embeddings = True):
        pass
    with ns["begin_block_swap_load"](3, {"": 0}):
        pass
    assert seen == [(0, {"placement": "spread", "embeddings": True}), (3, {"placement": "spread"})]


def _load_skip():
    src = open(UTILS, encoding = "utf-8").read()
    mod = ast.parse(src)
    names = ("_skip_aware_flag", "skip_checkpointing")
    ns = {"find_decoder_layers": lambda m: m.layers}
    for n in mod.body:
        if isinstance(n, ast.FunctionDef) and n.name in names:
            exec(ast.get_source_segment(src, n), ns)
    return ns


def _skip_model(n = 4, swapped = None):
    class Layer:
        gradient_checkpointing = False

    model = types.SimpleNamespace(layers = [Layer() for _ in range(n)])
    for layer in model.layers:
        layer.gradient_checkpointing = True
    if swapped is not None:
        model._unsloth_block_swap = types.SimpleNamespace(indices = swapped)
    return model


def test_skipped_layers_stay_off_when_checkpointing_is_turned_back_on():
    ns = _load_skip()
    model = _skip_model()
    assert ns["skip_checkpointing"](model, 2) == [1, 3]
    # for_training / gradient_checkpointing_enable write the flag on every layer again.
    for layer in model.layers:
        layer.gradient_checkpointing = True
    assert [l.gradient_checkpointing for l in model.layers] == [True, False, True, False]
    for layer in model.layers:
        layer.gradient_checkpointing = False
    assert not any(l.gradient_checkpointing for l in model.layers)


def test_skip_leaves_block_swapped_layers_checkpointed():
    ns = _load_skip()
    model = _skip_model(4, swapped = [0, 2])
    assert ns["skip_checkpointing"](model, "max") == [1, 3]
    assert [l.gradient_checkpointing for l in model.layers] == [True, False, True, False]


def test_skip_zero_changes_nothing():
    ns = _load_skip()
    model = _skip_model()
    assert ns["skip_checkpointing"](model, 0) == []
    assert all(l.gradient_checkpointing for l in model.layers)


def _method(path, cls_name, name):
    mod = ast.parse(open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read())
    cls = next(n for n in mod.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    return next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_generic_load_refuses_unified_memory_before_loading_to_host():
    fn = _method("vision.py", "FastBaseModel", "from_pretrained")
    begin = min(
        n.lineno
        for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "begin_block_swap_load"
    )
    hits = _refusals(fn, "is_integrated_unified_memory_gpu")
    assert hits and min(hits) < begin


def test_headless_load_swaps_onto_the_retained_layers_card():
    torch = pytest.importorskip("torch")
    mod = ast.parse(open(UTILS, encoding = "utf-8").read())
    fn = next(
        n for n in mod.body if isinstance(n, ast.FunctionDef) and n.name == "finish_block_swap_load"
    )
    built = []
    card = torch.device("cuda", 1)
    ns = {
        "torch": types.SimpleNamespace(
            device = torch.device, cuda = types.SimpleNamespace(current_device = lambda: 0)
        ),
        "_new_block_swap": lambda layers, idx, depth, device, placement: built.append(device)
        or object(),
    }
    exec(compile(ast.Module(body = [fn], type_ignores = []), UTILS, "exec"), ns)

    class _Layer:
        def __init__(self, device):
            self.device = device

        def parameters(self):
            return [types.SimpleNamespace(device = self.device)]

    class _Headless:
        def get_output_embeddings(self):
            return None

    layers = _Layers([_Layer(card), _Layer(torch.device("cpu")), _Layer(torch.device("cpu"))])
    ns["finish_block_swap_load"](_Headless(), types.SimpleNamespace(layers = layers, indices = [1, 2]))
    assert built == [card]


@pytest.mark.parametrize("path", ["llama.py", "vision.py"])
def test_auto_plan_never_counts_on_an_embedding_offload_the_platform_refuses(path):
    src = open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read()
    call = next(
        n
        for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "resolve_auto_block_swap"
    )
    kw = next(k for k in call.keywords if k.arg == "offload_embedding")
    assert "_offload_embedding_unsupported_platform" in ast.get_source_segment(src, kw.value)


def test_load_time_swappers_allocate_the_planned_prefetch_depth():
    torch = pytest.importorskip("torch")
    mod = ast.parse(open(UTILS, encoding = "utf-8").read())
    fns = [
        n
        for n in mod.body
        if isinstance(n, ast.FunctionDef)
        and n.name in ("finish_block_swap_load", "planned_prefetch_depth")
    ]
    built = []
    ns = {
        "torch": types.SimpleNamespace(
            device = torch.device, cuda = types.SimpleNamespace(current_device = lambda: 0)
        ),
        "_new_block_swap": lambda layers, idx, depth, device, placement: built.append(depth)
        or object(),
    }
    exec(compile(ast.Module(body = fns, type_ignores = []), UTILS, "exec"), ns)

    class _Headless:
        def get_output_embeddings(self):
            return None

    depth = ns["planned_prefetch_depth"]({"prefetch_depth": 1})
    ns["finish_block_swap_load"](
        _Headless(), types.SimpleNamespace(layers = _Layers([]), indices = [0]), depth
    )
    assert built == [1] and ns["planned_prefetch_depth"](None) == 2
    # Both load paths hand the planner's depth to the swapper they build.
    for path, call in (
        ("llama.py", "attach_block_swap_layers"),
        ("vision.py", "finish_block_swap_load"),
    ):
        src = open(os.path.join(HERE, "unsloth", "models", path), encoding = "utf-8").read()
        node = next(
            n
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == call
        )
        assert "planned_prefetch_depth(device_map_planner_kwargs)" in ast.get_source_segment(
            src, node
        )
