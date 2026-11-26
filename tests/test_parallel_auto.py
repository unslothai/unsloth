# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""UNSLOTH_PARALLEL=auto: DDP when the model and its training state fit on every rank, else native FSDP2.

* parsing: `auto` is its own value, an explicit layout parses and translates exactly as before, and what
  already fixes the layout (UNSLOTH_FSDP2, tp / pp / cp, an accelerate FSDP config) is refused;
* sizing: the safetensors headers of a tiny checkpoint, 16-bit vs 4-bit on load;
* on a gloo CPU world of 2: fits -> ddp, does not fit -> fsdp2, ranks disagree -> fsdp2 on every rank;
* end to end: a Trainer under auto trains exactly like the explicit layout it picked (UNSLOTH_FSDP2=0 / 1)."""

import multiprocessing
import os

if multiprocessing.parent_process() is not None or getattr(
    multiprocessing.current_process(), "_inheriting", False
):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # spawned gloo ranks: CPU only, also on GPU hosts

import importlib  # noqa: E402
import sys  # noqa: E402
import types  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fsdp2_native_units import REPO, _load_native  # noqa: E402

TINY = "trl-internal-testing/tiny-Qwen3ForCausalLM"
STEPS = 2
_ENV = (
    "UNSLOTH_PARALLEL",
    "UNSLOTH_PARALLEL_AUTO_RESOLVED",
    "UNSLOTH_PARALLEL_AUTO_GB",
    "UNSLOTH_FSDP2",
    "UNSLOTH_TP_SIZE",
    "UNSLOTH_PP_SIZE",
    "UNSLOTH_CP_SIZE",
    "ACCELERATE_USE_FSDP",
    "FSDP_VERSION",
    "WORLD_SIZE",
)


def _modules():
    """unsloth.parallel_env / unsloth.parallel_auto from this tree without importing the GPU package."""
    for name, path in (("unsloth", ("unsloth",)), ("unsloth.models", ("unsloth", "models"))):
        if name not in sys.modules:
            pkg = types.ModuleType(name)
            pkg.__path__ = [os.path.join(REPO, *path)]
            sys.modules[name] = pkg
    env = importlib.import_module("unsloth.parallel_env")
    auto = importlib.import_module("unsloth.parallel_auto")
    assert os.path.dirname(auto.__file__) == os.path.join(REPO, "unsloth"), auto.__file__
    return env, auto


def _architecture_skip_modules():
    """vision.py's _architecture_skip_modules, taken verbatim from the source (vision.py needs the GPU package)."""
    import ast

    path = os.path.join(REPO, "unsloth", "models", "vision.py")
    tree = ast.parse(open(path, encoding = "utf-8").read())
    fn = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_architecture_skip_modules"
    )
    ns = {"_mamba_ssm_importable": lambda: False}
    exec(compile(ast.Module(body = [fn], type_ignores = []), path, "exec"), ns)
    return ns["_architecture_skip_modules"]


def _real_quant_skip_modules(model_types):
    """parallel_auto._quant_skip_modules with the real lists: unsloth_zoo's and vision.py's (from source)."""
    from unsloth_zoo.peft_utils import SKIP_QUANTIZATION_MODULES
    return list(SKIP_QUANTIZATION_MODULES) + list(_architecture_skip_modules()(model_types))


def _stub_loader_facts(
    auto,
    setattr,
    model_types = ("llama",),
    supports_bf16 = True,
    force_float32 = (),
):
    """What the loaders know about the GPU and the architecture, without importing them (GPU package)."""
    setattr(auto, "_loader_precision", lambda: (supports_bf16, list(force_float32)), raising = False)
    setattr(auto, "_model_types", lambda name, call, revision: list(model_types), raising = False)
    setattr(auto, "_hidden_size", lambda name, call, revision: None, raising = False)
    setattr(auto, "_quant_skip_modules", _real_quant_skip_modules, raising = False)


def _stub_device_type(allow_bitsandbytes = True):
    module = types.ModuleType("unsloth.device_type")
    module.ALLOW_BITSANDBYTES = allow_bitsandbytes
    return module


def _stub_name_mapping(mapping = None, calls = None):
    """unsloth.models.loader_utils stand-in: the real one needs the GPU package. Identity unless mapped."""
    module = types.ModuleType("unsloth.models.loader_utils")

    def get_model_name(
        model_name,
        load_in_4bit = True,
        **kwargs,
    ):
        if calls is not None:
            calls.append((model_name, load_in_4bit))
        return (mapping or {}).get((model_name, load_in_4bit), model_name)

    module.get_model_name = get_model_name
    return module


@pytest.fixture
def clean(monkeypatch):
    for key in _ENV:
        monkeypatch.delenv(key, raising = False)
    env, auto = _modules()
    auto._STATE.update(choice = None, reason = None)
    monkeypatch.setitem(sys.modules, "unsloth.models.loader_utils", _stub_name_mapping())
    monkeypatch.setitem(sys.modules, "unsloth.device_type", _stub_device_type())
    # Hub names in these tests are fake: no PeftConfig probe over the network (the PEFT test restores it).
    monkeypatch.setattr(auto, "_peft_base_real", getattr(auto, "_peft_base", None), raising = False)
    monkeypatch.setattr(auto, "_peft_base", lambda name, call, revision: None, raising = False)
    _stub_loader_facts(auto, monkeypatch.setattr)
    yield monkeypatch, env, auto
    auto._STATE.update(choice = None, reason = None)


def _snapshot():
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(TINY, allow_patterns = ["*.json", "*.safetensors"])
    except Exception as exc:  # offline host without the cached tiny repo
        pytest.skip(f"{TINY} unavailable: {exc}")


def test_auto_parses_and_explicit_layouts_are_unchanged(clean):
    mp, env, auto = clean
    assert env.parse_parallel_spec("auto") == {"auto": True}
    assert env.parse_parallel_spec("  AUTO ") == {"auto": True}
    assert env.parse_parallel_spec("fsdp=1") == {"fsdp": True}
    assert env.parse_parallel_spec("fsdp=0") == {"fsdp": False}
    assert env.parse_parallel_spec("dp_shard=2") == {"dp": 2, "fsdp": True}
    assert env.translate(env.parse_parallel_spec("fsdp=1"), {"WORLD_SIZE": "2"}) == {
        "UNSLOTH_FSDP2": "1"
    }
    with pytest.raises(ValueError, match = "is not key=value"):
        env.parse_parallel_spec("auto,tp=2")
    # auto at import writes nothing: the decision waits for the model.
    environ = {"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2"}
    assert env.apply_parallel_env(environ) == {"auto": True}
    assert environ == {"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2"}
    assert auto.auto_requested(environ) and not auto.auto_requested({"UNSLOTH_PARALLEL": "fsdp=1"})


@pytest.mark.parametrize(
    "extra, match",
    [
        ({"UNSLOTH_FSDP2": "1"}, "UNSLOTH_FSDP2=1 already does"),
        ({"UNSLOTH_FSDP2": "0"}, "UNSLOTH_FSDP2=0 already does"),
        ({"UNSLOTH_TP_SIZE": "2"}, "UNSLOTH_TP_SIZE=2 asks for another axis"),
        ({"UNSLOTH_CP_SIZE": "2"}, "UNSLOTH_CP_SIZE=2 asks for another axis"),
        ({"ACCELERATE_USE_FSDP": "true"}, "accelerate launch"),
        ({"UNSLOTH_PARALLEL_AUTO_GB": "lots"}, "number of GiB"),
    ],
)
def test_auto_refuses_a_layout_already_fixed(clean, extra, match):
    _, env, _ = clean
    with pytest.raises(ValueError, match = match):
        env.apply_parallel_env({"UNSLOTH_PARALLEL": "auto", "WORLD_SIZE": "2", **extra})


def test_child_process_inherits_the_decision(clean):
    _, env, auto = clean
    environ = {
        "UNSLOTH_PARALLEL": "auto",
        "UNSLOTH_FSDP2": "0",
        "UNSLOTH_PARALLEL_AUTO_RESOLVED": "ddp",
        "WORLD_SIZE": "2",
    }
    env.apply_parallel_env(environ)
    assert auto.resolved_choice() == "ddp"
    auto._STATE.update(choice = None)
    with pytest.raises(ValueError):  # a marker that disagrees with UNSLOTH_FSDP2 is not inherited
        env.apply_parallel_env({**environ, "UNSLOTH_PARALLEL_AUTO_RESOLVED": "fsdp2"})


@pytest.mark.parametrize(
    "axis", ["UNSLOTH_TP_SIZE", "UNSLOTH_PP_SIZE", "UNSLOTH_CP_SIZE", "UNSLOTH_EP_SIZE"]
)
def test_inherited_decision_still_refuses_another_axis(clean, axis, monkeypatch):
    _, _, auto = clean
    monkeypatch.delenv(axis, raising = False)
    environ = {
        "UNSLOTH_PARALLEL": "auto",
        "UNSLOTH_FSDP2": "0",
        "UNSLOTH_PARALLEL_AUTO_RESOLVED": "ddp",
        "WORLD_SIZE": "2",
        axis: "2",
    }
    with pytest.raises(ValueError, match = axis):
        auto.check_auto_env(environ)


def test_inherited_fsdp2_keeps_the_accelerate_env_its_parent_exported(clean):
    _, _, auto = clean
    environ = {
        "UNSLOTH_PARALLEL": "auto",
        "UNSLOTH_FSDP2": "1",
        "UNSLOTH_PARALLEL_AUTO_RESOLVED": "fsdp2",
        "ACCELERATE_USE_FSDP": "true",
        "FSDP_VERSION": "2",
        "WORLD_SIZE": "2",
    }
    auto.check_auto_env(environ)
    assert auto.resolved_choice() == "fsdp2"


def test_decision_and_state_arithmetic(clean):
    _, _, auto = clean
    gib = 1024**3
    # 5 GiB weights + 1 GiB state + 20% of 14 GiB = 8.8 GiB: fits 14 GiB, not 8 GiB.
    assert auto.decide(5 * gib, 1 * gib, 14 * gib)[0] is True
    assert auto.decide(5 * gib, 1 * gib, 8 * gib, total = 14 * gib)[0] is False
    assert auto.decide(0, 0, 0)[0] is False  # nothing sized: never DDP
    assert auto.training_state_bytes(10, 4, "adamw_torch") == 10 * (8 + 8)
    assert auto.training_state_bytes(10, 2, "adamw_8bit") == 10 * (4 + 2)


def test_checkpoint_headers_size_the_load(clean):
    _, _, auto = clean
    tensors = auto._checkpoint_tensors(_snapshot())
    assert tensors and all(len(t) == 3 for t in tensors)
    raw16 = sum(
        __import__("math").prod(shape) * min(auto._DTYPE_BYTES[dt], 2) for _, dt, shape in tensors
    )
    w16, lora16 = auto.checkpoint_estimate(tensors)
    # Quantizing on load takes the loader's skip modules (none for this generic check).
    takes_skip = (
        "skip_modules" in __import__("inspect").signature(auto.checkpoint_estimate).parameters
    )
    w4, lora4 = auto.checkpoint_estimate(
        tensors, load_in_4bit = True, **({"skip_modules": []} if takes_skip else {})
    )
    wfull, train_full = auto.checkpoint_estimate(tensors, full_finetuning = True, load_in_4bit = True)
    assert w16 == raw16
    assert w4 < w16 and lora4 == lora16 > 0
    assert wfull == w16  # full finetuning does not quantize
    assert train_full == sum(__import__("math").prod(s) for _, _, s in tensors)


def test_load_sizing_reads_the_repo_the_loader_maps_to(clean):
    """unsloth/mistral-7b-bnb-4bit with load_in_4bit = False loads unsloth/mistral-7b in 16-bit: size that one."""
    mp, _, auto = clean
    calls, fetched = [], []
    mapping = {("unsloth/mistral-7b-bnb-4bit", False): "unsloth/mistral-7b"}
    mp.setitem(sys.modules, "unsloth.models.loader_utils", _stub_name_mapping(mapping, calls))
    headers = {
        # 4096 x 4096 linear: 8 MiB packed 4-bit codes vs 32 MiB in 16-bit
        "unsloth/mistral-7b-bnb-4bit": [("model.layers.0.mlp.up_proj.weight", "U8", (8388608, 1))],
        "unsloth/mistral-7b": [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))],
    }

    def tensors(name, **kwargs):
        fetched.append(name)
        return headers[name]

    _auto_env(mp, auto, budget_mib = 25)
    mp.setattr(auto, "_checkpoint_tensors", tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        load_in_16bit = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(
        from_pretrained, ("unsloth/mistral-7b-bnb-4bit",), {"load_in_4bit": False}
    )
    assert calls == [("unsloth/mistral-7b-bnb-4bit", False)] and fetched == ["unsloth/mistral-7b"]
    assert got == "fsdp2", auto._STATE["reason"]  # 32 MiB of 16-bit weights do not fit 25 MiB


def test_peft_adapter_dir_sizes_its_base(clean, tmp_path):
    """A local adapter dir loads peft_config.base_model_name_or_path too: size the base plus the adapter."""
    pytest.importorskip("peft")
    from peft import LoraConfig
    from safetensors.torch import save_file

    mp, _, auto = clean
    mp.setattr(auto, "_peft_base", auto._peft_base_real)
    adapter = tmp_path / "adapter"
    LoraConfig(r = 4, target_modules = ["up_proj"], base_model_name_or_path = "org/base").save_pretrained(
        adapter
    )
    save_file(
        {"base_model.model.up_proj.lora_A.weight": torch.zeros(4, 4096)},
        adapter / "adapter_model.safetensors",
    )
    real, fetched = auto._checkpoint_tensors, []

    def tensors(name, **kwargs):
        fetched.append(name)
        if name == "org/base":
            return [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))]  # 32 MiB in 16-bit
        return real(name, **kwargs)

    _auto_env(mp, auto, budget_mib = 25)
    mp.setattr(auto, "_checkpoint_tensors", tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        load_in_16bit = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, (str(adapter),), {"load_in_4bit": False})
    assert fetched == [str(adapter), "org/base"], fetched
    assert got == "fsdp2", auto._STATE["reason"]


def test_no_bitsandbytes_turns_4bit_off_before_mapping(clean):
    """ALLOW_BITSANDBYTES false: the loaders clear load_in_4bit first, so a -bnb-4bit name loads its float repo."""
    mp, _, auto = clean
    calls, fetched = [], []
    mapping = {("unsloth/mistral-7b-bnb-4bit", False): "unsloth/mistral-7b"}
    mp.setitem(sys.modules, "unsloth.models.loader_utils", _stub_name_mapping(mapping, calls))
    mp.setitem(sys.modules, "unsloth.device_type", _stub_device_type(allow_bitsandbytes = False))
    headers = {
        "unsloth/mistral-7b-bnb-4bit": [("model.layers.0.mlp.up_proj.weight", "U8", (8388608, 1))],
        "unsloth/mistral-7b": [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))],
    }
    _auto_env(mp, auto, budget_mib = 25)
    mp.setattr(auto, "_checkpoint_tensors", lambda name, **k: fetched.append(name) or headers[name])

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        load_in_16bit = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, ("unsloth/mistral-7b-bnb-4bit",), {})
    assert calls == [("unsloth/mistral-7b-bnb-4bit", False)] and fetched == ["unsloth/mistral-7b"]
    assert got == "fsdp2", auto._STATE["reason"]


def test_gather_uses_gloo_on_cuda_without_nccl(clean, monkeypatch):
    """A Windows CUDA build has no NCCL: the decision's process group falls back to gloo."""
    import torch.distributed as dist

    for name in ("unsloth.fsdp2_launch", "unsloth.models.fsdp2_native"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    _load_native()
    created = []
    monkeypatch.setattr(dist, "is_initialized", lambda: bool(created))
    monkeypatch.setattr(dist, "is_nccl_available", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", lambda *a, **k: created.append((a, k)))
    monkeypatch.setattr(dist, "get_world_size", lambda *a: 1)
    monkeypatch.setattr(
        dist, "all_gather_object", lambda out, obj, *a, **k: out.__setitem__(0, obj)
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    monkeypatch.setenv("LOCAL_RANK", "0")
    _, _, auto = clean
    assert auto._gather({"fits": True}) == [{"fits": True}]
    ((args, kwargs),) = created
    assert (args[0] if args else kwargs.get("backend")) == "gloo", created


def test_full_finetuning_counts_float32_master_weights(clean):
    """Full finetuning on a float16 GPU keeps float32 master weights, grads and DDP buckets: 20 bytes per parameter."""
    mp, _, auto = clean
    _stub_loader_facts(auto, mp.setattr, supports_bf16 = False)  # a T4
    tensors = [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))]  # 16 Mi parameters
    # 16-bit sizing: 32 MiB + 192 MiB state fits 350 MiB (280 after the margin); float32 master: 64 + 256 does not.
    _auto_env(mp, auto, budget_mib = 350)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        full_finetuning = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, ("org/model",), {"full_finetuning": True})
    assert got == "fsdp2", auto._STATE["reason"]
    assert "weights 64.0 MiB + training state 0.25 GiB" in auto._STATE["reason"], auto._STATE[
        "reason"
    ]


def test_full_finetuning_a_prequantized_checkpoint_declines(clean):
    """Full finetuning dequantizes a packed bitsandbytes checkpoint before training: not sized from its packed bytes."""
    mp, _, auto = clean
    _stub_loader_facts(auto, mp.setattr)
    tensors = [("model.layers.0.mlp.up_proj.weight", "U8", (2048, 1))]
    _auto_env(mp, auto, budget_mib = 4096)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        full_finetuning = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, ("org/model-bnb-4bit",), {"full_finetuning": True})
    assert got == "fsdp2", auto._STATE["reason"]
    assert "dequantizes" in auto._STATE["reason"], auto._STATE["reason"]


def test_fresh_lora_counts_its_adapter_parameters(clean):
    """A fresh r = 16 adapter's float32 parameters are resident memory, not only their gradients and moments."""
    mp, _, auto = clean
    _stub_loader_facts(auto, mp.setattr)
    tensors = [
        ("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))
    ]  # 32 MiB, LoRA 131072 params
    _auto_env(mp, auto, budget_mib = 4096)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = False,
        **kwargs,
    ):
        return model_name

    auto.resolve_for_load(from_pretrained, ("org/model",), {"load_in_4bit": False})
    assert "weights 32.5 MiB" in auto._STATE["reason"], auto._STATE["reason"]


def test_packed_lora_uses_the_hidden_size_side():
    """A packed rectangular layer (18944 x 3584) sizes its r = 16 adapter from hidden_size, not a square guess."""
    from unsloth import parallel_auto as auto

    tensors = [("model.layers.0.mlp.up_proj.weight", "U8", (18944 * 3584 // 2, 1))]
    _, square = auto.checkpoint_estimate(tensors)
    _, exact = auto.checkpoint_estimate(tensors, hidden_size = 3584)
    assert exact == 16 * (18944 + 3584) == 360448
    assert square < exact


def test_trainer_sizing_keeps_each_trainable_dtype(clean):
    """Mixed trainable dtypes: a float32 parameter keeps 4-byte grads and buckets even when a float16 one follows."""
    _, _, auto = clean
    model = torch.nn.Module()
    model.a = torch.nn.Parameter(torch.zeros(1_000_000, dtype = torch.float32))
    model.b = torch.nn.Parameter(torch.zeros(1, dtype = torch.float16))
    captured = {}

    def fake_resolve(estimate, install_trainer_init):
        captured["value"] = estimate()
        return "ddp"

    import unittest.mock as um

    with (
        um.patch.object(auto, "_resolve", fake_resolve),
        um.patch.object(auto, "auto_requested", lambda: True),
    ):
        auto.resolve_for_trainer(model)
    assert captured["value"][1] == 1_000_000 * (2 * 4 + 8) + 1 * (2 * 2 + 8)


def test_quantize_on_load_keeps_the_loaders_dense_modules(clean):
    """WavLM's attention projections stay dense under load_in_4bit (vision._architecture_skip_modules)."""
    mp, _, auto = clean
    _stub_loader_facts(auto, mp.setattr, model_types = ("wavlm",))
    tensors = [
        ("encoder.layers.0.attention.q_proj.weight", "BF16", (4096, 4096))
    ]  # 32 MiB dense, ~8.5 MiB 4-bit
    _auto_env(mp, auto, budget_mib = 25)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, ("org/wavlm",), {})
    assert got == "fsdp2", auto._STATE["reason"]
    assert "weights 32.5 MiB" in auto._STATE["reason"], auto._STATE["reason"]


def test_unmodelled_loads_decline_with_a_reason(clean):
    """Conservative by construction: what the estimate does not model picks FSDP2 and says why."""
    mp, _, auto = clean
    _auto_env(mp, auto, budget_mib = 1024)
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: [("a.weight", "BF16", (64, 64))])

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        **kwargs,
    ):
        return model_name

    cases = [
        ({"fast_inference": True}, {}, "fast_inference"),
        ({"dtype": "int8"}, {}, "dtype 'int8' is not sized"),
        ({}, {"UNSLOTH_FORCE_FLOAT32": "1"}, "UNSLOTH_FORCE_FLOAT32"),
    ]
    for kwargs, env, why in cases:
        auto._STATE.update(choice = None, reason = None)
        for key, value in env.items():
            mp.setenv(key, value)
        assert auto.resolve_for_load(from_pretrained, ("org/model",), kwargs) == "fsdp2"
        assert why in auto._STATE["reason"], auto._STATE["reason"]
        for key in env:
            mp.delenv(key)
    auto._STATE.update(choice = None, reason = None)
    _stub_loader_facts(
        auto, mp.setattr, model_types = ("gemma3",), supports_bf16 = False, force_float32 = ("gemma3",)
    )
    assert (
        auto.resolve_for_load(from_pretrained, ("org/model",), {"load_in_4bit": False}) == "fsdp2"
    )
    assert "forced float32" in auto._STATE["reason"], auto._STATE["reason"]


def test_float_weights_count_at_the_loaded_dtype(clean):
    _, _, auto = clean
    n = 4096 * 4096
    bf16 = [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))]
    fp32 = [("model.layers.0.mlp.up_proj.weight", "F32", (4096, 4096))]
    assert (
        auto.checkpoint_estimate(bf16, dtype_bytes = 4)[0] == 4 * n
    )  # bf16 checkpoint, float32 load
    assert auto.checkpoint_estimate(bf16, dtype_bytes = 2)[0] == 2 * n
    assert auto.checkpoint_estimate(fp32, dtype_bytes = 2)[0] == 2 * n  # fp32 checkpoint, 16-bit load
    # bitsandbytes quant state beside packed codes keeps its stored float32
    packed = [("a.weight", "U8", (1024, 1)), ("a.weight.absmax", "F32", (256,))]
    assert auto.checkpoint_estimate(packed, dtype_bytes = 2)[0] == 1024 + 4 * 256


def _loader_unc_guard_module():
    """unsloth.models.loader with only the real network-share guard: importing the whole loader would need the
    _utils module _auto_env stubs out, so the guard is taken from the loader's source instead."""
    import ast

    source = open(os.path.join(REPO, "unsloth", "models", "loader.py"), encoding = "utf-8").read()
    fn = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_refuse_network_share_adapter_base"
    )
    module = types.ModuleType("unsloth.models.loader")
    module.os = os
    exec(compile(ast.Module(body = [fn], type_ignores = []), "loader.py", "exec"), module.__dict__)
    return module


def _auto_env(mp, auto, budget_mib):
    mp.setattr(auto, "_world", lambda environ: 2)
    mp.setattr(auto, "_gather", lambda entry: [entry, entry])
    utils = types.ModuleType("unsloth.models._utils")
    utils.patch_gradient_accumulation_fix = lambda cls: None
    mp.setitem(sys.modules, "unsloth.models._utils", utils)
    mp.setitem(sys.modules, "unsloth.models.loader", _loader_unc_guard_module())
    for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL_AUTO_RESOLVED"):
        mp.setenv(key, "")
        mp.delenv(key)
    mp.setenv("UNSLOTH_PARALLEL", "auto")
    mp.setenv("UNSLOTH_PARALLEL_AUTO_GB", str(budget_mib / 1024))
    mp.setenv("WORLD_SIZE", "2")


def test_budget_measures_the_local_rank_gpu(clean, monkeypatch):
    """Rank 1 sizes cuda:1, not the current device (cuda:0), and leaves the current device alone."""
    _, _, auto = clean
    gib, calls = 1024**3, []
    free = {0: 2 * gib, 1: 9 * gib}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d = None: (free[d], 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: 0)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d = None: 0)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: calls.append(d))
    monkeypatch.setenv("LOCAL_RANK", "1")
    assert auto._budget() == (9 * gib, 16 * gib)
    assert calls == []
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)  # one visible GPU per rank
    assert auto._budget() == (2 * gib, 16 * gib)


def test_budget_excludes_live_unrelated_allocations(clean, monkeypatch):
    """Reserved memory holding other live tensors is not available; free cache + the model's own bytes are."""
    _, _, auto = clean
    gib = 1024**3
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d = None: (4 * gib, 16 * gib))
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda d = None: 10 * gib)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda d = None: 9 * gib)
    monkeypatch.delenv("LOCAL_RANK", raising = False)
    assert auto._budget() == (5 * gib, 16 * gib)
    assert auto._budget(9 * gib, 9 * gib) == (14 * gib, 16 * gib)


def test_import_installs_the_trainer_wrapper_while_auto_is_unresolved(clean, monkeypatch):
    """`import unsloth` then a plain transformers model: the wrapper that resolves auto must be installed."""
    from transformers import Trainer

    from test_fsdp2_native_trainer_init_any_loader_gloo import _load_launch

    _, _, auto = clean
    for name in ("unsloth", "unsloth.models", "unsloth.models._utils", "unsloth.fsdp2_launch"):
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
    monkeypatch.setattr(Trainer, "__init__", Trainer.__init__)
    monkeypatch.setattr(Trainer, "_test_wrapped", False, raising = False)
    monkeypatch.setenv("UNSLOTH_PARALLEL", "auto")
    monkeypatch.setenv("WORLD_SIZE", "2")
    calls = []
    launch = _load_launch(calls)
    launch.install_native_fsdp2_trainer_init()
    assert calls == [Trainer]
    assert Trainer.__init__.__name__ == "_unsloth_trainer_init"
    auto._STATE.update(choice = "ddp")
    calls.clear()
    launch.install_native_fsdp2_trainer_init()
    auto._STATE.update(choice = None)
    monkeypatch.setenv("WORLD_SIZE", "1")
    launch.install_native_fsdp2_trainer_init()
    assert calls == []


@pytest.mark.parametrize(
    "name, extra, expected",
    [
        ("org/model", {}, "ddp"),  # 4-bit on load fits the budget
        ("org/model-bf16", {}, "fsdp2"),  # the loader loads a -bf16 repo in 16bit: does not fit
        (
            "org/model",
            {"quantization_config": object()},
            "fsdp2",
        ),  # precision not known here: not sized
    ],
)
def test_load_sizing_follows_the_loaders_precision(clean, name, extra, expected):
    mp, _, auto = clean
    # One 4096 x 4096 linear: ~8.5 MiB in 4-bit, 32 MiB in 16-bit; 25 MiB per rank (80% after the margin).
    tensors = [("model.layers.0.mlp.up_proj.weight", "BF16", (4096, 4096))]
    mp.setattr(auto, "_checkpoint_tensors", lambda *a, **k: tensors)
    mp.setattr(auto, "_world", lambda environ: 2)
    mp.setattr(auto, "_gather", lambda entry: [entry, entry])
    utils = types.ModuleType("unsloth.models._utils")
    utils.patch_gradient_accumulation_fix = lambda cls: None
    mp.setitem(sys.modules, "unsloth.models._utils", utils)
    for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL_AUTO_RESOLVED"):
        mp.setenv(key, "")
        mp.delenv(key)
    mp.setenv("UNSLOTH_PARALLEL", "auto")
    mp.setenv("UNSLOTH_PARALLEL_AUTO_GB", str(25 / 1024))
    mp.setenv("WORLD_SIZE", "2")

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        load_in_16bit = False,
        **kwargs,
    ):
        return model_name

    assert auto.resolve_for_load(from_pretrained, (name,), dict(extra)) == expected, auto._STATE[
        "reason"
    ]


def _init(rank, world, port):
    import datetime

    import torch.distributed as dist

    os.environ.update(
        MASTER_ADDR = "127.0.0.1",
        MASTER_PORT = str(port),
        RANK = str(rank),
        WORLD_SIZE = str(world),
        LOCAL_RANK = str(rank),
        LOCAL_WORLD_SIZE = str(world),
    )
    for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL_AUTO_RESOLVED"):
        os.environ.pop(key, None)
    dist.init_process_group(
        "gloo", rank = rank, world_size = world, timeout = datetime.timedelta(seconds = 120)
    )


def _decision_worker(rank, world, port, out, snapshot):
    """resolve_for_load on a from_pretrained-shaped loader, one case after another in one process group."""
    _init(rank, world, port)
    import torch.distributed as dist

    try:
        _load_native()
        # FSDP2 installs the Trainer wrapper through models._utils, which needs the GPU package.
        utils = types.ModuleType("unsloth.models._utils")
        utils.patch_gradient_accumulation_fix = lambda cls: None
        sys.modules["unsloth.models._utils"] = utils
        sys.modules["unsloth.models.loader_utils"] = _stub_name_mapping()
        sys.modules["unsloth.device_type"] = _stub_device_type()
        _, auto = _modules()
        _stub_loader_facts(auto, lambda obj, name, value, raising = False: setattr(obj, name, value))

        def from_pretrained(
            model_name = "unused",
            load_in_4bit = True,
            full_finetuning = False,
            **kwargs,
        ):
            return model_name

        res = {}
        for case, budgets in (
            ("fits", ("64", "64")),
            ("tight", ("1e-6", "1e-6")),
            ("split", ("64", "1e-6")),
        ):
            auto._STATE.update(choice = None, reason = None)
            os.environ.pop("UNSLOTH_FSDP2", None)
            os.environ.update(UNSLOTH_PARALLEL = "auto", UNSLOTH_PARALLEL_AUTO_GB = budgets[rank])
            if case == "fits":
                # Through the decorator every Unsloth from_pretrained carries (DDP: a plain load follows).
                load = importlib.import_module("unsloth.models.fsdp2_load")
                res["loaded"] = load.sharded_load_env(from_pretrained)(snapshot, load_in_4bit = False)
                choice = auto.resolved_choice()
            else:
                choice = auto.resolve_for_load(
                    from_pretrained, (snapshot,), {"load_in_4bit": False}
                )
            again = auto.resolve_for_load(from_pretrained, (snapshot,), {})  # once per process
            res[case] = dict(
                choice = choice,
                again = again,
                fsdp2_env = os.environ.get("UNSLOTH_FSDP2"),
                marker = os.environ.get("UNSLOTH_PARALLEL_AUTO_RESOLVED"),
                reason = auto._STATE["reason"],
            )
        out.put((rank, res))
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


def _spawn(fn, *args):
    import torch.multiprocessing as mp

    from test_fsdp2_native_trainer_ckpt_gloo import _free_port

    ctx = mp.get_context("spawn")
    out = ctx.Queue()
    mp.start_processes(
        fn, args = (2, _free_port(), out, *args), nprocs = 2, join = True, start_method = "spawn"
    )
    results = {}
    while not out.empty():
        rank, r = out.get(timeout = 5)
        results[rank] = r
    assert sorted(results) == [0, 1], results
    for r in results.values():
        assert "crash" not in r, r["crash"]
    return results


def test_ranks_agree_fits_ddp_tight_fsdp2_split_fsdp2():
    snapshot = _snapshot()
    results = _spawn(_decision_worker, snapshot)
    for rank, r in results.items():
        print("AUTO_DECISION", rank, r)
        assert r["loaded"] == snapshot, r  # the decorated load ran after the decision
        assert r["fits"]["choice"] == r["fits"]["again"] == "ddp", r
        assert r["fits"]["fsdp2_env"] == "0" and r["fits"]["marker"] == "ddp", r
        assert r["tight"]["choice"] == "fsdp2" and r["tight"]["fsdp2_env"] == "1", r
        assert r["split"]["choice"] == r["split"]["again"] == "fsdp2", r
        assert r["split"]["fsdp2_env"] == "1" and "disagree" in r["split"]["reason"], r
        assert "rank 1 needs" in r["split"]["reason"], r


def _rows():
    g = torch.Generator().manual_seed(0)
    rows = []
    for _ in range(8):
        ids = torch.randint(3, 100, (16,), generator = g)
        rows.append(
            {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}
        )
    return rows


def _lora_model():
    import transformers
    from peft import LoraConfig, get_peft_model

    config = transformers.AutoConfig.from_pretrained(TINY)
    config.pad_token_id = None
    torch.manual_seed(0)
    model = transformers.AutoModelForCausalLM.from_config(config, torch_dtype = torch.float32)
    model = get_peft_model(
        model, LoraConfig(r = 4, lora_alpha = 8, target_modules = ["q_proj", "v_proj"])
    )
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.02, generator = g)
    return model


def _e2e_worker(rank, world, port, out, tmp, budget_gb, expected):
    _init(rank, world, port)
    import torch.distributed as dist
    try:
        native = _load_native()
        from test_fsdp2_native_trainer_ckpt_gloo import _stub_unsloth_save
        from test_fsdp2_native_trainer_init_any_loader_gloo import _load_launch

        _stub_unsloth_save()
        _, auto = _modules()
        from torch.distributed.tensor import DTensor
        from torch.nn.parallel import DistributedDataParallel
        from transformers import Trainer, TrainingArguments

        res = {}
        for mode in ("auto", "explicit"):
            auto._STATE.update(choice = None, reason = None)
            for key in ("UNSLOTH_FSDP2", "UNSLOTH_PARALLEL", "UNSLOTH_PARALLEL_AUTO_GB"):
                os.environ.pop(key, None)
            if mode == "auto":
                os.environ.update(UNSLOTH_PARALLEL = "auto", UNSLOTH_PARALLEL_AUTO_GB = budget_gb)
            else:
                os.environ["UNSLOTH_FSDP2"] = "1" if expected == "fsdp2" else "0"
            # What `import unsloth` runs: the Trainer wrapper (the real _unsloth_trainer_init source) installs
            # under native FSDP2, and under auto while it is unresolved (a plain transformers model, no Unsloth load).
            installed = []
            _load_launch(installed).install_native_fsdp2_trainer_init()
            args = TrainingArguments(
                output_dir = os.path.join(tmp, mode),
                report_to = [],
                use_cpu = True,
                per_device_train_batch_size = 2,
                max_steps = STEPS,
                learning_rate = 1e-2,
                lr_scheduler_type = "constant",
                optim = "adamw_torch",
                weight_decay = 0.0,
                max_grad_norm = 1.0,
                save_strategy = "no",
                logging_steps = 1,
                disable_tqdm = True,
                dataloader_drop_last = True,
                seed = 0,
            )
            trainer = Trainer(model = _lora_model(), args = args, train_dataset = _rows())
            trainer.train()
            params = {
                n: (p.full_tensor() if isinstance(p, DTensor) else p).detach().clone()
                for n, p in trainer.model.named_parameters()
                if "lora_" in n
            }
            res[mode] = dict(
                fsdp2_env = os.environ.get("UNSLOTH_FSDP2"),
                sharded = bool(native.is_native_fsdp2(trainer.model)),
                ddp = isinstance(trainer.model_wrapped, DistributedDataParallel),
                losses = [h["loss"] for h in trainer.state.log_history if "loss" in h],
                params = params,
                choice = auto._STATE["choice"],
                installed = bool(installed),
            )
        a, b = res["auto"], res["explicit"]
        out.put(
            (
                rank,
                dict(
                    auto = {k: v for k, v in a.items() if k != "params"},
                    explicit = {k: v for k, v in b.items() if k != "params"},
                    n_params = len(a["params"]),
                    max_diff = max(
                        (a["params"][n] - b["params"][n]).abs().max().item() for n in a["params"]
                    ),
                ),
            )
        )
    except Exception as exc:
        import traceback
        out.put((rank, {"crash": f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("budget_gb, expected", [("64", "ddp"), ("1e-6", "fsdp2")])
def test_auto_trains_like_the_explicit_layout_it_picked(tmp_path, budget_gb, expected):
    pytest.importorskip("peft")
    _snapshot()
    results = _spawn(_e2e_worker, str(tmp_path), budget_gb, expected)
    for rank, r in results.items():
        print("AUTO_E2E", rank, r)
        a, b = r["auto"], r["explicit"]
        assert a["installed"] and a["choice"] == expected, r
        assert a["fsdp2_env"] == b["fsdp2_env"] == ("1" if expected == "fsdp2" else "0"), r
        assert a["sharded"] == b["sharded"] == (expected == "fsdp2"), r
        assert a["ddp"] == b["ddp"] == (expected == "ddp"), r
        assert len(a["losses"]) == STEPS and a["losses"] == b["losses"], r
        assert r["n_params"] > 0 and r["max_diff"] == 0.0, r


@pytest.mark.parametrize("base", [r"\\evil.example\share\model", "//evil.example/share/model"])
def test_hub_adapter_network_share_base_is_refused_before_any_probe(clean, base):
    """A Hub adapter's UNC base_model_name_or_path must not reach get_model_name / a filesystem probe while
    sizing (Windows: SMB + NTLM to a host the repo chose); the loaders refuse it the same way."""
    mp, _, auto = clean
    windows_share = lambda p: p.replace("\\", "/").startswith(
        "//"
    )  # the Windows answer, on any host
    mp.setitem(
        sys.modules,
        "unsloth_zoo.vision_utils",
        types.SimpleNamespace(_is_network_share = windows_share),
    )
    calls, fetched = [], []
    mp.setitem(sys.modules, "unsloth.models.loader_utils", _stub_name_mapping(calls = calls))
    mp.setattr(auto, "_peft_base", lambda name, call, revision: base)

    def tensors(name, **kwargs):
        fetched.append(name)
        return [("model.layers.0.mlp.up_proj.weight", "BF16", (64, 64))]

    _auto_env(mp, auto, budget_mib = 25)
    mp.setattr(auto, "_checkpoint_tensors", tensors)

    def from_pretrained(
        model_name = "x",
        load_in_4bit = True,
        load_in_16bit = False,
        **kwargs,
    ):
        return model_name

    got = auto.resolve_for_load(from_pretrained, ("someone/adapter",), {"load_in_4bit": False})
    assert base not in fetched and all(name != base for name, _ in calls), (fetched, calls)
    assert got == "fsdp2" and "network share" in auto._STATE["reason"], auto._STATE["reason"]
