# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bf16 and the torch.compile kill switches answer for the current device, as torch does: an idle
older card beside it (e.g. RTX 4090 + GTX 1080) must not take bf16 or compile from a model that runs
on the 4090. A load that places layers on the older card is checked after loading. The capability
cache must also follow torch.cuda.set_device."""

import functools
import inspect
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

import unsloth.device_type as dt

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"
DEVICE_TYPE_PY = REPO_ROOT / "unsloth" / "device_type.py"

if not hasattr(dt, "torch"):
    pytest.skip(
        "device_type stub or MLX host; the real probes are not loaded", allow_module_level = True
    )

SM80, SM86, SM75, SM90, SM61 = (8, 0), (8, 6), (7, 5), (9, 0), (6, 1)

# (visible caps, current device): bf16 and the pre-Volta floor follow the current device only.
CASES = [
    pytest.param([SM80, SM80], 0, id = "sm80x2"),
    pytest.param([SM86, SM75], 0, id = "sm86+sm75"),
    pytest.param([SM86, SM75], 1, id = "sm86+sm75@1"),
    pytest.param([SM75, SM86], 0, id = "sm75+sm86"),
    pytest.param([SM90, SM61], 0, id = "sm90+sm61"),
    pytest.param([SM90, SM61], 1, id = "sm90+sm61@1"),
    pytest.param([SM61], 0, id = "sm61"),
]


def _fake_torch(caps, archs = None):
    state = {"current": 0}

    def get_device_capability(device = None):
        return caps[state["current"] if device is None else device]

    def get_device_properties(device = None):
        i = state["current"] if device is None else device
        major, minor = caps[i] if caps else (9, 0)
        return types.SimpleNamespace(
            major = major, minor = minor, gcnArchName = (archs or [""] * len(caps))[i]
        )

    # Real torch answers for the current device only.
    def is_bf16_supported(including_emulation = True):
        return including_emulation or caps[state["current"]][0] >= 8

    def set_device(i):
        state["current"] = i

    return types.SimpleNamespace(
        device = __import__("torch").device,
        version = types.SimpleNamespace(hip = None, cuda = "12.8"),
        cuda = types.SimpleNamespace(
            device_count = lambda: len(caps),
            current_device = lambda: state["current"],
            set_device = set_device,
            is_available = lambda: True,
            get_device_capability = get_device_capability,
            get_device_properties = get_device_properties,
            is_bf16_supported = is_bf16_supported,
        ),
    )


def _block(path, start, end):
    source = path.read_text(encoding = "utf-8")
    return start + source.split(start, 1)[1].split(end, 1)[0]


def _module_namespace(module, fake, monkeypatch, **overrides):
    # Helpers read their own module's torch, so point that at the fake too.
    monkeypatch.setattr(dt, "torch", fake)
    namespace = dict(vars(module))
    namespace.update(torch = fake, DEVICE_TYPE = "cuda", inspect = inspect, functools = functools)
    namespace.update(overrides)
    return namespace


def _run_gpu_init_chain(fake, monkeypatch):
    import unsloth._gpu_init as gpu_init

    body = _block(
        GPU_INIT,
        'if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():',
        '\nelif DEVICE_TYPE == "hip":',
    )
    namespace = _module_namespace(gpu_init, fake, monkeypatch)
    exec(compile(body, str(GPU_INIT), "exec"), namespace)
    return namespace


def _run_model_utils_chain(fake, monkeypatch):
    import unsloth.models._utils as model_utils

    body = _block(
        MODEL_UTILS,
        'if DEVICE_TYPE == "cuda" and not torch.cuda.is_available():',
        '\nelif DEVICE_TYPE == "hip":',
    )
    monkeypatch.setattr(model_utils, "torch", fake)
    namespace = _module_namespace(
        model_utils, fake, monkeypatch, _package_available = lambda name: False
    )
    exec(compile(body, str(MODEL_UTILS), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("caps,current", CASES)
def test_gpu_init_bf16_follows_the_current_device(monkeypatch, capsys, caps, current):
    fake = _fake_torch(caps)
    fake.cuda.set_device(current)
    namespace = _run_gpu_init_chain(fake, monkeypatch)
    bf16 = caps[current][0] >= 8
    assert namespace["SUPPORTS_BFLOAT16"] is bf16
    assert bool(fake.cuda.is_bf16_supported(False)) is bf16
    # Emulation is torch's own per-device answer and stays untouched.
    assert fake.cuda.is_bf16_supported(including_emulation = True) is True
    # An idle older card is no reason to print anything at import.
    assert "bfloat16" not in capsys.readouterr().out


@pytest.mark.parametrize("caps,current", CASES)
def test_model_utils_bf16_follows_the_current_device(monkeypatch, caps, current):
    fake = _fake_torch(caps)
    fake.cuda.set_device(current)
    namespace = _run_model_utils_chain(fake, monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is (caps[current][0] >= 8)


@pytest.mark.parametrize("chain", ["gpu_init", "model_utils"])
@pytest.mark.parametrize("world_size,bf16", [("2", False), ("1", True), ("", True)])
def test_a_distributed_launch_keeps_the_weakest_card_for_bf16(monkeypatch, chain, world_size, bf16):
    # DDP ranks import before set_device and must agree on one dtype: the weakest visible card decides.
    monkeypatch.setenv("WORLD_SIZE", world_size)
    fake = _fake_torch([SM86, SM75])
    run = _run_gpu_init_chain if chain == "gpu_init" else _run_model_utils_chain
    assert run(fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16


@pytest.mark.parametrize("current", [0, 1])
def test_a_distributed_launch_keeps_the_public_bf16_probe_on_the_weakest_card(monkeypatch, current):
    # save.py reads torch.cuda.is_bf16_supported() for merged 16-bit saves: every rank must agree.
    monkeypatch.setenv("WORLD_SIZE", "2")
    fake = _fake_torch([SM90, SM75])
    fake.cuda.set_device(current)
    namespace = _run_gpu_init_chain(fake, monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is False
    assert fake.cuda.is_bf16_supported(False) is False
    fake.cuda.set_device(1 - current)
    assert fake.cuda.is_bf16_supported(False) is False
    assert fake.cuda.is_bf16_supported(including_emulation = True) is True

    monkeypatch.setenv("WORLD_SIZE", "1")
    fake = _fake_torch([SM90, SM75])
    _run_gpu_init_chain(fake, monkeypatch)
    assert fake.cuda.is_bf16_supported(False) is True
    fake.cuda.set_device(1)
    assert fake.cuda.is_bf16_supported(False) is False


def test_capability_cache_follows_set_device(monkeypatch):
    fake = _fake_torch([SM86, SM75])
    _run_model_utils_chain(fake, monkeypatch)
    patched = fake.cuda.get_device_capability
    assert patched() == SM86
    fake.cuda.set_device(1)
    assert patched() == SM75
    assert patched(None) == SM75
    assert patched(0) == SM86
    assert patched(1) == SM75
    fake.cuda.set_device(0)
    assert patched() == SM86


def test_capability_cache_resolves_device_spellings(monkeypatch):
    import torch

    import unsloth.models._utils as model_utils

    if not hasattr(model_utils, "_cache_device_capability"):
        pytest.fail("no per-device capability cache")
    calls = []

    def probe(device = None):
        calls.append(device)
        return {0: SM86, 1: SM75}[device]

    fake = _fake_torch([SM86, SM75])
    monkeypatch.setattr(model_utils, "torch", fake)
    cached = model_utils._cache_device_capability(probe)
    assert cached(1) == SM75
    assert cached("cuda:1") == SM75
    assert cached(torch.device("cuda", 1)) == SM75
    assert cached() == SM86
    assert cached("cuda") == SM86
    assert calls == [0, 1]  # one real probe per card, read once while wrapping
    with pytest.raises(KeyError):
        cached("cpu")  # not a CUDA device: passed through uncached, as torch would reject it
    assert model_utils._cache_device_capability(cached) is cached  # re-import does not stack
    assert inspect.signature(cached) == inspect.signature(probe)


def test_capability_cache_single_device_keeps_the_plain_cache(monkeypatch):
    import unsloth.models._utils as model_utils

    calls = []

    def probe(device = None):
        calls.append(device)
        return SM86

    monkeypatch.setattr(model_utils, "torch", _fake_torch([SM86]))
    cached = model_utils._cache_device_capability(probe)
    assert hasattr(cached, "cache_info")  # functools.cache: no per-call Python work
    assert cached() == cached() == cached(0) == SM86
    assert calls == [None, 0]
    assert model_utils._cache_device_capability(cached) is cached


def test_capability_cache_identical_cards_keep_the_plain_cache(monkeypatch):
    import unsloth.models._utils as model_utils

    monkeypatch.setattr(model_utils, "torch", _fake_torch([SM90, SM90, SM90]))
    cached = model_utils._cache_device_capability(lambda device = None: SM90)
    assert hasattr(cached, "cache_info")  # every spelling of every card gets one answer
    assert cached() == cached(2) == SM90


def _run_pre_volta(fake, monkeypatch, capsys):
    applied = []

    def record(major, *args, **kwargs):
        applied.append(major)
        return major is not None and major < 7

    body = _block(
        DEVICE_TYPE_PY,
        'if DEVICE_TYPE == "cuda" and torch.cuda.is_available():',
        '\nif DEVICE_TYPE == "hip":\n    try:\n        import bitsandbytes',
    )
    namespace = _module_namespace(dt, fake, monkeypatch, apply_pre_volta_compile_workaround = record)
    exec(compile(body, str(DEVICE_TYPE_PY), "exec"), namespace)
    return applied, capsys.readouterr().out


@pytest.mark.parametrize("caps,current", CASES)
def test_pre_volta_switch_reads_the_current_device(monkeypatch, capsys, caps, current):
    fake = _fake_torch(caps)
    fake.cuda.set_device(current)
    applied, out = _run_pre_volta(fake, monkeypatch, capsys)
    assert applied == [caps[current][0]]
    if caps[current][0] >= 7:
        assert out == ""
    else:
        assert out == (
            "Unsloth: GPUs older than Volta (compute capability < 7.0) cannot run "
            "torch.compile - training without it.\n"
        )


@pytest.mark.parametrize("world_size,major", [("2", 6), ("1", 9)])
def test_a_distributed_launch_keeps_the_weakest_card_for_the_compile_switch(
    monkeypatch, capsys, world_size, major
):
    # A rank may be placed on the GTX 1080 after import: keep the weakest card, as for bf16.
    monkeypatch.setenv("WORLD_SIZE", world_size)
    fake = _fake_torch([SM90, SM61])
    applied, _ = _run_pre_volta(fake, monkeypatch, capsys)
    assert applied == [major]


def test_pre_volta_switch_skips_an_unreadable_card(monkeypatch, capsys):
    fake = _fake_torch([SM61, SM90])
    real = fake.cuda.get_device_capability

    def flaky(device = None):
        if device == 1:
            raise RuntimeError("device wedged")
        return real(device)

    fake.cuda.get_device_capability = flaky
    applied, _ = _run_pre_volta(fake, monkeypatch, capsys)
    assert applied == [6]


@pytest.mark.parametrize(
    "archs,applied",
    [
        (["gfx906:sramecc+:xnack-"], True),
        # An idle gfx906 beside the current card is checked after loading, not here.
        (["gfx1100", "gfx906:sramecc+:xnack-"], False),
        (["gfx906", "gfx1100"], True),
        (["gfx90a", "gfx942"], False),
    ],
)
def test_gfx906_switch_reads_the_current_device(monkeypatch, capsys, archs, applied):
    fake = _fake_torch([SM90] * len(archs), archs = archs)
    environ = {}
    dynamo_config = types.SimpleNamespace(disable = False)
    body = _block(
        DEVICE_TYPE_PY,
        "# gfx906 (MI50",
        '\n\nif DEVICE_TYPE == "cuda" and torch.cuda.is_available():',
    )
    namespace = _module_namespace(
        dt,
        fake,
        monkeypatch,
        DEVICE_TYPE = "hip",
        os = types.SimpleNamespace(environ = environ),
        apply_pre_volta_compile_workaround = lambda major: dt.apply_pre_volta_compile_workaround(
            major, environ = environ, dynamo_config = dynamo_config
        ),
    )
    exec(compile(body, str(DEVICE_TYPE_PY), "exec"), namespace)
    out = capsys.readouterr().out
    # Regions compiled before this check read the live config, not the env vars.
    assert dynamo_config.disable is applied
    if applied:
        assert environ == {
            "TORCHDYNAMO_DISABLE": "1",
            "TORCH_COMPILE_DISABLE": "1",
            "UNSLOTH_COMPILE_DISABLE": "1",
        }
        assert out == (
            "Unsloth: gfx906 (MI50 / Radeon VII) detected - torch.compile disabled "
            "(community-maintained legacy GCN path).\n"
        )
    else:
        assert environ == {} and out == ""


def _grpo_all_sm90_expression():
    source = (REPO_ROOT / "unsloth" / "models" / "rl.py").read_text(encoding = "utf-8")
    block = source.split("all_sm90 = (", 1)[1].split("\n            )", 1)[0]
    return eval("(" + block + ")")


@pytest.mark.parametrize(
    "caps,unreadable,expected",
    [
        ([(10, 0), (10, 0)], set(), True),
        ([(10, 0), (8, 0)], set(), False),
        ([(8, 0), (10, 0)], set(), False),
        # A card whose capability query raises must turn TMA off, not abort the import.
        ([(10, 0), (10, 0)], {1}, False),
    ],
)
def test_grpo_tma_options_need_every_visible_card(caps, unreadable, expected):
    import builtins

    fake = _fake_torch(caps)
    real_cap = fake.cuda.get_device_capability

    def get_device_capability(device = None):
        if device in unreadable:
            raise RuntimeError("unreadable")
        return real_cap(device)

    fake.cuda.get_device_capability = get_device_capability

    def visible(with_index = False):
        out = []
        for i in range(fake.cuda.device_count()):
            try:
                cap = tuple(fake.cuda.get_device_capability(i))
            except Exception:
                continue
            out.append((i, cap) if with_index else cap)
        return out

    stub = types.SimpleNamespace(cuda_visible_capabilities = visible)
    safe_builtins = dict(vars(builtins))
    safe_builtins["__import__"] = lambda name, *a, **k: stub
    expression = _grpo_all_sm90_expression()
    assert "get_device_capability()" not in expression
    assert eval(expression, {"__builtins__": safe_builtins, "torch": fake}) is expected


@pytest.mark.parametrize(
    "caps,current,bf16",
    [
        ([(9, 0), (9, 0), (7, 5)], 0, True),
        ([(9, 0), (9, 0), (7, 5)], 2, False),
        ([(9, 0), (9, 0), (9, 0)], 0, True),
    ],
)
def test_an_unreadable_card_does_not_matter(monkeypatch, capsys, caps, current, bf16):
    # Card 1 unreadable: only the current device is read.
    fake = _fake_torch(caps)
    real_cap = fake.cuda.get_device_capability

    def get_device_capability(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return real_cap(device)

    fake.cuda.get_device_capability = get_device_capability
    fake.cuda.set_device(current)
    assert _run_gpu_init_chain(fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16
    assert _run_model_utils_chain(fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16


@pytest.mark.parametrize("world_size,applied", [("2", True), ("1", False)])
def test_a_distributed_launch_keeps_every_card_for_the_gfx906_switch(
    monkeypatch, capsys, world_size, applied
):
    monkeypatch.setenv("WORLD_SIZE", world_size)
    archs = ["gfx1100", "gfx906:sramecc+:xnack-"]
    fake = _fake_torch([SM90] * len(archs), archs = archs)
    environ = {}
    dynamo_config = types.SimpleNamespace(disable = False)
    body = _block(
        DEVICE_TYPE_PY,
        "# gfx906 (MI50",
        '\n\nif DEVICE_TYPE == "cuda" and torch.cuda.is_available():',
    )
    namespace = _module_namespace(
        dt,
        fake,
        monkeypatch,
        DEVICE_TYPE = "hip",
        os = types.SimpleNamespace(environ = environ),
        apply_pre_volta_compile_workaround = lambda major: dt.apply_pre_volta_compile_workaround(
            major, environ = environ, dynamo_config = dynamo_config
        ),
    )
    exec(compile(body, str(DEVICE_TYPE_PY), "exec"), namespace)
    capsys.readouterr()
    assert dynamo_config.disable is applied


class _Placed:
    def __init__(self, device_map):
        self.hf_device_map = device_map


def _check_placement(
    monkeypatch,
    caps,
    device_map,
    dtype,
    current = 0,
    model = None,
    at_train_start = False,
):
    import torch

    import unsloth.models._utils as model_utils

    fake = _fake_torch(caps)
    fake.cuda.set_device(current)
    fake.cuda.get_device_name = lambda i: f"card{i}"
    fake.device = torch.device
    fake.bfloat16, fake.float16 = torch.bfloat16, torch.float16
    disabled = []
    monkeypatch.setattr(model_utils, "torch", fake)
    monkeypatch.setattr(model_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(model_utils, "DEVICE_COUNT", len(caps))
    monkeypatch.setattr(
        model_utils,
        "apply_pre_volta_compile_workaround",
        lambda major: disabled.append(major) or True,
    )
    model = _Placed(device_map) if model is None else model
    if at_train_start:
        # The once-per-train() hook every Unsloth trainer path runs (plain Trainer loop, SFT, RL).
        model_utils._unsloth_reset_stray_compile_cache(types.SimpleNamespace(model = model))
    else:
        model_utils.check_mixed_gpu_placement(model, dtype)
    return disabled


def test_layers_on_a_pre_volta_card_turn_compile_off(monkeypatch, capsys):
    import torch

    disabled = _check_placement(
        monkeypatch, [SM90, SM61], {"model.layers.0": 0, "model.layers.1": 1}, torch.float16
    )
    out = capsys.readouterr().out
    assert disabled == [0]
    assert "GPU 1 (card1)" in out and "torch.compile" in out


def test_bf16_layers_on_a_card_without_bf16_warn(monkeypatch, capsys):
    import torch

    disabled = _check_placement(
        monkeypatch, [SM86, SM75], {"model.layers.0": 0, "lm_head": 1}, torch.bfloat16
    )
    out = capsys.readouterr().out
    assert disabled == []
    assert "bfloat16 layers on GPU 1 (card1)" in out


@pytest.mark.parametrize(
    "caps,device_map",
    [
        ([SM86, SM61], {"": 0}),  # everything on the current card: nothing to check
        ([SM90, SM90], {"model.layers.0": 0, "model.layers.1": 1}),  # identical cards
        ([SM86, SM75], {"model": "cpu", "lm_head": 0}),
    ],
)
def test_placement_on_capable_cards_changes_nothing(monkeypatch, capsys, caps, device_map):
    import torch
    assert _check_placement(monkeypatch, caps, device_map, torch.bfloat16) == []
    assert capsys.readouterr().out == ""


def test_set_device_to_a_pre_volta_card_turns_compile_off(monkeypatch, capsys):
    # Imported on the H100, then torch.cuda.set_device(1) and the whole model loads on the GTX 1080.
    import torch

    disabled = _check_placement(monkeypatch, [SM90, SM61], {"": 1}, torch.float16, current = 1)
    out = capsys.readouterr().out
    assert disabled == [0]
    assert "GPU 1 (card1)" in out and "torch.compile" in out


def test_bf16_load_on_a_current_card_without_bf16_warns(monkeypatch, capsys):
    # SUPPORTS_BFLOAT16 stays the import-time answer; the post-load check covers a set_device later.
    import torch

    disabled = _check_placement(monkeypatch, [SM86, SM75], {"": 1}, torch.bfloat16, current = 1)
    out = capsys.readouterr().out
    assert disabled == []
    assert "bfloat16 layers on GPU 1 (card1)" in out


def test_one_gpu_hosts_never_walk_the_placement(monkeypatch, capsys):
    import torch

    import unsloth.models._utils as model_utils

    class _Untouchable:
        @property
        def hf_device_map(self):
            raise AssertionError("one-GPU hosts must not walk the placement")

    fake = _fake_torch([SM61])
    monkeypatch.setattr(model_utils, "torch", fake)
    monkeypatch.setattr(model_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(model_utils, "DEVICE_COUNT", 1)
    monkeypatch.setattr(
        model_utils,
        "apply_pre_volta_compile_workaround",
        lambda major: pytest.fail("one-GPU hosts keep the import-time switch"),
    )
    model_utils.check_mixed_gpu_placement(_Untouchable(), torch.bfloat16)
    assert capsys.readouterr().out == ""


def test_diffusion_loader_checks_the_placement():
    import ast

    tree = ast.parse(
        (REPO_ROOT / "unsloth" / "models" / "diffusion.py").read_text(encoding = "utf-8")
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "check_mixed_gpu_placement"
    ]
    assert calls, "FastDiffusionModel.from_pretrained must check where device_map put the weights"
    assert [ast.unparse(a) for a in calls[0].args] == ["model", "dtype"]


def test_both_llama_load_branches_check_the_placement():
    """num_labels loads go through AutoModelForSequenceClassification; they must check device_map too."""
    import ast

    tree = ast.parse((REPO_ROOT / "unsloth" / "models" / "llama.py").read_text(encoding = "utf-8"))

    def checks(nodes):
        return any(
            isinstance(n, ast.Call) and getattr(n.func, "id", None) == "check_mixed_gpu_placement"
            for stmt in nodes
            for n in ast.walk(stmt)
        )

    branches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "num_labels is not None"
        and any(
            "AutoModelForSequenceClassification.from_pretrained" in ast.unparse(s) for s in n.body
        )
    ]
    assert branches, "sequence-classification load branch not found"
    for branch in branches:
        assert checks(
            branch.body
        ), "the num_labels branch must check where device_map put the weights"
        assert branch.orelse and checks(
            branch.orelse
        ), "the causal LM branch must check the placement"


def test_sentence_transformer_direct_loads_check_the_placement():
    """FastSentenceTransformer's direct SentenceTransformer(model_name, ...) loads (incl. the compiled
    fast encoder path) bypass the other loaders, so each must check where the weights landed."""
    import ast

    tree = ast.parse(
        (REPO_ROOT / "unsloth" / "models" / "sentence_transformer.py").read_text(encoding = "utf-8")
    )
    found = 0
    for block in ast.walk(tree):
        body = getattr(block, "body", None)
        if not isinstance(body, list):
            continue
        for i, stmt in enumerate(body):
            if not (
                isinstance(stmt, ast.Assign)
                and isinstance(stmt.value, ast.Call)
                and getattr(stmt.value.func, "id", None) == "SentenceTransformer"
                and stmt.value.args
                and getattr(stmt.value.args[0], "id", None) == "model_name"
            ):
                continue
            found += 1
            assert any(
                isinstance(n, ast.Call)
                and getattr(n.func, "id", None) == "check_mixed_gpu_placement"
                for later in body[i + 1 :]
                for n in ast.walk(later)
            ), f"SentenceTransformer load at line {stmt.lineno} must check the placement"
    assert found >= 2


class _ParamsOn:
    """A model with no hf_device_map (an extracted text core) whose parameters sit on these devices."""

    def __init__(
        self,
        *devices,
        dtype = None,
    ):
        import torch
        dtype = torch.float16 if dtype is None else dtype
        self._params = [types.SimpleNamespace(device = torch.device(d), dtype = dtype) for d in devices]

    def parameters(self):
        return iter(self._params)


def test_a_core_without_a_device_map_is_checked_on_every_card_it_spans(monkeypatch, capsys):
    import torch

    model = _ParamsOn("cuda:0", "cuda:0", "cuda:1", "cpu")
    disabled = _check_placement(monkeypatch, [SM90, SM61], None, torch.float16, model = model)
    assert disabled == [0]
    assert "GPU 1 (card1)" in capsys.readouterr().out


def test_a_core_on_capable_cards_only_changes_nothing(monkeypatch, capsys):
    import torch

    # The discarded sibling's card (GPU 1) is no longer among the retained parameters.
    model = _ParamsOn("cuda:0", "cuda:0")
    assert _check_placement(monkeypatch, [SM90, SM61], None, torch.bfloat16, model = model) == []
    assert capsys.readouterr().out == ""


def test_text_only_composite_loads_check_the_retained_core():
    """The placement check runs after _text_trainable_core drops the vision / audio siblings."""
    import ast

    tree = ast.parse((REPO_ROOT / "unsloth" / "models" / "vision.py").read_text(encoding = "utf-8"))
    checked = 0
    for block in ast.walk(tree):
        body = getattr(block, "body", None)
        if not isinstance(body, list):
            continue
        calls = [
            (i, getattr(n.func, "id", None))
            for i, stmt in enumerate(body)
            for n in ast.walk(stmt)
            if isinstance(n, ast.Call)
            and getattr(n.func, "id", None) in ("_text_trainable_core", "check_mixed_gpu_placement")
        ]
        core = [i for i, name in calls if name == "_text_trainable_core"]
        check = [i for i, name in calls if name == "check_mixed_gpu_placement"]
        # Outer blocks hold both inside one statement; the block that lists them as siblings decides.
        if core and check and min(core) != min(check):
            checked += 1
            assert min(check) > min(
                core
            ), "check the placement of the retained core, not the composite"
    assert checked


class _MovedAfterLoad(_ParamsOn):
    """Loaded with device_map = {"": 0} on the H100, then moved with .to("cuda:1"): the stale
    hf_device_map still names GPU 0."""

    hf_device_map = {"": 0}


@pytest.mark.parametrize("model_cls", [_ParamsOn, _MovedAfterLoad], ids = ["cpu_load", "to_after"])
def test_a_model_moved_onto_a_pre_volta_card_after_loading_is_checked_at_train_start(
    monkeypatch, capsys, model_cls
):
    import torch

    model = model_cls("cuda:1", "cuda:1")
    disabled = _check_placement(
        monkeypatch, [SM90, SM61], None, torch.float16, model = model, at_train_start = True
    )
    assert disabled == [0]
    assert "GPU 1 (card1)" in capsys.readouterr().out


def test_train_start_on_capable_cards_changes_nothing(monkeypatch, capsys):
    import torch

    model = _MovedAfterLoad("cuda:0", "cpu")
    disabled = _check_placement(
        monkeypatch, [SM90, SM61], None, torch.float16, model = model, at_train_start = True
    )
    assert disabled == []
    assert capsys.readouterr().out == ""


def test_bf16_weights_placed_on_a_card_without_bf16_warn_at_train_start(monkeypatch, capsys):
    # Loaded in bf16 on CPU, then the Trainer put it on the T4: the live check infers the dtype.
    import torch

    model = _ParamsOn("cuda:1", "cuda:1", dtype = torch.bfloat16)
    disabled = _check_placement(
        monkeypatch, [SM86, SM75], None, None, model = model, at_train_start = True
    )
    assert disabled == []
    assert "bfloat16 layers on GPU 1 (card1)" in capsys.readouterr().out


def test_bf16_weights_only_on_the_bf16_card_do_not_warn_at_train_start(monkeypatch, capsys):
    import torch

    model = _ParamsOn("cuda:0", dtype = torch.bfloat16)
    model._params += _ParamsOn("cuda:1", dtype = torch.float16)._params
    disabled = _check_placement(
        monkeypatch, [SM86, SM75], None, None, model = model, at_train_start = True
    )
    assert disabled == []
    assert capsys.readouterr().out == ""


class _PlacedByTrainer(_ParamsOn):
    """A FastSentenceTransformer loaded with device_map = "cpu" whose LoRA compile is pending."""

    _compile_pending = True
    _compile_mode = "default"

    def place(self, device):
        import torch
        for p in self._params:
            p.device = torch.device(device)


@pytest.mark.parametrize("moved", ["trainer_init", "before_train"])
def test_sentence_transformer_trainer_checks_the_placement_before_the_deferred_compile(
    monkeypatch, capsys, moved
):
    import sys

    import unsloth.models.sentence_transformer as st

    model = _PlacedByTrainer("cpu", "cpu")
    # Load time: everything on CPU, so nothing to check yet.
    disabled = _check_placement(monkeypatch, [SM90, SM61], None, None, model = model)
    assert disabled == []
    model.place("cuda:0")  # loaded on CPU, the user picks the GTX 1080 via the current device
    compiled = []

    class SentenceTransformerTrainer:
        def __init__(
            self,
            model = None,
            args = None,
        ):
            self.model = model
            if moved == "trainer_init":
                model.place("cuda:1")  # Trainer moves the model to args.device

        def train(self):
            return list(disabled)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformerTrainer = SentenceTransformerTrainer),
    )
    monkeypatch.setattr(
        st.FastSentenceTransformer, "_estimate_compile_threshold", staticmethod(lambda *a, **k: 1)
    )
    monkeypatch.setattr(
        st.FastSentenceTransformer,
        "_apply_torch_compile",
        staticmethod(lambda m, mode = "default": compiled.append(mode)),
    )
    st._patch_sentence_transformer_trainer()
    trainer = SentenceTransformerTrainer(
        model = model, args = types.SimpleNamespace(max_steps = 100, fp16 = False, bf16 = False)
    )
    assert compiled == ["default"]
    if moved == "before_train":
        model.place("cuda:1")
    # The stubbed switch leaves Dynamo on, so the train() re-check may fire again after __init__'s.
    assert set(trainer.train()) == {0}
    assert "GPU 1 (card1)" in capsys.readouterr().out


def _public_bf16_helper(
    fake,
    monkeypatch,
    world_size = "1",
):
    """The import-time chain of models/_utils.py, then its public is_bfloat16_supported, both from
    the real source, so the helper reads exactly the state that chain leaves behind."""
    monkeypatch.setenv("WORLD_SIZE", world_size)
    namespace = _run_model_utils_chain(fake, monkeypatch)
    body = _block(MODEL_UTILS, "def is_bfloat16_supported(", "\n\n\ndef ")
    exec(compile(body, str(MODEL_UTILS), "exec"), namespace)
    return namespace["is_bfloat16_supported"]


@pytest.mark.parametrize("caps,start", [([SM61, SM86], 0), ([SM90, SM75], 0), ([SM75, SM86], 1)])
def test_public_bf16_helper_follows_set_device_after_import(monkeypatch, caps, start):
    # set_device after import: notebooks' fp16 = not is_bfloat16_supported() must match the new card.
    fake = _fake_torch(caps)
    fake.cuda.set_device(start)
    helper = _public_bf16_helper(fake, monkeypatch)
    assert helper() is (caps[start][0] >= 8)
    other = 1 - start
    fake.cuda.set_device(other)
    assert helper() is (caps[other][0] >= 8)
    assert helper() is bool(fake.cuda.is_bf16_supported(False))
    fake.cuda.set_device(start)
    assert helper() is (caps[start][0] >= 8)


@pytest.mark.parametrize("current", [0, 1])
def test_public_bf16_helper_keeps_the_weakest_card_on_a_distributed_launch(monkeypatch, current):
    # Every rank must pick one dtype: set_device onto the rank's own card changes nothing.
    fake = _fake_torch([SM90, SM75])
    helper = _public_bf16_helper(fake, monkeypatch, world_size = "2")
    fake.cuda.set_device(current)
    assert helper() is False


@pytest.mark.parametrize("caps", [[SM86], [SM90, SM90, SM90]], ids = ["one_gpu", "same_cards"])
def test_public_bf16_helper_is_the_constant_on_homogeneous_hosts(monkeypatch, caps):
    fake = _fake_torch(caps)
    helper = _public_bf16_helper(fake, monkeypatch)
    calls = []
    fake.cuda.current_device = lambda: calls.append(1) or 0
    assert helper() is True
    assert calls == []


LOADER_FILES = [
    REPO_ROOT / "unsloth" / "models" / name
    for name in ("llama.py", "vision.py", "loader.py", "diffusion.py", "sentence_transformer.py")
]


@pytest.mark.parametrize("path", LOADER_FILES, ids = lambda p: p.name)
def test_loader_dtype_choices_ask_the_helper_at_call_time(path):
    # The module-level constant is the import-time answer, wrong after a set_device.
    import ast

    stale = []
    for node in ast.walk(ast.parse(path.read_text(encoding = "utf-8"))):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        names = [
            n for n in ast.walk(node) if isinstance(n, ast.Name) and n.id == "SUPPORTS_BFLOAT16"
        ]
        if any(isinstance(n.ctx, ast.Load) for n in names) and not any(
            isinstance(n.ctx, ast.Store) for n in names
        ):
            stale.append(f"{node.name}:{node.lineno}")
    assert stale == []


def test_an_unreadable_peer_keeps_the_cards_that_were_read(monkeypatch):
    # One card's capability query raising must not drop the per-device table for the others.
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    fake = _fake_torch([SM90, None, SM75])
    read = fake.cuda.get_device_capability

    def get_device_capability(device = None):
        if read(device) is None:
            raise RuntimeError("capability query failed")
        return read(device)

    fake.cuda.get_device_capability = get_device_capability
    namespace = _run_model_utils_chain(fake, monkeypatch)
    assert namespace["_BF16_BY_DEVICE"] == {0: True, 2: False}
