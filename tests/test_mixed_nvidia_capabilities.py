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
    # Every DDP rank must pick one dtype, and a rank's own card is not current yet at import: a mixed
    # host keeps the weakest visible card there, as before. One process follows the current device.
    monkeypatch.setenv("WORLD_SIZE", world_size)
    fake = _fake_torch([SM86, SM75])
    run = _run_gpu_init_chain if chain == "gpu_init" else _run_model_utils_chain
    assert run(fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16


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


class _Placed:
    def __init__(self, device_map):
        self.hf_device_map = device_map


def _check_placement(
    monkeypatch,
    caps,
    device_map,
    dtype,
    current = 0,
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
    model_utils.check_mixed_gpu_placement(_Placed(device_map), dtype)
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
        if isinstance(n, ast.If) and ast.unparse(n.test) == "num_labels is not None"
        and any("AutoModelForSequenceClassification.from_pretrained" in ast.unparse(s) for s in n.body)
    ]
    assert branches, "sequence-classification load branch not found"
    for branch in branches:
        assert checks(branch.body), "the num_labels branch must check where device_map put the weights"
        assert branch.orelse and checks(branch.orelse), "the causal LM branch must check the placement"


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
                isinstance(n, ast.Call) and getattr(n.func, "id", None) == "check_mixed_gpu_placement"
                for later in body[i + 1 :]
                for n in ast.walk(later)
            ), f"SentenceTransformer load at line {stmt.lineno} must check the placement"
    assert found >= 2
