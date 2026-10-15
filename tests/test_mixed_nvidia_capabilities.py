# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""bf16 and the torch.compile kill switches are process-wide, so on a mixed host (e.g. RTX 3090 +
T4 with device_map="auto") they must hold on the weakest visible card, not on device 0. The
capability cache must also follow torch.cuda.set_device."""

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

CASES = [
    pytest.param([SM80, SM80], True, None, id = "sm80x2"),
    pytest.param([SM86, SM75], False, None, id = "sm86+sm75"),
    pytest.param([SM75, SM86], False, None, id = "sm75+sm86"),
    pytest.param([SM90, SM61], False, 6, id = "sm90+sm61"),
    pytest.param([SM61], False, 6, id = "sm61"),
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


@pytest.mark.parametrize("caps,bf16,_floor", CASES)
def test_gpu_init_bf16_needs_every_visible_card(monkeypatch, caps, bf16, _floor):
    fake = _fake_torch(caps)
    namespace = _run_gpu_init_chain(fake, monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16
    assert bool(fake.cuda.is_bf16_supported(False)) is bf16
    assert bool(fake.cuda.is_bf16_supported()) is bf16
    # Emulation is torch's own per-device answer and stays untouched.
    assert fake.cuda.is_bf16_supported(including_emulation = True) is True


@pytest.mark.parametrize("caps,bf16,_floor", CASES)
def test_mixed_host_says_why_bf16_is_off(monkeypatch, capsys, caps, bf16, _floor):
    _run_gpu_init_chain(_fake_torch(caps), monkeypatch)
    out = capsys.readouterr().out
    mixed = any(c[0] >= 8 for c in caps) and any(c[0] < 8 for c in caps)
    assert ("lack bfloat16" in out) is mixed
    if mixed:
        assert str([i for i, c in enumerate(caps) if c[0] < 8]) in out


@pytest.mark.parametrize("caps,bf16,_floor", CASES)
def test_model_utils_bf16_needs_every_visible_card(monkeypatch, caps, bf16, _floor):
    namespace = _run_model_utils_chain(_fake_torch(caps), monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16


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
    assert calls == [1, 0]  # one real probe per index
    with pytest.raises(KeyError):
        cached("cpu")  # not a CUDA device: passed through uncached, as torch would reject it
    assert model_utils._cache_device_capability(cached) is cached  # re-import does not stack
    assert inspect.signature(cached) == inspect.signature(probe)


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


@pytest.mark.parametrize("caps,_bf16,floor", CASES)
def test_pre_volta_switch_reads_the_weakest_visible_card(monkeypatch, capsys, caps, _bf16, floor):
    applied, out = _run_pre_volta(_fake_torch(caps), monkeypatch, capsys)
    assert applied == [min(c[0] for c in caps)]
    if floor is None:
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
        (["gfx1100", "gfx906:sramecc+:xnack-"], True),
        (["gfx906", "gfx1100"], True),
        (["gfx90a", "gfx942"], False),
    ],
)
def test_gfx906_switch_reads_every_visible_card(monkeypatch, capsys, archs, applied):
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
        ([(9, 0), (9, 0), (7, 5)], 0, False),  # a known old card still counts
        ([(9, 0), (9, 0), (7, 5)], 2, False),
        ([(9, 0), (9, 0), (9, 0)], 0, True),  # current device is sm90, as on main
    ],
)
def test_a_partial_scan_judges_every_card_it_read(monkeypatch, capsys, caps, current, bf16):
    # Card 1 unreadable: the gate judges the readable cards, which include the current device,
    # so it is never less strict than main's current-device rule.
    fake = _fake_torch(caps)
    real_cap = fake.cuda.get_device_capability

    def get_device_capability(device = None):
        if device == 1:
            raise RuntimeError("unreadable")
        return real_cap(device)

    fake.cuda.get_device_capability = get_device_capability
    fake.cuda.set_device(current)
    namespace = _run_gpu_init_chain(fake, monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16
    out = capsys.readouterr().out
    assert ("[2]" in out) is (not bf16) and "[1]" not in out
    assert _run_model_utils_chain(fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16


def test_bf16_notice_names_real_device_indices(monkeypatch, capsys):
    # Device 2 is the pre-Ampere card while device 0 is current: the notice names 2.
    _run_gpu_init_chain(_fake_torch([(9, 0), (8, 6), (7, 5)]), monkeypatch)
    out = capsys.readouterr().out
    assert "[2]" in out and "[0]" not in out and "[1]" not in out


@pytest.mark.skipif(not __import__("torch").cuda.is_available(), reason = "needs a CUDA device")
@pytest.mark.parametrize("spelling", [None, 0, "cuda:0", "device", "tensor"])
def test_capability_cache_traces_in_fullgraph(spelling):
    # Attention helpers call get_device_capability(Q.device) inside compiled regions.
    import torch

    import unsloth.models._utils as model_utils

    cached = model_utils._cache_device_capability(inspect.unwrap(torch.cuda.get_device_capability))

    def f(x):
        arg = {"device": torch.device("cuda"), "tensor": x.device}.get(spelling, spelling)
        return x * (cached() if arg is None else cached(arg))[0]

    torch._dynamo.reset()
    x = torch.ones(2, device = "cuda")
    assert torch.compile(f, fullgraph = True)(x)[0].item() == torch.cuda.get_device_capability(0)[0]
