# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""XPU bf16 must be native on the XPU in use (the current device, as torch.xpu.is_bf16_supported()
asks), judged by its own aspects (torch's default probe counts emulation). An unused sibling that
emulates bf16 never narrows it."""

import re
import types
from pathlib import Path

import pytest

pytest.importorskip("torch")

import unsloth.device_type as dt

REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_INIT = REPO_ROOT / "unsloth" / "_gpu_init.py"
MODEL_UTILS = REPO_ROOT / "unsloth" / "models" / "_utils.py"

if not hasattr(dt, "torch"):
    pytest.skip(
        "device_type stub or MLX host; the real probes are not loaded", allow_module_level = True
    )

# Per device: a dict of aspect values, "old" (torch without the aspect) or None (unreadable).
NATIVE = {"has_bfloat16_conversions": True}
EMULATED = {"has_bfloat16_conversions": False, "has_subgroup_matrix_multiply_accumulate": True}
NO_OCLOC = {"has_bfloat16_conversions": False}

# (devices, current device, native bf16)
CASES = [
    pytest.param([NATIVE], 0, True, id = "1dev-native"),
    pytest.param([EMULATED], 0, False, id = "1dev-emulated"),
    pytest.param([NO_OCLOC], 0, True, id = "1dev-no-libocloc"),
    pytest.param(["old"], 0, True, id = "1dev-old-torch"),
    pytest.param([None], 0, True, id = "1dev-unreadable"),
    pytest.param([NATIVE, NATIVE], 0, True, id = "2dev-native"),
    pytest.param([NATIVE, EMULATED], 0, True, id = "2dev-mixed"),
    pytest.param([EMULATED, NATIVE], 1, True, id = "2dev-mixed-rev"),
    pytest.param([NATIVE, EMULATED], 1, False, id = "2dev-mixed-current-emulated"),
    pytest.param([EMULATED, NATIVE], 0, False, id = "2dev-mixed-rev-current-emulated"),
    pytest.param([EMULATED, EMULATED], 0, False, id = "2dev-emulated"),
    pytest.param([EMULATED, EMULATED], 1, False, id = "2dev-emulated-dev1"),
    pytest.param([NO_OCLOC, NO_OCLOC], 0, True, id = "2dev-no-libocloc"),
    pytest.param([EMULATED, NO_OCLOC], 1, True, id = "2dev-emulated+no-libocloc"),
    pytest.param([NATIVE, NO_OCLOC], 0, True, id = "2dev-native+no-libocloc"),
    pytest.param([NO_OCLOC, NATIVE], 1, True, id = "2dev-no-libocloc+native"),
    pytest.param(["old", "old"], 0, True, id = "2dev-old-torch"),
    pytest.param(["old", EMULATED], 0, True, id = "2dev-old-attr+emulated"),
    pytest.param(["old", EMULATED], 1, False, id = "2dev-old-attr+emulated-current"),
    pytest.param([None, NO_OCLOC], 0, True, id = "2dev-unreadable+no-libocloc"),
]


@pytest.fixture(autouse = True)
def _single_process(monkeypatch):
    for name in ("LOCAL_WORLD_SIZE", "WORLD_SIZE", "LOCAL_RANK", "RANK"):
        monkeypatch.delenv(name, raising = False)


def _fake_torch(devices, current = 0):
    def get_device_properties(device = None):
        spec = devices[current if device is None else device]
        if spec is None:
            raise RuntimeError("unreadable device")
        props = types.SimpleNamespace(name = f"Intel XPU {device}")
        if spec != "old":
            for name, value in spec.items():
                setattr(props, name, value)
        return props

    def is_bf16_supported(including_emulation: bool = True) -> bool:
        return including_emulation or bool(
            getattr(get_device_properties(), "has_bfloat16_conversions", True)
        )

    return types.SimpleNamespace(
        xpu = types.SimpleNamespace(
            device_count = lambda: len(devices),
            current_device = lambda: current,
            is_available = lambda: True,
            get_device_properties = get_device_properties,
            is_bf16_supported = is_bf16_supported,
        ),
    )


def _xpu_arm(path):
    source = path.read_text(encoding = "utf-8")
    match = re.search(
        r'\nelif DEVICE_TYPE == "xpu":\n((?:    #.*\n)*    SUPPORTS_BFLOAT16 = .*?)\nelif DEVICE_TYPE == "npu":',
        source,
        re.S,
    )
    return 'if DEVICE_TYPE == "xpu":\n' + match.group(1)


def _run(
    path,
    module,
    fake,
    monkeypatch,
    zoo_shares = True,
):
    monkeypatch.setattr(dt, "torch", fake)
    namespace = dict(vars(module))
    namespace.update(torch = fake, DEVICE_TYPE = "xpu", zoo_shares_xpu_bf16 = lambda: zoo_shares)
    exec(compile(_xpu_arm(path), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("devices,current,bf16", CASES)
def test_gpu_init_xpu_bf16_is_native_on_the_current_device(monkeypatch, devices, current, bf16):
    import unsloth._gpu_init as gpu_init

    fake = _fake_torch(devices, current)
    original = fake.xpu.is_bf16_supported
    assert _run(GPU_INIT, gpu_init, fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16
    assert fake.xpu.is_bf16_supported is original


@pytest.mark.parametrize("devices,current,bf16", CASES)
def test_model_utils_xpu_bf16_is_native_on_the_current_device(monkeypatch, devices, current, bf16):
    import unsloth.models._utils as model_utils
    namespace = _run(MODEL_UTILS, model_utils, _fake_torch(devices, current), monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16


@pytest.mark.parametrize(
    "device, bf16",
    [
        (0, True),
        (1, False),
        ("xpu:1", False),
        ("xpu", True),
        ([0, 1], False),
        ("cpu", True),
    ],
)
def test_xpu_bf16_judges_the_devices_a_model_uses(monkeypatch, device, bf16):
    import torch

    fake = _fake_torch([NATIVE, EMULATED], 0)
    fake.device = torch.device
    monkeypatch.setattr(dt, "torch", fake)
    assert dt.xpu_supports_native_bf16(device) is bf16


# torchrun imports unsloth before prepare_device_map selects the rank's XPU, so every rank reads
# current device 0 at import: judge the launch's XPUs together instead, one answer for all ranks.
LAUNCH_CASES = [
    pytest.param(
        [NATIVE, EMULATED], {"LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"}, False, id = "2ranks-mixed"
    ),
    pytest.param(
        [EMULATED, NATIVE],
        {"LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"},
        False,
        id = "2ranks-mixed-rev",
    ),
    pytest.param(
        [NATIVE, NATIVE], {"LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"}, True, id = "2ranks-native"
    ),
    pytest.param(
        [NATIVE, EMULATED], {"LOCAL_WORLD_SIZE": "1", "WORLD_SIZE": "2"}, True, id = "1rank-per-node"
    ),
    pytest.param([NATIVE, EMULATED], {"WORLD_SIZE": "2"}, False, id = "no-local-world-all-visible"),
    pytest.param(
        [NATIVE], {"LOCAL_WORLD_SIZE": "2", "WORLD_SIZE": "2"}, True, id = "affinity-mask-one-visible"
    ),
    pytest.param(
        [NATIVE, EMULATED], {"LOCAL_WORLD_SIZE": "1", "WORLD_SIZE": "1"}, True, id = "single-process"
    ),
    pytest.param(
        [NATIVE, EMULATED], {"LOCAL_WORLD_SIZE": "x", "WORLD_SIZE": "y"}, True, id = "garbage-env"
    ),
]


@pytest.mark.parametrize("devices,env,bf16", LAUNCH_CASES)
@pytest.mark.parametrize("path_name", ["GPU_INIT", "MODEL_UTILS"])
def test_xpu_bf16_under_a_launch_judges_every_rank_xpu(monkeypatch, devices, env, bf16, path_name):
    import unsloth._gpu_init as gpu_init
    import unsloth.models._utils as model_utils

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    module = gpu_init if path_name == "GPU_INIT" else model_utils
    namespace = _run(globals()[path_name], module, _fake_torch(devices, 0), monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16


@pytest.mark.parametrize("devices,env,_bf16", LAUNCH_CASES)
def test_launch_answer_matches_zoo(monkeypatch, devices, env, _bf16):
    zoo_dt = pytest.importorskip("unsloth_zoo.device_type")
    if not hasattr(zoo_dt, "_xpu_launch_indices"):
        pytest.skip("installed unsloth_zoo predates _xpu_launch_indices")
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fake = _fake_torch(devices, 0)
    monkeypatch.setattr(dt, "torch", fake)
    monkeypatch.setattr(zoo_dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is zoo_dt._xpu_bf16_supported()


@pytest.mark.parametrize(
    "devices, bf16_after_switch",
    [
        pytest.param([NATIVE, EMULATED], False, id = "native-then-emulated"),
        pytest.param([EMULATED, NATIVE], True, id = "emulated-then-native"),
    ],
)
def test_is_bfloat16_supported_follows_set_device_after_import(
    monkeypatch, devices, bf16_after_switch
):
    # unsloth#14622 on XPU: import on XPU 0, then torch.xpu.set_device(1) before loading. The loader
    # asks is_bfloat16_supported(); it must answer for XPU 1, as the trainer's zoo probe does.
    import unsloth.models._utils as model_utils

    state = {"current": 0}
    fake = _fake_torch(devices, 0)
    fake.xpu.current_device = lambda: state["current"]
    monkeypatch.setattr(dt, "torch", fake)
    monkeypatch.setattr(model_utils, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(model_utils, "zoo_shares_xpu_bf16", lambda: True)
    monkeypatch.setattr(model_utils, "SUPPORTS_BFLOAT16", not bf16_after_switch)  # cached at import
    assert model_utils.is_bfloat16_supported() is (not bf16_after_switch)
    state["current"] = 1
    assert model_utils.is_bfloat16_supported() is bf16_after_switch


def test_is_bfloat16_supported_keeps_the_cached_answer_off_xpu(monkeypatch):
    import unsloth.models._utils as model_utils

    for answer in (True, False):
        monkeypatch.setattr(model_utils, "DEVICE_TYPE", "cuda")
        monkeypatch.setattr(model_utils, "SUPPORTS_BFLOAT16", answer)
        assert model_utils.is_bfloat16_supported() is answer
    monkeypatch.setattr(model_utils, "DEVICE_TYPE", "xpu")
    monkeypatch.setattr(model_utils, "zoo_shares_xpu_bf16", lambda: False)  # older zoo: old answer
    assert model_utils.is_bfloat16_supported() is False


def test_xpu_unreadable_current_device_keeps_true(monkeypatch):
    def boom():
        raise RuntimeError("no context")

    fake = _fake_torch([EMULATED], 0)
    fake.xpu.current_device = boom
    monkeypatch.setattr(dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is True


def test_xpu_unreadable_count_keeps_true(monkeypatch):
    def boom():
        raise RuntimeError("no driver")

    fake = types.SimpleNamespace(xpu = types.SimpleNamespace(device_count = boom))
    monkeypatch.setattr(dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is True


@pytest.mark.parametrize("devices,current,_bf16", CASES)
def test_matches_zoo_answer(monkeypatch, devices, current, _bf16):
    """Must match zoo: a mismatch loads bf16 then trains in float16."""
    zoo_dt = pytest.importorskip("unsloth_zoo.device_type")
    if not hasattr(zoo_dt, "_xpu_bf16_supported"):
        pytest.skip("installed unsloth_zoo predates _xpu_bf16_supported")
    fake = _fake_torch(devices, current)
    monkeypatch.setattr(dt, "torch", fake)
    monkeypatch.setattr(zoo_dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is zoo_dt._xpu_bf16_supported()


@pytest.mark.parametrize("path_name", ["GPU_INIT", "MODEL_UTILS"])
def test_older_zoo_keeps_the_old_xpu_answer(monkeypatch, path_name):
    """An older zoo still trains in bf16 on emulated XPUs; loading fp16 would disagree with it."""
    import unsloth._gpu_init as gpu_init
    import unsloth.models._utils as model_utils

    module = gpu_init if path_name == "GPU_INIT" else model_utils
    fake = _fake_torch([EMULATED])
    namespace = _run(globals()[path_name], module, fake, monkeypatch, zoo_shares = False)
    assert namespace["SUPPORTS_BFLOAT16"] is True


def test_zoo_shares_xpu_bf16_tracks_the_helper(monkeypatch):
    zoo_dt = pytest.importorskip("unsloth_zoo.device_type")
    monkeypatch.setattr(zoo_dt, "_xpu_bf16_supported", lambda: True, raising = False)
    assert dt.zoo_shares_xpu_bf16() is True
    monkeypatch.delattr(zoo_dt, "_xpu_bf16_supported")
    assert dt.zoo_shares_xpu_bf16() is False
