# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Intel XPU bf16 must be native bf16 on every visible XPU. torch.xpu.is_bf16_supported() defaults to
including_emulation=True and is then always True, and models/_utils.py hard-coded True. Same answer
as unsloth_zoo.device_type._xpu_bf16_supported, which the trainer reads."""

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
EMULATED = {"has_bfloat16_conversions": False}

CASES = [
    pytest.param([NATIVE], True, id = "1dev-native"),
    pytest.param([EMULATED], False, id = "1dev-emulated"),
    pytest.param(["old"], True, id = "1dev-old-torch"),
    pytest.param([None], True, id = "1dev-unreadable"),
    pytest.param([NATIVE, NATIVE], True, id = "2dev-native"),
    pytest.param([NATIVE, EMULATED], False, id = "2dev-mixed"),
    pytest.param([EMULATED, NATIVE], False, id = "2dev-mixed-rev"),
    pytest.param([EMULATED, EMULATED], False, id = "2dev-emulated"),
    pytest.param(["old", "old"], True, id = "2dev-old-torch"),
    pytest.param(["old", EMULATED], False, id = "2dev-old-attr+emulated"),
    pytest.param([None, EMULATED], True, id = "2dev-unreadable+emulated"),
]


def _fake_torch(devices):
    def get_device_properties(device = None):
        spec = devices[device or 0]
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
            current_device = lambda: 0,
            is_available = lambda: True,
            get_device_properties = get_device_properties,
            is_bf16_supported = is_bf16_supported,
        ),
    )


def _xpu_arm(path):
    # The xpu arm of the SUPPORTS_BFLOAT16 chain (models/_utils.py has other xpu arms above it).
    source = path.read_text(encoding = "utf-8")
    match = re.search(
        r'\nelif DEVICE_TYPE == "xpu":\n((?:    #.*\n)*    SUPPORTS_BFLOAT16 = .*?)\nelif DEVICE_TYPE == "npu":',
        source,
        re.S,
    )
    return 'if DEVICE_TYPE == "xpu":\n' + match.group(1)


def _run(path, module, fake, monkeypatch):
    monkeypatch.setattr(dt, "torch", fake)
    namespace = dict(vars(module))
    namespace.update(torch = fake, DEVICE_TYPE = "xpu")
    exec(compile(_xpu_arm(path), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("devices,bf16", CASES)
def test_gpu_init_xpu_bf16_is_native_on_every_device(monkeypatch, devices, bf16):
    import unsloth._gpu_init as gpu_init

    fake = _fake_torch(devices)
    original = fake.xpu.is_bf16_supported
    assert _run(GPU_INIT, gpu_init, fake, monkeypatch)["SUPPORTS_BFLOAT16"] is bf16
    # torch's own probe stays untouched: Studio reads its strict answer.
    assert fake.xpu.is_bf16_supported is original


@pytest.mark.parametrize("devices,bf16", CASES)
def test_model_utils_xpu_bf16_is_native_on_every_device(monkeypatch, devices, bf16):
    import unsloth.models._utils as model_utils
    namespace = _run(MODEL_UTILS, model_utils, _fake_torch(devices), monkeypatch)
    assert namespace["SUPPORTS_BFLOAT16"] is bf16


def test_xpu_unreadable_count_keeps_true(monkeypatch):
    def boom():
        raise RuntimeError("no driver")

    fake = types.SimpleNamespace(xpu = types.SimpleNamespace(device_count = boom))
    monkeypatch.setattr(dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is True


@pytest.mark.parametrize("devices,_bf16", CASES)
def test_matches_zoo_answer(monkeypatch, devices, _bf16):
    """The trainer asks zoo's device_is_bf16_supported(); a different answer here loads bf16 and
    then trains in float16."""
    zoo_dt = pytest.importorskip("unsloth_zoo.device_type")
    if not hasattr(zoo_dt, "_xpu_bf16_supported"):
        pytest.skip("installed unsloth_zoo predates _xpu_bf16_supported")
    fake = _fake_torch(devices)
    monkeypatch.setattr(dt, "torch", fake)
    monkeypatch.setattr(zoo_dt, "torch", fake)
    assert dt.xpu_supports_native_bf16() is zoo_dt._xpu_bf16_supported()
