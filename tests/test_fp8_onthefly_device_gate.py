import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: F401
from unsloth.models import loader_utils

torchao = pytest.importorskip("torchao")


def _spoof(
    monkeypatch,
    device_type,
    torchao_version = "0.18.0",
    capability = (9, 0),
    transformers_version = "5.17.0",
):
    monkeypatch.setattr(loader_utils, "DEVICE_TYPE_TORCH", device_type)
    monkeypatch.setattr(loader_utils, "transformers_version", transformers_version)
    monkeypatch.setattr(torchao, "__version__", torchao_version)
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: device_type == "cuda")
    monkeypatch.setattr(torch.version, "cuda", "12.8" if device_type == "cuda" else None)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)


@pytest.mark.parametrize(
    "load_in_fp8, transformers_version, mode",
    [(True, "5.17.0", "row"), (True, "4.57.6", "row"), ("block", "4.57.6", "block")],
)
def test_xpu_is_admitted(monkeypatch, load_in_fp8, transformers_version, mode):
    _spoof(monkeypatch, "xpu", transformers_version = transformers_version)
    mode_out = loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference = False)
    assert mode_out == mode


@pytest.mark.parametrize("load_in_fp8", [True, "block"])
def test_xpu_with_vllm_is_refused(monkeypatch, load_in_fp8):
    _spoof(monkeypatch, "xpu", transformers_version = "4.57.6")
    with pytest.raises(ValueError, match = "fast_inference = False"):
        loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference = True)


def test_xpu_block_on_transformers_5_is_refused(monkeypatch):
    _spoof(monkeypatch, "xpu", transformers_version = "5.17.0")
    with pytest.raises(ValueError, match = "transformers < 5"):
        loader_utils._get_fp8_mode_and_check_settings("block", fast_inference = False)


@pytest.mark.parametrize(
    "load_in_fp8, torchao_version, refused",
    [
        (True, "0.17.0", True),
        (True, "0.18.0", False),
        ("block", "0.17.0", True),
        ("block", "0.18.0", False),
    ],
)
def test_xpu_needs_torchao_0_18(monkeypatch, load_in_fp8, torchao_version, refused):
    _spoof(monkeypatch, "xpu", torchao_version, transformers_version = "4.57.6")
    if refused:
        with pytest.raises(ValueError, match = "on Intel XPU requires torchao 0.18.0"):
            loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference = False)
    else:
        loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference = False)


@pytest.mark.parametrize(
    "capability, torchao_version, refused",
    [
        ((8, 9), "0.18.0", True),
        ((9, 0), "0.14.0", True),
        ((9, 0), "0.15.0", False),
    ],
)
def test_cuda_rules_are_unchanged(monkeypatch, capability, torchao_version, refused):
    _spoof(monkeypatch, "cuda", torchao_version, capability)
    for load_in_fp8 in (True, "block"):
        for fast_inference in (False, True):
            if refused:
                with pytest.raises(ValueError):
                    loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference)
            else:
                loader_utils._get_fp8_mode_and_check_settings(load_in_fp8, fast_inference)


@pytest.mark.parametrize("device_type", ["mps", "npu", "cpu"])
def test_other_devices_are_still_refused(monkeypatch, device_type):
    _spoof(monkeypatch, device_type)
    with pytest.raises(ValueError, match = "requires H100 GPUs or after"):
        loader_utils._get_fp8_mode_and_check_settings(True, fast_inference = False)
