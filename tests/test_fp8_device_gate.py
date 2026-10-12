# SPDX-License-Identifier: AGPL-3.0-only
# verify_fp8_support_if_applicable admits FP8 checkpoints on CUDA and XPU only.
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

import unsloth  # noqa: F401
from unsloth.models import _utils


def _config(quant_method):
    return SimpleNamespace(quantization_config = {"quant_method": quant_method})


@pytest.mark.parametrize("quant_method", ["fp8", "fbgemm_fp8"])
def test_xpu_is_admitted(monkeypatch, quant_method):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", "xpu")
    _utils.verify_fp8_support_if_applicable(_config(quant_method))


@pytest.mark.parametrize("device_type", ["hip", "mps", "npu", "cpu"])
@pytest.mark.parametrize("quant_method", ["fp8", "fbgemm_fp8"])
def test_other_devices_are_still_refused(monkeypatch, device_type, quant_method):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", device_type)
    with pytest.raises(ValueError, match = "FP8 quantization is only supported on"):
        _utils.verify_fp8_support_if_applicable(_config(quant_method))


@pytest.mark.parametrize("device_type", ["xpu", "hip", "mps", "cpu"])
def test_non_fp8_checkpoints_load_anywhere(monkeypatch, device_type):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", device_type)
    _utils.verify_fp8_support_if_applicable(_config("bitsandbytes"))
    _utils.verify_fp8_support_if_applicable(SimpleNamespace())


@pytest.mark.parametrize(
    "capability, quant_method, refused",
    [
        ((8, 0), "fp8", True),
        ((8, 9), "fp8", False),
        ((8, 9), "fbgemm_fp8", True),
        ((9, 0), "fbgemm_fp8", False),
    ],
)
def test_cuda_capability_rules_are_unchanged(monkeypatch, capability, quant_method, refused):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "spoofed")
    if refused:
        with pytest.raises(ValueError):
            _utils.verify_fp8_support_if_applicable(_config(quant_method))
    else:
        _utils.verify_fp8_support_if_applicable(_config(quant_method))
