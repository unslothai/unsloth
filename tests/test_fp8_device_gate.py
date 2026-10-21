# SPDX-License-Identifier: AGPL-3.0-only
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
    # transformers 4.57.x: no 16-bit fallback below sm89, so the gate still refuses.
    monkeypatch.setattr(_utils, "_fp8_checkpoint_dequantizes_below_sm89", lambda: False)
    monkeypatch.setattr(_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "spoofed")
    if refused:
        with pytest.raises(ValueError):
            _utils.verify_fp8_support_if_applicable(_config(quant_method))
    else:
        _utils.verify_fp8_support_if_applicable(_config(quant_method))


def _spoof_cuda(monkeypatch, capability):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", "cuda")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: capability)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a: "spoofed")


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 6)])
def test_fp8_checkpoint_loads_dequantized_below_sm89(monkeypatch, capability):
    # T4 / A100 / A10: transformers 5.x decodes the weights to 16-bit, nothing needs FP8 compute.
    _spoof_cuda(monkeypatch, capability)
    monkeypatch.setattr(_utils, "_fp8_checkpoint_dequantizes_below_sm89", lambda: True)
    _utils.verify_fp8_support_if_applicable(_config("fp8"))
    _utils.verify_fp8_support_if_applicable(_config("fp8"), fast_inference = False)


@pytest.mark.parametrize("capability", [(7, 5), (8, 0)])
def test_vllm_load_of_fp8_checkpoint_still_needs_sm89(monkeypatch, capability):
    _spoof_cuda(monkeypatch, capability)
    monkeypatch.setattr(_utils, "_fp8_checkpoint_dequantizes_below_sm89", lambda: True)
    with pytest.raises(ValueError, match = "compute capability 8.9"):
        _utils.verify_fp8_support_if_applicable(_config("fp8"), fast_inference = True)


@pytest.mark.parametrize("capability", [(7, 5), (8, 0), (8, 9)])
def test_fbgemm_fp8_rule_is_unchanged_by_the_fallback(monkeypatch, capability):
    _spoof_cuda(monkeypatch, capability)
    monkeypatch.setattr(_utils, "_fp8_checkpoint_dequantizes_below_sm89", lambda: True)
    with pytest.raises(ValueError, match = "FBGEMM"):
        _utils.verify_fp8_support_if_applicable(_config("fbgemm_fp8"))


@pytest.mark.parametrize("device_type", ["hip", "mps", "cpu"])
def test_fallback_does_not_widen_other_devices(monkeypatch, device_type):
    monkeypatch.setattr(_utils, "DEVICE_TYPE", device_type)
    monkeypatch.setattr(_utils, "_fp8_checkpoint_dequantizes_below_sm89", lambda: True)
    with pytest.raises(ValueError, match = "only supported on CUDA and XPU"):
        _utils.verify_fp8_support_if_applicable(_config("fp8"))


def test_fallback_detection_follows_the_installed_transformers():
    import inspect

    from transformers.utils.quantization_config import FineGrainedFP8Config

    has_flag = "dequantize" in inspect.signature(FineGrainedFP8Config.__init__).parameters
    assert _utils._fp8_checkpoint_dequantizes_below_sm89() is has_flag


def test_real_check_admits_t4_on_this_transformers(monkeypatch):
    """No stub on the detector: what a T4 user on the installed transformers gets."""
    from transformers.utils.quantization_config import FineGrainedFP8Config
    import inspect

    _spoof_cuda(monkeypatch, (7, 5))
    if "dequantize" in inspect.signature(FineGrainedFP8Config.__init__).parameters:
        _utils.verify_fp8_support_if_applicable(_config("fp8"))
    else:
        with pytest.raises(ValueError, match = "compute capability 8.9"):
            _utils.verify_fp8_support_if_applicable(_config("fp8"))
