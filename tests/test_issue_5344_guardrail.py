# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for the unslothai/unsloth#5344 silent-quantization-bypass guardrail.

Covers two failure modes the helper detects:
  1. total bypass: load_in_4bit was requested but no bnb modules exist.
  2. partial bypass: bnb quantized nn.Linear but a large fraction of weight
     bytes live in non-nn.Linear Parameters (e.g. Gemma-4 MoE fused experts).
"""

import warnings

import torch
import torch.nn as nn


# unsloth must be imported before transformers per its loading order, but
# these tests do not exercise the real loader. Import the helper directly.
from unsloth.models.vision import _warn_if_quantization_silently_dropped


class _PretendLinear4bit(nn.Module):
    """type(m).__name__ == 'Linear4bit' so the guardrail counts it as quantized."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1, dtype = torch.uint8),
            requires_grad = False,
        )


_PretendLinear4bit.__name__ = "Linear4bit"


def _unquantized_model():
    return nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))


def _quantized_model():
    return nn.Sequential(nn.Linear(4, 4), _PretendLinear4bit())


def test_fires_when_4bit_requested_but_no_bnb_modules():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_4bit=True was requested" in m for m in msgs), msgs
    assert any("issues/5344" in m for m in msgs), msgs


def test_silent_when_4bit_succeeded():
    model = _quantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m for m in msgs), msgs


def test_silent_for_full_finetuning():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = True,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m or "load_in_8bit" in m for m in msgs), msgs


def test_silent_when_no_quantization_requested():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("load_in_4bit" in m or "load_in_8bit" in m for m in msgs), msgs


def test_fires_for_8bit_silent_bypass():
    model = _unquantized_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("load_in_8bit=True was requested" in m for m in msgs), msgs


class _MoEFusedExpertWrapper(nn.Module):
    """Mimics Gemma4TextExperts: fused 3D weights stored as nn.Parameter, not
    as separate nn.Linear instances. bnb's replace_with_bnb_linear skips this."""

    # Gemma-4's real shape is (128, 1408, 2816), which is ~1 GB of BF16 to
    # allocate in a unit test. Only the bulk-weight floor matters here, so this
    # sits just over it at ~16 MB and exercises the same branch.
    def __init__(
        self,
        num_experts = 2,
        intermediate = 1024,
        hidden = 4100,
    ):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros((num_experts, intermediate, hidden), dtype = torch.bfloat16),
            requires_grad = False,
        )


def _partial_quant_model():
    return nn.Sequential(_PretendLinear4bit(), _MoEFusedExpertWrapper())


def test_fires_on_partial_quant_moe_experts():
    model = _partial_quant_model()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert any("gate_up_proj" in m for m in msgs), msgs


class _NormParam(nn.Module):
    """An RMSNorm-like module: large BF16 weight whose name is in the skip list."""

    def __init__(self, dim = 8 * 1024 * 1024 + 10):
        super().__init__()
        self.norm_weight = nn.Parameter(
            torch.zeros(dim, dtype = torch.bfloat16),
            requires_grad = False,
        )


def test_silent_when_only_skip_list_tensors_unquantized():
    model = nn.Sequential(_PretendLinear4bit(), _NormParam())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


class _PretendLinear8bitLt(nn.Module):
    """type(m).__name__ == 'Linear8bitLt', the 8-bit half of the bnb pair."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1, dtype = torch.int8),
            requires_grad = False,
        )


_PretendLinear8bitLt.__name__ = "Linear8bitLt"


def test_fires_when_4bit_request_lands_as_8bit():
    model = nn.Sequential(nn.Linear(4, 4), _PretendLinear8bitLt())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("came back quantized to 8bit" in m for m in msgs), msgs
    # It must NOT claim full precision: the model is quantized, wrongly.
    assert not any("is in full precision" in m for m in msgs), msgs


def test_fires_when_8bit_request_lands_as_4bit():
    model = nn.Sequential(nn.Linear(4, 4), _PretendLinear4bit())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = False,
            load_in_8bit = True,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("came back quantized to 4bit" in m for m in msgs), msgs


class _SkippedOutProj(nn.Module):
    """Falcon-H1 / Nemotron-H leave out_proj unquantized on purpose: the mamba
    kernels cannot consume a 4-bit one (tiiuae/Falcon-H1#13)."""

    def __init__(self, dim = 8 * 1024 * 1024 + 10):
        super().__init__()
        self.out_proj_weight = nn.Parameter(
            torch.zeros(dim, dtype = torch.bfloat16),
            requires_grad = False,
        )


def test_silent_when_unquantized_module_was_configured_as_skipped():
    model = nn.Sequential(_PretendLinear4bit(), _SkippedOutProj())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
            quantization_config = {
                "load_in_4bit": True,
                "llm_int8_skip_modules": ["out_proj"],
            },
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


def test_fires_when_the_skip_list_does_not_cover_it():
    # Negative control: same model, a skip list naming something else.
    model = nn.Sequential(_PretendLinear4bit(), _SkippedOutProj())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
            quantization_config = {
                "load_in_4bit": True,
                "llm_int8_skip_modules": ["some_other_module"],
            },
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs


class _FakeConfig:
    def __init__(self, quantization_config):
        self.quantization_config = quantization_config


class _ModelWithConfig(nn.Sequential):
    def __init__(
        self,
        *mods,
        quantization_config = None,
    ):
        super().__init__(*mods)
        self.config = _FakeConfig(quantization_config)


def test_silent_when_the_checkpoint_is_natively_the_other_width():
    # load_in_4bit defaults True, so a pre-quantized 8-bit repo must stay quiet.
    model = _ModelWithConfig(
        nn.Linear(4, 4),
        _PretendLinear8bitLt(),
        quantization_config = {"load_in_8bit": True},
    )
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert msgs == [], msgs


def test_fires_when_the_checkpoint_declares_the_width_it_did_not_produce():
    # Negative control: config claims 4-bit, modules are 8-bit. A real drop.
    model = _ModelWithConfig(
        nn.Linear(4, 4),
        _PretendLinear8bitLt(),
        quantization_config = {"load_in_4bit": True},
    )
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("came back quantized to 8bit" in m for m in msgs), msgs


class _OffloadedExperts(nn.Module):
    """A fused expert block left in float because device_map put it on CPU.

    With `llm_int8_enable_fp32_cpu_offload=True`, transformers appends every key
    mapped to "cpu" or "disk" to `modules_to_not_convert`
    (quantizers/quantizer_bnb_8bit.py), so these weights are unquantized BY
    DESIGN and consume no accelerator memory.
    """

    def __init__(self, device):
        super().__init__()
        self.gate_up_proj = nn.Parameter(
            torch.zeros((2, 1024, 4100), dtype = torch.bfloat16, device = device),
            requires_grad = False,
        )


class _Quantized(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(1, dtype = torch.uint8, device = device),
            requires_grad = False,
        )


_Quantized.__name__ = "Linear4bit"


def test_silent_when_the_bulk_weight_is_offloaded_off_the_quantized_device():
    # The quantized payload is on meta (standing in for an accelerator); the
    # float experts are on CPU. Warning that VRAM is near full precision would
    # be wrong: those bytes are not on the device the claim is about.
    model = nn.Sequential(_Quantized("meta"), _OffloadedExperts("cpu"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


def test_fires_when_the_bulk_weight_shares_the_quantized_device():
    # Negative control: same shapes and dtypes, one device.
    model = nn.Sequential(_Quantized("cpu"), _OffloadedExperts("cpu"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert not any("sits off those devices" in m for m in msgs), msgs


class _Params4bitBf16Storage(nn.Parameter):
    """A packed 4-bit payload whose storage dtype is bfloat16.

    `bnb_4bit_quant_storage=torch.bfloat16` is what FSDP QLoRA requires, and it
    gives the packed Params4bit a FLOATING dtype. Keying "is this quantized" on
    uint8 both counted this as suspect bulk weight and left quantized_bytes at
    zero, which then suppressed the partial-bypass warning through the
    `quantized_bytes > 0` gate.
    """


_Params4bitBf16Storage.__name__ = "Params4bit"


class _Linear4bitFloatStorage(nn.Module):
    def __init__(self, numel = 8 * 1024 * 1024 + 16):
        super().__init__()
        self.weight = _Params4bitBf16Storage(
            torch.zeros(numel, dtype = torch.bfloat16), requires_grad = False
        )


_Linear4bitFloatStorage.__name__ = "Linear4bit"


def test_float_storage_payload_counts_as_quantized_not_as_a_suspect():
    model = nn.Sequential(_Linear4bitFloatStorage(), _MoEFusedExpertWrapper())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


def test_float_storage_still_reports_a_real_partial_bypass():
    # Negative control: the experts dominate, so the warning must survive.
    model = nn.Sequential(_Linear4bitFloatStorage(numel = 1024), _MoEFusedExpertWrapper())
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert any("gate_up_proj" in m for m in msgs), msgs


class _Conv2dHeavyUNet(nn.Module):
    """A diffusion-shaped model: Linear4bit attention plus large Conv2d weights.

    bnb converts only nn.Linear and Conv1D
    (transformers/integrations/bitsandbytes.py:189), so the convs stay float on a
    perfectly good 4-bit load. Running the LLM-tuned partial-bypass ratio over
    this would warn on every correctly quantized diffusion model.
    """

    def __init__(self):
        super().__init__()
        self.attn_to_q = _PretendLinear4bit()
        self.conv_in = nn.Parameter(
            torch.zeros(8 * 1024 * 1024 + 16, dtype = torch.bfloat16),
            requires_grad = False,
        )


def test_check_partial_false_reports_total_bypass_but_not_the_ratio():
    model = _Conv2dHeavyUNet()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
            check_partial = False,
        )
    msgs = [str(w.message) for w in caught]
    assert not any("partially applied" in m for m in msgs), msgs


def test_check_partial_false_still_catches_a_total_bypass():
    # Negative control: the half that IS naming-agnostic must survive the gate.
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
            check_partial = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("was requested but no bitsandbytes" in m for m in msgs), msgs


def test_the_same_model_does_warn_when_partial_checking_is_on():
    # The gate is a gate: default True still runs the ratio.
    model = _Conv2dHeavyUNet()
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs


def test_float_weight_on_another_accelerator_stays_suspect():
    """Multi-GPU: the payload landed on one device, the fused experts on another.

    Those bytes ARE accelerator-resident full precision, so excusing them as
    "offload" suppresses a real partial bypass. Needs a real accelerator,
    because the whole point is a device that is neither cpu nor meta.
    """
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("needs a real accelerator to be a device that is not cpu/meta")
    model = nn.Sequential(_Quantized("cpu"), _OffloadedExperts("cuda:0"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs
    assert not any("sits off those devices" in m for m in msgs), msgs


def test_cpu_only_load_keeps_its_cpu_floats_suspect():
    # A bnb load whose payload IS on cpu must not have its cpu floats excused:
    # the exclusion is for weights moved OFF the payload's devices, not for cpu
    # as such.
    model = nn.Sequential(_Quantized("cpu"), _OffloadedExperts("cpu"))
    with warnings.catch_warnings(record = True) as caught:
        warnings.simplefilter("always")
        _warn_if_quantization_silently_dropped(
            model,
            load_in_4bit = True,
            load_in_8bit = False,
            full_finetuning = False,
        )
    msgs = [str(w.message) for w in caught]
    assert any("partially applied" in m for m in msgs), msgs


if __name__ == "__main__":
    test_fires_when_4bit_requested_but_no_bnb_modules()
    test_silent_when_4bit_succeeded()
    test_silent_for_full_finetuning()
    test_silent_when_no_quantization_requested()
    test_fires_for_8bit_silent_bypass()
    test_fires_on_partial_quant_moe_experts()
    test_silent_when_only_skip_list_tensors_unquantized()
    test_fires_when_4bit_request_lands_as_8bit()
    test_fires_when_8bit_request_lands_as_4bit()
    test_silent_when_unquantized_module_was_configured_as_skipped()
    test_fires_when_the_skip_list_does_not_cover_it()
    test_silent_when_the_checkpoint_is_natively_the_other_width()
    test_fires_when_the_checkpoint_declares_the_width_it_did_not_produce()
    test_silent_when_the_bulk_weight_is_offloaded_off_the_quantized_device()
    test_fires_when_the_bulk_weight_shares_the_quantized_device()
    test_float_storage_payload_counts_as_quantized_not_as_a_suspect()
    test_float_storage_still_reports_a_real_partial_bypass()
    test_check_partial_false_reports_total_bypass_but_not_the_ratio()
    test_check_partial_false_still_catches_a_total_bypass()
    test_the_same_model_does_warn_when_partial_checking_is_on()
    test_float_weight_on_another_accelerator_stays_suspect()
    test_cpu_only_load_keeps_its_cpu_floats_suspect()
    print("All 22 guardrail tests passed.")
