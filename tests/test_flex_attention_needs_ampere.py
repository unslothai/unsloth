# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flex attention must not be chosen on a card that cannot run its kernel.

`Gemma3_(4B)-Vision-GRPO` passes on A100 and dies on both a Colab T4 and a
Kaggle T4 with `RuntimeError: expected scalar type Half but found Float`, raised
by torch's own eager fallback:

    torch/_higher_order_ops/flex_attention.py, sdpa_dense_backward
    grad_value = softmax_scores.to(query.dtype).transpose(-2, -1) @ grad_out

That line casts the scores to the query dtype and leaves `grad_out` alone. It is
only reached when the HOP runs uncompiled, which is what happens on sm75, and a
T4 also forces fp16, so query is Half against a Float grad.

`gemma3` is in `_FLEX_PREFERRED_MODELS` and its sdpa is disabled, so flex is
exactly the path it took there, while the only availability question asked was
`is_torch_flex_attn_available()` -- a torch-version check with nothing to say
about the card. Verified by running the notebook on a Colab T4 with flex off:
PASS in 1007s, against the failure at 1180s with it on.
"""

import sys
import types
from unittest import mock

import pytest

import unsloth.models._utils as U


class _Model:
    _supports_flex_attn = True


def _supports(model_type = "gemma3"):
    return U._supports_flex_attention(_Model, {}, model_type)


def _cuda(capabilities, hip = None):
    """Patch just enough of torch for the vendor/capability probe."""
    return mock.patch.multiple(
        U.torch.cuda,
        is_available = lambda: bool(capabilities),
        device_count = lambda: len(capabilities),
        get_device_capability = lambda index = 0: capabilities[index],
    ), mock.patch.object(U.torch.version, "hip", hip, create = True)


@pytest.mark.parametrize("capability", [(7, 0), (7, 5)])
def test_a_pre_ampere_card_does_not_get_flex(capability):
    """(7, 5) is the T4 this was measured on; (7, 0) is V100, same fallback."""
    cuda, hip = _cuda([capability])
    with cuda, hip:
        assert U._flex_attention_gpu_is_supported() is False
        assert _supports() is False


@pytest.mark.parametrize("capability", [(8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (12, 0)])
def test_ampere_and_newer_are_untouched(capability):
    """A100, A10, L4, H100, B200, RTX 50xx. The notebook passes on A100 with
    flex on, so the fix must not take it away from them."""
    cuda, hip = _cuda([capability])
    with cuda, hip:
        assert U._flex_attention_gpu_is_supported() is True


def test_a_mixed_box_follows_its_weakest_card():
    """One process, one attn_implementation. A T4 beside an A100 still cannot
    run the kernel, so the pair has to fall back together."""
    cuda, hip = _cuda([(8, 0), (7, 5)])
    with cuda, hip:
        assert U._flex_attention_gpu_is_supported() is False


def test_rocm_is_not_judged_by_a_cuda_capability():
    """`get_device_capability` answers on ROCm too, and its numbers are not
    CUDA's. Reading them would disable flex on AMD cards for no reason."""
    cuda, hip = _cuda([(7, 5)], hip = "6.2.0")
    with cuda, hip:
        assert U._flex_attention_gpu_is_supported() is True


def test_no_cuda_device_is_left_alone():
    """CPU, MPS and XPU boxes keep whatever they had."""
    cuda, hip = _cuda([])
    with cuda, hip:
        assert U._flex_attention_gpu_is_supported() is True


def test_an_unreadable_device_fails_open():
    """A probe that raises must not silently switch every user's attention
    backend. Same stance as the `is_torch_flex_attn_available` guard below it."""

    def _boom(index = 0):
        raise RuntimeError("no CUDA driver")

    with mock.patch.multiple(
        U.torch.cuda, is_available = lambda: True, device_count = lambda: 1, get_device_capability = _boom
    ):
        assert U._flex_attention_gpu_is_supported() is True


def test_the_gate_runs_before_the_torch_version_check():
    """`is_torch_flex_attn_available` is a torch-version answer and says yes on
    a T4. If it were consulted first the card check could never refuse."""
    stub = types.ModuleType("transformers.utils.import_utils")
    stub.is_torch_flex_attn_available = lambda: True
    cuda, hip = _cuda([(7, 5)])
    with cuda, hip, mock.patch.dict(sys.modules, {"transformers.utils.import_utils": stub}):
        assert _supports() is False
