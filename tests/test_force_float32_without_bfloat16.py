# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A FORCE_FLOAT32 family must not be loaded in bfloat16 on a device without bfloat16.

The selection branch for these architectures is entered when the caller asked for float16
OR the device has no bfloat16 -- and it then chose bfloat16 unconditionally. On the second
of those two conditions that picks the one dtype the device cannot run:

  * RDNA 1/2 (gfx101x, gfx103x): Triton cannot lower bf16, so a bf16 tensor reaching a
    Triton kernel aborts the process inside LLVM with no Python exception at all,
    "Cannot select: intrinsic %llvm.amdgcn.fdot2.bf16.bf16" (unslothai/unsloth#7922).
  * Pre-Ampere NVIDIA (T4, V100): "expected mat1 and mat2 to have the same dtype, but
    got: BFloat16 != Half" (unslothai/unsloth#7506).

bfloat16 stays the choice wherever it is usable -- it is what reduces the outliers that
make float16 unusable for these families in the first place. float32 is the fallback.

These tests are pure decision-table checks and need no GPU.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")

from unsloth.models._utils import force_float32_dtype
from unsloth import device_type


def test_bfloat16_is_still_chosen_wherever_it_works():
    """The whole point of the FORCE_FLOAT32 list is that bf16 reduces outliers. A change
    that returned float32 everywhere would 'fix' RDNA2 by regressing every other GPU."""
    assert force_float32_dtype(True) is torch.bfloat16


def test_float32_replaces_bfloat16_when_the_device_has_none():
    assert force_float32_dtype(False) is torch.float32


def test_float16_is_never_the_answer():
    """float16 is what the FORCE_FLOAT32 list exists to avoid: these architectures have
    activations outside its finite range and silently NaN at training time."""
    for supports in (True, False):
        assert force_float32_dtype(supports) is not torch.float16


@pytest.mark.parametrize(
    "arch, lacks_bf16",
    [
        ("gfx1010", True),    # RDNA1
        ("gfx1030", True),    # RDNA2, RX 6800
        ("gfx1034", True),    # RDNA2, RX 6500 XT -- the card in unslothai/unsloth#7922
        ("gfx1034:sramecc-:xnack-", True),   # the suffixed form torch reports
        ("gfx1100", False),   # RDNA3 has bf16
        ("gfx1151", False),   # RDNA3.5, Strix Halo
        ("gfx1201", False),   # RDNA4
        ("gfx90a", False),    # CDNA, MI210
        ("gfx942", False),    # CDNA3, MI300
        ("", False),          # unreadable: must fail OPEN, never assume gfx10
        (None, False),
    ],
)
def test_only_gfx10_is_treated_as_lacking_bfloat16(arch, lacks_bf16):
    """The prefix has to stay 5 characters. `startswith("gfx1")` would swallow RDNA3,
    RDNA3.5 and RDNA4, which all have bf16, and drop every one of them to float32."""
    assert device_type.arch_lacks_bf16(arch) is lacks_bf16


def test_an_unreadable_arch_does_not_narrow_anything():
    """Detection must fail open. An arch that cannot be read is not evidence of gfx10,
    and answering True there would push working RDNA3/CDNA/NVIDIA users onto float32."""
    for unreadable in ("", None, "unknown", "   "):
        assert device_type.arch_lacks_bf16(unreadable) is False


def test_both_call_sites_use_the_shared_helper():
    """Two copies of this rule drifted apart once already: loader.py promoted to bfloat16
    and vision.py re-promoted float16 to bfloat16 in an `elif` that skipped the
    bfloat16-unsupported downgrade underneath it. One helper, two callers."""
    from unsloth.models import loader, vision

    for module in (loader, vision):
        source = inspect.getsource(module)
        assert "force_float32_dtype(SUPPORTS_BFLOAT16)" in source, (
            f"{module.__name__} chooses the FORCE_FLOAT32 dtype itself instead of "
            f"calling the shared helper"
        )


def test_vision_force_float32_branch_no_longer_skips_the_downgrade():
    """That branch is an `elif`, so it bypasses `elif dtype == bfloat16 and not
    SUPPORTS_BFLOAT16` below it. It must therefore handle an incoming bfloat16 itself,
    or a caller passing dtype=torch.bfloat16 on a gfx10 card keeps it."""
    from unsloth.models import vision

    source = inspect.getsource(vision)
    assert "dtype == torch.float16 or (dtype == torch.bfloat16 and not SUPPORTS_BFLOAT16)" in source
