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

"""On RDNA1 (gfx101x) Triton's AMD buffer ops silently do nothing.

Every Triton kernel launches with hipSuccess and leaves its outputs untouched, because the
buffer resource descriptor Triton builds is laid out for gfx10.3+ and gfx10.1 reads it
differently. Found on an RX 5700 XT (gfx1010): a ten-line ``x * 2`` kernel returned 0 of
1024 correct values, and every value with ``AMDGCN_USE_BUFFER_OPS=0``. unsloth sets that
variable at import when such a GPU is visible. These are decision-table checks, no GPU.
"""

import pytest

torch = pytest.importorskip("torch")

from unsloth import device_type


@pytest.mark.parametrize(
    "arch, lacks",
    [
        ("gfx1010", True),  # RDNA1, RX 5700 XT: where it was measured
        ("gfx1010:xnack-", True),  # the suffixed form torch reports
        ("gfx1011", True),  # RDNA1, Radeon Pro V520
        ("gfx1012", True),  # RDNA1, RX 5500 XT
        ("gfx1013", True),  # RDNA1, Cyan Skillfish
        ("gfx1030", False),  # RDNA2 reads the newer descriptor, buffer ops are fine
        ("gfx1034", False),  # RDNA2, RX 6500 XT: measured correct with buffer ops on
        ("gfx1034:sramecc-:xnack-", False),
        ("gfx1100", False),  # RDNA3
        ("gfx1201", False),  # RDNA4
        ("gfx90a", False),  # CDNA
        ("gfx942", False),  # CDNA3
        ("", False),  # unreadable: fail open, never guess
        (None, False),
    ],
)
def test_only_gfx101x_loses_buffer_ops(arch, lacks):
    """The prefix has to be exactly ``gfx101``. ``gfx10`` would also switch RDNA2 to global
    loads, which is slower for no reason; ``gfx1`` would reach RDNA3 and RDNA4."""
    assert device_type.arch_lacks_buffer_ops(arch) is lacks


def test_an_unreadable_arch_does_not_narrow_anything():
    for unreadable in ("", None, "unknown", "   "):
        assert device_type.arch_lacks_buffer_ops(unreadable) is False


def test_bf16_gate_and_buffer_ops_gate_disagree_on_rdna2():
    """Both gates exist because the two problems have different scopes: all of gfx10 lacks
    bf16, only gfx10.1 mishandles the descriptor. A refactor that merges them breaks RDNA2."""
    assert device_type.arch_lacks_bf16("gfx1034") is True
    assert device_type.arch_lacks_buffer_ops("gfx1034") is False
    assert device_type.arch_lacks_bf16("gfx1010") is True
    assert device_type.arch_lacks_buffer_ops("gfx1010") is True
