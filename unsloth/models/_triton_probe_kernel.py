# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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

"""The kernel ``_utils.triton_runs_on_device`` builds on a ROCm card. Imported only by the probe, which
guards the import: a Triton that cannot load just answers False there.

It lives at module level on purpose: Triton before 3.4 resolves a kernel's names through
``fn.__globals__`` only, so a kernel defined in a function with a local ``import triton.language as tl``
fails to compile with ``NameError: tl is not defined`` and the probe would turn FlexAttention off."""

import triton
import triton.language as tl


@triton.jit
def _add_one(X, N, BLOCK: tl.constexpr):  # BLOCK: 16, a fixed probe size
    offsets = tl.arange(0, BLOCK)
    tl.store(X + offsets, tl.load(X + offsets, offsets < N) + 1, offsets < N)
