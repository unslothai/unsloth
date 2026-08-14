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

"""Unlock the ROCm AOTriton attention kernels torch ships but hides.

On a ROCm build torch gates its AOTriton flash / mem-efficient SDPA kernels behind
TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL. With the gate closed every sub-quadratic backend
declines, SDPA falls through to MATH, and MATH materialises the whole
batch x heads x tokens x tokens score matrix. Peak VRAM then grows with the SQUARE of the
token count.

That matters more for finetuning than for inference: the scores are needed again for the
backward pass, so they are either held or recomputed per layer. For a 32-head model in fp16
the score tensor alone is 1.07 GB at 4k tokens, 4.3 GB at 8k and 17.2 GB at 16k, before the
softmax output. On a 16 GB card that is the difference between a 4-8k context and a 32k one.

Measured on an RX 9060 XT (gfx1200, torch 2.11.0+rocm7.13.0) at B=1 H=16 N=9408 D=64:

    math            387.2 ms   peak 12.14 GiB
    flash            17.3 ms   peak  0.18 GiB   max|d| vs math = 1.221e-04
    mem_efficient    24.6 ms   peak  0.20 GiB   max|d| vs math = 1.221e-04

1.221e-04 is 2**-13, fp16 epsilon: these agree with the reference to rounding, so the gate
is not buying accuracy. It is only costing the kernel.

Stdlib only, and it MUST run before torch is imported: torch reads the variable once while
loading its C++ extension, so setting it afterwards is dead code. `unsloth/__init__.py` calls
this at the top for exactly that reason, which is also why the check cannot ask torch whether
this is even a ROCm build.

The Studio backend opens the same gate in its own process (unslothai#8323). The two are
deliberately separate copies: this one has to stay importable with nothing but stdlib on the
library path, and that one is vendored into a backend that never imports unsloth.
"""

import os

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def enable_rocm_aotriton_attention(env = None) -> bool:
    """Open the AOTriton gate unless the user already had an opinion.

    Returns whether this call set it. Any pre-existing value wins, including "0": that is
    the opt-out for someone who hits an AOTriton bug and wants the math fallback back.

    Set unconditionally rather than only on ROCm, because knowing the build means importing
    torch, and by then the value has already been read. Non-ROCm torch never looks at a
    TORCH_ROCM_* variable, so the cost of being wrong is one unused entry in the environment.
    """
    target = os.environ if env is None else env
    if AOTRITON_ENV in target:
        return False
    target[AOTRITON_ENV] = "1"
    return True
