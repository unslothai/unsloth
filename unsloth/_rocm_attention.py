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

"""Enable PyTorch's ROCm AOTriton SDPA kernels.

Without the environment gate, PyTorch can fall back to quadratic MATH attention. This module
stays stdlib-only so the package can enable the gate before importing torch.
"""

import os

AOTRITON_ENV = "TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"


def enable_rocm_aotriton_attention(env = None) -> bool:
    """Set the gate unless a value already exists, including the "0" opt-out.

    Avoid importing torch for platform detection. Non-ROCm builds ignore the variable.
    """
    target = os.environ if env is None else env
    if AOTRITON_ENV in target:
        return False
    target[AOTRITON_ENV] = "1"
    return True
