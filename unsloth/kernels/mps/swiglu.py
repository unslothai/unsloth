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

import torch


def mps_swiglu_forward(e: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """PyTorch-native SwiGLU forward pass for MPS."""
    return torch.nn.functional.silu(e) * g


def mps_swiglu_backward(dw: torch.Tensor, e: torch.Tensor, g: torch.Tensor):
    """PyTorch-native SwiGLU backward pass for MPS, matching Triton kernel behavior."""
    e_f32 = e.to(torch.float32)
    se = torch.sigmoid(e_f32)
    f = (se * e_f32).to(e.dtype)

    # h = f * g
    h = f * g
    # df = dw * f
    df = dw * f
    # dg = dw * g
    dg = dw * g
    # de = dg * se * (1 + e * (1 - se))
    de = (dg.to(torch.float32) * se * (1.0 + e_f32 * (1.0 - se))).to(e.dtype)

    return h, df, de
