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
import math


def mps_geglu_exact_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch-native GEGLU (Exact) forward pass for MPS."""
    return torch.nn.functional.gelu(gate, approximate = "none") * up


def mps_geglu_exact_backward(dw: torch.Tensor, e: torch.Tensor, g: torch.Tensor):
    """PyTorch-native GEGLU (Exact) backward pass for MPS, matching Triton kernel behavior."""
    e_f32 = e.to(torch.float32)
    # f = 0.5 * e * (1 + erf(e / sqrt(2)))
    f_partial = 0.5 * (torch.erf(e_f32 / math.sqrt(2.0)) + 1.0)
    f = (f_partial * e_f32).to(e.dtype)

    h = f * g
    df = dw * f
    dg = dw * g

    # df/de = 1/2 * (1 + erf(1/sqrt(2) * e)) + 1/sqrt(2*pi) * e * exp(-1/2 * e^2)
    t = 0.3989422804014327  # 1/sqrt(2*pi)
    df_de = f_partial + t * e_f32 * torch.exp(-0.5 * e_f32 * e_f32)

    de = (dg.to(torch.float32) * df_de).to(e.dtype)
    return h, df, de


def mps_geglu_approx_forward(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """PyTorch-native GEGLU (Approximate) forward pass for MPS."""
    return torch.nn.functional.gelu(gate, approximate = "tanh") * up


def mps_geglu_approx_backward(dw: torch.Tensor, e: torch.Tensor, g: torch.Tensor):
    """PyTorch-native GEGLU (Approximate) backward pass for MPS, matching Triton kernel behavior."""
    e_f32 = e.to(torch.float32)
    s = 0.7978845608028654  # math.sqrt(2 / math.pi)

    # T = 1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))
    # T2 = 0.5 * T
    # inner = s * e * (1 + 0.044715 * e^2)
    inner = s * e_f32 * (1.0 + 0.044715 * e_f32 * e_f32)
    tanh_inner = torch.tanh(inner)
    T = 1.0 + tanh_inner
    T2 = 0.5 * T

    f = (T2 * e_f32).to(e.dtype)
    h = f * g
    df = dw * f
    dg = dw * g

    # df/de = T2 + 0.5 * T * (2 - T) * (s * e * (1 + 3 * 0.044715 * e^2))
    # Matches Triton's: Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    a = s * e_f32
    b = a * 0.044715 * e_f32 * e_f32
    Q2 = -T2 * (T - 2.0) * (a + 3.0 * b)
    df_de = T2 + Q2

    de = (dg.to(torch.float32) * df_de).to(e.dtype)
    return h, df, de
