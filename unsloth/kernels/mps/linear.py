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


def mps_gemv(
    X: torch.Tensor, W: torch.Tensor, out: torch.Tensor = None
) -> torch.Tensor:
    """
    Optimized GEMV for MPS.
    Uses torch.mv or torch.matmul depending on input shape.
    """
    # X shape: (bsz, seq_len, in_dim) or (seq_len, in_dim)
    # W shape: (out_dim, in_dim)

    if X.dim() == 3:
        # Flatten batch and seq if they are both 1
        if X.shape[0] == 1 and X.shape[1] == 1:
            return torch.matmul(X.view(-1), W.t()).view(1, 1, -1)
        return torch.matmul(X, W.t(), out=out)

    if X.dim() == 2 and X.shape[0] == 1:
        return torch.matmul(X.view(-1), W.t()).view(1, -1)

    return torch.matmul(X, W.t(), out=out)


def mps_linear_forward(
    X: torch.Tensor, W: torch.Tensor, bias: torch.Tensor = None
) -> torch.Tensor:
    """
    Basic linear forward for MPS.
    """
    out = torch.matmul(X, W.t())
    if bias is not None:
        out += bias
    return out
