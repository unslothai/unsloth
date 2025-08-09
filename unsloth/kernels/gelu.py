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
import torch.nn.functional as F


@torch.compiler.disable
def fast_gelu(x: torch.Tensor, approximate: str | None = None) -> torch.Tensor:
    """Fast GeLU wrapper. Uses torch.nn.functional.gelu with optional approximation.

    approximate: None | "tanh"
    """
    if approximate is None:
        return F.gelu(x)
    return F.gelu(x, approximate=approximate)


class FastGELU(torch.nn.Module):
    def __init__(self, approximate: str | None = None):
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return fast_gelu(x, self.approximate)


