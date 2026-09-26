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

"""CP attention must not train segment 2 on segment 1, or real tokens on pads."""

import torch

from unsloth.distributed.parallel import _blocked_key_mask, sdpa_flash_fn


def main():
    device = "cuda"
    seq = 8
    pos = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], device = device)
    keep = _blocked_key_mask(None, pos, 1, seq, device)
    assert keep is not None
    # Query in the second segment cannot see the first four keys.
    assert not bool(keep[0, 0, 4, 0])
    assert not bool(keep[0, 0, 7, 3])
    assert bool(keep[0, 0, 5, 4])
    assert not bool(keep[0, 0, 5, 6])

    pad = torch.ones(1, seq, dtype = torch.long, device = device)
    pad[0, 6:] = 0
    keep_pad = _blocked_key_mask(pad, None, 1, seq, device)
    assert keep_pad is not None
    assert not bool(keep_pad[0, 0, 3, 6])
    assert bool(keep_pad[0, 0, 3, 1])

    dense = _blocked_key_mask(None, torch.arange(seq, device = device).view(1, -1), 1, seq, device)
    assert dense is None

    torch.manual_seed(0)
    q = torch.randn(1, seq, 2, 8, device = device, dtype = torch.bfloat16, requires_grad = True)
    k = torch.randn(1, seq, 1, 8, device = device, dtype = torch.bfloat16, requires_grad = True)
    v = torch.randn(1, seq, 1, 8, device = device, dtype = torch.bfloat16, requires_grad = True)
    out = sdpa_flash_fn(q, k, v, None, is_causal = True, position_ids = pos)
    out[:, 4:].float().sum().backward()
    leaked = v.grad[:, :4].abs().max().item()
    own = v.grad[:, 4:].abs().max().item()
    print(f"packed leak={leaked:.3e} own={own:.3e}")
    if leaked > 1e-5 or own == 0:
        raise SystemExit(f"packed mask failed leak={leaked} own={own}")
    print("cp mask ok")


if __name__ == "__main__":
    main()
