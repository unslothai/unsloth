# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.

"""bitsandbytes.utils subset for Unsloth's bitsandbytes stand-in."""

import json

import torch


def pack_dict_to_tensor(source_dict):
    """JSON of the non-tensor quant state fields as a uint8 tensor (bitsandbytes' state_dict format)."""
    return torch.tensor(list(json.dumps(source_dict).encode("utf-8")), dtype = torch.uint8)


def unpack_tensor_to_dict(tensor_data):
    return json.loads(bytes(tensor_data.cpu().numpy()).decode("utf-8"))


def sync_gpu(t: torch.Tensor):
    if t.device.type == "cuda":
        torch.cuda.synchronize()
    elif t.device.type == "xpu":
        torch.xpu.synchronize()
