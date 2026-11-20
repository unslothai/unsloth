# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""MXFP4-packed routed experts of the native DeepSeek-V4.1 (the release's fp4 experts kept packed by convert.py).
Kept out of modeling.py: unsloth_zoo's compiler wraps that module's free functions in torch.compile, and these
run at load / save time only."""

import json
import os

from transformers.modeling_utils import PreTrainedModel

__all__ = [
    "converted_experts_format",
    "zoo_saves_mxfp4_dense",
    "packed_experts_state_dict_hook",
    "packed_experts_load_pre_hook",
]


def converted_experts_format(path):
    try:
        with open(os.path.join(str(path), "config.json"), encoding = "utf-8") as f:
            return json.load(f).get("unsloth_experts_format")
    except Exception:
        return None


def zoo_saves_mxfp4_dense():
    """True if unsloth_zoo's MXFP4 dequantize-on-save patch wraps save_pretrained."""
    fn, seen = PreTrainedModel.save_pretrained, set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        if getattr(fn, "_unsloth_mxfp4_patched", False):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


def packed_experts_state_dict_hook(module, state_dict, prefix, local_metadata):
    # A packed stack saves as its blocks + scales (what convert.py writes), so save / reload round-trips.
    for name in ("gate_up", "down"):
        param = module._parameters.get(f"{name}_proj")
        if getattr(param, "mxfp4_scales", None) is None:
            continue
        state_dict.pop(prefix + f"{name}_proj", None)
        state_dict[prefix + f"{name}_blocks"] = param.data
        state_dict[prefix + f"{name}_scales"] = param.mxfp4_scales
    return state_dict


def packed_experts_load_pre_hook(module, state_dict, prefix, *args):
    # load_state_dict into a finalized stack: blocks + scales become one Mxfp4ExpertParam source.
    for name in ("gate_up", "down"):
        param = module._parameters.get(f"{name}_proj")
        blocks, scales = prefix + f"{name}_blocks", prefix + f"{name}_scales"
        if (
            getattr(param, "mxfp4_scales", None) is None
            or blocks not in state_dict
            or scales not in state_dict
        ):
            continue
        from unsloth_zoo.mxfp4_dequant import Mxfp4ExpertParam

        state_dict[prefix + f"{name}_proj"] = Mxfp4ExpertParam(
            state_dict.pop(blocks),
            mxfp4_scales = state_dict.pop(scales),
            mxfp4_transposed = False,
            mxfp4_dtype = param.mxfp4_dtype,
        )
