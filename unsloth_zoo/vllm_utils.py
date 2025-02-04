# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "patch_vllm",
]
from typing import Optional, List, Tuple, Dict, Any
from transformers.utils.import_utils import _is_package_available

if _is_package_available("vllm"):

    # Allow unsloth dynamic quants to work
    def is_layer_skipped_bnb(prefix: str, llm_int8_skip_modules):
        # Split the prefix into its dot-separated components
        components = prefix.split('.')
        # Check if any of the skip modules exactly matches any component
        vllm_check = any(
            module_name in components
            for module_name in llm_int8_skip_modules
        )

        # Allow certain layers to not be quantized
        components = set(".".join(components[:i+1]) for i in range(len(components)))
        unsloth_check = len(set(llm_int8_skip_modules) & components) != 0
        
        return vllm_check or unsloth_check
    pass

    # Fix force using torch.bfloat16 all the time and make it dynamic
    def _apply_4bit_weight(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # only load the bitsandbytes module when needed
        from bitsandbytes import matmul_4bit

        original_type = x.dtype
        original_shape = x.shape
        reshape_after_matmul = False
        if x.ndim > 2:
            x = x.reshape(-1, x.size(-1))
            reshape_after_matmul = True

        qweight = layer.weight
        quant_states = qweight.bnb_quant_state
        offsets = qweight.bnb_shard_offsets
        inference_dtype = quant_states[0].dtype
        bf_x = x.to(inference_dtype) # Originally used bfloat16

        out_dim_0 = x.shape[0]
        out_dim_1 = sum(
            [quant_state[1].shape[0] for quant_state in quant_states.items()])
        out = torch.empty(out_dim_0,
                            out_dim_1,
                            dtype=inference_dtype,
                            device=x.device)

        current_index = 0
        for i in range(len(quant_states)):
            output_size = quant_states[i].shape[0]
            # It is more efficient to use out kwarg like
            # matmul_4bit(..., out = ...).  Infeasible now due to the bug
            # https://github.com/TimDettmers/bitsandbytes/issues/1235.
            # Need to change  after the bug is fixed.
            out[:, current_index:current_index + output_size] = matmul_4bit(
                bf_x, qweight[offsets[i]:offsets[i + 1]].t(), quant_states[i])

            current_index += output_size

        out = out.to(original_type)

        if reshape_after_matmul:
            out = out.view(*original_shape[:-1], out.size(-1))

        if bias is not None:
            out += bias

        return out
    pass

    def patch_vllm_bitsandbytes():
        import vllm.model_executor.layers.quantization.bitsandbytes
        vllm.model_executor.layers.quantization.bitsandbytes.is_layer_skipped_bnb = is_layer_skipped_bnb
        vllm.model_executor.layers.quantization.bitsandbytes.BitsAndBytesLinearMethod._apply_4bit_weight = _apply_4bit_weight
    pass
else:
    def patch_vllm_bitsandbytes():
        return
    pass
pass


if _is_package_available("bitsandbytes"):
    import bitsandbytes.functional
    from bitsandbytes.utils import pack_dict_to_tensor, unpack_tensor_to_dict

    # Force offsets to be in float32 and not bfloat16 / float16
    @classmethod
    def from_dict(cls, qs_dict: Dict[str, Any], device: torch.device) -> "QuantState":
        """
        unpacks components of state_dict into QuantState
        where necessary, convert into strings, torch.dtype, ints, etc.

        qs_dict: based on state_dict, with only relevant keys, striped of prefixes.

        item with key `quant_state.bitsandbytes__[nf4/fp4]` may contain minor and non-tensor quant state items.
        """

        # unpacking tensor with non-tensor components
        qs_key = [k for k, v in qs_dict.items() if "quant_state" in k and isinstance(v, torch.Tensor)]
        if not len(qs_key) and "quant_type" not in qs_dict:
            raise ValueError("Expected packed or unpacked quant_state items, found neither")
        elif len(qs_key) != 1 or qs_key[0].split(".")[-1] not in cls.valid_qs_type_keys:
            raise ValueError(
                f"There should be exactly one `quant_state` item with ending from {cls.valid_qs_type_keys}.\nDetected {qs_key}.",
            )

        # unpacking minor and non-tensor quant state items if necessary
        if len(qs_key) == 1:
            first_qs_key = qs_key[0]
            qs_dict.update(unpack_tensor_to_dict(qs_dict.pop(first_qs_key)))

        qs_dict = {k.split(".")[-1]: v for k, v in qs_dict.items()}  # strip prefixes
        assert set(qs_dict.keys()).issubset(cls.valid_qs_keys)

        if "nested_absmax" in qs_dict:
            # Must use float32 and disable autocasting - vLLM fails!
            # offset = torch.tensor(float(qs_dict["nested_offset"])).to(device)
            with torch.autocast(device_type = "cuda", enabled = False):
                offset = torch.tensor(qs_dict["nested_offset"], dtype = torch.float32, device = "cuda")
            state2 = cls(
                absmax=qs_dict["nested_absmax"].to(device),
                blocksize=qs_dict["nested_blocksize"],
                code=qs_dict["nested_quant_map"].to(device),
                dtype=getattr(torch, qs_dict["nested_dtype"]),
            )
        else:
            offset, state2 = None, None

        quant_state = cls(
            quant_type=qs_dict["quant_type"],
            absmax=qs_dict["absmax"].to(device),
            blocksize=qs_dict["blocksize"],
            code=qs_dict["quant_map"].to(device),
            dtype=getattr(torch, qs_dict["dtype"]),
            shape=torch.Size(qs_dict["shape"]) if qs_dict["shape"] is not None else None,
            offset=offset,
            state2=state2,
        )
        return quant_state
    pass

    def patch_bitsandbytes_quant_state():
        bitsandbytes.functional.QuantState.from_dict = from_dict
    pass
else:
    def patch_bitsandbytes_quant_state():
        return
    pass
pass


def patch_vllm():
    patch_bitsandbytes_quant_state()
    patch_vllm_bitsandbytes()
pass


# Unsloth Zoo - Utilities for Unsloth
# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
