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

from .cross_entropy_loss import (
    fast_cross_entropy_loss,
    post_patch_loss_function,
    patch_loss_functions,
)
from .rms_layernorm import (
    fast_rms_layernorm,
    patch_rms_layernorm,
    unpatch_rms_layernorm,
)
from .layernorm import (
    fast_layernorm,
    patch_layernorm,
)
from .rope_embedding import fast_rope_embedding, inplace_rope_embedding
from .swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
from .geglu import (
    geglu_exact_forward_kernel,
    geglu_exact_backward_kernel,
    geglu_approx_forward_kernel,
    geglu_approx_backward_kernel,
)
from .fast_lora import (
    get_lora_parameters,
    get_lora_parameters_bias,
    apply_lora_mlp_swiglu,
    apply_lora_mlp_geglu_exact,
    apply_lora_mlp_geglu_approx,
    apply_lora_qkv,
    apply_lora_o,
    fast_lora_forward,
)
from .utils import fast_dequantize, fast_gemv, QUANT_STATE, fast_linear_forward, matmul_lora

from .flex_attention import (
    HAS_FLEX_ATTENTION,
    slow_attention_softcapping,
    slow_inference_attention_softcapping,
    create_flex_attention_causal_mask,
    create_flex_attention_sliding_window_mask,
)

import os
if "UNSLOTH_ZOO_IS_PRESENT" not in os.environ:
    try:
        print("🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.")
    except:
        print("Unsloth: Will patch your computer to enable 2x faster free finetuning.")
    pass
pass
del os
