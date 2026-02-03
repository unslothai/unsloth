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
import functools
from ...device_type import DEVICE_TYPE
from . import USE_MPS_FALLBACK

# Flag to try torch.compile on fallbacks
# Require torch 2.4+ for better MPS support in inductor
_SUPPORTS_MPS_COMPILE = hasattr(torch, "compile") and torch.__version__ >= "2.4.0"


@functools.lru_cache(maxsize=None)
def _get_compiled_fn(fn):
    """Caches and returns a compiled version of the function if possible."""
    if _SUPPORTS_MPS_COMPILE:
        try:
            # Using inductor with MPS backend
            return torch.compile(fn, backend="inductor")
        except Exception:
            return fn
    return fn


def dispatch_rms_layernorm(X, W, eps, gemma=False):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .rms_layernorm import mps_rms_layernorm

        fn = _get_compiled_fn(mps_rms_layernorm)
        return fn(X, W, eps, gemma)
    else:
        from ..rms_layernorm import Fast_RMS_Layernorm

        return Fast_RMS_Layernorm.apply(X, W, eps, gemma)


def dispatch_layernorm(X, W, b, eps):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .layernorm import mps_layernorm

        fn = _get_compiled_fn(mps_layernorm)
        return fn(X, W, b, eps)
    else:
        from ..layernorm import Fast_Layernorm

        return Fast_Layernorm.apply(X, W, b, eps)


def dispatch_rope_embedding(Q, K, cos, sin, rope_indices=None):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .rope_embedding import mps_rope_embedding_qk

        fn = _get_compiled_fn(mps_rope_embedding_qk)
        return fn(Q, K, cos, sin)
    else:
        from ..rope_embedding import fast_rope_embedding

        return fast_rope_embedding(Q, K, cos, sin, rope_indices)


def dispatch_cross_entropy_loss(logits, labels, logit_softcapping=0, logit_scaling=0):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .cross_entropy_loss import mps_cross_entropy_loss

        fn = _get_compiled_fn(mps_cross_entropy_loss)
        return fn(logits, labels, logit_softcapping, logit_scaling)
    else:
        from ..cross_entropy_loss import fast_cross_entropy_loss

        return fast_cross_entropy_loss(logits, labels, logit_softcapping, logit_scaling)


def dispatch_swiglu_fg(e, g):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .swiglu import mps_swiglu_forward

        fn = _get_compiled_fn(mps_swiglu_forward)
        return fn(e, g)
    else:
        from ..swiglu import swiglu_fg_kernel

        return swiglu_fg_kernel(e, g)


def dispatch_swiglu_backward(dw, e, g):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .swiglu import mps_swiglu_backward

        fn = _get_compiled_fn(mps_swiglu_backward)
        return fn(dw, e, g)
    else:
        from ..swiglu import swiglu_DWf_DW_dfg_kernel

        return swiglu_DWf_DW_dfg_kernel(dw, e, g)


def dispatch_geglu_exact_forward(gate, up):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .geglu import mps_geglu_exact_forward

        fn = _get_compiled_fn(mps_geglu_exact_forward)
        return fn(gate, up)
    else:
        from ..geglu import geglu_exact_forward_kernel

        return geglu_exact_forward_kernel(gate, up)


def dispatch_geglu_exact_backward(dw, e, g):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .geglu import mps_geglu_exact_backward

        fn = _get_compiled_fn(mps_geglu_exact_backward)
        return fn(dw, e, g)
    else:
        from ..geglu import geglu_exact_backward_kernel

        return geglu_exact_backward_kernel(dw, e, g)


def dispatch_geglu_approx_forward(gate, up):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .geglu import mps_geglu_approx_forward

        fn = _get_compiled_fn(mps_geglu_approx_forward)
        return fn(gate, up)
    else:
        from ..geglu import geglu_approx_forward_kernel

        return geglu_approx_forward_kernel(gate, up)


def dispatch_geglu_approx_backward(dw, e, g):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .geglu import mps_geglu_approx_backward

        fn = _get_compiled_fn(mps_geglu_approx_backward)
        return fn(dw, e, g)
    else:
        from ..geglu import geglu_approx_backward_kernel

        return geglu_approx_backward_kernel(dw, e, g)


def dispatch_matmul_lora(X, W, W_quant, A, B, s):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .fast_lora import mps_matmul_lora

        fn = _get_compiled_fn(mps_matmul_lora)
        return fn(X, W, W_quant, A, B, s)
    else:
        from ..utils import matmul_lora

        return matmul_lora(X, W, W_quant, A, B, s)


def dispatch_gemv(X, W, quant_state, out=None):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .linear import mps_gemv

        fn = _get_compiled_fn(mps_gemv)
        return fn(X, W, out=out)
    else:
        from ..utils import fast_gemv

        return fast_gemv(X, W, quant_state, out=out)


def dispatch_lora_mlp_swiglu(
    X,
    gateW,
    gateW_quant,
    gateA,
    gateB,
    gateS,
    upW,
    upW_quant,
    upA,
    upB,
    upS,
    downW,
    downW_quant,
    downA,
    downB,
    downS,
):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .fast_lora import mps_apply_lora_mlp_swiglu

        fn = _get_compiled_fn(mps_apply_lora_mlp_swiglu)
        return fn(
            X,
            gateW,
            gateW_quant,
            gateA,
            gateB,
            gateS,
            upW,
            upW_quant,
            upA,
            upB,
            upS,
            downW,
            downW_quant,
            downA,
            downB,
            downS,
        )
    else:
        from ..fast_lora import LoRA_MLP, swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel

        return LoRA_MLP.apply(
            X,
            gateW,
            gateW_quant,
            gateA,
            gateB,
            gateS,
            upW,
            upW_quant,
            upA,
            upB,
            upS,
            downW,
            downW_quant,
            downA,
            downB,
            downS,
            swiglu_fg_kernel,
            swiglu_DWf_DW_dfg_kernel,
        )


def dispatch_lora_qkv(
    X,
    QW,
    QW_quant,
    QA,
    QB,
    QS,
    KW,
    KW_quant,
    KA,
    KB,
    KS,
    VW,
    VW_quant,
    VA,
    VB,
    VS,
    inplace=True,
):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .fast_lora import mps_apply_lora_qkv

        fn = _get_compiled_fn(mps_apply_lora_qkv)
        return fn(
            X,
            QW,
            QW_quant,
            QA,
            QB,
            QS,
            KW,
            KW_quant,
            KA,
            KB,
            KS,
            VW,
            VW_quant,
            VA,
            VB,
            VS,
        )
    else:
        from ..fast_lora import LoRA_QKV

        return LoRA_QKV.apply(
            X,
            QW,
            QW_quant,
            QA,
            QB,
            QS,
            KW,
            KW_quant,
            KA,
            KB,
            KS,
            VW,
            VW_quant,
            VA,
            VB,
            VS,
            inplace,
        )


def dispatch_lora_o(X, OW, OW_quant, OA, OB, OS):
    if DEVICE_TYPE == "mps" and USE_MPS_FALLBACK:
        from .fast_lora import mps_apply_lora_o

        fn = _get_compiled_fn(mps_apply_lora_o)
        return fn(X, OW, OW_quant, OA, OB, OS)
    else:
        from ..fast_lora import LoRA_W

        return LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)
