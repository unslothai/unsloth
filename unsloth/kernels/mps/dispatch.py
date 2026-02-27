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


def _use_mps_fallback():
    """Dynamically read USE_MPS_FALLBACK from the parent module.

    This MUST be a function call (not a cached import) because the flag is
    mutated at runtime by llama.py / loader.py when gradient checkpointing
    is enabled.  A bare ``from . import USE_MPS_FALLBACK`` would snapshot
    the immutable bool at import time and never see later mutations.
    """
    import unsloth.kernels.mps as _mps_module
    return getattr(_mps_module, "USE_MPS_FALLBACK", True)

# Flag to try torch.compile on fallbacks
# Require torch 2.4+ for better MPS support in inductor
_SUPPORTS_MPS_COMPILE = hasattr(torch, "compile") and torch.__version__ >= "2.4.0"

# One-time diagnostic logging flag
_LOGGED_FALLBACK_STATE = False

# =============================================================================
# Metal / MLX Kernel Priority System
# =============================================================================
# Priority order for Apple Silicon:
#   1. Metal kernels (highest performance, custom fused ops)
#   2. MLX fast ops (mx.fast.* operations)
#   3. MPS fallback (PyTorch-native with torch.compile)
#   4. CUDA/Triton (default path for NVIDIA GPUs)
# =============================================================================

# Lazy-loaded availability flags for Metal/MLX
_METAL_AVAILABLE = None
_MLX_AVAILABLE = None


def _is_metal_available():
    """Check if Metal kernels are available (cached)."""
    global _METAL_AVAILABLE
    if _METAL_AVAILABLE is None:
        try:
            from ..metal import is_metal_available
            _METAL_AVAILABLE = is_metal_available()
        except ImportError:
            _METAL_AVAILABLE = False
    return _METAL_AVAILABLE


def _is_mlx_available():
    """Check if MLX is available (cached)."""
    global _MLX_AVAILABLE
    if _MLX_AVAILABLE is None:
        try:
            from ..mlx import is_mlx_available
            _MLX_AVAILABLE = is_mlx_available()
        except ImportError:
            _MLX_AVAILABLE = False
    return _MLX_AVAILABLE


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
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX RMSNorm (mx.fast.rms_norm - fastest for both train/inference)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_rms_norm_autograd
            return mlx_rms_norm_autograd(X, W, eps, gemma)
        # Priority 2: Metal kernel (fused, highest performance)
        if _is_metal_available():
            from ..metal.rms_layernorm import Metal_RMSLayerNorm
            return Metal_RMSLayerNorm.apply(X, W, eps, gemma)
        # Priority 3: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .rms_layernorm import mps_rms_layernorm
            fn = _get_compiled_fn(mps_rms_layernorm)
            return fn(X, W, eps, gemma)
    # Default: CUDA/Triton path
    from ..rms_layernorm import Fast_RMS_Layernorm
    return Fast_RMS_Layernorm.apply(X, W, eps, gemma)


def dispatch_layernorm(X, W, b, eps):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX LayerNorm (mx.fast.layer_norm - fastest for both train/inference)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_layer_norm_autograd
            return mlx_layer_norm_autograd(X, W, b, eps)
        # Priority 2: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .layernorm import mps_layernorm
            fn = _get_compiled_fn(mps_layernorm)
            return fn(X, W, b, eps)
    # Default: CUDA/Triton path
    from ..layernorm import Fast_Layernorm
    return Fast_Layernorm.apply(X, W, b, eps)


def dispatch_rope_embedding(Q, K, cos, sin, rope_indices=None):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX RoPE (uses mx.fast.rope or optimized MLX arithmetic)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_rope_autograd
            return mlx_rope_autograd(Q, K, cos, sin)
        # Priority 2: MPS fallback (PyTorch-native with custom autograd)
        if _use_mps_fallback():
            from .rope_embedding import mps_rope_embedding_qk
            fn = _get_compiled_fn(mps_rope_embedding_qk)
            return fn(Q, K, cos, sin)
        # Priority 3: Pure PyTorch (when fallback disabled, e.g., gradient checkpointing)
        from .lora_pytorch import pytorch_rope_embedding_qk
        return pytorch_rope_embedding_qk(Q, K, cos, sin)
    # Default: CUDA/Triton path
    from ..rope_embedding import fast_rope_embedding
    return fast_rope_embedding(Q, K, cos, sin, rope_indices)


def dispatch_cross_entropy_loss(
    logits, 
    labels, 
    logit_softcapping=0, 
    logit_scaling=0, 
    n_items=None,
    hidden_states=None,
    lm_head_weight=None,
):
    if DEVICE_TYPE == "mps":
        # Check if CCE should be used
        use_cce = os.environ.get("UNSLOTH_USE_FAST_CROSS_ENTROPY", "0") == "1"
        
        # When USE_MPS_FALLBACK is disabled (gradient checkpointing active),
        # use standard PyTorch cross-entropy to avoid custom autograd issues
        if not _use_mps_fallback():
            import torch.nn.functional as F
            # ... (omitted for brevity)
            if logits is None:
                # If logits is not provided but hidden_states is, we must compute logits
                logits = hidden_states @ lm_head_weight.T
            
            batch, seq_len, vocab_size = logits.shape
            logits_flat = logits.reshape(-1, vocab_size)
            labels_flat = labels.reshape(-1)
            # Use standard F.cross_entropy which works with gradient checkpointing
            loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction='none')
            if n_items is None:
                n_items = torch.count_nonzero(labels != -100)
            if n_items == 0:
                return loss.sum() * 0.0 + (logits.sum() * 0.0 if logits is not None else 0.0)
            return loss.sum() / n_items

        if use_cce and _is_mlx_available() and hidden_states is not None and lm_head_weight is not None:
            # Use MLX Chunked Cross Entropy (CCE) via bridge
            from ..mlx.fast_ops import mlx_cce_loss_autograd
            return mlx_cce_loss_autograd(
                hidden_states, 
                lm_head_weight, 
                labels, 
                logit_softcapping, 
                logit_scaling, 
                n_items
            )

        from .cross_entropy_loss import mps_cross_entropy_loss
        fn = _get_compiled_fn(mps_cross_entropy_loss)
        return fn(logits, labels, logit_softcapping, logit_scaling, n_items)
    else:
        from ..cross_entropy_loss import fast_cross_entropy_loss
        return fast_cross_entropy_loss(logits, labels, logit_softcapping, logit_scaling, n_items)


def dispatch_swiglu_fg(e, g):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX SwiGLU (fastest for both train/inference)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_swiglu_autograd
            return mlx_swiglu_autograd(e, g)
        # Priority 2: Metal kernel (fused SwiGLU)
        if _is_metal_available():
            from ..metal.swiglu import metal_swiglu_forward
            return metal_swiglu_forward(e, g)
        # Priority 3: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .swiglu import mps_swiglu_forward
            fn = _get_compiled_fn(mps_swiglu_forward)
            return fn(e, g)
        # Priority 4: Pure PyTorch (when fallback disabled, e.g. gradient checkpointing)
        # SwiGLU: f = silu(e) * g = (e * sigmoid(e)) * g
        return torch.nn.functional.silu(e) * g
    # Default: CUDA/Triton path
    from ..swiglu import swiglu_fg_kernel
    return swiglu_fg_kernel(e, g)


def dispatch_swiglu_backward(dw, e, g):
    if DEVICE_TYPE == "mps":
        # Priority 1: Metal kernel (fused SwiGLU backward)
        if _is_metal_available():
            from ..metal.swiglu import metal_swiglu_backward
            return metal_swiglu_backward(dw, e, g)
        # Priority 2: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .swiglu import mps_swiglu_backward
            fn = _get_compiled_fn(mps_swiglu_backward)
            return fn(dw, e, g)
        # Priority 3: Pure PyTorch backward (when fallback disabled)
        # SwiGLU backward: h = silu(e) * g
        # df/de = sigmoid(e) * (1 + e * (1 - sigmoid(e)))
        # dh/de = df/de * g, dh/dg = silu(e)
        se = torch.sigmoid(e)
        f = e * se
        df = se * (1.0 + e * (1.0 - se))
        # dw is dh (upstream gradient already multiplied)
        # Return: (dw * g * df, dw * f)  but in the convention of the kernel:
        # h = f * g, so DW (from downstream) comes in as dw
        # de = dw * df * g, dg = dw * f
        return dw * df * g, dw * f, dw * g * df
    # Default: CUDA/Triton path
    from ..swiglu import swiglu_DWf_DW_dfg_kernel
    return swiglu_DWf_DW_dfg_kernel(dw, e, g)


def dispatch_geglu_exact_forward(gate, up):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX GeGLU exact (fastest for both train/inference)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_geglu_exact_autograd
            return mlx_geglu_exact_autograd(gate, up)
        # Priority 2: Metal kernel (fused GEGLU exact)
        if _is_metal_available():
            from ..metal.geglu import metal_geglu_exact_forward
            return metal_geglu_exact_forward(gate, up)
        # Priority 3: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .geglu import mps_geglu_exact_forward
            fn = _get_compiled_fn(mps_geglu_exact_forward)
            return fn(gate, up)
        # Priority 4: Pure PyTorch (when fallback disabled)
        # GeGLU exact: h = gelu(gate) * up
        return torch.nn.functional.gelu(gate) * up
    # Default: CUDA/Triton path
    from ..geglu import geglu_exact_forward_kernel
    return geglu_exact_forward_kernel(gate, up)


def dispatch_geglu_exact_backward(dw, e, g):
    if DEVICE_TYPE == "mps":
        # Priority 1: Metal kernel (fused GEGLU exact backward)
        if _is_metal_available():
            from ..metal.geglu import metal_geglu_exact_backward
            return metal_geglu_exact_backward(dw, e, g)
        # Priority 2: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .geglu import mps_geglu_exact_backward
            fn = _get_compiled_fn(mps_geglu_exact_backward)
            return fn(dw, e, g)
        # Priority 3: Pure PyTorch backward (when fallback disabled)
        # GeGLU exact backward
        import math
        f = torch.nn.functional.gelu(e)
        # GELU'(x) derivative via standard formula
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        cdf = 0.5 * (1.0 + torch.erf(e / math.sqrt(2.0)))
        pdf = sqrt_2_over_pi * torch.exp(-0.5 * e * e)
        df = cdf + e * pdf
        return dw * df * g, dw * f, dw * g * df
    # Default: CUDA/Triton path
    from ..geglu import geglu_exact_backward_kernel
    return geglu_exact_backward_kernel(dw, e, g)


def dispatch_geglu_approx_forward(gate, up):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX GeGLU approx (fastest for both train/inference)
        if _is_mlx_available():
            from ..mlx.fast_ops import mlx_geglu_approx_autograd
            return mlx_geglu_approx_autograd(gate, up)
        # Priority 2: Metal kernel (fused GEGLU approx)
        if _is_metal_available():
            from ..metal.geglu import metal_geglu_approx_forward
            return metal_geglu_approx_forward(gate, up)
        # Priority 3: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .geglu import mps_geglu_approx_forward
            fn = _get_compiled_fn(mps_geglu_approx_forward)
            return fn(gate, up)
        # Priority 4: Pure PyTorch (when fallback disabled)
        # GeGLU approx: h = gelu_approx(gate) * up
        return torch.nn.functional.gelu(gate, approximate='tanh') * up
    # Default: CUDA/Triton path
    from ..geglu import geglu_approx_forward_kernel
    return geglu_approx_forward_kernel(gate, up)


def dispatch_geglu_approx_backward(dw, e, g):
    if DEVICE_TYPE == "mps":
        # Priority 1: Metal kernel (fused GEGLU approx backward)
        if _is_metal_available():
            from ..metal.geglu import metal_geglu_approx_backward
            return metal_geglu_approx_backward(dw, e, g)
        # Priority 2: MPS fallback (PyTorch-native)
        if _use_mps_fallback():
            from .geglu import mps_geglu_approx_backward
            fn = _get_compiled_fn(mps_geglu_approx_backward)
            return fn(dw, e, g)
        # Priority 3: Pure PyTorch backward (when fallback disabled)
        # GeGLU approx backward using tanh approximation
        import math
        sqrt_2_over_pi = math.sqrt(2.0 / math.pi)
        inner = sqrt_2_over_pi * (e + 0.044715 * e * e * e)
        tanh_val = torch.tanh(inner)
        f = 0.5 * e * (1.0 + tanh_val)
        # Derivative of tanh-approx GELU
        dtanh = 1.0 - tanh_val * tanh_val
        dinner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * e * e)
        df = 0.5 * (1.0 + tanh_val) + 0.5 * e * dtanh * dinner
        return dw * df * g, dw * f, dw * g * df
    # Default: CUDA/Triton path
    from ..geglu import geglu_approx_backward_kernel
    return geglu_approx_backward_kernel(dw, e, g)


def dispatch_matmul_lora(X, W, W_quant, A, B, s):
    if DEVICE_TYPE == "mps" and _use_mps_fallback():
        from .fast_lora import mps_matmul_lora

        fn = _get_compiled_fn(mps_matmul_lora)
        return fn(X, W, W_quant, A, B, s)
    else:
        from ..utils import matmul_lora

        return matmul_lora(X, W, W_quant, A, B, s)


def dispatch_gemv(X, W, quant_state, out=None):
    if DEVICE_TYPE == "mps" and _use_mps_fallback():
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
    gate_multiplier=None,
    down_multiplier=None,
):
    if DEVICE_TYPE == "mps":
        # One-time diagnostic logging of USE_MPS_FALLBACK state
        global _LOGGED_FALLBACK_STATE
        if not _LOGGED_FALLBACK_STATE:
            _LOGGED_FALLBACK_STATE = True
            _fallback_val = _use_mps_fallback()
            import logging
            _logger = logging.getLogger("unsloth")
            _logger.warning_once(
                f"Unsloth MPS dispatch: USE_MPS_FALLBACK={_fallback_val}, "
                f"grad_enabled={torch.is_grad_enabled()}"
            ) if hasattr(_logger, 'warning_once') else _logger.warning(
                f"Unsloth MPS dispatch: USE_MPS_FALLBACK={_fallback_val}, "
                f"grad_enabled={torch.is_grad_enabled()}"
            )

        # Priority 1: MLX fast_lora (compiled graph fusion + batch-1 GEMV)
        # Only use MLX path for inference — MLX ops are outside PyTorch autograd
        if _is_mlx_available() and not torch.is_grad_enabled():
            from ..mlx.fast_lora import apply_lora_mlp_swiglu
            return apply_lora_mlp_swiglu(
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
                gate_multiplier=gate_multiplier,
                down_multiplier=down_multiplier,
            )
        # Priority 2: MPS fallback with custom autograd for memory efficiency
        if _use_mps_fallback():
            from .fast_lora import MPSLoRA_MLP
            from ..swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
            return MPSLoRA_MLP.apply(
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
    # Priority 3: Pure PyTorch implementation for MPS (when fallback is disabled)
    # This avoids custom autograd functions which have issues with gradient checkpointing
    if DEVICE_TYPE == "mps":
        from .lora_pytorch import pytorch_lora_mlp_swiglu
        return pytorch_lora_mlp_swiglu(
            X,
            gateW, gateW_quant, gateA, gateB, gateS,
            upW, upW_quant, upA, upB, upS,
            downW, downW_quant, downA, downB, downS,
        )
    
    # Default: CUDA/Triton path
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


def dispatch_lora_mlp_geglu_exact(
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
    gate_multiplier=None,
    down_multiplier=None,
):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX fast_lora (compiled graph fusion + batch-1 GEMV)
        # Only use MLX path for inference — MLX ops are outside PyTorch autograd
        if _is_mlx_available() and not torch.is_grad_enabled():
            from ..mlx.fast_lora import apply_lora_mlp_geglu
            return apply_lora_mlp_geglu(
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
                gate_multiplier=gate_multiplier,
                down_multiplier=down_multiplier,
            )
        # Priority 2: MPS fallback with custom autograd for memory efficiency
        if _use_mps_fallback():
            from .fast_lora import MPSLoRA_MLP
            from ..geglu import geglu_exact_forward_kernel, geglu_exact_backward_kernel
            return MPSLoRA_MLP.apply(
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
                geglu_exact_forward_kernel,
                geglu_exact_backward_kernel,
            )
    # Priority 3: Pure PyTorch implementation for MPS (when fallback is disabled)
    # This avoids custom autograd functions which have issues with gradient checkpointing
    if DEVICE_TYPE == "mps":
        from .lora_pytorch import pytorch_lora_mlp_geglu_exact
        return pytorch_lora_mlp_geglu_exact(
            X,
            gateW, gateW_quant, gateA, gateB, gateS,
            upW, upW_quant, upA, upB, upS,
            downW, downW_quant, downA, downB, downS,
        )
    
    # Default: CUDA/Triton path
    from ..fast_lora import LoRA_MLP, geglu_exact_forward_kernel, geglu_exact_backward_kernel

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
        geglu_exact_forward_kernel,
        geglu_exact_backward_kernel,
    )


def dispatch_lora_mlp_geglu_approx(
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
    gate_multiplier=None,
    down_multiplier=None,
):
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX fast_lora (compiled graph fusion + batch-1 GEMV)
        # Only use MLX path for inference — MLX ops are outside PyTorch autograd
        if _is_mlx_available() and not torch.is_grad_enabled():
            from ..mlx.fast_lora import apply_lora_mlp_geglu

            return apply_lora_mlp_geglu(
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
                gate_multiplier=gate_multiplier,
                down_multiplier=down_multiplier,
            )
        # Priority 2: MPS fallback with custom autograd for memory efficiency
        if _use_mps_fallback():
            from .fast_lora import MPSLoRA_MLP
            from ..geglu import geglu_approx_forward_kernel, geglu_approx_backward_kernel
            return MPSLoRA_MLP.apply(
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
                geglu_approx_forward_kernel,
                geglu_approx_backward_kernel,
            )
    # Priority 3: Pure PyTorch implementation for MPS (when fallback is disabled)
    # This avoids custom autograd functions which have issues with gradient checkpointing
    if DEVICE_TYPE == "mps":
        from .lora_pytorch import pytorch_lora_mlp_geglu_approx
        return pytorch_lora_mlp_geglu_approx(
            X,
            gateW, gateW_quant, gateA, gateB, gateS,
            upW, upW_quant, upA, upB, upS,
            downW, downW_quant, downA, downB, downS,
        )
    
    # Default: CUDA/Triton path
    from ..fast_lora import (
        LoRA_MLP,
        geglu_approx_forward_kernel,
        geglu_approx_backward_kernel,
    )

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
        geglu_approx_forward_kernel,
        geglu_approx_backward_kernel,
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
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX fast_lora (compiled graph fusion + batch-1 GEMV)
        # Only use MLX path for inference — MLX ops are outside PyTorch autograd
        if _is_mlx_available() and not torch.is_grad_enabled():
            from ..mlx.fast_lora import apply_lora_qkv
            return apply_lora_qkv(
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
        # Priority 2: MPS fallback (PyTorch-native, preserves autograd)
        if _use_mps_fallback():
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
    # Priority 3: Pure PyTorch implementation for MPS (when fallback is disabled)
    # This avoids custom autograd functions which have issues with gradient checkpointing
    if DEVICE_TYPE == "mps":
        from .lora_pytorch import pytorch_lora_qkv
        return pytorch_lora_qkv(
            X,
            QW, QW_quant, QA, QB, QS,
            KW, KW_quant, KA, KB, KS,
            VW, VW_quant, VA, VB, VS,
        )
    
    # Default: CUDA/Triton path
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
    if DEVICE_TYPE == "mps":
        # Priority 1: MLX fast_lora (compiled graph fusion + batch-1 GEMV)
        # Only use MLX path for inference — MLX ops are outside PyTorch autograd
        if _is_mlx_available() and not torch.is_grad_enabled():
            from ..mlx.fast_lora import apply_lora_o
            return apply_lora_o(X, OW, OW_quant, OA, OB, OS)
        # Priority 2: MPS fallback (PyTorch-native, preserves autograd)
        if _use_mps_fallback():
            from .fast_lora import mps_apply_lora_o
            fn = _get_compiled_fn(mps_apply_lora_o)
            return fn(X, OW, OW_quant, OA, OB, OS)
        # Priority 3: Pure PyTorch implementation for MPS (when fallback is disabled)
        # This avoids custom autograd functions which have issues with gradient checkpointing
        from .lora_pytorch import pytorch_lora_o
        return pytorch_lora_o(X, OW, OW_quant, OA, OB, OS)
    # Default: CUDA/Triton path
    from ..fast_lora import LoRA_W
    return LoRA_W.apply(X, OW, OW_quant, OA, OB, OS)
