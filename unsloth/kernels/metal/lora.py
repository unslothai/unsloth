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

from unsloth.kernels.mlx.quantization import MLXQuantizedWeight
import mlx.core as mx

def _mlx_matmul_lora(X, W, W_quant, A, B, s):
    """
    Internal helper for MatMul + LoRA in MLX.
    """
    
    # Base projection
    if isinstance(W, MLXQuantizedWeight):
        # Quantized MatMul: (x, w, scales, biases, transpose=True)
        out = mx.quantized_matmul(
            X,
            W.weight,
            scales=W.scales,
            biases=W.biases,
            transpose=True,
            group_size=W.group_size,
        )
    else:
        # Standard Linear: X @ W.T
        out = X @ W.T
        
    if A is not None:
        # LoRA: (X @ A.T) @ B.T * s
        # A, B are MLX arrays here [Rank, In], [Out, Rank]
        XA = X @ A.T
        lora_out = (XA @ B.T)
        out = out + (lora_out * s)
        
    return out


def apply_lora_qkv_mlx(X, QW, QW_quant, QA, QB, QS, KW, KW_quant, KA, KB, KS, VW, VW_quant, VA, VB, VS):
    """
    MLX implementation of fused QKV projection with LoRA.
    """
    Q = _mlx_matmul_lora(X, QW, QW_quant, QA, QB, QS)
    K = _mlx_matmul_lora(X, KW, KW_quant, KA, KB, KS)
    V = _mlx_matmul_lora(X, VW, VW_quant, VA, VB, VS)
    return Q, K, V

