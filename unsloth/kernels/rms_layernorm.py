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

import triton
import triton.language as tl
import torch
from typing import Tuple
from .utils import calculate_settings, torch_gpu_device


@triton.jit
def _rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    normed = normed.to(W_row.dtype)  # Exact copy from HF
    output = normed * W_row
    tl.store(Y + col_offsets, output, mask = mask)


def _rms_layernorm_backward(
    dY,
    dY_row_stride: tl.constexpr,
    dX,
    dX_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    GEMMA: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fast RMS Layernorm kernel for the backward pass
    Inspiration from a Triton tutorial:
    https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html
    """
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    dY += row_idx * dY_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    if GEMMA:
        dX += row_idx * dX_row_stride
    else:
        dX = dY

    dY_row = tl.load(dY + col_offsets, mask = mask, other = 0).to(tl.float32)
    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    # Get saved row variance
    inv_var = tl.load(r).to(tl.float32)
    normed = X_row * inv_var

    if GEMMA:
        dY_W = dY_row * (W_row + 1.0)
    else:
        dY_W = dY_row * W_row

    rowsum_dY_normed = tl.sum(dY_W * normed, axis = 0)
    output = inv_var / n_cols * (n_cols * dY_W - normed * rowsum_dY_normed)
    tl.store(dX + col_offsets, output, mask = mask)


_rms_layernorm_backward = triton.jit(_rms_layernorm_backward)
_rms_layernorm_backward = triton.heuristics(
    {
        "GEMMA": lambda args: bool(args["GEMMA"]),
    }
)(_rms_layernorm_backward)


@triton.jit
def _gemma_rms_layernorm_forward(
    Y,
    Y_row_stride: tl.constexpr,
    X,
    X_row_stride: tl.constexpr,
    W,
    W_row_stride: tl.constexpr,
    r,
    r_row_stride: tl.constexpr,
    n_cols: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Copies google-deepmind/gemma layers.py#L31 and keras-nlp gemma/rms_normalization.py#L33 exactly:
    # essentially all in float32.
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    Y += row_idx * Y_row_stride
    X += row_idx * X_row_stride
    r += row_idx * r_row_stride

    X_row = tl.load(X + col_offsets, mask = mask, other = 0).to(tl.float32)
    W_row = tl.load(W + col_offsets, mask = mask, other = 0).to(tl.float32)

    row_var = tl.sum(X_row * X_row, axis = 0) / n_cols
    # Explicit float32 scalar to ensure correct type promotion on HIP/ROCm.
    eps_f32 = tl.full((), eps, tl.float32)
    inv_var = tl.math.rsqrt(row_var + eps_f32)
    tl.store(r, inv_var)
    normed = X_row * inv_var
    output = normed * (W_row + 1.0)

    tl.store(Y + col_offsets, output, mask = mask)


def _rms_forward(X, W, eps, gemma, wrap):
    # X: [n_rows, n_cols] contiguous, W contiguous. Returns Y and the saved 1 / rms per row.
    n_rows, n_cols = X.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    Y = torch.empty((n_rows, n_cols), dtype = X.dtype, device = X.device)
    r = torch.empty(n_rows, dtype = torch.float32, device = X.device)
    fx = _gemma_rms_layernorm_forward if gemma else _rms_layernorm_forward
    wrap(fx)[(n_rows,)](
        Y,
        Y.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        r,
        r.stride(0),
        n_cols,
        eps,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )
    return Y, r


def _rms_backward(dY, dX, X, W, r, eps, gemma, wrap):
    # Non-Gemma writes dX over dY (the kernel ignores dX); Gemma writes into dX.
    n_rows, n_cols = dY.shape
    BLOCK_SIZE, num_warps = calculate_settings(n_cols)
    wrap(_rms_layernorm_backward)[(n_rows,)](
        dY,
        dY.stride(0),
        dX,
        dX.stride(0),
        X,
        X.stride(0),
        W,
        W.stride(0),
        r,
        r.stride(0),
        n_cols,
        eps,
        GEMMA = gemma,
        BLOCK_SIZE = BLOCK_SIZE,
        num_warps = num_warps,
    )


_eager_kernel = lambda kernel: kernel


class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X: torch.Tensor,
        W: torch.Tensor,
        eps: float,
        gemma: bool = False,
    ):
        shape = X.shape
        dim: int = shape[-1]
        X = X.reshape(-1, dim).contiguous()
        # kernels read W at unit stride, and this W is the one saved for backward.
        W = W.contiguous()
        with torch_gpu_device(X.device):
            Y, r = _rms_forward(X, W, eps, gemma, _eager_kernel)
        ctx.eps = eps
        ctx.GEMMA = gemma
        ctx.save_for_backward(X, W, r)
        return Y.view(*shape)

    @staticmethod
    def backward(ctx, dY: torch.Tensor):
        shape = dY.shape
        dim: int = shape[-1]
        dY = dY.reshape(-1, dim).contiguous()
        X, W, r = ctx.saved_tensors
        dX = torch.empty_like(dY) if ctx.GEMMA else dY
        with torch_gpu_device(dY.device):
            _rms_backward(dY, dX, X, W, r, ctx.eps, ctx.GEMMA, _eager_kernel)
        dX = dX.view(*shape)
        return dX, None, None, None


# The autograd.Function above reuses the incoming gradient as dX, a mutation torch.compile cannot
# trace, so compiled graphs take these ops instead: the same kernels, with fresh outputs.
_TRACEABLE = hasattr(torch.library, "triton_op") and hasattr(torch.library, "wrap_triton")


def _traced_kernel(kernel):
    # wrap_triton takes the JITFunction under triton.heuristics; every heuristic here only turns
    # a bool argument into a constexpr, which the callers pass explicitly.
    if isinstance(kernel, triton.runtime.autotuner.Heuristics):
        kernel = kernel.fn
    return torch.library.wrap_triton(kernel)


if _TRACEABLE:

    @torch.library.triton_op("unsloth::rms_layernorm", mutates_args = ())
    def _rms_layernorm_op(
        X: torch.Tensor, W: torch.Tensor, eps: float, gemma: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return _rms_forward(X, W, eps, gemma, _traced_kernel)

    @torch.library.triton_op("unsloth::rms_layernorm_backward", mutates_args = ())
    def _rms_layernorm_backward_op(
        dY: torch.Tensor, X: torch.Tensor, W: torch.Tensor, r: torch.Tensor, eps: float, gemma: bool
    ) -> torch.Tensor:
        dX = torch.empty_like(dY)
        # Non-Gemma rewrites its dY argument, so it gets a copy.
        _rms_backward(dY if gemma else dX.copy_(dY), dX, X, W, r, eps, gemma, _traced_kernel)
        return dX

    def _rms_setup_context(ctx, inputs, output):
        X, W, eps, gemma = inputs
        ctx.eps, ctx.gemma = eps, gemma
        ctx.save_for_backward(X, W, output[1])

    def _rms_layernorm_op_backward(ctx, dY, dr):
        X, W, r = ctx.saved_tensors
        dX = torch.ops.unsloth.rms_layernorm_backward(dY.contiguous(), X, W, r, ctx.eps, ctx.gemma)
        return dX, None, None, None

    _rms_layernorm_op.register_autograd(
        _rms_layernorm_op_backward, setup_context = _rms_setup_context
    )


@torch.compiler.disable
def _fast_rms_layernorm_untraced(X, W, eps, gemma):
    return Fast_RMS_Layernorm.apply(X, W, eps, gemma)


def fast_rms_layernorm(
    layernorm,
    X: torch.Tensor,
    gemma: bool = False,
):
    W: torch.Tensor = layernorm.weight
    eps: float = (
        layernorm.variance_epsilon if hasattr(layernorm, "variance_epsilon") else layernorm.eps
    )
    if not torch.compiler.is_compiling():
        return Fast_RMS_Layernorm.apply(X, W, eps, gemma)
    if not _TRACEABLE or X.device.type != "cuda":
        return _fast_rms_layernorm_untraced(X, W, eps, gemma)
    shape = X.shape
    Y, _ = torch.ops.unsloth.rms_layernorm(
        X.reshape(-1, shape[-1]).contiguous(), W.contiguous(), eps, gemma
    )
    return Y.view(shape)


from transformers.models.llama.modeling_llama import LlamaRMSNorm


class Unsloth_LlamaRMSNorm(LlamaRMSNorm):
    def forward(self, X):
        return fast_rms_layernorm(self, X, gemma = False)


try:
    from transformers.models.mllama.modeling_mllama import MllamaTextRMSNorm
    class Unsloth_MllamaTextRMSNorm(MllamaTextRMSNorm):
        def forward(self, X):
            return fast_rms_layernorm(self, X, gemma = False)


except (ImportError, AttributeError):
    pass


def patch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = Unsloth_LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = Unsloth_MllamaTextRMSNorm
    except (ImportError, AttributeError, NameError):
        pass
    return


def unpatch_rms_layernorm():
    import transformers.models.llama.modeling_llama

    transformers.models.llama.modeling_llama.LlamaRMSNorm = LlamaRMSNorm
    try:
        import transformers.models.mllama.modeling_mllama
        transformers.models.mllama.modeling_mllama.MllamaTextRMSNorm = MllamaTextRMSNorm
    except (ImportError, AttributeError, NameError):
        pass
    return


def test_rms_layernorm(
    dim = 1024,
    eps = 1e-5,
    dtype = torch.float16,
    bsz = 21,
    random_state = 3407,
    seqlen = 3341,
):
    from transformers.models.llama.modeling_llama import LlamaRMSNorm

    layernorm = LlamaRMSNorm((dim,), eps = eps).to("cuda")
    torch.cuda.manual_seed(random_state)
    torch.manual_seed(random_state)
    torch.nn.init.uniform_(layernorm.weight)
    X = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda")
    XX = X.clone()
    X.requires_grad_(True)
    XX.requires_grad_(True)
    Y = layernorm(X)
    YY = torch.randn((bsz, seqlen, dim), dtype = dtype, device = "cuda", requires_grad = True)
    Y.backward(YY)
    correct_grad = X.grad.clone()
    Y = fast_rms_layernorm(layernorm, XX)
    Y.backward(YY)
    assert torch.amax(correct_grad - XX.grad).item() <= 0.05


def testing_suite_layernorm():
    for dim in [512, 1024, 2048]:
        for dtype in [torch.float16, torch.bfloat16]:
            with torch.autocast(device_type = "cuda", dtype = dtype):
                for seqlen in [3341, 2048, 349]:
                    for random_state in [3407, 42]:
                        test_rms_layernorm(
                            dim = dim,
                            eps = 1e-5,
                            dtype = dtype,
                            bsz = 21,
                            random_state = random_state,
                            seqlen = seqlen,
                        )
