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

import triton
MAX_FUSED_SIZE : int = 65536
next_power_of_2 = triton.next_power_of_2
import functools

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch
torch_Tensor = torch.Tensor
from packaging.version import Version
if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")
pass


# tl.math.tanh now is libdevice.tanh
from packaging.version import Version
import triton
import triton.language as tl
if Version(triton.__version__) >= Version("3.0.0"):
    from triton.language.extra import libdevice
    triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh
    # No casting in old Triton versions
    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)
    pass
pass


def calculate_settings(n : int) -> (int, int,):
    BLOCK_SIZE : int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(f"Cannot launch Triton kernel since n = {n} exceeds "\
                           f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}.")
    num_warps : int = 4
    if   BLOCK_SIZE >= 32768: num_warps = 32
    elif BLOCK_SIZE >=  8192: num_warps = 16
    elif BLOCK_SIZE >=  2048: num_warps = 8
    return BLOCK_SIZE, num_warps
pass


import bitsandbytes as bnb
import ctypes

# https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1330/files
HAS_CUDA_STREAM = Version(bnb.__version__) > Version("0.43.3")
get_ptr = bnb.functional.get_ptr

if torch.cuda.device_count() > 1:
    torch_cuda_device = torch.cuda.device
else:
    from contextlib import nullcontext
    def torch_cuda_device(device): return nullcontext()
pass
_cuda_getCurrentRawStream = torch._C._cuda_getCurrentRawStream
c_void_p = ctypes.c_void_p
def _get_tensor_stream(tensor: torch_Tensor) -> c_void_p:
    return c_void_p(_cuda_getCurrentRawStream(tensor.device.index))
pass

# Get array of CUDA streams and other buffers
global CUDA_STREAMS
global WEIGHT_BUFFERS
global ABSMAX_BUFFERS

_CUDA_STREAMS = {
    (index := torch.cuda.device(i).idx) : ctypes.c_void_p(torch._C._cuda_getCurrentRawStream(index))
    for i in range(torch.cuda.device_count())
}
CUDA_STREAMS   = [None] * (max(_CUDA_STREAMS.keys()) + 1)
WEIGHT_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
ABSMAX_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
for k, v in _CUDA_STREAMS.items(): CUDA_STREAMS[k] = v
CUDA_STREAMS = tuple(CUDA_STREAMS)
del _CUDA_STREAMS

# Bitsandbytes operations
ctypes_c_int   = ctypes.c_int
ctypes_c_int32 = ctypes.c_int32
cdequantize_blockwise_fp32      = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4  = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4  = bnb.functional.lib.cdequantize_blockwise_bf16_nf4
cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16
torch_mm = torch.mm
torch_mv = torch.mv
torch_matmul = torch.matmul
torch_addmm  = torch.addmm
torch_empty  = torch.empty

def QUANT_STATE(W): return getattr(W, "quant_state", None)

def get_lora_parameters(proj):
    # For DPO or disabled adapters
    base_layer = getattr(proj, "base_layer", proj) # (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, getattr(W, "quant_state", None), None, None, None
    pass

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None: adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]
    
    return (
        W,
        getattr(W, "quant_state", None),
        proj.lora_A [adapter].weight,
        proj.lora_B [adapter].weight,
        proj.scaling[adapter],
    )
pass


def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = getattr(proj, "base_layer", proj) # (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, getattr(W, "quant_state", None), None, None, None, base_layer.bias
    pass

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None: adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    return (
        W,
        getattr(W, "quant_state", None),
        proj.lora_A [adapter].weight,
        proj.lora_B [adapter].weight,
        proj.scaling[adapter],
        base_layer.bias,
    )
pass

if HAS_CUDA_STREAM:
    @torch.inference_mode
    def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
        if quant_state is None: return W
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            # Old quant_state as a list of lists
            absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        global CUDA_STREAMS
        device = W.device
        device_index = device.index
        CUDA_STREAM = CUDA_STREAMS[device_index]

        n_elements_absmax = absmax.numel()

        # Create weight matrix
        if use_global_buffer:

            # Use same buffers for faster inference
            size = shape[0]*shape[1]
            global WEIGHT_BUFFERS
            global ABSMAX_BUFFERS
            WEIGHT_BUFFER = WEIGHT_BUFFERS[device_index]
            ABSMAX_BUFFER = ABSMAX_BUFFERS[device_index]
            if WEIGHT_BUFFER is None:
                WEIGHT_BUFFERS[device_index] = WEIGHT_BUFFER = torch_empty(size, dtype = dtype, device = device, requires_grad = False)
                ABSMAX_BUFFERS[device_index] = ABSMAX_BUFFER = torch_empty(n_elements_absmax, dtype = torch.float32, device = device, requires_grad = False)

            if size > WEIGHT_BUFFER.numel(): WEIGHT_BUFFER.resize_(size)
            if n_elements_absmax > ABSMAX_BUFFER.numel(): ABSMAX_BUFFER.resize_(n_elements_absmax)

            out = WEIGHT_BUFFER[:size].view(shape)
            out_absmax = ABSMAX_BUFFER[:n_elements_absmax]
        else:
            if out is None:
                out = torch_empty(shape, dtype = dtype, device = device, requires_grad = False)
            else:
                assert(out.shape == shape)
                assert(out.dtype == dtype)
            out_absmax = torch_empty(n_elements_absmax, dtype = torch.float32, device = device, requires_grad = False)
        pass

        # NF4 dequantization of statistics
        ptr_out_absmax = get_ptr(out_absmax)
        with torch_cuda_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
                ctypes_c_int(blocksize2), ctypes_c_int(n_elements_absmax), CUDA_STREAM
            )
            out_absmax += offset

            # Dequantize W
            fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
                 cdequantize_blockwise_bf16_nf4
            fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
               ctypes_c_int(blocksize), ctypes_c_int(out.numel()), CUDA_STREAM,)
        pass
        # Careful returning transposed data
        is_transposed = (True if W.shape[0] == 1 else False)
        return out.t() if is_transposed else out
    pass
else:
    @torch.inference_mode
    def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
        if quant_state is None: return W
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            # Old quant_state as a list of lists
            absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass

        n_elements_absmax = absmax.numel()
        device = W.device

        # Create weight matrix
        if out is None:
            out = torch_empty(shape, dtype = dtype, device = device, requires_grad = False)
        else:
            assert(out.shape == shape)
            assert(out.dtype == dtype)
        out_absmax = torch_empty(n_elements_absmax, dtype = torch.float32, device = device, requires_grad = False)

        # Do dequantization
        ptr_out_absmax = get_ptr(out_absmax)
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), ptr_out_absmax,
            ctypes_c_int(blocksize2), ctypes_c_int(n_elements_absmax),
        )
        out_absmax += offset

        fx = cdequantize_blockwise_fp16_nf4 if dtype == torch.float16 else \
             cdequantize_blockwise_bf16_nf4
        fx(get_ptr(None), get_ptr(W), ptr_out_absmax, get_ptr(out),
           ctypes_c_int(blocksize), ctypes_c_int(out.numel()),)

        # Careful returning transposed data
        is_transposed = (True if W.shape[0] == 1 else False)
        return out.t() if is_transposed else out
    pass
pass


if HAS_CUDA_STREAM:
    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None: return torch_matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            stats      = quant_state.code
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        global CUDA_STREAMS
        device = W.device
        device_index = device.index
        CUDA_STREAM = CUDA_STREAMS[device_index]
        
        # assert(dtype == X.dtype)
        bout = shape[0]

        if out is None:
            out = torch_empty((1, 1, bout,), dtype = dtype, device = device)
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd+1)//2
        m = ctypes_c_int32(m)
        n = ctypes_c_int32(n)
        k = ctypes_c_int32(k)
        lda = ctypes_c_int32(lda)
        ldb = ctypes_c_int32(ldb)
        ldc = ctypes_c_int32(ldc)

        df = torch_empty(absmax.shape, dtype = torch.float32, device = device)
        with torch_cuda_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df),
                ctypes_c_int(blocksize2), ctypes_c_int(df.numel()), CUDA_STREAM,
            )
            df += offset
            absmax = df

            fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else \
                cgemm_4bit_inference_naive_bf16

            blocksize = ctypes_c_int32(blocksize)
            fx(m, n, k, get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(stats), get_ptr(out),
               lda, ldb, ldc, blocksize, CUDA_STREAM,)
        pass

        return out
    pass
else:
    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None: return torch.matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax     = quant_state.absmax
            shape      = quant_state.shape
            dtype      = quant_state.dtype
            blocksize  = quant_state.blocksize
            stats      = quant_state.code
            offset     = quant_state.offset
            state2     = quant_state.state2
            absmax2    = state2.absmax
            code2      = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        # assert(dtype == X.dtype)
        bout = shape[0]
        device = W.device

        if out is None:
            out = torch_empty((1, 1, bout,), dtype = dtype, device = device)
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd+1)//2
        m = ctypes_c_int32(m)
        n = ctypes_c_int32(n)
        k = ctypes_c_int32(k)
        lda = ctypes_c_int32(lda)
        ldb = ctypes_c_int32(ldb)
        ldc = ctypes_c_int32(ldc)

        df = torch_empty(absmax.shape, dtype = torch.float32, device = device)
        cdequantize_blockwise_fp32(
            get_ptr(code2), get_ptr(absmax), get_ptr(absmax2), get_ptr(df),
            ctypes_c_int(blocksize2), ctypes_c_int(df.numel()),
        )
        df += offset
        absmax = df

        fx = cgemm_4bit_inference_naive_fp16 if dtype == torch.float16 else \
            cgemm_4bit_inference_naive_bf16

        blocksize = ctypes_c_int32(blocksize)
        fx(m, n, k, get_ptr(X), get_ptr(W), get_ptr(absmax), get_ptr(stats), get_ptr(out),
           lda, ldb, ldc, blocksize,)

        return out
    pass
pass


def fast_linear_forward(proj, X, temp_lora = None, out = None):

    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1: return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch_matmul(X, W.t(), out = out)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)
        out = torch_matmul(X, W, out = out)
    pass

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)
        pass
        
        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch_mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch_mm(X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora)
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        pass
        out = out.view(bsz, 1, out_dim)
    pass

    if bias is not None: out += bias

    return out
pass


def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype
    W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False
    pass
    out = torch_matmul(X, W, out = out)
    if W_quant is not None: del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))
    pass
    
    return out.view(batch, seq_len, -1) if reshape else out
pass
