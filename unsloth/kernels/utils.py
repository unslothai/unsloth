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

import importlib
import triton
import ctypes

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2
import functools
from typing import Optional

from ..device_type import (
    is_hip,
    get_device_type,
    DEVICE_TYPE,
    DEVICE_TYPE_TORCH,
    DEVICE_COUNT,
    ALLOW_PREQUANTIZED_MODELS,
)
from .fp8 import weight_dequant, fp8_linear
import functools

# torch.cuda.amp.custom_fwd is deprecated >= 2.4
import torch

torch_Tensor = torch.Tensor
from unsloth_zoo.utils import Version

if DEVICE_TYPE == "xpu" and Version(torch.__version__) < Version("2.6.0"):
    raise RuntimeError(
        "Intel xpu currently supports unsloth with torch.version >= 2.6.0"
    )

if Version(torch.__version__) < Version("2.4.0"):
    torch_amp_custom_fwd = torch.cuda.amp.custom_fwd
    torch_amp_custom_bwd = torch.cuda.amp.custom_bwd
else:
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "cuda")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "cuda")

if DEVICE_TYPE == "xpu":
    torch_amp_custom_fwd = torch.amp.custom_fwd(device_type = "xpu")
    torch_amp_custom_bwd = torch.amp.custom_bwd(device_type = "xpu")


# tl.math.tanh now is libdevice.tanh
import triton
import triton.language as tl

if Version(triton.__version__) >= Version("3.0.0"):
    if DEVICE_TYPE == "xpu":
        triton_tanh = tl.extra.intel.libdevice.tanh
    else:
        from triton.language.extra import libdevice

        triton_tanh = libdevice.tanh
    triton_cast = tl.cast
else:
    triton_tanh = tl.math.tanh

    # No casting in old Triton versions
    @triton.jit
    def triton_cast(x, dtype):
        return x.to(dtype)


@functools.lru_cache(1)
def is_cdna():
    return is_hip() and triton.runtime.driver.active.get_current_target().arch in (
        "gfx940",
        "gfx941",
        "gfx942",
    )


def calculate_settings(
    n: int,
) -> (
    int,
    int,
):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps


HAS_CUDA_STREAM = False
import bitsandbytes as bnb

# https://github.com/bitsandbytes-foundation/bitsandbytes/pull/1330/files
HAS_CUDA_STREAM = Version(bnb.__version__) > Version("0.43.3")
get_ptr = bnb.functional.get_ptr

if DEVICE_TYPE == "xpu":
    HAS_XPU_STREAM = True

if DEVICE_COUNT > 1:
    if DEVICE_TYPE in ("cuda", "hip"):
        torch_gpu_device = torch.cuda.device
    elif DEVICE_TYPE == "xpu":
        torch_gpu_device = torch.xpu.device
else:
    from contextlib import nullcontext

    def torch_gpu_device(device):
        return nullcontext()


# INTEL GPU Specific Logic
if DEVICE_TYPE == "xpu":
    _gpu_getCurrentRawStream = torch._C._xpu_getCurrentRawStream
# NVIDIA GPU Default Logic
else:
    _gpu_getCurrentRawStream = torch._C._cuda_getCurrentRawStream

c_void_p = ctypes.c_void_p


def _get_tensor_stream(tensor: torch_Tensor) -> c_void_p:
    return c_void_p(_gpu_getCurrentRawStream(tensor.device.index))


# Get array of CUDA streams and other buffers
global CUDA_STREAMS
global XPU_STREAMS
global WEIGHT_BUFFERS
global ABSMAX_BUFFERS

# INTEL GPU Specific Logic
if DEVICE_TYPE == "xpu":
    _XPU_STREAMS = {
        (index := torch.xpu.device(i).idx): ctypes.c_void_p(
            torch._C._xpu_getCurrentRawStream(index)
        )
        for i in range(DEVICE_COUNT)
    }
    XPU_STREAMS = [None] * (max(_XPU_STREAMS.keys()) + 1)
    WEIGHT_BUFFERS = [None] * (max(_XPU_STREAMS.keys()) + 1)
    ABSMAX_BUFFERS = [None] * (max(_XPU_STREAMS.keys()) + 1)
    for k, v in _XPU_STREAMS.items():
        XPU_STREAMS[k] = v
    XPU_STREAMS = tuple(XPU_STREAMS)
    del _XPU_STREAMS
else:
    # NVIDIA GPU Default Logic
    _CUDA_STREAMS = {
        (index := torch.cuda.device(i).idx): ctypes.c_void_p(
            torch._C._cuda_getCurrentRawStream(index)
        )
        for i in range(DEVICE_COUNT)
    }
    CUDA_STREAMS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    WEIGHT_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    ABSMAX_BUFFERS = [None] * (max(_CUDA_STREAMS.keys()) + 1)
    for k, v in _CUDA_STREAMS.items():
        CUDA_STREAMS[k] = v
    CUDA_STREAMS = tuple(CUDA_STREAMS)
    del _CUDA_STREAMS

# Bitsandbytes operations
ctypes_c_int = ctypes.c_int
ctypes_c_int32 = ctypes.c_int32
cdequantize_blockwise_fp32 = bnb.functional.lib.cdequantize_blockwise_fp32
cdequantize_blockwise_fp16_nf4 = bnb.functional.lib.cdequantize_blockwise_fp16_nf4
cdequantize_blockwise_bf16_nf4 = bnb.functional.lib.cdequantize_blockwise_bf16_nf4

if DEVICE_TYPE == "xpu":
    # https://github.com/bitsandbytes-foundation/bitsandbytes/blob/c3b8de268fdb55a88f92feada23fc811a1e6877a/bitsandbytes/backends/xpu/ops.py#L115
    # for xpu, inference gemv using above link
    cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemv_4bit_inference_fp16
    cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemv_4bit_inference_bf16
else:
    cgemm_4bit_inference_naive_fp16 = bnb.functional.lib.cgemm_4bit_inference_naive_fp16
    cgemm_4bit_inference_naive_bf16 = bnb.functional.lib.cgemm_4bit_inference_naive_bf16


torch_device_stream = (
    torch.xpu.current_stream if DEVICE_TYPE == "xpu" else torch.cuda.current_stream
)

torch_mm = torch.mm
torch_mv = torch.mv
torch_matmul = torch.matmul
torch_addmm = torch.addmm
torch_empty = torch.empty
torch_float32 = torch.float32
torch_float16 = torch.float16
torch_bfloat16 = torch.bfloat16


# Check whether torchao can be imported to get Float8Tensor
if importlib.util.find_spec("torchao") is not None:
    try:
        from torchao.quantization import Float8Tensor
    except:
        import torchao

        if Version(torchao.__version__) >= Version("0.15.0"):
            print(
                f"Unsloth: `from torchao.quantization import Float8Tensor` failed on version={torchao.__version__}"
            )
        Float8Tensor = type(None)
else:
    Float8Tensor = type(None)


def QUANT_STATE(W):
    return getattr(W, "quant_state", None)


def get_lora_parameters(proj):
    """
    Return a 5-tuple of (weight, weight quant_state, lora A, lora B, and lora scale).
    If QAT is enabled, additionally fake quantize the base layer and lora weights.
    """
    # For DPO or disabled adapters
    base_layer = getattr(
        proj, "base_layer", proj
    )  # (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    # Optionally apply fake quantization to base layer weights for QAT
    if hasattr(base_layer, "weight_fake_quantizer"):
        weight_fake_quantizer = getattr(base_layer, "weight_fake_quantizer", None)
        if weight_fake_quantizer is not None:
            W = weight_fake_quantizer(W)

    # Get quant state for 4bit or FP8
    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    if getattr(base_layer, "quant_method", None) == "fp8":
        # we need to somehow store and pass this information :)
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    # Optionally apply fake quantization to lora weights for QAT
    lora_A_linear = proj.lora_A[adapter]
    lora_B_linear = proj.lora_B[adapter]
    A = lora_A_linear.weight
    B = lora_B_linear.weight
    if hasattr(lora_A_linear, "weight_fake_quantizer"):
        lora_A_fake_quantizer = getattr(lora_A_linear, "weight_fake_quantizer", None)
        if lora_A_fake_quantizer is not None:
            A = lora_A_fake_quantizer(A)
    if hasattr(lora_B_linear, "weight_fake_quantizer"):
        lora_B_fake_quantizer = getattr(lora_B_linear, "weight_fake_quantizer", None)
        if lora_B_fake_quantizer is not None:
            B = lora_B_fake_quantizer(B)

    return (
        W,
        W_quant,
        A,
        B,
        proj.scaling[adapter],
    )


def get_lora_parameters_bias(proj):
    # For DPO or disabled adapters
    base_layer = getattr(
        proj, "base_layer", proj
    )  # (proj.base_layer if hasattr(proj, "base_layer") else proj)
    W = base_layer.weight

    # Get quant state for 4bit or FP8
    W_quant = getattr(W, "quant_state", None)
    if W_quant is None:
        W_quant = getattr(base_layer, "weight_scale_inv", None)
        if W_quant is None:
            W_quant = getattr(base_layer, "weight_scale", None)

    # if not hasattr(proj, "disable_adapters") or proj.disable_adapters or proj.merged:
    if getattr(proj, "disable_adapters", True) or proj.merged:
        return W, W_quant, None, None, None, base_layer.bias

    if getattr(base_layer, "quant_method", None) == "fp8":
        # we need to somehow store and pass this information :)
        W.block_size = getattr(base_layer, "block_size", [128, 128])
        W_quant.block_size = W.block_size

    adapter = getattr(proj, "active_adapters", None)
    if adapter is None:
        adapter = getattr(proj, "active_adapter", ("default"))
    adapter = adapter[0]

    return (
        W,
        W_quant,
        proj.lora_A[adapter].weight,
        proj.lora_B[adapter].weight,
        proj.scaling[adapter],
        base_layer.bias,
    )


def _maybe_fake_quantize_activations(
    X: torch.Tensor, proj: torch.nn.Module
) -> torch.Tensor:
    """
    If QAT is enabled, fake quantize the input activations.
    Otherwise, just return the input activations as is.
    Weights are fake quantized separately in `get_lora_parameters`.
    """
    base_layer = getattr(proj, "base_layer", proj)
    activation_fake_quantizer = getattr(base_layer, "activation_fake_quantizer", None)
    if activation_fake_quantizer is not None:
        X = activation_fake_quantizer(X)
    return X


# INTEL GPU Specific Logic
if DEVICE_TYPE == "xpu" and HAS_XPU_STREAM:

    @torch.inference_mode
    def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
        # TODO: After adding XPU BNB support, check this function
        if isinstance(W, Float8Tensor):
            return W.dequantize()
        if quant_state is None:
            return W
        if W.dtype == torch.float8_e4m3fn:
            return weight_dequant(W, quant_state)
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
            blocksize2 = state2.blocksize
        else:
            # Old quant_state as a list of lists
            absmax, shape, dtype, blocksize, compressed_stats, _, _ = quant_state
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        global XPU_STREAMS
        device = W.device
        device_index = device.index
        XPU_STREAM = XPU_STREAMS[device_index]

        n_elements_absmax = absmax.numel()
        # Create weight matrix
        if use_global_buffer:
            # Use same buffers for faster inference
            size = shape[0] * shape[1]
            global WEIGHT_BUFFERS
            global ABSMAX_BUFFERS
            WEIGHT_BUFFER = WEIGHT_BUFFERS[device_index]
            ABSMAX_BUFFER = ABSMAX_BUFFERS[device_index]
            if WEIGHT_BUFFER is None:
                WEIGHT_BUFFERS[device_index] = WEIGHT_BUFFER = torch_empty(
                    size, dtype = dtype, device = device, requires_grad = False
                )
                ABSMAX_BUFFERS[device_index] = ABSMAX_BUFFER = torch_empty(
                    n_elements_absmax,
                    dtype = torch.float32,
                    device = device,
                    requires_grad = False,
                )

            if size > WEIGHT_BUFFER.numel():
                WEIGHT_BUFFER.resize_(size)
            if n_elements_absmax > ABSMAX_BUFFER.numel():
                ABSMAX_BUFFER.resize_(n_elements_absmax)

            out = WEIGHT_BUFFER[:size].view(shape)
            out_absmax = ABSMAX_BUFFER[:n_elements_absmax]
        else:
            if out is None:
                out = torch_empty(
                    shape, dtype = dtype, device = device, requires_grad = False
                )
            else:
                assert out.shape == shape
                assert out.dtype == dtype
            out_absmax = torch_empty(
                n_elements_absmax,
                dtype = torch_float32,
                device = device,
                requires_grad = False,
            )

        # NF4 dequantization of statistics
        ptr_out_absmax = get_ptr(out_absmax)
        with torch_gpu_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2),
                get_ptr(absmax),
                get_ptr(absmax2),
                ptr_out_absmax,
                ctypes_c_int(blocksize2),
                ctypes_c_int(n_elements_absmax),
                XPU_STREAM,
            )
            out_absmax += offset

            # Dequantize W
            fx = (
                cdequantize_blockwise_fp16_nf4
                if dtype == torch_float16
                else cdequantize_blockwise_bf16_nf4
            )
            fx(
                get_ptr(None),
                get_ptr(W),
                ptr_out_absmax,
                get_ptr(out),
                ctypes_c_int(blocksize),
                ctypes_c_int(out.numel()),
                XPU_STREAM,
            )
        # Careful returning transposed data
        is_transposed = True if W.shape[0] == 1 else False
        return out.t() if is_transposed else out

# NVIDIA GPU Default Logic
elif DEVICE_TYPE in ("cuda", "hip") and HAS_CUDA_STREAM:

    @torch.inference_mode
    def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
        if isinstance(W, Float8Tensor):
            return W.dequantize()
        if quant_state is None:
            return W
        if W.dtype == torch.float8_e4m3fn:
            return weight_dequant(W, quant_state)
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
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
            size = shape[0] * shape[1]
            global WEIGHT_BUFFERS
            global ABSMAX_BUFFERS
            WEIGHT_BUFFER = WEIGHT_BUFFERS[device_index]
            ABSMAX_BUFFER = ABSMAX_BUFFERS[device_index]
            if WEIGHT_BUFFER is None:
                WEIGHT_BUFFERS[device_index] = WEIGHT_BUFFER = torch_empty(
                    size, dtype = dtype, device = device, requires_grad = False
                )
                ABSMAX_BUFFERS[device_index] = ABSMAX_BUFFER = torch_empty(
                    n_elements_absmax,
                    dtype = torch_float32,
                    device = device,
                    requires_grad = False,
                )

            if size > WEIGHT_BUFFER.numel():
                WEIGHT_BUFFER.resize_(size)
            if n_elements_absmax > ABSMAX_BUFFER.numel():
                ABSMAX_BUFFER.resize_(n_elements_absmax)

            out = WEIGHT_BUFFER[:size].view(shape)
            out_absmax = ABSMAX_BUFFER[:n_elements_absmax]
        else:
            if out is None:
                out = torch_empty(
                    shape, dtype = dtype, device = device, requires_grad = False
                )
            else:
                assert out.shape == shape
                assert out.dtype == dtype
            out_absmax = torch_empty(
                n_elements_absmax,
                dtype = torch_float32,
                device = device,
                requires_grad = False,
            )
        pass

        # NF4 dequantization of statistics
        ptr_out_absmax = get_ptr(out_absmax)
        with torch_gpu_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2),
                get_ptr(absmax),
                get_ptr(absmax2),
                ptr_out_absmax,
                ctypes_c_int(blocksize2),
                ctypes_c_int(n_elements_absmax),
                CUDA_STREAM,
            )
            out_absmax += offset

            # Dequantize W
            fx = (
                cdequantize_blockwise_fp16_nf4
                if dtype == torch_float16
                else cdequantize_blockwise_bf16_nf4
            )
            fx(
                get_ptr(None),
                get_ptr(W),
                ptr_out_absmax,
                get_ptr(out),
                ctypes_c_int(blocksize),
                ctypes_c_int(out.numel()),
                CUDA_STREAM,
            )
        pass
        # Careful returning transposed data
        is_transposed = True if W.shape[0] == 1 else False
        return out.t() if is_transposed else out

    pass
else:

    @torch.inference_mode
    def fast_dequantize(W, quant_state = None, out = None, use_global_buffer = False):
        if isinstance(W, Float8Tensor):
            return W.dequantize()
        if quant_state is None:
            return W
        if W.dtype == torch.float8_e4m3fn:
            return weight_dequant(W, quant_state)
        if type(quant_state) is not list:
            # New quant_state as a class
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
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
            assert out.shape == shape
            assert out.dtype == dtype
        out_absmax = torch_empty(
            n_elements_absmax, dtype = torch_float32, device = device, requires_grad = False
        )

        # Do dequantization
        ptr_out_absmax = get_ptr(out_absmax)
        cdequantize_blockwise_fp32(
            get_ptr(code2),
            get_ptr(absmax),
            get_ptr(absmax2),
            ptr_out_absmax,
            ctypes_c_int(blocksize2),
            ctypes_c_int(n_elements_absmax),
        )
        out_absmax += offset

        fx = (
            cdequantize_blockwise_fp16_nf4
            if dtype == torch_float16
            else cdequantize_blockwise_bf16_nf4
        )
        fx(
            get_ptr(None),
            get_ptr(W),
            ptr_out_absmax,
            get_ptr(out),
            ctypes_c_int(blocksize),
            ctypes_c_int(out.numel()),
        )

        # Careful returning transposed data
        is_transposed = True if W.shape[0] == 1 else False
        return out.t() if is_transposed else out

    pass


# INTEL GPU Specific Logic
if DEVICE_TYPE == "xpu" and HAS_XPU_STREAM:

    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None:
            return torch_matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            stats = quant_state.code
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = (
                quant_state
            )
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        global XPU_STREAMS
        device = W.device
        device_index = device.index
        XPU_STREAM = XPU_STREAMS[device_index]

        # assert(dtype == X.dtype)
        bout = shape[0]

        if out is None:
            out = torch_empty(
                (
                    1,
                    1,
                    bout,
                ),
                dtype = dtype,
                device = device,
            )
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        if DEVICE_TYPE == "xpu":
            m = 1
            n = shape[0]
        else:
            n = 1
            m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd + 1) // 2
        m = ctypes_c_int32(m)
        n = ctypes_c_int32(n)
        k = ctypes_c_int32(k)
        lda = ctypes_c_int32(lda)
        ldb = ctypes_c_int32(ldb)
        ldc = ctypes_c_int32(ldc)

        df = torch_empty(absmax.shape, dtype = torch_float32, device = device)
        with torch_gpu_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2),
                get_ptr(absmax),
                get_ptr(absmax2),
                get_ptr(df),
                ctypes_c_int(blocksize2),
                ctypes_c_int(df.numel()),
                XPU_STREAM,
            )
            df += offset
            absmax = df

            fx = (
                cgemm_4bit_inference_naive_fp16
                if dtype == torch_float16
                else cgemm_4bit_inference_naive_bf16
            )

            blocksize = ctypes_c_int32(blocksize)
            fx(
                m,
                n,
                k,
                get_ptr(X),
                get_ptr(W),
                get_ptr(absmax),
                get_ptr(stats),
                get_ptr(out),
                lda,
                ldb,
                ldc,
                blocksize,
                XPU_STREAM,
            )

        return out

elif DEVICE_TYPE in ("cuda", "hip") and HAS_CUDA_STREAM:

    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None:
            return torch_matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            stats = quant_state.code
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = (
                quant_state
            )
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
            out = torch_empty(
                (
                    1,
                    1,
                    bout,
                ),
                dtype = dtype,
                device = device,
            )
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd + 1) // 2
        m = ctypes_c_int32(m)
        n = ctypes_c_int32(n)
        k = ctypes_c_int32(k)
        lda = ctypes_c_int32(lda)
        ldb = ctypes_c_int32(ldb)
        ldc = ctypes_c_int32(ldc)

        df = torch_empty(absmax.shape, dtype = torch_float32, device = device)
        with torch_gpu_device(device):
            cdequantize_blockwise_fp32(
                get_ptr(code2),
                get_ptr(absmax),
                get_ptr(absmax2),
                get_ptr(df),
                ctypes_c_int(blocksize2),
                ctypes_c_int(df.numel()),
                CUDA_STREAM,
            )
            df += offset
            absmax = df

            fx = (
                cgemm_4bit_inference_naive_fp16
                if dtype == torch_float16
                else cgemm_4bit_inference_naive_bf16
            )

            blocksize = ctypes_c_int32(blocksize)
            fx(
                m,
                n,
                k,
                get_ptr(X),
                get_ptr(W),
                get_ptr(absmax),
                get_ptr(stats),
                get_ptr(out),
                lda,
                ldb,
                ldc,
                blocksize,
                CUDA_STREAM,
            )
        pass

        return out

    pass
else:

    def fast_gemv(X, W, quant_state, out = None):
        if quant_state is None:
            return torch_matmul(X, W, out = out)
        # For fast X @ W where seq_len == 1
        # From https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L1469
        _, q_len, hd = X.shape
        # assert(q_len == 1)

        if type(quant_state) is not list:
            # https://github.com/TimDettmers/bitsandbytes/pull/763/files
            absmax = quant_state.absmax
            shape = quant_state.shape
            dtype = quant_state.dtype
            blocksize = quant_state.blocksize
            stats = quant_state.code
            offset = quant_state.offset
            state2 = quant_state.state2
            absmax2 = state2.absmax
            code2 = state2.code
            blocksize2 = state2.blocksize
        else:
            absmax, shape, dtype, blocksize, compressed_stats, quant_type, stats = (
                quant_state
            )
            offset, state2 = compressed_stats
            absmax2, code2, blocksize2, _, _, _, _ = state2
        pass
        # assert(dtype == X.dtype)
        bout = shape[0]
        device = W.device

        if out is None:
            out = torch_empty(
                (
                    1,
                    1,
                    bout,
                ),
                dtype = dtype,
                device = device,
            )
        # else:
        #     assert(out.shape == (1, 1, bout,))
        # pass

        n = 1
        m = shape[0]
        k = shape[1]
        lda = shape[0]
        ldc = shape[0]
        ldb = (hd + 1) // 2
        m = ctypes_c_int32(m)
        n = ctypes_c_int32(n)
        k = ctypes_c_int32(k)
        lda = ctypes_c_int32(lda)
        ldb = ctypes_c_int32(ldb)
        ldc = ctypes_c_int32(ldc)

        df = torch_empty(absmax.shape, dtype = torch_float32, device = device)
        cdequantize_blockwise_fp32(
            get_ptr(code2),
            get_ptr(absmax),
            get_ptr(absmax2),
            get_ptr(df),
            ctypes_c_int(blocksize2),
            ctypes_c_int(df.numel()),
        )
        df += offset
        absmax = df

        fx = (
            cgemm_4bit_inference_naive_fp16
            if dtype == torch_float16
            else cgemm_4bit_inference_naive_bf16
        )

        blocksize = ctypes_c_int32(blocksize)
        fx(
            m,
            n,
            k,
            get_ptr(X),
            get_ptr(W),
            get_ptr(absmax),
            get_ptr(stats),
            get_ptr(out),
            lda,
            ldb,
            ldc,
            blocksize,
        )

        return out

    pass


def fast_linear_forward(proj, X, temp_lora = None, out = None):
    W, W_quant, lora_A, lora_B, lora_S, bias = get_lora_parameters_bias(proj)
    bsz, q_len, in_dim = X.shape
    if q_len != 1:
        return matmul_lora(X, W, W_quant, lora_A, lora_B, lora_S)

    if W_quant is None:
        out = torch_matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant, bias)
    elif bsz == 1 and q_len == 1:
        out = fast_gemv(X, W, W_quant, out = out)
    else:
        W = fast_dequantize(W.t(), W_quant, use_global_buffer = True)
        out = torch_matmul(X, W, out = out)

    # Add in LoRA weights
    if lora_A is not None:
        out_dim = out.shape[2]
        dtype = X.dtype

        if not hasattr(lora_A, "_fast_lora"):
            lora_A._fast_lora = lora_A.to(dtype)
            lora_B._fast_lora = lora_B.to(dtype)

        if bsz == 1:
            out = out.view(out_dim)
            temp_lora = torch_mv(lora_A._fast_lora, X.ravel(), out = temp_lora)
            out.addmv_(lora_B._fast_lora, temp_lora, alpha = lora_S)
        else:
            out = out.view(bsz, out_dim)
            temp_lora = torch_mm(
                X.view(bsz, in_dim), lora_A._fast_lora.t(), out = temp_lora
            )
            out.addmm_(temp_lora, lora_B._fast_lora.t(), alpha = lora_S)
        out = out.view(bsz, 1, out_dim)

    if bias is not None:
        out += bias

    return out


def matmul_lora(X, W, W_quant, A, B, s, out = None):
    dtype = X.dtype

    if X.dim() == 3:
        batch, seq_len, d = X.shape
        X = X.view(-1, X.shape[-1])
        reshape = True
    else:
        reshape = False

    if isinstance(W, Float8Tensor):
        assert W.ndim == 2
        if W.block_size[0] == W.shape[0] and W.block_size[1] == 1:
            # In the backward pass, rowwise scaled becomes colwise scaled after we
            # transpose the weight tensor. Use this case to detect backward.
            # TODO: would be simpler if we simply don't call `matmul_lora` in backward
            W = W.dequantize()
        else:
            W = W.contiguous()
        out = torch_matmul(X, W.t(), out = out)
    elif W.dtype == torch.float8_e4m3fn:
        out = fp8_linear(X, W, W_quant)
    else:
        W = fast_dequantize(W, W_quant, use_global_buffer = True)
        out = torch_matmul(X, W.t(), out = out)
    if W_quant is not None:
        del W

    if A is not None:
        # LoRA is enabled
        A, B = A.t(), B.t()
        XA = torch_matmul(X, A.to(dtype))
        out.addmm_(XA, B.to(dtype), alpha = s)
        # out += (X @ A.to(dtype)) @ (s * B.to(dtype))

    return out.view(batch, seq_len, -1) if reshape else out
