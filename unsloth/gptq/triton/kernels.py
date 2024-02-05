import itertools
from logging import getLogger

import torch
import triton
import triton.language as tl

logger = getLogger(__name__)


def dequant_ref(qstate):
    # assert bits == 4, "Only 4-bit quantization is supported"
    qweight, scales, qzeros, wf, g_idx, bits = (
        qstate.qweight,
        qstate.scales,
        qstate.qzeros,
        qstate.wf,
        qstate.g_idx,
        qstate.bits,
    )

    zeros = torch.bitwise_right_shift(
        torch.unsqueeze(qzeros, 2).expand(-1, -1, 32 // bits), wf.unsqueeze(0)
    ).to(torch.int16 if bits == 8 else torch.int8)
    zeros = torch.bitwise_and(zeros, (2**bits) - 1)

    zeros = zeros + 1
    zeros = zeros.reshape(scales.shape)

    weights = torch.bitwise_right_shift(
        torch.unsqueeze(qweight, 1).expand(-1, 32 // bits, -1), wf.unsqueeze(-1)
    ).to(torch.int16 if bits == 8 else torch.int8)
    weights = torch.bitwise_and(weights, (2**bits) - 1)
    weights = weights.reshape(weights.shape[0] * weights.shape[1], weights.shape[2])
    weights = scales[g_idx] * (weights - zeros[g_idx])
    return weights


def make_dequant_configs(block_sizes, num_warps):
    configs = []
    for bs, ws in itertools.product(block_sizes, num_warps):
        configs.append(triton.Config({"X_BLOCK": bs}, num_warps=ws))
    return configs


DEFAULT_DEQUANT_CONFIGS = make_dequant_configs([128, 256, 512, 1024], [4, 8])


@triton.autotune(DEFAULT_DEQUANT_CONFIGS, key=["numels"])
@triton.jit
def dequant_kernel_248(
    g_idx_ptr,
    scales_ptr,
    qweight_ptr,
    qzeros_ptr,
    out_ptr,
    numels,
    maxq: tl.constexpr,
    bits: tl.constexpr,
    outfeatures: tl.constexpr,
    num_groups: tl.constexpr,
    X_BLOCK: tl.constexpr = 1024,
):
    # Block indexing
    xoffset = tl.program_id(0) * X_BLOCK
    x_index = xoffset + tl.arange(0, X_BLOCK)
    xmask = x_index < numels
    row_idx = x_index // outfeatures
    col_idx = x_index % outfeatures

    elements_per_feature: tl.constexpr = 32 // bits

    # Load parameters
    g_idx = tl.load(g_idx_ptr + (row_idx), None, eviction_policy="evict_last")
    qweights = tl.load(
        qweight_ptr + (col_idx + (outfeatures * (row_idx // elements_per_feature))),
        None,
    )

    wf_weights = (row_idx % elements_per_feature) * bits

    wf_zeros = (col_idx % elements_per_feature) * bits

    tmp1 = g_idx + num_groups
    tmp2 = g_idx < 0
    tl.device_assert(g_idx >= 0, "index out of bounds: 0 <= tmp0 < 0")
    groups = tl.where(tmp2, tmp1, g_idx)  # tmp3 are g_idx

    scales = tl.load(scales_ptr + (col_idx + (outfeatures * groups)), None).to(
        tl.float32
    )

    # Unpack weights
    weights = qweights >> wf_weights  # bit shift qweight

    weights = weights & maxq

    # Unpack zeros
    qzero_ncols: tl.constexpr = outfeatures // elements_per_feature
    qzeros = tl.load(
        qzeros_ptr + ((qzero_ncols * groups) + (col_idx // elements_per_feature)),
        None,
        eviction_policy="evict_last",
    )
    zeros = qzeros >> wf_zeros
    zeros = zeros & maxq

    # Dequantize
    zeros = zeros + 1
    weights = weights - zeros
    weights = weights.to(tl.float32)
    weights = scales * weights

    tl.store(out_ptr + (x_index), weights, mask=xmask)


def dequant248(qweight, scales, qzeros, g_idx, bits, maxq=None):
    """Launcher for triton dequant kernel
    Only valid for bits = 2, 4, 8

    """

    assert bits in [2, 4, 8], "Only 2, 4, 8-bit GPTQ quantization is supported"
    num_groups = scales.shape[0]
    outfeatures = scales.shape[1]
    infeatures = g_idx.shape[0]

    out = torch.empty((infeatures, outfeatures), device="cuda", dtype=torch.float16)
    numels = out.numel()
    maxq = 2**bits - 1 if maxq is None else maxq
    grid = lambda meta: (triton.cdiv(numels, meta["X_BLOCK"]),)

    dequant_kernel_248[grid](
        g_idx,
        scales,
        qweight,
        qzeros,
        out,
        numels,
        maxq=maxq,
        bits=bits,
        outfeatures=outfeatures,
        num_groups=num_groups,
    )
    return out
