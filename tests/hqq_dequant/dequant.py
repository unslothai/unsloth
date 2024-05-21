import itertools
import sys

import torch
import triton
import triton.language as tl
from triton import next_power_of_2

from .pack import pack, unpack


@triton.jit
def _dequant_kernel(
    q_ptr,
    scales_ptr,
    zeros_ptr,
    output_ptr,
    stride_qm,
    stride_qn,
    stride_dqm,
    stride_dqn,
    num_groups,  # used in heuristics
    BLOCK_SIZE: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    NBITS: tl.constexpr,
    AXIS: tl.constexpr,
    QUANT_ZERO: tl.constexpr = False,
    qz_scale_ptr=None,
    qz_zero_ptr=None,
    IS_BFLOAT16: tl.constexpr = False,
):
    pid = tl.program_id(0)

    if NBITS == 4:
        # Offsets for loading q
        if AXIS == 1:
            within_group_offsets = tl.arange(0, GROUP_SIZE)
            across_group_offset = pid * (BLOCK_SIZE // 2)
            across_group_offsets = across_group_offset + tl.arange(0, BLOCK_SIZE // 2)

            q_offsets_m = across_group_offsets * stride_qm
            q_offsets_n = within_group_offsets * stride_qn
        else:
            within_group_offsets = tl.arange(0, GROUP_SIZE // 2)
            across_group_offset = pid * (BLOCK_SIZE)
            across_group_offsets = across_group_offset + tl.arange(0, BLOCK_SIZE)

            q_offsets_m = within_group_offsets * stride_qm
            q_offsets_n = across_group_offsets * stride_qn

        q = tl.load(q_ptr + q_offsets_m[:, None] + q_offsets_n[None, :])

        # Reset offsets after packed loading to full size for loading scales / zeros and storing dequantized values
        if AXIS == 1:
            across_group_offset = pid * BLOCK_SIZE
            across_group_offsets = across_group_offset + tl.arange(0, BLOCK_SIZE)
        else:
            within_group_offsets = tl.arange(0, GROUP_SIZE)

        # Load scales / zeros
        scales = tl.load(scales_ptr + across_group_offsets)
        zeros = tl.load(zeros_ptr + across_group_offsets)
    elif NBITS == 8:
        within_group_offsets = tl.arange(0, GROUP_SIZE)
        across_group_offset = pid * BLOCK_SIZE
        across_group_offsets = across_group_offset + tl.arange(0, BLOCK_SIZE)

        # Load q
        if AXIS == 1:
            q_offsets_m = across_group_offsets * stride_qm
            q_offsets_n = within_group_offsets * stride_qn
        else:
            q_offsets_m = within_group_offsets * stride_qm
            q_offsets_n = across_group_offsets * stride_qn

        q = tl.load(q_ptr + q_offsets_m[:, None] + q_offsets_n[None, :])

        # Load scales
        scales = tl.load(scales_ptr + across_group_offsets)
        zeros = tl.load(zeros_ptr + across_group_offsets)
    else:
        tl.static_assert(False, "Only NBITS = {4, 8} supported for now")

    if QUANT_ZERO:
        # Only scalar qz_scale and qz_zero for now
        qz_scale = tl.load(qz_scale_ptr)
        qz_zero = tl.load(qz_zero_ptr)
        zeros = (zeros - qz_zero) * qz_scale

    # Unpack
    if NBITS == 4:
        # Unpack qweights -- h/t jlebar!
        _4_i8 = tl.full((1,), 4, dtype=tl.int8)
        q_lo = (q << _4_i8) >> _4_i8
        q_hi = q >> _4_i8

        # Problems with direct convert to bfloat16
        if IS_BFLOAT16:
            q_lo = q_lo.to(tl.float16)
            q_hi = q_hi.to(tl.float16)

        q = tl.join(
            q_lo.to(scales_ptr.dtype.element_ty),
            q_hi.to(scales_ptr.dtype.element_ty),
        ).permute(0, 2, 1)

        if AXIS == 1:
            q = tl.reshape(q, BLOCK_SIZE, GROUP_SIZE)
        else:
            q = tl.reshape(q, GROUP_SIZE, BLOCK_SIZE)

    # Compute
    # Broadcast across grouping dimension
    if AXIS == 1:
        zeros = zeros[:, None]
        scales = scales[:, None]
    else:
        zeros = zeros[None, :]
        scales = scales[None, :]

    dq = (q - zeros) * scales

    # Store
    if AXIS == 1:
        # Write to [num_groups, group_size] tensor, intra-group along axis 1, inter-group agross axis 0
        dq_offsets_m = across_group_offsets * stride_dqm
        dq_offsets_n = within_group_offsets * stride_dqn
    else:
        # Write out to [group_size, num_groups], intra-group along axis 0, inter-group along axis 1
        dq_offsets_m = within_group_offsets * stride_dqm
        dq_offsets_n = across_group_offsets * stride_dqn

    tl.store(output_ptr + dq_offsets_m[:, None] + dq_offsets_n[None, :], dq)


def get_autotune_configs():
    configs = []

    for block_size in [32, 64, 128, 256, 512, 1024]:
        for num_warps in [2, 4, 8]:
            configs.append(
                triton.Config({"BLOCK_SIZE": block_size}, num_warps=num_warps)
            )

    return configs


HEURISTICS = {
    "BLOCK_SIZE": lambda args: min(args["num_groups"], args["BLOCK_SIZE"]),
}

_base_kernel = _dequant_kernel
_checked_kernel = triton.heuristics(values=HEURISTICS)(_base_kernel)
_autotuned_kernel = triton.autotune(configs=get_autotune_configs(), key=["num_groups"])(
    _checked_kernel
)


def hqq_dequant(
    q,
    scales,
    zeros,
    group_size,
    nbits,
    axis,
    output_shape,
    quant_zero=False,
    qz_scales=None,
    qz_zeros=None,
    compute_dtype=None,
    block_size=None,
    autotune=True,
):
    """
    Stand-alone triton `hqq` dequantization kernel
    - Handles both axis == 0 and axis == 1
    - Currently, only supports the `hqq_base_quant_config` [settings](https://github.com/mobiusml/hqq/blob/aad68687e042ed628b5a655969406d501a203949/hqq/core/quantize.py#L872) for `quant_scale` and `quant_zero`:
        - no quantization for scales
        - quant_zero: `channelwise=False`, `group_size=None`, nbits=8 -- that is single scale and single zero

    Assuming a final quantized tensor shape of `M x N`:
    - launches a 1-D grid of shape `M // BLOCK_SIZE` for `axis = 1` or `N // BLOCK_SIZE` for `axis = 0`,
    with `N = GROUP_SIZE` or `M = GROUP_SIZE`, respectively, where `GROUP_SIZE` is the grouping used for quantization.

    For `axis = 1`:
        q is a 2-D `M x N` tensor viewed as `num_groups x group_size` where `num_groups == M * N // group_size`

        Each block
        - Loads:
            - q: `BLOCK_SIZE` number of groups along `axis = 0` and `GROUP_SIZE` elements along `axis=1`
            - scales / zeros:`BLOCK_SIZE` number of scales / zeros
        - Computes:
            - `BLOCK_M x GROUP_SIZE` dequantized elements, where scales and zeros are broadcasted to `GROUP_SIZE` along `axis = 1`

    For `axis = 0`:
        q is a 2-D `M x N` tensor viewed as `GROUP_SIZE x num_groups` where `num_groups == M * N // GROUP_SIZE`

        Grouped elements are no longer contiguous in this case since stride == `num_groups` along `axis = 0`.

        Each block
        - Loads:
            - q: `GROUP_SIZE x BLOCK_SIZE`
            - `BLOCK_SIZE` scales / zeros
        - Computes:
            - `GROUP_SIZE x BLOCK_SIZE` dequantized elements, where scales and zeros are broadcasted to `GROUP_SIZE` along `axis = 0`

        TODO:
        - Check generated `ptx` for both cases to check data access patterns.
        - Fused scales and zeros to compile to `fma.{f16,bf16x2}`
        - Add compiler hints (`multiple_of`, `max_contiguous`)
        - Add masking / predication to to handle all tile shapes or further optimize data access
        - Handle bfloat16 conversion
    Args:
        q (torch.uint8): 2-D quantized tensor, shape (num_groups, group_size) if axis ==1 else (group_size, num_groups)
        scales (torch.float16, torch.bfloat16, or torch.float32): 1-D tensor shape (num_groups,)
        zeros: 1-D tensor shape (num_groups,), if dtype is i8 then zeros are quantized, otherwise same dtype as scales
        qz_scales (scalar tensor): scale factors for quantized zeros, only relevant if zeros are quantized
        qz_zeros (scalar tensor): zero points for quantized zeros, only relevant if zeros are quantized
        axis (axis): Either 0 or 1.
        compute_dtype: dtype of the dequantized tensor. Defaults to scales.dtype

    Returns:
        torch.Tensor: 2-D dequantized tensor, shape (num_groups, group_size) if axis ==1 else (group_size, num_groups) with dtype {scales}.dtype
    """
    assert (
        block_size is not None
    ) ^ autotune, "Must specify either block_size or autotune"
    assert q.dtype in (torch.uint8, torch.int8)
    if not quant_zero:
        assert scales.dtype == zeros.dtype
    else:
        assert zeros.dtype in (torch.uint8, torch.int8)
        assert qz_scales is not None and qz_zeros is not None
        assert (
            qz_scales.shape == torch.Size([]) and qz_scales.numel() == 1
        ), "Only scalar qz_scales is supported (channelwise=False)"
        assert (
            qz_zeros.shape == torch.Size([]) and qz_zeros.numel() == 1
        ), "Only scalar qz_zeros is supported (channelwise=False)"

    assert (
        scales.shape == zeros.shape
    )  # True even if zeros are quantized, since we only handle `channelwise=False` case
    assert scales.ndim == 1
    assert zeros.ndim == 1
    assert nbits in [4, 8], "nbits must be 4 or 8"

    num_groups = len(scales)

    if nbits == 4:
        assert (
            q.shape == torch.Size([num_groups // 2, group_size])
            if axis == 1
            else q.shape == torch.Size([group_size // 2, num_groups])
        )
    else:
        assert (
            q.shape == torch.Size([num_groups, group_size])
            if axis == 1
            else q.shape == torch.Size([group_size, num_groups])
        )

    if axis == 1:
        dq_shape = (num_groups, group_size)
    else:
        dq_shape = (group_size, num_groups)

    dq_dtype = compute_dtype or scales.dtype
    dq = torch.empty(dq_shape, dtype=dq_dtype, device=q.device)

    grid = lambda meta: (triton.cdiv(num_groups, meta["BLOCK_SIZE"]),)

    # Kernel params
    args = [q, scales, zeros, dq]
    kwargs = dict(
        # M=dq.shape[0],
        # N=dq.shape[1],
        stride_qm=q.stride(0),
        stride_qn=q.stride(1),
        stride_dqm=dq.stride(0),
        stride_dqn=dq.stride(1),
        num_groups=num_groups,
        GROUP_SIZE=group_size,
        AXIS=axis,
        QUANT_ZERO=quant_zero,
        qz_scale_ptr=qz_scales,
        qz_zero_ptr=qz_zeros,
        NBITS=nbits,
        IS_BFLOAT16=dq_dtype == torch.bfloat16,
    )

    if autotune:
        kernel = _autotuned_kernel
    else:
        kernel = _checked_kernel
        kwargs.update({"BLOCK_SIZE": block_size})
        block_size = min(block_size, next_power_of_2(num_groups))

    # Launch kernel
    kernel[grid](*args, **kwargs)

    return dq.view(output_shape)


def ref_dequant(t, scales, zeros, quant_zero=True, qz_scales=None, qz_zeros=None):
    if quant_zero:
        zeros = (zeros - qz_zeros) * qz_scales

    dq = (t - zeros) * scales
    return dq


if __name__ == "__main__":
    torch.manual_seed(0)

    NBITS = [8, 4]
    AXES = [0, 1]
    QUANT_ZERO = [False, True]
    dtype = torch.bfloat16
    device = "cuda"
    quant_dtype = torch.uint8
    group_size = 8
    M = N = 16
    assert (M * N) % group_size == 0
    num_groups = (M * N) // group_size
    BLOCK_SIZES = [num_groups, num_groups // 2]

    q_min, q_max = 0, 16

    configs = itertools.product(AXES, BLOCK_SIZES, NBITS, QUANT_ZERO)
    for axis, block_size, nbits, quant_zero in configs:
        print(f"{axis=} {block_size=} {nbits=} {quant_zero=}", file=sys.stderr)
        q = torch.randint(
            low=q_min, high=q_max, size=(M, N), dtype=quant_dtype, device=device
        )
        original_shape = q.shape

        if axis == 1:
            q = q.reshape(num_groups, -1)
        else:
            q = q.reshape(-1, num_groups)

        scales = 1 / q.to(torch.float32).max(axis=axis, keepdim=True)[0].abs()
        zeros = torch.randn_like(scales)

        if quant_zero:
            _min, _max = zeros.min(), zeros.max()
            max_v = 2**8 - 1
            min_v = 0
            qz_scales = (max_v / (_max - _min)).clamp(max=2e4)
            qz_zeros = -_min * qz_scales
            zeros = (
                torch.round(zeros * qz_scales + qz_zeros)
                .clamp(min_v, max_v)
                .to(quant_dtype)
            )
            print(f"qz_scales: {qz_scales}, qz_zeros: {qz_zeros}")
        else:
            qz_scales = None
            qz_zeros = None

        print(f"q:\n{q.shape}\n{q}")
        q = pack(q, nbits=nbits)
        print(f"q_packed:\n{q.shape}\n{q}")  # \n{q}")
        if nbits == 4:
            if axis == 1:
                assert q.shape[0] == num_groups // 2
            else:
                assert q.shape[0] == group_size // 2

        q_ref = unpack(q, nbits=nbits)
        print(f"q_ref shape:\n{q_ref.shape}\n{q_ref}")

        dq_ref = ref_dequant(
            q_ref,
            scales,
            zeros,
            quant_zero=quant_zero,
            qz_scales=qz_scales,
            qz_zeros=qz_zeros,
        ).view(original_shape)

        block_size = num_groups // 2
        dq = hqq_dequant(
            q,
            scales.view(-1),
            zeros.view(-1),
            output_shape=original_shape,
            nbits=nbits,
            group_size=group_size,
            quant_zero=quant_zero,
            qz_scales=qz_scales,
            qz_zeros=qz_zeros,
            axis=axis,
            BLOCK_SIZE=block_size,
        ).view(original_shape)

        print((dq_ref - dq).abs().max(), file=sys.stderr)
