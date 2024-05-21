import itertools

import pytest
import torch
from hqq.core.quantize import HQQBackend, HQQLinear, hqq_base_quant_config

from .dequant import hqq_dequant
from .utils import patch_hqq_packing

torch.manual_seed(0)

SHAPES = [(128, 128)]  # , (4096, 4096), (4096, 11008)]
AXES = [1, 0]
GROUP_SIZES = [64]  # , 128]
NBITS = [8, 4]
DTYPES = [torch.bfloat16, torch.float32, torch.float16]  # , torch.bfloat16]
QUANT_SCALES = [False]
QUANT_ZEROS = [False, True]  # , True]
BLOCK_SIZES = [32, 128, 1024, "autotune"]
device = "cuda"
HQQLinear.set_backend(HQQBackend.PYTORCH)


ALL_TEST_CONFIGS = list(
    itertools.product(
        SHAPES, AXES, GROUP_SIZES, NBITS, DTYPES, QUANT_SCALES, QUANT_ZEROS, BLOCK_SIZES
    )
)

TESTS = ALL_TEST_CONFIGS[:]


@pytest.mark.parametrize(
    "shape, axis, group_size, nbits, dtype, quant_scale, quant_zero, block_size",
    TESTS,
    ids=lambda arg: str(arg),
)
def test_hqq_dequant(
    shape, axis, group_size, nbits, dtype, quant_scale, quant_zero, block_size
):
    M, N = shape
    linear = torch.nn.Linear(M, N, dtype=dtype, device=device)

    quant_cfg = hqq_base_quant_config(
        nbits=nbits, group_size=group_size, axis=axis, quant_zero=quant_zero
    )
    hqq_linear = HQQLinear(
        linear,
        device=device,
        quant_config=quant_cfg,
        compute_dtype=dtype,
        del_orig=False,
    )
    dq_ref = hqq_linear.dequantize()

    with patch_hqq_packing():
        hqq_linear = HQQLinear(
            linear,
            device=device,
            quant_config=quant_cfg,
            compute_dtype=dtype,
            del_orig=False,
        )
        q = hqq_linear.W_q

        meta = hqq_linear.meta
        output_shape = meta["shape"]

        scales = meta["scale"].view(-1)
        if quant_zero:
            zeros = meta["zero_q"].view(-1)
            meta_zero = meta["meta_zero"]
            qz_scale = meta_zero["scale"]
            qz_zero = meta_zero["zero"]
        else:
            zeros = meta["zero"].view(-1)
            qz_scale, qz_zero = None, None

        num_groups = len(scales)

        assert num_groups == (M * N) // group_size

        autotune = block_size == "autotune"
        dq = hqq_dequant(
            q,
            scales=scales,
            zeros=zeros,
            group_size=group_size,
            nbits=nbits,
            axis=axis,
            output_shape=output_shape,
            quant_zero=quant_zero,
            qz_scales=qz_scale,
            qz_zeros=qz_zero,
            autotune=autotune,
            block_size=block_size if not autotune else None,
        )

        if dtype == torch.bfloat16:
            atol, rtol = 1e-3, 1e-4
        elif dtype == torch.float16:
            atol, rtol = 1e-3, 1e-5
        else:
            atol, rtol = 1e-5, 1e-5
        assert torch.allclose(dq, dq_ref, atol=atol, rtol=rtol)
