import itertools
from types import MethodType

import torch
from dequant import hqq_dequant
from hqq.core.quantize import HQQBackend, HQQLinear, Quantizer, hqq_base_quant_config
from tabulate import tabulate
from triton.testing import do_bench
from utils import patch_hqq_packing

torch.manual_seed(0)

SHAPES = [(4096, 4096)]  # , (4096, 4096), (4096, 11008)]
AXES = [1, 0]
GROUP_SIZES = [64, 128]
NBITS = [4]
DTYPES = [torch.bfloat16]
QUANT_SCALES = [False]
QUANT_ZEROS = [False, True]
NUM_ITERS = [10]
BLOCK_SIZES = [32, 64, 128, 256, 512, 1024, "autotune"]
device = "cuda"
backend = HQQBackend.PYTORCH
HQQLinear.set_backend(backend)


BENCH_CONFIGS = list(
    itertools.product(
        SHAPES,
        AXES,
        GROUP_SIZES,
        NBITS,
        DTYPES,
        QUANT_SCALES,
        QUANT_ZEROS,
        BLOCK_SIZES,
        NUM_ITERS,
    )
)


def run_bench(fn, num_iters):
    times = do_bench(lambda: [fn() for _ in range(num_iters)])
    return times


data = []

for (
    shape,
    axis,
    group_size,
    nbits,
    dtype,
    quant_scale,
    quant_zero,
    block_size,
    num_iters,
) in BENCH_CONFIGS[:]:
    if backend == HQQBackend.ATEN and axis == 1:
        continue
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
    if quant_zero:
        zero_q = hqq_linear.meta["zero_q"]
        meta_zero = hqq_linear.meta["meta_zero"]
        z_ref = Quantizer.dequantize(zero_q, meta_zero)

    HEADERS = [
        "kernel",
        "shape",
        "axis",
        "group_size",
        "nbits",
        "dtype",
        "quant_scale",
        "quant_zero",
        "block_size",
        f"hqq({backend})",
        "triton",
        "speedup",
    ]

    common_args = [
        shape,
        axis,
        group_size,
        nbits,
        dtype,
        quant_scale,
        quant_zero,
    ]

    ref_fn = lambda: hqq_linear.dequantize()
    ref_t = run_bench(ref_fn, num_iters)

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
        block_size = min(block_size, num_groups) if not autotune else None

        test_fn = lambda: hqq_dequant(
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
            block_size=block_size,
            autotune=autotune,
        )

        test_t = run_bench(test_fn, num_iters=num_iters)
        data.append(
            [
                "triton",
                *common_args,
                "autotune" if autotune else block_size,
                ref_t,
                test_t,
                f"{ref_t / test_t:.2f}x",
            ]
        )

print(tabulate(data, headers=HEADERS, floatfmt=".4f"))
