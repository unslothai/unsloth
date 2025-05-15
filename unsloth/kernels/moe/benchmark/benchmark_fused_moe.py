import argparse
import time

import torch
from grouped_gemm.kernels.autotuning import (
    DEFAULT_K_BLOCK_SIZES,
    DEFAULT_M_BLOCK_SIZES,
    DEFAULT_N_BLOCK_SIZES,
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
)
from grouped_gemm.kernels.tuning import (
    KernelConfigBackward_dW,
    KernelConfigBackward_dX,
    KernelConfigForward,
    KernelResult,
    TritonTuningContext,
)
from grouped_gemm.reference.moe_block import Qwen3MoeFusedGroupedGEMMBlock
from transformers import AutoConfig
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from triton.testing import do_bench
from utils import (
    create_kernel_configs,
    post_process_results,
    save_results,
)

SEED = 42

def run_benchmark_forward(
    config: Qwen3MoeConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
):
    torch.manual_seed(SEED)  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size
    bs = 1

    # Reference op -- HF
    moe_block = Qwen3MoeSparseMoeBlock(config).to(device, dtype)

    # Triton kernel grouped gemm version of MoE Block -- this is what we're testing
    fused_gemm_block = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
        moe_block,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
    ).to(device, dtype)
    X = torch.randn(bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True)

    ref_output, _ = moe_block(X)
    # Forward
    bench_forward_ref = lambda: moe_block(X)
    bench_forward_fused = lambda: fused_gemm_block(X)

    ref_forward_time = do_bench(bench_forward_ref)
    with TritonTuningContext(kernel_config_fwd) as ctx:
        fused_forward_time = do_bench(bench_forward_fused)
    
    if not ctx.success:
        return 0, 1

    print(
        f"Forward: ref {ref_forward_time:.4f}, fused {fused_forward_time:.4f}, speedup {ref_forward_time / fused_forward_time:.1f}x"
    )
    return ref_forward_time, fused_forward_time

def run_benchmark_backward(
    config: Qwen3MoeConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    dX_only: bool = False,
    dW_only: bool = False,
):
    torch.manual_seed(SEED)  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size
    bs = 1

    # Reference op -- HF
    moe_block = Qwen3MoeSparseMoeBlock(config).to(device, dtype)

    # Triton kernel grouped gemm version of MoE Block -- this is what we're testing
    fused_gemm_block = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
        moe_block,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
        dX_only=dX_only,
        dW_only=dW_only,
    ).to(device, dtype)

    X = torch.randn(bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True)
    X_test = X.detach().clone().requires_grad_(True)
    
    output, _ = moe_block(X)

    # Prevent autotuning forward pass
    from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel
    _autotuned_grouped_gemm_forward_kernel.configs = _autotuned_grouped_gemm_forward_kernel.configs[:20]
    test_output, _ = fused_gemm_block(X_test)

    # Bench
    grad_output = torch.randn_like(output)
    bench_backward_ref = lambda: output.backward(grad_output, retain_graph=True) # noqa: E731
    bench_backward_fused = lambda: test_output.backward(grad_output, retain_graph=True) # noqa: E731

    ref_backward_time = do_bench(bench_backward_ref, grad_to_none=[X, *moe_block.parameters()])
    fused_backward_time = do_bench(bench_backward_fused, grad_to_none=[X_test, *fused_gemm_block.parameters()])
    print(
        f"Backward: ref {ref_backward_time:.4f}, fused {fused_backward_time:.4f}, speedup {ref_backward_time / fused_backward_time:.1f}x"
    )
    return ref_backward_time, fused_backward_time


def run_benchmark(
    mode: str,
    model_config: Qwen3MoeConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
):
    
    if mode == "forward":
        
        ref_time, fused_time = run_benchmark_forward(
            model_config,
            seqlen,
            dtype,
            permute_x,
            permute_y,
            autotune,
            kernel_config_fwd,
            kernel_config_bwd_dW,
            kernel_config_bwd_dX,
        )
    elif mode == "dW":
        ref_time, fused_time = run_benchmark_backward(
            model_config,
            seqlen,
            dtype,
            permute_x,
            permute_y,
            autotune,
            kernel_config_fwd,
            kernel_config_bwd_dW,
            kernel_config_bwd_dX,
            dW_only=True,
        )
    elif mode == "dX":
        ref_time, fused_time = run_benchmark_backward(
            model_config,
            seqlen,
            dtype,
            permute_x,
            permute_y,
            autotune,
            kernel_config_fwd,
            kernel_config_bwd_dW,
            kernel_config_bwd_dX,
            dX_only=True,
        )
    elif mode == "backward":
        ref_time, fused_time = run_benchmark_backward(
            model_config,
            seqlen,
            dtype,
            permute_x,
            permute_y,
            autotune,
            kernel_config_fwd,
            kernel_config_bwd_dW,
            kernel_config_bwd_dX,
            dX_only=False,
            dW_only=False,
        )

    return ref_time, fused_time

# NOTE: better to use autotuner for now, since the MoE block needs 2 different kernel configs for forward (2 grouped gemms, gate_up_proj and down_proj)
# and the backward pass needs 4 different kernel configs (2 grouped gemms each for dW and dX)
# The benchmark only supports 1 kernel config at a time so the same config will be used for both grouped gemms, which is suboptimal.

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16")
    parser.add_argument("--permute_x", action="store_true")
    parser.add_argument("--permute_y", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--BLOCK_SIZE_M", nargs=2, type=int, default=[DEFAULT_M_BLOCK_SIZES[0], DEFAULT_M_BLOCK_SIZES[-1]])
    parser.add_argument("--BLOCK_SIZE_N", nargs=2, type=int, default=[DEFAULT_N_BLOCK_SIZES[0], DEFAULT_N_BLOCK_SIZES[-1]])
    parser.add_argument("--BLOCK_SIZE_K", nargs=2, type=int, default=[DEFAULT_K_BLOCK_SIZES[0], DEFAULT_K_BLOCK_SIZES[-1]])
    parser.add_argument("--num_warps", nargs=2, type=int, default=[DEFAULT_NUM_WARPS[0], DEFAULT_NUM_WARPS[-1]])
    parser.add_argument("--num_stages", nargs=2, type=int, default=[DEFAULT_NUM_STAGES[0], DEFAULT_NUM_STAGES[-1]])
    parser.add_argument("--use_tma_load_w", action="store_true") # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument("--use_tma_load_x", action="store_true") # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument("--use_tma_load_dy", action="store_true") # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument("--mode", type=str, choices=["forward", "backward", "dW", "dX"], default="forward")
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    model_id = "Qwen/Qwen3-30B-A3B"
    model_config = AutoConfig.from_pretrained(model_id)

    mode = args.mode

    if args.autotune:
        print(
            f"Benchmarking {model_id} {mode}: seqlen={args.seqlen}, dtype={args.dtype}, permute_x={args.permute_x}, permute_y={args.permute_y}, autotune"
        )
        start_time = time.time()
        ref_time, fused_time = run_benchmark(
            args.mode,
            model_config,
            seqlen=args.seqlen,
            dtype=args.dtype,
            permute_x=args.permute_x,
            permute_y=args.permute_y,
            autotune=args.autotune,
        )
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.4f} seconds")

    else:
        kernel_configs = create_kernel_configs(args, args.permute_x, args.permute_y)
        print(f"Running {len(kernel_configs)} kernel configs")
        default_kernel_config_fwd = KernelConfigForward(permute_x=args.permute_x, permute_y=args.permute_y)
        default_kernel_config_bwd_dW = KernelConfigBackward_dW(permute_x=args.permute_x, permute_y=args.permute_y)
        default_kernel_config_bwd_dX = KernelConfigBackward_dX(permute_x=args.permute_x, permute_y=args.permute_y)
        results = []
        for kernel_config in kernel_configs:
            if args.mode == "forward":
                kernel_config_fwd = kernel_config
                kernel_config_bwd_dW = default_kernel_config_bwd_dW
                kernel_config_bwd_dX = default_kernel_config_bwd_dX
            elif args.mode == "dW":
                kernel_config_fwd = default_kernel_config_fwd
                kernel_config_bwd_dW = kernel_config
                kernel_config_bwd_dX = default_kernel_config_bwd_dX
            elif args.mode == "dX":
                kernel_config_fwd = default_kernel_config_fwd
                kernel_config_bwd_dW = default_kernel_config_bwd_dW
                kernel_config_bwd_dX = kernel_config
            else:
                raise ValueError(f"Invalid mode: {args.mode}")
            print(
                f"Benchmarking {model_id} {args.mode} with seqlen={args.seqlen}, dtype={args.dtype}, permute_x={args.permute_x}, permute_y={args.permute_y}, kernel_config_fwd={kernel_config_fwd}, kernel_config_bwd_dW={kernel_config_bwd_dW}, kernel_config_bwd_dX={kernel_config_bwd_dX}"
            )
            
            ref_time, fused_time = run_benchmark(
                args.mode,
                model_config,
                seqlen=args.seqlen,
                dtype=args.dtype,
                permute_x=kernel_config.permute_x,
                permute_y=kernel_config.permute_y,
                autotune=False,
                kernel_config_fwd=kernel_config_fwd,
                kernel_config_bwd_dW=kernel_config_bwd_dW,
                kernel_config_bwd_dX=kernel_config_bwd_dX,
            )
            results.append(KernelResult(
                torch_time=ref_time,
                triton_time=fused_time,
                speedup=ref_time / fused_time,
                kernel_config=kernel_config,
            ))
        df = post_process_results(results, args.mode, args.seqlen, args.dtype, args.autotune)
        save_results(df, args.results_dir, args.mode, args.seqlen, args.dtype, args.autotune)
