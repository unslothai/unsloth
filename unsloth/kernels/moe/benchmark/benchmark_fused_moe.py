import argparse
import time
from contextlib import nullcontext

import torch
from transformers import AutoConfig
from transformers.models.llama4 import Llama4TextConfig
from transformers.models.llama4.modeling_llama4 import Llama4TextMoe
from transformers.models.qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from triton.testing import do_bench
from utils import (
    create_kernel_configs,
    get_autotuner,
    post_process_results,
    postprocess_autotune_results,
    save_results,
)

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
from grouped_gemm.reference.layers.llama4_moe import Llama4TritonTextMoe
from grouped_gemm.reference.layers.qwen3_moe import Qwen3MoeFusedGroupedGEMMBlock

SEED = 42
LLAMA4_ID = "meta-llama/Llama-4-Scout-17B-16E"
QWEN3_MODEL_ID = "Qwen/Qwen3-30B-A3B"


def run_benchmark_forward(
    ref_model: torch.nn.Module,
    tt_model: torch.nn.Module,
    config: AutoConfig,
    seqlen: int,
    dtype: torch.dtype,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    bs: int = 1,
):
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size

    X = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True
    )

    # Forward
    bench_forward_ref = lambda: ref_model(X)  # noqa: E731
    bench_forward_fused = lambda: tt_model(X)  # noqa: E731

    ref_forward_time = do_bench(bench_forward_ref)

    if not autotune:
        assert kernel_config_fwd is not None
        tuning_context = TritonTuningContext(kernel_config_fwd)
    else:
        tuning_context = nullcontext()

    with tuning_context:
        fused_forward_time = do_bench(bench_forward_fused)

    if (not autotune) and (not tuning_context.success):
        return 0, 1

    print(
        f"Forward: ref {ref_forward_time:.4f}, fused {fused_forward_time:.4f}, speedup {ref_forward_time / fused_forward_time:.1f}x"
    )
    return ref_forward_time, fused_forward_time


def run_benchmark_backward(
    ref_model: torch.nn.Module,
    tt_model: torch.nn.Module,
    config: AutoConfig,
    seqlen: int,
    dtype: torch.dtype,
    bs=1,
):
    torch.manual_seed(
        SEED
    )  # Should not be needed when running using pytest -- autouse fixture in conftest.py
    device = "cuda"
    hidden_size = config.hidden_size

    X = torch.randn(
        bs, seqlen, hidden_size, dtype=dtype, device=device, requires_grad=True
    )
    X_test = X.detach().clone().requires_grad_(True)

    output, _ = ref_model(X)

    # Prevent autotuning forward pass
    from grouped_gemm.kernels.forward import _autotuned_grouped_gemm_forward_kernel

    _autotuned_grouped_gemm_forward_kernel.configs = (
        _autotuned_grouped_gemm_forward_kernel.configs[:20]
    )
    test_output, _ = tt_model(X_test)

    # Bench
    grad_output = torch.randn_like(output)
    bench_backward_ref = lambda: output.backward(grad_output, retain_graph=True)  # noqa: E731
    bench_backward_fused = lambda: test_output.backward(grad_output, retain_graph=True)  # noqa: E731

    ref_backward_time = do_bench(
        bench_backward_ref, grad_to_none=[X, *ref_model.parameters()]
    )
    fused_backward_time = do_bench(
        bench_backward_fused, grad_to_none=[X_test, *tt_model.parameters()]
    )
    print(
        f"Backward: ref {ref_backward_time:.4f}, fused {fused_backward_time:.4f}, speedup {ref_backward_time / fused_backward_time:.1f}x"
    )
    return ref_backward_time, fused_backward_time


def setup_model(
    config: Qwen3MoeConfig | Llama4TextConfig,
    dtype,
    permute_x,
    permute_y,
    autotune,
    kernel_config_fwd,
    kernel_config_bwd_dW,
    kernel_config_bwd_dX,
    dX_only=False,
    dW_only=False,
    overlap_router_shared=False,
    device="cuda",
):
    if isinstance(config, Qwen3MoeConfig):
        ref_model = Qwen3MoeSparseMoeBlock(config).to(device, dtype)

        # Triton kernel grouped gemm version of MoE Block -- this is what we're testing
        tt_model = Qwen3MoeFusedGroupedGEMMBlock.from_hf(
            ref_model,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dX_only=dX_only,
            dW_only=dW_only,
        ).to(device, dtype)

    elif isinstance(config, Llama4TextConfig):
        ref_model = Llama4TextMoe(config).to(device, dtype)
        tt_model = Llama4TritonTextMoe(
            config,
            overlap_router_shared=overlap_router_shared,
            permute_x=permute_x,
            permute_y=permute_y,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
            kernel_config_bwd_dW=kernel_config_bwd_dW,
            kernel_config_bwd_dX=kernel_config_bwd_dX,
            dX_only=dX_only,
            dW_only=dW_only,
        ).to(device, dtype)

    else:
        raise ValueError(f"Unrecognized config {type(config).__name__}")

    return ref_model, tt_model


def run_benchmark(
    mode: str,
    model_config: Qwen3MoeConfig | Llama4TextConfig,
    seqlen: int,
    dtype: torch.dtype,
    permute_x: bool,
    permute_y: bool,
    autotune: bool,
    kernel_config_fwd: KernelConfigForward = None,
    kernel_config_bwd_dW: KernelConfigBackward_dW = None,
    kernel_config_bwd_dX: KernelConfigBackward_dX = None,
    overlap_router_shared: bool = False,
    results_dir: str = None,
):
    if autotune:
        autotuner = get_autotuner(mode)
    if mode == "dW":
        dW_only = True
    elif mode == "dX":
        dX_only = True
    else:
        dW_only = dX_only = False

    ref_model, tt_model = setup_model(
        model_config,
        dtype=dtype,
        permute_x=permute_x,
        permute_y=permute_y,
        autotune=autotune,
        kernel_config_fwd=kernel_config_fwd,
        kernel_config_bwd_dW=kernel_config_bwd_dW,
        kernel_config_bwd_dX=kernel_config_bwd_dX,
        dX_only=dX_only,
        dW_only=dW_only,
        overlap_router_shared=overlap_router_shared,
    )

    if mode == "forward":
        ref_time, fused_time = run_benchmark_forward(
            ref_model,
            tt_model,
            config=model_config,
            seqlen=seqlen,
            dtype=dtype,
            autotune=autotune,
            kernel_config_fwd=kernel_config_fwd,
        )
    else:
        ref_time, fused_time = run_benchmark_backward(
            ref_model, tt_model, config=model_config, seqlen=seqlen, dtype=dtype
        )

    if autotune:
        if mode == "backward":
            autotuner_dW, autotuner_dX = autotuner
            postprocess_autotune_results(
                autotuner_dW, "dW", ref_time, fused_time, results_dir
            )
            postprocess_autotune_results(
                autotuner_dX, "dX", ref_time, fused_time, results_dir
            )
        else:
            postprocess_autotune_results(
                autotuner, mode, ref_time, fused_time, results_dir
            )

    return ref_time, fused_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="benchmark_results")
    parser.add_argument("--model", type=str, choices=["llama4", "qwen3"], required=True)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument(
        "--dtype", type=str, choices=["bfloat16", "float16"], default="bfloat16"
    )
    parser.add_argument("--permute_x", action="store_true")
    parser.add_argument("--permute_y", action="store_true")
    parser.add_argument("--autotune", action="store_true")
    parser.add_argument("--overlap_router_shared", action="store_true")
    parser.add_argument(
        "--BLOCK_SIZE_M",
        nargs=2,
        type=int,
        default=[DEFAULT_M_BLOCK_SIZES[0], DEFAULT_M_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--BLOCK_SIZE_N",
        nargs=2,
        type=int,
        default=[DEFAULT_N_BLOCK_SIZES[0], DEFAULT_N_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--BLOCK_SIZE_K",
        nargs=2,
        type=int,
        default=[DEFAULT_K_BLOCK_SIZES[0], DEFAULT_K_BLOCK_SIZES[-1]],
    )
    parser.add_argument(
        "--num_warps",
        nargs=2,
        type=int,
        default=[DEFAULT_NUM_WARPS[0], DEFAULT_NUM_WARPS[-1]],
    )
    parser.add_argument(
        "--num_stages",
        nargs=2,
        type=int,
        default=[DEFAULT_NUM_STAGES[0], DEFAULT_NUM_STAGES[-1]],
    )
    parser.add_argument(
        "--use_tma_load_w", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--use_tma_load_x", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--use_tma_load_dy", action="store_true"
    )  # No need to specify, will automatically parametrize these for each kernel config
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "backward", "dW", "dX"],
        default="forward",
    )
    args = parser.parse_args()
    args.dtype = getattr(torch, args.dtype)

    model_id = QWEN3_MODEL_ID if args.model == "qwen3" else LLAMA4_ID
    model_config = AutoConfig.from_pretrained(model_id)
    model_config = model_config.text_config if args.model == "llama4" else model_config

    mode = args.mode

    if args.autotune:
        # logging.basicConfig(level=logging.INFO)
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
            overlap_router_shared=args.overlap_router_shared,
            results_dir=args.results_dir,
        )
        end_time = time.time()
        print(f"Total time: {end_time - start_time:.4f} seconds")

    # NOTE: better to use autotuner for now, since the MoE block needs 2 different kernel configs for forward (2 grouped gemms, gate_up_proj and down_proj)
    # and the backward pass needs 4 different kernel configs (2 grouped gemms each for dW and dX)
    # The benchmark only supports 1 kernel config at a time so the same config will be used for both grouped gemms, which is suboptimal.
    else:
        assert False, "Use autotune for now"
        kernel_configs = create_kernel_configs(args, args.permute_x, args.permute_y)
        print(f"Running {len(kernel_configs)} kernel configs")
        default_kernel_config_fwd = KernelConfigForward(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
        default_kernel_config_bwd_dW = KernelConfigBackward_dW(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
        default_kernel_config_bwd_dX = KernelConfigBackward_dX(
            permute_x=args.permute_x, permute_y=args.permute_y
        )
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
            results.append(
                KernelResult(
                    torch_time=ref_time,
                    triton_time=fused_time,
                    speedup=ref_time / fused_time,
                    kernel_config=kernel_config,
                )
            )
        df = post_process_results(
            results, args.mode, args.seqlen, args.dtype, args.autotune
        )
        save_results(
            df, args.results_dir, args.mode, args.seqlen, args.dtype, args.autotune
        )
