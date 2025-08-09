## MoE Grouped GEMM

Optimized implementation of `MoE MLP Block`.
Licensed under AGPLv3.

### Background

`MoE MLP` requires the following steps:
- Calculate `topk_weights` and `topk_indices`
- If using a grouped gemm implementation, calculate permutation indices needed to rearrange tokens grouped by expert
- For each expert:
    - `expert_tokens`: gather the tokens assigned to the expert
    - `first_gemm`: `gate / up proj` @ `expert_tokens`
    - `silu_and_mul`: `silu` and `mul` of `first_gemm`
    - `second_gemm`: `silu_and_mul` @ `down proj`
    - `scatter_second_gemm`: scatter the `second_gemm` to the original token order
    - `topk_weight_mul`: `second_gemm` @ `topk_weights`
    - `final_output`: if `topk > 1`, `topk_weight_mul.view(num_tokens, topk, -1).sum(dim=1)` else `topk_weight_mul`

One way to eliminate the loop is to use a grouped GEMM, where all expert GEMMs are computed within a single kernel, which iterates over tiles of the expert GEMMs as individual GEMMs, where each GEMM, the `A` matrix is `M' x K` and the `B` matrix is `K x N`, where `M'` is the number of tokens assigned to the expert and `B` is the weight matrix for that expert.

This requires an additional permute (and subsequent copy) of the hidden states such that the tokens assigned to each expert are contiguous in memory before running the first grouped GEMM within the Expert MLP.
Additionally, after the second grouped GEMM, the hidden states must be permuted back to the original token order and multiplied by `topk_weights` to get the final output.

### Optimizations
This repo implements a grouped GEMM-based MoE MLP with the following optimizations:
- Eliminates the loop over experts by performing gemms as a grouped GEMM, computing the expert gemms within a single fused triton kernel
- Fuses the permutation of hidden states from token order (original input order) to expert order (tokens grouped by expert) within the prologue of first the first grouped GEMM
- Fuses the (un)permutation of hidden states from expert order back to token order in second GEMM
- Fuses the mul of hidden states by expert weights within epilogue of second GEMM (only implemented for inference, not for training)

### Structure
- `grouped_gemm/interface.py`: wrappers for the individual forward / backward kernels as well as the `torch.autograd.Function`
- `grouped_gemm/kernels/forward.py`: forward kernel
- `grouped_gemm/kernels/backward.py`: backward dX and dW kernels
- `grouped_gemm/kernels/tuning.py`: manual tuning utils
- `grouped_gemm/kernels/autotuning.py`: autotuning utils
- `grouped_gemm/reference/moe_block.py`: contains `Qwen3MoeFusedGroupedGEMMBlock`, a reference implementation of Huggingface `Qwen3SparseMOEBlock` with fused triton kernel in-place of original HF expert computation
- `grouped_gemm/reference/moe_ops.py`: supporting ops (routing, token sorting, etc.) and reference MoE block using a torch-native grouped gemm approach.

### Tests
- `grouped_gemm/tests/test_grouped_gemm.py`: unit tests for forward, backward grouped gemm kernels as well as the wrapped grouped gemm autograd.Function.  Best not to run this entire test suite at once due to the large number of parametrized unit tests.  Rather, use filters to run specific
sets of tests.  E.g., to run forward tests with autotune turned on: `pytest -sv -k "forward and autotune" --tb=short tests/test_grouped_gemm.py`.  Use the test function names and parameter ids for words to filter on.
- `grouped_gemm/tests/test_qwen3_moe.py`: end to end test for Qwen3 MoE block.  IMPORTANT: read `tests/run_qwen3_moe_tests.sh` as well as notes in the test itself for complications when running parametrized pytest test suites and triton / autotune.  TLDR: use the test script and NOT pytest to run the tests.

### Benchmarks
- `grouped_gemm/benchmark/benchmark_fused_moe.py`: benchmarks HF `Qwen3SpareMOEBlock` or `Llama4TextMoe` against the fused implementation


Running with these flags on an `H100` to bench forward pass (run with `--help` to see all available flags):

For `Qwen3-30B-A3B`:
```
python benchmark/benchmark_fused_moe.py --model qwen3 --mode forward --seqlen 1024 --permute_x --permute_y --autotune
```

For the backward bench:
```
python benchmark/benchmark_fused_moe.py --model qwen3 --mode backward --seqlen 1024 --permute_x --permute_y --autotune
```

For `Llama-4-Scout-17B-16E`:
```
python benchmark/benchmark_fused_moe.py --model llama4 --autotune --mode=forward --permute_y
```
Ditto for backwards.

### Notes
- Tested and benched on `H100`, though should run on Ampere and possibly even earlier gpu generations though the autotuning configs will need to be adjusted.
- The env I used to develop the kernel was `pytorch 2.7/2.8` and `pytorch-triton 3.3`.
- The kernels can be run either as autotuned (see `autotuning.py`) or with manually specified config (see `tuning.py`).  Recommended to run using autotuner since the MoE block requires 2 configs for the forward (2 grouped gemms) and 4 for the backwards (dX and dW per grouped gemm, 2 grouped gemms).
- Running with autotuning turned off with the default manual kernel config will result is **highly** sub-optimal performance as it is only meant for testing / debugging purposes.
- I've tried to strike a balance between compilation time and autotuning search space -- can probably squeeze even more performance for specific workloads.
- The Llama4 reference layer is still highly under-optimized as there are many low-hanging opportunities for further speedups around routing and shared expert calculation.

TODO:
- TMA store: implemented but not enabled currently due to non-determinism arising from triton pipelining bug.
- Warp specialization: Hopper support for WS not yet enabled on triton 3.3x branch which ships with latest pytorch 2.7.  
- Additional optimizations:
    - Fused / optimized implementations of routing, token sorting, etc.
    - Better software pipelining within grouped gemm
    - Threadblock swizzling for better L2 caching
    - Llama4
        - Fused gather / topk weight merging 
        - Custom topk, gather indices kernel
        - Shared expert fusion with experts calculation