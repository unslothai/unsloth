## Efficient Fused Cross Entropy Loss

Memory-efficient cross entropy implementation that only materializes the derivatives of the language modeling head layer without storing the logits and chunks the computation of the logits such that the full logits tensor is never realized.

This is a direct adaptation of this [repo](https://github.com/mgmalek/efficient_cross_entropy/tree/main).

## Contents

- [Overview](#overview)
- [Changes](#changes)
- [Tests](#tests)
- [Benchmarks](#benchmarks)
- [Profiling](#profiling)
- [Next Steps](#next-steps)

## <a id="overview">Overview</a>

In short:

- the logits, derivative with respect to the hidden state inputs to the language modeling head layer (`dX` hereafter), and the derivative with respect to the logits projection weights (`dW` hereafter) are computed in chunks
- the logits are overwritten by its derivatives within a custom loss kernel to avoid additional memory allocations.

See the original [repo](https://github.com/mgmalek/efficient_cross_entropy/tree/main) for an excellent explanation of the design.

## <a id="changes">Changes</a>

The following changes were made to the original kernel:

- Reshape inputs and labels to adapt the `3-D` language modeling tensors with the required shapes of the kernel.
- Upcast `loss` to `float32`, which in the original kernel was initialized to the autocasted / in-feat dtype.
- Add `torch.cuda.amp.{custom_fwd,custom_bwd}` to the `autograd.Function`.

All changes are enumerated in `unsloth/kernels/fused_cel.py`.

Additionally, adapter layers and configs in `fused_cel.py` enable integration with `transformers` and `unsloth`.

## <a id="tests">Tests</a>

See `tests/test_CEL.py` for correctness checks.

The comments in the tests describe numerical edge cases.

## <a id="benchmarks">Benchmarks</a>

Following are results from preliminary testing on a `L4` NVIDIA GPU comparing a small `llama-like` [model](https://huggingface.co/hf-internal-testing/tiny-random-LlamaForCausalLM):

_Correctness_

## <a id="next-steps">Next Steps</a>

- [ ] Integrate with `FastLanguageModel`
- [ ]
