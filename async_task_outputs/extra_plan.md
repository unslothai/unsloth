**(1) Summary & Findings**

The target change should be treated as a measurement-driven speed-tier restructure, not as a blanket rewrite. The current evidence in the prompt already says the safest eager patches are installed broadly, while earlier direct in-place rewrites measured roughly neutral. The plan should preserve that bias: make the new tier behavior explicit, keep class-level monkey patches reversible and idempotent, and only promote additional dense-path in-place changes when they pass correctness and performance gates on each supported transformer family.

The proposed tier model is coherent:

`off`:
No speed patches beyond whatever the base runtime requires.

`default`:
Enable channels-last where applicable, cuDNN tuning/configuration, compiled GGUF dequant-only path, and eager monkey patches for all supported dense and GGUF diffusion transformer paths.

`max`:
Enable everything in `default`, then add regional compilation of repeated transformer blocks, TF32 where supported and acceptable, and fused-QKV where the model class and attention implementation can safely use it.

The important semantic change is that `max` becomes a strict superset of `default`. That means compiled GGUF dequant must no longer be assumed mutually exclusive with regional transformer compilation until tests prove it has to be. The expected implementation should first try composition, then fall back only with a narrow capability flag and a documented reason.

The current Diffusers transformer structures look favorable for regional compilation because the target model families expose repeated transformer blocks. PyTorch’s regional compilation guidance explicitly targets repeated regions to reduce compile cold-start cost while still compiling hot blocks: [PyTorch regional compilation recipe](https://docs.pytorch.org/tutorials/recipes/regional_compilation.html). The lower-level API also describes nested compile regions as reusable repeated regions, with recompilation only when guards such as shape, dtype, device, or stride change: [torch.compiler API](https://docs.pytorch.org/docs/stable/torch.compiler_api.html).

The in-place dense-path optimization opportunity is narrower than it first appears. The common formulas are residual adds, gated residual adds, and AdaLayerNorm-style modulation such as `norm * (1 + scale) + shift`. These are usually safe only when the mutated tensor is a fresh activation, is not reused by another branch, is not returned as an alias that callers expect to remain independent, and does not interfere with compiler/CUDA graph assumptions. For most eager inference paths, in-place modulation of fresh normalization outputs is more promising than mutating incoming block inputs.

Channels-last should remain part of `default`, but its benefit is likely concentrated in 4D image/conv paths rather than the 3D sequence tensors used inside DiT transformer blocks. PyTorch documents channels-last as a memory format for 4D NCHW tensors: [channels-last tutorial](https://docs.pytorch.org/tutorials/intermediate/memory_format_tutorial.html). The implementation should avoid forcing channels-last on sequence tensors or creating hidden layout conversions that erase the win.

TF32 belongs in `max`, not `default`, because it can change numerical results for float32 matmul/convolution. PyTorch documents TF32 as faster but lower mantissa precision on supported NVIDIA devices: [CUDA semantics: TF32](https://docs.pytorch.org/docs/stable/notes/cuda.html). It should be enabled with saved/restored global flags and validated by image/tensor tolerances, not assumed harmless.

Global reusable buffers are plausible but high-risk. PyTorch’s CUDA caching allocator already tries to reach a steady state by reusing freed memory, which explains why manual reusable buffers can benchmark neutral: [PyTorch CUDA allocator overview](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html). Manual buffers are most likely useful only if profiling shows allocator churn or peak memory spikes that the allocator cannot smooth out. They are most dangerous inside compiled regions, because mutated inputs or dynamic buffer resizing can cause graph breaks, skipped CUDA graphs, or recompiles. PyTorch’s CUDA graph docs call out mutation and static-address constraints: [CUDAGraph Trees](https://docs.pytorch.org/docs/stable/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html).

**(2) Important Issues To Investigate Or Look Further Into**

The first issue is whether compiled GGUF dequant composes with regional compile. Test both application orders: compile dequant first then compile repeated blocks, and compile repeated blocks first then wrap dequant. The desired behavior is one stable compiled dequant path reused inside or beside the regionally compiled block, without graph breaks, recompilation loops, stale dequant buffers, or correctness drift.

The second issue is dense-path in-place mutation safety. For each supported transformer family, classify candidate mutations into three groups:

Safe-looking:
In-place modulation on fresh normalization outputs, such as multiply/add on the output of LayerNorm or RMSNorm before attention/MLP.

Conditionally safe:
In-place residual adds to `hidden_states` or `encoder_hidden_states`, only if no caller-visible alias, hook, adapter, attention processor, or later branch needs the pre-add value.

Risky:
In-place activation rewrites inside feed-forward modules, especially where activation inputs may be reused, where approximate GELU implementations are module-specific, or where compiler fusion already handles the elementwise chain.

The third issue is model-specific modulation behavior. Qwen-style modulation can use per-token selection paths, while Z-Image-style modulation includes noisy/clean token selection and tanh gates. These branches may be correct in eager mode but create extra compiler guards or less stable fused graphs. They need per-model tests rather than a single shared assumption.

The fourth issue is regional compile granularity. Compiling the whole transformer may produce stronger fusion but higher cold start and more graph-break risk. Compiling repeated blocks should be the default `max` strategy, but the exact block boundary must include enough computation to matter while excluding dynamic preprocessing, image packing/unpacking, scheduler logic, progress callbacks, and optional non-tensor control flow.

The fifth issue is reversibility and idempotence. Since the system uses class-level monkey patches, tier switching must be safe across repeated pipeline loads, multiple model classes, and switching from `max` back to `default` or `off`. Every patch needs an original-method registry, an installed-state key, and a clean unpatch path.

The sixth issue is numerical tolerance. “No accuracy beyond fp noise” should be operationalized as tensor-level and image-level checks. Dense BF16/FP16 paths can use tolerances appropriate to dtype. FP32 with TF32 enabled needs separate thresholds and should be reported as a `max`-only behavior change.

The seventh issue is global-buffer lifetime. The arbiter guarantees one active pipeline, but that does not automatically guarantee safe buffer reuse across CUDA streams, async execution, callbacks, interrupted requests, shape changes, or teardown. Any reusable buffer scheme must be keyed by device, dtype, shape, and possibly stream, and it must never return a tensor that aliases the scratch buffer unless ownership is explicit.

The eighth issue is channels-last scope. The plan should verify that channels-last conversion is applied only to compatible modules/tensors and does not add repeated `contiguous` or layout conversion costs inside transformer blocks.

The ninth issue is fused-QKV compatibility. Fused-QKV must be tested against dense safetensors, GGUF, LoRA/adapters, custom attention processors, model offload, and torch.compile. It should not be enabled by class name alone; it needs structural checks on projection shapes, bias handling, dtype, quantization wrappers, and processor expectations.

**(3) Approximate Plan On How To Resolve / Investigate / Do Task**

1. Establish a baseline matrix.

Measure current `off`, current `default`, and current `max` behavior before edits. Capture latency, warmup time, compile time, peak memory, allocation count if available, output tensor deltas, and generated image deltas. Use fixed seeds, fixed scheduler settings, identical prompts, identical dimensions, and a small number of representative resolutions.

Matrix dimensions:
Dense safetensors and GGUF.
Each supported transformer family.
Cold run and warmed run.
Eager and compiled where applicable.
At least one static-shape repeated run and one changed-shape run.

2. Define the new tier contract first.

Make the speed-tier resolver produce an explicit feature set instead of scattered conditionals. The contract should be:

`default = channels_last + cudnn + eager_patches + gguf_dequant_compile`

`max = default + regional_compile + tf32 + fused_qkv`

This should be represented as capability flags that downstream patch/install code consumes. The key invariant is that `max` starts from the already-resolved `default` feature set, not from a separate branch.

3. Normalize patch ordering.

Use one deterministic install order:

Runtime/global backend flags.
Memory format setup.
Class-level eager patches.
GGUF dequant compile wrapper.
Fused-QKV transforms if selected.
Regional compile if selected.

The reason to apply regional compile late is that the compiler should see the final call structure, including eager-patched operations and dequant wrappers. If tests prove this order breaks GGUF compiled dequant, test the reverse order and preserve the best one behind a narrowly named compatibility branch.

4. Harden reversible patch infrastructure.

Use a registry keyed by class object and patch name. Store original callables once. Reinstalling the same tier should be a no-op. Downgrading or disabling should restore exactly the original method when no other active tier requires the patch. Add explicit metadata such as installed model classes, active feature flags, and whether a patch is dense-only, GGUF-only, or shared.

5. Revisit dense eager patches with model-specific candidate functions.

For dense paths, evaluate these candidates independently:

Ada modulation out-of-place replacement:
Use fused formulas such as `addcmul` where it reduces temporary tensors and has already shown 1-ULP behavior.

Ada modulation in-place:
Only mutate fresh normalization outputs. Example conceptual rewrite: normalize into a fresh tensor, multiply by scale term in-place, then add shift in-place. Do not mutate source `hidden_states`.

Residual gated add:
Prefer `add_` only when the residual target is not needed elsewhere and the operation is outside a compiled CUDA graph-sensitive boundary. Test `x.add_(gate * residual)` versus `x = x + gate * residual`; if `gate * residual` still allocates the dominant temporary, this may not help.

Activation in-place:
Treat as experimental. Only use if the specific activation input is fresh and compiler/eager benchmarks show a win. GELU is less attractive than SiLU because module implementations and approximations vary.

6. Build correctness tests before accepting each in-place candidate.

For each target transformer family, run the original and patched block with identical randomized inputs covering:
Batch 1 and batch 2.
Small and realistic sequence lengths.
BF16, FP16, and FP32 where supported.
With and without text/context stream.
With modulation-index or noisy/clean-token branches where applicable.
With attention masks and RoPE/frequency inputs where applicable.

Assertions:
No shape, dtype, device, stride, or memory-format regression unless intentional.
Output max/mean error within dtype-specific tolerance.
No input tensor unexpectedly changed unless the candidate explicitly allows mutation.
No returned tensor aliases a scratch buffer that will be reused.
No autograd requirement is introduced; inference mode remains clean.

7. Build compile-safety tests.

For every accepted eager/in-place patch, test:
Plain eager.
`torch.compile` on the block.
Regional compile on repeated blocks.
`max` with GGUF compiled dequant active.
Shape-stable repeated calls.
Shape-changed calls that should either reuse dynamic guards or recompile a bounded number of times.

Record graph breaks and recompiles. Any patch that increases recompiles or disables CUDA graph capture without a clear latency win should stay out of `default` and likely out of `max`.

8. Test compiled GGUF dequant composition directly.

Use a small GGUF-backed transformer path and run four configurations:

GGUF dequant compile only.
Regional compile only.
GGUF dequant compile then regional compile.
Regional compile then GGUF dequant compile.

Pass/fail criteria:
Same outputs within quantized tolerance.
No crash during first compile.
No repeated recompilation across identical inputs.
No extra CPU fallback in hot path.
No material latency regression versus dequant-only `default`.

If composition works, keep `max` as a true superset. If not, implement a targeted compatibility mode that preserves `max` semantics as much as possible, such as compiling the dequant op outside the regional block or disabling only the conflicting nested compile boundary, not disabling all default GGUF optimizations.

9. Re-examine global reusable buffers as an experiment, not a default feature.

Start by profiling whether per-forward allocation is a real bottleneck in diffusion DiT blocks. Collect allocation counts, peak memory, reserved memory, and latency with allocator warmup. If allocator churn is not visible, do not add global buffers to default or max.

If profiling shows a real issue, prototype eager-only scratch buffers with strict constraints:
One buffer owner per active pipeline.
Keyed by device, dtype, shape, and role.
Never resized inside a compiled graph.
Never used for tensors returned to callers.
Invalidated on model unload, dtype change, device change, or resolution change.
Disabled automatically when regional compile is active unless compile tests prove it is stable.

10. Compare manual buffers against compiler-managed buffers.

Because `torch.compile` and CUDA graphs can already manage static addresses and memory pools, benchmark global buffers separately for:
Eager dense.
Eager GGUF.
Default GGUF compiled dequant.
Max regional compile.
Max regional compile plus compiled dequant.

Expected result: global buffers may be neutral or harmful under `max`. If so, keep them opt-in/debug-only.

11. Add acceptance benchmarks.

Use representative image-generation calls rather than only synthetic block tests. Include:
Single prompt, fixed resolution.
Changed resolution.
Changed batch size.
Dense and GGUF.
Each speed tier.
Warmup excluded and included numbers.
Peak memory.
Compile cold-start time.
Steady-state tokens/steps per second or seconds per denoise step.

Promote a candidate only if it has a consistent win or a clear memory benefit without correctness or compile regression.

12. Document final behavior in runtime metadata.

Expose which optimizations are active for the current pipeline: eager patches, compiled dequant, regional compile, TF32, fused-QKV, channels-last, global buffers if enabled. This makes support/debugging much easier and helps prove `max` is actually a superset of `default`.

**(4) Potential Bugs Or Issues Worth Looking Into**

In-place residual mutation may corrupt caller-visible tensors if a block input is also referenced by another branch, hook, adapter, debug capture, or later residual path. This is the highest-risk dense optimization.

In-place modulation may accidentally mutate a tensor view. Broadcasted scale/shift tensors, chunked modulation outputs, and selected per-token modulation branches can produce non-contiguous or aliased views. Mutate only the fresh normalized activation, never the chunked modulation tensors.

`addcmul` rewrites can silently alter rounding order. The existing 1-ULP finding is encouraging, but each model-specific formula still needs per-dtype tolerances and image-level checks.

Activation rewrites can break approximate GELU semantics or FeedForward module assumptions. They may also be optimized away by Inductor already, producing no eager win and worse compile behavior.

Regional compile can specialize on shape, stride, dtype, device, boolean flags, or optional inputs. Diffusion workloads vary by height, width, prompt length, batch size, guidance mode, and model-specific options, so recompilation control is essential.

Compiled GGUF dequant inside a regionally compiled block may create nested compile conflicts, graph breaks, duplicate compilation, stale dequant buffers, or CPU fallback. This is the main `max`-as-superset risk.

Applying fused-QKV after compilation may compile the wrong graph; applying it before capability checks may break custom attention processors or quantized projections. The transform order needs tests.

TF32 global flags can leak outside the active pipeline if not restored. This can affect unrelated inference or training sessions in the same process.

Channels-last conversion can be a no-op for transformer sequence tensors, or worse, can introduce layout conversions if applied indiscriminately. Limit it to compatible tensors/modules.

Global reusable buffers can create cross-call data hazards if a second request starts before the previous CUDA work has completed, even with single-active-pipeline scheduling at the Python level.

Global buffers can pin peak VRAM after a large resolution request and hurt later smaller requests unless there is a clear release/shrink policy.

Global buffers stored as module attributes can become compiler guards or static inputs, causing recompilation when shape/device/dtype changes.

Manual buffers may fight CUDA graph memory planning. PyTorch’s compiled CUDA graph path has static-address assumptions; mutated external buffers can cause CUDA graph skips or partitioning.

Class-level monkey patches can stack if install is not idempotent, or restore the wrong method if multiple tiers install overlapping patches.

Tier downgrades can leave stale compiled modules, fused projections, backend flags, or patched methods active unless teardown is explicit.

LoRA/adapters may depend on projection module structure. Fused-QKV and in-place attention rewrites must verify adapter compatibility.

Offload or device movement can invalidate buffers and compiled graphs. Any optimization cache must be cleared on device map changes, CPU/GPU offload transitions, dtype changes, and model unload.

The final recommendation is to implement the tier restructure first, then promote only the already-proven eager patches into dense/default behavior, then run the GGUF dequant plus regional compile composition tests, and only after that revisit in-place residuals, activations, and global buffers as individually gated experiments.