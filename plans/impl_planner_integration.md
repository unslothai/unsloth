# Integrating the `-ot` spill planner into Studio's launch path

Fork B. Angle: the **seam**, not the algorithm. The planner core (`plan_placement`)
is fork A's; here it is stubbed so the diff compiles and the seam is testable.

All line numbers are `studio/backend/core/inference/llama_cpp.py` at `ac600e8`
(origin/main), except where a file is named.

---

## 1. What we are replacing, and why

Measured on Qwen3.8-27B UD-Q4_K_XL and Qwen3.6-35B-A3B UD-Q4_K_XL, 128K, one B200:

| placement | generation |
| --- | ---: |
| everything resident | 75.37 t/s |
| FFN spilled with `-ot`, cache resident | 13.63 t/s |
| KV cache spilled to host | **1.03 t/s** |

The KV cache lives on whatever device holds its layer:
`src/llama-kv-cache.cpp:215`, `buft = ggml_backend_dev_buffer_type(model.dev_layer(il))`.
So **`-ngl` drags the cache off the GPU with the layer**, while **`-ot` does not**,
because it overrides a tensor's buffer type without touching `dev_layer()`.
Verified empirically: `-ot 'blk\..*=CPU'` still reports `offloaded 66/66 layers to GPU`
and keeps the whole cache on CUDA0.

`--fit on` is what Studio hands the doesn't-fit case to today. Its step 3
(`common/fit.cpp:402`) is *"iteratively fill the back to front with dense layers"* --
whole layers, so on a dense model it takes the cache with it. (Its MoE branch already
puts expert tensors in system memory and is fine; this change is aimed at dense.)

So the target is exactly the arm where Studio gives up and delegates.

## 2. The seam

`_select_gpus` (9184) documents its own contract at 9210-9213:

```
Returns (gpu_indices, use_fit):
  - ([1], False)       fits on 1 GPU at the headroom threshold
  - ([1, 2], False)    needs 2 GPUs
  - (None, True)       too large, let --fit handle it
```

`(None, True)` is precisely "does not fit, delegate". It surfaces at the
placement block as the middle arm:

| line | arm | emits |
| --- | --- | --- |
| 18418 | manual, `gpu_layers >= 0` | `--gpu-layers N --fit off` (+ `--n-cpu-moe`, `--tensor-split`) |
| **18472** | **`elif use_fit:`** | **`--fit on`** |
| 18474 | `elif gpu_indices is not None:` | `-ngl -1 --fit off`, `fully_gpu_offloaded = True` |

**The planner belongs in the 18472 arm and nowhere else.** The other two arms are
cases where placement is already decided (by the user, or by Studio's own proof),
and touching them would be re-placing a load that already fits.

## 3. `--fit on` and the plan are mutually exclusive

They cannot coexist, for a reason that is easy to get backwards:

- `common/fit.cpp:377` aborts only when `mparams->n_gpu_layers != default_mparams.n_gpu_layers`.
- The default **is** `-1` (`src/llama-model.cpp:2453`).

So emitting `-ngl -1` does **not** hold the fitter off. A running fitter would
re-derive its own layer split on top of our `-ot` plan and spill whole layers,
reintroducing exactly the cache eviction we are removing. The plan therefore emits
`--fit off` explicitly.

The plan emits **`-ngl -1 --fit off` plus the `-ot` patterns**:

- `-ngl -1` keeps every layer *assigned* to a GPU, so `dev_layer(il)` is a GPU and
  the cache stays resident. This is the entire trick.
- `--fit off` stops llama.cpp from re-placing layers underneath us.
- `-ot ...` moves only the named weight tensors to the host.

`-ot` accumulates across repeated flags on llama-server
(`common/arg.cpp:2657` -> `parse_tensor_buffer_overrides` push_backs into the shared
vector), so one flag per pattern is safe. Note the intra-value separator there is
`,`, **not** the `;` that `llama-bench` uses; emitting one pattern per flag avoids
depending on either.

## 4. Spill order

From the measurements, cost per GiB of VRAM recovered:

| step | frees | generation cost | per GiB |
| --- | ---: | ---: | ---: |
| FFN tensors | 10.09 GiB | 82% | **8.1%** |
| lm_head, *after* FFN | 0.97 GiB | 16% | 17% |
| lm_head, *alone* | 0.97 GiB | 43% | 43% |
| `-ngl` (any) | varies | cache leaves -> ~1 t/s | never |

So: **FFN first, lm_head only after FFN is exhausted, `-ngl` never.**
`token_embd` needs no rule: `llama-model.cpp:1339` pins `dev_input` to the CPU
unconditionally, so it is never on the GPU to spill.

## 5. Precedence: when NOT to plan

This is the lesson from PR 9565, where ~20 review items were all one bug --
the footprint priced for a placement the child does not actually get.

**Abstain (emit `--fit on`, exactly today's behaviour) whenever anything else owns
placement.** Abstaining is always safe: it is byte-identical to current main.

| condition | why |
| --- | --- |
| `_args_place_tensors_on_cpu(extra_args)` (3263) | user `-ot` / `--cpu-moe` / positive `--n-cpu-moe` |
| `_env_places_tensors_on_cpu(env)` | the env twins, which the child inherits |
| `-ngl` / `--gpu-layers` in extras, or `LLAMA_ARG_N_GPU_LAYERS` | user owns the layer count |
| `--device` / `-dev` in extras, or `LLAMA_ARG_DEVICE` | user owns the device set |
| `--fit` set explicitly in extras | user asked for the fitter |
| planner returns `None` | cannot size it; do not guess |

User extras are appended **after** Studio's flags (19127-19140), so a user flag
would last-wins anyway. Abstaining is stronger than relying on that: it also stops
Studio *logging* and *reporting* a plan the child will not run.

## 6. Retry paths

Every retry that changes placement or resizing must drop the plan, because the
plan's headroom arithmetic no longer holds. The plan's tokens are recorded so
`_without_subsequence` (4098) can remove them, mirroring how `_mem_managed` is
dropped at 19863.

| retry | line | action |
| --- | --- | --- |
| `_spawn_and_wait` internal `--fit off` retry | 19845 | **no interaction**: `_fit_off_retry_eligible` (22425) returns False when `use_fit` is True, and the planned arm is the `use_fit` arm |
| CPU fallback `-cpu` | 19937 | builds its own `replay`; must not inherit the plan |
| arch-crash `-archfallback` | 20317 | drop plan, fall back to `--fit on` |
| no-flash `-noflash` | 20363 | drop plan (KV dtype changes -> headroom changes) |
| no-flash MTP `-noflash-mtp` | 20436 | same |
| `-retry` | 20537 | drop plan |
| `-mmproj-cpu` | 20588 | drop plan (projector moves) |

Fallback on drop is `--fit on`, i.e. today's behaviour, which is the safe direction.

## 7. Gate and compatibility

Default **OFF**, `UNSLOTH_SMART_OFFLOAD=1` to enable, matching the existing env-flag
idiom (3070, 3544). Rationale: this changes the placement of large loads, the
measurements come from one machine class (B200 + 192-core host, ~125-140 GiB/s
effective host bandwidth), and consumer PCIe will look different.

Compatibility argument, in three parts:

1. **Disabled** -> the helper returns `[]` before doing anything, so the argv is
   byte-identical to main. Pinned by test.
2. **Enabled and the model fits** -> `gpu_indices is not None`, so the 18472 arm is
   never reached. Identical.
3. **Enabled, does not fit, anything else owns placement** -> abstains. Identical.

Only case 4 (enabled, does not fit, Studio owns placement, planner produced a plan)
differs, which is the case that is 5.5x faster.

Frontend and per-model config: unchanged in this diff. A user-facing toggle is
follow-up work; the env gate keeps this reviewable.

## 8. Risks

- **Cost model is machine-specific.** The *orderings* (cache over weights, FFN
  before lm_head, never `-ngl`) should hold anywhere; the ratios should not be
  quoted as universal. The planner should be conservative and the gate default-off
  until measured on consumer hardware.
- **Partial FFN spill granularity** is fork A's problem; the seam only needs an
  ordered pattern list.
- **`--n-cpu-moe` overlap.** On MoE, `--fit on`'s step 3 already does the right
  thing. The planner should prefer to abstain on MoE unless it beats the fitter,
  rather than duplicate it. Flagged for fork A.
- **Multi-GPU.** `-ot ...=CPU` is device-agnostic, so it composes with a layer
  split, but the per-device headroom arithmetic is the planner's job.
- **Metal / unified memory.** `-ot` to CPU on unified memory moves bytes within one
  pool; the win is unlikely to exist there. The planner should abstain on Apple.
