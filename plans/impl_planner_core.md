# Smart placement planner: core algorithm

## What this replaces

Today a load that does not fit is handed to llama.cpp's `--fit on`. Its step 2
shrinks the context, which is right, and its step 3 spills **whole layers** via
`n_gpu_layers`, which is wrong for anything with an attention cache: a layer's KV
cache is allocated on `model.dev_layer(il)` (`llama-kv-cache.cpp:215`), so moving
a layer to the host moves its cache with it.

Measured on Qwen3.6-35B-A3B at 128K, one B200:

| placement | generation |
| --- | ---: |
| everything resident | 182 t/s |
| weights spilled, cache resident (`-ot`) | 71.63 t/s |
| cache spilled, weights resident (`-nkvo 1`) | **3.24 t/s** |

Keeping the cache resident and paying for host-resident weights is **22x** faster
than the reverse. `-ot` achieves exactly that, because it overrides tensor buffer
types without touching layer assignment: measured `offloaded 66/66 layers to GPU`
with `CUDA0 KV buffer size = 16.00 MiB` and **no CPU KV buffer at all**, even with
every block tensor forced to the host.

So the planner emits `-ot` patterns and never `-ngl`.

## The spill ladder

Rungs in increasing cost, each measured on Qwen3.8-27B dense at 128K:

| rung | frees | generation | cost per GiB |
| --- | ---: | ---: | ---: |
| 0 nothing spilled | - | 75.37 | - |
| 1 FFN to host | 10.09 GiB | 13.63 | **8.1%** |
| 2 FFN + lm_head | 10.86 GiB | 11.39 | lm_head adds 16% |
| never: `-ngl` | - | ~1.03 | catastrophic |

lm_head **alone** costs 43% for 0.97 GiB (42.60 t/s), but only 16% when added on
top of an FFN spill. Once FFN offload has made generation host-bandwidth-bound,
another ~10% of traffic buys a ~16% slowdown instead of dominating an otherwise
on-device step. Hence rung 2 is strictly after rung 1, never before.

`token_embd` is never charged to VRAM at all: `llama-model.cpp:1339` pins
`dev_input` to the CPU unconditionally ("very little benefit to offloading the
input layer"). Confirmed by measurement: baseline `CPU_Mapped model buffer size =
682.03 MiB`, exactly `token_embd.weight`.

`output` / lm_head rides the layer list at index `n_layer_all`, so with
`i_gpu_start = n_layer_all + 1 - n_gpu_layers` it lands on the GPU for any
`-ngl >= 1`. It is the *first* thing offloaded, not the last, which is why loads
report `offloaded 66/66 layers` for 65 blocks.

## What is spillable, per architecture

Not all of `ffn_*` is equal. On the MoE model:

| bucket | size | share | spillable? |
| --- | ---: | ---: | --- |
| `ffn_(up\|gate\|down)_exps` | 18.320 GiB | 88.0% | **yes** |
| `ffn_(up\|gate\|down)_shexp` | 0.125 GiB | 0.6% | **no** |
| `ffn_gate_inp*` (routers) | 0.078 GiB | 0.4% | no |
| `attn_*` | 1.017 GiB | 4.9% | no |
| `ssm_*` | 0.267 GiB | 1.3% | no |

Shared experts run on **every** token, exactly like a dense FFN, so spilling them
costs dense-like bandwidth for 0.6% of the model. Routers are tiny and on the
critical path. Only the sparse `_exps` tensors are worth spilling, which is what
the measured `-ot '.ffn_(up|down|gate)_exps.=CPU'` did.

This is also why MoE tolerates spill so much better than dense: only ~8 of 256
experts per layer are read per token, whereas a dense FFN is fully activated.
Measured cost of full FFN spill: **2.5x** on MoE (182 to 71.63), **5.5x** on
dense (75.37 to 13.63). The mmap-penalty ratio (4.64x MoE vs 2.94x dense)
points the other way and is a different quantity; it must not be used to predict
this.

## Pattern encoding, and a trap

Patterns are matched with `std::regex_search` (`llama-model-loader.cpp:1174`),
i.e. **unanchored substring search**. During this investigation
`-ot 'output\.weight=CPU'` silently also matched every `blk.N.attn_output.weight`
and moved 16 attention output projections to the host. The arithmetic gave it
away: 994.63 MiB (lm_head) + 357.16 MiB (16 attn_output) = 1351.79 MiB against an
observed 1351.82 MiB drop.

**Every pattern this planner emits is fully anchored**, `^...$`. Anchoring also
makes `ffn_(up|gate|down)\.weight$` safe against `ffn_gate_inp.weight`, since the
literal `\.` cannot match the `_` in `gate_inp`.

- all blocks: `^blk\.\d+\.ffn_(up|gate|down)_exps\.weight$`
- some blocks: `^blk\.(3|7|11)\.ffn_(up|gate|down)_exps\.weight$`
- lm_head: `^output\.weight$`

One alternation-grouped pattern per rung keeps argv short; llama-bench and
llama-server both accept `;`-separated overrides in a single `-ot`.

## Partial spill: which blocks

The dominant cost is **bytes moved per token**, so the selection should free the
deficit while spilling as few bytes as possible. These are dynamic quants with
very non-uniform blocks (Q2 FFN spans 49.8 to 209.2 MiB), so the choice matters
for overshoot: taking one 209 MiB block to cover a 50 MiB deficit wastes 159 MiB
of bandwidth every token.

Policy `LARGEST_FIRST` (default): take blocks largest-first while the remaining
deficit exceeds the smallest remaining block, then cover the residual with the
**smallest block that still covers it**. That is best-fit-decreasing, which keeps
both the spilled byte total and the block count low. Low block count matters
because each host-resident run is a graph split.

`FRONT_FIRST` and `BACK_FIRST` are offered for experimentation.

**Honest gap: we have not measured whether block choice matters at all.** Every
`-ot` measurement to date spilled either all blocks or none. It is plausible that
contiguous runs schedule better than scattered ones (adjacent host blocks can
merge into one graph split), which would favour a contiguous policy over
largest-first. Until that is measured, the default is justified by byte-minimality
alone, and the policy is configurable rather than baked in.

## Context versus residency

`--fit` shrinks context first, and our data agrees with that ordering: a resident
smaller context beats a spilled larger one by a wide margin. But context is a
user-visible feature, not a free variable, and Studio already has its own context
planner. Silently shrinking a user's request to win a benchmark is the wrong
default.

So `ContextPolicy` is explicit:

- `NEVER_REDUCE` (**default**) - honour the request, spill as needed.
- `PREFER_RESIDENT` - reduce context (to `min_ctx`) if that avoids spilling.
- `FIT_ONLY` - reduce only when no rung of the ladder fits.

## KV quantisation

`q8_0` halves the cache but measured **35% slower** generation on the MoE model,
and only four matched K/V combinations are compiled without
`GGML_CUDA_FA_ALL_QUANTS` (a mismatched pair silently falls to CPU; observed as a
59-minute stall at 8272% CPU). So it is **off by default**, and when enabled the
planner only ever emits matched pairs.

## Abstention

If any needed quantity is missing, return a plan with `changed = False` and no
flags. Abstaining is always safe: llama.cpp's own defaults apply. This mirrors
the existing `_fit_derived_load_mode` contract.

The same applies when even rung 2 does not fit: `-ot` cannot help when the
*resident floor* (attention + norms + recurrent state + cache + overhead) exceeds
VRAM. The planner reports `insufficient` and declines to emit `--load-mode none`,
because mmap is the only thing that makes an over-commit pageable rather than
OOM-killed. It attaches a recommendation (smaller quant, or less context) rather
than silently producing a launch that dies.

## Load mode

`--load-mode none` whenever anything is host-resident **and** host RAM can hold
the host-resident part. mmap costs 2 to 4.6x on host-resident weight reads
(`none` and `dio` are within 1.2% of each other; `mmap+mlock` does not rescue it,
340.96 vs 1396.86, so the penalty is not page eviction). Where host RAM cannot
hold it, mmap must stay.

## Module split

- `offload_layout.py` - reads a GGUF and produces `ModelLayout`. Does file IO.
- `offload_planner.py` - pure functions over `ModelLayout`. No IO, no globals.

Keeping them apart is what makes the planner exhaustively testable without
fixtures.
