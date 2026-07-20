# Multi-GPU (FSDP2) design for the Studio diffusion DiT trainer

Status: design only. No implementation in this document's scope.

## Goal and scope

Extend `core/training/diffusion_dit_trainer.py` with an optional multi-GPU path
for the dense base precisions (bf16 and fp8). The single-GPU path stays the
default and is untouched; the FSDP2 path activates only when the run is
launched with `world_size > 1` and `base_precision` is `bf16` or `fp8`.

Out of scope, explicitly:

- torchao int8 (`Int8WeightOnlyConfig`): the quantized weight is a tensor
  subclass, and `fully_shard` would need to wrap a DTensor around that
  subclass. DTensor-over-subclass composition is a known sharp edge (dispatch
  ordering between the two `__torch_dispatch__` layers is not guaranteed, and
  reduce-scatter over the packed int8 payload is undefined). int8 stays
  single-GPU only; the trainer should raise a clear error when int8 + multi-GPU
  is requested rather than attempt it.
- nf4 (bitsandbytes): bnb 4-bit `Params4bit` are likewise incompatible with
  `fully_shard`. Multi-GPU QLoRA would need per-rank replicated bases (DDP on
  the LoRA params only), which is a separate, simpler design; noted below.
- Pipeline/tensor/context parallelism. The Studio DiTs fit one node; FSDP2
  data parallelism is the whole design.

## Wrapping plan (vendored, not imported)

Vendor a minimal FSDP2 helper into `core/training/` modeled on the reference
recipe pattern (an auto-pipeline that loads the diffusers pipeline, then
parallelizes only the transformer):

1. Only the transformer is wrapped. The VAE and text encoders are already
   freed before the loop in our phased load, so there is nothing else to
   shard. The conditioning phase (prompt encode + latent cache build) runs on
   rank 0 only and broadcasts/serializes results through the persistent
   conditioning cache (`diffusion_train_extras.PersistentConditioningCache`),
   which doubles as the cross-rank handoff: ranks != 0 wait on the cache files
   instead of loading the encoders at all.
2. Per-block units: walk the transformer's `nn.ModuleList` block lists and
   `fully_shard` each repeated block as its own unit, then `fully_shard` the
   root with `reshard_after_forward=False` (root params are reused immediately
   in backward). All non-final blocks reshard after forward; keep the last
   block gathered so backward prefetch starts warm.
3. Explicit prefetch chains (`set_modules_to_forward_prefetch` depth 1,
   `set_modules_to_backward_prefetch` depth 2) as an opt-in flag; defaults
   off for the first landing to keep the state space small.
4. Mixed precision policy: `param_dtype=bf16, reduce_dtype=fp32,
   output_dtype=bf16, cast_forward_inputs=True`. fp32 gradient reduction is
   the important part: the LoRA params train in fp32 (our
   `cast_training_params` call), so their reduce must not round through bf16.
5. CPU offload (`CPUOffloadPolicy`) stays exposed but default-off, matching
   the reference configs (none of their diffusion examples enable it).

## PEFT LoRA over FSDP2: ordering rules

The ordering that works (and that the reference implementation encodes):

1. Load the dense bf16 transformer.
2. Attach the LoRA (`add_adapter`) BEFORE `fully_shard`, so FSDP2 sees the
   final module structure and the `lora_A/lora_B` params become DTensors with
   proper gradient reduction.
3. Do NOT freeze the base before sharding in a way that creates never-gathered
   units; freeze base params AFTER `fully_shard` (freezing before can break
   LoRA gradient reduction wiring inside a unit that mixes frozen and
   trainable params). Our current code freezes before `add_adapter`; the FSDP2
   path must reorder this: attach adapter, shard, then `requires_grad_(False)`
   on non-LoRA params.
4. Keep LoRA dtype equal to the base param dtype (bf16 storage) inside the
   sharded units so reduce-scatter sees one dtype per unit; the fp32 master
   copy lives in the optimizer (switch from `cast_training_params` to an
   optimizer-side master-weight approach, e.g. torch AdamW with fused fp32
   master weights, when sharded).
5. Saving: gather the LoRA state dict on rank 0 via
   `get_peft_model_state_dict` over full-tensor materialization
   (`DTensor.full_tensor()` per param); the LoRA set is megabytes, so a plain
   rank-0 gather is fine (no DCP needed). The EMA shadow
   (`diffusion_train_extras.LoRAEMA`) tracks only LoRA params, which are small
   enough to keep replicated on every rank; update from `full_tensor()` views
   on rank 0 only.

## fp8 over FSDP2

torchao `convert_to_float8_training` composes with FSDP2 when applied BEFORE
`fully_shard` (Float8Linear is a module swap, not a tensor subclass on the
stored weight; weights stay bf16 in memory). Order: load dense, attach LoRA,
convert frozen linears to Float8Linear (our existing `_fp8_module_filter`
excludes `lora_` and non-16-divisible shapes), then shard. Two extra knobs
from the reference worth carrying as config, both default off:

- `enable_fsdp_float8_all_gather` (tensorwise only): all-gathers the fp8-cast
  weight instead of bf16, saving interconnect bandwidth.
- `precompute_float8_dynamic_scale_for_fsdp`: batches the per-step scale
  computation; only meaningful with the fp8 all-gather.

Both interact only with the tensorwise recipe (what we use); if we ever move
to rowwise, they must be forced off.

## Per-family notes: would Wan/Hunyuan-style custom strategies apply?

The reference registers custom parallelization strategies for two video DiTs:
Wan (a TP plan over its ffn/time-embedder plus per-block activation
checkpointing) and HunyuanVideo (per-block non-reentrant checkpoint wrapping,
flash-varlen attention masks). Mapped onto our trainable families:

- flux.1 / flux.2 (dev, klein): homogeneous double/single-stream block lists;
  the generic per-block wrap covers them. No custom strategy needed. FLUX.2-dev
  (32B) is the family that actually needs FSDP2 to train dense at all on
  sub-80GB cards.
- qwen-image: homogeneous MMDiT block list; generic wrap. Its unpadded text
  stream is irrelevant to sharding.
- z-image: heterogeneous single-stream blocks (~11 distinct shapes). Still one
  ModuleList, so the generic per-block wrap applies; the only interaction is
  with regional compile (more distinct graphs), which we already handle via
  the recompile-limit bump.
- krea-2: generic wrap; its custom conditioning modules (text_fusion,
  time_mod_proj) sit outside the block list and land in the root unit, which
  is fine at their size.
- A Wan-style TP plan would only become relevant if we adopt video families
  (Wan/Hunyuan themselves) into the trainer; for the current image DiTs,
  sequence lengths (~4k tokens) never reach the ~30k-token activation
  pressure that motivated Wan's per-block checkpoint + TP combination. Our
  existing non-reentrant gradient checkpointing flag is the equivalent lever
  and already composes with FSDP2.

## Launch and process model

The Studio trainer runs as a spawned subprocess today. Multi-GPU wraps that
same entrypoint with `torchrun --nproc-per-node=N`; the event/stop protocol
stays rank-0-only (ranks != 0 swallow events, poll the same stop flag file).
Determinism: seed per rank as `cfg.seed + rank` for the noise/timestep draws
while keeping the permutation sampler on the shared seed with a rank-strided
view, so the global batch covers the dataset exactly as the single-GPU run
does at the same effective batch size.

## Failure containment

- If `fully_shard` or the fp8 conversion raises on any rank, abort the run
  (no silent single-GPU fallback mid-launch: ranks would desync).
- Preflight in `training_precision_preflight_error`: reject
  world_size > 1 with base_precision in ("nf4", "int8", "mxfp8") before any
  eviction, with the DTensor-over-subclass rationale in the int8 message.
