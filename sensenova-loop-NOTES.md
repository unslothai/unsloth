# SenseNova-Loop registration notes (draft)

Branch `feat/sensenova-loop`, drafted for human review. Not pushed, no PR.

Target: `sensenova/SenseNova-Loop-9B-Preview`, class
`SenseNovaLoopForConditionalGeneration`, config model_type `sensenova_loop` (text_config
`qwen3_5_text`). The checkpoint is remote code: a 395-line modeling file over native
transformers Qwen3.5, `SenseNovaLoopConfig(Qwen3_5Config)`, text model subclassing
`hf.Qwen3_5TextModel`. Layers 4-27 execute twice per forward (56 layer executions from 32
stored layers) with one cache slot per visit; the loop layers are `MHCDecoderLayer`
subclasses carrying two fp32 routers (`mhc_attn`, `mhc_ffn`, about 12.6 MB total) that the
remote classes keep in fp32 via `_keep_in_fp32_modules` and reject casts in `_apply`.

All numbers below come from tonight's real 4-bit loads on the 8 GB card: the first
measured set on transformers 5.18.0, and a verification run on 5.17.0
(`loop9b\smoke-517.json`: load 33 s, execution plan 56 slots with 14 full-attention
entries, fp32 MHC router params confirmed, KV 57,344 B/token, coherent output, peak
5.63 GiB with the embedding staged on CPU and the visual tower parked).

## The mapping

- `unsloth/models/sensenova_loop.py` (new): registration file, `longcat_lsa.py` style.
  `SENSENOVA_LOOP_MODEL_TYPE`, config-based detection, the int8 embedding module,
  `convert_sensenova_loop_checkpoint` (the checkpoint-side conversion), and
  `register_sensenova_loop(model, tokenizer, require_int8)` which the loader calls after
  the plain remote-code load. No Auto-factory registration and no `import_fixes` hook: the
  checkpoint's `auto_map` already points at its own classes, and the existing remote-code
  chain (trust passthrough, compiler self-gating via `_config_uses_remote_code`) is reused
  unchanged.
- `unsloth/models/loader.py`: floor constant `SUPPORTS_SENSENOVA_LOOP` (line 120),
  FastLanguageModel chain floor guard (line 1825), FastModel env chain entry with the
  floor raise, the static-generation env var and the converted-checkpoint detection
  (line 2921), FLA advisory prefix (line 899), post-load `register_sensenova_loop` call
  (line 3476).
- `unsloth/models/llama.py`: `patch_peft_model` chain entry with SwiGLU (line 5271),
  composite-safe `_decoder_layers` resolution and fused-path exclusion (lines 5330-5339),
  fused LoRAMLP exclusion in the patch loop (line 5351).
- `unsloth/models/__init__.py`: guarded export like `falcon_h1` (lines 29-38).

## Gap (a): generation cache type

The custom text model requires a DynamicCache: its forward raises TypeError for any other
cache (modeling file, the `isinstance(past_key_values, DynamicCache)` guard) and rebuilds a
plan-sized DynamicCache only when handed an EMPTY foreign cache; a populated foreign cache
is rejected. Unsloth's `unsloth_base_fast_generate` (installed on every load,
`vision.py:4871`) forces `cache_implementation = "static"` on transformers >= 4.56 even
when `UNSLOTH_DISABLE_STATIC_GENERATION = 1` is set, so the env var alone would not have
protected this model.

Fix, in two layers:
- FastModel env chain (`loader.py`): `UNSLOTH_DISABLE_STATIC_GENERATION = 1`
  (covers transformers < 4.56 and keeps the house pattern).
- `register_sensenova_loop`: `model._supports_static_cache = False`, which is the switch
  `unsloth_base_fast_generate` actually reads (`vision.py:2084`) to decline static caches,
  plus `generation_config.cache_implementation = "dynamic"` as the default for vanilla
  transformers generate and TRL.

Measured on the 5.18.0 set: the broken decode chain ran 0.08 tok/s versus 3.77 tok/s with
the DynamicCache path pinned; KV costs 57,344 bytes per token (56 execution slots), the
same per-token KV cost the 5.17.0 run recorded.

## Gap (b): tokenizer pre-tokenizer regex

This checkpoint's tokenizer ships Mistral's broken pre-tokenizer regex; transformers
documents `fix_mistral_regex = True` for it and warns on load. Unsloth's tokenizer backend
repair (`_repair_tokenizer_backend_from_json`) reinstalls the repo `tokenizer.json`
backend, which would overwrite a fix applied during the tokenizer load, so the fix must
run after the whole load.

Fix: `register_sensenova_loop` calls transformers' own
`TokenizersBackend._patch_mistral_regex(fix_mistral_regex = True)` on the loaded backend,
from the loader's post-load hook, and stamps `tokenizer.fix_mistral_regex = True` for
introspection. Detection rides transformers' own hub-tag/config logic. Offline: that
detection cannot run, so the fix silently declines (same as transformers itself offline).

## Gap (c): embedding footprint

bitsandbytes does not quantize `nn.Embedding`, so the stock checkpoint's 4-bit load
carries the whole bf16 token table: 248,320 x 4096 x 2 B = 2,034,237,440 bytes (1.89 GiB)
beside about 5.47 GiB of 4-bit weights, about 7.36 GiB total. The fit check runs during
`from_pretrained`, so on the 8 GB card (about 6.76 GiB usable) the plain all-on-GPU load
is refused before any post-load step can run; offload_layers or host-RAM staging then
becomes mandatory and keeps its margin flakiness.

Fix, checkpoint side (the path that makes the 8 GB fit deterministic):

- `convert_sensenova_loop_checkpoint` rewrites ONLY the shard holding
  `model.language_model.embed_tokens.weight` (the bf16 table, about 2.03 GB): the same
  key is stored as int8 (1,017,118,720 bytes, about 0.95 GiB) plus a new bf16 per-row
  `...weight_scale`, and `model.safetensors.index.json` gains the scale entry with the
  total adjusted. Every other file is untouched, so the disk cost is one shard write
  (about 1.1 GB new, briefly both versions on disk). In place by default, or into an
  `output_dir` copy; idempotent.
- At load, the loader detects the `weight_scale` key header-only (hub repos via
  `get_safetensors_metadata`, local dirs via the index) and patches the remote build:
  the module-level `_configure_text_model` the remote classes call swaps `embed_tokens`
  for `SenseNovaLoopInt8Embedding`, whose parameters are named `weight` (int8) and
  `weight_scale` to match the checkpoint keys, so the int8 table loads directly and no
  key renames are needed. After conversion a plain 4-bit load carries about 6.42 GiB of
  weights against about 6.76 GiB usable: no offload_layers, no host-RAM dependence, no
  block-swap margin. If the build patch ever fails to reach the model, the loader raises
  instead of letting the int8 table land in a bf16 `nn.Embedding` as garbage.
- The post-load conversion (bf16 table staged through host RAM, int8 copy lands after
  the VRAM copy is freed) stays as the fallback for stock checkpoints and adapter loads.

Measured on the 5.18.0 set: QLoRA training peaked at 7.96 of 8.19 GiB with the int8 table
in place, and the model scored 29/30 on the first 30 HumanEval problems.

## Transformers pin

Resolved as of 2026-11-18: verified on transformers 5.17.0 (`loop9b\smoke-517.json`, load
33 s, plan 56/14, fp32 routers, KV 57,344 B/token, coherent output, peak 5.63 GiB) and on
5.18.0 (the first measured set, earlier the same day). `SUPPORTS_SENSENOVA_LOOP` pins
5.17.0, so the claim is "works within the existing tested pin", not a ceiling raise;
versions below 5.17.0 remain untested.

## Remaining external items

- unsloth_zoo compile map entry: the loop model stays uncompiled (remote code), so
  nothing in unsloth needs it today, but a future native rewrite would need a zoo compile
  map entry for `sensenova_loop` (and its looped-layer execution plan) before any compile
  path can be enabled.
- pyproject: unsloth's transformers range must actually admit the tested pins; if the
  declared range excludes 5.17.0+ it has to be widened for this mapping to be usable at
  all. Left untouched in this draft.

## Known limitations of this draft

- Adapter (`is_peft`) loads skip the converted-checkpoint detection and use the post-load
  fallback, so a converted base under an adapter load still needs the plain load to fit
  first.
- `resize_token_embeddings` recreates a plain bf16 table; the int8 conversion does not
  follow a resize. `resize_model_vocab` loads run the conversion after the resize, but
  later user-side resizes drop it.
- Only plain `nn.Embedding` tables are converted post-load; a PEFT-wrapped embedding
  (`modules_to_save`) or an Embedding subclass keeps its own storage and forward.
- A tied-embedding variant would need the lm_head to share the dequantized table; this
  checkpoint has `tie_word_embeddings = false`, so the case is unhandled.
- Full-save paths that bypass `model.save_pretrained` (some unsloth save helpers,
  `push_to_hub` internals) would need the same dequantize step. Adapter-only saves are
  unaffected: the wrapper is installed on the transformer model, not on the PeftModel.
- Fused LoRAMLP/QKV/O fast paths are excluded for this model until proven on
  twice-executed loop layers; PEFT's own LoRA forwards carry it in the meantime.
- Offline tokenizer loads keep the broken regex (transformers' own detection needs the
  hub tag or the config fetch), and offline conversion requires a local snapshot.
- Only `py_compile` was run on the changed files; no GPU or load verification was done in
  this draft.
