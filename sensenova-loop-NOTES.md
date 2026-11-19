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

All numbers below come from tonight's real 4-bit load on the 8 GB card, transformers 5.18.0.

## The mapping

- `unsloth/models/sensenova_loop.py` (new): registration file, `longcat_lsa.py` style.
  `SENSENOVA_LOOP_MODEL_TYPE`, config-based detection, the int8 embedding module, and
  `register_sensenova_loop(model, tokenizer)` which the loader calls after the plain
  remote-code load. No Auto-factory registration and no `import_fixes` hook: the
  checkpoint's `auto_map` already points at its own classes, and the existing remote-code
  chain (trust passthrough, compiler self-gating via `_config_uses_remote_code`) is reused
  unchanged.
- `unsloth/models/loader.py`: floor constant `SUPPORTS_SENSENOVA_LOOP` (line 120),
  FastLanguageModel chain floor guard (line 1825), FastModel env chain entry (line 2919),
  FLA advisory prefix (line 899), post-load `register_sensenova_loop` call (line 3462).
- `unsloth/models/llama.py`: `patch_peft_model` chain entry with SwiGLU (line 5271),
  composite-safe `_decoder_layers` resolution and fused-path exclusion (lines 5330-5339),
  fused LoRAMLP exclusion in the patch loop (line 5351).
- `unsloth/models/__init__.py`: guarded export like `falcon_h1` (lines 30-37).

## Gap (a): generation cache type

The custom text model requires a DynamicCache: its forward raises TypeError for any other
cache (modeling file, the `isinstance(past_key_values, DynamicCache)` guard) and rebuilds a
plan-sized DynamicCache only when handed an EMPTY foreign cache; a populated foreign cache
is rejected. Unsloth's `unsloth_base_fast_generate` (installed on every load,
`vision.py:4871`) forces `cache_implementation = "static"` on transformers >= 4.56 even
when `UNSLOTH_DISABLE_STATIC_GENERATION = 1` is set, so the env var alone would not have
protected this model.

Fix, in two layers:
- FastModel env chain (`loader.py:2927`): `UNSLOTH_DISABLE_STATIC_GENERATION = 1`
  (covers transformers < 4.56 and keeps the house pattern).
- `register_sensenova_loop`: `model._supports_static_cache = False`, which is the switch
  `unsloth_base_fast_generate` actually reads (`vision.py:2084`) to decline static caches,
  plus `generation_config.cache_implementation = "dynamic"` as the default for vanilla
  transformers generate and TRL.

Measured tonight: the broken decode chain ran 0.08 tok/s versus 3.77 tok/s with the
DynamicCache path pinned; KV costs 57,344 bytes per token (56 execution slots).

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

bitsandbytes does not quantize `nn.Embedding`, so the 4-bit load carries the whole bf16
token table: 248,320 x 4096 x 2 B = 2,034,237,440 bytes (1.89 GiB) beside about 5.47 GiB
of 4-bit weights, about 7.36 GiB total, which does not fit an 8 GB card once training
overhead is added.

Fix: `SenseNovaLoopInt8Embedding` re-stores the table as int8 (1,017,118,720 bytes, about
0.95 GiB) plus a bf16 per-row scale vector (0.5 MiB), dequantizing only the gathered rows
(one 4096-wide vector per token, negligible compute). The conversion stages the bf16
table through host RAM, drops the old module so the VRAM copy is freed, then moves the
int8 table onto the GPU: the card never holds both copies, and the end state is all-VRAM
with no host-RAM offload. Saved checkpoints write the original bf16
`embed_tokens.weight` key back (wrapped `save_pretrained`), so vanilla transformers
reloads them.

Measured tonight: QLoRA training peaked at 7.96 of 8.19 GiB with the int8 table in place,
and the model scored 29/30 on the first 30 HumanEval problems.

## Transformers pin: unresolved

The mapping is pinned to transformers 5.18.0, the only version with a real 4-bit load
behind it. 5.17.0 is untested; whether native `qwen3_5` support (which the remote file
subclasses) arrived earlier and 5.17 would work is unmeasured. `SUPPORTS_SENSENOVA_LOOP`
therefore states 5.18.0 as the floor, with the question left open rather than silently
widened.

## Remaining external items

- unsloth_zoo compile map entry: the loop model stays uncompiled (remote code), so
  nothing in unsloth needs it today, but a future native rewrite would need a zoo compile
  map entry for `sensenova_loop` (and its looped-layer execution plan) before any compile
  path can be enabled.
- pyproject ceiling: `pyproject.toml` does not currently require transformers 5.18+; if
  5.18 proves to be a hard requirement rather than a measurement artifact, the ceiling
  must be raised there and in any CI pins. Left untouched in this draft.

## Known limitations of this draft

- `resize_token_embeddings` recreates a plain bf16 table; the int8 conversion does not
  follow a resize. `resize_model_vocab` loads run the conversion after the resize, but
  later user-side resizes drop it.
- Only plain `nn.Embedding` tables are converted; a PEFT-wrapped embedding
  (`modules_to_save`) or an Embedding subclass keeps its own storage and forward.
- A tied-embedding variant would need the lm_head to share the dequantized table; this
  checkpoint has `tie_word_embeddings = false`, so the case is unhandled.
- Full-save paths that bypass `model.save_pretrained` (some unsloth save helpers,
  `push_to_hub` internals) would need the same dequantize step. Adapter-only saves are
  unaffected: the wrapper is installed on the transformer model, not on the PeftModel.
- Fused LoRAMLP/QKV/O fast paths are excluded for this model until proven on
  twice-executed loop layers; PEFT's own LoRA forwards carry it in the meantime.
- Offline tokenizer loads keep the broken regex (transformers' own detection needs the
  hub tag or the config fetch).
- Only `py_compile` was run on the changed files; no GPU or load verification was done in
  this draft.
