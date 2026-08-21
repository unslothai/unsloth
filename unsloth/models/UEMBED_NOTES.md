# UEmbed / Qwen3.5 Backbone Constraints

This note documents known constraints when using the Qwen3.5 (`qwen3_5`)
backbone for UEmbed-style embedding fine-tuning in this repo. It is a
standalone reference; it does not change any runtime behavior.

1. bf16-only (fp16 forced off): `qwen3_5` is in the fp16 blocklist, so
   `float16` training is forced off for this arch. This excludes fp16-only
   GPUs such as the T4. Source: `unsloth/models/loader.py:125-140`
   (`_FORCE_FLOAT32_FALLBACK` entry `"qwen3_5"`).

2. Packing / padding-free disabled: `qwen3_5` uses hybrid linear-attention
   (GDN) layers that carry a recurrent gated-delta state plus a causal
   conv1d; these leak across sequence boundaries once packing flattens the
   batch. This is detected structurally (not by model name) via
   `_is_hybrid_linear_attention_model`. Source: `unsloth/trainer.py:100-112`.

3. `transformers >= 5.2` required. The UEmbed model card asks for `>= 5.4`.

4. `trust_remote_code=True` is required to load the model.

5. No `auto_map` in `config.json`: the `Qwen3_5ForEmbedding` class lives
   only in the upstream repo's source, not as a registered `AutoModel`
   entry point. Loading therefore routes through `AutoModel` to the base
   `Qwen3_5Model`, which returns `last_hidden_state` rather than a
   pooled/embedding output.

6. Output modes and save scope:
   - Dense: offset-lasttoken pooling
     (`last_index - num_eos_tokens`).
   - Sparse: SPLADE (`splade.last` / `splade.max`, `log1p(relu(...))`).
   - Save scope: LoRA and merged 16-bit save only.
   - GGUF / llama.cpp export is OUT OF SCOPE.

This is a documentation-only summary of currently observed constraints; it
does not imply full or tested support for every Qwen3.5 configuration.
