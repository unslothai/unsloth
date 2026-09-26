## Examples

A collection of lightweight end-to-end scripts to quickly verify that Unsloth works on your device.
Each example is a single self-contained Python script that uses a tiny model and a minimal number of steps, and prints a clear `[PASS]`, `[FAIL]`, or `[SKIP]` at the end.

## Available Examples

| Example | Model | What it covers |
| --- | --- | --- |
| `sft_lora.py` | `unsloth/Qwen3-0.6B` | LoRA SFT with `SFTTrainer` |
| `sft_qlora.py` | `unsloth/Qwen3-0.6B` | 4-bit QLoRA fine-tuning + merge to 16-bit |
| `dpo_lora.py` | `unsloth/Qwen3-0.6B` | LoRA DPO with `DPOTrainer` |
| `grpo_fast_inference.py` | `unsloth/Qwen3-0.6B` | GRPO with `fast_inference=True` (vLLM rollout) |
| `save_and_export.py` | `unsloth/Qwen3-0.6B` | Merge LoRA adapter, verify inference, export to GGUF |
| `vision_sft_lora.py` | `unsloth/Qwen3.5-0.8B` | Vision LoRA SFT with `FastVisionModel` |
| `moe_sft_lora.py` | `allenai/OLMoE-1B-7B-0924-Instruct` | MoE LoRA SFT with `FastModel` and `SFTTrainer` |

## Success Criteria

An example checks that the pipeline runs, not training quality.
Each example uses criteria specific to its pipeline.

For fine-tuning examples, each verifies that:
- every step's logged loss is finite
- every step's logged `grad_norm` is finite - greater than zero (to show gradients reached the LoRA
  adapter) and below a loose `1e3` divergence bound

## Requirements

Examples require a GPU to run.

Some scripts have optional dependencies:
- `sft_qlora.py` — requires `bitsandbytes`
- `grpo_fast_inference.py` — requires `vllm`
- `save_and_export.py` — GGUF export requires a `llama-quantize` build (Unsloth will try to install it)

Scripts skip cleanly with a `[SKIP]` message if a dependency is not available.

## Running an Example

An example is a simple Python script that can be run directly with:

```bash
python examples/sft_lora.py
```

No additional arguments are needed.

To run all examples:

```bash
bash examples/run_all.sh
```

The overall summary will be logged at the end.

## Adding Examples

Requirements to follow:
1. Tiny, ungated models and few steps - For model accessibility and so that examples are kept fast.
2. Validate an observable outcome - apply the success criteria above, plus any criteria specific to the pipeline (e.g. a phrase trained into a LoRA is reproduced after merging).
3. Print a clear `[PASS]` or `[FAIL]` at the end. Every failed check is listed in `[FAIL]`.
4. Self-contained — One simple file.
5. Device-agnostic — use Unsloth's `DEVICE_TYPE_TORCH`/`DEVICE_TYPE` from `unsloth.device_type`.
6. Skip with `[SKIP]` if a required backend (e.g. vLLM, bitsandbytes) is unavailable.

**Logging tags used in scripts:**

| Tag | Description |
| --- | --- |
| `[INFO]` | Progress update |
| `[SKIP]` | Skipped due to missing dependency or unsupported device |
| `[PASS]` | Outcome verified |
| `[FAIL]` | Ran but did not meet success criteria |
| `[WARNING]` | Warns user something notable happened |
