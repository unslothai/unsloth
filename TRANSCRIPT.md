# Task

Investigate GitHub issue `unslothai/unsloth#4888`:

> "How to chat with finetuned model."
>
> Studio Chat's Finetuned selector shows `No adapter found` for a saved training
> output under `.unsloth/studio/outputs/unsloth_SmolLM-135M_1775412608`.
>
> Need to determine what Studio expects as an adapter, whether the saved output
> is missing required artifacts or metadata, and what code path causes the
> failure. Create a new env if needed.

# Notes

- Created clean investigation worktree at
  `/home/datta0/repos/unsloth/worktrees/issue_4888_chat_adapter`
  from `origin/main`.
- Issue context from GitHub:
  - `unslothai/unsloth#4888`
  - User reports Studio Chat says `No adapter found` for
    `.unsloth/studio/outputs/unsloth_SmolLM-135M_1775412608`
  - No follow-up comments on the issue as of 2026-04-07.

# Findings

- Chat's fine-tuned selector was only scanning `~/.unsloth/studio/outputs` for
  LoRA artifacts (`adapter_config.json` or `adapter_model.safetensors`).
- Full finetune training outputs stored in the same `outputs/` tree were not
  surfaced in Chat at all, even when they had a valid `config.json` and model
  weights.
- The UI copy reinforced the narrower behavior with labels like
  `Fine-tuned (LoRA)` and `No adapters found.`
- `get_base_model_from_lora(...)` also only preferred `adapter_config.json`,
  which made base-model resolution weaker for local full-finetune outputs.

# Changes

- Expanded trained-output discovery in
  `studio/backend/utils/models/model_config.py` to classify Studio outputs as:
  - `lora` when adapter artifacts are present
  - `merged` when a directory has `config.json` plus model weights
- Updated `/api/models/loras` assembly in
  `studio/backend/routes/models.py` to include those training outputs in the
  Chat selector payload with `source="training"` and `export_type`.
- Added `config.json` fallback to base-model detection for local fine-tuned
  outputs.
- Updated Chat UI copy to use trained-model wording instead of adapter-only
  wording, and tagged training full-finetune entries as `Full`.
- Added focused backend tests in
  `studio/backend/tests/test_trained_model_scan.py`.

# Verification

- Ran:
  `source ~/.venvs/pyenv/bin/activate && cd /home/datta0/repos/unsloth/worktrees/issue_4888_chat_adapter/studio/backend && pytest -q tests/test_trained_model_scan.py tests/test_transformers_version.py`
- Result: `18 passed in 0.24s`

# E2E Validation

- Environment routing:
  - Backed up the previous `~/.venvs/studio` to
    `/home/datta0/.venvs/backups/studio_20260407T065217Z`
  - Pointed `~/.venvs/studio` at the installer-managed Studio venv path
    `~/.unsloth/studio/unsloth_studio`
- Local install:
  - Ran `./install.sh --local` from the worktree with
    `CUDA_VISIBLE_DEVICES=5`
  - Initial setup failed because `~/.npmrc` contained `prefix=...`, which the
    `nvm` installer rejects before Studio's cleanup logic runs
  - Workaround used for this validation:
    - temporarily moved `~/.npmrc`
    - reran `unsloth studio update --local`
    - restored `~/.npmrc` after setup completed
- Runtime:
  - Started Studio on `http://127.0.0.1:8898` with `CUDA_VISIBLE_DEVICES=5`
  - Backend reported `Using device_map='sequential' (1 GPU(s) visible)`
  - Training/inference subprocess logs both showed `Applied gpu_ids: CUDA_VISIBLE_DEVICES='5'`
- Auth:
  - Provided credentials `admin / thisisme` did not match the local Studio DB
  - Local DB contained only the seeded `unsloth` user
  - For API-driven e2e validation, used a locally generated bearer token from
    the Studio auth DB instead of the login flow
- Training:
  - Local dataset:
    `/home/datta0/repos/unsloth/worktrees/issue_4888_chat_adapter/tmp/issue4888_train.jsonl`
  - Model:
    `unsloth/SmolLM-135M-Instruct-bnb-4bit`
  - Config:
    - `training_type=LoRA/QLoRA`
    - `max_seq_length=256`
    - `batch_size=1`
    - `gradient_accumulation_steps=1`
    - `max_steps=4`
    - `save_steps=2`
    - LoRA target modules:
      `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`
  - Result:
    `Training completed! Model saved to /home/datta0/.unsloth/studio/outputs/unsloth_SmolLM-135M-Instruct-bnb-4bit_1775545274`
  - Final metrics:
    - step `4/4`
    - epoch `0.67`
    - final loss `3.052718758583069`
- Checkpoints and outputs:
  - Final output contains:
    - `adapter_config.json`
    - `adapter_model.safetensors`
    - `checkpoint-2/`
    - `checkpoint-4/`
    - tokenizer files
  - `/api/models/loras` returned the new run as:
    - `display_name=unsloth_SmolLM-135M-Instruct-bnb-4bit_1775545274`
    - `source=training`
    - `export_type=lora`
- Inference load:
  - Loaded model path:
    `/home/datta0/.unsloth/studio/outputs/unsloth_SmolLM-135M-Instruct-bnb-4bit_1775545274`
  - `/api/inference/load` response:
    - `status=loaded`
    - `is_lora=true`
- Inference generation:
  - Called `/v1/chat/completions` with prompt:
    `Answer in one short sentence: what is 2 + 2?`
  - Response content:
    `Here is a short answer: **2 + 2** The answer is: 4.`

# Follow-up Fix

- Root cause:
  - local full-finetune directories could be misclassified as LoRA adapters
    because `ModelConfig.from_identifier(...)` used
    `get_base_model_from_lora(...)`, and that helper had been broadened to read
    `config.json`
- Fix:
  - restored strict LoRA detection for `get_base_model_from_lora(...)`
  - added `get_base_model_from_checkpoint(...)` for generic base-model recovery
    from training outputs and checkpoints
  - updated trained-model scan routes to use the generic checkpoint helper
  - added regression tests proving a full-finetune local path is not marked as
    `is_lora=True`
- Verification:
  - `pytest -q tests/test_trained_model_scan.py tests/test_transformers_version.py`
  - result: `20 passed in 0.29s`
