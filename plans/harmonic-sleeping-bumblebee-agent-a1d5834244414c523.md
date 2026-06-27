# Mistral + Gemma tool-call parser audit (read-only)

Audit target (branch `studio-tools-deepseek-glm-kimi`):
- unsloth/studio/backend/core/inference/tool_call_parser.py
- unsloth/studio/backend/core/tool_healing.py

This file is the audit deliverable. No source files were edited.

## Ground truth established
- HF `Mistral-7B-Instruct-v0.3` AND `Ministral-8B-Instruct-2410` chat templates both emit the
  PRE-v11 ARRAY form: `[TOOL_CALLS] [{...,"id":"<9char>"}, ...]</s>` (v0.3 has a space after the
  trigger; Ministral does not). `arguments` key (via `tool.function|tojson`), injected 9-char `id`,
  array directly followed by `</s>`. NOTE: the "ministral_args" dir is actually the array template,
  NOT [ARGS]. The `[ARGS]` / `[CALL_ID]` / v11 forms are mistral_common (v11+) only and were
  inferred from vLLM / SGLang / llama.cpp (Magistral/Small-3.2/Devstral templates were gated).
- Gemma3 (gemma-3-27b-it) HF template has NO tool markup. `<|tool_call>call:NAME{...}<tool_call|>`
  is Gemma 4; ground truth = vLLM gemma4/functiongemma + SGLang gemma4_detector + llama.cpp
  gemma4 PEG / gemma4_to_json.

## Findings (see assistant message for full detail)
