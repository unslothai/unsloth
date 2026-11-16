# API Model Configurations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let an API client select sequential GGUF launch configurations through existing model aliases.
**Architecture:** Store optional complete `load_config` on aliases, retain trusted configuration in target-bound ASGI scope, and make automatic switching compare and apply it under existing lifecycle gates. Preserve it through the existing UI API parser.
**Tech Stack:** Python/FastAPI/Pydantic, TypeScript, pytest and Node tests; no new production dependencies.
**Spec:** docs/superpowers/specs/2026-11-15-api-model-configurations-design.md

## Global Constraints

- Write clean, concise, idiomatic code with clear names and straightforward control flow; choose the simplest design meeting requirements.
- Keep prose/comments purposeful; avoid unrelated abstractions or file restructuring.
- Every new test body starts with a one-line behavior comment and proves a distinct behavior; parameterize shared cases.
- Python floor 3.10. No new dependencies or database migration. Existing aliases remain compatible.
- load_config has all three fields: strict integer max_seq_length 1..1048576, existing KV cache type, speculative_type auto/off/ngram; unknown keys fail.
- First version: GGUF text generation, Keep multiple models loaded Off; unsupported targets/modes fail before teardown.
- Scope metadata is target-bound and server-created; existing auth/auto-switch/loaded-only/keyless policies remain authoritative.
- GOAT two-pass review after each task and the final branch, with immutable commit ranges. No publish, merge, or upstream messages.

## Review Focus

- Concurrent different configurations of one model must recheck resident settings under gates, including failed/cancelled loads.
- A caller-supplied JSON field or nested model name cannot supply trusted configuration.
- Non-GGUF/missing/embedding targets and multi-model/loaded-only policies must not silently answer with a wrong configuration.
- Extra llama flags and stashed launch intent cannot override configured context/cache/speculation.
- Legacy UI save, catalog filtering and target authorization must not discard settings or expose inaccessible models.

### Task 1: Alias schema, persistence and trusted request metadata

**Files:** Modify `studio/backend/utils/model_aliases.py`, `studio/backend/routes/settings.py`; Test `studio/backend/tests/test_model_aliases.py`.
**Interfaces:** Produces `ModelAliasLoadConfig` Pydantic schema; `MODEL_ALIAS_CONFIG_SCOPE_KEY`; `request_alias_load_config(scope, model) -> dict`; `alias_config_matches(backend, config) -> bool`; aliases retain optional load_config. Existing alias helpers keep their behavior.

- [x] **Step 1: Write failing tests** for normalization and settings response roundtrip; parameterize invalid context, dtype, speculation, missing and extra fields. Middleware tests assert target-bound scope from a real stored alias and no metadata from forged/nested input. Catalog tests keep both configured aliases and visibility filtering. Matcher tests use real GgufLoadIntent and verify requested Auto intent.
- [x] **Step 2: Run tests to verify failure**: `.venv/bin/python .superpowers/run-tests.py studio/backend/tests/test_model_aliases.py -q --tb=short`. Expected: new behavior assertions fail.
- [x] **Step 3: Implement** small schema/helpers in model_aliases.py; add optional schema to ModelAliasEntry with exclude-none serialization; preserve configuration during normalization. Attach scope metadata only for a rewritten top-level model; metadata helper checks target equality. Matcher compares last_load_intent n_ctx/cache_type_kv/speculative_type.
- [x] **Step 4: Verify** same command. Expected: all alias tests pass.
- [x] **Step 5: Commit and GOAT review** task diff, then full changed files and consumers. Record confirmed/refuted suspicions before fixes.

### Task 2: Configuration-aware automatic switching

**Files:** Modify `studio/backend/routes/inference.py`; Test `studio/backend/tests/test_openai_auto_switch.py`.
**Interfaces:** Consumes Task 1 helpers; produces configured sequential API switching using existing loader and locks.

- [x] **Step 1: Write failing tests** using existing _FakeBackend/_LoadRecorder/_wired and GgufLoadIntent: A→B→B→A yields three loads with exact fields; load failure leaves prior intent; disabled switch refuses mismatch and reuses match; loaded-only refuses mismatch; missing/non-GGUF/multi-model targets fail before load; raw JSON cannot inject scope metadata; saved context/cache/spec extra flags cannot defeat config. Add concurrent configs test retaining existing lock/gates.
- [x] **Step 2: Run focused tests** `.venv/bin/python .superpowers/run-tests.py studio/backend/tests/test_openai_auto_switch.py -k alias_config -q --tb=short`. Expected: new cases fail because same model bypasses reconfiguration.
- [x] **Step 3: Implement** extract trusted config at hook entry, enforce supported policy/backend; configuration-aware resident fast paths and _already_serving; unresolved configured targets fail before auto-download. Overlay max_seq_length/custom_context_length/kv_cache_dtype/speculative_type before existing model_override_load_kwargs, preserving global state. Existing lifecycle gates perform actual load and post-wait comparison.
- [x] **Step 4: Verify** `.venv/bin/python .superpowers/run-tests.py studio/backend/tests/test_model_aliases.py studio/backend/tests/test_openai_auto_switch.py -q --tb=short`. Expected: all pass (baseline environment isolated by runner).
- [x] **Step 5: Commit and GOAT review** lifecycle boundaries, authorization, failure cleanup and concurrency.

### Task 3: UI roundtrip and contributor-facing usage

**Files:** Modify `studio/frontend/src/features/settings/api/model-aliases.ts`; create `studio/frontend/tests/model-aliases-config.test.ts`, `studio/API_MODEL_CONFIGURATIONS.md`.
**Interfaces:** Consume optional load_config schema; preserve its API fields in frontend parsing/saving without new UI controls.

- [x] **Step 1: Write failing Node test** import API module through existing TS test loader pattern, stub only HTTP/auth transport, assert a load→rename→save retains complete configuration and plain aliases stay unchanged. If necessary isolate pure parse helper in the existing module/test loader.
- [x] **Step 2: Run** test with a supported Node version (`uvx --from nodejs-wheel node --experimental-strip-types --test tests/model-aliases-config.test.ts` from studio/frontend). Expected: configuration lost before parser fix.
- [x] **Step 3: Implement** optional TypeScript type and preservation; document owner API JSON for fast/long aliases, selecting model, sequential reload, supported modes, authentication and hardware limits. Include PR-ready problem/result and validation in local handoff.
- [x] **Step 4: Verify** Node test; backend task-2 suite; default `.venv/bin/python -m pytest -q --tb=short` (record any environment/baseline limitations); lint/format changed Python and frontend checks if feasible.
- [x] **Step 5: Commit and GOAT review** complete task, then fresh reviewer for complete branch using GOAT plus requesting-code-review template. Fix verified blockers with failing regressions; retain local branch for user.

## Execution outcome

Implemented all three tasks. Final review identified four correctness defects; their regressions failed before the fix and pass afterward, including a real shared-account loader join. Default ML/security pytest remains unavailable in this lightweight environment because PyTorch is not installed; relevant backend and complete frontend results are recorded in the handoff.
