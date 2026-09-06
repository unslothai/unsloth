Worker 02 implemented account-scoped storage on `mu/wt-02`. The storage changes are committed in `0b78dd3fb5` (`Scope storage schemas, caches, paths and usage writes to each account`). This report and the AST guard are included in the following commit. Production changes stay within the allowlist.

The strict AST guard still fails on 16 existing captures outside this worker's allowlist. All executed functional regression suites pass after updating obsolete fixtures and separating the Hub test harness from the backend harness.

Changes made:

- Hub storage roots now re-export the canonical implementations, including dataset uploads, recipe datasets and temporary roots. Shared cache paths retain their existing locations.
- All six storage schema flags became sets of resolved database paths, with mutations protected by the existing locks. Each module exposes `reset_schema_state_for_tests()`. Existing fixtures now reset sets instead of booleans.
- WAL keepers are keyed by database path. `close_wal_keeper_for(path)` closes one account's keeper; `close_wal_keeper()` closes all keepers during shutdown. Reopening one database replaces only its own keeper.
- API usage queue items carry the submitting `AccountContext`; the writer uses `run_as` for every attempt, including retries. Profile caches and invalidation are scoped by account ID.
- Settings caches, invalidation generations, remembered embedding resolutions, and HF validation caches, budgets and in-flight requests use account-scoped keys. Existing TTLs and owner credential encryption formats remain unchanged.
- A shared `LazyPath` accessor keeps imported path constants usable with path operations, `str`, `os.fspath` and explicit calls. Dataset, recipe-worker, seed-upload and plugin cache roots resolve when used. Both `list_preview_targets` and the additionally discovered `scan_checkpoints` default now resolve inside the function.
- Added 32 storage regression cases and an AST guard with six detector tests. The guard scans backend Python files outside test directories and checks imported root/path calls at module scope, including function and lambda defaults and class bodies.

Single-account evidence: all 20 frozen contract tests pass. The owner retains the historical database, assets, datasets, outputs, exports, project and temporary layouts. Existing credential, WAL, rollback-journal, settings, checkpoint and dataset tests pass. New tests verify that warm owner settings reads perform no storage queries and warm database opens do not repeat database-file realpath resolution. No account-count or policy queries were added to these paths. Exact CPU-cycle parity was not benchmarked; account-key bookkeeping remains necessary for the requested isolation.

Tests were run from `studio/backend` with this environment:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR"
```

Contract, isolation and AST guard command, run twice: initially **48 passed, 1 failed**; finally **58 passed, 1 failed** after expanding coverage. The sole failure is the strict repository scan described below.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_storage.py tests/test_account_path_imports.py -q -n 8 --timeout=330
```

Initial combined regression command: **1,959 passed, 6 failed, 2 skipped**. Four failures came from tests assigning `_schema_ready = True`; those fixtures were updated to initialized-path sets. Two failures came from the Hub harness installing a `loggers` stub during collection; running the suites separately resolves that collision.

```bash
PYTHONPATH=. python -m pytest tests/test_studio_db_write_lock_contention.py tests/test_chat_history_storage.py tests/test_chat_generation_runs.py tests/test_chat_generation_lease_compat.py tests/test_credential_secrets.py tests/test_credential_routes.py tests/test_credential_rotation_race.py tests/test_mcp_servers.py tests/test_providers_db_models.py tests/test_profile_stats.py tests/test_model_memory_settings.py tests/test_vram_budget_settings.py tests/test_embedding_model_settings.py tests/test_openai_auto_switch.py tests/test_hf_token_validation.py tests/test_rag_store.py tests/test_rag_unavailable_quiet.py tests/test_rag_embedding_identity.py tests/test_sandbox_files_and_storage_roots.py tests/test_checkpoints_scan.py tests/test_data_recipe_seed.py tests/test_dataset_upload_limits.py tests/test_data_recipe_sampling_progress.py tests/test_data_recipe_github_progress.py hub/tests/test_dataset_services.py hub/tests/test_dataset_native_drop_upload.py -q -n 8 --timeout=330
```

Separate backend regression command: **1,906 passed, 0 failed, 2 skipped**.

```bash
PYTHONPATH=. python -m pytest tests/test_studio_db_write_lock_contention.py tests/test_chat_history_storage.py tests/test_chat_generation_runs.py tests/test_chat_generation_lease_compat.py tests/test_credential_secrets.py tests/test_credential_routes.py tests/test_credential_rotation_race.py tests/test_mcp_servers.py tests/test_providers_db_models.py tests/test_profile_stats.py tests/test_model_memory_settings.py tests/test_vram_budget_settings.py tests/test_embedding_model_settings.py tests/test_openai_auto_switch.py tests/test_hf_token_validation.py tests/test_rag_store.py tests/test_rag_unavailable_quiet.py tests/test_rag_embedding_identity.py tests/test_sandbox_files_and_storage_roots.py tests/test_checkpoints_scan.py tests/test_data_recipe_seed.py tests/test_dataset_upload_limits.py tests/test_data_recipe_sampling_progress.py tests/test_data_recipe_github_progress.py -q -n 8 --timeout=330
```

The complete Hub suite passed: **682 passed, 0 failed**.

```bash
PYTHONPATH=. python -m pytest hub/tests -q -n 8 --timeout=330
```

Additional tests whose schema fixtures changed: **1,043 passed, 0 failed, 22 skipped**.

```bash
PYTHONPATH=. python -m pytest tests/test_{chat_attachments,chat_history_routes,chat_thread_settings,deep_research_handoff_simulation,desktop_auth,external_tool_truncated_and_budget,llama_cpp_tool_loop,mcp_config_import,mcp_stdio_api_key_gate,mcp_stdio_improvements,mcp_stdio_pr5863,mcp_stdio_real_server,mcp_stdio_sessions,provider_max_output_tokens_contract,rag_nudge_roster_compat,research_progress_events,research_runs_storage,research_synthesis_recovery,training_history_delete,training_provenance,training_resume,web_rank}.py -q -n 8 --timeout=330
```

Intermediate contract and storage run: **46 passed, 0 failed**, before adding the six warm-connection cases.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_storage.py -q -n 8 --timeout=330
```

Final focused regression run after avoiding redundant database realpath work: **325 passed, 0 failed**.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_storage.py tests/test_studio_db_write_lock_contention.py tests/test_chat_generation_runs.py tests/test_credential_secrets.py tests/test_mcp_servers.py tests/test_providers_db_models.py tests/test_rag_store.py -q -n 8 --timeout=330
```

Before the storage commit, the required checks passed from the repository root:

```bash
ruff check $(git diff --name-only -- '*.py') studio/backend/utils/paths/lazy.py studio/backend/tests/test_account_storage.py studio/backend/tests/test_account_path_imports.py
python3 scripts/enforce_kwargs_spacing.py $(git diff --name-only -- '*.py') studio/backend/utils/paths/lazy.py studio/backend/tests/test_account_storage.py studio/backend/tests/test_account_path_imports.py
python3 scripts/verify_import_hoist.py
git diff --check
```

Before the guard/report commit, these checks passed:

```bash
ruff check studio/backend/tests/test_account_path_imports.py
python3 scripts/enforce_kwargs_spacing.py studio/backend/tests/test_account_path_imports.py
python3 scripts/verify_import_hoist.py
git diff --check
```

No frontend or Rust files changed, so their suites were not run. The contract suite emits its existing unawaited-coroutine warning; one existing source-inspection test emits an escape-sequence warning.

Integration notes and assumptions:

- Authentication and background-job domains must bind the correct account before calling storage, settings, dataset or recipe functions. Spawned processes must explicitly restore the account context. Unbound work remains the owner's under the frozen contract.
- API-monitor completion callbacks must enter `enqueue_api_usage` in the producing account's context. The queue preserves that context from submission onward.
- Account deletion should drain or stop that account's writers, then call `close_wal_keeper_for(account_database_path)` before renaming its workspace. This worker does not implement account lifecycle coordination. Managed-account initialization can call `run_as(account, open_wal_keeper)` to engage its keeper; the existing owner lifespan calls remain valid.
- Use `reset_schema_state_for_tests()` when replacing database fixtures. The reset clears schema bookkeeping, not database content.
- Call profile invalidation in the affected account's context. Existing settings setters already invalidate only that account's memo.
- Legacy root constants remain live `LazyPath` accessors. Call the accessor or use `Path(accessor)` when a concrete path must be retained. An accessor does not carry an account across a thread or process boundary.
- Storage follows the frozen context and roots contract without adding authorization policy. Managed contexts are assumed to come from authorized authenticated requests or explicitly bound jobs. Account IDs remain immutable and are never reused.

Known gaps: the strict AST guard remains deliberately unsuppressed and reports these 16 captures. Every listed production file is outside this worker's allowlist:

| File, relative to `studio/backend` | Lines | Captured function |
| --- | --- | --- |
| `auth/storage.py` | 21 | `auth_db_path` |
| `core/export/export.py` | 452 | `outputs_root` |
| `core/export/orchestrator.py` | 720 | `outputs_root` |
| `main.py` | 219, 221 | `studio_root` |
| `run.py` | 945, 1396, 1398 | `studio_root` |
| `utils/models/model_config.py` | 695 | `studio_root` |
| `utils/models/model_config.py` | 3251 | `outputs_root` |
| `utils/models/model_config.py` | 3292 | `exports_root` |
| `utils/transformers_version.py` | 389, 390, 391, 397, 400 | `studio_root` |

The four output/export defaults require their owning domains to defer resolution. The other twelve captures refer to shared authentication or installation paths; satisfying the requested broad guard there needs an integration decision that preserves existing startup behavior. No exceptions or expected-failure markers were added to hide them.
