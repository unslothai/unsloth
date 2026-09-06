Implemented account ownership for long-running Studio jobs on `mu/wt-04`. Changes stay within worker 04's allowlist.

The implementation:

- Adds shared ownership, path, credential, subprocess, stream, and retirement helpers in `core/training/account_jobs.py`.
- Tags training, diffusion, export, and recipe jobs with their starting `AccountContext`. Active ownership is released after workers and finalizers finish; completed results retain a separate account tag. Foreign callers receive neutral status or empty results and cannot cancel, reset, export another account's checkpoint, or read its metrics, logs, or streams.
- Reserves ownership during training validation and diffusion admission. A successor cannot take over while the previous account's finalizer is still writing. Training request records and cancellation tombstones also carry account identity.
- Uses `account_thread` for job threads and carries account data into training, export, recipe, and diffusion child processes. The child binds its account before importing its worker module. Restarted training pumps retain the original account, including jobs started before multi-account mode began.
- Namespaces RAG ingestion queues, workers, folder locks, and leases. Folder sync and research supervisors claim work in each account's context. Research event-wait executor calls also rebind the account. Diffusion history uses the account's TensorBoard root.
- Checks supplied local paths against the account's workspace, project, and temporary roots, including symlink resolution, snapshot hints, recipe path lists, native-drop paths, output paths, and imatrix paths. Server-resolved resource pins may additionally use the shared HF cache.
- Refuses ambient W&B and AWS credentials for managed accounts. Remote training resources and remote recipe seeds require explicit account credentials. Managed child environments lose ambient HF, AWS, W&B, and GitHub credentials; anonymous dataset and RAG Hub calls explicitly disable implicit authentication. Recipe workflow API keys now belong to the starting account.
- Gives managed accounts separate dataset-download registries and prevents them from deleting shared dataset caches.
- Implements `retire_account_jobs(account)`: blocks new work for that immutable account ID, requests cancellation, terminates captured child processes, and attempts every service even if one cancellation fails. Background storage guards prevent retired jobs from recreating renamed directories. Retirement preserves checkpoint files.

Validation used the supplied environment read-only, with `PYTHONDONTWRITEBYTECODE=1`. Studio home, HF home, temporary files, and caches were directed into this clone's `.tmp/`. Network-dependent tests ran with `HF_HUB_OFFLINE=1`.

Final results: **74/74 contract and isolation tests passed**, comprising 20 frozen-contract tests and 54 added tests. The final broad run produced **4,804 passed, 26 failed, 22 skipped**. All 26 failures exactly match the unchanged baseline. An additional service test selection produced **485 passed, 4 failed, 2 skipped**; all four failures also reproduce on the baseline.

Exact test commands and recorded results follow. Commands using `PYTHONPATH=.` ran from `studio/backend`; commands using `PYTHONPATH=studio/backend` ran from the clone root. Output redirections are omitted.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_job_accounts.py -q -n 8 --timeout=330
```

Final result: 74 passed, 0 failed; the preceding run passed 73 tests before the last teardown-ownership test was added. The frozen contract emits its existing unawaited-coroutine warning.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_job_accounts.py tests/test_training_*.py tests/test_export_*.py tests/test_diffusion_training.py tests/test_diffusion_lora_trainer.py tests/test_mlx_training_worker_config.py tests/test_s3_dataset.py tests/test_data_recipe_*.py tests/test_rag_*.py tests/test_research_*.py tests/test_dataset_check_format_missing.py tests/test_external_confirm_gate_and_saved_keys.py tests/test_credential_rotation_race.py tests/test_desktop_auth.py -q -n 8 --timeout=330
```

Final result: 4,804 passed, 26 failed, 22 skipped. The failure set exactly equals the baseline's 26 failures.

```bash
PYTHONPATH=. python -m pytest tests/test_training_*.py tests/test_export_*.py tests/test_diffusion_training.py tests/test_diffusion_lora_trainer.py tests/test_mlx_training_worker_config.py tests/test_s3_dataset.py tests/test_data_recipe_*.py tests/test_rag_*.py tests/test_research_*.py tests/test_dataset_check_format_missing.py -q -n 8 --timeout=330
```

Development result and baseline result: both 4,655 passed, 26 failed, 22 skipped, with identical failing node IDs. Baseline failures concern offline-mode assumptions in training/preflight/cache tests and CPU-build versus accelerator-detection expectations in export capability tests.

```bash
PYTHONPATH=. python -m pytest $(rg -l 'hub.services.datasets|core.data_recipe.service|core.data_recipe.huggingface|core.training.trainer' tests/test*.py) -q -n 8 --timeout=330
```

Result: 485 passed, 4 failed, 2 skipped. Baseline verification of those four failures:

```bash
PYTHONPATH=. python -m pytest tests/test_hub_token_caller_identity.py::test_an_anonymous_config_read_does_not_strip_the_process_credential 'tests/test_hub_token_caller_identity.py::test_the_config_probes_do_not_go_local_only_for_an_anonymous_caller[False-False]' tests/test_training_preflight.py::test_remote_train_fallback_keeps_auto_eval_remote tests/test_hub_token_caller_identity.py::test_seed_inspection_derives_its_policy_from_the_caller -q -n 8 --timeout=330
```

Baseline result: 0 passed, 4 failed; identical failure set. One overlaps the broad suite's failures.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_job_accounts.py tests/test_external_confirm_gate_and_saved_keys.py tests/test_credential_rotation_race.py tests/test_desktop_auth.py -q -n 8 --timeout=330
```

Development result: 147 passed, 0 failed. The final broad run includes the subsequently added owner/Alice workflow-key tests.

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_job_accounts.py::test_export_owner_single_mode_preserves_operation_and_status_bytes -q --timeout=330
```

Final result: 1 passed, 0 failed, including the explicit owner export-response assertion.

Earlier development runs, before fixes and additional coverage:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_job_accounts.py tests/test_training_before_spawn.py tests/test_training_start_idempotency.py tests/test_export_log_cursor.py tests/test_data_recipe_pump_resilience.py tests/test_research_progress_events.py tests/test_rag_job_events_queue_lifecycle.py -q -n 8 --timeout=330
```

Result: 109 passed, 4 failed. A cancellation-serialization regression was fixed; two new-test fixtures were corrected; the remaining offline failure was baseline-confirmed.

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_job_accounts.py studio/backend/tests/test_account_contract.py studio/backend/tests/test_training_start_idempotency.py -q -n 8 --timeout=330
```

Result: 80 passed, 2 failed, before the final export fixture correction; the other failure was the baseline offline case.

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_account_contract.py studio/backend/tests/test_job_accounts.py -q -n 8 --timeout=330
```

Successive development results as coverage grew: 57 passed/1 failed; 60 passed/0 failed; 64 passed/1 failed; 68 passed/0 failed. The failures were corrected new-test expectations/setup. Final coverage is the 74-pass run above.

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_account_contract.py studio/backend/tests/test_job_accounts.py studio/backend/tests/test_rag_embeddings.py studio/backend/tests/test_rag_embed_llama_server.py studio/backend/tests/test_training_start_idempotency.py -q -n 8 --timeout=330
```

Result: 235 passed, 1 failed; the failure was the baseline offline case.

```bash
PYTHONPATH=. python -m pytest tests/test_training_start_idempotency.py::test_cancelled_route_during_spawn_keeps_the_worker_result_authoritative -q --timeout=330
```

Initial baseline probe: 0 passed, 1 failed with the original `mu/base` training route. For the broader baseline comparisons, existing changed Python files were temporarily restored from `mu/base` in this clone, tested, and restored to the implementation afterward. No other clone or prohibited PR was used.

Required checks were run before commits and passed:

```bash
ruff check $(git diff mu/base --name-only -- '*.py')
python3 scripts/enforce_kwargs_spacing.py $(git diff mu/base --name-only -- '*.py')
python3 scripts/verify_import_hoist.py
git diff --check
```

Single-account regression evidence: the frozen owner-path contract passes; owner training start/status/cancel and export responses have explicit serialized golden checks; the legacy single-account multiprocessing dispatch is unchanged; arbitrary owner paths and ambient credentials remain accepted. The broad existing test suite has the same failure set and counts on the implementation and baseline. No real GPU training or hot-path timing benchmark was performed.

Integration notes and assumptions:

- Worker 01 should import `retire_account_jobs` from `core.training.account_jobs` and call it with the immutable `AccountContext` before renaming directories. From an async route, use `await asyncio.to_thread(retire_account_jobs, account)`. A retirement exception means directories must remain in place for retry. Recreating a username with a new account ID is unaffected by the retired-ID set.
- Worker 02 must supply per-path database schema initialization and the remaining dynamic storage roots. This domain intentionally leaves `storage/studio_db.py`, `storage/rag_db.py`, recipe worker `_ARTIFACT_ROOT`, and the two seed import-time roots untouched. Child dispatch now binds before importing the recipe worker. Cross-account database operation on the fully merged branch still needs integration testing.
- GPU interlocks may continue calling `is_training_active()` and diffusion `is_active()` as global liveness predicates. These deliberately remain global. HTTP status/data/control surfaces enforce account ownership; internal raw service attributes must not be serialized directly to callers.
- Dataset callers should use the account-aware service functions. The legacy exported dataset `registry` still refers to the owner's registry. The generic `hub/services/download_lifecycle.py` watcher and subprocess launcher are outside this allowlist: they receive the selected private registry and explicit HF-token policy, but their own thread boundaries still need account propagation by that domain before they perform account-private path work.
- RAG cancellation is cooperative between ingestion/sync stages; an already-running parser or embedding operation can finish before observing retirement. Storage reopening is blocked afterward. Shared embedding servers are not terminated when retiring one account.
- Managed remote training/recipe sources require explicit tokens; RAG's implicit model loading is anonymous. Managed private RAG models should be downloaded through the account-authenticated model-download flow first. Supplied local cache hints must satisfy containment; server-resolved pins can use the shared HF cache.
- CUDA/MLX training, external AWS/W&B/HF operations, and a fully merged multi-account application were not exercised. CPU subprocess tests verify account binding, private output roots, and removal of ambient credentials. No frontend or Rust files changed.

The implementation commits are `Bind long-running jobs and their results to the starting account` and `Complete account retirement and recipe credential isolation`. Temporary test files and logs were removed after recording these results.
