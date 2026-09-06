Worker 03 implemented the chat/settings route isolation changes on `mu/wt-03`. Code and tests are committed in `56e1c959e0` (`Isolate account chat cancellation and settings policy`). All implementation changes stay within the supplied allowlist; no storage modules were edited.

Changes and rationale:

- Added `cancel_account_run(request, run_id, supervisor_name = ...)` and routed chat-history cleanup and durable-chat cancellation through it. Multi-account requests signal only their account's active-generation events, without calling a supervisor or pending-cancel registry keyed by a bare client ID. Single-account installations retain the original supervisor callbacks and fallback behavior.
- Preserved account context in the durable-chat event-wait executor. A managed account's event stream now reads its own database instead of the owner's database.
- Guarded durable-run creation against a foreign account's active supervisor ID before committing a run. This prevents client-chosen IDs from sharing a legacy supervisor slot.
- Classified every settings route in a module-docstring table and grouped routes under shared dependencies. Personal preferences remain private. Upload-limit, helper-precache and download-transport reads use owner policy; their writes require the owner. Executable selection, cache location, model memory, VRAM, idle unloading, embedding-backend control, networking, keyless access, public-sharing controls and installation logs require the owner in multi-account mode.
- Bound keyless policy-cache refreshes to the owner's database and rejected managed-account writes through the utility. The installation-wide cache and its cache-hit path remain shared.
- Kept the owner's last-model key unchanged; managed accounts use immutable account IDs so renaming a username preserves its selection.
- Fenced the profile-statistics storage cache by account ID and database path at the route boundary. Multi-account calls serialize cache transitions; single-account calls retain the existing computation and cache.
- Prevented multi-account history clears from fencing or deleting the global search-thumbnail registry, which currently has no ownership metadata. Single-account thumbnail cleanup is unchanged.
- Left prompt and personalization storage code unchanged. Added tests for private threads, messages, attachments, exports, prompts, settings, recorded API usage, research listings, cancellation, event replay, and username reuse. Updated two structural tests for the shared helpers and made two existing embedding tests explicitly simulate online discovery with stubs.

Validation used the supplied environment read-only. Test homes, Hugging Face files and temporary files were contained in this clone. From `studio/backend`, the common setup was:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR"
```

The exact pytest commands and outcomes follow. Output was redirected to logs under `./.tmp/`.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_chat_settings.py -q -n 8 --timeout=330
```

Two runs: initially **84 passed, 4 failed**; after correcting an implementation typo and test expectations, **88 passed, 0 failed**.

```bash
PYTHONPATH=. python -m pytest \
  tests/test_account_contract.py tests/test_account_chat_settings.py \
  tests/test_chat_history_storage.py tests/test_chat_history_routes.py \
  tests/test_chat_attachments.py tests/test_chat_settings_payload.py \
  tests/test_chat_thread_settings.py tests/test_chat_generation_runs.py \
  tests/test_chat_generation_supervisor.py tests/test_chat_generation_run_lease.py \
  tests/test_chat_generation_lease_compat.py tests/test_api_profile_usage_routes.py \
  tests/test_profile_stats.py tests/test_personalization_settings.py \
  tests/test_chat_preferences_settings.py tests/test_current_date_prompt_settings.py \
  tests/test_download_transport_setting.py tests/test_media_generation_preset_settings.py \
  tests/test_keyless_api_access.py tests/test_keyless_api_access_adversarial.py \
  tests/test_last_local_model_setting.py tests/test_model_memory_settings.py \
  tests/test_vram_budget_settings.py tests/test_llama_cpp_path_settings.py \
  tests/test_openai_auto_download.py tests/test_openai_auto_switch.py \
  tests/test_preview_sharing_settings.py tests/test_remote_access_settings.py \
  tests/test_lan_access_settings.py tests/test_hf_cache_settings.py \
  tests/test_embedding_model_settings.py tests/test_embedding_model_resolve.py \
  tests/test_embedding_model_security_gate.py tests/test_debug_log_routes.py \
  tests/test_debug_log_self_feedback.py tests/test_llm_assist_startup_opt_in.py \
  tests/test_research_runs_storage.py tests/test_conversation_archive.py \
  tests/test_project_workspace_location.py tests/test_rag_linked_folders.py \
  tests/test_sandbox_files_and_storage_roots.py tests/test_parallel_slots_per_load.py \
  tests/test_llama_extra_args_compatibility.py tests/test_llama_compat_routes.py \
  tests/test_desktop_auth.py -q -n 8 --timeout=330
```

Result: **3,133 passed, 6 failed, 1 skipped** in 320.49 seconds. Four failures concerned structural assertions tied to previous helper/router names. Two embedding tests assumed online discovery despite `HF_HUB_OFFLINE=1`; both failures reproduced on the unchanged base. All six failures were resolved and their complete test modules passed in the final rerun below.

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_chat_settings.py tests/test_chat_history_routes.py tests/test_chat_generation_runs.py tests/test_profile_stats.py -q -n 8 --timeout=330
```

Two runs while adding coverage: **244 passed, 1 failed**, then **247 passed, 0 failed**. The failure was the thumbnail helper's structural assertion, subsequently updated without removing its same-threadpool-call requirement.

For the following negative-control run, the five changed production files were temporarily restored to their original `d010387b19` contents using local `git show`, then restored in a `finally` block. No other clone or external implementation was used.

```bash
PYTHONPATH=. python -m pytest \
  tests/test_account_chat_settings.py::test_event_wait_executor_retains_account_context \
  tests/test_account_chat_settings.py::test_keyless_cache_is_always_populated_from_the_owner \
  tests/test_account_chat_settings.py::test_profile_cache_cannot_follow_a_reused_username \
  tests/test_account_chat_settings.py::test_managed_last_model_key_survives_username_rename \
  tests/test_account_chat_settings.py::test_clear_history_cannot_reap_global_images_or_foreign_runs \
  tests/test_account_chat_settings.py::test_create_refuses_a_foreign_supervisor_slot_before_writing \
  tests/test_account_chat_settings.py::test_thread_cleanup_scopes_run_ids \
  'tests/test_account_chat_settings.py::test_every_owner_setting_rejects_managed_accounts[GET-/llama-cpp-path]' \
  tests/test_embedding_model_security_gate.py::test_settings_scan_scopes_module_subdirs \
  tests/test_embedding_model_security_gate.py::test_the_resolved_repo_is_what_gets_verified_and_scanned \
  -q -n 8 --timeout=330
```

Result: **14 failed, 1 passed** as a diagnostic control: 12 isolation failures demonstrated the original defects, two failures confirmed the existing offline-test assumptions, and one cancellation case already worked on the base.

Final rerun on the committed implementation:

```bash
PYTHONPATH=. python -m pytest \
  tests/test_account_contract.py tests/test_account_chat_settings.py \
  tests/test_chat_history_routes.py tests/test_chat_generation_runs.py \
  tests/test_profile_stats.py tests/test_remote_access_settings.py \
  tests/test_lan_access_settings.py tests/test_embedding_model_security_gate.py \
  tests/test_sandbox_files_and_storage_roots.py -q -n 8 --timeout=330
```

Result: **683 passed, 0 failed** in 9.57 seconds. This includes every module with a failure in the broad run. The full broad command was not repeated after those corrections. The final run retained the existing frozen-contract warning about its deliberately rejected, unawaited coroutine. The broad run also emitted an existing invalid-escape warning.

Before each commit, repository-root checks used this file list:

```bash
isolation_python_files=(
  studio/backend/routes/chat_generation_runs.py
  studio/backend/routes/chat_history.py
  studio/backend/routes/profile_stats.py
  studio/backend/routes/settings.py
  studio/backend/utils/keyless_api_access.py
  studio/backend/tests/test_account_chat_settings.py
  studio/backend/tests/test_chat_history_routes.py
  studio/backend/tests/test_embedding_model_security_gate.py
  studio/backend/tests/test_sandbox_files_and_storage_roots.py
)
ruff check "${isolation_python_files[@]}"
python3 scripts/enforce_kwargs_spacing.py "${isolation_python_files[@]}"
python3 scripts/verify_import_hoist.py
python3 scripts/verify_import_hoist.py --audit "${isolation_python_files[@]}"
git diff --check
```

Results: Ruff passed; keyword spacing was clean; the import verifier passed; the additional audit checked all nine Python files with zero analyzer errors or false positives; whitespace checks passed. No frontend or Rust files changed, so their suites were not run.

Single-account regression evidence:

- All 20 frozen-contract tests pass, including the owner's historical paths, identity, policy-cache behavior and generation behavior.
- Tests compare the owner's existing personalization row, including its timestamp and unknown fields, before and after Alice writes preferences; the row is unchanged. Owner chats, prompts and chat settings also survive Alice's use of identical IDs.
- The owner's last-model key retains the exact legacy username hash. Owner executable-settings access remains HTTP 200; managed access is HTTP 403 in multi-account mode.
- Tests preserve both legacy supervisor cancellation and the existing no-supervisor behavior. Existing history-clear tests verify owner thumbnail cleanup and its race protection.
- No database migration, owner directory relocation or personalization rewrite was introduced. Keyless cache hits and the per-event generation wait loop retain their existing paths; multi-account checks occur at route/setup boundaries. No performance benchmark was run.

Assumptions, integration notes and known gaps:

- **Worker 02 / storage:** public storage functions must resolve the bound account's database and initialize schemas per database. The current base has process-global schema-ready flags; the new route tests initialize each account database through public functions while resetting those flags in the fixture. This deliberately isolates route validation from worker 02's pending implementation. Queued API-usage writers must preserve account context; these tests validate already-recorded receipts and profile reads, not the writer's background-thread boundary.
- **Research routes:** `routes/research_runs.py` is outside this allowlist. Its listings and unknown-ID reads/cancels pass the private-database tests, but its event executor still loses account context, and its direct supervisor cancellation remains unsafe for colliding IDs. Its owner should use `cancel_account_run(request, run_id, supervisor_name = "research_supervisor")` after updating the caller's database, and carry account context into event waits with `partial(run_as, current_account(), db.wait_for_events)`.
- **Supervisors / inference:** shared supervisor task maps still require account-scoped keys and correctly bound background work. Register research cancellation events in `active_generations` for immediate cancellation; otherwise research workers observe cancellation/deletion through their database lease checks. Until supervisor maps are partitioned, durable-chat creation returns 404 when another account occupies the same active run ID. Account deletion should call `active_generations.cancel_all(account_id)`.
- **Images:** multi-account history clearing intentionally retains search thumbnails. The image domain must add ownership to registrations, persisted sidecars, reads and cleanup before an account-scoped reap can replace `_snapshot_chat_images()`'s empty snapshot. A supplied image ID alone cannot establish ownership.
- **Installation settings consumers:** the settings routes enforce the classification, but utility modules outside this allowlist still need owner-bound reads for installation policy and cache refreshes when called from managed inference/background work. This includes memory/VRAM, idle-unload/model overrides, embedding selection, upload limits, download policy and public-preview policy. Embedding selection is classified owner-only because its current backend is shared.
- **Other domains:** sandbox, project, archive and signed-preview implementations must enforce the bound account internally. The new HTTP isolation fixture stubs filesystem sandbox/archive cleanup; it does not claim to validate other workers' filesystem changes. Update checks, whisper/sd executable selection and cache cleanup have no endpoints in `routes/settings.py`; their route owners must apply `require_owner`.
- **Profile cache:** the route-level lock is a compatibility measure for the current storage cache. Worker 02 can include immutable account identity/database path in the storage fingerprint to remove the need to serialize multi-account cache transitions.

The code is committed, the report is committed separately, and the working tree is clean. The allowlist gaps above remain integration work; this branch alone is not a complete multi-account release.
