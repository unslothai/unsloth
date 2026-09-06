Worker 10 implemented the account-isolation test harness and performance gate on `mu/wt-10`. Final validation: **759 passed, 482 strict xfailed, zero failures or errors**. The separate timing gate passed all four 5% limits. No product code or existing shared fixtures changed.

Commits:

- `4bfec187e7`: Add account upgrade simulations, route isolation matrix, and performance gates.
- `ecfee73ef8`: Record isolation coverage and benchmark results and pin deactivated API key regression.
- The final documentation commit adds this report.

The implementation lives in the new `studio/backend/tests/multi_account/` directory and new files under `tests/studio/multi_account/`. Generated evidence is in `artifacts/`.

- The upgrade simulation seeds old auth columns, a frozen Studio schema, chats, settings, an encrypted credential, RAG metadata, and binary output/upload/sandbox/document files using SQLite and filesystem operations. A fresh subprocess then imports the real app, reads through storage APIs, checks status, and logs in as the owner.
- The downgrade simulation exercises the old named-column reads and password-update statement against the expanded auth schema. It verifies that the managed account's identity, credentials, and directory survive those owner operations.
- The route inventory imports every module below `routes/`, walks ordinary and lazy nested APIRouters, and includes hidden routes. It records 719 route/method pairs, including 109 with object-like path parameters. Thirteen have factories covering chats/messages, projects, training history, API keys, and MCP records. Each gets five credential scenarios and an additional owner-success case. Rejected requests preserve Alice's database contents.
- Login tests cover 0, 1, 2, back to 1, and back to 0 accounts; activation changes; status fields; concurrent creation; duplicate creation; and cache invalidation racing a stale count.
- Windows tests exercise reserved usernames, UUID storage components, NFC/NFD, case folding, UTF-16 lengths, and realistic MAX_PATH budgets. The budget table is in `artifacts/windows_path_budget.md`.
- The performance harness uses only local Git objects to materialize the pre-contract revision inside this clone. It measures real router calls with TestClient and verifies that its counters and 5% threshold detect added work.

Single-user regression evidence:

The upgrade probe preserves all seven seeded files byte-for-byte immediately after app import. After storage reads and owner login, all six non-auth files remain byte-identical at their original paths. No `accounts/` directory is created. Owner chats, messages, settings, the encrypted HF credential, and RAG metadata remain readable; the sandbox root remains unchanged; login succeeds; status retains its existing fields and values plus `login_mode: "single"`.

The auth database cannot remain byte-identical after adding columns or storing a login refresh token. The test instead requires exact preservation of every old owner column/value across those operations. This is the necessary interpretation of the upgrade requirement, not a claim that additive SQLite migration preserves database bytes.

The baseline is `mu/base~1`, commit `cc0cdab40e`, read from local Git. Its initialized status handler performs two `is_initialized()` reads and one password-change read; authentication performs one credential read; the former owner workspace path is `studio_root()`. Actual warm-path measurements match those counts:

| Operation | Base/head SQLite connections | Base/head SELECT queries | Base/head mkdir attempts | Directories created |
| --- | ---: | ---: | ---: | ---: |
| `/api/auth/status` | 3 / 3 | 3 / 3 | 3 / 3 | 0 / 0 |
| GET through the auth dependency | 1 / 1 | 1 / 1 | 1 / 1 | 0 / 0 |
| Owner `workspace_root()` 1,000 times | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |

The test also compares total traced SQL statement counts. Existing `mkdir(exist_ok = True)` attempts are distinguished from newly created directories. There are zero additional connections, queries, SQL statements, mkdir attempts, or created directories on these warm paths.

The timing gate runs three alternating base/head rounds. Each revision/round receives 2,000 status calls carrying an owner JWT header and 200 authenticated history listings of 100 threads, after 100 warmups per route. Status is public in production; its JWT header does not introduce an extra authentication dependency. Each reported percentile is the median of that percentile across rounds.

| Metric | Base ms | Head ms | Change |
| --- | ---: | ---: | ---: |
| Status p50 | 2.5825 | 2.6476 | +2.52% |
| Status p95 | 3.0317 | 3.1215 | +2.96% |
| History p50 | 4.0889 | 4.0533 | -0.87% |
| History p95 | 4.7257 | 4.8603 | +2.85% |

`artifacts/perf.json` retains every round and revision metadata. An earlier measurement overlapped this worker's tests and failed status p50 at +6.30%; its other three limits passed. That result is preserved in `artifacts/perf_contended.json`. The repeat ran after this worker's tests finished. No tolerance was widened and no failing measurement was discarded. Other workers' host load was not controlled.

Exact validation commands and results follow. Root-level runs used this environment:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/.tmp/home" HF_HOME="$PWD/.tmp/hf" TMPDIR="$PWD/.tmp/tmp" XDG_CACHE_HOME="$PWD/.tmp/cache" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR" "$XDG_CACHE_HOME"
```

From the repository root:

```bash
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/test_account_contract.py studio/backend/tests/multi_account -q -n 8 --timeout=330
PYTHONPATH=studio/backend python -m pytest studio/backend/tests/multi_account/test_perf_gate.py -q -n 8 --timeout=330
python tests/studio/multi_account/perf/compare.py --output artifacts/perf.json
```

The first pytest command was run three times during development: 56 passed / 60 failed / 481 xfailed; then 153 passed / 4 failed / 3 errors / 481 xfailed; then 160 passed / 481 xfailed. The failures were harness defects: an empty router mount prefix, MCP factory setup, and an invalid seeded chat model type. All were corrected. The separate performance-helper tests passed 7/7. The benchmark command first failed on that invalid fixture before collecting timings, then produced the retained contended result (3/4 limits passed), then the final result (4/4 passed).

From `studio/backend`, the final regression command used:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" XDG_CACHE_HOME="$PWD/../../.tmp/cache" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/multi_account tests/test_auth_status_bootstrap_deadline.py tests/test_auth_lookup_off_event_loop.py tests/test_desktop_auth.py tests/test_chat_history_storage.py tests/test_chat_history_routes.py tests/test_mcp_servers.py tests/test_sandbox_files_and_storage_roots.py tests/test_training_history_update.py tests/test_training_history_delete.py -q -n 8 --timeout=330
PYTHONPATH=. python -m tests.multi_account.inventory --output ../../artifacts/route_inventory.md
```

The regression command passed 759 tests with 481 strict xfails before the additional API-key regression, then **759 passed / 482 strict xfailed** in the final run. Both runs had one existing warning from the contract's intentionally rejected, unawaited coroutine. Inventory generation completed successfully.

Before every commit, the required checks were run from the repository root:

```bash
ruff check studio/backend/tests/multi_account tests/studio/multi_account
python3 scripts/enforce_kwargs_spacing.py $(rg --files studio/backend/tests/multi_account tests/studio/multi_account -g '*.py')
python3 scripts/verify_import_hoist.py
```

Final results: Ruff passed; keyword spacing required no changes; import-hoist verification reported no blockers. Additional checks passed:

```bash
python3 scripts/verify_import_hoist.py --audit $(rg --files studio/backend/tests/multi_account tests/studio/multi_account -g '*.py')
git diff --check
```

The audit covered 17 Python files with zero syntax skips, analyzer errors, or false-positive files. No frontend or Rust files changed; those suites were not run.

Integration notes and assumptions:

- `factories.py::FACTORIES` registers `module:METHOD:/path` entries. Extend `seed_resource()` for a new resource family; the matrix automatically supplies all five credential scenarios and an owner-success check. Regenerate `artifacts/route_inventory.md` after adding routes or factories.
- `inventory.py::WORKERS` is provisional because the worker-number/domain mapping was not supplied. Assumed ownership is 01 auth/accounts, 02 Studio/chat storage, 03 credentials/providers, 04 RAG/uploads/datasets, 05 training/export/recipes, 06 inference/media/GPU, 07 tools/MCP/research, and 08 installation/network/preview. Confirm these labels during integration; they are not an authoritative assignment of the other workers.
- Worker 02 is the provisional owner of the strict first-use schema regression: warming the owner's database leaves a process-wide schema-ready flag that prevents initialization of a fresh managed database. Remove the strict xfail once the storage implementation initializes each account's database.
- Worker 01 is the provisional owner of the strict deactivated-API-key regression: JWT deactivation already works, but a previously issued API key still authenticates an inactive account on the contract branch. Remove that strict xfail when the auth implementation rejects it.
- Account mutations must call `auth.policy.invalidate_account_cache()`. The fixture uses the contract's `create_initial_user()` and `get_account()`; deactivation tests apply SQL directly and invalidate explicitly because the managed-account API has not landed.
- Production integrations should continue using the frozen context/path helpers. This worker adds no production helper or required product-code call site.
- The benchmark and cost tests need local history containing `mu/base~1`; they never fetch. The standalone benchmark also accepts `--base-ref`.

Known gaps:

There are **96 uncovered object routes**, listed individually in `artifacts/route_inventory.md`. Their five missing-factory cases account for 480 of the 482 strict xfails; they fail at factory lookup and do not send requests. Completing those factories remains integration work. The two remaining xfails are the concrete schema-initialization and deactivated-API-key regressions above. Passing the current suite does not prove isolation for the uncovered routes.

The inventory covers APIRouters under `routes/`, not additional main-app endpoints, hub/picker routers, or external MCP mounts. The upgrade probe imports the real app but does not start its GPU/download/cleanup lifespan. Timing uses real auth/chat routers in a minimal app and covers warm paths, not cold startup or the full middleware stack. Frontend username-form rendering, real desktop transitions, native Windows filesystem behavior, tunnel integration, setup-code flows, account-directory retirement, and complete GPU/job lifecycle isolation still require their domain tests and integration coverage.

An old build can read the expanded auth database, but it ignores account identity, activation/setup fields, and managed directories. It cannot safely serve a multi-account installation or enforce deactivation. The downgrade test establishes owner-data compatibility only.

All changes are committed. Generated runtime scratch and the supplied untracked task-prompt file were removed; the working tree is clean.
