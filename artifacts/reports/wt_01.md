Implemented worker 01 account lifecycle and password-reset isolation on `mu/wt-01`. Changes stay within the allowlist, plus this report. The frozen schema is unchanged.

Implementation commits:

- `f23744d3ed` Add owner-managed accounts and scope password resets to one account
- `b70bc3822c` Keep accounts disabled when retirement races with reactivation

The accounts API authenticates each request before applying `auth.policy.require_owner`. Managed accounts receive the same generic 403 for every operation, including requests targeting themselves, another account, or the owner. Responses expose account metadata without hashes.

| Method and path | Request | Result |
| --- | --- | --- |
| `GET /api/accounts` | None | `{accounts: [...]}` |
| `POST /api/accounts` | `{username}` | 201 with `{account, setup_code, setup_code_expires_at}` |
| `POST /api/accounts/{account_id}/setup-code` | None | New setup code and expiry; previous credentials revoked |
| `PATCH /api/accounts/{account_id}` | `{is_active}` | Updated account metadata |
| `DELETE /api/accounts/{account_id}` | None | 204 after credential revocation and directory retirement |

Account metadata contains `account_id`, `username`, `role`, `is_active`, `created_at`, and `setup_code_pending`. Usernames are casefolded and validated against the required length, character set, and reserved names. Codes contain 256 bits of randomness, are hashed at rest, expire after 60 minutes, and are consumed atomically. Their retained password hash lets the issued session complete the existing change-password flow; the code cannot log in again. Failed setup logins use the existing account and IP rate-limit buckets and generic password error.

Deactivation and setup regeneration revoke only the target's refresh tokens, API keys, and JWT signing secret. Deletion also signals only that account's registered generations and resolves all three private roots with `run_as(account, ...)`. Existing roots are renamed with UTC retirement suffixes. Symlink targets and existing retirement destinations are preserved. Filesystem failures leave the account disabled and retryable, including when reactivation races with deletion. Reusing a username creates a fresh account ID; old files, passwords, access tokens, and late refresh tokens are not inherited.

The CLI accepts `unsloth studio reset-password --username NAME`. It requires the option when multiple accounts are active, without listing accounts. Reset updates only the target's password, JWT secret, setup state, refresh tokens, and API keys. Only owner resets clear the desktop credential and owner bootstrap files. Managed password changes and logout preserve owner bootstrap state.

Tests were run from `studio/backend` with this environment; bytecode generation was disabled to keep the shared environment read-only:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
```

Exact pytest commands, in execution order:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_lifecycle.py -q -n 8 --timeout=330 > ../../.tmp/account-tests.log 2>&1
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_lifecycle.py tests/test_auth_lookup_off_event_loop.py tests/test_auth_status_bootstrap_deadline.py tests/test_change_password_policy.py tests/test_desktop_auth.py tests/test_login_rate_limit.py tests/test_reset_password_command.py tests/test_password_prompt.py tests/test_password_prompt_backstop.py -q -n 8 --timeout=330 > ../../.tmp/auth-regression-tests.log 2>&1
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_lifecycle.py tests/test_auth_lookup_off_event_loop.py tests/test_auth_status_bootstrap_deadline.py tests/test_change_password_policy.py tests/test_desktop_auth.py tests/test_login_rate_limit.py tests/test_reset_password_command.py tests/test_password_prompt.py tests/test_password_prompt_backstop.py -q -n 8 --timeout=330 > ../../.tmp/auth-regression-tests-2.log 2>&1
PYTHONPATH=. python -m pytest tests/test_credential_rotation_race.py tests/test_api_key_expiry.py tests/test_health_reports_unified_memory.py tests/test_health_holds_verdict_during_mlx_repair.py tests/test_health_answers_within_probe_budget.py -q -n 8 --timeout=330 > ../../.tmp/credential-health-tests.log 2>&1
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_lifecycle.py tests/test_auth_lookup_off_event_loop.py tests/test_auth_status_bootstrap_deadline.py tests/test_change_password_policy.py tests/test_desktop_auth.py tests/test_login_rate_limit.py tests/test_reset_password_command.py tests/test_password_prompt.py tests/test_password_prompt_backstop.py -q -n 8 --timeout=330 > ../../.tmp/auth-regression-tests-final.log 2>&1
```

| Run | Passed | Failed | Explanation |
| --- | ---: | ---: | --- |
| Initial contract and lifecycle tests | 124 | 0 | Initial implementation |
| Initial expanded regression suite | 282 | 1 | Desktop health test's router stubs lacked the new accounts module |
| Expanded suite after fixture update and additional coverage | 290 | 0 | Fixed fixture; added logout, missing-ID, and retirement checks |
| Credential races, API-key expiry, and health | 75 | 0 | Existing regression tests |
| Final auth and lifecycle suite | 292 | 0 | Added retirement-race and username-reuse credential tests |

Final verification totals 367 passing tests and zero failures. Runs containing the frozen contract report its existing unawaited-coroutine warning in `test_run_as_refuses_a_coroutine`. No frontend or Rust changes were made, so those suites were not run.

The required checks were run before each implementation commit. These commands also validate the complete changed Python file set from the repository root:

```bash
ruff check studio/backend/auth/storage.py studio/backend/routes/auth.py studio/backend/routes/accounts.py studio/backend/models/auth.py studio/backend/main.py unsloth_cli/commands/studio.py studio/backend/tests/test_account_lifecycle.py studio/backend/tests/test_desktop_auth.py
python3 scripts/enforce_kwargs_spacing.py studio/backend/auth/storage.py studio/backend/routes/auth.py studio/backend/routes/accounts.py studio/backend/models/auth.py studio/backend/main.py unsloth_cli/commands/studio.py studio/backend/tests/test_account_lifecycle.py studio/backend/tests/test_desktop_auth.py
python3 scripts/verify_import_hoist.py
python3 scripts/verify_import_hoist.py --audit studio/backend/auth/storage.py studio/backend/routes/auth.py studio/backend/routes/accounts.py studio/backend/models/auth.py studio/backend/main.py unsloth_cli/commands/studio.py studio/backend/tests/test_account_lifecycle.py studio/backend/tests/test_desktop_auth.py
python3 scripts/verify_import_hoist.py --before mu/base --after HEAD studio/backend/auth/storage.py studio/backend/routes/auth.py studio/backend/routes/accounts.py studio/backend/models/auth.py studio/backend/main.py unsloth_cli/commands/studio.py studio/backend/tests/test_account_lifecycle.py studio/backend/tests/test_desktop_auth.py
```

All checks passed. The audit checked eight files with zero analyzer errors or false positives; comparison against `mu/base` reported all eight files clean. `git diff --check` also passed.

Single-user regression evidence: tests pin the owner's login response bytes for both password-change states, retain the hidden lowercase `unsloth` login, and verify one existing credential lookup without entering managed-account authentication. Single-mode uppercase login remains rejected as before. Tests also pin the reset CLI's default output and desktop/bootstrap cleanup. Existing desktop, password-prompt, credential-race, rate-limit, auth-status, and health tests pass. The frozen contract continues to verify historical owner paths and the cached policy hot path. No path helpers or owner password-update implementation were changed.

Integration notes and assumptions:

- Frontend account management should use the routes and payloads above, address mutations by immutable `account_id`, and show returned setup codes once. Login should submit the code as `password`, then use the returned session and code with the existing change-password request.
- Multi-user detection follows the frozen policy's active-account count. Disabled accounts do not keep the installation in multi mode. `setup_code_pending` means an unconsumed code is stored, including an expired code needing regeneration. After successful consumption, interrupted setup requires the existing session or a regenerated code.
- Other backend callers should use `storage.issue_account_setup_code`, `storage.set_account_active`, and `storage.delete_account`, which handle credential revocation and policy-cache invalidation. Managed password changes use `storage.update_account_password`; the historical `storage.update_password` remains the owner path.
- `storage.delete_account(account_id, retire)` invokes `retire(AccountContext)` under an auth database write lock after initial revocation. The callback must not write to `auth.db`. The API supplies `routes.accounts.retire_account_roots`, which calls `active_generations.cancel_all(account_id)` and resolves the three roots under that account context.
- `main.py` contains exactly one added router-mount line. Its inline module import satisfies the single-line allowlist restriction.

Known gaps: cancellation is cooperative and currently covers work registered with `active_generations`. Other domains must register their jobs or integrate account-scoped cancellation and stop writers before they recreate retired paths. Directory renames are not one atomic transaction across filesystems; partial retirement remains recoverable through the disabled account and a deletion retry. Already-authenticated requests and resources outside this domain still require the other workers' account isolation and lifecycle handling. Desktop multi-account login gating and account-management UI belong to those other domains.

The report is committed separately. Temporary test artifacts and the untracked task prompt were removed; the final working tree is clean.
