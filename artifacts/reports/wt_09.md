Worker 09 implemented the login form, owner Accounts settings, browser account transitions, and multi-account permission controls on `mu/wt-09`. Final verification passed 6,921 frontend tests, including 46 new tests, and all 20 backend contract tests. TypeScript is clean.

Implementation commits:

- `464aed0aac` Add account login, owner settings and browser isolation
- `3c090c9e95` Preserve managed setup sessions through redirects and cleanup retries
- `38abf181c5` Keep owner startup free of additional account status requests
- `a7b9f00658` Match backend Unicode case folding for account usernames

The login form preserves the single-account password-only UI and posts `unsloth`. Multi-account mode renders an accessible username field and setup-code hint. Usernames are trimmed and case-folded using Python Unicode 15.1 rules, with an ASCII fast path for the owner. A single-mode 401 rechecks status and reveals the username field without showing the rejected login error when the installation became multi-account. Managed password setup follows the session requirement instead of the public owner's bootstrap state. Cleanup retries retain an issued setup session in memory so they do not consume a setup code twice.

The owner-only Accounts tab lists accounts, creates accounts, displays and copies an expiring setup code, regenerates codes, deactivates/reactivates accounts, and requires a named retirement confirmation before deletion. Codes live only in component memory. Managed accounts cannot mount the panel or find it through navigation/search, including a persisted Accounts tab selection. The client uses the specified `/api/accounts` endpoints and authenticated fetch, preserving server error messages.

`lib/account-transition.ts` compares the normalized username with `unsloth.browser-account.v1`. Missing identity means the legacy owner. A different username clears account content, deletes both recipe IndexedDB databases, commits the new session, publishes the username last, and replaces the document. The shared listener reloads another tab at most once. Failed or blocked database deletion prevents session publication and navigation.

The explicit chrome allowlist is `theme`, `palette`, `unsloth_appearance_customization`, `unsloth_locale`, `sidebar_pinned`, `sidebar_width`, `chat_settings_width`, `unsloth_sidebar_navigate_open`, `unsloth_settings_active_tab`, `unsloth_loaded_models_collapsed`, `unsloth_loaded_models_dismissed`, and `unsloth-rag-preview-width`, plus the versioned notice prefix `unsloth_web_update_dismissed:`. Other `unsloth*` and `chat-draft*` keys are removed. Unrelated keys survive. The account marker is retained until the replacement session is ready; fresh authentication and installation-policy metadata are then written for that session.

Full access disappears from the shared permission menus and confirmation component in multi-account mode. Persisted and hydrated `full` settings reset to `auto`. An installation-policy hint survives reloads and tightens policy in peer tabs. Owner-only browsers make no additional startup status request; multi-account hints alone trigger revalidation.

Single-user regression evidence:

- First and repeated `unsloth` logins perform no content removal, IndexedDB deletion, or document replacement; an existing `full` preference remains intact.
- A successful single-mode login sends one login request with username `unsloth`, without a status retry.
- Owner bootstrap routing and the single-account desktop `/chat` route remain unchanged.
- The startup test rejects any additional status fetch for an owner-only browser.
- The existing frontend suite passed, and the frozen backend tests verified historical owner paths and cached policy behavior without backend changes.

Final frontend verification used these commands from `studio/frontend`:

```bash
export TMPDIR="$PWD/../../.tmp/tmp" npm_config_cache="$PWD/../../.tmp/npm"
node --experimental-strip-types --test "tests/**/*.test.ts"
npx tsc --noEmit -p tsconfig.json
npx tsc --noEmit -p tsconfig.app.json
npx tsc --noEmit -p tsconfig.test.json
npx tsc --noEmit --module esnext --moduleResolution bundler --target ES2022 --types node --skipLibCheck e2e/multi-account.spec.ts
./node_modules/.bin/playwright test e2e/multi-account.spec.ts --list
```

The complete frontend suite passed on all four runs: 6,908/0, 6,917/0, 6,920/0, and finally 6,921/0 passed/failed, with no skips. Each final TypeScript command exited successfully with zero diagnostics. The root TypeScript project only references other projects, so the app and test projects were checked explicitly. Playwright discovery passed twice and found one integration test. The integration test itself was not executed: this clone lacks worker 01's account endpoints and a configured disposable server with an owner password. It requires `STUDIO_E2E_URL` and `STUDIO_E2E_OWNER_PASSWORD` and covers setup, account switching, cross-tab document replacement, content cleanup, and backend denial of managed account administration.

Backend verification used these exact commands from `studio/backend`:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export PYTHONDONTWRITEBYTECODE=1 UNSLOTH_STUDIO_HOME="$PWD/../../.tmp/home" HF_HOME="$PWD/../../.tmp/hf" TMPDIR="$PWD/../../.tmp/tmp" HF_HUB_OFFLINE=1
mkdir -p "$UNSLOTH_STUDIO_HOME" "$HF_HOME" "$TMPDIR"
PYTHONPATH=. python -m pytest tests/test_account_contract.py -q -n 8 --timeout=330
```

Result: 20 passed, 0 failed. The existing coroutine-rejection contract test emitted one unawaited-coroutine warning.

Focused frontend commands also ran from the clone root; log redirection is omitted below:

```bash
node --experimental-strip-types --test studio/frontend/tests/account-transition.test.ts studio/frontend/tests/accounts-api.test.ts studio/frontend/tests/login-account-modes.test.ts
node --experimental-strip-types --test studio/frontend/tests/account-transition.test.ts studio/frontend/tests/accounts-api.test.ts studio/frontend/tests/login-account-modes.test.ts studio/frontend/tests/account-settings-ui.test.ts studio/frontend/tests/account-permissions.test.ts
node --experimental-strip-types --test studio/frontend/tests/account-auth-redirect.test.ts studio/frontend/tests/auth-fetch-timezone.test.ts
node --experimental-strip-types --test studio/frontend/tests/login-account-modes.test.ts studio/frontend/tests/account-auth-redirect.test.ts studio/frontend/tests/auth-fetch-timezone.test.ts
node --experimental-strip-types --test studio/frontend/tests/login-account-modes.test.ts studio/frontend/tests/account-session-routing.test.ts
```

Their results, in order, were 25/0, 33/0, 3/0, 14/0, then 19/0. Adding the peer-tab policy test produced 19 passed and 1 failed because the storage subscription was missing. After fixing it, the following command from `studio/frontend` passed 20/0:

```bash
node --experimental-strip-types --test tests/login-account-modes.test.ts tests/account-session-routing.test.ts
```

The final Unicode normalization regression also passed 25/0 from `studio/frontend`:

```bash
node --experimental-strip-types --test tests/account-transition.test.ts tests/login-account-modes.test.ts
```

Early typecheck commands were `./node_modules/.bin/tsc --noEmit -p tsconfig.json`, `./node_modules/.bin/tsc --noEmit -p tsconfig.app.json`, and `./node_modules/.bin/tsc --noEmit -p tsconfig.test.json`. They exposed one missing Accounts button-ref entry and 11 diagnostics from an incorrect test-context type. Both issues were fixed before the successful final checks.

`git diff --check` and `python3 scripts/verify_import_hoist.py` passed before every commit. No Python files changed, so the conditional `ruff check <changed .py files>` and `python3 scripts/enforce_kwargs_spacing.py <changed .py files>` checks had no inputs. New frontend files were formatted with the installed Biome formatter. npm emitted only the existing unknown `min-release-age` configuration warning.

Integration expectations and known gaps:

- Worker 01: the client expects `GET /api/accounts` to return `{ accounts: [{ account_id, username, role, is_active, created_at }] }`. Creation and setup-code regeneration should return `{ username, setup_code, expires_at }`, with an ISO timestamp. Activation/deletion accept any successful response, including an empty body. These response bodies were not specified in the supplied endpoint contract and need confirmation when merged.
- Worker 01: login/password-change tokens retain `access_token`, `refresh_token`, and `must_change_password`, with the canonical username in JWT `sub`. Optional JWT `role` controls display policy; existing owner tokens fall back to `sub === "unsloth"`. The setup form preserves the existing change-password payload, including the original code as `current_password`; the backend must accept this with the issued setup session after consuming the code for login.
- Worker 08: publish existing guard status through `setLoginMode(status.login_mode)` from `features/auth/login-client.ts`, or reuse `fetchAuthStatus()`. This is necessary to discover an installation-mode change made in a different browser without adding requests to owner startup. Route guards must preserve managed `must_change_password` state instead of replacing it with the public owner's bootstrap flag.
- Worker 08: desktop auto-auth must use `transitionBrowserAccount(username, route, commitSession)` before publishing a new session when it can replace a managed account. The callback stores tokens/password-change state and restores policy with `setLoginMode`. A `true` return means document replacement is underway. The excluded `tauri-auto-auth.ts` and `app/auth-guards.ts` were not edited; desktop managed-login behavior still requires worker 08's integration.
- The requested username-only browser marker cannot distinguish a deleted account from a recreated account with the same username. That case needs immutable account identity in the browser contract to guarantee browser-cache retirement; backend directories already use immutable IDs.
- Accounts strings were added to the English catalog; other locales use the existing English fallback. No existing translation keys were changed.
- Backend authorization and install-wide full-access rejection remain authoritative. Account lifecycle side effects, directory retirement, and integrated desktop/browser setup cannot be verified against the frozen backend alone.

All source changes stayed within the allowlist. No code was copied from either prohibited PR. The report is committed at `artifacts/report.md`; temporary test files and the supplied untracked prompt were removed, leaving the working tree clean.
