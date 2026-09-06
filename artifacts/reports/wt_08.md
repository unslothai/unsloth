Worker 08 implemented desktop and network admission isolation on `mu/wt-08`. Implementation commit: `7d9e23339b` (`Require account login for shared desktop installs and disable keyless access`).

Changes and rationale:

- Desktop login validates the local secret before checking account policy. Multiple accounts receive exactly `{"login_required":true,"login_mode":"multi"}` and no tokens. Returning to one account restores the existing owner token exchange.
- Rust accepts the new response as successful shell authentication, without provisioning another secret or treating it as a stale backend. Three new Rust tests cover the multi-account response, unchanged single-account response bytes, and malformed responses.
- The frontend opens the normal login route, clears the previous desktop session, and stops the Tauri route-guard shortcuts when login is required. A forced startup probe reports shell readiness so the existing AppProvider releases its startup screen. Ordinary API recovery, including recovery coalesced with that probe, still requires a session. `getPostAuthRoute` is unchanged.
- In multi-account mode, public auth status cannot overwrite the signed-in account's password-change flag with the owner's bootstrap state.
- One shared keyless admission gate refuses both inference and full scopes when multiple accounts exist, including loopback, LAN, approved dummy bearers, and middleware-cached settings. Previously recorded admissions are also rechecked against account policy. Stored grants remain intact and resume when the installation returns to one account. The frozen auth dependency already explicitly binds every admitted keyless request to `OWNER`; tests verify that it replaces an inherited managed-account context.
- Desktop initial-password privileges require the owner's subject as well as a desktop token. Bootstrap HTML injection is suppressed in multi-account installations, including local browsers.
- Bootstrap timeout and terminal startup gates already inspect `DEFAULT_ADMIN_USERNAME`. Their production code remains unchanged; tests prove that managed-account setup does not trigger them, and that an owner password change leaves managed credentials untouched. Tunnel and LAN process management also remain unchanged.

Validation used the supplied environment read-only. Temporary homes, caches, logs, and test files were confined to this clone. From the repository root:

```bash
source /mnt/disks/unslothai/daniel3/workspace_6/temp/venv-studio-review/bin/activate
export PYTHONDONTWRITEBYTECODE=1 UNSLOTH_STUDIO_HOME="$PWD/.tmp/home" HF_HOME="$PWD/.tmp/hf" TMPDIR="$PWD/.tmp/tmp" XDG_CACHE_HOME="$PWD/.tmp/cache" HF_HUB_OFFLINE=1
mkdir -p .tmp/home .tmp/hf .tmp/tmp .tmp/cache
cd studio/backend
```

The final backend command passed **614 tests, with zero failures**:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_desktop_network.py tests/test_desktop_auth.py tests/test_keyless_api_access.py tests/test_keyless_api_access_adversarial.py tests/test_index_bootstrap_origin.py tests/test_index_bootstrap_origin_extra.py tests/test_index_bootstrap_loopback.py tests/test_bootstrap_timeout.py tests/test_password_prompt.py tests/test_password_prompt_backstop.py tests/test_auth_status_bootstrap_deadline.py tests/test_secure_tunnel_gate.py tests/test_cloudflare_tunnel.py tests/test_bind_host_policy.py tests/test_lan_access_settings.py tests/test_quick_tunnel_streaming_routes.py tests/test_tunnel_safe_long_post.py -q -n 8 --timeout=330
```

It reported three warnings: the frozen contract's unawaited-coroutine warning and two existing duplicate OpenAPI operation-ID warnings for sandbox file routes. An earlier run of the same command passed 613 tests before the secure-banner byte comparison was added.

The targeted development command was:

```bash
PYTHONPATH=. python -m pytest tests/test_account_contract.py tests/test_account_desktop_network.py -q -n 8 --timeout=330
```

Its first run had 43 passes and 11 failures caused by incorrect new test expectations: token field order, checking dummy bearers before credential validation, and an unsupported password-update argument. After correcting those fixtures, it passed all 54 tests. The final broader run includes the additional banner test.

From `studio/frontend`, with temporary paths kept inside the clone:

```bash
export TMPDIR="$PWD/../../.tmp/tmp" XDG_CACHE_HOME="$PWD/../../.tmp/cache" npm_config_cache="$PWD/../../.tmp/npm"
node --experimental-strip-types --test tests/account-desktop-auth.test.ts
node --experimental-strip-types --test "tests/**/*.test.ts"
npx tsc --noEmit -p tsconfig.json
npx tsc --noEmit -p tsconfig.app.json --tsBuildInfoFile ../../.tmp/tsconfig.app.tsbuildinfo
```

Results: **14/14 targeted frontend tests passed; 6,889/6,889 full frontend tests passed; both TypeScript commands passed with zero errors**. Earlier runs passed 13 targeted and 6,888 full tests before adding the coalesced startup/API-recovery case. npm reported an existing unknown `min-release-age` configuration warning. The application type check overrides the build-info location to avoid writing through the shared `node_modules` symlink.

Regression sensitivity was checked by temporarily restoring the five changed Python/TypeScript implementation files from `mu/base`, keeping the new tests, then restoring the implementation in a `finally` block. These commands intentionally failed against the baseline:

```bash
# From studio/backend:
PYTHONPATH=. python -m pytest tests/test_account_desktop_network.py -q -n 8 --timeout=330
# From studio/frontend:
node --experimental-strip-types --test tests/account-desktop-auth.test.ts
```

Baseline results were **17 backend passes and 17 expected failures**, and **6 frontend passes and 8 expected failures**. This check preceded the additional secure-banner test. All implementation files were restored before final validation and commits.

The required checks were run before each commit, from the repository root:

```bash
ruff check studio/backend/main.py studio/backend/routes/auth.py studio/backend/utils/keyless_api_access.py studio/backend/tests/test_account_desktop_network.py
python3 scripts/enforce_kwargs_spacing.py studio/backend/main.py studio/backend/routes/auth.py studio/backend/utils/keyless_api_access.py studio/backend/tests/test_account_desktop_network.py
python3 scripts/verify_import_hoist.py
git diff --check
```

All passed. The spacing script normalized one new inline-import block on its first run. An additional import comparison against the staged implementation passed for all four Python files:

```bash
python3 scripts/verify_import_hoist.py --before mu/base --after "$(git write-tree)" studio/backend/main.py studio/backend/routes/auth.py studio/backend/utils/keyless_api_access.py studio/backend/tests/test_account_desktop_network.py
```

`command -v cargo` and `command -v rustc` found neither executable. Rust tests were added but **`cargo test` could not be run**.

Single-account regression evidence includes exact byte comparisons for the desktop token response, bootstrap HTML with a fixed nonce, and the secure startup banner. Existing desktop secret, refresh, initial-password, origin, tunnel gate, LAN, keyless, and password-prompt tests pass. The frozen account contract passes, including historical owner paths and its cached policy query-cost check. Existing cached desktop sessions still authenticate without an IPC exchange, refresh call, or status fetch. New backend policy checks use the contract's cached account count; normal bootstrap rendering with an already-changed password and disabled keyless scopes retain their early returns.

Integration assumptions and remaining work:

- The owner remains username `unsloth`, account ID `owner`. Account creation, deletion, activation, and deactivation must call `auth.policy.invalidate_account_cache()`; the frozen storage helpers used here already do so.
- Worker 01 must finish owner-only desktop-marker handling in `auth/authentication.py::_get_current_credential` and `is_desktop_access_token`, plus the refresh route's desktop password-change bypass. The base currently trusts `desktop=true` for any subject. This worker blocks managed tokens at `desktop-initial-password`, but the general must-change-password bypass remains outside this allowlist and is a known gap.
- The auth-form domain must render the username/setup controls from `login_mode` and populate the account's password-change flag from login/refresh responses. Those form files are outside this allowlist. This worker makes the existing login route reachable in Tauri and preserves the required `getPostAuthRoute` behavior.
- Installation-operation routes must authenticate before running `auth.policy.require_owner`, including LAN/tunnel settings and keyless settings changes. The network request matrix uses the real authentication and owner dependency in a small test app; it does not establish that other workers have attached those dependencies to every production route.
- No new Tauri command is needed. `desktop_auth` now returns either the existing token object or the login-required object. `isTauriLoginRequired()` is exported directly from `features/auth/tauri-auto-auth.ts`. Callers of `tauriAutoAuth({ force: true })` must interpret success as shell readiness; ordinary calls continue to require a session.

Quick-tunnel streaming behavior is preserved. The repository's existing streaming tests document GET responses buffering until the stream closes; first-party event streams therefore use POST, with GET fallback only on HTTP 405 for older backends. Both verbs retain authentication. The new tunnel-header matrix verifies managed JWT/API-key account binding through `Host: shared.trycloudflare.com` and `cf-connecting-ip`, owner-only denial, and account context surviving asynchronous GET/POST SSE generation. Existing tunnel-safe long-response tests also pass. These are in-process checks, not measurements of Cloudflare transport latency. No tunnel was started; the integrator must verify real first-event delivery, sustained streaming, and account isolation through the shared quick-tunnel URL.

Native desktop rendering and Rust execution remain unverified in this environment. Production edits stayed within the allowlist, with minimal hunks in the two permitted auth routes and `_inject_bootstrap`. Temporary artifacts were removed and the report was committed; the working tree is clean.
