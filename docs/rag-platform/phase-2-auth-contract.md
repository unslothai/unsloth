# Faz 2 native auth contract and runtime evidence

This document records the source/runtime decisions behind the generated route
inventory and endpoint coverage matrix. The pinned deployment is backend tag
`v0.26.4` at `cb93883f3f8c975eecb2fed81210effeb3bdb06f`; the separately maintained
backend `main` inspected for forward deltas was
`a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2` on 2026-08-12.

## Browser contract

- Password wire value is `base64(RSA_PKCS1v1_5(base64(UTF-8 password)))`.
- Login, registration and password reset return the single opaque session token
  in the `Authorization` response header. There is no platform refresh route or
  refresh token.
- Authenticated requests use `Authorization: Bearer <opaque token>`.
- `401` clears platform session state and redirects once to `/login`; it is never
  refreshed or retried. Logout always clears local state, including network
  failures.
- OAuth state and provider-code validation are performed by the active Go
  callback. The frontend accepts the callback credential only from the
  `ragflow_auth` SameSite cookie, never from the `auth` query marker, removes the
  cookie immediately, and chooses only the fixed `/chat` or `/login` return
  route.
- The RSA public key is supplied at build time and compared with the active
  backend key. It is non-secret; TLS remains the confidentiality boundary.

## Canonical UI routes

| Backend route | Runtime target | Product classification | Typed service / UI |
|---|---:|---|---|
| `POST /api/v1/auth/login` | Python 9380 | frontend-action | `loginPlatformUser`; Login |
| `POST /api/v1/auth/logout` | Python 9380 | frontend-action | `logoutPlatformUser`; account logout |
| `GET /api/v1/system/config` | Python 9380 | frontend-action | `getPlatformAuthCapabilities`; Login probe |
| `GET /api/v1/auth/login/channels` | Go 9384 | frontend-action | capability probe; OAuth buttons |
| `GET /api/v1/auth/login/:channel` | Go 9384 | frontend-action | `getPlatformOAuthLoginUrl`; provider redirect |
| `GET /api/v1/auth/oauth/:channel/callback` | Go 9384 | external-callback | `consumePlatformOAuthRedirect`; root bridge |
| `POST /api/v1/users` | Python 9380 | frontend-action | `registerPlatformUser`; Login → Hesap oluştur |
| forgot captcha → OTP → verify → reset | Python 9380 | frontend-action | four typed services; Login → Parolamı unuttum |
| `GET/PATCH /api/v1/users/me` | Python 9380 | frontend-action | session hydration; Settings → Profile identity plus email/create/update metadata; password change |
| `GET/PATCH /api/v1/users/me/models` | Python 9380 | api-only / contract-verified | Typed services and contract tests retained; product-owner-approved UI placement deferral, with negative Profile render coverage |

The six active Go `/v1/user/*` paths are `api-only` compatibility contracts.
The frontend intentionally uses their canonical `/api/v1` equivalents so it
does not create a second auth state machine.

## Source-only routes that are runtime-disabled

Backend `main` registers the following paths in
`internal/router/router_ee.go:31-39`, but the pinned v0.26.4 source does not
contain `router_ee.go`, so they are absent from the owned image and generated
hybrid proxy inventory. In `main`, every corresponding handler in
`internal/handler/user_auth_ee.go:25-58` returns `CodeNotImplemented`.

| Route | Class | Runtime-disabled evidence / UI decision |
|---|---|---|
| `GET /api/v1/auth/oauth/callback` | unsupported / runtime-disabled callback stub | source-only stub; absent in v0.26.4; no UI |
| `GET /api/v1/auth/oauth/github/callback` | unsupported / runtime-disabled static stub | source-only static stub is absent; active `:channel` callback handles the concrete URL (live 302); OAuth UI uses the active channel contract |
| `GET /api/v1/auth/oauth/lark/callback` | unsupported / runtime-disabled static stub | source-only static stub is absent; active `:channel` callback handles the concrete URL (live 302); OAuth UI uses the active channel contract |
| `GET /api/v1/auth/icbc/callback` | unsupported / runtime-disabled callback stub | source-only stub; absent in v0.26.4; no UI |
| `GET /api/v1/auth/azure/callback` | unsupported / runtime-disabled callback stub | source-only stub; absent in v0.26.4; no UI |
| `GET /api/v1/auth/azure/login` | unsupported / runtime-disabled action stub | source-only stub; absent in v0.26.4; no login choice |
| `POST /api/v1/auth/register/captcha` | unsupported / runtime-disabled action stub | source-only stub; live hybrid proxy returns 404; direct registration remains enabled |
| `POST /api/v1/auth/register/otp` | unsupported / runtime-disabled action stub | source-only stub; live hybrid proxy returns 404; no false registration step |
| `POST /api/v1/auth/register/otp/verify` | unsupported / runtime-disabled action stub | source-only stub; live hybrid proxy returns 404; no false registration step |

The active v0.26.4 channel-specific OAuth trio is not part of this disabled
set: Go source registers it at `internal/router/router.go:163-174`, implements
state-cookie + Redis validation in `internal/handler/oauth_login.go`, and the
hybrid proxy serves it on 9384. The UI renders OAuth only for channels returned
by the live `GET /api/v1/auth/login/channels` response; an empty array produces
no provider option.

The 2026-08-13 hybrid-proxy smoke returned HTTP 404 for the other seven
source-only paths and HTTP 302 for the GitHub/Lark concrete callback URLs. The
latter redirects are produced by the active parameterised callback, not by the
absent EE stubs; route inventory therefore records those two disabled source
registrations as having a reachable equivalent.

## Live UI and runtime evidence

- The owned runtime image was rebuilt and the active hybrid container recreated.
- `auth-key-contract.sh` loaded the encrypted PKCS#8 private key with the same
  PyCryptodome call as the backend and verified that its derived public-key hash
  equals the deployed public-key hash. Key material was not printed.
- In the local browser, registration through `POST /api/v1/users` navigated to
  `/chat`; a full navigation to `/chat` hydrated the same session through
  `GET /api/v1/users/me`; account-menu logout called the backend and returned to
  `/login`.
- The uniquely named smoke account and its root file, tenant link and tenant were
  removed in one exact-email database transaction after logout.
- Live `GET /api/v1/auth/login/channels` returned an empty list, so the login UI
  correctly rendered no OAuth-provider buttons.
- Plain Vite development startup was verified on port 5173. When no explicit
  proxy target is supplied, `/api/v1` defaults to the owned hybrid nginx entry
  point rather than falling through to the legacy `/api` proxy. Both capability
  probes return 200 and the login/register forms render. The gitignored local
  env carries only the active public key, never the private key.

## Automated evidence

- `auth-crypto.test.ts`: deployment-compatible RSA payload and fail-closed key
  configuration.
- `auth-api.test.ts`: fixture-backed header extraction; register, recovery,
  profile, password, tenant/model, hydration, 401 and logout contracts.
- `auth-session.test.ts` and `auth-guards.test.ts`: one-token persistence,
  protected/guest routing, one-time redirect and offline no-loop behavior.
- `oauth.test.ts`: cookie-only credential, state/cancel/error normalization,
  channel validation and fixed return paths.
- `platform-auth-form.test.tsx`: capability-gated registration/OAuth visibility
  and a complete UI login action.
