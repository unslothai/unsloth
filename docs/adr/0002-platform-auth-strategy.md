# ADR 0002 — Platform auth is a single opaque bearer token read from a response header; there is no refresh token, and the login-password encryption is obfuscation, not confidentiality

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B and every later phase; `src/integrations/platform-backend`, `features/auth`
* Supersedes: nothing. Superseded by: nothing.

## Context

The frontend already has an auth system, built for the Studio backend: an access
token plus a refresh token, both in `localStorage`, with silent rotation on 401.
The platform's auth model is different in kind, not merely in URL. Getting this
wrong produces a login screen that appears to work and a session that dies
unpredictably.

Everything below was read from `/Users/baran/Desktop/rag-backend`, and
cross-checked against `docs/rag-platform/fixtures/auth.json`, which was captured
from a running instance.

### The token is issued in a header, not in the body

`api/apps/restful_apis/user_api.py:124-134`:

```python
elif user:
    user.access_token = get_uuid()
    login_user(user)
    …
    return await construct_response(data=user.to_safe_dict(for_self=True), auth=user.get_id(), message=msg)
```

`construct_response` (`common/connection_utils.py:103-118`) puts that value in a
header and exposes it to browsers:

```python
if auth:
    response.headers["Authorization"] = auth
response.headers["Access-Control-Expose-Headers"] = "Authorization"
```

The JSON body cannot carry it, because `User.SENSITIVE_FIELDS`
(`api/db/db_models.py:680`) is `{"password", "access_token", "email"}` and
`to_safe_dict` strips them. The fixture confirms both halves: `auth.login`'s body
has no token field, and its recorded headers include `authorization`, while
`auth.whoami`'s do not.

**A client that reads `data.access_token` will always get `undefined`.**

### The wire token is not the database token

The column holds a bare UUID (`get_uuid()`), but the value sent to the client is
`User.get_id()` (`api/db/db_models.py:702-704`):

```python
def get_id(self):
    jwt = Serializer(secret_key=settings.get_secret_key())
    return jwt.dumps(str(self.access_token))
```

`Serializer` is `itsdangerous.url_safe.URLSafeTimedSerializer`
(`api/db/db_models.py:29`). So the wire token is a *signed, timestamped envelope
around* the UUID. It is not a JWT despite the local variable name, and it is not
the UUID. Verification reverses this (`api/apps/__init__.py:185-206`):

```python
jwt = Serializer(secret_key=settings.get_secret_key())
access_token = str(jwt.loads(auth_token))
…
if len(access_token.strip()) < 32: return None
user = UserService.query(access_token=access_token, status=StatusEnum.VALID.value)
```

Two consequences for the frontend: the token is opaque — nothing in it can be
parsed client-side — and it must be sent back verbatim.

### There is no expiry, and no refresh route

`jwt.loads(auth_token)` is called with **no `max_age`**. `URLSafeTimedSerializer`
embeds a timestamp but only enforces it when `max_age` is passed, and
`api/apps/__init__.py:188` is the only `loads` call in the tree. The token
therefore does not expire.

`grep -rn "refresh" api/apps/ --include="*.py"` returns only Google/Box OAuth
connector credentials (`connector_api.py:399,655`) — third-party provider
tokens, unrelated to platform sessions. **No `/auth/refresh` route exists.** The
plan's acceptance criterion ("Refresh token kullanılmayacağı açıkça
kararlaştırılmıştır") is therefore not a preference; it is the only option the
backend offers.

A session ends in exactly three ways:

1. `POST /auth/logout` (`user_api.py:274-298`) sets
   `user.access_token = f"INVALID_{secrets.token_hex(16)}"`. Server-side
   revocation is real, and `_load_user_from_session` rejects any token with that
   prefix (`api/apps/__init__.py:136`).
2. Account deletion or disabling (`user_api.py:346` applies the same
   invalidation).
3. **The signing key changes.** `get_secret_key()` caches
   `_get_or_create_secret_key()`, whose env and config branches are commented out
   (`common/settings.py:186-193`); it generates `secrets.token_hex(32)` and
   stores it in Redis under `ragflow:system:secret_key`. The upstream comment at
   `common/settings.py:180` names the failure directly: *"if REDIS evict keys due
   to lack of memory, new secret key will be generated, cause all requests 401"*.

So a mass 401 is a normal operational event, not necessarily a bug in the client.

### Three token kinds share one header

`_load_user` (`api/apps/__init__.py:144-229`) tries, in order: `AUTH_BETA`
(`APIToken.beta`), `AUTH_JWT` (the session envelope above), then `AUTH_API`
(`APIToken.token`). It accepts `Authorization: bearer <token>` case-insensitively
*and* a bare token with no scheme (`:156-163`). When no `Authorization` header is
present at all it falls back to the Quart session cookie
(`_load_user_from_session`, `:118-141`).

The frontend uses only the session-token path. The API-key paths exist for
programmatic callers and must not be conflated with login.

### The password encryption protects nothing

`login` decrypts the submitted password (`user_api.py:108-113`) with
`api/utils/crypt.py:38`:

```python
def decrypt(line):
    file_path = os.path.join(get_project_base_directory(), "conf", "private.pem")
    rsa_key = RSA.importKey(Path(file_path).read_text(), "Welcome")
    cipher = Cipher_pkcs1_v1_5.new(rsa_key)
    return cipher.decrypt(base64.b64decode(line), "Fail to decrypt password!").decode("utf-8")
```

The exact client-side transform is fixed by `crypt()`'s own docstring —
"`decrypt(crypt(input_string)) == base64(input_string)`" — so the wire value is:

```
base64( RSA_PKCS1v1_5_encrypt( base64(plaintext_password) ) )
```

and the werkzeug hash in the database is computed over the **base64** form, not
the plaintext (`user_service.py:99`: `check_password_hash(str(user.password), password)`
where `password` is what `decrypt` returned).

Upstream historically shipped a public, static key pair:

```
$ git log --format='%h %ad %an' --date=short -1 --all -- conf/private.pem
30791976d 2024-01-15 KevinHuSh
```

The matching public key was also hardcoded in upstream's web client
(`web/src/utils/index.ts:32`, used via `JSEncrypt`). That history explains the
wire contract but is not copied into Rag Platform's deliverable tree.

Faz 0 deletes `conf/private.pem` and `conf/public.pem` from the backend worktree
and ignores both paths. The derived image also removes the upstream image copies.
On first container start, `backend-entrypoint.sh` generates an RSA-2048 private
key in the named `rag-platform-key-material` volume (`0700` directory, `0600`
private key), derives the public half (`0644`) and symlinks both into
`/ragflow/conf`. Startup validates the private key and re-derives the public half
on every run, preventing a stale or mismatched pair.

The runtime-unique key removes the upstream static-key weakness, but RSA password
wrapping is still not a substitute for transport security. TLS remains mandatory
because it authenticates the server and protects the whole request and response,
not only one password field.

### What the Studio frontend does today

`features/auth/session.ts` defines `AUTH_TOKEN_KEY = "unsloth_auth_token"` and
`AUTH_REFRESH_TOKEN_KEY = "unsloth_auth_refresh_token"`, and `storeAuthTokens`
requires both. `features/auth/api.ts:147-189` implements `refreshSession()`
against `POST /api/auth/refresh` with in-flight deduplication and a
`logoutGeneration` counter. This is a competent implementation of a model the
platform does not have.

Separately, `api.ts:66-69` currently disables the 401/403 redirect:

```js
async function redirectToAuth(): Promise<void> {
  // TEMP (local dev, backend not attached): a 401/403 no longer bounces the app
  // to /login. Uncomment the block below to restore the real behavior.
  return;
```

so an expired or revoked platform session would leave the user on a broken page
rather than at the login screen. Recorded here because it directly affects the
only failure path this ADR has.

## Decision

**1. One token, no refresh, no rotation.** The platform session is a single
opaque bearer token. No refresh token is stored, no rotation is attempted, and no
`/auth/refresh` call is made. A 401 from the platform means *log in again* — it
is not a retryable condition.

**2. The token is read from the `Authorization` response header of
`POST /api/v1/auth/login`.** Not from the body. The platform client treats a
login whose response carries no `Authorization` header as a failed login, with an
explicit error, rather than proceeding tokenless.

**3. The token is opaque and is sent verbatim as `Authorization: Bearer <token>`.**
No client-side parsing, no expiry inspection, no claim reading. The backend
accepts a bare token too, but the frontend always sends the `Bearer` scheme so
one code path covers both backends' header construction.

**4. Two tokens, two storage keys, never crossed.** The platform token gets its
own key, distinct from `unsloth_auth_token`. `platformRequest` attaches only the
platform token; Studio's `authFetch` attaches only Studio's. Neither client ever
sees the other's key. Per ADR 0000 the existing `unsloth_*` keys are not renamed,
so the platform key is additive.

**5. Password encryption is implemented exactly as the backend requires, and
documented as non-protective.** The client performs
`base64(RSA_PKCS1v1_5(base64(password)))` with the public key, because the
backend will not accept anything else. The public key is a build-time constant in
our code — there is no route to fetch it, and it is already public in two places
upstream. This is written down here so nobody later mistakes it for a security
control: **the platform must be reached over TLS in any non-local deployment**,
and that is the actual mitigation.

**6. Password, plaintext or encrypted, is never logged, never stored, and never
persisted.** It lives in a local variable for the duration of the login call.
Same rule for the token: no logging of its value, not even truncated.

**7. Logout calls `POST /api/v1/auth/logout` before clearing local state.**
Server-side revocation exists and works; skipping it would leave a
non-expiring token valid forever. If the call fails, local state is cleared
anyway and the failure is surfaced — the user is logged out locally, and we do
not claim the server session was revoked when it was not.

**8. Mass 401 is a first-class, named failure.** Because the signing key lives in
Redis and is regenerated on eviction, every session can be invalidated at once
through no fault of the user. The platform client surfaces 401 as
"session ended — please sign in again", which is accurate for all three causes,
and does not retry.

**9. Login channels are read, not assumed.** `GET /api/v1/auth/login/channels`
returns the configured OAuth providers; the captured fixture returns `[]`. The
UI renders provider buttons from that response, so a deployment with no providers
shows none rather than dead buttons. The OAuth callback
(`GET /auth/oauth/<channel>/callback`) is classified `external-callback`.

**10. Password recovery uses the backend's four-step chain as-is.**
`/auth/password/forgot/captcha` → `/auth/password/forgot/otp` →
`/auth/password/forgot/otp/verify` → `/auth/password/reset`. No step is skipped
or synthesised client-side.

## Alternatives rejected

* **Reuse `features/auth/session.ts` and `refreshSession()` for the platform** —
  `storeAuthTokens` requires a refresh token that does not exist, and
  `refreshSession()` would POST to a route the platform does not serve, turning
  every 401 into a wasted round trip and a false "session restored" path.
* **Synthesise a refresh by re-POSTing `/auth/login` with stored credentials** —
  requires persisting the user's password. Directly violates the standing
  instruction not to write passwords to a persistent store, and converts one
  compromised `localStorage` into a permanent account takeover.
* **Treat the token as a JWT and read `exp` for proactive renewal** — it is an
  `itsdangerous` envelope, not a JWT; there is nothing to renew with, and the
  backend enforces no expiry anyway. Client-side "expiry" would log users out of
  still-valid sessions.
* **Share one `localStorage` key between both backends** — the two tokens are
  accepted by different hosts under different schemes. One key means the platform
  token eventually reaches the Studio server, or the reverse: sending a credential
  to a host that was never meant to receive it.
* **Fetch the RSA public key from the server at runtime** — no such route exists
  in the platform (the only `public_key` handling in `api/apps/` is Langfuse
  provider config). Studio's separate provider-key flow does fetch a key, and
  conflating the two would imply a protection the login path does not have.
* **Skip the encryption and send the plaintext password** — the backend runs
  `decrypt()` unconditionally and answers `Fail to crypt password` (`code`
  `SERVER_ERROR`); login would simply never succeed.
* **Keep the disabled `redirectToAuth()` as-is for the platform** — a revoked
  session would strand the user on a page of failing requests with no way back
  to login.

## Consequences

* Session lifetime is server-controlled and unbounded. There is no client-side
  expiry logic to get wrong, and no rotation race to deduplicate.
* Any hosted deployment **must** use TLS. The login-password encryption is
  decryptable by anyone who has the upstream repository, so without TLS the
  password is effectively in cleartext. This is a deployment obligation recorded
  here, in the same spirit as ADR 0000's AGPL §13 note.
* Redis eviction of `ragflow:system:secret_key` logs every user out
  simultaneously. Operationally this argues for a persistent secret key, which is
  a backend configuration change and therefore out of Faz 0 scope — the
  `RAGFLOW_SECRET_KEY` and `secret_key` config branches exist in
  `common/settings.py:186-193` but are commented out upstream. Recorded as a
  known limitation, not fixed here.
* The frontend holds two independent sessions. A user can be signed in to one
  backend and out of the other, and the UI must be able to say which.
* `features/auth/api.ts:66-69`'s disabled redirect is a pre-existing defect that
  becomes load-bearing once platform 401s can happen. Restoring it is scoped to
  the phase that implements platform login, not to this ADR, and it must not
  reintroduce a call to Studio's `/api/auth/status` for a platform 401.
* Because the token never expires and revocation is by database column, a token
  copied out of `localStorage` stays valid until an explicit logout. That is the
  backend's model; the frontend's contribution is to always call logout.
