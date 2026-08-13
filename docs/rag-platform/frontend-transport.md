# Rag Platform frontend transport

The browser client uses relative `/api/v1` URLs by default. While platform auth
is enabled, Vite sends that prefix to the owned method-aware hybrid nginx entry
point at `http://127.0.0.1` by default. `VITE_RAG_PLATFORM_PROXY_TARGET` may
override it. The default must not be Python port 9380: doing so bypasses active
Go routes such as login channels and OAuth callbacks. A local env file is still
required for `VITE_RAG_PLATFORM_AUTH_PUBLIC_KEY_B64`; it must contain the public
half mounted in the active backend container.

Production should keep the same relative URL and provide a same-origin reverse
proxy to the owned hybrid backend. This avoids cross-origin bearer-token
handling and gives streaming, upload limits and TLS one deployment boundary.
`VITE_RAG_PLATFORM_BASE_URL` exists for controlled non-browser/test deployments,
but an empty value is the production recommendation.

The existing Studio backend continues to use `src/lib/api-base.ts`. The Rag
Platform transport lives only under `src/integrations/platform-backend`; tokens,
timeouts and response envelopes must not cross between the two clients.

`VITE_RAG_PLATFORM_ENABLED` defaults to `true`. Setting it to `false` keeps the
Settings connection card in a disabled empty state and prevents its health
requests; it is a rollout control, not a substitute for the implemented client.

Rollback is additive: remove imports of the integration directory, remove the
`/api/v1` Vite proxy entry and restore the previous package scripts. No existing
auth, RAG or chat request is redirected by Faz 1.
