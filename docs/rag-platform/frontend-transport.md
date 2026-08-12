# Rag Platform frontend transport

The browser client uses relative `/api/v1` URLs by default. Development routes
that prefix through Vite read their destination from
`VITE_RAG_PLATFORM_PROXY_TARGET`; copy `studio/frontend/.env.example` to a local
`.env` when running the frontend directly.

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
