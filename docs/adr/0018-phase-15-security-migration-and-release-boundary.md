# ADR 0018 — Phase 15 security, migration and release boundary

## Status

Accepted for implementation; production release remains gated until every
runtime/CI check in this ADR passes against clean protected commits.

## Decisions

1. The owned frontend bundle replaces the upstream web bundle inside the
   Phase 15 backend image. The image uses `infiniflow/ragflow:v0.26.4` as its
   Python/MCP base, builds the Go executable from the reviewed backend release
   ref, and overlays only the explicitly owned current Python files. Nginx
   serves the UI and hybrid API from one origin. Production TLS terminates at a
   trusted edge; HTTP is redirected by that edge and the application proxy
   emits HSTS, CSP, nosniff, frame, referrer and permissions policy headers.
2. Cross-origin API access is denied by default. The owned proxy never emits
   wildcard CORS and hides an upstream wildcard header. Explicit cross-origin
   integrations require a separate reviewed policy.
3. SSE proxy buffering/cache are disabled and stream/upload timeouts are
   explicit. The container healthcheck probes Python API/admin, Go API/admin
   and the hybrid proxy.
4. The browser bearer token remains session-local platform state as described
   by ADR 0002. It is never put in a URL, analytics event, persisted query
   cache, migration export or log. Same-origin deployment, CSP and dependency
   gates reduce (but cannot eliminate) XSS exposure.
5. Legacy Project/Thread migration is always dry-run first and exportable.
   Supported Chat/Session fields are written idempotently; source data is not
   deleted. Chat descriptions carry a server-side non-secret migration marker,
   while Session mappings use a per-user local resume ledger because the
   backend Session create contract has no metadata/idempotency-key field.
6. The backend has no endpoint for inserting arbitrary historic Session
   messages. Messages, archive/fork/pair/container and local sandbox fields are
   preserved in export and reported as non-migratable; they are never guessed
   into another contract.
7. Production images may only be built from a clean backend commit that is on
   `origin/main` or tagged. Dirty worktree overlay builds are rejected. An
   explicit `RAG_PLATFORM_LOCAL_SMOKE=true` path may build a separately tagged
   `phase15-local-smoke` image for local verification only; its provenance
   labels record the dirty state and it is never a release candidate.

## Consequences

- A release requires an external TLS edge or ingress with an HTTPS canary URL.
- A migration resumed in a different browser can recover Chat markers from the
  server, but cannot discover already-created Session mappings if the original
  local ledger is unavailable. The export is therefore mandatory before run.
- Backend changes currently present only in a dirty worktree must be reviewed,
  committed and protected before the release-image gate can pass.

## Rollback

Roll back to the preceding pinned frontend/backend image pair and restore the
preceding proxy configuration. Migration never deletes source records; the
export and local ledger remain available for audit or resume.
