# ADR 0017 — Phase 14 management runtime overlay

- Status: Accepted
- Date: 2026-08-18
- Product: Rag Platform

## Context

ADR 0016 identified backend/runtime blockers that made the otherwise complete
management UI unusable: missing tenant detail/update/role routes, a no-op
ingestor shutdown, source-only AIMLAPI authorization, no explicit paired
public/embed token workflow, no beta-token abuse limit, and no authenticated
superuser runtime evidence. Runtime testing also found that admin login returned
a signed authorization value while protected middleware compared it directly
with the raw database token. Normal Go auth rejected every superuser before
tenant ownership checks, making a combined admin/tenant surface impossible.

## Decision

- Build the owned `rag-platform-backend:0.26.4` image from immutable backend
  authority commit `a0e091e75051f278ab21e7e1c2ce3d1fcccbd5a2`, overlaying only
  reviewed Phase 14 worktree files. Unrelated dirty files are excluded.
- Keep the upstream v0.26.4 Python runtime as the base, add the current AIMLAPI
  route module/utility, and use owned pure-Go analyzer/PDF adapters.
- Verify and unwrap signed admin authorization before database lookup. Raw
  access tokens are not accepted as admin credentials.
- Permit platform superusers to use their own tenant-scoped product routes.
  Service-layer membership, owner and IDOR checks remain authoritative.
- Add tenant detail/name update and member-role mutation. The owner cannot be
  demoted; only accepted non-owner members may become `normal` or `admin`.
- Use the existing admin (`token`, `beta`) pair as the public/embed lifecycle.
  The frontend base64-encodes usernames, redacts list results, reveals generated
  values once in memory, rotates create-before-revoke with cleanup, and never
  persists secrets.
- Rate-limit beta routes with a strict Redis token bucket keyed only by a
  SHA-256 digest: capacity 60, refill one/second, Redis failure fail-closed.
- Replace no-op shutdown with a fresh exact-heartbeat check and target-scoped
  NATS command; ingestor shutdown is idempotent.
- On startup create only missing Go-owned `ingestion_task` tables, with
  race-tolerant post-condition checks. Do not run broad current-main schema
  migration against the v0.26.4 database.

## Consequences

The Phase 14 routes are deployed through the owned hybrid proxy and covered by
typed services, UI paths, tests and secret-safe authenticated runtime smoke.
`acrbaran0@gmail.com` is a platform superuser and its tenant owner. Evidence
covers admin auth, tenant rollback/IDOR/owner guard, token lifecycle, beta
MCP/preview auth, request-61 rate limiting, AIMLAPI scoped poll and actual
ingestor process exit/restart.

Python-only forward alternates remain explicit runtime-disabled records when a
selected Go equivalent exists. Go-internal developer imports remain `internal`
and receive no browser action.
