# ADR 0016 — Phase 14 management and runtime boundary

- Status: Superseded by ADR 0017 after the Phase 14 runtime overlay
- Date: 2026-08-17
- Product: Rag Platform

## Context

Phase 14 combines two independently secured surfaces: the normal bearer session
used by tenant/user routes and the admin server's opaque `Authorization` value.
At the time of this decision the active deployment was hybrid and pinned to backend image `0.26.4`, while the
normative local backend worktree is ahead of that image. Route names and even
implemented behavior therefore cannot be inferred from the image tag alone.

The following source/runtime findings are binding:

1. `internal/admin/router.go` mounts Go admin at `/api/v1/admin`; protected
   routes require the admin middleware. `POST /admin/login` returns an opaque
   authorization header, not a normal bearer session.
2. `internal/admin/handler.go:1125-1140` implements `ShutdownIngestor` by
   generating a task id, but the only `ingestionManager.SubmitTask` block is
   commented out. The handler returns success without requesting shutdown.
3. The current `internal/router/router.go` marks tenant chunk/metadata store and
   `dev_insert_*_from_file` routes `Internal API only for GO`. The active runtime
   still recognizes the old non-`dev_` spelling behind authentication, while
   hybrid and direct probes return 404 for the current `dev_` spelling.
4. The pinned runtime serves underscore compilation-template-group routes. The
   local worktree declares the renamed hyphen form; the active hybrid proxy
   returns 404 for that worktree-only route.
5. The active runtime serves beta-token `POST /api/v1/mcp` and rejects an
   anonymous call with envelope code 401. Selected Python document preview and
   thumbnail routes accept JWT, API-key, and beta-token auth and deny anonymous
   requests; the Go alternates are shadowed by Python 9380. The local worktree
   additionally defines AIMLAPI device-authorization start/poll routes; those
   declarations are absent from the pinned v0.26.4 target and remain
   source-only. There is still no separate public/embed token
   create/rotate/revoke contract.
   Tenant source exposes list/invite/member remove/invite acceptance, but no
   separate tenant create/detail/update or tenant-role mutation contract.

## Decision

- Add one authenticated `/management` product route with separate System/admin,
  tenant/team, bot/channel/template, and compatibility areas.
- Require the already authenticated platform principal to have `superuser=true`
  before rendering admin reauthentication. This is only a UI guard; every admin
  request also passes the backend admin authorization middleware.
- Encrypt the admin password with the backend public key, keep the resulting
  opaque token in component memory only, send it as a raw authorization header,
  and clear it on logout/unmount. Never persist or log passwords, tokens,
  provider credentials, or sandbox keys.
- Route all requests through the typed management adapter. Components do not
  call the network directly. Responses are recursively redacted before render.
- Do not retry mutations. Abort the preceding identical UI execution, require
  `ONAYLA` plus an audit reason for dangerous operations, and refresh the
  management health snapshot after completion.
- Keep compatibility protocol endpoints outside the Phase 8 chat transport.
  They are classified `api-only` with contract/security evidence and receive no
  browser credential store.
- Classify explicitly Go-internal routes as `internal`, including when the old
  spelling remains reachable. Record worktree-only spellings as
  `runtime-disabled` with source, proxy, and direct/hybrid smoke evidence.
- Do not expose `DELETE /admin/ingestors`: returning a generated task id is not
  proof of shutdown. It remains a functional `runtime-disabled` blocker.
- Keep the active beta MCP route API-only with authorization contract evidence.
  Do not expose worktree-only AIMLAPI routes against an older
  runtime, and do not invent public/embed token lifecycle, tenant update,
  or tenant-role endpoints. These runtime/contract gaps are blockers rather
  than feature flags.

## Consequences at the time

The reachable Phase 14 inventory has a real UI path, typed adapter, explicit
permission/error/loading/empty/abort states, confirmation, audit, and redaction.
Source-only and internal surfaces remain visible in the inventory without fake
screens. Phase 14 cannot be marked complete until the backend supplies and
deploys the missing contracts and fixes ingestor shutdown, followed by
authenticated admin, IDOR, revoke, rate-limit, abuse, and rollback E2E evidence.

ADR 0017 records how these blockers were closed without changing the upstream
base-image identity or persisting credentials.
