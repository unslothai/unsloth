# ADR 0010: Platform capability registry and operational secrets

- Status: Accepted
- Date: 2026-08-16
- Scope: Faz 9 product shell and Settings operations

## Context

Rag Platform mode coexists in the frontend repository with legacy Studio pages.
Those pages include local model lifecycle, training, recipe authoring, export,
image/video generation, API monitoring and model-cache operations that the active
hybrid Rag Platform contract does not support. Static booleans had already hidden
some pages, but they did not express backend mode, navigation visibility, disabled
reasons or the difference between a supported capability and retained legacy code.

Phase 9 also exposes authenticated system status, aggregate usage, API tokens and
Langfuse configuration. Their backend responses contain fields that must not be
rendered or persisted: dependency error details, task-executor identifiers, full
listed token values and Langfuse secrets.

The Go API additionally registers `/api/v1/system/keys` and
`/api/v1/system/tokens` on the same `ListAPIKeys`, `CreateKey` and `DeleteKey`
handlers, backed by the same API-token service. The hybrid proxy selects Python
for token list/create, Go for token delete, and Go for the live `/system/keys`
alias.

## Decision

1. `platform-capabilities.ts` is the single product capability registry. It
   resolves explicit `platform`, `legacy` and `hybrid` modes and records both
   availability and navigation visibility.
2. Platform-only mode removes unsupported legacy routes from navigation and
   keeps their existing route guards from mounting network-owning pages. Agents
   remains visible as a disabled, explained destination until its planned phase;
   it has no click or network action.
3. Chat, Projects, Knowledge, Settings and the explained Agents destination form
   the platform information architecture. The user-visible product name remains
   only “Rag Platform”.
4. `/system/tokens` is the canonical API-token UI. Listed token values are mapped
   to masked display models; the raw value is retained only in component memory
   as the backend-required revoke key. A newly created token and its compatibility
   value are shown once, then discarded. Neither is written to browser persistence
   or logs.
5. `/system/keys` is an `api-only` compatibility alias. It remains in the typed
   service and contract/runtime tests, but receives no duplicate UI state machine.
6. System status adapters discard backend error strings and heartbeat keys or
   payloads. The UI receives only normalized service state, service type, latency
   and executor count. Aggregate stats expose only numeric time series.
7. Langfuse secret keys exist only in password input and request scope. Read and
   mutation responses are reduced to host, masked public key and project metadata;
   server-returned secrets are discarded immediately.
8. Operational requests share one UI error policy for 401, 403, 429, 5xx,
   timeout, abort and network failures. A global platform banner distinguishes
   disconnected/degraded state from a legitimate empty collection. Every
   background and retry request has an abort/cleanup path.

## Consequences

- Platform mode has no clickable navigation item whose page is unsupported or
  whose backend call is nonsensical.
- Legacy implementations remain available for explicit legacy/hybrid rollback
  without becoming accidental platform capabilities.
- The frontend cannot fully compensate for the backend list contract returning
  raw token values, but it minimizes exposure by never rendering or persisting
  them and by keeping the revoke key memory-only.
- A successful Langfuse runtime mutation requires real provider credentials.
  Local smoke tests therefore exercise the authenticated GET and all mutation
  auth boundaries; deterministic contract tests verify POST/PUT/DELETE bodies and
  secret redaction without writing fake tenant configuration.
