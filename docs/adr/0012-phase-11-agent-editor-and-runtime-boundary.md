# ADR 0012: Phase 11 agent editor and runtime boundary

- Status: Accepted
- Date: 2026-08-16
- Scope: Phase 11 only

## Context

The active hybrid deployment runs the pinned `v0.26.4` backend image, while the
normative local backend worktree has replaced `DELETE /api/v1/agents/:canvas_id/run`
with session-scoped `POST /api/v1/tasks/:session_id/cancel`. Both contracts must
remain visible: the former is the deployed cancellation route; the latter is a
source-only route until a matching image is deployed.

The existing frontend recipe editor models a different recipe/dataflow payload.
It does not preserve the Agent canvas invariants required by the backend:
`components.*.obj.component_name`, upstream/downstream edges, Agent globals,
history, retrieval, path and variables. Reusing it without a field-by-field
adapter would silently corrupt an Agent DSL.

Agents also combine several security-sensitive surfaces: server-sent execution
events, database passwords, MCP headers/variables, authenticated attachments,
third-party webhook callbacks and destructive lifecycle mutations.

## Decision

1. Phase 11 uses a dedicated Agent domain and typed API layer. React components
   never issue network requests directly.
2. The recipe editor is not reused. `/agents` exposes a dedicated canonical JSON
   canvas editor plus backend component catalog, input-form and debug panels.
   The minimum valid DSL is pinned in
   `fixtures/phase-11-agent-contract.json` and `EMPTY_PLATFORM_AGENT_DSL`.
3. The screen exposes CRUD, tags, draft save, publish, reset, deployed-runtime
   cancel, session-scoped cancel, run/completion SSE, document-component rerun,
   logs, session single/bulk deletion, versions, database test, file/attachment,
   webhook test/logs, MCP server lifecycle/import/test and plugin tools.
4. Mutations are single-flight. Duplicate clicks are disabled; delete/reset/
   publish-adjacent destructive actions require explicit confirmation. Abort,
   timeout, incomplete stream and unmount cleanup are first-class states.
5. Database passwords and MCP credential-bearing values remain only in local
   component state for the current form submission. They are never logged,
   persisted or included in fixtures. Attachment preview object URLs are
   revoked on replacement/unmount, and all attachment calls use the shared
   authenticated client.
6. `/api/v1/agents/:canvas_id/webhook` is an external callback, not a frontend
   action. Its six verbs are contract/security-classified only. The authenticated
   `/webhook/test` sibling is the product action and lets the user select every
   supported verb. `/api/v1/mcp` and the standalone `9382` MCP endpoints are
   protocol transports for MCP clients; MCP server management lives in the UI.
7. The deprecated `/agents/:agent_id/completions` shim remains API-only. The UI
   uses canonical `/agents/chat/completions` and does not create a second stream
   state machine.

## State machine and stream contract

The UI lifecycle is `idle → pending → succeeded|failed|aborted`, with at most one
mutation or stream pending. Run and completion streams accept native Agent SSE
events, preserve event names and ids, terminate only on `done`/`[DONE]`, and
raise a typed error if the connection closes without a terminal frame. Cancelling
the browser request aborts the reader and runs cleanup; cancelling the backend
task is a separate confirmed lifecycle action.

## Runtime boundary

The pinned image registers `DELETE /agents/:canvas_id/run`; the current local
source registers `POST /tasks/:session_id/cancel` instead. The inventory retains
the deployed route as enabled and adds the newer source route as
`runtime-disabled` with source/proxy/smoke evidence. The UI's deployed cancel
button uses the enabled route. The session cancel action is typed and visible
only because the same public task path is served by the active Python API; a
future image transition must regenerate inventory and rerun smoke before the Go
implementation is credited as enabled.

## Consequences

Agents have a dedicated, contract-preserving workspace rather than an unsafe
recipe-editor shortcut. Protocol callbacks and compatibility aliases stay
testable without being mislabeled as product screens. A future visual node
canvas may replace the JSON presentation only after an adapter proves lossless
round trips against this fixture and backend normalization tests.
