# ADR 0013: Phase 12 connectors, files and OAuth runtime boundary

- Status: Accepted
- Date: 2026-08-16
- Scope: Phase 12 only

## Context

Phase 12 combines tenant files, connector lifecycle, asynchronous indexing,
provider OAuth and three commit scopes. The local backend source is the contract
authority, while the active deployment is the pinned `v0.26.4` image behind the
owned method-aware hybrid proxy.

The active Python Google OAuth start handler contains `print(credentials)`.
Sending the uploaded client JSON to that implementation would leak a client
secret into process logs. The Go connector service exposes the same start/result
contract, generates PKCE verifier/challenge state, stores correlation state with
an expiry and does not print the credential. Conversely, the Python connector
test supports both REST API and BigQuery while the pinned Go test handler only
supports REST API.

Provider callbacks use unprefixed `/connectors/.../callback` paths. Before this
phase nginx and Vite proxied only `/api` and `/v1`, so a provider redirect could
fall through to the SPA index without reaching the backend callback handler.

## Decision

1. `/files` is the sole product route. It exposes file/folder list, search,
   upload, download, create, delete, move, rename, parent, ancestors, dataset
   linking and version history; connector CRUD/test/rebuild/logs; and
   workspace/folder/dataset commit create/list/detail/diff/tree/content/changes.
2. React components never issue network requests. Typed services own exact
   request fields, response mapping, timeout and abort behavior.
3. Google Drive, Gmail and Box OAuth start/result requests are explicit proxy
   overrides to Go `9384`. Connector test is an explicit override to Python
   `9380`, preserving BigQuery support. Connector rebuild is routed to Python
   `9380` as well: the deployed worker is Python's DB-polling
   `sync_data_source` process. The current local Python source marks an explicit
   rebuild immediately runnable; the pinned v0.26.4 implementation still
   consumes it on the connector refresh cadence, whereas the Go peer publishes
   to a task channel that this runtime does not consume. The UI first preserves
   and updates the dataset `connectors` link, because both worker selection and
   log listing join through `connector2kb`.
   Canonical file-parent lookup is also routed to Python `9380`: the Go route
   binds `:id` but its handler incorrectly reads `file_id` from the query and
   returns code 400 for every canonical call. Other routes continue through
   generated method-aware selection.
4. Nginx and Vite proxy unprefixed `/connectors/` callback paths. The frontend
   callback route intentionally lives under `/connector-oauth/:source/callback`
   so it cannot shadow the backend path.
5. The SPA bridge stores only `{source, flowId, returnTo, startedAt}` in
   `sessionStorage`. Provider credential JSON, Box secret, authorization code and
   returned credential/token payload remain in transient memory, are never
   rendered, logged, placed in URL state or persisted. Callback state must match
   the popup `window.name` or pending session correlation before the backend
   callback is called.
6. Legacy `/api/v1/file/*` aliases are API-only. The UI uses canonical
   `/api/v1/files`. Provider callback endpoints are external-callback contracts;
   their result is surfaced through the verified SPA bridge rather than a fake
   user action.
7. Commit add/modify operations obtain current content through the typed file
   download service. The UI rejects binary data and text over 1 MiB before a
   commit request; delete has no content. The Python and Go services compare
   committed `name`/`parent_id` metadata with the recursive live tree and expose
   rename and move changes explicitly. When one UI action both renames and moves
   a file, the typed adapter coalesces those two observations into one atomic
   `move` commit item carrying both old/new names and old/new parent IDs. This
   matches the database's `(commit_id, file_id)` uniqueness contract and keeps
   the resulting tree snapshot consistent.

## Runtime and security evidence

The generated proxy contains method/path overrides for all four OAuth
start/result calls, connector test and connector rebuild, plus an unprefixed
callback location.
The Phase 12 smoke test sends a unique non-secret marker in a fake local OAuth
client record and asserts that the marker is absent from container logs after
the request. It also probes the callback boundary, authenticated file and commit
routes, and performs cleanup without printing authorization material.

The compatible Phase 12 image applies the reviewed
`phase12-backend-v0.26.4.patch` to the pinned source before building the Go
server, runs the no-CGO binding and retrieval regression tests, and copies the
patched Python file-commit service into the final image. Commit routes are
explicitly sent to Python `9380` so detection and persistence use one service
implementation. Reconsidering these boundaries still requires a new
inventory/proxy/smoke run.

## Consequences

All backend-supported connector and file capabilities have a real UI path while
duplicate aliases and inbound protocol routes retain accurate classifications.
OAuth has an explicit, testable security boundary instead of relying on a
feature flag or silently accepting secret-bearing logs.
