# Rag Platform — P0 contract fixtures

Real request/response pairs captured from a running Rag Platform backend, one
file per flow. Nothing here is hand-written: every field in these files was sent
or received by the backend. Regenerate with

```
node scripts/rag-platform/capture-fixtures.mjs
```

The capture covers the eight flows the plan requires — login, dataset, document,
chunk, retrieval, chat, session, stream — plus a cleanup flow that removes what
the run created.

| File | Interactions | Flow |
| --- | --- | --- |
| `auth.json` | 6 | register, login, whoami, login channels, unauthorized, logout |
| `dataset.json` | 5 | create, list, get, update, not-found |
| `document.json` | 6 | upload, list, parse, status polls |
| `chunk.json` | 2 | list, list retry |
| `retrieval.json` | 1 | `POST /api/v1/retrieval` |
| `chat.json` | 4 | create with dataset, create, list, get |
| `session.json` | 3 | create, list, get |
| `stream.json` | 1 | completion with `stream: true` |
| `cleanup.json` | 2 | delete chat, delete dataset |

## Provenance

Every file stamps what it was captured against:

```json
"captured_against": {
  "base_url": "http://127.0.0.1",
  "source_image": "infiniflow/ragflow:v0.26.4",
  "proxy_scheme": "python",
  "api_version": "v1"
}
```

## Secret handling

Fixtures are committed, so no secret may reach them. Every value passes through
the capture script's `scrub()` before it is written, and each file carries the
policy it was written under:

- request bodies — the throwaway account's password never leaves the capture
  process;
- response bodies — keys matching `api_key` / `secret` / `token` / `password` /
  `credential`, and the throwaway account's e-mail, are replaced with stable
  placeholders;
- headers — only a small allowlist is recorded at all, and `Authorization` is
  always `<redacted:authorization>`.

Placeholders in use: `<redacted:authorization>`, `<redacted:email>`,
`<redacted:password>`. The session token lives in a local variable and is never
logged or written. The account is created fresh per run with a random local part
and is only ever used against the local stack.

## Caveat 1 — three success shapes are not captured

`chunk`, `retrieval` and `stream` recorded the backend's *rejection* contract,
not its success contract:

| Fixture | What was recorded |
| --- | --- |
| `chunk.list`, `chunk.list.retry1` | `code: 0`, `chunks: []`, `doc.chunk_count: 0` |
| `retrieval.search` | `code: 100`, `LookupError('Provider  not found for model .')` |
| `stream.completion` | `code: 100`, `LookupError('No default chat model for tenant.')` |
| `chat.create_with_dataset` | `code: 102`, `The dataset … doesn't own parsed file` |

All four are the same root cause, and the parse-status poll names it:

```
document.parse_status.retry1 →
  run: "FAIL", progress: -1,
  progress_msg: "Page(1~100000001): [ERROR]Fail to bind embedding model:
                 No default embedding model is set."
```

The test deployment has no embedding model and no default chat model
configured, so the document never parses, so there are no chunks to list, no
vectors to retrieve against, and no model to complete with. The upload and the
parse *request* both succeeded (`code: 0`) — the failure is in the async parse
task, which is why `document.json` holds a valid contract while `chunk.json`
does not.

**This is not a runtime-disabled record.** `POST /api/v1/retrieval` and
`POST /api/v1/chat/completions` are both `enabled` in the coverage matrix and
both answered on the first try; they are reachable and their error envelopes are
real. What is missing is deployment configuration, not a route. The routes
therefore stay classified as reachable in
`docs/rag-platform/endpoint-coverage-matrix.md`, and only the *success* payload
shape for these three flows remains uncaptured.

`stream.json` additionally notes `Response was not text/event-stream; recorded
as a single envelope` — the request was rejected before streaming began, so no
SSE frame shape was captured either. The phase that implements the completion
client must capture the SSE envelope itself rather than infer it from this file.

To close this caveat: configure an embedding model and a default chat model on
the tenant, then re-run the capture. The four interactions above are the ones to
re-check.

## Caveat 2 — the capture image is older than local backend HEAD

`source_image` is `infiniflow/ragflow:v0.26.4`, and the commit baked into that
image is behind `/Users/baran/Desktop/rag-backend` HEAD. The route mount points
differ: the image mounts commit routes under `/datasets` + `/workspace` +
`/folders`, while local HEAD uses `/datasets` + `/workspaces`.

Per the plan, the **local backend source is the authority for the contract**, so
`docs/rag-platform/route-inventory.md` (generated from local source) is correct
and the running image is stale. The consequence for these files is narrow but
real: a fixture path that the image served may not be the path local HEAD
registers.

Re-verify every fixture against a container built from local HEAD before any
phase treats a path in these files as the contract. Until then, take the field
*shapes* from the fixtures and the *paths* from the route inventory.
