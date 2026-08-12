# ADR 0001 — The Rag Platform backend is the primary backend; the Studio backend keeps only what the platform has no concept of

* Status: Accepted
* Date: 2026-08-12
* Scope: Faz 0B and every later phase; `studio/frontend/src`
* Supersedes: nothing. Superseded by: nothing.

## Context

The frontend currently talks to one backend: the Studio server, reached through
`apiUrl()` in `studio/frontend/src/lib/api-base.ts`, which resolves to
`http://127.0.0.1:<port>` under Tauri and to a same-origin relative path in the
browser. Every feature calls it — chat, RAG, training, model management, local
inference, settings.

The migration introduces a second backend. Without a stated rule for which
backend owns what, each phase would decide case by case and the answer would
drift.

### What was measured, not assumed

`docs/rag-platform/contract-matrix.md` scans every method+path literal in
`studio/frontend/src` and classifies it against the reachable platform route set:

| | Count |
|---|---|
| Scanned frontend method+path pairs | 272 |
| Mapped to a Rag Platform endpoint | 52 |
| Mapped, but the platform has no equivalent | 28 |
| Studio-local, not migrating | 192 |
| Not yet mapped | **0** |

The 192 Studio-local pairs are not a backlog. They are capabilities the platform
does not model at all:

| Studio-local family | Endpoints |
|---|---|
| Local inference server control (load/unload, audio, images, video, sandbox) | 50 |
| Hugging Face hub cache and download management | 33 |
| Fine-tuning | 26 |
| Local model file inventory, GGUF variants, folder scanning | 23 |
| Desktop application settings | 20 |
| Synthetic data recipe studio | 14 |
| Prompt library | 8 |
| Model export / merge / GGUF | 8 |
| Studio updater and install-source metadata | 4 |
| llama.cpp updater | 2 |
| Chat-template validation | 2 |
| Usage statistics | 1 |
| Desktop process control | 1 |

The platform is a retrieval-augmented-generation server. It has no training API,
no local model loader, no GGUF conversion, no Hugging Face cache. Migrating
these is not deferred work; there is nowhere to migrate them to.

Conversely the 52 mapped pairs are the product's core: datasets, documents,
chunks, retrieval, chat assistants, sessions and completions. Those are exactly
what the platform is built for, and re-implementing them on the Studio backend
would mean writing a second RAG engine.

### The desktop dimension

Some Studio-local calls are not merely local by convention — they are local by
physics. `useNativePathLeasesSupported()`
(`features/native-intents/use-native-readiness.ts:6`) returns `false` outside
Tauri, and gates the native-path lease flow that lets the app read a file the
user picked without copying its bytes through the browser. A remote platform
server cannot read a path on the user's disk. See ADR 0006.

### The 28 mapped-but-absent pairs

These are the genuinely hard cases: a frontend feature that the platform *nearly*
supports. They are enumerated with reasons in `contract-matrix.md`, and the
recurring themes are thread-scoped documents (platform scope is the dataset),
fork, archive, whole-history export, per-message editing, cross-dataset document
listing, refresh tokens and server-side generation cancellation. Each has its own
decision record or ADR; this ADR only fixes the rule that decides where they land.

## Decision

**1. The Rag Platform backend owns RAG.** Datasets, dataset documents, chunks,
retrieval, chat assistants, sessions, session messages and completions are
served by the platform. The frontend's corresponding calls are repointed, phase
by phase, per the target-phase column of the contract matrix.

**2. The Studio backend keeps only what the platform has no concept of.** The 13
families above stay. This is a positive decision with a stated test, not a
leftover: a capability stays Studio-local **only if no platform route can serve
it**, and the reason is recorded per endpoint in the contract matrix.

**3. Neither backend is a fallback for the other.** No call tries the platform
and silently retries against Studio, or vice versa. A dual-path call has two
contracts, two auth models and two failure modes, and its behaviour depends on
which one happened to answer. Each endpoint has exactly one owner.

**4. The platform is reached through its own typed client, not through
`apiUrl()`.** Per plan §1.2: integration code lives in
`src/integrations/platform-backend`, the client is `platformRequest`, errors are
`PlatformApiError`, configuration uses the `VITE_RAG_PLATFORM_` prefix. UI
components never call either backend directly — they go through a typed service,
a domain model and an adapter. The existing `ragRequest` in
`features/rag/api/rag-api.ts`, which hardcodes `const RAG_BASE = "/api/rag"`
against the Studio server, is the shape being replaced, not extended.

**5. A capability the platform cannot serve is stated, never faked.** If a
feature has no platform path it is either implemented client-side over platform
primitives, or removed from the UI, or recorded as unsupported in ADR 0004. What
it must never do is present a control that silently does nothing or reports
success it did not achieve.

**6. Feature flags do not defer migration.** Per the user's standing
instruction, a flag may not stand in for an unimplemented endpoint. The existing
flags in `config/disabled-features.ts` (`FEATURE_IMAGES`, `FEATURE_TRAIN`,
`FEATURE_PROJECTS`, `FEATURE_VIDEO`, `FEATURE_RECIPES`, `FEATURE_EXPORT`,
`FEATURE_API_MONITOR`, all currently `false`) are pre-existing page kill
switches, not migration state, and no new flag may be added to postpone platform
work.

## Alternatives rejected

* **Studio backend proxies the platform** — every request pays two hops, the
  platform's error envelope and status codes get flattened into Studio's, and
  streaming completions would have to be re-streamed through a server that has
  no reason to be in the path.
* **Platform backend becomes the only backend** — would require implementing
  training, local model loading, GGUF conversion and Hugging Face cache
  management on a RAG server. Deletes working product features to satisfy an
  architectural preference.
* **Per-call fallback between the two** — indeterminate behaviour by
  construction. Which backend answered decides what the user gets.
* **Route everything through one generic client with a base-URL switch** — the
  two backends differ in more than host: auth (bearer opaque UUID vs Studio
  token), error envelope (`{code, message, data}` vs FastAPI `detail`), and
  status conventions. One client would have to branch on all three internally,
  which is two clients with extra steps.
* **Keep `ragRequest`/`/api/rag` and point it at the platform** — its error
  parser is FastAPI-shaped (`formatFastApiDetail`), it has no notion of the
  platform's `code` field, and its path space is Studio's. Repointing it would
  make platform errors unreadable.

## Consequences

* 52 endpoints move to the platform across phases 1–14; 192 stay Studio-local
  with a per-endpoint reason; 28 need an explicit decision each.
* The app depends on both backends at once for the whole migration. A deployment
  that omits either loses real features, so both must be reachable and both must
  be health-checked.
* Two auth systems coexist. The platform's is fixed in ADR 0002; Studio's stays
  as-is. The frontend holds two tokens and must not send either to the wrong
  host.
* `contract-matrix.mjs` re-validates every declared platform target against
  `endpoint-coverage-matrix.json` on each run, so a target that stops being
  reachable fails the build instead of silently becoming wrong documentation.
* Desktop-only capabilities (native path leases) cannot migrate at all. In a
  browser-only deployment they are absent, which is a product consequence of the
  platform being remote, not a bug.
