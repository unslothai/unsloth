# ADR 0014 — Faz 12 retrieval pruning must preserve engine identity

* Status: Accepted
* Date: 2026-08-17
* Scope: Faz 12 real-data document upload and chat retrieval verification
* Supersedes: nothing. Superseded by: nothing.

## Context

Faz 12's real-data smoke test uploaded and processed the 34-page Medyasoft
information-security policy PDF successfully. Elasticsearch stored 59 active
chunks with 384-dimensional vectors, and a direct KNN query returned the
expected document hit.

The chat retrieval path still returned no knowledge. The active tenant index
also contained stale chunks for deleted documents. `PruneDeletedChunks`
correctly removed those stale entries, but rebuilt `RetrievalSearchResult.Chunks`
from the public field map. That map intentionally excludes engine metadata such
as Elasticsearch `_id` and `_index`. The subsequent KNN-only rerank pass therefore
received chunks without engine identity, produced no vector scores, and rejected
otherwise relevant results at the configured similarity threshold.

Lowering the dataset threshold would hide the defect while leaving vector
reranking broken. Reprocessing the document cannot restore metadata discarded
inside the request path.

## Decision

`PruneDeletedChunks` will continue to use the normalized field map to decide
whether a chunk belongs to an existing document, but it will retain the matching
raw engine chunk in `RetrievalSearchResult.Chunks`. Ordered chunk IDs, field
maps, highlights, query vectors, options, aggregations and index names remain
aligned after filtering.

The change is deliberately backend-local and contract-neutral:

* no route, request or response field changes;
* no persistence or schema migration;
* no change to authorization or document-deletion semantics;
* no frontend feature flag or threshold workaround;
* a regression test must prove that removing one stale document preserves
  `_id`, `_index`, ordering and index names for the surviving raw chunk.

## Consequences

The second Elasticsearch KNN pass can score surviving chunks even when stale
chunks were pruned earlier in the same retrieval. Other engines retain their raw
chunk representation as well. The additional in-request map is bounded by the
already limited candidate set and is discarded after the retrieval call.

The backend source change remains a small upstream-candidate bug fix. It must be
kept separate from the user's existing Elasticsearch hybrid-filter work and must
pass the focused NLP tests plus the real browser document-question smoke test.

The fix is deployed reproducibly in `rag-platform-backend:0.26.4`: the owned
Docker build checks and applies the exact pinned-source patch, runs
`TestPruneRetrievalSearchResultPreservesRawChunkIdentity`, builds the Go server,
and labels the image with the Phase 12 runtime-fix marker. The deployed build
produced image manifest `sha256:86a2004d577a8bb3f0aad07a73368a5299b5008f0e54c20800618813bd7a21cf`.
After recreation, health, grounded Medyasoft retrieval and document-source UI
evidence all passed on that image.

## Alternatives rejected

* **Lower the dataset similarity threshold** — masks missing vector scores and
  changes relevance behavior for every query.
* **Skip stale-chunk pruning** — can expose deleted-document content and violates
  deletion semantics.
* **Rebuild chunks from the normalized field map and synthesize engine metadata**
  — cannot reliably recover `_index` across multi-index searches.
