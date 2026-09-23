# Laya reranking

Studio can reorder retrieved document passages with the Decision API's Laya
model before they reach chat. It is off by default.

## Turning it on

In Settings > API, turn on the Decision API, then **Rerank document search**.
Reranking uses the model and device selected there and shares the one resident
copy with `/v1/systemone`, so it adds no memory beyond the Decision API itself.

A search never downloads, installs, or waits for the model. If the selected
checkpoint is downloaded but not loaded, the first eligible search starts
loading it in the background and returns the original retrieval order; later
searches are reranked once it is resident. A search also keeps the retrieval
order while a Decision API request is using the model.

Candidates are scored eight at a time. On CPU each passage of about 500 tokens
costs roughly 0.4 to 0.7 seconds on 8 cores, so the default of 20 candidates
adds several seconds to a search; choose GPU in the same section, or lower
`RAG_RERANK_CANDIDATES`, if chat turns with documents feel slow.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `RAG_RERANK_CANDIDATES` | `20` | Candidate count, bounded to 1–50. Requests for more results bypass reranking. |

## Behavior

Knowledge-base search, project/thread document search, automatic document
injection, and `/api/rag/search` use reranking in lexical, dense, and hybrid modes.
Whole-document injection, conversation recall, and web retrieval keep their
existing behavior.

Studio retrieves the candidate pool, applies the existing cosine similarity
floor, scores each question–passage pair with Laya, and returns the requested
number of passages. Automatic document injection decides whether to inject
from the unreranked results exactly as before; reranking only reorders
passages that already clear its similarity floor. Equal scores retain retrieval
order. Laya scores only affect ordering; they are not used as a confidence
threshold. Existing retrieval scores and citation identifiers are preserved,
with `rerankScore` added to reranked search results and citation metadata.

If any question–passage pair exceeds the checkpoint's context window, Studio
uses the original retrieval results for that request. The multilingual and
typed-decisions checkpoints read 1024 tokens, which fits Studio's default
500-token chunks; the English checkpoint reads 512, so most searches fall back
with it. Model failures are logged without document text and retried after a
60-second backoff.

Benchmark on your own documents before relying on the ordering. Compare against
reranking disabled and a conventional reranker using the same questions, context
budget, and answer model. Laya's relevance scores are not a guarantee that a
passage is useful or an answer is supported.
