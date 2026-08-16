# ADR 0009: Native chat stream and client-only cancellation

- Status: Accepted
- Date: 2026-08-14
- Scope: Faz 8 chat runtime

## Context

The existing frontend runtime sends OpenAI-compatible requests to
`POST /v1/chat/completions`. The hybrid Rag Platform deployment routes the
native completion contract to Go at `POST /api/v1/chat/completions`. That route
requires explicit `chat_id`, `session_id` and a question, and returns native SSE
envelopes with reasoning markers, references, usage and a terminal
`{"code":0,"data":true}` frame. The deprecated Python
`POST /api/v1/chats/<chat_id>/completions` alias is reachable, but is not the
active canonical product contract.

The backend exposes no endpoint that cancels an in-progress completion. A
browser can abort its HTTP stream, but cannot prove that server-side generation
or persistence stopped at the same time.

## Decision

1. The Rag Platform runtime uses only `POST /api/v1/chat/completions` and a
   dedicated native SSE adapter. The existing external OpenAI adapter remains a
   separate code path and is not reused to parse native frames.
2. A stream is successful only after the `data:true` terminal frame. EOF before
   that frame is an error, including after a native `final:true` payload.
3. Duplicate frames are ignored and cumulative legacy answers are reduced to
   deltas before entering the assistant runtime.
4. The browser abort signal closes the client connection and cleans up readers,
   timers and listeners. The UI explicitly states that Rag Platform has no
   server-side cancellation endpoint; it does not claim that backend generation
   stopped.
5. References are normalized into the existing document-preview citation
   model. Tokens, provider secrets, audio blobs and credentials are never
   written to persistent storage by the adapter.
6. Hybrid Elasticsearch retrieval keeps the lexical query and KNN filter as
   separate query trees. The KNN filter contains only dataset, authorization,
   availability and explicit metadata constraints; it must not require a
   literal match for the user's question. Native SSE error events are surfaced
   to the assistant runtime instead of being treated as empty successful turns.

## Consequences

- Completion, reasoning, reference and usage evolution can be tested without
  depending on OpenAI `choices[]` shapes.
- Users receive an honest stop result, but backend compute may continue after a
  client abort until the backend naturally observes disconnection or finishes.
- Generic document questions such as summarization can retrieve semantically
  relevant chunks even when their wording does not occur verbatim in a chunk.
- Backend/provider failures remain visible and retryable instead of appearing
  as a conversation with no assistant message.
- Adding true server cancellation later requires a new backend contract and a
  follow-up ADR; a feature flag cannot simulate that capability.
