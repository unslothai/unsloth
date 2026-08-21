# Automatic context compaction

Unsloth Studio automatically keeps long local chats inside the active model's
context window. The frontend opts local GGUF and Apple Silicon MLX chats into the
`truncate_oldest` policy; other API clients can opt in with the same field.

## How the limit is measured

The limit comes from the currently loaded model's `context_length`. Before a
generation starts, Studio reserves the requested output allowance and renders the
conversation with the model's actual chat template, tool catalogue, and reasoning
controls.

GGUF counts this rendered prompt in llama.cpp. MLX sends a request-scoped count to
the model-owning worker process, where the loaded tokenizer measures the exact prompt
that generation will use, including native-template fallback. The route performs this
work in the generation worker thread, so tokenization and archive lookup do not block
the async server loop or the chat UI.

Tool loops repeat the preflight before every model turn because tool results can grow
the prompt after the first generation.

The context meter shows the live window occupancy. A single llama.cpp pass can report
more processed prompt-plus-completion tokens than the configured window because its KV
window shifts during a very long generation; the meter caps active occupancy at the
window and keeps the larger processed-work total in its tooltip. That reporting value is
never used as a compaction latch. Every later model pass and every new user turn performs
the fit again.

## Compaction behavior

Studio reuses one policy for GGUF and MLX:

- The newest user turn and protected standing instructions stay in view.
- Persisted, tool-capable chats can start a checkpoint epoch. Evicted turns are
  archived, a compact carried-forward boundary remains in the prompt, and the model
  can retrieve older details with `search_conversation`.
- Temporary, incognito, threadless, tool-disabled, or client-owned tool requests use
  the rolling window instead. This avoids claiming that discarded history is
  searchable when no safe retrieval path exists.
- A failed preflight never breaks generation. Studio leaves the request unchanged and
  lets the inference backend report its normal context error.

The backend emits `context_truncated` metadata in streaming and non-streaming
responses. Studio uses it for the visible compaction notice and for the persisted
checkpoint boundary on the next turn.

## Media prompts

MLX image and audio prompts are currently left unchanged. Their effective token cost
depends on processor-side media expansion rather than tokenizer-only counting, so
silently compacting them would not provide an exact fit guarantee.

## Tests

The backend tests cover exact MLX prompt counting, request-scoped worker IPC,
checkpoint/rolling fitting, media fallback, and per-iteration tool-loop fitting. The
frontend contract test verifies that both GGUF and MLX chats opt into automatic
compaction.
