# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Turn a mid-stream llama-server error chunk into something the user can act on.

llama-server can fail a request after the stream has already opened. It sends one
``{"error": {"message": ...}}`` chunk and closes. Two callers read that stream and both
handled it badly:

- ``core.research_runs`` raised ``RuntimeError("Local model stream failed")``, discarding
  whatever the server said. The user saw a generic failure with no cause and no remedy.
- ``core.inference.llama_cpp``'s chat loops only ever read ``finish_reason`` out of
  ``choices``. An error chunk carries no ``choices``, so nothing was recorded: the reply
  was emitted with whatever text had accumulated and no ``incomplete`` stamp, which reads
  to the user as the model stopping mid-sentence for no reason.

The error worth naming is KV-cache exhaustion. With ``--parallel N --kv-unified``,
llama.cpp allocates ONE cache of ``n_ctx`` tokens but reports ``n_ctx_slot = n_ctx`` to
every slot, so N concurrent generations may each be admitted believing they own the whole
window. When their combined length crosses it, the decode fails and llama.cpp kills every
task involved, not just the one that tipped it over. Two chats running at once is enough.
The server's own wording for that is "Context size has been exceeded", which sounds like
the prompt was too long and sends the user off to shorten a conversation that was never
the problem.
"""

from typing import Any, Optional

# Starvation: concurrent generations drew on the same unified cache Two different failures share the word "context" and
# need different advice. Starvation: the decode could not find KV space because concurrent generations drew on the same
# unified cache. Nothing about this request was too big, so telling the user to shorten it is wrong. Matched on a
# substring because the server appends punctuation and, on some paths, batch details.
_STARVATION_MARKERS = (
    "context size has been exceeded",
    "failed to find free space in the kv cache",
    "failed to find a memory slot",
)

# oversize: the server's precise text (both token counts) is kept verbatim, only the remedy is added
# Oversize: the request alone did not fit, and the server says so precisely, including both token counts. That text is
# kept verbatim; only the remedy is added.
_OVERSIZE_MARKERS = (
    "exceeds the available context size",
    "exceeds the context size",
)

# Worded around the chat client's own substring test. `chat-adapter.ts::isContextLimitError` matches "context window"
# (among others) and shows "Context limit reached: the conversation has filled the model's context window... or start a
# new chat" -- the advice this message exists to deny, since nothing about the conversation was too long and a new chat
# fails identically while the other generation is still running. The backend `code` cannot rescue it: `chat-api.ts`
# rethrows an in-band error chunk as `new Error(message)` and the text is all the client has. "Context Length" names the
# setting and is not one of its markers.
KV_STARVATION_MESSAGE = (
    "The model ran out of context space while generating. This happens when several "
    "chats or research runs generate at the same time, because they all draw on one "
    "shared pool of context. Try again with fewer running at once, or raise the "
    "Context Length in Model settings."
)

_OVERSIZE_HINT = "Raise the Context Length in Model settings, or shorten the request."

_GENERIC_MESSAGE = "The model stopped generating early."


class LlamaStreamError(RuntimeError):
    """A mid-stream llama-server failure, carrying a message already fit to show.

    Typed rather than a bare ``RuntimeError`` for two reasons, both in
    ``routes/inference.py``:

    - ``_friendly_error`` ends with ``return "An internal error occurred"`` for any
      exception it does not recognise. A bare RuntimeError would have that message
      swapped out on the chat path, so the cause would still never reach the user
      even though it now survives the stream loop.
    - ``_classify_llama_generation_error`` flags an overflow by substring, matching
      "context" beside "length"/"window". The starvation text above says "context"
      while explaining that the context is shared, and names the Context Length
      setting as the remedy, so it would be classified as ``context_length_exceeded``
      and set the client compacting a conversation that was never too long.
      ``kv_starvation`` lets that path opt out explicitly.
    """

    def __init__(
        self,
        friendly: str,
        *,
        server_message: Optional[str] = None,
        kv_starvation: bool = False,
        context_oversize: bool = False,
    ):
        # keep the server's own text so _friendly_error's token-count regex still rewrites an oversize refusal
        # str(exc) stays the server's own text where there is one, so the existing token-count regex in _friendly_error
        # still matches an oversize refusal and rewrites it into the established "Message too long" wording.
        super().__init__(server_message or friendly)
        self.friendly = friendly
        self.server_message = server_message
        self.kv_starvation = kv_starvation
        self.context_oversize = context_oversize


def stream_error_from_chunk(chunk: Any) -> Optional[LlamaStreamError]:
    """A raisable error for a streamed chunk, or None when the chunk is not one."""
    message = error_message_from_chunk(chunk)
    if message is None:
        return None
    starvation = is_kv_starvation(message)
    return LlamaStreamError(
        describe_stream_error(message),
        server_message = message or None,
        kv_starvation = starvation,
        context_oversize = is_context_oversize(message),
    )


def error_message_from_chunk(chunk: Any) -> Optional[str]:
    """The server's own error text from a streamed chunk, or None if it is not an error.

    Accepts the two shapes llama-server emits: ``{"error": {"message": ...}}`` and a bare
    ``{"error": "..."}``. A chunk carrying an ``error`` key with neither shape is still an
    error, so it returns the empty string rather than None; only a non-error chunk is None.
    """
    if not isinstance(chunk, dict) or "error" not in chunk:
        return None
    error = chunk["error"]
    if isinstance(error, str):
        return error.strip()
    if isinstance(error, dict):
        message = error.get("message")
        if isinstance(message, str):
            return message.strip()
    return ""


def is_kv_starvation(message: Optional[str]) -> bool:
    """Whether the error is the shared-KV starvation case rather than an oversize request."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _STARVATION_MARKERS)


def is_context_oversize(message: Optional[str]) -> bool:
    """Whether the server refused the request because it alone did not fit."""
    if not message:
        return False
    lowered = message.lower()
    return any(marker in lowered for marker in _OVERSIZE_MARKERS)


def describe_stream_error(message: Optional[str], *, prefix: str = "") -> str:
    """A user-facing sentence for a mid-stream failure.

    Starvation gets the explanation above, because the server's own wording ("Context size
    has been exceeded") reads as though the request was too long and sends the user off to
    shorten a conversation that was never the problem. An oversize refusal already names
    both token counts, so it is kept verbatim and only gains the remedy. Anything else is
    passed through unchanged: it is the only information there is, and replacing it with a
    fixed string is what made these undiagnosable. ``prefix`` names the caller ("Deep
    Research") where the surrounding UI does not already.
    """
    if is_kv_starvation(message):
        body = KV_STARVATION_MESSAGE
    elif is_context_oversize(message):
        body = f"{message.rstrip('. ')}. {_OVERSIZE_HINT}"
    elif message:
        body = message
    else:
        body = _GENERIC_MESSAGE
    return f"{prefix}: {body}" if prefix else body
