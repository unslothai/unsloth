# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Stops the Windows proactor feeding bytes to h11 after uvicorn sent ConnectionClosed, which
printed a spurious LocalProtocolError traceback on clean shutdown; dropping that data beats
suppressing the error, which would hide real protocol errors."""

from __future__ import annotations

from functools import lru_cache
from typing import Union


@lru_cache(maxsize = 1)
def _shutdown_quiet_h11_protocol() -> Union[type, None]:
    """Build the ``H11Protocol`` subclass that ignores post-close reads."""
    try:
        import h11
        from uvicorn.protocols.http.h11_impl import H11Protocol
    except Exception:
        # Any uvicorn/h11 layout we do not recognise: leave uvicorn untouched.
        return None

    # In our_state CLOSED / ERROR / MUST_CLOSE uvicorn can no longer write a response, so feeding h11 more inbound bytes only produces the spurious 400 attempt above. MUST_CLOSE matters because send_400_response() closes the transport without sending ConnectionClosed, and a second proactor read in that window raises the same LocalProtocolError. Reads our_state and never their_state on purpose: a malformed request from a live client leaves our_state at IDLE or SEND_RESPONSE precisely so the server can still answer 400 (h11 docs, "error handling"), and h11 only reaches CLOSED or MUST_CLOSE from IDLE or DONE, so a response is either finished or was never started. Any non-HTTP bytes spanning more than one read get there, a browser sent to https://127.0.0.1:<port> included.
    terminal_states = (h11.MUST_CLOSE, h11.CLOSED, h11.ERROR)

    class _ShutdownQuietH11Protocol(H11Protocol):  # type: ignore[misc, valid-type]
        """H11Protocol that drops reads delivered after the connection closed."""

        def data_received(self, data: bytes) -> None:
            conn = getattr(self, "conn", None)
            if conn is not None and conn.our_state in terminal_states:
                return
            super().data_received(data)

    return _ShutdownQuietH11Protocol


def uvicorn_http_protocol() -> Union[str, type]:
    """The value for ``uvicorn.Config(http = ...)``: the patched h11 protocol only when uvicorn would have picked plain h11 anyway, so the httptools fast path is never silently disabled. httptools does not need this, since its own 400 path writes straight to the transport with no state machine to violate."""
    try:
        from uvicorn.protocols.http.auto import AutoHTTPProtocol
        from uvicorn.protocols.http.h11_impl import H11Protocol
    except Exception:
        return "auto"

    if AutoHTTPProtocol is not H11Protocol:
        return "auto"

    protocol_class = _shutdown_quiet_h11_protocol()
    if protocol_class is None:
        return "auto"
    return protocol_class
