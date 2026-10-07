# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Silence uvicorn's h11 shutdown traceback on Windows (issue #8404).

Every clean shutdown on Windows printed an unhandled asyncio traceback ending in ``h11._util.LocalProtocolError: can't handle event type Response when role=SERVER and state=CLOSED``. Nothing breaks, but a full traceback on a normal exit reads as a crash and gets reported as one.

The sequence, all of it in third-party code. First, ``uvicorn.Server.shutdown()`` walks the live connections and calls ``H11Protocol.shutdown()`` (``uvicorn/protocols/http/h11_impl.py``), which sends ``h11.ConnectionClosed()``, moving the h11 server state to CLOSED, then calls ``transport.close()``. Second, the browser is still polling ``/api/inference/status`` and ``/api/inference/monitor`` on that keep-alive connection, so a request can already be sitting in the socket, and on Windows the proactor transport still hands it to the protocol after ``close()``: ``_ProactorReadPipeTransport._loop_reading()`` assigns ``length`` from ``fut.result()`` before returning early on ``self._closing``, and its ``finally:`` clause calls ``_data_received()`` anyway (CPython ``Lib/asyncio/proactor_events.py``). ``close()`` does cancel ``_read_fut``, but a read whose overlapped ``WSARecv()`` already completed had ``_ov`` cleared by ``_OverlappedFuture.set_result()``, so ``cancel()`` is a no-op and the done callback queued before ``close()`` still runs. CPython 3.9 also set ``data = None`` in that branch and so never delivered, and 3.10 dropped that line when the reader moved to ``recv_into()``, which is why the report needs 3.10 or newer; the selector transport used on Linux and macOS removes the reader inside ``close()``, and uvloop calls ``_stop_reading()`` inside its own ``close()`` (and does not build on Windows at all), so this is Windows-only in practice. Third, h11 sees bytes after it expected EOF, so ``next_event()`` raises ``RemoteProtocolError``, uvicorn logs "Invalid HTTP request received." and calls ``send_400_response()``, whose very first ``self.conn.send(...)`` raises ``LocalProtocolError`` because the server state is CLOSED; nothing catches it, so it escapes ``data_received()`` back into the proactor read callback and asyncio's default handler prints the traceback.

The fix stops step 2 from reaching h11 at all rather than swallowing the exception at the end: once ``ConnectionClosed`` is sent and the transport closed, no further byte can legally be written on that connection, so the inbound data belongs to a request that will never be answered and dropping it is exactly what the selector transport already does. Suppressing the ``LocalProtocolError`` instead would hide genuine protocol errors on live connections and leave the equally misleading "Invalid HTTP request received." warning behind.
"""

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
        return None

    # In our_state CLOSED/ERROR/MUST_CLOSE no response can be written, so more input only yields a
    # spurious 400; our_state not their_state, since a live malformed request must still get its 400.
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
