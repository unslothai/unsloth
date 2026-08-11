# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Silence uvicorn's h11 shutdown traceback on Windows (issue #8404).

Every clean shutdown on Windows printed an unhandled asyncio traceback ending in
``h11._util.LocalProtocolError: can't handle event type Response when
role=SERVER and state=CLOSED``. Nothing breaks, but a full traceback on a normal
exit reads as a crash and gets reported as one.

The sequence, all of it in third-party code:

1. ``uvicorn.Server.shutdown()`` walks the live connections and calls
   ``H11Protocol.shutdown()`` (``uvicorn/protocols/http/h11_impl.py``), which
   sends ``h11.ConnectionClosed()`` -- moving the h11 server state to CLOSED --
   and then calls ``transport.close()``.
2. The browser is still polling ``/api/inference/status`` and
   ``/api/inference/monitor`` on that keep-alive connection, so a request can
   already be sitting in the socket. On Windows the proactor transport still
   hands it to the protocol after ``close()``:
   ``_ProactorReadPipeTransport._loop_reading()`` returns early on
   ``self._closing``, but its ``finally:`` clause calls ``_data_received()``
   anyway (CPython ``Lib/asyncio/proactor_events.py``). The overlapped
   ``WSARecv()`` is deliberately not cancelled -- see bpo-33694, cancelling it
   loses data. The selector transport used on Linux and macOS removes the
   reader inside ``close()``, so its read callback cannot fire again, which is
   why this is Windows-only in practice.
3. h11 sees bytes after it expected EOF, so ``next_event()`` raises
   ``RemoteProtocolError``. uvicorn logs "Invalid HTTP request received." and
   calls ``send_400_response()``, whose very first ``self.conn.send(...)``
   raises ``LocalProtocolError`` because the server state is CLOSED. Nothing
   catches it, so it escapes ``data_received()`` back into the proactor read
   callback and asyncio's default handler prints the traceback.

The fix is to stop step 2 from reaching h11 at all rather than to swallow the
exception at the end: once we have sent ``ConnectionClosed`` and closed the
transport, no further byte can legally be written on that connection, so the
inbound data belongs to a request that will never be answered and dropping it
is exactly what the selector transport already does. Suppressing the
``LocalProtocolError`` instead would also hide genuine protocol errors on live
connections, and it would leave the equally misleading "Invalid HTTP request
received." warning behind.
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
        # Any uvicorn/h11 layout we do not recognise: leave uvicorn untouched.
        return None

    # our_state CLOSED means we already sent h11.ConnectionClosed(); ERROR means
    # a previous send violated the state machine. In both cases uvicorn can no
    # longer write a response on this connection, so feeding it more inbound
    # bytes can only produce the spurious 400 attempt described above.
    terminal_states = (h11.CLOSED, h11.ERROR)

    class _ShutdownQuietH11Protocol(H11Protocol):  # type: ignore[misc, valid-type]
        """H11Protocol that drops reads delivered after the connection closed."""

        def data_received(self, data: bytes) -> None:
            conn = getattr(self, "conn", None)
            if conn is not None and conn.our_state in terminal_states:
                return
            super().data_received(data)

    return _ShutdownQuietH11Protocol


def uvicorn_http_protocol() -> Union[str, type]:
    """Return the value for ``uvicorn.Config(http = ...)``.

    Yields the patched h11 protocol only when uvicorn would have picked plain
    h11 anyway, so the httptools fast path is never silently disabled. httptools
    does not need this: its own 400 path writes straight to the transport with
    no state machine to violate.
    """
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
