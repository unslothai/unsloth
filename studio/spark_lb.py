# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Round-robin front end for several llama.cpp engines spread over two DGX Sparks."""

from __future__ import annotations

import argparse
import asyncio
import itertools
import time
from typing import List, Tuple

# Both engines are on this box or one cable away, so a connect that has not completed in two seconds
# is not slow, it is gone.
CONNECT_TIMEOUT = 2.0
# A backend that failed to connect is skipped as a first choice for this long, so a lost peer costs
# one CONNECT_TIMEOUT per window instead of one on every other request.
DOWN_FOR = 10.0


async def _pump(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    half_close: bool = False,
) -> None:
    forwarded_eof = False
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                # A client that half-closes after its request still reads the response; closing the
                # upstream socket here cut that response off. Forward the EOF and let the other
                # direction finish; the handler closes both sockets.
                if half_close and writer.can_write_eof():
                    writer.write_eof()
                    forwarded_eof = True
                break
            writer.write(chunk)
            await writer.drain()
    except (OSError, asyncio.IncompleteReadError):
        pass
    finally:
        if not forwarded_eof:
            try:
                writer.close()
            except Exception:
                pass


def _handler(backends: List[Tuple[str, int]], rr):
    down_until = {}

    async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        # Round robin picks where to START, not the only backend to try.
        start = next(rr)
        up_r = up_w = None
        now = time.monotonic()
        order = [(start + offset) % len(backends) for offset in range(len(backends))]
        order.sort(key = lambda i: down_until.get(i, 0.0) > now)  # stable: live ones first
        for i in order:
            host, port = backends[i]
            try:
                # Bounded.
                up_r, up_w = await asyncio.wait_for(
                    asyncio.open_connection(host, port), timeout = CONNECT_TIMEOUT
                )
                down_until.pop(i, None)
                break
            except (OSError, asyncio.TimeoutError):
                down_until[i] = time.monotonic() + DOWN_FOR
                continue
        if up_w is None:
            client_w.close()
            return
        try:
            await asyncio.gather(_pump(client_r, up_w, half_close = True), _pump(up_r, client_w))
        finally:
            for writer in (up_w, client_w):
                try:
                    writer.close()
                except Exception:
                    pass

    return handle


async def serve(listen_host: str, listen_port: int, backends: List[Tuple[str, int]]) -> None:
    rr = itertools.count()
    server = await asyncio.start_server(_handler(backends, rr), listen_host, listen_port)
    where = ", ".join(f"{h}:{p}" for h, p in backends)
    print(f"round-robin on {listen_host}:{listen_port} -> {where}", flush = True)
    async with server:
        await server.serve_forever()


def parse_backend(text: str) -> Tuple[str, int]:
    host, _, port = text.rpartition(":")
    return (host or "127.0.0.1", int(port))


def main(argv = None) -> int:
    p = argparse.ArgumentParser(prog = "spark_lb", description = __doc__)
    p.add_argument("--port", type = int, default = 8080, help = "port to listen on")
    p.add_argument("--host", default = "0.0.0.0")
    p.add_argument(
        "backends", nargs = "+", help = "engine endpoints, e.g. 127.0.0.1:8096 127.0.0.1:8097"
    )
    args = p.parse_args(argv)
    try:
        asyncio.run(serve(args.host, args.port, [parse_backend(b) for b in args.backends]))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
