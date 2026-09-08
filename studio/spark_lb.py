# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Round-robin front end for several llama.cpp engines spread over two DGX Sparks.

One model layer-split across two Sparks loses throughput because the nodes take turns; two
independent engines alternating requests beat a single Spark. That is vLLM's and SGLang's
trick: a pipeline needs `pp_size` DATA-INDEPENDENT batches in flight, because a single
autoregressive stream cannot be pipelined at all -- token t+1 depends on token t. llama.cpp's
RPC path cannot host two contexts in one process, so two engines it is. Balancing is per
CONNECTION, so a streaming response is never split and no HTTP parsing is needed.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
from typing import List, Tuple


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            writer.write(chunk)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


def _handler(backends: List[Tuple[str, int]], rr):
    async def handle(client_r: asyncio.StreamReader, client_w: asyncio.StreamWriter) -> None:
        host, port = backends[next(rr) % len(backends)]
        try:
            up_r, up_w = await asyncio.open_connection(host, port)
        except OSError:
            # One engine down must not take the front end with it; the client retries next.
            client_w.close()
            return
        await asyncio.gather(_pump(client_r, up_w), _pump(up_r, client_w))

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
