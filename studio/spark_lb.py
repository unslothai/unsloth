# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Round-robin front end for several llama.cpp engines spread over two DGX Sparks.

The reason this exists is a measured result. A single model layer-split across two Sparks
runs at **0.92x one Spark** -- the two nodes take turns, so the split buys capacity and
costs throughput. But running *two* independent engines, each split across both nodes, and
alternating requests between them reaches **1.35x one Spark** (124.4 vs 92.4 tok/s at 32
concurrent requests, against 70.3 for two engines confined to one node).

That is the same trick vLLM and SGLang use. Their pipeline schedulers need at least
`pp_size` *data-independent* batches in flight (vLLM's `EngineCore.batch_queue`, SGLang's
`running_mbs`), because a single autoregressive stream cannot be pipelined at all -- token
t+1 depends on token t. Two engines supply that independence without any change to
llama.cpp, whose RPC path cannot host two contexts in one process (a process-global socket
singleton, an unlocked send path, and a serial accept loop).

Balancing is per *connection*, not per request, so a streaming response is never split
across engines and no HTTP parsing is needed -- bytes are forwarded verbatim in both
directions.
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
            # One engine down should not take the front end with it; drop this
            # connection and let the client retry onto the next backend.
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
