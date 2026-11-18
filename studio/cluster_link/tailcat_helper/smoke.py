# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Native smoke for a built unsloth-link: genkey, then a tunnel between two helpers over a loopback
DERP carries 1 MiB both ways. Usage: smoke.py PATH_TO_UNSLOTH_LINK (stdlib only)."""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studio.cluster_link import tailnet  # noqa: E402

PAYLOAD = os.urandom(1 << 20)


def _echo() -> int:
    srv = socket.create_server(("127.0.0.1", 0))

    def run() -> None:
        conn, _ = srv.accept()
        with conn:
            while data := conn.recv(65536):
                conn.sendall(data)

    threading.Thread(target = run, daemon = True).start()
    return srv.getsockname()[1]


def main(binary: str) -> int:
    os.environ[tailnet.ENV_BIN] = binary
    os.environ["TS_DEBUG_TAILCAT_LOCAL_DERP"] = "1"
    with tempfile.TemporaryDirectory() as tmp:
        server_key, client_key = Path(tmp) / "server.key", Path(tmp) / "client.key"
        tailnet.new_server_key(server_key, fixed_region = False)
        client_pub = tailnet.new_client_key(client_key)
        port = _echo()
        with tailnet.Tailnet() as worker, tailnet.Tailnet() as master:
            address = worker.serve([port], key_path = server_key, allow = [client_pub])
            local = master.forward(address, port, key_path = client_key)
            with socket.create_connection(("127.0.0.1", local), timeout = 60) as sock:
                sock.settimeout(60)
                threading.Thread(target = sock.sendall, args = (PAYLOAD,), daemon = True).start()
                got = bytearray()
                while len(got) < len(PAYLOAD):
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    got += chunk
            info = master.probe(address, timeout_s = 20)
    if bytes(got) != PAYLOAD:
        print(f"round trip failed: {len(got)} of {len(PAYLOAD)} bytes")
        return 1
    print(f"ok: 1 MiB round trip, path direct={info.direct} rtt_ms={info.rtt_ms}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
