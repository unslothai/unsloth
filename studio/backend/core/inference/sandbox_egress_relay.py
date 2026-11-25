# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Runs inside bubblewrap's private network: relays 127.0.0.1:PORT to Studio's proxy socket, then execs the command."""

import _signal
import _socket
import os
import sys


def _serve(raw, path):
    # Imported after the fork, so the command does not wait on them to start.
    import socket
    import threading

    def pipe(source, sink):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                sink.sendall(data)
        except OSError:
            pass
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass

    def relay(client):
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            upstream.connect(path)
            back = threading.Thread(target = pipe, args = (upstream, client), daemon = True)
            back.start()
            pipe(client, upstream)
            back.join()
        except OSError:
            pass
        finally:
            upstream.close()
            client.close()

    listener = socket.socket(fileno = raw.detach())
    while True:
        client, _ = listener.accept()
        threading.Thread(target = relay, args = (client,), daemon = True).start()


def main(argv):
    path, port, command = argv[1], int(argv[2]), argv[3:]
    listener = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
    listener.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen(64)
    # Listening before the command starts; the namespace's init kills this child when the command exits.
    if os.fork() == 0:
        null = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(null, fd)
        try:
            _serve(listener, path)
        finally:
            os._exit(0)
    listener.close()
    # Python ignores these at startup and exec would pass that on: `yes | head` must still end quietly.
    for name in ("SIGPIPE", "SIGXFSZ"):
        if hasattr(_signal, name):
            _signal.signal(getattr(_signal, name), _signal.SIG_DFL)
    os.execvp(command[0], command)


if __name__ == "__main__":
    try:
        main(sys.argv)
    except Exception as exc:
        sys.stderr.write(f"Unsloth sandbox: could not start the network relay: {exc}\n")
        sys.exit(126)
