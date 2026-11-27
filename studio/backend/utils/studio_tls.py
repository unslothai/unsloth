# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""HTTPS from user-supplied --ssl-certfile/--ssl-keyfile; a TLS launch also serves plain HTTP on an
ephemeral 127.0.0.1 port (loopback_http_port) for Studio's own /v1 self-calls from children that
cannot trust the cert."""

from __future__ import annotations

import os
import ssl
from dataclasses import dataclass
from typing import Optional

import uvicorn


@dataclass(frozen = True)
class StudioTLS:
    certfile: str
    keyfile: str


def resolve_studio_tls(certfile: Optional[str], keyfile: Optional[str]) -> Optional[StudioTLS]:
    """The validated pair, or None for a plain HTTP launch. Exits with a readable message before anything binds."""
    certfile = (certfile or "").strip()
    keyfile = (keyfile or "").strip()
    if not certfile and not keyfile:
        return None
    if not (certfile and keyfile):
        raise SystemExit("--ssl-certfile and --ssl-keyfile must be passed together.")
    tls = StudioTLS(
        certfile = os.path.abspath(os.path.expanduser(certfile)),
        keyfile = os.path.abspath(os.path.expanduser(keyfile)),
    )
    for flag, path in (("--ssl-certfile", tls.certfile), ("--ssl-keyfile", tls.keyfile)):
        if not os.path.isfile(path):
            raise SystemExit(f"{flag}: file not found: {path}")
    try:
        server_ssl_context(tls)
    except (OSError, ssl.SSLError) as exc:
        raise SystemExit(
            f"Could not load the TLS certificate and key ({tls.certfile}, {tls.keyfile}): {exc}"
        ) from None
    return tls


def server_ssl_context(tls: StudioTLS) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # Native TLS (utils/native_tls.py) injects truststore, whose handshake hook verifies the PEER chain: a browser sends
    # none, so macOS / Windows fail every handshake. Keep truststore's class, uvloop isinstance-checks it.
    inner = getattr(ctx, "_ctx", None)
    if inner is not None and hasattr(inner, "sslobject_class"):
        inner.sslobject_class = ssl.SSLObject
    ctx.load_cert_chain(tls.certfile, tls.keyfile)
    return ctx


class StudioTLSConfig(uvicorn.Config):
    def __init__(self, app, *, tls: StudioTLS, **kwargs):
        super().__init__(app, ssl_certfile = tls.certfile, ssl_keyfile = tls.keyfile, **kwargs)
        self.studio_tls = tls

    def load(self) -> None:
        super().load()
        self.ssl = server_ssl_context(self.studio_tls)


def loopback_http_port(app_state) -> Optional[int]:
    port = getattr(app_state, "loopback_http_port", None)
    return port if isinstance(port, int) and port > 0 else None
