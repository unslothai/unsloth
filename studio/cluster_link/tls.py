# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""TLS contexts for cluster links and the identity a peer certificate carries."""

from __future__ import annotations

import hashlib
import ssl
import sys
from dataclasses import dataclass
from typing import Optional

NODE_URI_PREFIX = "unsloth-node:"
ROLE_URI_PREFIX = "unsloth-role:"
ROLES = ("master", "worker")


def plain_ssl_context_class() -> type:
    for cls in getattr(ssl.SSLContext, "__mro__", ()):
        if cls.__module__ == "ssl" and cls.__name__ == "SSLContext":
            return cls
    for name, module in list(sys.modules.items()):
        if name.endswith("truststore._ssl_constants"):
            original = getattr(module, "_original_SSLContext", None)
            if isinstance(original, type):
                return original
    return ssl.SSLContext


def _set(ctx: ssl.SSLContext, name: str, value) -> None:
    import _ssl
    getattr(_ssl._SSLContext, name).__set__(ctx, value)


def _base(purpose_server: bool) -> ssl.SSLContext:
    import _ssl

    cls = plain_ssl_context_class()
    ctx = cls(ssl.PROTOCOL_TLS_SERVER if purpose_server else ssl.PROTOCOL_TLS_CLIENT)
    _set(ctx, "minimum_version", ssl.TLSVersion.TLSv1_3)
    _set(ctx, "check_hostname", False)
    if hasattr(ctx, "keylog_filename"):
        _set(ctx, "keylog_filename", None)
    _set(ctx, "options", _ssl._SSLContext.options.__get__(ctx) | ssl.OP_NO_TICKET)
    if purpose_server:
        _set(ctx, "num_tickets", 0)
    return ctx


def mutual_context(*, server: bool, certfile: str, keyfile: str, rootfile: str) -> ssl.SSLContext:
    """Both sides present a certificate and both must chain to the cluster root, and nothing else."""
    ctx = _base(server)
    _set(ctx, "verify_mode", ssl.CERT_REQUIRED)
    ctx.load_cert_chain(certfile, keyfile)
    ctx.load_verify_locations(cafile = rootfile)
    return ctx


def bootstrap_server_context(*, certfile: str, keyfile: str) -> ssl.SSLContext:
    """The pairing listener: the joining worker has no certificate yet, and proves the token secret instead."""
    ctx = _base(True)
    _set(ctx, "verify_mode", ssl.CERT_NONE)
    ctx.load_cert_chain(certfile, keyfile)
    return ctx


def bootstrap_client_context() -> ssl.SSLContext:
    """The joining worker."""
    ctx = _base(False)
    _set(ctx, "verify_mode", ssl.CERT_NONE)
    return ctx


def fingerprint(der: bytes) -> str:
    return hashlib.sha256(der).hexdigest()


def _peer_der(ssl_object: Optional[ssl.SSLObject]) -> Optional[bytes]:
    return ssl_object.getpeercert(binary_form = True) if ssl_object is not None else None


def peer_fingerprint(ssl_object: Optional[ssl.SSLObject]) -> Optional[str]:
    der = _peer_der(ssl_object)
    return fingerprint(der) if der else None


@dataclass(frozen = True)
class PeerIdentity:
    fingerprint: str
    node_id: str
    role: str


def identity_from_der(der: bytes) -> Optional[PeerIdentity]:
    """The node id and role a cluster certificate names in its SAN URIs, or None if it names neither."""
    from cryptography import x509

    try:
        cert = x509.load_der_x509_certificate(der)
        uris = cert.extensions.get_extension_for_class(
            x509.SubjectAlternativeName
        ).value.get_values_for_type(x509.UniformResourceIdentifier)
    except Exception:
        return None
    node = [u[len(NODE_URI_PREFIX) :] for u in uris if u.startswith(NODE_URI_PREFIX)]
    role = [u[len(ROLE_URI_PREFIX) :] for u in uris if u.startswith(ROLE_URI_PREFIX)]
    if len(node) != 1 or len(role) != 1 or role[0] not in ROLES or not node[0]:
        return None
    return PeerIdentity(fingerprint = fingerprint(der), node_id = node[0], role = role[0])


def peer_identity(ssl_object: Optional[ssl.SSLObject]) -> Optional[PeerIdentity]:
    der = _peer_der(ssl_object)
    return identity_from_der(der) if der else None
