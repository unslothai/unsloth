# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""This node's keys and certificates, the master's cluster root, and the registry of paired peers."""

from __future__ import annotations

import contextlib
import datetime
import json
import os
import secrets
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from . import tls
from .model import LocalIdentity, Peer
from .paths import private_dir, state_dir, write_private

ROOT_DAYS = 3650
MASTER_CERT_DAYS = 3650
WORKER_CERT_DAYS = 730


def _x509():
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    return x509, hashes, serialization, ec, ExtendedKeyUsageOID, NameOID


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _key_pem(key) -> bytes:
    _, _, serialization, _, _, _ = _x509()
    return key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    )


def _cert_pem(cert) -> bytes:
    _, _, serialization, _, _, _ = _x509()
    return cert.public_bytes(serialization.Encoding.PEM)


def _der_of_pem(pem: bytes) -> bytes:
    x509, _, serialization, _, _, _ = _x509()
    return x509.load_pem_x509_certificate(pem).public_bytes(serialization.Encoding.DER)


def pem_fingerprint(pem: bytes) -> str:
    return tls.fingerprint(_der_of_pem(pem))


def _leaf(public_key, ca_key, ca_cert, node_id: str, role: str, days: int):
    x509, hashes, _, _, eku, name_oid = _x509()
    now = _now()
    return (
        x509.CertificateBuilder()
        .subject_name(
            x509.Name([x509.NameAttribute(name_oid.COMMON_NAME, f"unsloth {role} {node_id}")])
        )
        .issuer_name(ca_cert.subject)
        .public_key(public_key)
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = days))
        .add_extension(x509.BasicConstraints(ca = False, path_length = None), critical = True)
        .add_extension(
            x509.KeyUsage(
                digital_signature = True,
                content_commitment = False,
                key_encipherment = False,
                data_encipherment = False,
                key_agreement = True,
                key_cert_sign = False,
                crl_sign = False,
                encipher_only = False,
                decipher_only = False,
            ),
            critical = True,
        )
        .add_extension(x509.ExtendedKeyUsage([eku.SERVER_AUTH, eku.CLIENT_AUTH]), critical = False)
        .add_extension(
            x509.SubjectAlternativeName(
                [
                    x509.UniformResourceIdentifier(tls.NODE_URI_PREFIX + node_id),
                    x509.UniformResourceIdentifier(tls.ROLE_URI_PREFIX + role),
                ]
            ),
            critical = False,
        )
        .add_extension(
            x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical = False
        )
        .sign(ca_key, hashes.SHA256())
    )


def _identity_dir() -> Path:
    return state_dir() / "identity"


def _ca_dir() -> Path:
    return state_dir() / "ca"


def _read_identity_json() -> Optional[dict]:
    path = _identity_dir() / "identity.json"
    try:
        data = json.loads(path.read_text(encoding = "utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _local_identity(meta: dict) -> LocalIdentity:
    d = _identity_dir()
    cert = d / "cert.pem"
    return LocalIdentity(
        node_id = str(meta["node_id"]),
        role = str(meta["role"]),
        cluster_id = str(meta["cluster_id"]),
        certfile = str(cert),
        keyfile = str(d / "key.pem"),
        rootfile = str(d / "root.pem"),
        fingerprint = pem_fingerprint(cert.read_bytes()),
    )


def _load_ca():
    x509, _, serialization, _, _, _ = _x509()
    d = _ca_dir()
    key = serialization.load_pem_private_key((d / "root_key.pem").read_bytes(), password = None)
    cert = x509.load_pem_x509_certificate((d / "root.pem").read_bytes())
    return key, cert


def _create_ca(cluster_id: str):
    x509, hashes, _, ec, _, name_oid = _x509()
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(name_oid.COMMON_NAME, f"unsloth cluster {cluster_id}")])
    now = _now()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes = 5))
        .not_valid_after(now + datetime.timedelta(days = ROOT_DAYS))
        .add_extension(x509.BasicConstraints(ca = True, path_length = 0), critical = True)
        .add_extension(
            x509.KeyUsage(
                digital_signature = False,
                content_commitment = False,
                key_encipherment = False,
                data_encipherment = False,
                key_agreement = False,
                key_cert_sign = True,
                crl_sign = True,
                encipher_only = False,
                decipher_only = False,
            ),
            critical = True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), critical = False)
        .sign(key, hashes.SHA256())
    )
    write_private(_ca_dir() / "root_key.pem", _key_pem(key))
    write_private(_ca_dir() / "root.pem", _cert_pem(cert))
    return key, cert


def _existing_master() -> Optional[LocalIdentity]:
    meta = _read_identity_json()
    if meta is not None and meta.get("role") == "master" and (_ca_dir() / "root_key.pem").is_file():
        return _local_identity(meta)
    return None


def master_identity(create: bool = True) -> Optional[LocalIdentity]:
    """The master's transport identity, creating the cluster root and certificate on first use."""
    found = _existing_master()
    if found is not None or not create:
        return found
    # Studio and `unsloth cluster pair` may both get here first: two roots would orphan every
    # worker paired against the loser.
    with _locked(state_dir() / "identity.lock"):
        return _existing_master() or _create_master_identity()


def _create_master_identity() -> LocalIdentity:
    meta = _read_identity_json()
    if meta is not None and meta.get("role") == "worker":
        raise RuntimeError("this machine is paired as a worker; run `unsloth cluster leave` first")
    _, _, _, ec, _, name_oid = _x509()
    if (_ca_dir() / "root_key.pem").is_file():
        ca_key, ca_cert = _load_ca()
        cn = ca_cert.subject.get_attributes_for_oid(name_oid.COMMON_NAME)[0].value
        cluster_id = str(cn).rsplit(" ", 1)[-1]
    else:
        cluster_id = secrets.token_hex(16)
        ca_key, ca_cert = _create_ca(cluster_id)
    node_id = secrets.token_hex(8)
    key = ec.generate_private_key(ec.SECP256R1())
    cert = _leaf(key.public_key(), ca_key, ca_cert, node_id, "master", MASTER_CERT_DAYS)
    d = private_dir(_identity_dir())
    write_private(d / "key.pem", _key_pem(key))
    write_private(d / "cert.pem", _cert_pem(cert))
    write_private(d / "root.pem", _cert_pem(ca_cert))
    meta = {"node_id": node_id, "role": "master", "cluster_id": cluster_id}
    write_private(d / "identity.json", json.dumps(meta).encode("utf-8"))
    return _local_identity(meta)


def worker_identity() -> Optional[LocalIdentity]:
    meta = _read_identity_json()
    if meta is None or meta.get("role") != "worker":
        return None
    return _local_identity(meta)


def save_worker_identity(
    *,
    node_id: str,
    cluster_id: str,
    key_pem: bytes,
    cert_pem: bytes,
    root_pem: bytes,
    agent_port: Optional[int] = None,
) -> LocalIdentity:
    d = private_dir(_identity_dir())
    write_private(d / "key.pem", key_pem)
    write_private(d / "cert.pem", cert_pem)
    write_private(d / "root.pem", root_pem)
    meta = {"node_id": node_id, "role": "worker", "cluster_id": cluster_id}
    if agent_port:
        meta["agent_port"] = int(agent_port)
    write_private(d / "identity.json", json.dumps(meta).encode("utf-8"))
    return _local_identity(meta)


def worker_agent_port() -> Optional[int]:
    """The port this worker told the master it serves on when it paired."""
    meta = _read_identity_json() or {}
    port = meta.get("agent_port")
    return (
        port if isinstance(port, int) and not isinstance(port, bool) and 0 < port < 65536 else None
    )


def new_worker_key_and_csr() -> Tuple[bytes, bytes]:
    x509, hashes, _, ec, _, name_oid = _x509()
    key = ec.generate_private_key(ec.SECP256R1())
    csr = (
        x509.CertificateSigningRequestBuilder()
        .subject_name(x509.Name([x509.NameAttribute(name_oid.COMMON_NAME, "unsloth worker")]))
        .sign(key, hashes.SHA256())
    )
    _, _, serialization, _, _, _ = _x509()
    return _key_pem(key), csr.public_bytes(serialization.Encoding.PEM)


def csr_der(csr_pem: bytes) -> bytes:
    """DER of a CSR whose signature checks out; ValueError otherwise."""
    x509, _, serialization, ec, _, _ = _x509()
    try:
        csr = x509.load_pem_x509_csr(csr_pem)
    except Exception as exc:
        raise ValueError(f"not a CSR: {exc}") from None
    if not csr.is_signature_valid:
        raise ValueError("CSR signature does not verify")
    if not isinstance(csr.public_key(), ec.EllipticCurvePublicKey):
        raise ValueError("CSR key must be an EC key")
    return csr.public_bytes(serialization.Encoding.DER)


def cert_matches_key(cert_pem: bytes, key_pem: bytes) -> bool:
    x509, _, serialization, _, _, _ = _x509()
    cert = x509.load_pem_x509_certificate(cert_pem)
    key = serialization.load_pem_private_key(key_pem, password = None)
    fmt = (serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return cert.public_key().public_bytes(*fmt) == key.public_key().public_bytes(*fmt)


def issue_worker_cert(csr_pem: bytes, node_id: str) -> bytes:
    """Master side: a worker certificate for the CSR's key, naming ``node_id`` whatever the CSR asked for."""
    x509, _, _, _, _, _ = _x509()
    csr_der(csr_pem)
    csr = x509.load_pem_x509_csr(csr_pem)
    ca_key, ca_cert = _load_ca()
    cert = _leaf(csr.public_key(), ca_key, ca_cert, node_id, "worker", WORKER_CERT_DAYS)
    return _cert_pem(cert)


def root_pem() -> bytes:
    return (_identity_dir() / "root.pem").read_bytes()


def own_cert_pem() -> bytes:
    return (_identity_dir() / "cert.pem").read_bytes()


def forget_identity() -> None:
    """Drop this node's identity and registry (``unsloth cluster leave`` on a worker)."""
    for path in (
        _identity_dir() / "identity.json",
        _identity_dir() / "key.pem",
        _identity_dir() / "cert.pem",
        _identity_dir() / "root.pem",
        state_dir() / "peers.json",
    ):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def identity_lock() -> Iterator[None]:
    """Held by a master reset and by a pairing commit, so a reset cannot land between a pairing
    issuing a certificate and recording the worker (or after it, unseen by the reset's check)."""
    with _locked(state_dir() / "identity.lock"):
        yield


def forget_master() -> None:
    """Drop a master's identity, its cluster root and registry. Only for a master with no live workers:
    every worker paired to this root would be orphaned."""
    forget_identity()
    for path in (_ca_dir() / "root_key.pem", _ca_dir() / "root.pem"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def _locked(path: Path) -> Iterator[None]:
    private_dir(path.parent)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
        except OSError as exc:
            # Proceeding unlocked would let a CLI and Studio silently overwrite each other's writes.
            raise RuntimeError(f"could not lock {path}: {exc}") from None
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(fd)


class Registry:
    """``peers.json``: every peer this node paired with, removed ones kept as tombstones."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or state_dir() / "peers.json"
        self.peers: dict = {}

    @classmethod
    def open(cls) -> "Registry":
        reg = cls()
        reg.load()
        return reg

    def load(self) -> "Registry":
        try:
            data = json.loads(self.path.read_text(encoding = "utf-8"))
        except FileNotFoundError:
            data = {"peers": []}
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{self.path} could not be read: {exc}") from None
        peers, extra = {}, {}
        known = {f.name for f in fields(Peer)}
        for entry in data.get("peers", []) if isinstance(data, dict) else []:
            if not isinstance(entry, dict):
                continue
            # Fields from a newer version are carried through save, so an older CLI never drops
            # a peer (or a tombstone) it does not fully understand.
            try:
                peer = Peer(**{k: v for k, v in entry.items() if k in known})
            except TypeError:
                continue
            peers[peer.node_id] = peer
            extra[peer.node_id] = {k: v for k, v in entry.items() if k not in known}
        self.peers, self._extra = peers, extra
        return self

    def save(self) -> None:
        extra = getattr(self, "_extra", {})
        rows = [{**extra.get(p.node_id, {}), **asdict(p)} for p in self.peers.values()]
        body = json.dumps({"peers": rows}, indent = 2)
        write_private(self.path, body.encode("utf-8"))

    @contextlib.contextmanager
    def transaction(self) -> Iterator["Registry"]:
        """Reload, mutate and save under the lock, so a CLI and Studio never overwrite each other."""
        with _locked(self.path.with_name(self.path.name + ".lock")):
            self.load()
            yield self
            self.save()

    def workers(self) -> List[Peer]:
        return [p for p in self.peers.values() if p.role == "worker" and not p.removed]

    def master(self) -> Optional[Peer]:
        live = [p for p in self.peers.values() if p.role == "master" and not p.removed]
        return live[0] if live else None

    def add(self, peer: Peer) -> None:
        if not peer.added_at:
            peer.added_at = time.time()
        self.peers[peer.node_id] = peer

    def remove(self, node_id: str) -> bool:
        peer = self.peers.get(node_id)
        if peer is None or peer.removed:
            return False
        peer.removed = True
        return True

    def find(self, name_or_id: str) -> Optional[Peer]:
        if name_or_id in self.peers:
            return self.peers[name_or_id]
        named = [p for p in self.peers.values() if p.name == name_or_id and not p.removed]
        return named[0] if len(named) == 1 else None

    def authorize(self, pid: Optional[tls.PeerIdentity], role: str) -> Optional[Peer]:
        if pid is None or pid.role != role:
            return None
        peer = self.peers.get(pid.node_id)
        if peer is None or peer.removed or peer.role != role:
            return None
        if not secrets.compare_digest(peer.fingerprint, pid.fingerprint):
            return None
        return peer

    def record_address(
        self,
        node_id: str,
        host: str,
        port: Optional[int] = None,
    ) -> None:
        peer = self.peers.get(node_id)
        if peer is None:
            return
        peer.addrs = [host] + [a for a in peer.addrs if a != host][:7]
        if port:
            peer.agent_port = int(port)
