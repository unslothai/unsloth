# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Durable, atomic replay protection for signed native path grants."""

import base64
import concurrent.futures
import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from storage import studio_db
from utils import native_path_leases as leases
from utils.paths import studio_db_path


_SECRET = b"durable-native-path-lease-test-secret"
_RUST_V2_VECTOR = Path(__file__).parent / "fixtures" / "native_path_lease_v2_rust.json"


@pytest.fixture(autouse = True)
def _lease_state(monkeypatch):
    monkeypatch.setenv(
        leases.LEASE_SECRET_ENV,
        base64.urlsafe_b64encode(_SECRET).decode("ascii").rstrip("="),
    )
    monkeypatch.setattr(leases, "_CACHED_LEASE_SECRET", None)
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    leases._USED_NONCES.clear()
    saved_labels = dict(leases._NATIVE_PATH_LABELS)
    saved_redactions = list(leases._NATIVE_PATH_REDACTIONS)
    yield
    leases._USED_NONCES.clear()
    leases._NATIVE_PATH_LABELS.clear()
    leases._NATIVE_PATH_LABELS.update(saved_labels)
    leases._NATIVE_PATH_REDACTIONS[:] = saved_redactions


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _sign_folder(
    path: Path,
    *,
    operation: str = "open-project",
    nonce: str | None = None,
) -> str:
    metadata = path.stat()
    now_ms = int(time.time() * 1000)
    payload = {
        "version": 1,
        "operation": operation,
        "canonical_path": str(path),
        "path_kind": "document-folder",
        "path_type": "directory",
        "source_kind": "dialog",
        "token_id_hash": hashlib.sha256(b"native-path-token").hexdigest(),
        "issued_at_ms": now_ms,
        "expires_at_ms": now_ms + 120_000,
        "nonce": nonce or os.urandom(16).hex(),
        "display_label": path.name,
        "size_bytes": None,
        "modified_ms": None,
        "device_id": format(metadata.st_dev, "x"),
        "file_id": format(metadata.st_ino, "x"),
    }
    payload_b64 = _b64(json.dumps(payload).encode("utf-8"))
    signature = hmac.new(_SECRET, payload_b64.encode("ascii"), hashlib.sha256).digest()
    return f"{payload_b64}.{_b64(signature)}"


def _verify_folder(lease: str, operation: str = "open-project"):
    return leases.verify_native_path_lease(
        lease,
        operation = operation,
        expected_kind = "document-folder",
        expected_path_type = "directory",
    )


def _lease_payload(lease: str) -> dict:
    payload = lease.split(".", 1)[0]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def _sign_folder_v2(path: Path) -> str:
    payload = _lease_payload(_sign_folder(path))
    payload["version"] = 2
    nonce = os.urandom(leases._LEASE_V2_NONCE_BYTES)
    plaintext = json.dumps(payload, separators = (",", ":")).encode("utf-8")
    ciphertext = leases._xor_lease_stream(_SECRET, nonce, plaintext)
    envelope = nonce + ciphertext
    signature = hmac.new(
        _SECRET,
        leases._LEASE_V2_AUTH_DOMAIN + envelope,
        hashlib.sha256,
    ).digest()
    return f"2.{_b64(envelope)}.{_b64(signature)}"


def test_python_decoder_accepts_rust_v2_known_answer():
    vector = json.loads(_RUST_V2_VECTOR.read_text(encoding = "utf-8"))
    secret = base64.urlsafe_b64decode(
        vector["secret_base64url"] + "=" * (-len(vector["secret_base64url"]) % 4)
    )

    payload, envelope_version = leases._decode_authenticated_payload(vector["lease"], secret)

    assert envelope_version == 2
    assert payload == vector["payload"]
    envelope = base64.urlsafe_b64decode(
        vector["lease"].split(".")[1] + "=" * (-len(vector["lease"].split(".")[1]) % 4)
    )
    expected_nonce = base64.urlsafe_b64decode(
        vector["envelope_nonce_base64url"] + "=" * (-len(vector["envelope_nonce_base64url"]) % 4)
    )
    assert envelope[: leases._LEASE_V2_NONCE_BYTES] == expected_nonce


def test_v2_lease_is_opaque_and_authenticates_before_use(tmp_path):
    folder = tmp_path / "private-repository"
    folder.mkdir()
    lease = _sign_folder_v2(folder)

    assert len(lease.split(".")) == 3
    assert str(folder) not in lease
    envelope = base64.urlsafe_b64decode(lease.split(".")[1] + "=" * (-len(lease.split(".")[1]) % 4))
    assert str(folder).encode("utf-8") not in envelope
    assert _verify_folder(lease).canonical_path == folder

    tampered = f"2.{_b64(envelope[:-1] + bytes([envelope[-1] ^ 1]))}.{lease.rsplit('.', 1)[1]}"
    with pytest.raises(leases.NativePathLeaseError, match = "signature"):
        _verify_folder(tampered)


@pytest.mark.parametrize("segment", [1, 2])
def test_v2_lease_rejects_noncanonical_base64url(tmp_path, segment):
    folder = tmp_path / "private-repository"
    folder.mkdir()
    parts = _sign_folder_v2(folder).split(".")
    parts[segment] = f"{parts[segment][:2]}!{parts[segment][2:]}"

    with pytest.raises(leases.NativePathLeaseError, match = "invalid format"):
        _verify_folder(".".join(parts))


def test_v2_lease_rejects_wrong_signature_length(tmp_path):
    folder = tmp_path / "private-repository"
    folder.mkdir()
    parts = _sign_folder_v2(folder).split(".")
    parts[2] = _b64(base64.urlsafe_b64decode(parts[2] + "=" * (-len(parts[2]) % 4)) + b"x")

    with pytest.raises(leases.NativePathLeaseError, match = "invalid format"):
        _verify_folder(".".join(parts))


def test_every_retained_native_path_label_remains_redactable(monkeypatch):
    monkeypatch.setattr(leases, "_MAX_NATIVE_PATH_LABELS", 101)
    monkeypatch.setattr(leases, "_MAX_NATIVE_PATH_REDACTIONS", 101)
    for index in range(101):
        leases._remember_native_path_for_redaction(
            f"/private/project-{index}",
            f"project-{index}",
        )

    assert leases.display_label_for_native_path("/private/project-0") == "project-0"
    assert leases.redact_native_paths("failed in /private/project-0/file.py") == (
        "failed in <native_path>/file.py"
    )


def test_replay_is_rejected_after_process_state_reset(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)
    _verify_folder(lease)

    # A new backend process has neither the module cache nor the in-memory set.
    leases._USED_NONCES.clear()
    leases._CACHED_LEASE_SECRET = None
    script = """
import sys
from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

try:
    verify_native_path_lease(
        sys.stdin.read(),
        operation="open-project",
        expected_kind="document-folder",
        expected_path_type="directory",
    )
except NativePathLeaseError as exc:
    raise SystemExit(0 if str(exc) == "Native path grant was already used." else 3)
raise SystemExit(2)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd = Path(__file__).resolve().parents[1],
        env = os.environ.copy(),
        input = lease,
        text = True,
        capture_output = True,
        timeout = 15,
        check = False,
    )

    assert result.returncode == 0, result.stderr


def test_concurrent_consumers_admit_exactly_one(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)
    barrier = threading.Barrier(2)

    def consume() -> str:
        barrier.wait(timeout = 5)
        try:
            _verify_folder(lease)
        except leases.NativePathLeaseError as exc:
            return str(exc)
        return "accepted"

    with concurrent.futures.ThreadPoolExecutor(max_workers = 2) as pool:
        futures = [pool.submit(consume) for _ in range(2)]
        outcomes = sorted(future.result() for future in futures)

    assert outcomes == ["Native path grant was already used.", "accepted"]


def test_concurrent_backend_processes_admit_exactly_one(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)
    connection = studio_db.get_connection()
    connection.close()
    release = tmp_path / "release-workers"
    ready_paths = [tmp_path / f"worker-{index}-ready" for index in range(2)]
    script = """
import os
import sys
import time
from pathlib import Path
from utils.native_path_leases import NativePathLeaseError, verify_native_path_lease

Path(sys.argv[1]).touch()
release = Path(sys.argv[2])
deadline = time.monotonic() + 10
while not release.exists():
    if time.monotonic() >= deadline:
        raise SystemExit(4)
    time.sleep(0.01)
try:
    verify_native_path_lease(
        os.environ.pop("NATIVE_LEASE_TEST_VALUE"),
        operation="open-project",
        expected_kind="document-folder",
        expected_path_type="directory",
    )
except NativePathLeaseError as exc:
    if str(exc) != "Native path grant was already used.":
        raise SystemExit(3)
    print("replayed")
else:
    print("accepted")
"""
    environment = os.environ.copy()
    environment["NATIVE_LEASE_TEST_VALUE"] = lease
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(ready), str(release)],
            cwd = Path(__file__).resolve().parents[1],
            env = environment,
            text = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        for ready in ready_paths
    ]
    try:
        deadline = time.monotonic() + 10
        while not all(path.exists() for path in ready_paths):
            if time.monotonic() >= deadline:
                pytest.fail("concurrent lease workers did not reach the consume barrier")
            time.sleep(0.01)
        release.touch()
        completed = [process.communicate(timeout = 15) for process in processes]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout = 5)

    assert [process.returncode for process in processes] == [0, 0], completed
    assert sorted(stdout.strip() for stdout, _stderr in completed) == ["accepted", "replayed"]


def test_nonce_store_cleans_expired_rows_and_refuses_live_overflow():
    first = hashlib.sha256(b"first-nonce").digest()
    second = hashlib.sha256(b"second-nonce").digest()

    assert studio_db.consume_native_path_lease_nonce(first, 2_000, now_ms = 1_000, max_entries = 1)
    with pytest.raises(studio_db.NativePathLeaseReplayCapacityError):
        studio_db.consume_native_path_lease_nonce(second, 3_000, now_ms = 1_500, max_entries = 1)
    assert studio_db.consume_native_path_lease_nonce(second, 3_000, now_ms = 2_000, max_entries = 1)

    connection = studio_db.get_connection()
    try:
        rows = connection.execute(
            "SELECT nonce_digest, expires_at_ms FROM native_path_lease_consumptions"
        ).fetchall()
    finally:
        connection.close()
    assert [(bytes(row["nonce_digest"]), row["expires_at_ms"]) for row in rows] == [(second, 3_000)]


def test_nonce_store_persists_no_raw_nonce_path_or_secret(tmp_path):
    folder = tmp_path / "private-repository"
    folder.mkdir()
    raw_nonce = "raw-native-lease-nonce"
    lease = _sign_folder(folder, nonce = raw_nonce)
    _verify_folder(lease)

    connection = studio_db.get_connection()
    try:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(native_path_lease_consumptions)")
        ]
        row = connection.execute(
            "SELECT nonce_digest, expires_at_ms FROM native_path_lease_consumptions"
        ).fetchone()
    finally:
        connection.close()
    assert columns == ["nonce_digest", "expires_at_ms"]
    assert bytes(row["nonce_digest"]) == hashlib.sha256(raw_nonce.encode()).digest()

    database = studio_db_path()
    persisted = b"".join(
        candidate.read_bytes()
        for candidate in (database, Path(f"{database}-wal"), Path(f"{database}-shm"))
        if candidate.exists()
    )
    assert raw_nonce.encode() not in persisted
    assert str(folder).encode() not in persisted
    assert _SECRET not in persisted


def test_wrong_operation_and_changed_identity_do_not_consume_nonce(tmp_path):
    folder = tmp_path / "repository"
    folder.mkdir()
    operation_lease = _sign_folder(folder)
    with pytest.raises(leases.NativePathLeaseError, match = "operation"):
        _verify_folder(operation_lease, operation = "link-documents")
    assert _verify_folder(operation_lease).canonical_path == folder

    identity_lease = _sign_folder(folder)
    identity_nonce = str(_lease_payload(identity_lease)["nonce"])
    moved = tmp_path / "moved-repository"
    folder.rename(moved)
    folder.mkdir()
    with pytest.raises(leases.NativePathLeaseError, match = "changed"):
        _verify_folder(identity_lease)

    connection = studio_db.get_connection()
    try:
        stored = connection.execute(
            "SELECT 1 FROM native_path_lease_consumptions WHERE nonce_digest = ?",
            (hashlib.sha256(identity_nonce.encode("utf-8")).digest(),),
        ).fetchone()
    finally:
        connection.close()
    assert stored is None


def test_nonce_store_failure_rejects_an_otherwise_valid_lease(tmp_path, monkeypatch):
    folder = tmp_path / "repository"
    folder.mkdir()
    lease = _sign_folder(folder)

    def unavailable(*_args, **_kwargs):
        raise OSError("injected storage failure")

    monkeypatch.setattr(studio_db, "consume_native_path_lease_nonce", unavailable)
    with pytest.raises(leases.NativePathLeaseError, match = "protection is unavailable"):
        _verify_folder(lease)
