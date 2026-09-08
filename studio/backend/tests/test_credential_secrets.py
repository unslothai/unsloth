# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations


import base64
import os
import subprocess
import sys
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from auth import storage as auth_storage
from core.inference import key_exchange
from core.inference.key_exchange import decrypt_api_key, init_key_pair
from storage import credential_secrets


@pytest.fixture(autouse = True)
def isolated_databases(tmp_path, monkeypatch):
    auth_db = tmp_path / "auth.db"
    studio_db = tmp_path / "studio.db"
    monkeypatch.setattr(auth_storage, "DB_PATH", auth_db)
    monkeypatch.setattr(auth_storage, "_credential_encryption_key_cache", None)
    monkeypatch.setattr(credential_secrets, "studio_db_path", lambda: studio_db)
    monkeypatch.setattr(
        credential_secrets, "ensure_dir", lambda path: path.mkdir(parents = True, exist_ok = True)
    )
    monkeypatch.setattr(
        credential_secrets,
        "get_or_create_credential_encryption_key",
        auth_storage.get_or_create_credential_encryption_key,
    )
    credential_secrets._schema_ready = False
    yield studio_db
    credential_secrets._schema_ready = False
    auth_storage._credential_encryption_key_cache = None


def test_round_trip_scopes_and_ciphertext(isolated_databases):
    secret = "hf_example-secret-value"
    credential_secrets.save_hf_token(secret)

    assert credential_secrets.get_hf_token() == secret
    assert secret.encode() not in isolated_databases.read_bytes()

    credential_secrets.save_provider_api_key("provider-1", "sk-one")
    credential_secrets.save_provider_api_key("provider-2", "sk-two")
    assert credential_secrets.get_provider_api_key("provider-1") == "sk-one"
    assert credential_secrets.get_provider_api_key("provider-2") == "sk-two"


def test_upsert_and_delete_are_idempotent():
    credential_secrets.save_provider_api_key("provider-1", "first")
    credential_secrets.save_provider_api_key("provider-1", "second")
    assert credential_secrets.get_provider_api_key("provider-1") == "second"

    assert credential_secrets.save_provider_api_key_if_absent("provider-1", "legacy") is False
    assert credential_secrets.get_provider_api_key("provider-1") == "second"
    assert credential_secrets.save_hf_token_if_absent("hf_legacy") is True
    assert credential_secrets.save_hf_token_if_absent("hf_delayed") is False
    assert credential_secrets.get_hf_token() == "hf_legacy"
    assert credential_secrets.delete_provider_api_key("provider-1") is True
    assert credential_secrets.delete_provider_api_key("provider-1") is False


def test_tampering_and_key_loss_fail_closed(isolated_databases):
    credential_secrets.save_hf_token("hf_private")
    conn = sqlite3.connect(isolated_databases)
    try:
        row = conn.execute("SELECT ciphertext FROM credential_secrets").fetchone()
        damaged = bytearray(row[0])
        damaged[-1] ^= 1
        conn.execute("UPDATE credential_secrets SET ciphertext = ?", (bytes(damaged),))
        conn.commit()
    finally:
        conn.close()
    assert credential_secrets.get_hf_token() is None

    credential_secrets.save_hf_token("hf_replaced-key")
    auth_storage._credential_encryption_key_cache = None
    conn = auth_storage.get_connection()
    try:
        conn.execute(
            "UPDATE app_secrets SET value = ? WHERE key = ?",
            ("00" * 32, auth_storage._CREDENTIAL_ENCRYPTION_KEY_DB_KEY),
        )
        conn.commit()
    finally:
        conn.close()
    assert credential_secrets.get_hf_token() is None


def test_repeated_schema_initialization_and_concurrent_upserts():
    credential_secrets.get_connection().close()
    credential_secrets._schema_ready = False
    credential_secrets.get_connection().close()

    with ThreadPoolExecutor(max_workers = 4) as pool:
        list(
            pool.map(
                lambda value: credential_secrets.save_provider_api_key("provider-1", value),
                ["one", "two", "three", "four"],
            )
        )
    assert credential_secrets.get_provider_api_key("provider-1") in {
        "one",
        "two",
        "three",
        "four",
    }


def test_credential_key_persists_independently_of_password_changes():
    before = auth_storage.get_or_create_credential_encryption_key()
    auth_storage._credential_encryption_key_cache = None
    after = auth_storage.get_or_create_credential_encryption_key()
    assert before == after


def test_credentials_survive_real_process_restart(tmp_path):
    backend_dir = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "UNSLOTH_STUDIO_HOME": str(tmp_path / "studio-home"),
    }
    app_setup = """
from fastapi import FastAPI
from fastapi.testclient import TestClient
from auth.authentication import authenticated_via_api_key, get_current_credential, get_current_subject
import importlib.util
from pathlib import Path
import sys
import types
routes_dir = Path.cwd() / "routes"
routes_package = types.ModuleType("routes")
routes_package.__path__ = [str(routes_dir)]
sys.modules["routes"] = routes_package
def load_route(name, filename):
    spec = importlib.util.spec_from_file_location(name, routes_dir / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
load_route("routes.provider_credentials", "provider_credentials.py")
providers = load_route("restart_providers_route", "providers.py")
settings = load_route("restart_settings_route", "settings.py")

from core.inference.key_exchange import init_key_pair
init_key_pair()
app = FastAPI()
app.include_router(providers.router, prefix="/api/providers")
app.include_router(settings.router, prefix="/api/settings")
app.dependency_overrides[get_current_subject] = lambda: "alice"
app.dependency_overrides[get_current_credential] = lambda: ("alice", None)
app.dependency_overrides[authenticated_via_api_key] = lambda: False
client = TestClient(app)
"""
    write_credentials = (
        app_setup
        + """
import base64
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
hf_response = client.put("/api/settings/hugging-face-token", json={"token": "hf_restart"})
assert hf_response.status_code == 200, hf_response.text
public_key_pem = client.get("/api/providers/public-key").json()["public_key"]
public_key = serialization.load_pem_public_key(public_key_pem.encode())
encrypted_key = base64.b64encode(public_key.encrypt(
    b"sk_restart",
    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None),
)).decode()
provider_response = client.post("/api/providers/", json={
    "provider_type": "custom",
    "display_name": "Restart provider",
    "base_url": "https://provider.invalid/v1",
    "models": ["restart-model"],
    "encrypted_api_key": encrypted_key,
})
assert provider_response.status_code == 201, provider_response.text
"""
    )
    use_credentials_after_restart = (
        app_setup
        + """
hf_response = client.get("/api/settings/hugging-face-token")
assert hf_response.status_code == 200, hf_response.text
assert hf_response.json() == {"token": "hf_restart", "has_token": True}
provider_rows = client.get("/api/providers/").json()
assert len(provider_rows) == 1 and provider_rows[0]["has_api_key"] is True
seen = {}
class FakeProviderClient:
    def __init__(self, **kwargs):
        seen["api_key"] = kwargs["api_key"]
    async def chat_completion(self, **_kwargs):
        return {}
    async def close(self):
        return None
providers.ExternalProviderClient = FakeProviderClient
use_response = client.post("/api/providers/test", json={
    "provider_type": "custom",
    "provider_id": provider_rows[0]["id"],
    "base_url": "https://attacker.invalid/v1",
    "model_id": "restart-model",
})
assert use_response.status_code == 200, use_response.text
assert use_response.json()["success"] is True
assert seen["api_key"] == "sk_restart"
"""
    )

    subprocess.run(
        [sys.executable, "-c", write_credentials],
        cwd = backend_dir,
        env = env,
        check = True,
        timeout = 30,
    )
    subprocess.run(
        [sys.executable, "-c", use_credentials_after_restart],
        cwd = backend_dir,
        env = env,
        check = True,
        timeout = 30,
    )


# ── API key transport envelope ──────────────────────────────────────
#
# Keys reach the backend as "v1.<RSA-wrapped AES key>.<nonce>.<ciphertext||tag>", each part
# base64. Builds predating the envelope send bare RSA ciphertext instead.

# The reported OVH key length, and the last length bare RSA-OAEP-SHA256 could carry.
OVH_KEY_LENGTH = 238
LEGACY_MAX_LENGTH = 190

# Spelled out rather than imported: this is the contract encryptProviderApiKey writes, so
# reading it from the module would let both sides drift together.
ENVELOPE_VERSION = "v1"
ENVELOPE_AAD = b"unsloth-studio-provider-key-v1"

OAEP = padding.OAEP(
    mgf = padding.MGF1(algorithm = hashes.SHA256()),
    algorithm = hashes.SHA256(),
    label = None,
)


@pytest.fixture(scope = "module", autouse = True)
def key_pair():
    init_key_pair()


def _b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


def encrypt_envelope(
    plaintext: str,
    *,
    aad: bytes = ENVELOPE_AAD,
    nonce: bytes = b"\x00" * 12,
    version: str = ENVELOPE_VERSION,
) -> str:
    """The frontend half of the envelope, mirroring encryptProviderApiKey."""
    aes_key = b"\x11" * 32
    wrapped = key_exchange._private_key.public_key().encrypt(aes_key, OAEP)
    sealed = AESGCM(aes_key).encrypt(nonce, plaintext.encode("utf-8"), aad)
    return ".".join([version, _b64(wrapped), _b64(nonce), _b64(sealed)])


def encrypt_legacy(plaintext: str) -> str:
    """A build predating the envelope: bare RSA-OAEP ciphertext, base64."""
    return _b64(key_exchange._private_key.public_key().encrypt(plaintext.encode("utf-8"), OAEP))


@pytest.mark.parametrize(
    "length",
    [1, LEGACY_MAX_LENGTH - 1, LEGACY_MAX_LENGTH, LEGACY_MAX_LENGTH + 1, OVH_KEY_LENGTH, 4096],
)
def test_envelope_round_trips_at_any_length(length):
    api_key = "k" + "x" * (length - 1)
    assert decrypt_api_key(encrypt_envelope(api_key)) == api_key


def test_envelope_round_trips_a_non_ascii_key():
    api_key = "sk-café-日本語-🔑-Ünïcödé"
    assert decrypt_api_key(encrypt_envelope(api_key)) == api_key


def test_the_envelope_carries_a_key_bare_rsa_cannot():
    api_key = "k" * OVH_KEY_LENGTH
    with pytest.raises(ValueError):
        encrypt_legacy(api_key)
    assert decrypt_api_key(encrypt_envelope(api_key)) == api_key


def test_legacy_bare_rsa_ciphertext_still_decrypts():
    assert decrypt_api_key(encrypt_legacy("sk-legacy")) == "sk-legacy"


def test_a_tampered_ciphertext_is_rejected():
    parts = encrypt_envelope("sk-tamper").split(".")
    sealed = bytearray(base64.b64decode(parts[3]))
    sealed[0] ^= 0x01
    parts[3] = _b64(bytes(sealed))
    with pytest.raises(InvalidTag):
        decrypt_api_key(".".join(parts))


def test_a_different_nonce_is_rejected():
    parts = encrypt_envelope("sk-nonce").split(".")
    parts[2] = _b64(b"\x09" * 12)
    with pytest.raises(InvalidTag):
        decrypt_api_key(".".join(parts))


def test_an_envelope_sealed_under_other_associated_data_is_rejected():
    with pytest.raises(InvalidTag):
        decrypt_api_key(encrypt_envelope("sk-aad", aad = b"some-other-protocol"))


@pytest.mark.parametrize("parts", [2, 3, 5])
def test_a_v1_envelope_with_the_wrong_part_count_is_rejected(parts):
    with pytest.raises(ValueError):
        decrypt_api_key(".".join(["v1"] + ["QUJD"] * (parts - 1)))


def test_envelope_parts_are_decoded_strictly():
    """Whitespace a permissive decoder would drop, restoring an otherwise valid envelope."""
    parts = encrypt_envelope("sk-b64").split(".")
    parts[3] = parts[3][:4] + "\n" + parts[3][4:]
    with pytest.raises(ValueError):
        decrypt_api_key(".".join(parts))


def test_an_unknown_version_is_rejected_rather_than_read_as_legacy():
    with pytest.raises(ValueError, match = "Unsupported"):
        decrypt_api_key(encrypt_envelope("sk-future", version = "v2"))
