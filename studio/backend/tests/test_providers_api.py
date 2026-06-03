# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""
Integration tests for the external providers API.

Requires a running Unsloth Studio server. Configure via environment variables:

    export STUDIO_TEST_URL="http://localhost:8888"   # default
    export STUDIO_TEST_USER="unsloth"                # default
    export STUDIO_TEST_PASSWORD="..."                # required — see .bootstrap_password

    # Provider API keys — any left unset will have their tests automatically skipped
    export OPENAI_API_KEY="sk-..."
    export MISTRAL_API_KEY="..."
    export GOOGLE_API_KEY="..."
    export TOGETHER_API_KEY="..."
    export FIREWORKS_API_KEY="..."
    export PERPLEXITY_API_KEY="..."

Run:
    cd studio/backend
    pytest tests/test_providers_api.py -v -s
"""

import base64
import json
import os

import pytest
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

# ── Configuration ─────────────────────────────────────────────────

BASE_URL = os.getenv("STUDIO_TEST_URL", "http://localhost:8000")
USERNAME = os.getenv("STUDIO_TEST_USER", "unsloth")
PASSWORD = os.getenv("STUDIO_TEST_PASSWORD", "")

# These tests require a live Studio server reachable at BASE_URL with a known
# bootstrap password. Skip the whole module when that environment is missing
# (e.g. on CI runners) so pytest discovery does not error out.
pytestmark = pytest.mark.skipif(
    not PASSWORD,
    reason = "Integration test requires a running Studio server; set STUDIO_TEST_PASSWORD to enable.",
)

# Map provider_type → (env var name, model to use for inference test)
_PROVIDER_CONFIGS: dict[str, tuple[str, str]] = {
    "openai": ("OPENAI_API_KEY", "gpt-4o-mini"),
    "mistral": ("MISTRAL_API_KEY", "mistral-small-2506"),
    "gemini": ("GEMINI_API_KEY", "gemini-3-flash-preview"),
    "openrouter": ("OPENROUTER_API_KEY", "openai/gpt-4o-mini"),
    "anthropic": ("ANTHROPIC_API_KEY", "claude-haiku-4-5"),
    "deepseek": ("DEEPSEEK_API_KEY", "deepseek-chat"),
    "huggingface": ("HUGGINGFACE_API_KEY", "meta-llama/Llama-3.3-70B-Instruct"),
    "kimi": ("MOONSHOT_API_KEY", "moonshot-v1-8k"),
    "qwen": ("DASHSCOPE_API_KEY", "qwen-turbo"),
}

PROVIDER_KEYS: dict[str, str] = {
    ptype: os.getenv(env_var, "") for ptype, (env_var, _) in _PROVIDER_CONFIGS.items()
}

EXPECTED_PROVIDER_TYPES = set(_PROVIDER_CONFIGS.keys())

# ── Helpers ────────────────────────────────────────────────────────


def _url(path: str) -> str:
    return f"{BASE_URL}/{path.lstrip('/')}"


def _parse_sse_stream(response: requests.Response) -> tuple[str, bool]:
    """
    Read a streaming SSE response and return (assembled_text, saw_done).

    Each chunk is a JSON object with choices[0].delta.content.
    The stream ends with `data: [DONE]`.
    """
    reply_parts: list[str] = []
    saw_done = False

    for raw_line in response.iter_lines():
        if isinstance(raw_line, bytes):
            raw_line = raw_line.decode("utf-8")
        if not raw_line.startswith("data:"):
            continue
        data = raw_line[len("data:") :].strip()
        if data == "[DONE]":
            saw_done = True
            break
        try:
            chunk = json.loads(data)
            # Handle both error payloads and normal chunks
            if "error" in chunk:
                raise RuntimeError(f"Provider error in stream: {chunk['error']}")
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content") or ""
            if content:
                reply_parts.append(content)
        except (json.JSONDecodeError, IndexError, KeyError):
            pass  # skip malformed lines

    return "".join(reply_parts), saw_done


# ── Session-scoped fixtures ────────────────────────────────────────


@pytest.fixture(scope = "session")
def auth_headers() -> dict[str, str]:
    """
    Log in once per session and return auth headers.

    On a fresh Studio install the bootstrap password triggers a forced password
    change (must_change_password=True).  Any subsequent API call using that token
    returns 403 "Password change required".  This fixture detects that state,
    automatically completes the change-password flow, and re-logs in so all other
    tests get a fully usable token.

    The new password used during auto-change is:
        STUDIO_TEST_NEW_PASSWORD  (env var, optional)
        or PASSWORD + "-test"     (derived default)

    On the second run, set STUDIO_TEST_PASSWORD to the new password.
    """
    assert PASSWORD, (
        "STUDIO_TEST_PASSWORD is not set.\n"
        "Run: export STUDIO_TEST_PASSWORD=$(cat studio/backend/.bootstrap_password)"
    )

    resp = requests.post(
        _url("/api/auth/login"),
        json = {"username": USERNAME, "password": PASSWORD},
        timeout = 10,
    )
    assert resp.status_code == 200, f"Login failed ({resp.status_code}): {resp.text}"
    body = resp.json()
    token = body["access_token"]
    assert token, "access_token is empty"

    if body.get("must_change_password"):
        # Bootstrap token is restricted — only /api/auth/change-password works with it.
        # Auto-complete the forced change so the rest of the tests get a full token.
        new_password = os.getenv("STUDIO_TEST_NEW_PASSWORD") or f"{PASSWORD}-test"
        change_resp = requests.post(
            _url("/api/auth/change-password"),
            headers = {"Authorization": f"Bearer {token}"},
            json = {"current_password": PASSWORD, "new_password": new_password},
            timeout = 10,
        )
        assert (
            change_resp.status_code == 200
        ), f"Auto password-change failed ({change_resp.status_code}): {change_resp.text}"
        token = change_resp.json()["access_token"]

    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope = "session")
def public_key_pem(auth_headers: dict[str, str]) -> str:
    """Fetch RSA public key PEM once per session."""
    resp = requests.get(
        _url("/api/providers/public-key"),
        headers = auth_headers,
        timeout = 10,
    )
    assert resp.status_code == 200, f"Public key fetch failed: {resp.text}"
    pem = resp.json().get("public_key", "")
    assert pem.startswith("-----BEGIN PUBLIC KEY-----"), "Not a valid PEM public key"
    return pem


@pytest.fixture(scope = "session")
def vision_image_data_url() -> str:
    """
    Download the sloth image once per session and return it as a base64 data URI.

    Using a data URI instead of a remote URL ensures every provider receives
    the image inline — Gemini's OpenAI-compatible layer does not fetch external
    HTTP URLs, so raw image_url links silently produce empty replies for Gemini.
    """
    resp = requests.get(_VISION_IMAGE_URL, timeout = 30)
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0].strip()
    b64 = base64.b64encode(resp.content).decode("utf-8")
    return f"data:{content_type};base64,{b64}"


@pytest.fixture(scope = "session")
def encrypt_key(public_key_pem: str):
    """
    Return a callable encrypt_key(plaintext: str) -> str (base64 RSA-OAEP ciphertext).
    Uses the backend's RSA public key — mirrors what the frontend does.
    """
    # Decode PEM → load RSA public key
    pem_bytes = public_key_pem.encode("utf-8")
    rsa_pub = serialization.load_pem_public_key(pem_bytes)

    def _encrypt(plaintext: str) -> str:
        ciphertext = rsa_pub.encrypt(
            plaintext.encode("utf-8"),
            padding.OAEP(
                mgf = padding.MGF1(algorithm = hashes.SHA256()),
                algorithm = hashes.SHA256(),
                label = None,
            ),
        )
        return base64.b64encode(ciphertext).decode("utf-8")

    return _encrypt


# ── TestAuth ────────────────────────────────────────────────────────


class TestAuth:
    def test_login_returns_token(self):
        """POST /api/auth/login returns a non-empty access_token."""
        assert PASSWORD, "STUDIO_TEST_PASSWORD not set"
        resp = requests.post(
            _url("/api/auth/login"),
            json = {"username": USERNAME, "password": PASSWORD},
            timeout = 10,
        )
        assert (
            resp.status_code == 200
        ), f"Login failed ({resp.status_code}): {resp.text}"
        body = resp.json()
        assert body.get("access_token"), "access_token is missing or empty"
        assert body.get("token_type") == "bearer"


# ── TestPublicKey ────────────────────────────────────────────────────


class TestPublicKey:
    def test_public_key_is_valid_pem(
        self, auth_headers: dict[str, str], public_key_pem: str
    ):
        """GET /api/providers/public-key returns an importable RSA PEM key."""
        pem_bytes = public_key_pem.encode("utf-8")
        key = serialization.load_pem_public_key(pem_bytes)
        key_size = key.key_size  # type: ignore[attr-defined]
        assert key_size >= 2048, f"Key size too small: {key_size}"
        print(f"\n  RSA-{key_size} public key OK")


# ── TestRegistry ────────────────────────────────────────────────────


class TestRegistry:
    def test_registry_returns_all_providers(self, auth_headers: dict[str, str]):
        """GET /api/providers/registry returns all supported providers."""
        resp = requests.get(
            _url("/api/providers/registry"),
            headers = auth_headers,
            timeout = 10,
        )
        assert resp.status_code == 200, f"Registry failed: {resp.text}"
        providers = resp.json()
        assert (
            len(providers) == 9
        ), f"Expected 9 providers, got {len(providers)}: {providers}"
        print(f"\n  {'Provider':<12} {'Base URL'}")
        print(f"  {'-'*12} {'-'*45}")
        for p in providers:
            print(f"  {p['provider_type']:<12} {p['base_url']}")

    def test_registry_has_expected_types(self, auth_headers: dict[str, str]):
        """All expected provider_type values are present in the registry."""
        resp = requests.get(
            _url("/api/providers/registry"),
            headers = auth_headers,
            timeout = 10,
        )
        assert resp.status_code == 200
        returned_types = {p["provider_type"] for p in resp.json()}
        missing = EXPECTED_PROVIDER_TYPES - returned_types
        assert not missing, f"Missing provider types: {missing}"

    def test_registry_entries_have_required_fields(self, auth_headers: dict[str, str]):
        """Each registry entry has provider_type, display_name, base_url, default_models."""
        resp = requests.get(
            _url("/api/providers/registry"), headers = auth_headers, timeout = 10
        )
        assert resp.status_code == 200
        for entry in resp.json():
            for field in (
                "provider_type",
                "display_name",
                "base_url",
                "default_models",
                "model_list_mode",
            ):
                assert field in entry, f"Missing field '{field}' in entry: {entry}"
            assert entry["model_list_mode"] in ("remote", "curated")
            assert isinstance(entry["default_models"], list)
            assert len(entry["default_models"]) > 0


# ── TestProviderCRUD ────────────────────────────────────────────────


class TestProviderCRUD:
    """
    These tests run sequentially within the class and share state via class variables.
    They create, read, update, and delete a single test provider config.
    """

    _created_id: str = ""

    def test_create_provider(self, auth_headers: dict[str, str]):
        """POST /api/providers/ creates a provider config and returns 201."""
        resp = requests.post(
            _url("/api/providers/"),
            headers = auth_headers,
            json = {"provider_type": "openai", "display_name": "Test OpenAI (pytest)"},
            timeout = 10,
        )
        assert (
            resp.status_code == 201
        ), f"Create failed ({resp.status_code}): {resp.text}"
        body = resp.json()
        assert body.get("id"), "No id in response"
        assert body["provider_type"] == "openai"
        assert body["display_name"] == "Test OpenAI (pytest)"
        assert body["is_enabled"] is True
        TestProviderCRUD._created_id = body["id"]
        print(f"\n  created id={body['id']}")

    def test_list_includes_created(self, auth_headers: dict[str, str]):
        """GET /api/providers/ includes the newly created config."""
        assert (
            TestProviderCRUD._created_id
        ), "No created_id (run test_create_provider first)"
        resp = requests.get(_url("/api/providers/"), headers = auth_headers, timeout = 10)
        assert resp.status_code == 200
        ids = [p["id"] for p in resp.json()]
        assert (
            TestProviderCRUD._created_id in ids
        ), f"Created id {TestProviderCRUD._created_id!r} not found in list: {ids}"
        print(f"\n  found id={TestProviderCRUD._created_id} in list of {len(ids)}")

    def test_update_display_name(self, auth_headers: dict[str, str]):
        """PUT /api/providers/{id} updates the display_name."""
        assert TestProviderCRUD._created_id, "No created_id"
        new_name = "Test OpenAI (pytest updated)"
        resp = requests.put(
            _url(f"/api/providers/{TestProviderCRUD._created_id}"),
            headers = auth_headers,
            json = {"display_name": new_name},
            timeout = 10,
        )
        assert (
            resp.status_code == 200
        ), f"Update failed ({resp.status_code}): {resp.text}"
        assert resp.json()["display_name"] == new_name
        print(f"\n  updated display_name to '{new_name}'")

    def test_delete_provider(self, auth_headers: dict[str, str]):
        """DELETE /api/providers/{id} removes the config (204) and it's gone from list."""
        assert TestProviderCRUD._created_id, "No created_id"
        resp = requests.delete(
            _url(f"/api/providers/{TestProviderCRUD._created_id}"),
            headers = auth_headers,
            timeout = 10,
        )
        assert (
            resp.status_code == 204
        ), f"Delete failed ({resp.status_code}): {resp.text}"

        # Confirm gone from list
        list_resp = requests.get(
            _url("/api/providers/"), headers = auth_headers, timeout = 10
        )
        ids = [p["id"] for p in list_resp.json()]
        assert TestProviderCRUD._created_id not in ids, "Deleted provider still in list"
        print(f"\n  deleted id={TestProviderCRUD._created_id} confirmed gone")


# ── TestProviderInference ────────────────────────────────────────────


# Build parametrize list: (provider_type, model, api_key) for configured providers only
_INFERENCE_PARAMS = [
    pytest.param(
        ptype,
        model,
        PROVIDER_KEYS.get(ptype, ""),
        id = ptype,
        marks = pytest.mark.skipif(
            not PROVIDER_KEYS.get(ptype, ""),
            reason = f"no {env_var} set",
        ),
    )
    for ptype, (env_var, model) in _PROVIDER_CONFIGS.items()
]


class TestProviderInference:
    """
    Live inference tests — one parametrized set per provider.
    Each test is automatically skipped when the provider's API key env var is not set.
    """

    @pytest.mark.parametrize("provider_type,model,api_key", _INFERENCE_PARAMS)
    def test_connection(
        self,
        auth_headers: dict[str, str],
        encrypt_key,
        provider_type: str,
        model: str,
        api_key: str,
    ):
        """POST /api/providers/test → success: true."""
        encrypted = encrypt_key(api_key)
        resp = requests.post(
            _url("/api/providers/test"),
            headers = auth_headers,
            json = {"provider_type": provider_type, "encrypted_api_key": encrypted},
            timeout = 30,
        )
        assert (
            resp.status_code == 200
        ), f"Request failed ({resp.status_code}): {resp.text}"
        body = resp.json()
        assert (
            body["success"] is True
        ), f"Connection test failed for {provider_type}: {body.get('message')}"
        print(f"\n  [{provider_type}] connection OK — {body['message']}")

    @pytest.mark.parametrize("provider_type,model,api_key", _INFERENCE_PARAMS)
    def test_list_models(
        self,
        auth_headers: dict[str, str],
        encrypt_key,
        provider_type: str,
        model: str,
        api_key: str,
    ):
        """POST /api/providers/models → non-empty list, print first 3."""
        encrypted = encrypt_key(api_key)
        resp = requests.post(
            _url("/api/providers/models"),
            headers = auth_headers,
            json = {"provider_type": provider_type, "encrypted_api_key": encrypted},
            timeout = 30,
        )
        assert (
            resp.status_code == 200
        ), f"Request failed ({resp.status_code}): {resp.text}"
        models = resp.json()
        assert isinstance(models, list), f"Expected list, got {type(models)}"
        assert len(models) > 0, f"No models returned for {provider_type}"
        preview = [m["id"] for m in models[:3]]
        print(f"\n  [{provider_type}] {len(models)} models — first 3: {preview}")

    @pytest.mark.parametrize("provider_type,model,api_key", _INFERENCE_PARAMS)
    def test_chat_inference(
        self,
        auth_headers: dict[str, str],
        encrypt_key,
        provider_type: str,
        model: str,
        api_key: str,
    ):
        """POST /v1/chat/completions with provider fields → streamed reply."""
        encrypted = encrypt_key(api_key)
        payload = {
            "messages": [{"role": "user", "content": "Say hello in one sentence."}],
            "stream": True,
            "temperature": 0.7,
            "max_tokens": 64,
            "provider_type": provider_type,
            "external_model": model,
            "encrypted_api_key": encrypted,
        }
        with requests.post(
            _url("/v1/chat/completions"),
            headers = {**auth_headers, "Content-Type": "application/json"},
            json = payload,
            stream = True,
            timeout = 60,
        ) as resp:
            assert (
                resp.status_code == 200
            ), f"Chat completions failed ({resp.status_code}): {resp.text[:500]}"
            reply, saw_done = _parse_sse_stream(resp)

        assert reply.strip(), f"Empty reply from {provider_type}/{model}"
        assert saw_done, f"Stream did not end with [DONE] for {provider_type}/{model}"
        print(f'\n  [{provider_type}/{model}] reply: "{reply.strip()}"')


# ── TestVisionInference ─────────────────────────────────────────────

# Sloth photo — used to test vision routing across providers
_VISION_IMAGE_URL = (
    "https://www.travelexcellence.com/images/where-to-see-sloths-in-costa-rica.jpg"
)

_VISION_PARAMS = [
    pytest.param(
        ptype,
        model,
        PROVIDER_KEYS.get(ptype, ""),
        id = ptype,
        marks = pytest.mark.skipif(
            not PROVIDER_KEYS.get(ptype, ""),
            reason = f"no key for {ptype}",
        ),
    )
    for ptype, (_, model) in _PROVIDER_CONFIGS.items()
    if ptype in {"openai", "mistral", "gemini", "anthropic", "openrouter"}
]


class TestVisionInference:
    """
    Send a 1×1 white PNG alongside a text question to each vision-capable provider.
    Verifies that image content parts survive the proxy and the provider replies.
    """

    @pytest.mark.parametrize("provider_type,model,api_key", _VISION_PARAMS)
    def test_vision_chat_inference(
        self,
        auth_headers: dict[str, str],
        encrypt_key,
        vision_image_data_url: str,
        provider_type: str,
        model: str,
        api_key: str,
    ):
        """Image URL + text message → non-empty streamed reply."""
        encrypted = encrypt_key(api_key)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Which animal is in this image? Reply in one word.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": vision_image_data_url},
                        },
                    ],
                }
            ],
            "stream": True,
            "max_tokens": 215,
            "provider_type": provider_type,
            "external_model": model,
            "encrypted_api_key": encrypted,
        }
        with requests.post(
            _url("/v1/chat/completions"),
            headers = {**auth_headers, "Content-Type": "application/json"},
            json = payload,
            stream = True,
            timeout = 60,
        ) as resp:
            assert (
                resp.status_code == 200
            ), f"Vision request failed ({resp.status_code}): {resp.text[:300]}"
            reply, saw_done = _parse_sse_stream(resp)

        assert reply.strip(), f"Empty reply from {provider_type}/{model}"
        assert saw_done, f"Stream did not end with [DONE] for {provider_type}/{model}"
        print(f"\n  [{provider_type}/{model}] vision reply: {reply.strip()!r}")


# ── TestLocalInferenceUnaffected ────────────────────────────────────


class TestLocalInferenceUnaffected:
    def test_chat_without_provider(self, auth_headers: dict[str, str]):
        """
        POST /v1/chat/completions without provider fields must not return 422 or 500.

        200 = a local model is loaded and responded.
        503 = no model loaded (expected in test environment — that's fine).
        Any other 4xx/5xx (except 503) = regression in request handling.
        """
        resp = requests.post(
            _url("/v1/chat/completions"),
            headers = {**auth_headers, "Content-Type": "application/json"},
            json = {
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False,
            },
            timeout = 15,
        )
        allowed = {200, 400, 503}
        assert resp.status_code in allowed, (
            f"Unexpected status {resp.status_code} for local inference path: {resp.text[:300]}\n"
            f"This likely means the provider fields broke the base request schema."
        )
        status_label = (
            "local model responded"
            if resp.status_code == 200
            else "no model loaded (expected)"
        )
        print(f"\n  status={resp.status_code} ({status_label}) — local path unaffected")
