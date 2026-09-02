# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio

import pytest
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient

from auth import authentication


API_KEY = f"{authentication.API_KEY_PREFIX}office"
CREDENTIAL_SECRET = "credential-secret"
KEYLESS_ADMIN_SECRET = "keyless-admin-secret"
DUMMY_X_API_KEY = "not-needed"
INVALID_API_KEY = f"{authentication.API_KEY_PREFIX}invalid"
JWT_LIKE_TOKEN = "header.payload.signature"
SUBJECT = "office-user"
TEST_PATH = "/protected"
VIA_API_KEY_PATH = "/via-api-key"
CREDENTIAL_PATH = "/credential"
PASSWORD_CHANGE_PATH = "/password-change"
DESKTOP_ONLY_PATH = "/desktop-only"
DESKTOP_REQUIRED_DETAIL = "This action requires the Unsloth desktop app."


def _protected_route(subject: str = Depends(authentication.get_current_subject)) -> dict[str, str]:
    return {"subject": subject}


def _via_api_key_route(
    via_api_key: bool = Depends(authentication.authenticated_via_api_key),
) -> dict[str, bool]:
    return {"via_api_key": via_api_key}


def _credential_route(
    credential: tuple[str, str | None] = Depends(authentication.get_current_credential),
) -> dict[str, str | None]:
    subject, generation = credential
    return {"subject": subject, "generation": generation}


def _password_change_route(
    subject: str = Depends(authentication.get_current_subject_allow_password_change),
) -> dict[str, str]:
    return {"subject": subject}


def _desktop_only_route(
    subject: str = Depends(authentication.get_current_subject_allow_password_change),
    is_desktop: bool = Depends(authentication.authenticated_via_desktop_jwt),
) -> dict[str, str]:
    if not is_desktop:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = DESKTOP_REQUIRED_DETAIL,
        )
    return {"subject": subject}


def test_reload_secret_delegates_to_loader(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(authentication, "load_jwt_secret", lambda: observed.append(True))

    authentication.reload_secret()

    assert observed == [True]


def test_x_api_key_authenticates_protected_routes(monkeypatch) -> None:
    app = FastAPI()
    app.get(TEST_PATH)(_protected_route)
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        lambda token: (SUBJECT, CREDENTIAL_SECRET) if token == API_KEY else None,
    )

    response = TestClient(app).get(
        TEST_PATH,
        headers = {authentication.X_API_KEY_HEADER: API_KEY},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"subject": SUBJECT}


def test_x_api_key_is_reported_as_programmatic_auth() -> None:
    app = FastAPI()
    app.get(VIA_API_KEY_PATH)(_via_api_key_route)

    response = TestClient(app).get(
        VIA_API_KEY_PATH,
        headers = {authentication.X_API_KEY_HEADER: API_KEY},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"via_api_key": True}


def test_bearer_api_key_is_still_reported_as_programmatic_auth() -> None:
    app = FastAPI()
    app.get(VIA_API_KEY_PATH)(_via_api_key_route)

    response = TestClient(app).get(
        VIA_API_KEY_PATH,
        headers = {"Authorization": f"Bearer {API_KEY}"},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"via_api_key": True}


def test_x_api_key_preserves_credential_generation(monkeypatch) -> None:
    app = FastAPI()
    app.get(CREDENTIAL_PATH)(_credential_route)
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        lambda token: (SUBJECT, CREDENTIAL_SECRET) if token == API_KEY else None,
    )

    response = TestClient(app).get(
        CREDENTIAL_PATH,
        headers = {authentication.X_API_KEY_HEADER: API_KEY},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {
        "subject": SUBJECT,
        "generation": authentication.credential_generation(CREDENTIAL_SECRET),
    }


def test_x_api_key_takes_precedence_over_keyless_admission(monkeypatch) -> None:
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        lambda token: (SUBJECT, CREDENTIAL_SECRET) if token == API_KEY else None,
    )
    monkeypatch.setattr(
        authentication,
        "get_user_and_secret",
        lambda username: (
            ("salt", "hash", KEYLESS_ADMIN_SECRET, False)
            if username == authentication.DEFAULT_ADMIN_USERNAME
            else None
        ),
    )

    subject, generation = asyncio.run(
        authentication.get_current_credential(authentication._KEYLESS_CREDENTIALS, API_KEY)
    )

    assert subject == SUBJECT
    assert generation == authentication.credential_generation(CREDENTIAL_SECRET)


def test_x_api_key_is_not_reported_as_credentialless_keyless_auth() -> None:
    assert (
        asyncio.run(
            authentication.authenticated_without_credential(
                authentication._KEYLESS_CREDENTIALS,
                API_KEY,
            )
        )
        is False
    )


def test_keyless_admission_still_resolves_to_admin_without_x_api_key(monkeypatch) -> None:
    monkeypatch.setattr(
        authentication,
        "get_user_and_secret",
        lambda username: (
            ("salt", "hash", KEYLESS_ADMIN_SECRET, False)
            if username == authentication.DEFAULT_ADMIN_USERNAME
            else None
        ),
    )

    subject, generation = asyncio.run(
        authentication.get_current_credential(authentication._KEYLESS_CREDENTIALS)
    )

    assert subject == authentication.DEFAULT_ADMIN_USERNAME
    assert generation == authentication.credential_generation(KEYLESS_ADMIN_SECRET)


def test_keyless_admission_still_counts_as_programmatic_without_x_api_key() -> None:
    assert asyncio.run(
        authentication.authenticated_via_api_key(authentication._KEYLESS_CREDENTIALS)
    )


def test_dummy_x_api_key_keeps_keyless_admission(monkeypatch) -> None:
    monkeypatch.setattr(
        authentication,
        "get_user_and_secret",
        lambda username: (
            ("salt", "hash", KEYLESS_ADMIN_SECRET, False)
            if username == authentication.DEFAULT_ADMIN_USERNAME
            else None
        ),
    )

    subject, generation = asyncio.run(
        authentication.get_current_credential(
            authentication._KEYLESS_CREDENTIALS,
            DUMMY_X_API_KEY,
        )
    )

    assert subject == authentication.DEFAULT_ADMIN_USERNAME
    assert generation == authentication.credential_generation(KEYLESS_ADMIN_SECRET)
    assert (
        asyncio.run(
            authentication.authenticated_without_credential(
                authentication._KEYLESS_CREDENTIALS,
                DUMMY_X_API_KEY,
            )
        )
        is True
    )


def test_bearer_credentials_take_precedence_over_dummy_x_api_key() -> None:
    credentials = HTTPAuthorizationCredentials(
        scheme = "Bearer",
        credentials = API_KEY,
    )

    assert (
        asyncio.run(
            authentication.authenticated_without_credential(
                credentials,
                DUMMY_X_API_KEY,
            )
        )
        is False
    )


def test_x_api_key_allows_password_change_dependency(monkeypatch) -> None:
    app = FastAPI()
    app.get(PASSWORD_CHANGE_PATH)(_password_change_route)
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        lambda token: (SUBJECT, CREDENTIAL_SECRET) if token == API_KEY else None,
    )

    response = TestClient(app).get(
        PASSWORD_CHANGE_PATH,
        headers = {authentication.X_API_KEY_HEADER: API_KEY},
    )

    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"subject": SUBJECT}


def test_x_api_key_password_change_routes_reach_desktop_guard(monkeypatch) -> None:
    app = FastAPI()
    app.get(DESKTOP_ONLY_PATH)(_desktop_only_route)
    monkeypatch.setattr(
        authentication,
        "validate_api_key_with_credential",
        lambda token: (SUBJECT, CREDENTIAL_SECRET) if token == API_KEY else None,
    )

    response = TestClient(app, raise_server_exceptions = False).get(
        DESKTOP_ONLY_PATH,
        headers = {authentication.X_API_KEY_HEADER: API_KEY},
    )

    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json() == {"detail": DESKTOP_REQUIRED_DETAIL}


def test_invalid_x_api_key_is_rejected(monkeypatch) -> None:
    app = FastAPI()
    app.get(TEST_PATH)(_protected_route)
    monkeypatch.setattr(authentication, "validate_api_key_with_credential", lambda _token: None)

    response = TestClient(app).get(
        TEST_PATH,
        headers = {authentication.X_API_KEY_HEADER: INVALID_API_KEY},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": authentication._invalid_api_key_detail(INVALID_API_KEY)}


def test_x_api_key_header_rejects_session_tokens() -> None:
    app = FastAPI()
    app.get(TEST_PATH)(_protected_route)

    response = TestClient(app).get(
        TEST_PATH,
        headers = {authentication.X_API_KEY_HEADER: JWT_LIKE_TOKEN},
    )

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": authentication._invalid_api_key_detail(JWT_LIKE_TOKEN)}


def test_missing_credentials_are_rejected() -> None:
    app = FastAPI()
    app.get(TEST_PATH)(_protected_route)

    response = TestClient(app).get(TEST_PATH)

    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json() == {"detail": authentication.NOT_AUTHENTICATED_DETAIL}


def test_dependency_rejects_missing_credentials_when_security_yields_none() -> None:
    with pytest.raises(HTTPException) as caught:
        asyncio.run(authentication.get_current_subject(None, None))

    assert caught.value.status_code == status.HTTP_401_UNAUTHORIZED
    assert caught.value.detail == authentication.NOT_AUTHENTICATED_DETAIL
