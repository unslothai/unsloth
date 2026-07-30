# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from fastapi import Depends, FastAPI, status
from fastapi.testclient import TestClient

from auth import authentication


API_KEY = f"{authentication.API_KEY_PREFIX}office"
CREDENTIAL_SECRET = "credential-secret"
INVALID_API_KEY = f"{authentication.API_KEY_PREFIX}invalid"
JWT_LIKE_TOKEN = "header.payload.signature"
SUBJECT = "office-user"
TEST_PATH = "/protected"
VIA_API_KEY_PATH = "/via-api-key"
CREDENTIAL_PATH = "/credential"
PASSWORD_CHANGE_PATH = "/password-change"


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
