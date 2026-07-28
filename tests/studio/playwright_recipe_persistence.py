# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Real-browser recipe persistence lifecycle integration test.

Run against a disposable Studio instance after its bootstrap password has been
changed::

    BASE_URL=http://127.0.0.1:18894 STUDIO_PW=... \
        python tests/studio/playwright_recipe_persistence.py

The test intentionally uses the existing Python Playwright UI-smoke harness.
It does not add a frontend test runner or emulate IndexedDB.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import secrets
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import Page, Response, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "studio" / "backend"
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(BACKEND))

from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    install_view_transition_killer,
    wait_for_health,
)
from storage import studio_db  # noqa: E402


def load_auth_storage():
    """Load the auth storage helper without importing JWT-dependent auth APIs."""
    path = BACKEND / "auth" / "storage.py"
    spec = importlib.util.spec_from_file_location("_studio_auth_storage", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load auth storage helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


auth_storage = load_auth_storage()


BASE = os.environ["BASE_URL"].rstrip("/")
ACCOUNT_A_PASSWORD = os.environ["STUDIO_PW"]
ART = Path(os.environ.get("PW_ART_DIR", "logs/playwright_recipe_persistence"))
ART.mkdir(parents = True, exist_ok = True)

RECIPE_ID = f"recipe-sync-{secrets.token_hex(6)}"
ACCOUNT_A_NAME = f"Legacy recipe A {RECIPE_ID}"
ACCOUNT_B_NAME = f"Legacy recipe B {RECIPE_ID}"
LEGACY_SOURCE = "recipe-indexeddb-v1"


def info(message: str) -> None:
    print(f"[recipe-sync] {message}", flush = True)


def request_json(
    path: str,
    *,
    method: str = "GET",
    body: object | None = None,
) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        f"{BASE}{path}",
        data = data,
        method = method,
        headers = {"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout = 30) as response:
        return json.load(response)


def login(username: str, password: str) -> dict:
    payload = request_json(
        "/api/auth/login",
        method = "POST",
        body = {"username": username, "password": password},
    )
    assert payload.get("access_token"), f"login for {username!r} returned no access token"
    assert payload.get("refresh_token"), f"login for {username!r} returned no refresh token"
    assert not payload.get(
        "must_change_password"
    ), f"account {username!r} still requires a password change"
    return payload


def jwt_subject(token: str) -> str:
    encoded = token.split(".")[1]
    encoded += "=" * (-len(encoded) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    subject = payload.get("sub")
    assert isinstance(subject, str) and subject
    return subject


def set_tokens(page: Page, tokens: dict) -> None:
    page.evaluate(
        """({accessToken, refreshToken}) => {
            const key = "unsloth_auth_token";
            const previous = localStorage.getItem(key);
            localStorage.setItem(key, accessToken);
            localStorage.setItem("unsloth_auth_refresh_token", refreshToken);
            window.dispatchEvent(new StorageEvent("storage", {
                key,
                oldValue: previous,
                newValue: accessToken,
                storageArea: localStorage,
            }));
        }""",
        {
            "accessToken": tokens["access_token"],
            "refreshToken": tokens["refresh_token"],
        },
    )


def delete_legacy_database(page: Page) -> None:
    page.evaluate(
        """() => new Promise((resolve, reject) => {
            const request = indexedDB.deleteDatabase("unsloth-data-recipes");
            request.onsuccess = () => resolve(null);
            request.onerror = () => reject(request.error);
            request.onblocked = () => reject(new Error("legacy IndexedDB delete was blocked"));
        })"""
    )


def clear_legacy_import_claim(page: Page) -> None:
    page.evaluate(
        """() => new Promise((resolve, reject) => {
            localStorage.removeItem("user-assets:recipe-indexeddb-v1:owner");
            const request = indexedDB.deleteDatabase("unsloth-user-assets-migration-claims");
            request.onsuccess = () => resolve(null);
            request.onerror = () => reject(request.error);
            request.onblocked = () => reject(new Error("migration claim delete was blocked"));
        })"""
    )


def put_legacy_recipe(page: Page, name: str) -> None:
    page.evaluate(
        """({id, name}) => new Promise((resolve, reject) => {
            const request = indexedDB.open("unsloth-data-recipes", 1);
            request.onupgradeneeded = () => {
                if (!request.result.objectStoreNames.contains("recipes")) {
                    request.result.createObjectStore("recipes", {keyPath: "id"});
                }
            };
            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                const database = request.result;
                const transaction = database.transaction("recipes", "readwrite");
                transaction.objectStore("recipes").put({
                    id,
                    name,
                    payload: {nodes: []},
                    revision: 0,
                    createdAt: 1_700_000_000_000,
                    updatedAt: 1_700_000_000_001,
                });
                transaction.oncomplete = () => {
                    database.close();
                    resolve(null);
                };
                transaction.onerror = () => {
                    database.close();
                    reject(transaction.error);
                };
            };
        })""",
        {"id": RECIPE_ID, "name": name},
    )


def is_user_assets_response(
    response: Response,
    *,
    method: str,
    suffix: str,
    token: str | None = None,
) -> bool:
    if response.request.method != method:
        return False
    if urlparse(response.url).path != f"/api/user-assets{suffix}":
        return False
    if token is None:
        return True
    return response.request.headers.get("authorization") == f"Bearer {token}"


def wait_for_recipe_list(page: Page, tokens: dict) -> Response:
    with page.expect_response(
        lambda response: is_user_assets_response(
            response,
            method = "GET",
            suffix = "/recipes",
            token = tokens["access_token"],
        ),
        timeout = 30_000,
    ) as response_info:
        set_tokens(page, tokens)
    response = response_info.value
    assert response.status == 200, f"recipe list returned {response.status}: {response.text()}"
    return response


def remove_legacy_ledger(subject: str) -> None:
    """Simulate a lost import ledger so the retained tombstone is exercised."""

    connection = studio_db.get_connection()
    try:
        connection.execute(
            """
            DELETE FROM user_asset_legacy_imports
            WHERE owner_subject = ? AND source = ?
              AND entity_kind = 'recipe' AND legacy_id = ?
            """,
            (subject, LEGACY_SOURCE, RECIPE_ID),
        )
        connection.commit()
    finally:
        connection.close()


def cleanup_assets(*subjects: str) -> None:
    connection = studio_db.get_connection()
    try:
        for subject in subjects:
            connection.execute(
                "DELETE FROM data_recipe_executions WHERE owner_subject = ? AND recipe_id = ?",
                (subject, RECIPE_ID),
            )
            connection.execute(
                "DELETE FROM user_asset_legacy_imports WHERE owner_subject = ? AND legacy_id = ?",
                (subject, RECIPE_ID),
            )
            connection.execute(
                "DELETE FROM data_recipes WHERE owner_subject = ? AND id = ?",
                (subject, RECIPE_ID),
            )
        connection.commit()
    finally:
        connection.close()


def main() -> None:
    wait_for_health(BASE, timeout = 30.0, info = info)
    status = request_json("/api/auth/status")
    account_a_username = status.get("default_username")
    assert isinstance(account_a_username, str) and account_a_username

    account_b_username = f"recipe_sync_b_{secrets.token_hex(6)}"
    account_b_password = f"RecipeSync-{secrets.token_urlsafe(18)}"
    auth_storage.create_initial_user(
        account_b_username,
        account_b_password,
        secrets.token_urlsafe(48),
        must_change_password = False,
    )

    tokens_a: dict | None = None
    tokens_b: dict | None = None
    subject_a = account_a_username
    subject_b = account_b_username
    browser = None
    try:
        tokens_a = login(account_a_username, ACCOUNT_A_PASSWORD)
        tokens_b = login(account_b_username, account_b_password)
        subject_a = jwt_subject(tokens_a["access_token"])
        subject_b = jwt_subject(tokens_b["access_token"])
        assert subject_a != subject_b

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless = True,
                args = chromium_launch_args(),
            )
            context = browser.new_context(
                viewport = {"width": 1280, "height": 900},
                reduced_motion = "reduce",
            )
            install_view_transition_killer(context)
            page = context.new_page()
            page.set_default_timeout(30_000)
            page.goto(f"{BASE}/login", wait_until = "domcontentloaded")
            set_tokens(page, tokens_a)
            page.goto(f"{BASE}/chat", wait_until = "domcontentloaded")

            info("seed native legacy IndexedDB and migrate it as account A")
            delete_legacy_database(page)
            put_legacy_recipe(page, ACCOUNT_A_NAME)
            migration_requests = []
            page.on(
                "request",
                lambda request: (
                    migration_requests.append(request)
                    if request.method == "POST"
                    and urlparse(request.url).path == "/api/user-assets/legacy-import"
                    else None
                ),
            )
            with page.expect_response(
                lambda response: is_user_assets_response(
                    response,
                    method = "POST",
                    suffix = "/legacy-import",
                    token = tokens_a["access_token"],
                ),
                timeout = 30_000,
            ) as migration_info:
                page.goto(f"{BASE}/data-recipes", wait_until = "domcontentloaded")
            migration = migration_info.value
            assert migration.status == 200
            migration_body = migration.json()
            assert migration_body["recipes"][0]["outcome"] in {"imported", "redacted"}
            migration_payload = migration.request.post_data_json
            assert migration_payload["source"] == LEGACY_SOURCE
            assert migration_payload["confirmSubject"] == subject_a
            assert [item["id"] for item in migration_payload["recipes"]] == [RECIPE_ID]
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "visible")

            info("remove browser source and prove a hard reload reads server persistence")
            delete_legacy_database(page)
            posts_before_reload = len(migration_requests)
            with page.expect_response(
                lambda response: is_user_assets_response(
                    response,
                    method = "GET",
                    suffix = "/recipes",
                    token = tokens_a["access_token"],
                ),
                timeout = 30_000,
            ) as reload_list_info:
                page.reload(wait_until = "domcontentloaded")
            reload_list = reload_list_info.value
            assert reload_list.status == 200
            reloaded_recipe = next(
                recipe for recipe in reload_list.json()["recipes"] if recipe["id"] == RECIPE_ID
            )
            assert reloaded_recipe["name"] == ACCOUNT_A_NAME
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "visible")
            page.wait_for_timeout(500)
            assert len(migration_requests) == posts_before_reload

            info("switch A -> B -> A in the same page and reject cross-account cache reuse")
            list_b = wait_for_recipe_list(page, tokens_b)
            assert list_b.json()["recipes"] == []
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "hidden")
            list_a = wait_for_recipe_list(page, tokens_a)
            assert any(recipe["id"] == RECIPE_ID for recipe in list_a.json()["recipes"])
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "visible")

            info("delete on the server, clear only the ledger, and force legacy tombstone handling")
            put_legacy_recipe(page, ACCOUNT_A_NAME)
            with page.expect_response(
                lambda response: (
                    response.request.method == "DELETE"
                    and urlparse(response.url).path == f"/api/user-assets/recipes/{RECIPE_ID}"
                    and response.request.headers.get("authorization")
                    == f"Bearer {tokens_a['access_token']}"
                ),
                timeout = 30_000,
            ) as delete_info:
                page.get_by_role("button", name = f"Delete {ACCOUNT_A_NAME}").click()
            assert delete_info.value.status == 204
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "hidden")
            remove_legacy_ledger(subject_a)

            with page.expect_response(
                lambda response: is_user_assets_response(
                    response,
                    method = "POST",
                    suffix = "/legacy-import",
                    token = tokens_a["access_token"],
                ),
                timeout = 30_000,
            ) as retired_info:
                page.reload(wait_until = "domcontentloaded")
            retired = retired_info.value
            assert retired.status == 200
            retired_recipes = retired.json()["recipes"]
            assert len(retired_recipes) == 1
            assert retired_recipes[0]["id"] == RECIPE_ID
            assert retired_recipes[0]["outcome"] == "id_retired"
            assert retired_recipes[0]["reason"] == "id_retired"
            assert not retired_recipes[0].get("redactedPaths")
            page.get_by_text(ACCOUNT_A_NAME, exact = True).wait_for(state = "hidden")

            info("prove account A's tombstone does not suppress account B's same legacy id")
            put_legacy_recipe(page, ACCOUNT_B_NAME)
            # The test has replaced the legacy source with B's synthetic data;
            # release A's one-device migration claim before exercising B's import.
            clear_legacy_import_claim(page)
            with page.expect_response(
                lambda response: is_user_assets_response(
                    response,
                    method = "POST",
                    suffix = "/legacy-import",
                    token = tokens_b["access_token"],
                ),
                timeout = 30_000,
            ) as account_b_import_info:
                set_tokens(page, tokens_b)
            account_b_import = account_b_import_info.value
            assert account_b_import.status == 200
            assert account_b_import.json()["recipes"][0]["outcome"] in {"imported", "redacted"}
            page.get_by_text(ACCOUNT_B_NAME, exact = True).wait_for(state = "visible")

            list_a_after_delete = wait_for_recipe_list(page, tokens_a)
            assert all(
                recipe["id"] != RECIPE_ID for recipe in list_a_after_delete.json()["recipes"]
            )
            page.get_by_text(ACCOUNT_B_NAME, exact = True).wait_for(state = "hidden")
            page.screenshot(path = str(ART / "recipe-persistence-pass.png"), full_page = True)
            info("PASS recipe persistence lifecycle")
            context.close()
            browser.close()
            browser = None
    except Exception:
        if browser is not None:
            try:
                page.screenshot(path = str(ART / "recipe-persistence-failure.png"), full_page = True)
            except Exception:
                pass
        raise
    finally:
        try:
            try:
                cleanup_assets(subject_a, subject_b)
            finally:
                auth_storage.revoke_user_refresh_tokens(account_b_username)
                auth_storage.delete_user(account_b_username)
        finally:
            if browser is not None:
                browser.close()


if __name__ == "__main__":
    main()
