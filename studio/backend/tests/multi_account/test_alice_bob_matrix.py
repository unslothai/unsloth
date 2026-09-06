# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sqlite3
from contextlib import closing

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from auth import policy
from storage import studio_db
from utils.account_context import run_as

from .factories import FACTORIES, initialize_workspaces, seed_resource, snapshot_resource
from .inventory import OBJECT_ROUTES, ROUTES, collect_routes, looks_like_object_id, render_inventory, walk_router, worker_for
from .support import bearer

ACTORS = ("owner", "right", "wrong", "unauthenticated", "deactivated")


def matrix_parameters():
    for case in OBJECT_ROUTES:
        for actor in ACTORS:
            marks = []
            if case.key not in FACTORIES:
                marks.append(pytest.mark.xfail(strict = True, reason = f"worker {worker_for(case)}"))
            yield pytest.param(case, actor, id = f"{case.key}[{actor}]", marks = marks)


@pytest.mark.parametrize("case,actor", list(matrix_parameters()))
def test_object_route_account_matrix(case, actor, request):
    assert case.key in FACTORIES, f"Uncovered resource factory: {case.key}; see artifacts/route_inventory.md"
    accounts = request.getfixturevalue("accounts")
    auth_db = request.getfixturevalue("isolated_auth")
    factory = FACTORIES[case.key]
    initialize_workspaces(accounts)
    params = seed_resource(factory, accounts["alice"])
    before = snapshot_resource(accounts["alice"])
    username = {"owner": "unsloth", "right": "alice", "wrong": "bob", "deactivated": "alice"}.get(actor)
    headers = bearer(username) if username else {}
    if actor == "deactivated":
        with closing(sqlite3.connect(auth_db.DB_PATH)) as conn:
            conn.execute("UPDATE auth_user SET is_active=0 WHERE username='alice'")
            conn.commit()
        policy.invalidate_account_cache()

    app = FastAPI()
    app.include_router(case.router, prefix = "/matrix")
    with TestClient(app, raise_server_exceptions = False) as client:
        response = client.request(
            case.method, "/matrix" + case.path.format(**params), headers = headers, json = factory.body
        )
    expected = {"owner": {404}, "right": {factory.success}, "wrong": {404},
                "unauthenticated": {401, 403}, "deactivated": {401}}
    assert response.status_code in expected[actor], (case.key, actor, response.status_code, response.text)
    if actor == "right":
        if factory.fragment:
            assert factory.fragment in response.text
    else:
        assert snapshot_resource(accounts["alice"]) == before
        if factory.name == "api-key":
            assert len(auth_db.list_api_keys("alice")) == 1


@pytest.mark.parametrize("case", [case for case in OBJECT_ROUTES if case.key in FACTORIES], ids = lambda case: case.key)
def test_owner_can_still_use_own_resource(case, accounts):
    initialize_workspaces(accounts)
    factory = FACTORIES[case.key]
    params = seed_resource(factory, accounts["unsloth"])
    app = FastAPI()
    app.include_router(case.router, prefix = "/matrix")
    with TestClient(app, raise_server_exceptions = False) as client:
        response = client.request(
            case.method, "/matrix" + case.path.format(**params), headers = bearer("unsloth"), json = factory.body
        )
    assert response.status_code == factory.success, response.text


@pytest.mark.xfail(strict = True, reason = "worker 02")
def test_first_database_use_in_each_account_initializes_its_schema(accounts):
    for account in accounts.values():
        assert run_as(account, studio_db.list_chat_threads) == []


def test_inventory_contains_hidden_routes_and_no_duplicate_method_paths():
    assert ROUTES == collect_routes()
    assert len({case.key for case in ROUTES}) == len(ROUTES)
    assert "routes.rag:GET:/jobs/{job_id}/events" in {case.key for case in ROUTES}
    assert set(FACTORIES) <= {case.key for case in OBJECT_ROUTES}, "A registered route disappeared or changed shape"
    generated = {(parameter.values[0].key, parameter.values[1]) for parameter in matrix_parameters()}
    assert generated == {(case.key, actor) for case in OBJECT_ROUTES for actor in ACTORS}
    report = render_inventory()
    assert all(f"`{case.path}`" in report for case in OBJECT_ROUTES)


def test_nested_router_prefixes_are_collected():
    nested, parent = APIRouter(), APIRouter()

    @nested.get("/{item_id}", include_in_schema = False)
    def get_item(item_id: str):
        return item_id

    parent.include_router(nested, prefix = "/items")
    assert [path for path, _ in walk_router(parent)] == ["/items/{item_id}"]


@pytest.mark.parametrize("name", ["id", "thread_id", "run_id", "job_id", "server_id", "key_id", "filename", "ref"])
def test_object_parameter_detection(name):
    assert looks_like_object_id(name)
