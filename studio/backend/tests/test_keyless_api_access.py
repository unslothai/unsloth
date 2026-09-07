# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import asyncio
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from auth import storage
from auth.authentication import (
    KEYLESS_FALLBACK_SCHEME,
    KEYLESS_SCHEME,
    admitted_without_credential,
    admitted_without_session,
    create_access_token,
    get_current_subject,
    security,
)
from utils.keyless_api_access import (
    KeylessToolPolicyMiddleware,
    KEYLESS_ADMISSION_STATE_KEY,
    KEYLESS_API_ACCESS_SETTING_KEY,
    _reset_scope_cache,
    access_exposure,
    asgi_request_is_keyless,
    get_keyless_api_access_scope,
    get_keyless_api_access_settings,
    get_keyless_api_tools_enabled,
    keyless_request_allowed,
    scope_covers,
    set_keyless_api_access,
)


# fmt: off
@pytest.fixture(autouse = True)
def isolated_auth_db(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
    monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
    monkeypatch.setattr(storage, "_bootstrap_password", None)
    monkeypatch.setattr(storage, "_api_key_pbkdf2_salt_cache", None)
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()
    yield
    storage._reset_api_key_hash_cache()
    _reset_scope_cache()


def seed_user(*, must_change_password = False):
    storage.create_initial_user(username = storage.DEFAULT_ADMIN_USERNAME,
                                password = "human-password-123", jwt_secret = secrets.token_urlsafe(64),
                                must_change_password = must_change_password)


def app_state(**overrides):
    state = SimpleNamespace(bind_host = "127.0.0.1", secure = False,
                            remote_access_is_colab = False, lan_access_is_colab = False,
                            lan_access_secure_launch = False, cloudflare_url = None)
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


def asgi_scope(*, path = "/v1/chat/completions", method = None, root_path = "",
               headers = None, state = None, server = ("127.0.0.1", 8000),
               client = ("127.0.0.1", 50000),):
    return {
        "type": "http",
        "method": method or ("GET" if path.startswith("/v1/models") else "POST"),
        "path": path,
        "root_path": root_path,
        "query_string": b"",
        "scheme": "http",
        "server": server,
        "client": client,
        "headers": [
            (name.lower().encode(), value.encode()) for name, value in (headers or {}).items()
        ],
        "app": SimpleNamespace(state = state or app_state()),
    }


def request_for(**kwargs):
    return Request(asgi_scope(**kwargs))


def bearer_request(token, **kwargs):
    return request_for(headers = {"Authorization": f"Bearer {token}"}, **kwargs)


def resolve(request):
    return asyncio.run(security(request))


def subject_of(request):
    return asyncio.run(get_current_subject(resolve(request)))


INFERENCE_POST_PATHS = "/v1/chat/completions /v1/chat/count_tokens /v1/completions /v1/embeddings /v1/messages /v1/messages/count_tokens /v1/responses".split()


def test_exact_route_matrix_matches_registered_topology():
    from routes.inference import router

    denied_posts = "/v1/load /v1/unload /v1/validate /v1/generate/stream /v1/audio/speech /v1/images/generations /v1/external/openai/containers/create /v1x/chat".split()
    allowed = {("POST", path) for path in INFERENCE_POST_PATHS} | {
        ("GET", "/v1/models"), ("GET", "/v1/models/unsloth/model")}
    denied = {("POST", path) for path in denied_posts} | {
        ("POST", "/v1/models"), ("GET", "/v1/chat/completions"), ("GET", "/v1/sandbox/abc")}
    assert all(scope_covers("inference", method, path) for method, path in allowed)
    assert not any(scope_covers("inference", method, path) for method, path in denied)
    assert scope_covers("inference", "GET", "/studio/v1/models/", "/studio/")
    assert not scope_covers("inference", "GET", "/studio-v2/v1/models", "/studio")
    assert not scope_covers("off", "POST", "/v1/chat/completions")
    assert scope_covers("full", "POST", "/api/train/start")

    registered = {(method, route.path) for route in router.routes
                  for method in getattr(route, "methods", set())}
    intended = {("POST", path.removeprefix("/v1")) for path in INFERENCE_POST_PATHS} | {
        ("GET", "/models"), ("GET", "/models/{model_id:path}")}
    assert intended <= registered


def test_settings_are_immediate_and_fail_closed(monkeypatch):
    import storage.studio_db as studio_db

    set_keyless_api_access("full", tools = True)
    assert keyless_request_allowed(request_for()) is True
    assert get_keyless_api_tools_enabled() is True
    set_keyless_api_access("off")
    assert keyless_request_allowed(request_for()) is False
    assert get_keyless_api_tools_enabled() is False

    set_keyless_api_access("full", tools = True)
    monkeypatch.setattr(studio_db, "get_app_settings",
                        lambda *_a, **_k: (_ for _ in ()).throw(OSError("db unavailable")))
    _reset_scope_cache()
    assert (get_keyless_api_access_scope(), get_keyless_api_tools_enabled()) == ("off", False)


def test_stale_refresh_cannot_reopen_a_closed_scope(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    _reset_scope_cache()
    read_done, write_done = threading.Event(), threading.Event()
    real_read = keyless._read_settings
    observed = []

    def delayed_read():
        value = real_read()
        read_done.set()
        assert write_done.wait(timeout = 10)
        return value

    monkeypatch.setattr(keyless, "_read_settings", delayed_read)
    reader = threading.Thread(target = lambda: observed.append(keyless._settings()))
    reader.start()
    try:
        assert read_done.wait(timeout = 10)
        set_keyless_api_access("off", tools = False)
    finally:
        write_done.set()
        reader.join(timeout = 10)
    assert observed == [("off", False)]


def test_concurrent_stale_cache_misses_coalesce_into_one_sqlite_read(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("inference", tools = False)
    _reset_scope_cache()

    clock = [0.0]
    monkeypatch.setattr(keyless.time, "monotonic", lambda: clock[0])
    real_read = keyless._read_settings
    call_count = 0
    count_lock = threading.Lock()
    release = threading.Event()
    n_threads = 25
    entered = threading.Barrier(n_threads + 1)

    def counted_read():
        nonlocal call_count
        with count_lock:
            call_count += 1
        assert release.wait(timeout = 10)
        return real_read()

    monkeypatch.setattr(keyless, "_read_settings", counted_read)

    results = [None] * n_threads

    def worker(i):
        entered.wait(timeout = 10)
        results[i] = keyless._settings()

    threads = [threading.Thread(target = worker, args = (i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    entered.wait(timeout = 10)
    time.sleep(0.2)  # let followers queue
    clock[0] = keyless._SETTINGS_CACHE_TTL_S + 1.0
    release.set()
    for t in threads:
        t.join(timeout = 10)

    assert call_count == 1
    assert results.count(("inference", False)) == 1
    assert results.count(("off", False)) == n_threads - 1


def test_concurrent_keyless_checks_through_the_real_entrypoint_stay_bounded(monkeypatch):
    import utils.keyless_api_access as keyless
    from utils.keyless_api_access import asgi_request_is_keyless

    set_keyless_api_access("inference", tools = False)
    keyless._cached_settings = (
        time.monotonic() - keyless._SETTINGS_CACHE_TTL_S - 1,
        "inference",
        False,
    )

    real_read = keyless._read_settings
    call_count = 0
    count_lock = threading.Lock()
    release = threading.Event()

    def counted_read():
        nonlocal call_count
        with count_lock:
            call_count += 1
        assert release.wait(timeout = 10)
        return real_read()

    monkeypatch.setattr(keyless, "_read_settings", counted_read)

    scope = asgi_scope(path = "/v1/models")
    n_requests = 20
    pool_workers = 4

    with ThreadPoolExecutor(max_workers = pool_workers) as pool:
        futures = [pool.submit(asgi_request_is_keyless, scope) for _ in range(n_requests)]
        time.sleep(0.2)  # fill the pool
        late_future = pool.submit(asgi_request_is_keyless, scope)
        release.set()
        results = [f.result(timeout = 10) for f in futures]
        late_result = late_future.result(timeout = 10)

    assert call_count == 1
    assert results.count(True) == 1
    assert results.count(False) == n_requests - 1
    assert isinstance(late_result, bool)


def test_async_stale_cache_miss_broadcasts_one_refresh_result(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("inference", tools = True)
    _reset_scope_cache()

    owner_started = threading.Event()
    release_owner = threading.Event()
    real_once = keyless._settings_once
    call_count = 0
    count_lock = threading.Lock()

    def counted_once():
        nonlocal call_count
        with count_lock:
            call_count += 1
        return real_once()

    def delayed_read():
        owner_started.set()
        assert release_owner.wait(timeout = 10)
        return "inference", True

    monkeypatch.setattr(keyless, "_settings_once", counted_once)
    monkeypatch.setattr(keyless, "_read_settings", delayed_read)

    async def exercise():
        cancelled = asyncio.create_task(keyless._settings_async())
        requests = [asyncio.create_task(keyless._settings_async()) for _ in range(100)]
        try:
            for _ in range(100):
                if owner_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert owner_started.is_set()
            await asyncio.sleep(0.1)
            assert call_count == 1
            cancelled.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled
        finally:
            release_owner.set()
        results = await asyncio.gather(*requests)
        assert [result[:2] for result in results] == [("inference", True)] * 100
        assert len({result[2] for result in results}) == 1
        assert call_count == 1

    asyncio.run(exercise())


def test_slow_refresh_does_not_exhaust_the_anyio_worker_pool(monkeypatch):
    import anyio.to_thread
    import utils.keyless_api_access as keyless
    from starlette.concurrency import run_in_threadpool

    set_keyless_api_access("inference", tools = False)
    keyless._cached_settings = (
        time.monotonic() - keyless._SETTINGS_CACHE_TTL_S - 1,
        "inference",
        False,
    )

    owner_started = threading.Event()
    release_owner = threading.Event()
    real_read = keyless._read_settings_from_db
    real_once = keyless._settings_once
    once_count = 0
    count_lock = threading.Lock()

    def delayed_read():
        owner_started.set()
        assert release_owner.wait(timeout = 10)
        return real_read()

    def counted_once():
        nonlocal once_count
        with count_lock:
            once_count += 1
        return real_once()

    monkeypatch.setattr(keyless, "_read_settings", delayed_read)
    monkeypatch.setattr(keyless, "_settings_once", counted_once)

    async def exercise():
        limiter = anyio.to_thread.current_default_thread_limiter()
        original_tokens = limiter.total_tokens
        limiter.total_tokens = 4

        observed = []

        async def downstream(scope, *_args):
            observed.append(scope["state"][KEYLESS_ADMISSION_STATE_KEY])

        middleware = KeylessToolPolicyMiddleware(downstream)
        requests = [
            asyncio.create_task(middleware(asgi_scope(), lambda: None, lambda _message: None))
            for _ in range(4)
        ]
        try:
            for _ in range(100):
                if owner_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert owner_started.is_set()
            assert once_count == 1

            unrelated = asyncio.create_task(run_in_threadpool(lambda: "available"))
            assert await asyncio.wait_for(unrelated, timeout = 1) == "available"
        finally:
            release_owner.set()
            await asyncio.gather(*requests)
            limiter.total_tokens = original_tokens
        assert observed == [True] * 4
        assert once_count == 1

    asyncio.run(exercise())


def test_failed_refresh_does_not_reuse_permissive_stale_settings(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("inference", tools = True)
    keyless._cached_settings = (
        time.monotonic() - keyless._SETTINGS_CACHE_TTL_S - 1,
        "inference",
        True,
    )
    refresh_started = threading.Event()
    release_refresh = threading.Event()

    def failed_read():
        refresh_started.set()
        assert release_refresh.wait(timeout = 10)
        raise OSError("db unavailable")

    monkeypatch.setattr(keyless, "_read_settings_from_db", failed_read)

    async def exercise():
        observed = []

        async def downstream(scope, *_args):
            observed.append(scope["state"][KEYLESS_ADMISSION_STATE_KEY])

        middleware = KeylessToolPolicyMiddleware(downstream)
        requests = [
            asyncio.create_task(middleware(asgi_scope(), lambda: None, lambda _message: None))
            for _ in range(2)
        ]
        try:
            for _ in range(100):
                if refresh_started.is_set() and keyless._settings_refresh_inflight is not None:
                    break
                await asyncio.sleep(0.01)
            assert refresh_started.is_set()
        finally:
            release_refresh.set()
            await asyncio.gather(*requests)
        assert observed == [False, False]

    asyncio.run(exercise())


def test_async_settings_tasks_do_not_retain_closed_event_loops(monkeypatch):
    import gc
    import weakref
    import utils.keyless_api_access as keyless

    loop_refs = []

    async def immediate_settings():
        return "off", False, 0

    monkeypatch.setattr(keyless, "_refresh_settings_async", immediate_settings)

    async def read_settings():
        loop_refs.append(weakref.ref(asyncio.get_running_loop()))
        assert (await keyless._settings_async())[:2] == ("off", False)

    for _ in range(3):
        asyncio.run(read_settings())
    gc.collect()

    assert all(ref() is None for ref in loop_refs)


def test_scope_write_preserves_tools_during_an_inflight_refresh(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    _reset_scope_cache()

    owner_started = threading.Event()
    release_owner = threading.Event()
    real_read = keyless._read_settings_from_db
    read_count = 0
    count_lock = threading.Lock()

    def delayed_first_read():
        nonlocal read_count
        with count_lock:
            read_count += 1
            first = read_count == 1
        if first:
            owner_started.set()
            assert release_owner.wait(timeout = 10)
        return real_read()

    monkeypatch.setattr(keyless, "_read_settings_from_db", delayed_first_read)
    reader = threading.Thread(target = keyless._settings)
    reader.start()
    try:
        assert owner_started.wait(timeout = 10)
        set_keyless_api_access("inference")
    finally:
        release_owner.set()
        reader.join(timeout = 10)

    assert not reader.is_alive()
    assert get_keyless_api_access_settings() == ("inference", True)


def test_scope_write_aborts_when_preserved_tools_cannot_be_read(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    monkeypatch.setattr(
        keyless,
        "_read_settings_from_db",
        lambda: (_ for _ in ()).throw(OSError("db unavailable")),
    )

    with pytest.raises(OSError, match = "db unavailable"):
        set_keyless_api_access("inference")

    assert get_keyless_api_access_settings() == ("full", True)


def test_disable_without_tools_does_not_depend_on_a_preservation_read(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("full", tools = True)
    monkeypatch.setattr(
        keyless,
        "_read_settings_from_db",
        lambda: (_ for _ in ()).throw(OSError("db unavailable")),
    )

    assert set_keyless_api_access("off") == ("off", False)
    assert get_keyless_api_access_settings() == ("off", False)


def test_keyless_write_has_no_fallible_post_commit_read(monkeypatch):
    import storage.studio_db as studio_db
    import utils.keyless_api_access as keyless

    real_get_connection = studio_db.get_connection
    result_read_attempted = False

    class GuardedConnection:
        def __init__(self, connection):
            self.connection = connection

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def execute(self, sql, *args):
            nonlocal result_read_attempted
            if "SELECT key, value_json FROM app_settings ORDER BY key" in " ".join(sql.split()):
                result_read_attempted = True
                raise RuntimeError("result read failed after commit")
            return self.connection.execute(sql, *args)

        def close(self):
            self.connection.close()
            raise RuntimeError("close failed after commit")

    monkeypatch.setattr(
        studio_db,
        "get_connection",
        lambda: GuardedConnection(real_get_connection()),
    )

    assert set_keyless_api_access("inference", tools = True) == ("inference", True)
    assert result_read_attempted is False
    monkeypatch.setattr(studio_db, "get_connection", real_get_connection)
    keyless._reset_scope_cache()
    assert get_keyless_api_access_settings() == ("inference", True)


def test_committed_disable_is_fail_closed_before_cache_publication(monkeypatch):
    import storage.studio_db as studio_db
    import utils.keyless_api_access as keyless

    set_keyless_api_access("inference", tools = True)
    keyless._cached_settings = (
        time.monotonic() - keyless._SETTINGS_CACHE_TTL_S - 1,
        "inference",
        True,
    )

    refresh_started = threading.Event()
    release_refresh = threading.Event()
    write_committed = threading.Event()
    release_writer = threading.Event()
    real_read = keyless._read_settings
    real_upsert = studio_db.upsert_app_settings

    def delayed_read():
        refresh_started.set()
        assert release_refresh.wait(timeout = 10)
        return real_read()

    def delayed_upsert(settings, **kwargs):
        result = real_upsert(settings, **kwargs)
        write_committed.set()
        assert release_writer.wait(timeout = 10)
        return result

    monkeypatch.setattr(keyless, "_read_settings", delayed_read)
    monkeypatch.setattr(studio_db, "upsert_app_settings", delayed_upsert)
    refresh_result = []
    refresh = threading.Thread(target = lambda: refresh_result.append(keyless._settings()))
    writer = threading.Thread(target = set_keyless_api_access, args = ("off",),
                              kwargs = {"tools": False})
    refresh.start()
    assert refresh_started.wait(timeout = 10)
    writer.start()
    try:
        assert write_committed.wait(timeout = 10)
        release_refresh.set()
        refresh.join(timeout = 10)
        assert refresh_result == [("off", False)]
        assert asgi_request_is_keyless(asgi_scope(path = "/v1/models")) is False
        assert get_keyless_api_tools_enabled() is False
    finally:
        release_writer.set()
        writer.join(timeout = 10)
        if refresh.is_alive():
            release_refresh.set()
            refresh.join(timeout = 10)

    assert not writer.is_alive() and not refresh.is_alive()
    assert get_keyless_api_access_settings() == ("off", False)


def test_middleware_rechecks_a_setting_changed_during_classification(monkeypatch):
    import utils.keyless_api_access as keyless

    set_keyless_api_access("inference", tools = True)
    classification_started = threading.Event()
    release_classification = threading.Event()
    real_classifier = keyless.asgi_request_is_keyless

    def delayed_classifier(scope, settings):
        admitted = real_classifier(scope, settings)
        classification_started.set()
        assert release_classification.wait(timeout = 10)
        return admitted

    monkeypatch.setattr(keyless, "asgi_request_is_keyless", delayed_classifier)

    async def exercise():
        observed = []

        async def downstream(scope, *_args):
            observed.append(scope["state"][KEYLESS_ADMISSION_STATE_KEY])

        request = asyncio.create_task(KeylessToolPolicyMiddleware(downstream)(
            asgi_scope(), lambda: None, lambda _message: None))
        try:
            for _ in range(100):
                if classification_started.is_set():
                    break
                await asyncio.sleep(0.01)
            assert classification_started.is_set()
            await asyncio.to_thread(set_keyless_api_access, "off", tools = False)
        finally:
            release_classification.set()
            await request
        assert observed == [False]

    asyncio.run(exercise())


def test_middleware_linearizes_admission_before_a_concurrent_disable():
    set_keyless_api_access("inference", tools = True)
    publication_started = threading.Event()
    writer_done = threading.Event()
    committed_before_publication = []

    class PublicationGate(dict):
        def setdefault(self, key, default = None,):
            if key == "state":
                publication_started.set()
                committed_before_publication.append(writer_done.wait(timeout = 0.2))
            return super().setdefault(key, default)

    scope = PublicationGate(asgi_scope())

    def disable():
        assert publication_started.wait(timeout = 10)
        set_keyless_api_access("off", tools = False)
        writer_done.set()

    writer = threading.Thread(target = disable)
    writer.start()

    async def exercise():
        observed = []

        async def downstream(request_scope, *_args):
            observed.append(request_scope["state"][KEYLESS_ADMISSION_STATE_KEY])

        await KeylessToolPolicyMiddleware(downstream)(scope, lambda: None, lambda _message: None)
        return observed

    try:
        assert asyncio.run(exercise()) == [True]
    finally:
        writer.join(timeout = 10)

    assert not writer.is_alive()
    assert committed_before_publication == [False]
    assert get_keyless_api_access_settings() == ("off", False)


def test_overlapping_writes_publish_in_commit_order(monkeypatch):
    import storage.studio_db as studio_db

    first_committed = threading.Event()
    release_first = threading.Event()
    second_committed = threading.Event()
    real_upsert = studio_db.upsert_app_settings

    def delayed_upsert(settings, **kwargs):
        result = real_upsert(settings, **kwargs)
        if settings[KEYLESS_API_ACCESS_SETTING_KEY] == "full":
            first_committed.set()
            assert release_first.wait(timeout = 10)
        else:
            second_committed.set()
        return result

    monkeypatch.setattr(studio_db, "upsert_app_settings", delayed_upsert)
    first = threading.Thread(target = set_keyless_api_access,
                             args = ("full",), kwargs = {"tools": True})
    second = threading.Thread(target = set_keyless_api_access,
                              args = ("off",), kwargs = {"tools": False})
    first.start()
    assert first_committed.wait(timeout = 10)
    second.start()
    try:
        assert not second_committed.wait(timeout = 0.1)
    finally:
        release_first.set()
        first.join(timeout = 10)
        second.join(timeout = 10)

    assert not first.is_alive() and not second.is_alive()
    assert get_keyless_api_access_settings() == ("off", False)


def test_full_scope_is_loopback_only():
    set_keyless_api_access("full")
    for server, client, bind_host in (
        (("127.0.0.1", 8000), ("127.0.0.1", 50000), "127.0.0.1"),
        (("::1", 8000), ("::1", 50000), "::1"),
        (("::ffff:127.0.0.1", 8000), ("::ffff:127.0.0.1", 50000), "::1"),
        (("localhost", 8000), ("localhost", 50000), "localhost"),
    ):
        assert keyless_request_allowed(request_for(
            server = server, client = client, state = app_state(bind_host = bind_host)))


def test_public_browser_and_private_lan_boundaries(monkeypatch):
    import lan_access
    from utils import host_policy

    set_keyless_api_access("inference")
    for overrides in ({"remote_access_is_colab": True}, {"secure": True},
                      {"cloudflare_url": "https://x.trycloudflare.com"}):
        assert not keyless_request_allowed(request_for(state = app_state(**overrides)))
    assert not keyless_request_allowed(request_for(headers = {"Origin": "https://evil.example"}))
    assert access_exposure(app_state(
        bind_host = "studio.lan", lan_access_launch_managed = True,
        lan_access_launch_addresses = ("192.168.1.24",),
    )) == "private_lan"

    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"running": True, "port": 8888,
                                 "addresses": ["192.168.1.24"]})
    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    assert access_exposure(app_state()) == "private_lan"
    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"addresses": ["100.64.0.10"]})
    assert access_exposure(app_state()) == "network"
    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"running": True, "port": 8888,
                                 "addresses": ["192.168.1.24"]})
    monkeypatch.setattr(host_policy, "_remote_connector_active", True)
    lan_request = request_for(method = "GET", path = "/v1/models",
                              server = ("192.168.1.24", 8888), client = ("192.168.1.90", 54321))
    assert keyless_request_allowed(lan_request)
    set_keyless_api_access("full")
    assert not keyless_request_allowed(lan_request)
    assert not keyless_request_allowed(request_for(state = app_state(bind_host = "0.0.0.0")))


def test_credentials_never_downgrade_to_keyless():
    seed_user()
    set_keyless_api_access("full")
    assert resolve(request_for()).scheme == KEYLESS_SCHEME
    for token in ("not-needed", "lm-studio", "ollama"):
        assert resolve(bearer_request(token)).scheme == KEYLESS_FALLBACK_SCHEME
    set_keyless_api_access("inference")
    with pytest.raises(HTTPException):
        subject_of(bearer_request("not-needed", path = "/v1/load"))
    set_keyless_api_access("full")

    for header in ("Bearer arbitrary", "garbage", "Basic abc", "Bearer"):
        with pytest.raises(HTTPException):
            subject_of(request_for(headers = {"Authorization": header}))
    duplicate = asgi_scope()
    duplicate["headers"] = [
        (b"authorization", b"Bearer not-needed"),
        (b"authorization", b"Bearer not-needed"),
    ]
    with pytest.raises(HTTPException):
        subject_of(Request(duplicate))

    session = create_access_token(storage.DEFAULT_ADMIN_USERNAME)
    assert subject_of(bearer_request(session)) == storage.DEFAULT_ADMIN_USERNAME
    expired_session = create_access_token(
        storage.DEFAULT_ADMIN_USERNAME, expires_delta = timedelta(seconds = -60))
    with pytest.raises(HTTPException):
        subject_of(bearer_request(expired_session))

    valid, _ = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "valid", expires_at = None)
    assert subject_of(bearer_request(valid)) == storage.DEFAULT_ADMIN_USERNAME
    for name, expires in (
        ("expired", (datetime.now(timezone.utc) - timedelta(days = 1)).isoformat()),
        ("revoked", None),
    ):
        raw, row = storage.create_api_key(
            username = storage.DEFAULT_ADMIN_USERNAME, name = name, expires_at = expires
        )
        if name == "revoked":
            storage.revoke_api_key(storage.DEFAULT_ADMIN_USERNAME, row["id"])
        with pytest.raises(HTTPException):
            subject_of(bearer_request(raw))


def _middleware_policy(scope):
    from state.tool_policy import get_tool_policy, reset_tool_policy, set_tool_policy

    observed = []

    async def downstream(*_args):
        observed.append(get_tool_policy())

    set_tool_policy(True)
    try:
        asyncio.run(KeylessToolPolicyMiddleware(downstream)(
            scope, lambda: None, lambda _message: None))
    finally:
        reset_tool_policy()
    return observed


def test_tool_policy_and_api_identity(monkeypatch):
    from core.inference.llama_keepwarm import _carries_bearer_credentials
    from routes import inference

    seed_user()
    set_keyless_api_access("inference", tools = False)
    request = request_for()
    assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME
    assert request.state.keyless_api_admitted is True
    assert admitted_without_credential(resolve(request)) is True
    assert admitted_without_session(request) is True
    assert inference._request_has_api_key(request) is True
    assert inference._request_used_api_key(request) is True
    assert inference._request_is_saved_credential_workflow(request) is False
    scope = asgi_scope(state = app_state())
    scope["state"] = {KEYLESS_ADMISSION_STATE_KEY: True}
    assert _carries_bearer_credentials(scope, "/v1/chat/completions") is True
    assert _middleware_policy(asgi_scope()) == [False]

    set_keyless_api_access("inference", tools = True)
    assert _middleware_policy(asgi_scope()) == [True]
    key, _ = storage.create_api_key(
        username = storage.DEFAULT_ADMIN_USERNAME, name = "tools", expires_at = None)
    set_keyless_api_access("inference", tools = False)
    assert _middleware_policy(asgi_scope(headers = {"Authorization": f"Bearer {key}"})) == [True]

    set_keyless_api_access("off")

    async def enabled_between_layers(scope, *_args):
        set_keyless_api_access("inference")
        with pytest.raises(HTTPException):
            await security(Request(scope))

    asyncio.run(KeylessToolPolicyMiddleware(enabled_between_layers)(
        asgi_scope(), lambda: None, lambda _message: None))


def test_desktop_password_setup_does_not_block_keyless_access(monkeypatch):
    import lan_access
    from utils import host_policy

    seed_user(must_change_password = True)
    monkeypatch.setattr(lan_access, "lan_listener_status",
                        lambda: {"running": True, "port": 8888,
                                 "addresses": ["192.168.1.24"]})
    monkeypatch.setattr(host_policy, "_lan_connector_active", True)
    set_keyless_api_access("inference")

    routes = [("POST", path) for path in INFERENCE_POST_PATHS] + [
        ("GET", "/v1/models"),
        ("GET", "/v1/models/unsloth/model"),
    ]
    transports = [
        {},
        {"server": ("192.168.1.24", 8888), "client": ("192.168.1.90", 54321)},
    ]
    for method, path in routes:
        for transport in transports:
            for bearer in (None, "not-needed"):
                request = (
                    request_for(method = method, path = path, **transport)
                    if bearer is None
                    else bearer_request(bearer, method = method, path = path, **transport)
                )
                assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME

    set_keyless_api_access("full")
    for bearer in (None, "not-needed"):
        request = (
            request_for(method = "POST", path = "/api/train/start")
            if bearer is None
            else bearer_request(bearer, method = "POST", path = "/api/train/start")
        )
        assert subject_of(request) == storage.DEFAULT_ADMIN_USERNAME


def test_setup_is_still_owed_after_keyless_access():
    """Admitting a keyless caller must not settle the password setup it skipped.

    The UI routes to /change-password off ``/api/auth/status``, which takes no auth
    dependency, and off the 403 a browser session still gets. Keyless reaches neither.
    """
    from routes.auth import auth_status

    seed_user(must_change_password = True)
    set_keyless_api_access("inference")
    assert subject_of(request_for()) == storage.DEFAULT_ADMIN_USERNAME

    assert storage.requires_password_change(storage.DEFAULT_ADMIN_USERNAME)
    assert auth_status().requires_password_change is True

    # The literal frontend/src/features/auth/api.ts matches to redirect the UI.
    record = storage.get_user_and_secret(storage.DEFAULT_ADMIN_USERNAME)
    token = create_access_token(subject = storage.DEFAULT_ADMIN_USERNAME, secret = record[2])
    with pytest.raises(HTTPException) as excinfo:
        subject_of(bearer_request(token))
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Password change required"


def test_browser_guards_hold_before_password_setup():
    """A page on another site stays out while the seeded password is still in place.

    The gate that no longer applies to keyless callers was incidentally doubling for
    these, so they are pinned here on their own footing.
    """
    seed_user(must_change_password = True)
    set_keyless_api_access("inference")
    for headers in ({"Sec-Fetch-Site": "cross-site"}, {"Sec-Fetch-Site": "none"},
                    {"Host": "evil.example"}):
        assert not keyless_request_allowed(request_for(headers = headers))
    assert keyless_request_allowed(request_for(headers = {"Sec-Fetch-Site": "same-origin"}))


def test_protected_side_effect_guards_remain_wired():
    import inspect
    from routes import auth, inference, mcp_servers, preview, rag, training_history, video

    auth_source = inspect.getsource(auth)
    assert auth_source.count("_require_a_credential_of_its_own(") >= 6
    assert "_require_a_credential_of_its_own" in inspect.getsource(auth.change_password)
    assert all(
        "request_admitted_without_credential" in inspect.getsource(handler)
        for handler in (inference._maybe_auto_switch_model, inference.openai_chat_completions)
    )
    assert all(
        "authenticated_without_credential" in inspect.getsource(handler)
        and "not no_credential" in inspect.getsource(handler)
        for handler in (
            preview.list_previews,
            training_history.list_training_runs,
            training_history.get_training_run_detail,
            training_history.update_training_run,
        )
    )
    assert all(
        "request_admitted_without_credential" in inspect.getsource(handler)
        for handler in (video.get_gallery_video_signed_url, rag.document_file_url)
    )
    assert "request_admitted_without_credential" in inspect.getsource(
        inference.openai_image_generations
    )
    assert all(
        "no_credential" in inspect.getsource(handler)
        for handler in (mcp_servers.list_mcp_servers, mcp_servers.update_mcp_server)
    )
    main_source = (Path(__file__).parents[1] / "main.py").read_text(encoding = "utf-8")
    assert 'app.state.secure = os.environ.get("UNSLOTH_SECURE") == "1"' in main_source
    assert security.scheme_name == "HTTPBearer"


def test_keyless_idle_restore_requires_the_requested_model(monkeypatch):
    from core.inference import llama_keepwarm as kw; import auth.authentication as authentication
    from studio.backend.tests import test_openai_auto_switch as auto
    backend = auto._FakeBackend(None); rec = auto._LoadRecorder(backend)
    auto._wire(monkeypatch, enabled = False, resolves_to = None, backend = backend, recorder = rec)
    monkeypatch.setattr(auto.settings, "idle_unload_is_configured", lambda: True)
    monkeypatch.setattr(kw, "_last_unloaded_model", ("/cache/snap/A", "Q4_K_M", "org/A-GGUF"))
    monkeypatch.setattr(authentication, "request_admitted_without_credential", lambda _r: True)
    auto._run_hook("org/B-GGUF"); assert rec.calls == []
    auto._run_hook("org/A-GGUF:Q4_K_M"); assert len(rec.calls) == 1
# fmt: on
