# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A request body is read before the route's auth dependency runs, so the body cap middleware must
not buffer a large body (up to the 500 MiB default) for a caller without a credential. Signed-in
callers, API keys and keyless transports keep the full cap."""

import asyncio
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jwt
import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

_MIB = 1024 * 1024
_SECRET = "test-secret-for-the-preauth-body-cap-0123456789"


@pytest.fixture(scope = "module")
def main_module():
    import main as _main  # noqa: F401
    return _main


@pytest.fixture(autouse = True)
def _no_keyless_and_one_user(monkeypatch):
    import auth.storage as storage
    import utils.keyless_api_access as keyless

    monkeypatch.setattr(keyless, "keyless_request_allowed", lambda request: False)
    real = storage.get_user_and_secret

    def fake(subject):
        subject.encode("utf-8")  # as sqlite3 binding does: a lone surrogate raises
        if subject == "alice":
            return ("salt", "hash", _SECRET, False)
        return None

    monkeypatch.setattr(storage, "get_user_and_secret", fake)
    yield
    monkeypatch.setattr(storage, "get_user_and_secret", real)


def _wired_middleware(main_module, inner):
    """MaxBodyMiddleware with the exact arguments main.app registers it with."""
    for entry in main_module.app.user_middleware:
        if entry.cls is main_module.MaxBodyMiddleware:
            return main_module.MaxBodyMiddleware(inner, *entry.args, **entry.kwargs)
    raise AssertionError("MaxBodyMiddleware is not registered on main.app")


def _drive(
    main_module,
    path,
    headers,
    chunks,
    declared = True,
):
    seen = {"inner_bytes": None, "read_chunks": 0}
    sent = []

    async def inner(scope, receive, send):
        body = b""
        while True:
            msg = await receive()
            body += msg.get("body", b"")
            if not msg.get("more_body"):
                break
        seen["inner_bytes"] = len(body)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    queue = list(chunks)

    async def receive():
        if not queue:
            return {"type": "http.disconnect"}
        seen["read_chunks"] += 1
        data = queue.pop(0)
        return {"type": "http.request", "body": data, "more_body": bool(queue)}

    async def send(msg):
        sent.append(msg)

    raw_headers = [(k.lower().encode(), v.encode()) for k, v in headers.items()]
    if declared:
        raw_headers.append((b"content-length", str(sum(map(len, chunks))).encode()))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "root_path": "",
        "query_string": b"",
        "headers": raw_headers,
        "client": ("203.0.113.9", 5555),
        "server": ("127.0.0.1", 8888),
        "scheme": "http",
    }
    asyncio.run(_wired_middleware(main_module, inner)(scope, receive, send))
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    return status, seen


def _session_token(secret = _SECRET, subject = "alice"):
    exp = datetime.now(timezone.utc) + timedelta(minutes = 5)
    return jwt.encode({"sub": subject, "exp": exp}, secret, algorithm = "HS256")


_BIG = [b"x" * _MIB] * 4


@pytest.mark.parametrize("declared", [True, False])
def test_unauthenticated_large_body_is_refused_before_buffering(main_module, declared):
    status, seen = _drive(main_module, "/api/auth/login", {}, _BIG, declared = declared)
    assert status == 401
    assert seen["inner_bytes"] is None
    assert seen["read_chunks"] <= 2


def test_forged_session_token_does_not_unlock_the_full_cap(main_module):
    headers = {"Authorization": f"Bearer {_session_token(secret = 'not-the-secret')}"}
    status, seen = _drive(main_module, "/api/chat/threads", headers, _BIG, declared = False)
    assert status == 401
    assert seen["inner_bytes"] is None


def test_malformed_subject_does_not_fail_open(main_module):
    headers = {"Authorization": f"Bearer {_session_token(secret = 'x', subject = chr(0xD800))}"}
    status, seen = _drive(main_module, "/api/chat/threads", headers, _BIG, declared = False)
    assert status == 401
    assert seen["inner_bytes"] is None


def test_unauthenticated_small_body_still_reaches_the_route(main_module):
    status, seen = _drive(main_module, "/api/auth/login", {}, [b'{"username":"a"}'])
    assert status == 200
    assert seen["inner_bytes"] == len(b'{"username":"a"}')


@pytest.mark.parametrize("declared", [True, False])
def test_signed_in_large_body_keeps_the_full_cap(main_module, declared):
    headers = {"Authorization": f"Bearer {_session_token()}"}
    status, seen = _drive(main_module, "/api/chat/threads", headers, _BIG, declared = declared)
    assert status == 200
    assert seen["inner_bytes"] == 4 * _MIB


def test_signed_in_multipart_upload_passes_through(main_module):
    headers = {"Authorization": f"Bearer {_session_token()}"}
    status, seen = _drive(main_module, "/api/datasets/upload", headers, _BIG)
    assert status == 200
    assert seen["inner_bytes"] == 4 * _MIB


def test_unauthenticated_multipart_upload_is_refused_before_spooling(main_module):
    status, seen = _drive(main_module, "/api/datasets/upload", {}, _BIG)
    assert status == 401
    assert seen["inner_bytes"] is None


@pytest.mark.parametrize("path", ["/api/chat/threads", "/api/datasets/upload"])
def test_chunked_body_with_small_content_length_is_still_checked(main_module, path):
    # h11 frames by Transfer-Encoding but keeps the Content-Length header in the scope.
    headers = {"Content-Length": "1", "Transfer-Encoding": "chunked"}
    status, seen = _drive(main_module, path, headers, _BIG, declared = False)
    assert status in (401, 411)
    assert seen["inner_bytes"] is None
