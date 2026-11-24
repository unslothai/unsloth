# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The whole-thread save decodes a large JSON body in the threadpool, with the same value and the same errors."""

import base64
import json
import os
import random
import sys
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from auth.authentication import get_current_subject
from routes import chat_history
from storage import studio_db
from utils.api_errors import install_api_error_handlers

FALLBACK = chat_history._SPLIT_JSON_FALLBACK
# Over the off-loop threshold, so every body below takes the threadpool path.
IMAGE = "data:image/png;base64," + base64.b64encode(bytes(range(256)) * 6000).decode()
PATH = "/api/chat/threads/t/messages?echoMessages=false"


def _same(a, b) -> bool:
    # repr tells 1 from 1.0 from True, -0.0 from 0.0 and keeps key order; == alone does not.
    return type(a) is type(b) and repr(a) == repr(b)


def _message(i, **overrides):
    message = {
        "id": f"m{i}",
        "threadId": "t",
        "parentId": f"m{i - 1}" if i else None,
        "role": "user",
        "content": [
            {"type": "text", "text": f"hi {i} \u00e9 \U0001f600"},
            {"type": "image", "image": IMAGE},
        ],
        "createdAt": 10 + i,
    }
    message.update(overrides)
    return message


GOOD = {"messages": [_message(0), _message(1)], "pruneMissing": False}
RAW = json.dumps(GOOD).encode()


def _deep(depth: int) -> bytes:
    nested = "[" * depth + "]" * depth
    return json.dumps(GOOD).replace('"pruneMissing": false', f'"x": {nested}').encode()


BODIES = {
    "valid": RAW,
    "valid_prune": json.dumps({**GOOD, "pruneMissing": True, "deletedMessageIds": []}).encode(),
    "wrong_type_messages": json.dumps({"messages": "x" + IMAGE}).encode(),
    "wrong_type_created_at": json.dumps({"messages": [_message(0, createdAt = "abc")]}).encode(),
    "wrong_type_role": json.dumps({"messages": [_message(0, role = 5)]}).encode(),
    "wrong_type_prune": json.dumps({"messages": [_message(0)], "pruneMissing": "maybe"}).encode(),
    "coerced_prune": json.dumps({"messages": [_message(0)], "pruneMissing": "true"}).encode(),
    "out_of_range_created_at": json.dumps({"messages": [_message(0, createdAt = 2**70)]}).encode(),
    "missing_messages": json.dumps({"pruneMissing": False, "pad": IMAGE}).encode(),
    "missing_id": json.dumps(
        {"messages": [{k: v for k, v in _message(0).items() if k != "id"}]}
    ).encode(),
    "extra_fields": json.dumps({"messages": [_message(0, extra = 1)], "unknown": IMAGE}).encode(),
    "thread_mismatch": json.dumps({"messages": [_message(0, threadId = "other")]}).encode(),
    "metadata_not_dict": json.dumps({"messages": [_message(0, metadata = [1])]}).encode(),
    "deleted_ids_ints": json.dumps({"messages": [_message(0)], "deletedMessageIds": [1]}).encode(),
    "truncated": RAW[:-7],
    "trailing_comma_array": RAW.replace(b'}], "pruneMissing"', b'},], "pruneMissing"'),
    "trailing_comma_object": RAW[:-1] + b",}",
    "extra_data": RAW + b" x",
    "second_object": RAW + b"{}",
    "unquoted_key": b"{messages: []" + b" " * 2_000_000 + b"}",
    "single_quoted_key": RAW.replace(b'"pruneMissing"', b"'pruneMissing'"),
    "control_character": RAW.replace(b"hi 0", b"hi\x010"),
    "bad_escape": RAW.replace(b"hi 0", b"hi\\q0"),
    "missing_colon": RAW.replace(b'"pruneMissing": false', b'"pruneMissing" false'),
    "missing_comma": RAW.replace(b'}, {"id": "m1"', b'} {"id": "m1"'),
    "whitespace_variety": json.dumps(GOOD, indent = "\t", separators = (" ,\r\n", " :\n ")).encode(),
    "utf8_bom": b"\xef\xbb\xbf" + RAW,
    "utf16": json.dumps(GOOD).encode("utf-16"),
    "utf32": json.dumps(GOOD).encode("utf-32"),
    "invalid_utf8": RAW.replace(b"hi 0", b"hi \xff 0"),
    "raw_non_ascii": json.dumps(GOOD, ensure_ascii = False).encode(),
    "lone_surrogate": RAW.replace(b"hi 0", b"hi \\ud800 0"),
    "surrogate_bytes": RAW.replace(b"hi 0", b"hi \xed\xa0\x80 0"),
    "non_finite": RAW.replace(
        b'"pruneMissing": false',
        b'"n": [NaN, Infinity, -Infinity, 1e400, -0.0, -0, 123456789012345678901]',
    ),
    "duplicate_top_key": RAW[:-1] + b', "messages": ' + json.dumps([_message(5)]).encode() + b"}",
    "top_level_array": json.dumps([_message(0)]).encode(),
    "top_level_null": b"null" + b" " * 2_000_000,
    "deep_150": _deep(150),
    "deep_250": _deep(250),
    "deep_200000": _deep(200_000),
    "empty": b"",
    "only_whitespace": b" " * 2_000_000,
}
CONTENT_TYPES = [
    "application/json",
    None,
    "text/plain",
    "application/merge-patch+json",
    "application/json; charset=utf-8",
    "application/x-www-form-urlencoded",
]

CASES = [(name, body, "application/json") for name, body in BODIES.items()] + [
    (name, BODIES[name], content_type)
    for name in ("valid", "truncated", "wrong_type_messages", "empty")
    for content_type in CONTENT_TYPES[1:]
]


@pytest.fixture
def chat_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_PROJECTS_HOME", str(tmp_path / "Projects"))
    monkeypatch.setattr(studio_db, "_schema_ready", set())


def _reset_thread(tmp_root, n: int) -> None:
    home = tmp_root / f"case{n}"
    os.environ["UNSLOTH_STUDIO_HOME"] = str(home)
    os.environ["UNSLOTH_STUDIO_PROJECTS_HOME"] = str(home / "Projects")
    studio_db._schema_ready = set()
    studio_db.upsert_chat_thread(
        {"id": "t", "title": "Chat", "modelType": "base", "modelId": "m", "createdAt": 1}
    )


def _old_app(subject: bool) -> FastAPI:
    app = FastAPI()
    install_api_error_handlers(app)
    app.put(
        "/api/chat/threads/{thread_id}/messages",
        response_model = chat_history.ChatMessageListResponse,
    )(chat_history.replace_thread_messages)
    if subject:
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def _new_app(subject: bool) -> FastAPI:
    app = FastAPI()
    install_api_error_handlers(app)
    app.include_router(chat_history.router, prefix = "/api/chat")
    if subject:
        app.dependency_overrides[get_current_subject] = lambda: "test-user"
    return app


def _exchange(client, body: bytes, content_type) -> tuple:
    headers = {} if content_type is None else {"content-type": content_type}
    r = client.put(PATH, content = body, headers = headers)
    rows = studio_db.list_chat_messages("t")
    return r.status_code, dict(r.headers), r.content, json.dumps(rows, sort_keys = True)


@pytest.mark.parametrize("subject", [True, False], ids = ["signed-in", "anonymous"])
def test_every_body_class_gets_the_same_response_and_rows(
    chat_home, tmp_path, subject, monkeypatch
):
    accepted = []
    real = chat_history._split_json_loads

    def counting(body):
        value = real(body)
        accepted.append(value is not FALLBACK)
        return value

    monkeypatch.setattr(chat_history, "_split_json_loads", counting)
    cases = iter(range(10_000))
    with (
        TestClient(_old_app(subject), raise_server_exceptions = False) as old,
        TestClient(_new_app(subject), raise_server_exceptions = False) as new,
    ):
        for name, body, content_type in CASES:
            _reset_thread(tmp_path, next(cases))
            before = _exchange(old, body, content_type)
            _reset_thread(tmp_path, next(cases))
            after = _exchange(new, body, content_type)
            assert after == before, (name, content_type, before[0], before[2][:300])
    assert any(accepted) and not all(accepted)


def test_a_large_body_is_decoded_off_the_event_loop(chat_home, tmp_path, monkeypatch):
    threads = []
    loads = []
    real_split = chat_history._split_json_loads
    real_loads = chat_history.json.loads

    def recording_split(body):
        threads.append(threading.current_thread() is threading.main_thread())
        return real_split(body)

    def recording_loads(value, *args, **kwargs):
        if len(value) >= len(RAW):
            loads.append(value)
        return real_loads(value, *args, **kwargs)

    monkeypatch.setattr(chat_history, "_split_json_loads", recording_split)
    _reset_thread(tmp_path, 0)
    with TestClient(_new_app(True)) as client:
        monkeypatch.setattr(chat_history.json, "loads", recording_loads)
        r = client.put(PATH, content = RAW, headers = {"content-type": "application/json"})
        monkeypatch.setattr(chat_history.json, "loads", real_loads)
    assert r.status_code == 200
    assert threads == [False]
    assert loads == []
    assert [m["content"][1]["image"] for m in studio_db.list_chat_messages("t")] == [IMAGE] * 2


def test_a_small_body_keeps_the_event_loop_decode(chat_home, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(chat_history, "_split_json_loads", lambda body: calls.append(body))
    _reset_thread(tmp_path, 0)
    small = json.dumps({"messages": [_message(0, content = "hi")]}).encode()
    with TestClient(_new_app(True)) as client:
        r = client.put(PATH, content = small, headers = {"content-type": "application/json"})
    assert r.status_code == 200 and calls == []


def _random_value(rng, depth = 0):
    kind = rng.randrange(9 if depth < 6 else 6)
    if kind == 0:
        return None
    if kind == 1:
        return rng.random() < 0.5
    if kind == 2:
        return rng.choice([0, -1, 7, 2**63, -(2**70), 10**30])
    if kind == 3:
        return rng.choice([0.0, -0.0, 0.1, 1e16, 1e-7, 1.5e300, float("nan"), float("inf")])
    if kind in (4, 5):
        alphabet = 'ab "\\/\t\n\x00\x7f\u2028\u00e9\u6f22\U0001f600\ud800'
        return "".join(rng.choice(alphabet) for _ in range(rng.randrange(12)))
    if kind in (6, 7):
        return [_random_value(rng, depth + 1) for _ in range(rng.randrange(5))]
    return {f"k{rng.randrange(4)}": _random_value(rng, depth + 1) for _ in range(rng.randrange(5))}


def _random_body(rng) -> bytes:
    obj = {f"k{rng.randrange(6)}": _random_value(rng) for _ in range(rng.randrange(6))}
    if rng.random() < 0.5:
        obj["messages"] = [_random_value(rng, 1) for _ in range(rng.randrange(6))]
    text = json.dumps(
        obj,
        ensure_ascii = rng.random() < 0.5,
        indent = rng.choice([None, 0, 2, "\t"]),
        separators = rng.choice([None, (",", ":"), (" , ", " : "), (",\r\n", ":\n")]),
    )
    if rng.random() < 0.3:
        # Duplicate a top-level key: the decoder keeps the first position and the last value.
        text = text[:-1].rstrip() + (", " if len(obj) else "") + '"k0": [1, {"a": 2}]}'
    text = rng.choice(["", " ", "\r\n\t"]) + text + rng.choice(["", " ", "\n\t \r"])
    encoding = rng.choice(["utf-8", "utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-32-be"])
    return text.encode(encoding, "surrogatepass")


def test_split_decode_matches_json_loads_on_random_bodies():
    rng = random.Random(1234)
    accepted = 0
    for _ in range(3000):
        body = _random_body(rng)
        expected = json.loads(body)
        got = chat_history._split_json_loads(body)
        assert got is not FALLBACK, body[:200]
        assert _same(got, expected), body[:200]
        accepted += 1
        for _ in range(4):
            raw = bytearray(body)
            for _ in range(rng.randrange(1, 4)):
                op, at = rng.randrange(3), rng.randrange(len(raw) + 1)
                if op == 0 and raw:
                    del raw[min(at, len(raw) - 1)]
                elif op == 1:
                    raw[at:at] = bytes([rng.choice(b'{}[],:" \\tn0-.eE\xff')])
                elif raw:
                    raw[min(at, len(raw) - 1)] = rng.choice(b'{}[],:" \\x1')
            mutated = bytes(raw)
            got = chat_history._split_json_loads(mutated)
            if got is FALLBACK:
                continue
            assert _same(got, json.loads(mutated)), mutated[:200]
    assert accepted == 3000


@pytest.mark.parametrize("name", sorted(BODIES))
def test_split_decode_accepts_only_what_json_loads_decodes_identically(name):
    body = BODIES[name]
    got = chat_history._split_json_loads(body)
    if got is FALLBACK:
        return
    assert _same(got, json.loads(body))


def test_split_decode_leaves_deep_and_non_object_bodies_to_json_loads():
    for name in (
        "deep_250",
        "deep_200000",
        "top_level_array",
        "top_level_null",
        "truncated",
        "empty",
    ):
        assert chat_history._split_json_loads(BODIES[name]) is FALLBACK, name
    assert chat_history._split_json_loads(BODIES["deep_150"]) is not FALLBACK


def test_openapi_schema_unchanged():
    old = _old_app(True).openapi()
    new = _new_app(True).openapi()
    path = "/api/chat/threads/{thread_id}/messages"
    assert new["paths"][path]["put"] == old["paths"][path]["put"]
    for name in ("ChatMessage", "ChatMessageSyncRequest", "ChatMessageListResponse"):
        assert new["components"]["schemas"][name] == old["components"]["schemas"][name]
