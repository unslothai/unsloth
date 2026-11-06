# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The harness authenticates once per arm, so any run longer than the token's lifetime gets 401s."""

from __future__ import annotations

import base64
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from studiobench.runtime import lifecycle  # noqa: E402
from studiobench.runtime.lifecycle import (  # noqa: E402
    HttpError,
    StudioAuth,
    auth_request_json,
    authenticate,
    jwt_expiry,
    request_json,
    seed_init_script,
)
from studiobench.runtime.seeder import Seeder  # noqa: E402

# Compressed token lifetime, long enough that a loaded machine cannot expire one mid-request.
TOKEN_TTL_S = 6.0
# Keep the real margin-to-TTL ratio; a margin longer than the TTL would rotate every call.
TEST_MARGIN_S = 1.0
PASSWORD = "studiobench-bench-password"


def _b64(payload: dict) -> str:
    raw = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


class _State:
    def __init__(self) -> None:
        self.logins = 0
        self.login_attempts = 0
        self.rejections = 0
        self.reject_next = 0
        self.lock = threading.Lock()

    def mint(self) -> str:
        exp = time.time() + TOKEN_TTL_S
        return f"{_b64({'alg': 'HS256'})}.{_b64({'sub': 'bench', 'exp': exp})}.sig"


class _Handler(BaseHTTPRequestHandler):
    state: _State

    def log_message(self, *_args) -> None:  # noqa: D102
        pass

    def _send(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _authorised(self) -> bool:
        with self.state.lock:
            if self.state.reject_next > 0:
                self.state.reject_next -= 1
                self.state.rejections += 1
                return False
        header = self.headers.get("Authorization") or ""
        token = header.split(" ", 1)[-1] if header.startswith("Bearer ") else ""
        exp = jwt_expiry(token)
        if exp is None or exp <= time.time():
            with self.state.lock:
                self.state.rejections += 1
            return False
        return True

    def do_POST(self) -> None:  # noqa: N802
        body = self._body()
        if self.path == "/api/auth/login":
            with self.state.lock:
                self.state.login_attempts += 1
            if body.get("password") != PASSWORD:
                self._send(401, {"detail": "bad password"})
                return
            with self.state.lock:
                self.state.logins += 1
            self._send(
                200,
                {
                    "access_token": self.state.mint(),
                    "refresh_token": f"refresh-{self.state.logins}",
                    "must_change_password": False,
                },
            )
            return
        if not self._authorised():
            self._send(401, {"detail": "Not authenticated"})
            return
        self._send(200, {"ok": True, "path": self.path})

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/auth/status":
            self._send(200, {"requires_password_change": False})
            return
        if not self._authorised():
            self._send(401, {"detail": "Not authenticated"})
            return
        self._send(200, {"ok": True, "path": self.path})

    def do_PUT(self) -> None:  # noqa: N802
        self._body()
        if not self._authorised():
            self._send(401, {"detail": "Not authenticated"})
            return
        self._send(200, {"ok": True, "path": self.path})


@pytest.fixture()
def studio():
    state = _State()
    handler = type("_Bound", (_Handler,), {"state": state})
    server = HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", state
    finally:
        server.shutdown()
        server.server_close()


def test_the_token_a_run_was_handed_stops_working(studio):
    """Control: the server rejects an expired token, so a single held token fails once it lapses."""
    base_url, _state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    frozen = auth.access_token

    assert request_json(f"{base_url}/api/chat/threads", method = "POST", token = frozen, body = {})
    time.sleep(TOKEN_TTL_S + 0.5)
    with pytest.raises(HttpError) as caught:
        request_json(f"{base_url}/api/chat/threads", method = "POST", token = frozen, body = {})
    assert caught.value.status == 401


def test_the_seeder_keeps_working_after_its_token_expires(studio, monkeypatch):
    """Seeder.create_thread must keep working after the token expires, by rotating it proactively."""
    monkeypatch.setattr(lifecycle, "TOKEN_REFRESH_MARGIN_S", TEST_MARGIN_S)
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    seeder = Seeder(base_url = base_url, auth = auth, model_id = "m", log = lambda *_a: None)

    assert seeder.create_thread()
    assert auth.rotations == 0
    logins_after_setup = state.logins

    time.sleep(TOKEN_TTL_S + 0.5)
    assert seeder.create_thread()
    assert auth.rotations == 1
    assert state.logins == logins_after_setup + 1
    assert state.rejections == 0
    assert (auth.expires_at or 0) > time.time()


def test_the_token_is_replaced_before_it_expires_not_after_it_fails(studio, monkeypatch):
    """Tokens rotate before they expire: a 900 s seeding PUT cannot be cheaply retried after a 401."""
    monkeypatch.setattr(lifecycle, "TOKEN_REFRESH_MARGIN_S", TEST_MARGIN_S)
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    time.sleep(TOKEN_TTL_S - TEST_MARGIN_S / 2)

    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert auth.rotations == 1
    assert state.rejections == 0


def test_a_401_that_arrives_anyway_is_recovered(studio):
    """The reactive half. A token this process believes is fresh can still be refused: a clock
    offset against the server, or an Unsloth restarted underneath the run. One retry, then the
    refusal is real and is raised."""
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    auth.expires_at = time.time() + 10_000
    state.reject_next = 1

    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert auth.rotations == 1
    assert state.rejections == 1


def test_a_refusal_that_survives_a_fresh_login_is_raised(studio):
    """Not looped on. Two refusals in a row is a real 401 and the caller has to see it."""
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    auth.expires_at = time.time() + 10_000
    state.reject_next = 2

    with pytest.raises(HttpError) as caught:
        auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert caught.value.status == 401


def test_a_login_that_is_refused_is_not_retried_as_if_it_were_the_request(studio):
    """A refused login is not retried, or failed logins count double toward the five-a-minute lockout."""
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    auth.password = "not-the-password"
    auth.expires_at = time.time() - 1
    attempts_before = state.login_attempts

    with pytest.raises(HttpError) as caught:
        auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert caught.value.status == 401
    assert state.login_attempts == attempts_before + 1


def test_a_clock_that_makes_every_token_look_stale_stops_the_proactive_half(studio):
    """A clock that makes every token look stale must stop proactive refresh, not re-login per request."""
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    assert auth.proactive is True

    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert auth.rotations == 1
    assert auth.proactive is False

    logins = state.logins
    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert state.logins == logins
    assert auth.rotations == 1


def test_a_failing_rotation_hook_does_not_fail_the_request(studio):
    """`on_rotate` re-seeds a Playwright context, which can throw for reasons that have nothing to
    do with authentication -- a closed context, a page that crashed. The token has already been
    replaced by then and the request has to go out."""
    monkeypatch_error = RuntimeError("Target page, context or browser has been closed")
    base_url, _state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)

    def _boom(_auth):
        raise monkeypatch_error

    auth.on_rotate = _boom
    auth.expires_at = time.time() - 1

    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert auth.rotations == 1
    assert "Target page" in (auth.hook_error or "")


def test_the_margin_outlasts_the_longest_authenticated_request():
    """TOKEN_REFRESH_MARGIN_S must be at least the 900 s seeding PUT timeout, or a token lapses mid-PUT."""
    from studiobench.runtime.lifecycle import TOKEN_REFRESH_MARGIN_S

    seed_put_timeout_s = 900
    assert TOKEN_REFRESH_MARGIN_S >= seed_put_timeout_s


def test_rotating_notifies_whoever_seeded_the_browser(studio):
    """The page's localStorage is seeded from a SNAPSHOT of these values, and an init script
    re-runs on every navigation, so the owner of that context is told when they go stale."""
    base_url, _state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    seen: list[str] = []
    auth.on_rotate = lambda a: seen.append(a.access_token)

    auth.rotate()
    assert seen == [auth.access_token]


def test_an_opaque_token_falls_back_to_the_documented_lifetime():
    """A token whose `exp` cannot be read is assumed to live `ACCESS_TOKEN_TTL_S`, not forever."""
    from studiobench.runtime.lifecycle import ACCESS_TOKEN_TTL_S

    auth = StudioAuth(
        access_token = "not-a-jwt",
        refresh_token = "",
        base_url = "http://127.0.0.1:1",
        username = "bench",
        password = PASSWORD,
    )
    assert auth.seconds_left() == pytest.approx(ACCESS_TOKEN_TTL_S, abs = 5)


def test_the_page_is_seeded_with_the_refresh_key_the_app_actually_reads():
    """Seed the refresh token under the key the SPA reads (AUTH_REFRESH_TOKEN_KEY in session.ts)."""
    session_ts = (
        Path(__file__).resolve().parents[5] / "studio/frontend/src/features/auth/session.ts"
    )
    if not session_ts.exists():
        pytest.skip("the frontend source is not in this tree")
    key = ""
    for line in session_ts.read_text(encoding = "utf-8").splitlines():
        if "AUTH_REFRESH_TOKEN_KEY" in line and "=" in line:
            key = line.split('"')[1]
            break
    assert key, "AUTH_REFRESH_TOKEN_KEY was not found in session.ts"

    auth = StudioAuth(
        access_token = "access-token",
        refresh_token = "refresh-token",
        base_url = "http://127.0.0.1:1",
        username = "bench",
        password = PASSWORD,
    )
    script = seed_init_script(auth, [])
    assert f'"{key}": "refresh-token"' in script or f'"{key}":"refresh-token"' in script


def _seed_script_for(exp: float, label: str) -> str:
    """A seed script carrying a JWT that expires at `exp`."""
    token = f"{_b64({'alg': 'HS256'})}.{_b64({'sub': 'bench', 'exp': exp})}.{label}"
    auth = StudioAuth(
        access_token = token,
        refresh_token = f"refresh-{label}",
        base_url = "http://127.0.0.1:1",
        username = "bench",
        password = PASSWORD,
    )
    return seed_init_script(auth, [])


def _run_in_node(scripts: list) -> dict:
    """Run init scripts against a localStorage shim and report what is in storage afterwards."""
    import json as _json
    import shutil
    import subprocess

    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = (
        "const store = new Map();\n"
        "globalThis.window = { localStorage: {\n"
        "  getItem: (k) => (store.has(k) ? store.get(k) : null),\n"
        "  setItem: (k, v) => store.set(k, String(v)),\n"
        "}, atob: (s) => Buffer.from(s, 'base64').toString('binary') };\n"
        + "\n".join(scripts)
        + "\nconsole.log(JSON.stringify(Object.fromEntries(store)));\n"
    )
    out = subprocess.run(
        ["node", "-e", harness], capture_output = True, text = True, timeout = 60, check = True
    )
    return _json.loads(out.stdout.strip().splitlines()[-1])


def test_the_freshest_seed_script_wins_whatever_order_they_run_in():
    """Init scripts run in no defined order on every navigation, so the fresher seed must win regardless."""
    now = time.time()
    stale = _seed_script_for(now - 60, "stale")
    fresh = _seed_script_for(now + 3600, "fresh")

    for order, name in ((f"{stale}\n{fresh}", "stale first"), (f"{fresh}\n{stale}", "fresh first")):
        storage = _run_in_node([order])
        assert storage["unsloth_auth_token"].endswith(".fresh"), name
        assert storage["unsloth_auth_refresh_token"] == "refresh-fresh", name


def test_a_seed_script_still_seeds_an_empty_page():
    """The control on the guard: with nothing in storage, the seed is written as it always was."""
    storage = _run_in_node([_seed_script_for(time.time() + 3600, "first")])
    assert storage["unsloth_auth_token"].endswith(".first")
    assert storage["unsloth_auth_refresh_token"] == "refresh-first"
    assert storage["unsloth_chat_connections_enabled"] == "true"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
