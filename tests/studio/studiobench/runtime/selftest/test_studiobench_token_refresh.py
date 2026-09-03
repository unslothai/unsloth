# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The run is longer than the token it was handed.

`ACCESS_TOKEN_EXPIRE_MINUTES` is 60 and this harness authenticated ONCE per arm, at setup, before
the first install. A standard A/B at four repetitions is 24 cells of the 243 second standard film
-- 97 minutes of film alone -- so the token the seeder holds expires part way through and every
request after that answers 401. It presents as an intermittent failure and it is not one: it is
the clock, and it is reproducible to the second.

The fake Unsloth below issues tokens with a six second life instead of an hour's, and the tests drive
`token()` with a one second margin, so the ratio between the two is the harness's own (15 minutes
against 60) at a scale a test can wait for. `test_the_token_a_run_was_handed_stops_working` is the
control that shows the server really does stop accepting an expired token, so the tests underneath
are not passing for some other reason.
"""

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

#: How long the fake Unsloth's access tokens live. An hour compressed into something a test can sit
#: through, and long enough that a loaded machine cannot expire one mid-request.
TOKEN_TTL_S = 6.0
#: The margin the tests below drive `token()` with. The REAL ratio matters: a margin far shorter
#: than the token's life is what the harness runs with (15 minutes against 60), and a margin longer
#: than the whole life would make every call rotate and prove nothing.
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
        #: Set to reject EVERY token once, whatever its `exp` says: an Unsloth restarted underneath
        #: the run, or a clock this process cannot see.
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
    """THE CONTROL, and the defect itself: hold one token and it expires under you.

    This is what the harness did -- authenticate once, then send `auth.access_token` on every
    request for the rest of the run -- reproduced at six seconds instead of sixty minutes.
    """
    base_url, _state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    frozen = auth.access_token

    assert request_json(f"{base_url}/api/chat/threads", method = "POST", token = frozen, body = {})
    time.sleep(TOKEN_TTL_S + 0.5)
    with pytest.raises(HttpError) as caught:
        request_json(f"{base_url}/api/chat/threads", method = "POST", token = frozen, body = {})
    assert caught.value.status == 401


def test_the_seeder_keeps_working_after_its_token_expires(studio, monkeypatch):
    """The fix, at the call site the review named: `Seeder.create_thread` past the expiry.

    The margin is a sixth of the token's life here, so the first cell's thread is created on the
    token the run was handed -- no rotation -- and only the one after the expiry rotates. That is
    the shape of a real run: one login per hour, not one per request.
    """
    monkeypatch.setattr(lifecycle, "TOKEN_REFRESH_MARGIN_S", TEST_MARGIN_S)
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    seeder = Seeder(base_url = base_url, auth = auth, model_id = "m", log = lambda *_a: None)

    assert seeder.create_thread()
    assert auth.rotations == 0
    logins_after_setup = state.logins

    time.sleep(TOKEN_TTL_S + 0.5)
    # The token the run was handed is dead by now -- the control above proves the server refuses
    # it -- and this still has to work.
    assert seeder.create_thread()
    assert auth.rotations == 1
    assert state.logins == logins_after_setup + 1
    # And it never had to be told: the server was not asked to refuse anything.
    assert state.rejections == 0
    assert (auth.expires_at or 0) > time.time()


def test_the_token_is_replaced_before_it_expires_not_after_it_fails(studio, monkeypatch):
    """PROACTIVE, which is the half a 401 handler alone does not give you.

    A 900 second seeding PUT that is valid when it is written and expired when the server finishes
    reading it cannot be retried cheaply -- the whole thread goes up the wire again -- so the token
    is replaced while it still has margin left. Here the margin is the whole of its life, so the
    request never sees a 401 at all.
    """
    monkeypatch.setattr(lifecycle, "TOKEN_REFRESH_MARGIN_S", TEST_MARGIN_S)
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    # Inside the margin and NOT yet expired: the server would still accept the old token, and it
    # is replaced anyway.
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
    # Fresh by this process's reckoning, and refused by the server regardless.
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
    """ONE login attempt, not two.

    `token()` can itself raise a 401 -- the password is wrong, or the account is locked -- and
    catching that alongside the request's own 401 would answer it with a SECOND login. The backend
    locks an account after five failures in a minute (`routes/auth.py`, `_LOGIN_MAX_FAILS`), so
    burning the bucket at double rate reaches the lockout twice as fast and the run then dies on a
    429 that says nothing about the password.
    """
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
    """The runaway guard. `needs_refresh` reads the server's `exp` against THIS process's clock.

    An Unsloth 45 minutes behind, or one whose `ACCESS_TOKEN_EXPIRE_MINUTES` is shorter than the
    margin -- which is exactly this fake studio, six seconds against fifteen minutes -- makes every
    token ever issued look like it is about to expire, and every request would then log in again
    and append another init script to the browser context. One rotation is enough to find that out.
    """
    base_url, state = studio
    auth = authenticate(base_url, "bench", PASSWORD, new_password = PASSWORD)
    assert auth.proactive is True

    assert auth_request_json(auth, f"{base_url}/api/chat/threads", method = "POST", body = {})
    assert auth.rotations == 1
    assert auth.proactive is False

    # And it does not keep doing it. The token is valid, so the request goes out on it untouched.
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
    """The invariant the margin exists for, pinned rather than argued.

    Seeding a 1M-token thread is ONE `PUT` with a 900 second timeout, so a token that is merely
    valid when the request is written is not enough: it has to still be valid when the server
    finishes reading the body. Lower the margin under that and this fails.
    """
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
    """The other half of the same defect, and the one that produced the Playwright symptom.

    The SPA reads its refresh token from `AUTH_REFRESH_TOKEN_KEY`. Seeded under any other name the
    page has an access token and no way to renew it, so the first 401 after the hour is up sends
    `authFetch` down the branch that clears the tokens and navigates to the login route -- which
    Playwright reports as `Execution context was destroyed, most likely because of a navigation`.

    The key is read out of the frontend source rather than copied here, so this fails if the app
    renames it.
    """
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
    """An init script re-runs on EVERY navigation and Playwright does not define the order that
    several of them run in, so "the one added last wins" is not a property this can rely on.

    The stale script must not put the token the run started with back over the one the SPA -- or a
    later re-seed -- rotated to. Both orders, one answer.
    """
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
