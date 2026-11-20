# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import shutil
import socket
import sys
import threading
import time

import pytest

from core.inference import browser_agent, tools
from core.inference.network_gate import NetworkGate
from core.inference.web_access_policy import check_url_access
from state import network_requests
from utils import network_firewall as firewall
from utils.account_context import (
    OWNER,
    AccountContext,
    as_api_caller,
    bind_account,
    reset_account,
)


@pytest.fixture
def store(monkeypatch):
    import storage.studio_db as studio_db

    saved = {}
    monkeypatch.setattr(
        studio_db, "get_app_setting", lambda key, fallback = None: saved.get(key, fallback)
    )
    monkeypatch.setattr(studio_db, "upsert_app_settings", saved.update)
    monkeypatch.setattr(firewall, "_cache", None)
    yield saved
    firewall._cache = None


def test_nothing_is_blocked_until_the_owner_adds_something(store):
    assert firewall.blocked_domains() == ()
    assert check_url_access("https://example.cn/page", None)[0] is True


def test_a_blocked_site_covers_its_subdomains_and_nothing_else(store):
    firewall.set_network_firewall(blocked_sites = ["example.com"])
    assert check_url_access("https://example.com/a", None)[0] is False
    assert check_url_access("https://api.example.com/a", None)[0] is False
    assert check_url_access("https://notexample.com/a", None)[0] is True
    assert check_url_access("https://example.com.evil.org/a", None)[0] is True


def test_a_bare_ending_blocks_every_address_under_it(store):
    firewall.set_network_firewall(blocked_sites = [".cn", "co.uk"])
    for url in ("https://www.aliyun.cn/", "https://a.b.gov.cn/x", "https://www.bbc.co.uk/"):
        assert check_url_access(url, None)[0] is False, url
    for url in ("https://cn.example.com/", "https://example.com/cn", "https://example.uk/"):
        assert check_url_access(url, None)[0] is True, url


def test_a_block_wins_over_a_chat_that_allows_the_site(store):
    firewall.set_network_firewall(blocked_sites = ["example.com"])
    policy = {"allowedDomains": ["example.com"], "blockedDomains": []}
    assert check_url_access("https://example.com/", policy)[0] is False


def test_saved_sites_are_cleaned_and_bad_ones_are_refused(store):
    saved = firewall.set_network_firewall(
        blocked_sites = [
            " Example.COM ",
            "https://docs.example.org/path?q=1",
            "*.example.net",
            ".su",
            "example.com",
        ],
    )
    assert saved["blocked_sites"] == ["example.com", "docs.example.org", "example.net", "su"]
    assert firewall.get_network_firewall() == saved
    for change in (
        {"blocked_sites": ["not a site"]},
        {"allowed_sites": ["http://"]},
        {"blocked_sites": "example.com"},
        {"mode": "strict"},
        {"allowed_sites": [f"site{n}.com" for n in range(firewall.MAX_SITES + 1)]},
    ):
        with pytest.raises(ValueError):
            firewall.set_network_firewall(**change)
    assert firewall.get_network_firewall() == saved


def test_a_fresh_install_is_open_with_the_package_hosts_ready_to_allow(store):
    fresh = firewall.get_network_firewall()
    assert fresh["mode"] == "open"
    assert fresh["allowed_sites"] == list(firewall.DEFAULT_ALLOWED_SITES)
    assert firewall.confines_code() is False
    assert firewall.decide("anything.example") == "allow"


@pytest.mark.parametrize(
    "mode, host, verdict",
    [
        ("open", "example.org", "allow"),
        ("open", "blocked.example", "deny"),
        ("open", "shop.example.cn", "deny"),
        ("ask", "pypi.org", "allow"),
        ("ask", "files.pythonhosted.org", "allow"),
        ("ask", "cdn-lfs.huggingface.co", "allow"),
        ("ask", "example.org", "ask"),
        ("ask", "evil-pypi.org", "ask"),
        ("ask", "sub.blocked.example", "deny"),
        ("ask", "mirror.example.cn", "deny"),
        ("ask", "140.82.112.3", "ask"),
        ("allowlist", "PyPI.org.", "allow"),
        ("allowlist", "example.org", "deny"),
        ("allowlist", "140.82.112.3", "deny"),
        ("allowlist", "allowed.blocked.example", "deny"),
    ],
)
def test_one_connection_is_allowed_denied_or_asked_about(store, mode, host, verdict):
    firewall.set_network_firewall(
        mode = mode,
        allowed_sites = [*firewall.DEFAULT_ALLOWED_SITES, "allowed.blocked.example"],
        blocked_sites = ["blocked.example", "cn"],
    )
    assert firewall.decide(host) == verdict
    assert firewall.confines_code() is (mode != "open")


def test_adding_a_site_to_one_list_takes_it_off_the_other(store):
    firewall.set_network_firewall(blocked_sites = ["example.org"])
    saved = firewall.add_site("allowed_sites", "https://Example.org/page")
    assert "example.org" in saved["allowed_sites"] and saved["blocked_sites"] == []
    saved = firewall.add_site("blocked_sites", "example.org")
    assert "example.org" not in saved["allowed_sites"] and saved["blocked_sites"] == ["example.org"]


def test_a_save_applies_to_the_next_request(store):
    assert check_url_access("https://example.com/", None)[0] is True
    firewall.set_network_firewall(blocked_sites = ["example.com"])
    assert check_url_access("https://example.com/", None)[0] is False
    firewall.set_network_firewall(blocked_sites = [])
    assert check_url_access("https://example.com/", None)[0] is True


def test_an_unreadable_store_keeps_the_last_known_blocks(store, monkeypatch):
    firewall.set_network_firewall(blocked_sites = ["example.com"])
    assert firewall.blocked_domains() == ("example.com",)

    def broken():
        raise OSError("database is locked")

    monkeypatch.setattr(firewall, "get_network_firewall", broken)
    monkeypatch.setattr(firewall, "_cache", (0.0, firewall._cache[1]))
    assert firewall.blocked_domains() == ("example.com",)


def test_page_fetch_refuses_a_blocked_site_before_dns(store, monkeypatch):
    firewall.set_network_firewall(blocked_sites = ["cn"])
    resolved = []
    monkeypatch.setattr(
        tools,
        "_validate_and_resolve_host",
        lambda hostname, port: resolved.append(hostname) or (True, "", ["1.1.1.1"]),
    )
    result = tools._fetch_page_text("https://example.cn/article")
    assert "Blocked" in result and "example.cn" in result
    assert resolved == []


def test_reading_a_blocked_site_is_refused_without_asking_first(store):
    blocked = {"url": "https://example.cn/article"}
    assert tools.is_high_risk_tool_call("web_search", blocked) is True
    firewall.set_network_firewall(blocked_sites = ["cn"])
    assert tools.is_high_risk_tool_call("web_search", blocked) is False
    assert tools.is_potentially_unsafe_tool_call("web_search", blocked) is False
    assert tools.is_high_risk_tool_call("web_search", {"url": "example.cn/article"}) is False
    assert tools.is_high_risk_tool_call("web_search", {"url": "https://example.com/"}) is True


def test_web_search_drops_blocked_results_and_refills(store, monkeypatch):
    firewall.set_network_firewall(blocked_sites = ["blocked.example"])
    asked = []

    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            asked.append(max_results)
            return [
                {"title": "Bad", "href": "https://blocked.example/1", "body": "B"},
                {"title": "Good", "href": "https://fine.example/1", "body": "B"},
            ]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    result = tools._web_search("q", max_results = 1)
    assert "https://fine.example/1" in result
    assert "blocked.example" not in result
    assert asked[0] > 1


def test_the_browser_agent_is_refused_and_its_panel_gets_the_blocks(store):
    firewall.set_network_firewall(blocked_sites = ["example.com", "cn"])
    with pytest.raises(ValueError, match = "example.cn"):
        browser_agent.normalize_navigation_url("https://example.cn/", None)
    withheld = browser_agent.format_result(
        "browser_navigate", {"ok": True, "url": "https://www.example.com/", "text": "secret"}, None
    )
    assert "withheld" in withheld and "secret" not in withheld


class TestRoute:
    @pytest.fixture
    def make(self, store):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        import routes.settings as settings

        def _make(account, via_api_key = False):
            app = FastAPI()
            app.include_router(settings.router)

            async def subject():
                token = bind_account(account)
                try:
                    yield account.username
                finally:
                    reset_account(token)

            app.dependency_overrides[settings.get_current_subject] = subject
            app.dependency_overrides[settings.authenticated_via_api_key] = lambda: via_api_key
            return TestClient(app, raise_server_exceptions = False)

        return _make

    _SAVE = {
        "mode": "ask",
        "allowed_sites": ["PyPI.org"],
        "blocked_sites": ["Example.com"],
    }

    def test_owner_saves_and_reads(self, make, monkeypatch):
        from core.inference import os_sandbox

        monkeypatch.setattr(os_sandbox, "network_confinement_available", lambda: True)
        with make(OWNER) as client:
            first = client.get("/network-firewall").json()
            assert first["mode"] == "open" and first["blocked_sites"] == []
            assert first["allowed_sites"] == list(firewall.DEFAULT_ALLOWED_SITES)
            assert first["level"] == "locked"
            put = client.put("/network-firewall", json = self._SAVE)
            assert put.status_code == 200, put.text
            read = client.get("/network-firewall").json()
            added = client.post(
                "/network-firewall/site", json = {"kind": "blocked", "host": "pypi.org"}
            )
        assert (read["mode"], read["allowed_sites"]) == ("ask", ["pypi.org"])
        assert read["blocked_sites"] == ["example.com"]
        assert added.json()["allowed_sites"] == []
        assert added.json()["blocked_sites"] == ["example.com", "pypi.org"]

    def test_a_computer_that_cannot_lock_code_in_says_it_only_watches(self, make, monkeypatch):
        from core.inference import os_sandbox
        monkeypatch.setattr(os_sandbox, "network_confinement_available", lambda: False)
        with make(OWNER) as client:
            assert client.get("/network-firewall").json()["level"] == "watched"

    def test_recent_connections_are_listed_newest_first_per_website(self, make):
        from state import network_requests

        network_requests._log.clear()
        network_requests.record("python", "pypi.org", 443, "allowed", 100, 5000)
        network_requests.record("terminal", "example.org", 443, "refused")
        network_requests.record("python", "pypi.org", 443, "allowed", 50, 1000)
        with make(OWNER) as client:
            rows = client.get("/network-firewall/connections").json()["connections"]
        network_requests._log.clear()
        assert [(row["host"], row["verdict"], row["count"]) for row in rows] == [
            ("pypi.org", "allowed", 2),
            ("example.org", "refused", 1),
        ]
        assert (rows[0]["sent"], rows[0]["received"]) == (150, 6000)

    def test_a_bad_site_is_a_400_that_names_it(self, make, store):
        with make(OWNER) as client:
            put = client.put(
                "/network-firewall", json = {**self._SAVE, "blocked_sites": ["not a site"]}
            )
        assert put.status_code == 400
        assert "not a website" in put.json()["detail"]
        assert store == {}

    @pytest.mark.parametrize(
        "account, via_api_key",
        [(AccountContext("a" * 32, "alice"), False), (OWNER, True)],
        ids = ["managed-account", "owner-api-key"],
    )
    def test_only_the_owners_ui_session_may_read_or_write(self, make, store, account, via_api_key):
        with make(account, via_api_key) as client:
            statuses = [
                client.put("/network-firewall", json = self._SAVE).status_code,
                client.get("/network-firewall").status_code,
                client.get("/network-firewall/connections").status_code,
                client.post(
                    "/network-firewall/site", json = {"kind": "allowed", "host": "example.com"}
                ).status_code,
            ]
        assert statuses == [403, 403, 403, 403]
        assert store == {}


class _Chat:
    def __init__(self, answer):
        self.answer = answer
        self.events = []

    def __call__(self, event):
        if not isinstance(event, dict):
            return
        self.events.append(event)
        question = event["network_request"]
        if self.answer and not question.get("done"):
            threading.Thread(
                target = network_requests.resolve_request,
                args = (question["id"], self.answer, question["session_id"], OWNER.account_id),
                daemon = True,
            ).start()

    @property
    def questions(self):
        return [e["network_request"] for e in self.events if not e["network_request"].get("done")]


@pytest.fixture
def asking(store):
    firewall.set_network_firewall(mode = "ask", blocked_sites = ["cn"])
    network_requests._chats.clear()
    network_requests._log.clear()
    yield
    network_requests._chats.clear()
    network_requests._log.clear()


def _gate(
    chat,
    thread = "thread-1",
    cancel = None,
):
    return NetworkGate("python", "session-1", thread, chat, cancel)


def test_the_gate_lets_listed_websites_through_and_refuses_blocked_ones_without_asking(asking):
    chat = _Chat("allow")
    gate = _gate(chat)
    assert gate.check("pypi.org", 443) is None
    assert "blocked website" in gate.check("example.cn", 443)
    assert chat.events == []


def test_an_unknown_website_is_asked_about_once_per_chat(asking):
    chat = _Chat("allow")
    gate = _gate(chat)
    assert gate.check("Example.org.", 443) is None
    assert gate.check("example.org", 443) is None
    assert _gate(chat).check("example.org", 80) is None
    (question,) = chat.questions
    assert (question["host"], question["port"], question["tool"]) == ("example.org", 443, "python")
    assert question["can_always"] is True
    assert chat.events[-1]["network_request"] == {"id": question["id"], "done": True}
    other_chat = _Chat("deny")
    assert "not approved" in _gate(other_chat, thread = "thread-2").check("example.org", 443)
    assert len(other_chat.questions) == 1


def test_a_declined_website_is_not_asked_about_again_in_that_call_but_is_in_the_next(asking):
    chat = _Chat("deny")
    gate = _gate(chat)
    assert "not approved in the chat" in gate.check("example.org", 443)
    assert "not approved in the chat" in gate.check("example.org", 443)
    assert len(chat.questions) == 1
    assert "not approved in the chat" in _gate(chat).check("example.org", 443)
    assert len(chat.questions) == 2
    assert network_requests.recent_connections()[0]["verdict"] == "refused"


def test_a_call_with_no_chat_to_ask_in_is_refused(asking):
    assert "nobody is in the chat" in _gate(None).check("example.org", 443)
    token = bind_account(as_api_caller(OWNER))
    try:
        gate = _gate(_Chat("allow"))
    finally:
        reset_account(token)
    assert "nobody is in the chat" in gate.check("example.org", 443)


def test_a_question_ends_when_the_call_is_stopped(asking):
    chat = _Chat(None)
    cancel = threading.Event()
    gate = _gate(chat, cancel = cancel)
    threading.Timer(0.3, cancel.set).start()
    started = time.monotonic()
    assert "not approved" in gate.check("example.org", 443)
    assert time.monotonic() - started < 5
    assert chat.questions[0]["id"] not in network_requests._pending
    assert (
        network_requests.resolve_request(
            chat.questions[0]["id"], "allow", "session-1", OWNER.account_id
        )
        is False
    )


def test_another_account_cannot_answer_the_question(asking):
    request_id, slot = network_requests.begin_request("session-1", OWNER.account_id, "example.org")
    assert network_requests.resolve_request(request_id, "allow", "session-1", "b" * 32) is False
    assert (
        network_requests.resolve_request(request_id, "allow", "other-session", OWNER.account_id)
        is False
    )
    assert network_requests.pending_host(request_id, "session-1", OWNER.account_id) == "example.org"
    assert (
        network_requests.resolve_request(request_id, "allow", "session-1", OWNER.account_id) is True
    )
    assert (
        network_requests.resolve_request(request_id, "deny", "session-1", OWNER.account_id) is False
    )
    assert network_requests.wait_request(request_id, slot, lambda: False, 1) == "allow"


class TestDecisionRoute:
    @pytest.fixture
    def client(self, asking):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from routes import inference

        def _make(account, via_api_key = False):
            app = FastAPI()
            app.include_router(inference.studio_router, prefix = "/api/inference")

            async def subject():
                token = bind_account(as_api_caller(account) if via_api_key else account)
                try:
                    yield account.username
                finally:
                    reset_account(token)

            app.dependency_overrides[inference.get_current_subject] = subject
            return TestClient(app, raise_server_exceptions = False)

        return _make

    def _post(
        self,
        client,
        request_id,
        decision,
        session = "session-1",
    ):
        return client.post(
            "/api/inference/network-decision",
            json = {"request_id": request_id, "session_id": session, "decision": decision},
        )

    def test_always_allow_adds_the_asked_website_to_the_allowed_list(self, client):
        request_id, slot = network_requests.begin_request(
            "session-1", OWNER.account_id, "example.org"
        )
        with client(OWNER) as http:
            assert self._post(http, request_id, "always").status_code == 200
            assert self._post(http, request_id, "always").status_code == 404
        assert slot["decision"] == "always"
        assert "example.org" in firewall.get_network_firewall()["allowed_sites"]
        assert firewall.decide("example.org") == "allow"

    @pytest.mark.parametrize("via_api_key", [False, True], ids = ["managed-account", "owner-api-key"])
    def test_only_the_owners_ui_may_always_allow(self, client, via_api_key):
        account = OWNER if via_api_key else AccountContext("a" * 32, "alice")
        request_id, slot = network_requests.begin_request(
            "session-1", account.account_id, "example.org"
        )
        with client(account, via_api_key) as http:
            assert self._post(http, request_id, "always").status_code == 403
            assert slot["decision"] is None
            assert self._post(http, request_id, "allow").status_code == 200
        assert "example.org" not in firewall.get_network_firewall()["allowed_sites"]

    def test_an_unknown_or_foreign_question_is_a_404(self, client):
        request_id, slot = network_requests.begin_request(
            "session-1", OWNER.account_id, "example.org"
        )
        with client(AccountContext("a" * 32, "alice")) as http:
            assert self._post(http, request_id, "allow").status_code == 404
        with client(OWNER) as http:
            assert self._post(http, "nope", "allow").status_code == 404
            assert self._post(http, request_id, "allow", session = "other").status_code == 404
        assert slot["decision"] is None


def _connect(port, target):
    with socket.create_connection(("127.0.0.1", port), timeout = 10) as sock:
        sock.sendall(f"CONNECT {target} HTTP/1.1\r\nHost: {target}\r\n\r\n".encode())
        return sock.recv(4096).decode()


@pytest.mark.skipif(sys.platform == "win32", reason = "the egress proxy backs the POSIX sandboxes")
def test_the_proxy_asks_the_gate_before_it_dials_and_reports_what_moved(asking, monkeypatch):
    from core.inference import sandbox_egress

    upstream, far_end = socket.socketpair()
    dialled = []
    monkeypatch.setattr(
        sandbox_egress, "_dial", lambda host, port: dialled.append(host) or upstream
    )
    chat = _Chat("deny")
    proxy = sandbox_egress.EgressProxy(gate = _gate(chat))
    try:
        refused = _connect(proxy.port, "example.cn:443")
        assert refused.startswith("HTTP/1.1 403") and "blocked website" in refused
        declined = _connect(proxy.port, "example.org:443")
        assert declined.startswith("HTTP/1.1 403") and "not approved" in declined
        assert dialled == []
        with socket.create_connection(("127.0.0.1", proxy.port), timeout = 10) as sock:
            sock.sendall(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n")
            assert sock.recv(4096).startswith(b"HTTP/1.1 200")
            sock.sendall(b"hello")
            assert far_end.recv(16) == b"hello"
            far_end.sendall(b"a longer reply")
            assert sock.recv(64) == b"a longer reply"
        far_end.close()
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            row["verdict"] == "allowed" for row in network_requests.recent_connections()
        ):
            time.sleep(0.05)
    finally:
        proxy.close()
    assert dialled == ["pypi.org"]
    rows = {row["host"]: row for row in network_requests.recent_connections()}
    assert (rows["pypi.org"]["sent"], rows["pypi.org"]["received"]) == (5, 14)
    assert rows["example.cn"]["verdict"] == rows["example.org"]["verdict"] == "refused"


def test_closing_the_proxy_ends_a_waiting_question(asking):
    from core.inference import sandbox_egress

    if sys.platform == "win32":
        pytest.skip("the egress proxy backs the POSIX sandboxes")
    chat = _Chat(None)
    proxy = sandbox_egress.EgressProxy(gate = _gate(chat))
    answer = []
    asker = threading.Thread(target = lambda: answer.append(_connect(proxy.port, "example.org:443")))
    asker.start()
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and not chat.questions:
        time.sleep(0.05)
    assert chat.questions
    proxy.close()
    asker.join(timeout = 5)
    assert not asker.is_alive()
    question = chat.questions[0]["id"]
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and question in network_requests._pending:
        time.sleep(0.05)
    assert question not in network_requests._pending


_SESSION = "__LOCALID_network_firewall"
_WATCHED_FETCH = (
    "import os, socket, urllib.request\n"
    "print('PROXY', bool(os.environ.get('HTTPS_PROXY')))\n"
    "for target in (('93.184.216.34', 443), ('example.org', 443)):\n"
    "    try:\n"
    "        socket.create_connection(target, timeout = 5)\n"
    "        print('DIRECT', target[0])\n"
    "    except PermissionError as exc:\n"
    "        print('HOOK', exc)\n"
    "    except OSError as exc:\n"
    "        print('OSERROR', exc)\n"
    "try:\n"
    "    socket.getaddrinfo('secret-data.example.org', 443)\n"
    "    print('RESOLVED')\n"
    "except PermissionError:\n"
    "    print('NO LOOKUP')\n"
    "try:\n"
    "    urllib.request.urlopen('https://example.org/', timeout = 20)\n"
    "    print('REACHED')\n"
    "except Exception as exc:\n"
    "    print('REFUSED', exc)\n"
)


def test_code_outside_the_os_sandbox_is_pointed_at_the_checkpoint_and_kept_off_direct_routes(
    asking,
):
    chat = _Chat("deny")
    out = tools._python_exec(
        _WATCHED_FETCH, None, 60, _SESSION, False, chat, tool_execution_mode = "software"
    )
    assert "PROXY True" in out, out
    assert out.count("HOOK") == 2 and "DIRECT" not in out, out
    assert "NO LOOKUP" in out and "RESOLVED" not in out, out
    assert "REACHED" not in out and "REFUSED" in out, out
    assert [q["host"] for q in chat.questions] == ["example.org"]
    assert "[Unsloth network] Refused: the connection to example.org was not approved" in out, out


def test_open_mode_leaves_code_outside_the_os_sandbox_as_it_was(store):
    out = tools._python_exec(
        "import os\nprint('PROXY', bool(os.environ.get('HTTPS_PROXY')), os.environ.get('_UNSLOTH_EGRESS_ONLY'))",
        None,
        30,
        _SESSION,
        tool_execution_mode = "software",
    )
    assert out.strip() == "PROXY False None", out


def test_a_checkpoint_that_cannot_start_refuses_the_call(asking, monkeypatch):
    from core.inference import sandbox_egress

    def refuse(*args, **kwargs):
        raise OSError("no sockets left")

    monkeypatch.setattr(sandbox_egress, "EgressProxy", refuse)
    for run, source in ((tools._python_exec, "print(1)"), (tools._bash_exec, "echo 1")):
        out = run(source, None, 30, _SESSION, tool_execution_mode = "software")
        assert out == firewall.CHECKPOINT_FAILED, out


def test_a_chat_is_asked_before_it_sends_a_website_more_than_the_budget(asking, monkeypatch):
    monkeypatch.setattr(firewall, "UPLOAD_BUDGET_BYTES", 1000)
    chat = _Chat("allow")
    gate = _gate(chat)
    assert gate.sending("PyPI.org", 443, 600) is True
    assert chat.questions == []
    assert gate.sending("pypi.org", 443, 600) is True
    (question,) = chat.questions
    assert (question["kind"], question["host"], question["sent"]) == ("upload", "pypi.org", 1200)
    assert question["can_always"] is False
    assert _gate(chat).sending("pypi.org", 443, 5000) is True
    assert len(chat.questions) == 1
    assert _gate(chat, thread = "thread-2").sending("pypi.org", 443, 900) is True
    assert len(chat.questions) == 1


def test_a_declined_or_unanswerable_upload_is_cut_off(asking, monkeypatch):
    monkeypatch.setattr(firewall, "UPLOAD_BUDGET_BYTES", 1000)
    chat = _Chat("deny")
    gate = _gate(chat)
    assert gate.sending("pypi.org", 443, 2000) is False
    assert gate.sending("pypi.org", 443, 10) is False
    assert len(chat.questions) == 1
    assert "sending more than" in gate.note() and "pypi.org" in gate.note()
    assert _gate(None, thread = "thread-3").sending("pypi.org", 443, 2000) is False


@pytest.mark.skipif(sys.platform == "win32", reason = "socketpair-backed proxy test")
def test_the_proxy_stops_forwarding_once_an_upload_is_refused(asking, monkeypatch):
    from core.inference import sandbox_egress

    monkeypatch.setattr(firewall, "UPLOAD_BUDGET_BYTES", 1000)
    upstream, far_end = socket.socketpair()
    monkeypatch.setattr(sandbox_egress, "_dial", lambda host, port: upstream)
    chat = _Chat("deny")
    proxy = sandbox_egress.EgressProxy(gate = _gate(chat))
    try:
        with socket.create_connection(("127.0.0.1", proxy.port), timeout = 10) as sock:
            sock.sendall(b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n")
            assert sock.recv(4096).startswith(b"HTTP/1.1 200")
            sock.sendall(b"a" * 800)
            assert len(far_end.recv(4096)) == 800
            sock.sendall(b"b" * 800)
            far_end.settimeout(5)
            assert far_end.recv(4096) == b""
    finally:
        proxy.close()
        far_end.close()
    assert [q["kind"] for q in chat.questions] == ["upload"]


def test_opening_a_page_asks_through_the_same_card_and_lists(asking, monkeypatch):
    fetched = []
    monkeypatch.setattr(
        tools,
        "_validate_and_resolve_host",
        lambda hostname, port: fetched.append(hostname) or (False, "Blocked: test stops here.", []),
    )
    chat = _Chat("deny")
    out = tools.execute_tool(
        "web_search", {"url": "https://example.org/a"}, session_id = "session-1", output_callback = chat
    )
    assert "example.org was not approved in the chat" in out and fetched == []
    assert [(q["tool"], q["host"]) for q in chat.questions] == [("web", "example.org")]
    chat.answer = "allow"
    tools.execute_tool(
        "web_search", {"url": "https://example.org/b"}, session_id = "session-1", output_callback = chat
    )
    tools.execute_tool(
        "web_search", {"url": "https://example.org/c"}, session_id = "session-1", output_callback = chat
    )
    assert fetched == ["example.org", "example.org"] and len(chat.questions) == 2
    tools.execute_tool("web_search", {"url": "https://pypi.org/simple"}, output_callback = chat)
    assert fetched[-1] == "pypi.org" and len(chat.questions) == 2
    assert tools.is_high_risk_tool_call("web_search", {"url": "https://example.org/a"}) is False


def test_allowed_only_refuses_an_unlisted_page_and_a_search_still_runs(asking, monkeypatch):
    firewall.set_network_firewall(mode = "allowlist")
    chat = _Chat("allow")
    out = tools.execute_tool("web_search", {"url": "https://example.org/a"}, output_callback = chat)
    assert "example.org is not an allowed website" in out and chat.questions == []

    class FakeDDGS:
        def __init__(self, **_kwargs):
            pass

        def text(
            self,
            query,
            max_results = 5,
            **kwargs,
        ):
            return [{"title": "Fine", "href": "https://example.org/1", "body": "B"}]

    monkeypatch.setattr("ddgs.DDGS", FakeDDGS)
    found = tools.execute_tool("web_search", {"query": "q"}, output_callback = chat)
    assert "https://example.org/1" in found and chat.questions == []


def test_allowed_only_keeps_the_browser_agent_to_the_allowed_websites(asking):
    firewall.set_network_firewall(mode = "allowlist")
    with pytest.raises(ValueError, match = "not an allowed website"):
        browser_agent.normalize_navigation_url("https://example.org/", None)
    withheld = browser_agent.format_result(
        "browser_navigate", {"ok": True, "url": "https://example.org/", "text": "secret"}, None
    )
    assert "withheld" in withheld and "secret" not in withheld


def test_early_page_reads_are_not_started_while_the_rules_may_ask(asking):
    from core.inference import studio_tool_loop

    assert studio_tool_loop._web_fetch_host("web_search", {"url": "https://pypi.org/x"}) is None
    firewall.set_network_firewall(mode = "open")
    assert (
        studio_tool_loop._web_fetch_host("web_search", {"url": "https://pypi.org/x"}) == "pypi.org"
    )


def test_full_access_is_outside_the_firewall(asking):
    out = tools._python_exec(
        "import os\nprint(2, os.environ.get('_UNSLOTH_EGRESS_ONLY'))", None, 30, _SESSION, True
    )
    assert out.strip() == "2 None", out


def test_the_static_host_screen_steps_aside_when_the_proxy_decides(store):
    computed = "import requests\nurl = input()\nrequests.get(url)\n"
    unlisted = "import requests\nrequests.get('https://example.org/data')\n"
    assert tools._check_code_safety(unlisted) is not None
    assert tools._check_code_safety(computed) is not None
    assert tools._check_code_safety(unlisted, network_confined = True) is None
    assert tools._check_code_safety(computed, network_confined = True) is None
    metadata = "import requests\nrequests.get('http://169.254.169.254/latest')\n"
    assert tools._check_code_safety(metadata, network_confined = True) is not None


def test_network_code_is_not_asked_about_twice_under_the_rules(store):
    computed = {"code": "import requests\nurl = input()\nrequests.get(url)\n"}
    unlisted = {"code": "import requests\nrequests.get('https://example.org/data')\n"}
    removes = {
        "code": "import shutil, requests\nrequests.get('https://example.org')\nshutil.rmtree('/home')\n"
    }
    for call in (computed, unlisted, removes):
        assert tools.is_high_risk_tool_call("python", call) is True
    firewall.set_network_firewall(mode = "ask")
    assert tools.is_high_risk_tool_call("python", computed) is False
    assert tools.is_high_risk_tool_call("python", unlisted) is False
    assert tools.is_high_risk_tool_call("python", removes) is True


def test_open_mode_refuses_a_blocked_website_written_out_in_code_even_when_approved(store):
    firewall.set_network_firewall(blocked_sites = ["github.com", "cn"])
    for url in ("https://github.com/a/b", "https://example.cn/x"):
        code = f"import requests\nrequests.get('{url}')\n"
        error = tools._check_code_safety(code, host_approved = True)
        assert error is not None and "blocked in Settings" in error, (url, error)
    assert (
        tools._check_code_safety("import requests\nrequests.get('https://pypi.org/simple')\n")
        is None
    )


def _isolated() -> bool:
    if sys.platform != "linux":
        return False
    from core.inference import os_sandbox
    return os_sandbox.network_confinement_available()


needs_bwrap = pytest.mark.skipif(not _isolated(), reason = "needs bubblewrap with a private network")


@needs_bwrap
def test_sandboxed_python_and_terminal_reach_the_network_only_through_the_gate(asking):
    session = "__LOCALID_network_firewall"
    asked = _Chat("deny")
    direct = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout = 5)\n"
        "    print('DIRECT')\n"
        "except OSError as exc:\n"
        "    print('NO ROUTE', type(exc).__name__)\n"
    )
    assert "NO ROUTE" in tools._python_exec(direct, None, 60, session, False, asked)
    fetch = (
        "import urllib.request, urllib.error\n"
        "for url in ('https://example.cn/', 'https://example.org/'):\n"
        "    try:\n"
        "        urllib.request.urlopen(url, timeout = 20)\n"
        "        print('REACHED', url)\n"
        "    except Exception as exc:\n"
        "        print('REFUSED', url, exc)\n"
    )
    out = tools._python_exec(fetch, None, 60, session, False, asked)
    assert "REACHED" not in out and out.count("REFUSED") == 2, out
    assert "[Unsloth network] Refused: example.cn is a blocked website" in out, out
    assert "example.org was not approved in the chat" in out, out
    assert [q["host"] for q in asked.questions] == ["example.org"]
    if shutil.which("curl"):
        out = tools._bash_exec(
            "curl -sS -m 20 https://example.cn/; echo rc=$?", None, 60, session, False, asked
        )
        assert "blocked website" in out or "rc=56" in out, out


@needs_bwrap
def test_sandboxed_curl_reaches_an_allowed_website_through_the_proxy(asking, monkeypatch):
    import http.server

    from core.inference import sandbox_egress

    if shutil.which("curl") is None:
        pytest.skip("curl is not installed")

    class Hello(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "5")
            self.end_headers()
            self.wfile.write(b"hello")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Hello)
    threading.Thread(target = server.serve_forever, daemon = True).start()
    # The origin listens on loopback, which the proxy refuses; let only that through.
    real = sandbox_egress.refused_address
    monkeypatch.setattr(
        sandbox_egress, "refused_address", lambda address: address != "127.0.0.1" and real(address)
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    session = "__LOCALID_network_firewall"
    try:
        refused = tools._bash_exec(f"curl -sS --noproxy '' {url}", None, 60, session)
        assert "nobody is in the chat" in refused, refused
        firewall.set_network_firewall(allowed_sites = ["127.0.0.1"])
        assert tools._bash_exec(f"curl -sS --noproxy '' {url}", None, 60, session) == "hello"
    finally:
        server.shutdown()
        server.server_close()
    row = next(r for r in network_requests.recent_connections() if r["verdict"] == "allowed")
    assert row["host"] == "127.0.0.1" and row["received"] > 5 and row["sent"] > 0
