# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Tests for the per-(ip, username) login rate limiter.

Covers:
  - bucket key is (client-ip, username.lower())
  - X-Forwarded-For honoured only when UNSLOTH_STUDIO_TRUST_FORWARDED is set
  - 429 detail body does NOT leak the client IP
  - One username failing doesn't lock out a different user from the same IP
  - One IP failing doesn't lock out the same user from a different IP
"""

import os
import sys
from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))


@pytest.fixture(autouse = True)
def _reset_buckets():
    """Clear the in-memory bucket dicts between tests."""
    from routes import auth as auth_routes

    auth_routes._LOGIN_BUCKETS.clear()
    auth_routes._LOGIN_IP_BUCKETS.clear()
    for _shard in auth_routes._LOGIN_IP_OVERFLOW:
        _shard.clear()
    auth_routes._LAST_IP_PRUNE = 0.0
    yield
    auth_routes._LOGIN_BUCKETS.clear()
    auth_routes._LOGIN_IP_BUCKETS.clear()
    for _shard in auth_routes._LOGIN_IP_OVERFLOW:
        _shard.clear()
    auth_routes._LAST_IP_PRUNE = 0.0


@pytest.fixture
def env_no_proxy(monkeypatch):
    monkeypatch.delenv("UNSLOTH_STUDIO_TRUST_FORWARDED", raising = False)


@pytest.fixture
def env_trust_proxy(monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_TRUST_FORWARDED", "1")


class _FakeRequest:
    def __init__(
        self,
        client_host = "127.0.0.1",
        headers = None,
    ):
        from starlette.datastructures import Headers
        self.client = type("Client", (), {"host": client_host})()
        self.headers = Headers(headers or {})


# ---------- _client_ip ----------


class TestClientIp:
    def test_uses_request_client_host_by_default(self, env_no_proxy):
        from routes.auth import _client_ip
        assert _client_ip(_FakeRequest("203.0.113.5")) == "203.0.113.5"

    def test_ignores_xff_when_trust_off(self, env_no_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"x-forwarded-for": "198.51.100.7, 10.0.0.1"},
        )
        # Proxy header is spoofable; without the opt-in, trust the direct connection.
        assert _client_ip(req) == "127.0.0.1"

    def test_honours_first_xff_when_trust_on(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"x-forwarded-for": "198.51.100.7, 10.0.0.1"},
        )
        assert _client_ip(req) == "198.51.100.7"

    def test_falls_back_to_client_host_when_xff_missing(self, env_trust_proxy):
        from routes.auth import _client_ip
        assert _client_ip(_FakeRequest("203.0.113.9")) == "203.0.113.9"

    def test_honours_forwarded_header_when_trust_on(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest(
            "127.0.0.1",
            {"forwarded": 'for="198.51.100.42";proto=https'},
        )
        assert _client_ip(req) == "198.51.100.42"

    def test_unknown_when_no_client(self, env_no_proxy):
        from routes.auth import _client_ip

        req = _FakeRequest()
        req.client = None
        assert _client_ip(req) == "_unknown"

    def test_xff_strips_ipv4_port(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest("127.0.0.1", {"x-forwarded-for": "198.51.100.7:50001, 10.0.0.1"})
        assert _client_ip(req) == "198.51.100.7"

    def test_xff_strips_bracketed_ipv6_port(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest("127.0.0.1", {"x-forwarded-for": "[2001:db8::1]:50001, 10.0.0.1"})
        assert _client_ip(req) == "2001:db8::1"

    def test_forwarded_strips_ipv4_port(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest("127.0.0.1", {"forwarded": 'for="198.51.100.7:50001";proto=https'})
        assert _client_ip(req) == "198.51.100.7"

    def test_forwarded_strips_bracketed_ipv6_port(self, env_trust_proxy):
        from routes.auth import _client_ip
        req = _FakeRequest("127.0.0.1", {"forwarded": 'for="[2001:db8::1]:50001";proto=https'})
        assert _client_ip(req) == "2001:db8::1"

    def test_forwarded_isolates_first_element(self, env_trust_proxy):
        from routes.auth import _client_ip

        # Pick the first Forwarded element only, else suffix variations create
        # attacker-controlled buckets.
        req = _FakeRequest(
            "127.0.0.1",
            {"forwarded": "for=198.51.100.42, for=10.0.0.1;proto=https"},
        )
        assert _client_ip(req) == "198.51.100.42"

    def test_xff_invalid_ip_falls_back_to_client_host(self, env_trust_proxy):
        from routes.auth import _client_ip

        # A garbage XFF must not propagate into the bucket key.
        req = _FakeRequest("127.0.0.1", {"x-forwarded-for": "not-an-ip"})
        assert _client_ip(req) == "127.0.0.1"


# ---------- bucket compose / blocking ----------


class TestBucketKeyAndBlocking:
    def test_record_per_user_isolates_other_users(self, env_no_proxy):
        from routes.auth import (
            _bucket_key,
            _record_login_failure,
            _login_blocked,
            _LOGIN_MAX_FAILS,
        )

        req = _FakeRequest("203.0.113.1")
        for _ in range(_LOGIN_MAX_FAILS):
            _record_login_failure(_bucket_key(req, "alice"))
        assert _login_blocked(_bucket_key(req, "alice")) > 0
        # bob's account from the same IP is unaffected by alice's typos
        assert _login_blocked(_bucket_key(req, "bob")) == 0

    def test_record_per_ip_isolates_other_ips(self, env_no_proxy):
        from routes.auth import (
            _bucket_key,
            _record_login_failure,
            _login_blocked,
            _LOGIN_MAX_FAILS,
        )

        req_a = _FakeRequest("203.0.113.1")
        req_b = _FakeRequest("203.0.113.2")
        for _ in range(_LOGIN_MAX_FAILS):
            _record_login_failure(_bucket_key(req_a, "alice"))
        assert _login_blocked(_bucket_key(req_a, "alice")) > 0
        # Same username, different IP, not blocked.
        assert _login_blocked(_bucket_key(req_b, "alice")) == 0

    def test_username_lowercased_in_key(self, env_no_proxy):
        from routes.auth import _bucket_key

        req = _FakeRequest("203.0.113.1")
        assert _bucket_key(req, "Alice") == _bucket_key(req, "alice")
        assert _bucket_key(req, "ALICE") == _bucket_key(req, "alice")

    def test_rotating_usernames_hit_ip_aggregate_cap(self, env_no_proxy, monkeypatch):
        """Spraying nonexistent usernames from one IP must still be throttled."""
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        req = _FakeRequest("203.0.113.10")
        for idx in range(5):
            auth_routes._record_login_failure(auth_routes._unknown_user_key(req))
            # Per-(ip,username) alone wouldn't throttle distinct usernames; the IP aggregate must.
        assert auth_routes._login_blocked(auth_routes._unknown_user_key(req)) > 0

    def test_unknown_user_bucket_is_single_sentinel(self, env_no_proxy):
        """Random unknown usernames from one IP collapse to one bucket."""
        from routes import auth as auth_routes

        req = _FakeRequest("203.0.113.11")
        unknown_key = auth_routes._unknown_user_key(req)
        for _ in range(20):
            auth_routes._record_login_failure(unknown_key)
        # Exactly one sentinel bucket for this IP regardless of usernames sprayed.
        ip_keys = [k for k in auth_routes._LOGIN_BUCKETS if k[0] == "203.0.113.11"]
        assert len(ip_keys) == 1
        assert ip_keys[0][1].startswith("\x00")

    def test_account_bucket_cap_bounded(self, env_no_proxy, monkeypatch):
        """The per-account bucket dict cannot grow without bound."""
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        req = _FakeRequest("203.0.113.12")
        for idx in range(50):
            auth_routes._record_login_failure((req.client.host, f"user-{idx}"))
        # Hard cap respected; further keys don't allocate.
        assert len(auth_routes._LOGIN_BUCKETS) <= 10

    def test_ip_bucket_cap_bounds_without_disabling_throttling(self, env_no_proxy, monkeypatch):
        """The per-IP dict is bounded, but saturating it must NOT disable
        throttling: a new IP that keeps failing after the cap is hit is still
        blocked (now via the shared overflow counter)."""
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        # Saturate the per-IP dict with distinct source IPs.
        for idx in range(50):
            auth_routes._record_login_failure((f"198.51.100.{idx}", "admin"))
        assert len(auth_routes._LOGIN_IP_BUCKETS) <= 10  # bounded

        # A brand-new IP arriving after saturation is still throttled: it can't get
        # its own bucket, so its failures land in the shared overflow counter.
        victim = ("203.0.113.99", "admin")
        for _ in range(5):
            auth_routes._record_login_failure(victim)
        assert auth_routes._login_blocked(victim) > 0

    def test_saturating_spray_cannot_reset_a_hot_ip_bucket(self, env_no_proxy, monkeypatch):
        """An IP flooding the dict must not evict (and reset) its own hot bucket.

        With FIFO eviction the oldest-inserted bucket -- the attacker's own, now
        blocked -- was popped once enough fresh IPs arrived, letting the attacker
        retry as first-seen. The overflow counter must keep it throttled.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        # Neutralize account-bucket blocking so this isolates the per-IP path.
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)

        attacker = ("203.0.113.7", "admin")
        for _ in range(5):
            auth_routes._record_login_failure(attacker)
        assert auth_routes._login_blocked(attacker) > 0  # attacker is throttled

        # Attacker sprays many distinct IPs to try to push its own bucket out.
        for idx in range(100):
            auth_routes._record_login_failure((f"198.51.100.{idx}", "admin"))

        # Still throttled: its hot bucket survived rather than being evicted.
        assert auth_routes._login_blocked(attacker) > 0

    def test_overflow_is_sharded_so_a_hot_ip_does_not_block_unrelated_ips(
        self, env_no_proxy, monkeypatch
    ):
        """A saturating spray must not globally deny login: a hot overflow shard
        throttles only the IPs that hash to it, not every new unbucketed client.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        # Neutralize account-bucket blocking so this isolates the per-IP path.
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)

        # Saturate the bucket dict so further new IPs fall through to overflow.
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))

        # Drive one IP's real overflow shard hot.
        attacker_ip = "198.51.100.7"
        for _ in range(5):
            auth_routes._record_login_failure((attacker_ip, "admin"))
        assert auth_routes._login_blocked((attacker_ip, "admin")) > 0

        # A new IP in a *different* shard must not be denied (a single global
        # counter would block it; a sharded one preserves per-source isolation).
        attacker_shard = auth_routes._overflow_shard(attacker_ip)
        victim_ip = next(
            f"203.0.113.{i}"
            for i in range(256)
            if auth_routes._overflow_shard(f"203.0.113.{i}") is not attacker_shard
        )
        assert auth_routes._login_blocked((victim_ip, "admin")) == 0

    def test_overflow_throttle_survives_capacity_freeing(self, env_no_proxy, monkeypatch):
        """A source throttled via overflow must stay throttled even if a bucket
        frees up before the window expires; otherwise a fresh bucket resets it.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        # Neutralize account-bucket blocking so this isolates the per-IP path.
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)

        # Saturate the dict, then drive a source's overflow shard hot.
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        attacker = ("198.51.100.7", "admin")
        for _ in range(5):
            auth_routes._record_login_failure(attacker)
        assert auth_routes._login_blocked(attacker) > 0

        # A successful login from another IP frees a bucket slot.
        auth_routes._clear_login_bucket(("10.0.0.0", "admin"))
        assert len(auth_routes._LOGIN_IP_BUCKETS) < auth_routes._LOGIN_MAX_BUCKETS

        # Still throttled (overflow shard still hot), and a new failure that now
        # gets a fresh per-IP bucket must not reset the throttle.
        assert auth_routes._login_blocked(attacker) > 0
        auth_routes._record_login_failure(attacker)
        assert auth_routes._login_blocked(attacker) > 0

    def test_overflow_shard_is_memory_bounded_under_cardinality_spray(
        self, env_no_proxy, monkeypatch
    ):
        """A high-cardinality spray must not grow overflow memory without bound:
        each shard tracks at most _LOGIN_IP_OVERFLOW_MAX distinct IPs.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_OVERFLOW_MAX", 8)

        # Saturate the dict, then spray thousands of distinct one-off IPs.
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        for idx in range(5000):
            auth_routes._record_login_failure((f"198.51.{idx // 256}.{idx % 256}", "admin"))

        assert all(len(shard) <= 8 for shard in auth_routes._LOGIN_IP_OVERFLOW)

    def test_overflow_eviction_does_not_inherit_count_onto_new_ip(self, env_no_proxy, monkeypatch):
        """Evicting a hot entry to make room must not hand its failure count to the
        new source; one attempt from an unrelated IP must not 429 it.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_OVERFLOW_MAX", 2)
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)
        # Force every overflow IP into one shard so we can saturate it.
        shard0 = auth_routes._LOGIN_IP_OVERFLOW[0]
        monkeypatch.setattr(auth_routes, "_overflow_shard", lambda _ip: shard0)

        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        # Fill the shard (cap 2) with two hot IPs at/over the threshold.
        for _ in range(5):
            auth_routes._record_login_failure(("198.51.100.1", "admin"))
        for _ in range(5):
            auth_routes._record_login_failure(("198.51.100.2", "admin"))
        assert len(shard0) == 2

        # A new IP evicts the lowest-count entry; it must start clean, so one
        # failure leaves it below the threshold and unblocked.
        new_ip = ("203.0.113.50", "admin")
        auth_routes._record_login_failure(new_ip)
        assert auth_routes._login_blocked(new_ip) == 0

    def test_overflow_count_migrates_into_new_bucket(self, env_no_proxy, monkeypatch):
        """Straddling the overflow -> bucket transition must not double the per-IP
        limit: the overflow count carries into the freshly created bucket.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)

        # Saturate, then push one IP to 4 overflow failures (one below threshold).
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        attacker = ("198.51.100.7", "admin")
        for _ in range(4):
            auth_routes._record_login_failure(attacker)
        assert auth_routes._login_blocked(attacker) == 0  # 4 < 5

        # Free a slot so the next failure lands in a fresh per-IP bucket.
        auth_routes._clear_login_bucket(("10.0.0.0", "admin"))
        # One more failure must throttle (4 carried + 1 = 5), not reset to 1.
        auth_routes._record_login_failure(attacker)
        assert auth_routes._login_blocked(attacker) > 0

    def test_overflow_migration_is_bounded_not_one_entry_per_failure(
        self, env_no_proxy, monkeypatch
    ):
        """A saturated IP can rack up many overflow failures; migrating them into a
        fresh bucket must allocate at most the per-IP threshold worth of entries,
        not one deque entry per recorded failure (which would let a single later
        attempt allocate an arbitrarily large deque under the login lock).
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100000)

        # Saturate the dict, then hammer one IP far past the threshold in overflow.
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        attacker_ip = "198.51.100.7"
        attacker = (attacker_ip, "admin")
        for _ in range(5000):
            auth_routes._record_login_failure(attacker)
        # The stored overflow count is clamped at the threshold, not 5000.
        entry = auth_routes._overflow_shard(attacker_ip).get(attacker_ip)
        assert entry is not None and entry[0] <= auth_routes._LOGIN_IP_MAX_FAILS

        # Free a slot so the next failure migrates the overflow count into a bucket.
        auth_routes._clear_login_bucket(("10.0.0.0", "admin"))
        auth_routes._record_login_failure(attacker)
        bucket = auth_routes._LOGIN_IP_BUCKETS[attacker_ip]
        # Bounded by the threshold (+1 for the triggering failure), not ~5000.
        assert len(bucket) <= auth_routes._LOGIN_IP_MAX_FAILS + 1
        # Still throttled -- bounding the migration must not weaken the limit.
        assert auth_routes._login_blocked(attacker) > 0

    def test_successful_login_clears_overflow_throttle(self, env_no_proxy, monkeypatch):
        """A successful login resets the IP's throttle, including overflow, so a
        single later typo is not immediately blocked.
        """
        from routes import auth as auth_routes

        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_BUCKETS", 10)
        monkeypatch.setattr(auth_routes, "_LOGIN_IP_MAX_FAILS", 5)
        monkeypatch.setattr(auth_routes, "_LOGIN_MAX_FAILS", 100)

        # Saturate the dict, then push one IP into overflow until it is throttled.
        for idx in range(10):
            auth_routes._record_login_failure((f"10.0.0.{idx}", "admin"))
        ip = ("198.51.100.7", "admin")
        for _ in range(5):
            auth_routes._record_login_failure(ip)
        assert auth_routes._login_blocked(ip) > 0

        # A successful login from that IP clears its overflow entries...
        auth_routes._clear_login_bucket(ip)
        assert auth_routes._login_blocked(ip) == 0
        # ...and a single subsequent failure does not immediately re-block it.
        auth_routes._record_login_failure(ip)
        assert auth_routes._login_blocked(ip) == 0


# ---------- /login 429 body ----------


class TestLogin429Body:
    @pytest.fixture
    def login_client(self, tmp_path, monkeypatch):
        from auth import storage
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from routes.auth import router as auth_router
        import secrets as _secrets

        monkeypatch.setattr(storage, "DB_PATH", tmp_path / "auth.db")
        monkeypatch.setattr(storage, "_BOOTSTRAP_PW_PATH", tmp_path / ".bootstrap_password")
        monkeypatch.setattr(storage, "_bootstrap_password", None)
        storage.create_initial_user(
            username = storage.DEFAULT_ADMIN_USERNAME,
            password = "human-password-123",
            jwt_secret = _secrets.token_urlsafe(64),
            must_change_password = False,
        )

        app = FastAPI()
        app.include_router(auth_router, prefix = "/api/auth")
        return TestClient(app)

    def test_429_detail_does_not_leak_ip(self, env_no_proxy, login_client):
        from routes.auth import _LOGIN_MAX_FAILS

        # Drive 6 failures from the same client IP / username
        for _ in range(_LOGIN_MAX_FAILS):
            r = login_client.post(
                "/api/auth/login",
                json = {"username": "unsloth", "password": "wrong"},
            )
            assert r.status_code == 401
        r = login_client.post(
            "/api/auth/login",
            json = {"username": "unsloth", "password": "wrong"},
        )
        assert r.status_code == 429
        detail = r.json()["detail"]
        # The 429 body must not interpolate the source IP
        assert "127.0.0.1" not in detail
        assert "Too many" in detail
        # Retry-After header is still set for clients
        assert "Retry-After" in r.headers
