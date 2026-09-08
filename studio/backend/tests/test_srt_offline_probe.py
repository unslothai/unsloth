# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Offline admission controls use real local sockets and a labelled launch seam."""

import json
import os
import socket
import sys
from types import SimpleNamespace

import pytest

from core.inference import srt_probe


@pytest.mark.skipif(sys.platform != "linux", reason = "Linux namespace probe")
@pytest.mark.parametrize("dns_error", [socket.EAI_AGAIN, socket.EAI_FAIL, None])
@pytest.mark.parametrize("outcome", ["success", "refused", "timeout"])
def test_local_probe_needs_no_public_dns(monkeypatch, dns_error, outcome):
    from core.inference import tools

    real_resolver = socket.getaddrinfo
    public_lookups = []

    def resolver(host, *args, **kwargs):
        if host != "127.0.0.1":
            public_lookups.append(host)
            if dns_error is not None:
                raise socket.gaierror(dns_error, "controlled external DNS refusal")
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("192.0.2.1", 443))]
        return real_resolver(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    monkeypatch.setattr(tools, "_build_safe_env", lambda work: {"PATH": "/usr/bin:/bin"})
    requests = []
    verified = []
    released = []

    def request_for(argv, *args, **kwargs):
        requests.append(json.loads(argv[-1]))
        return {}

    def communicate(timeout):
        assert timeout == 35
        if outcome == "timeout":
            raise srt_probe.subprocess.TimeoutExpired("controlled", timeout)
        return (b"UNSLOTH_SRT_NATIVE_PROBE_OK" if outcome == "success" else b"refused"), None

    proc = SimpleNamespace(returncode = 0 if outcome == "success" else 1,
                           communicate = communicate, kill = lambda: None, wait = lambda **kw: None)
    monkeypatch.setattr(srt_probe.srt_adapter, "request_for", request_for)
    monkeypatch.setattr(srt_probe.srt_adapter, "spawn", lambda *args, **kwargs: proc)
    monkeypatch.setattr(srt_probe.srt_adapter, "verify_success", lambda p: verified.append(p))
    monkeypatch.setattr(srt_probe.srt_adapter, "release_control", lambda p: released.append(p))
    if outcome == "timeout":
        with pytest.raises(srt_probe.srt_adapter.SrtError, match = "timed out"):
            srt_probe._native_probe()
    else:
        available, reason = srt_probe._native_probe()
        assert available is (outcome == "success"), reason
    assert len(requests) == 1
    assert requests[0]["udp_port"] > 0
    assert not public_lookups
    assert bool(verified) is (outcome == "success")
    assert released == [proc]


def test_probe_failure_cache_expiry_and_forced_recovery(monkeypatch):
    monkeypatch.setattr(srt_probe, "_cache", {})
    monkeypatch.setattr(srt_probe.srt_adapter, "installation_identity", lambda: "offline-test")
    now = [100.0]
    monkeypatch.setattr(srt_probe.time, "monotonic", lambda: now[0])
    calls = []

    def native(**kwargs):
        calls.append(1)
        return len(calls) > 1, "controlled result"

    monkeypatch.setattr(srt_probe, "_native_probe", native)
    assert not srt_probe.probe()[0]
    now[0] += 59
    assert not srt_probe.probe()[0]
    assert len(calls) == 1
    now[0] += 2
    assert srt_probe.probe()[0]
    assert srt_probe.probe(force = True)[0]
    assert len(calls) == 3


@pytest.mark.skipif(
    sys.platform != "linux" or os.environ.get("UNSLOTH_SRT_NATIVE_TESTS") != "1",
    reason = "requires prepared native Linux SRT runtime",
)
def test_native_required_probe_with_external_dns_refused(monkeypatch):
    real_resolver = socket.getaddrinfo

    def resolver(host, *args, **kwargs):
        if host != "127.0.0.1":
            raise socket.gaierror(socket.EAI_FAIL, "controlled external DNS refusal")
        return real_resolver(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", resolver)
    available, reason = srt_probe.probe(force = True)
    assert available, reason
