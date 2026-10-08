# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vendored truststore keeps verification on when handshakes overlap (upstream issue #209).

macOS and Windows switch OpenSSL verification off on the shared context during a wrap and
verify against the OS store afterwards. The flag flipping is reproduced here on any platform,
so the race is exercised without a network or a real handshake.
"""

from __future__ import annotations

import base64
import contextlib
import importlib.util
import ssl
import sys
import threading
from pathlib import Path

import pytest

_PACKAGE = Path(__file__).resolve().parent.parent / "vendor" / "truststore"


def _load_vendored(name, monkeypatch):
    # Private name: an installed truststore must not stand in. Restore the stdlib class first,
    # since native TLS may already have injected truststore into ssl (macOS / Windows default).
    stdlib = next(c for c in ssl._SSLContext.__subclasses__() if c.__module__ == "ssl")
    monkeypatch.setattr(ssl, "SSLContext", stdlib)
    spec = importlib.util.spec_from_file_location(
        name, _PACKAGE / "__init__.py", submodule_search_locations = [str(_PACKAGE)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module, sys.modules[f"{name}._api"]


@pytest.fixture()
def truststore(monkeypatch):
    name = "_unsloth_vendored_truststore_under_test"
    module, api = _load_vendored(name, monkeypatch)

    @contextlib.contextmanager
    def flipping(ctx):
        check_hostname, verify_mode = ctx.check_hostname, ctx.verify_mode
        ctx.check_hostname = False
        api._original_super_SSLContext.verify_mode.__set__(ctx, ssl.CERT_NONE)
        try:
            yield
        finally:
            ctx.check_hostname = check_hostname
            api._original_super_SSLContext.verify_mode.__set__(ctx, verify_mode)

    monkeypatch.setattr(api, "_configure_context", flipping)
    # raising=False: an unpatched copy has no such flag, and must fail on behaviour instead.
    monkeypatch.setattr(api, "_HOLDS_POLICY", True, raising = False)
    yield module, api
    for key in [k for k in sys.modules if k.startswith(name)]:
        sys.modules.pop(key, None)


class _FakeSock:
    def __init__(self, context):
        self.context = context

    def get_unverified_chain(self):
        return []

    def close(self):
        pass


def _overlap(
    module,
    api,
    monkeypatch,
    during_b = None,
):
    """A enters, B enters while A is open, A verifies and leaves, then B leaves."""
    seen = {}
    a_in, b_in, a_verified = threading.Event(), threading.Event(), threading.Event()

    def impl(
        ssl_context,
        cert_chain,
        server_hostname = None,
    ):
        seen[threading.current_thread().name] = (
            ssl_context.verify_mode,
            ssl_context.check_hostname,
        )
        if threading.current_thread().name == "A":
            a_verified.set()

    monkeypatch.setattr(api, "_verify_peercerts_impl", impl)
    ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def fake_wrap(sock, **kwargs):
        if threading.current_thread().name == "A":
            a_in.set()
            assert b_in.wait(5)
        else:
            assert a_in.wait(5)
            b_in.set()
            if during_b is not None:
                during_b(ctx)
            assert a_verified.wait(5)
        return _FakeSock(ctx._ctx)

    ctx._ctx.wrap_socket = fake_wrap
    threads = [
        threading.Thread(
            target = ctx.wrap_socket, args = (None,), kwargs = {"server_hostname": "h"}, name = n
        )
        for n in "AB"
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    return ctx, seen


def test_overlapping_wraps_verify_and_leave_the_context_verified(truststore, monkeypatch):
    module, api = truststore
    ctx, seen = _overlap(module, api, monkeypatch)
    assert seen == {"A": (ssl.CERT_REQUIRED, True), "B": (ssl.CERT_REQUIRED, True)}
    assert (ctx.verify_mode, ctx.check_hostname) == (ssl.CERT_REQUIRED, True)
    assert (ctx._ctx.verify_mode, ctx._ctx.check_hostname) == (ssl.CERT_REQUIRED, True)


def test_a_snapshot_restore_during_a_wrap_cannot_pin_cert_none(truststore, monkeypatch):
    # urllib3 reads verify_mode before load_verify_locations and writes it back afterwards.
    module, api = truststore

    def snapshot_restore(ctx):
        mode = ctx.verify_mode
        ctx.verify_mode = mode

    ctx, seen = _overlap(module, api, monkeypatch, during_b = snapshot_restore)
    assert seen["B"] == (ssl.CERT_REQUIRED, True)
    assert ctx._ctx.verify_mode == ssl.CERT_REQUIRED


def test_a_policy_set_during_a_wrap_applies_when_it_closes(truststore, monkeypatch):
    module, api = truststore

    def relax(ctx):
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

    ctx, _ = _overlap(module, api, monkeypatch, during_b = relax)
    assert (ctx._ctx.verify_mode, ctx._ctx.check_hostname) == (ssl.CERT_NONE, False)


def test_contexts_are_still_freed(truststore):
    import gc
    import weakref

    module, api = truststore
    refs = []
    for _ in range(20):
        ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        refs.append(weakref.ref(ctx._ctx))
        del ctx
    gc.collect()
    assert not [r for r in refs if r() is not None]
    assert len(api._POLICY_OWNERS) == 0


def test_setters_inside_a_window_follow_the_ssl_module_rules(truststore):
    module, api = truststore
    ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with ctx._verification_window():
        ctx.check_hostname = True
        assert (ctx.check_hostname, ctx.verify_mode) == (True, ssl.CERT_REQUIRED)
        with pytest.raises(ValueError):
            ctx.verify_mode = ssl.CERT_NONE
    assert (ctx._ctx.check_hostname, ctx._ctx.verify_mode) == (True, ssl.CERT_REQUIRED)


def test_a_failed_window_holds_no_policy(truststore, monkeypatch):
    module, api = truststore

    @contextlib.contextmanager
    def broken(ctx):
        raise OSError("bad CA file")
        yield

    monkeypatch.setattr(api, "_configure_context", broken)
    ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    with pytest.raises(OSError):
        with ctx._verification_window():
            pass
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    assert (ctx._ctx.check_hostname, ctx._ctx.verify_mode) == (False, ssl.CERT_NONE)


def test_backends_that_never_flip_flags_write_settings_through(truststore, monkeypatch):
    # Linux: OpenSSL enforces the live context, so a setting must reach it immediately.
    module, api = truststore
    monkeypatch.setattr(api, "_HOLDS_POLICY", False)
    monkeypatch.setattr(api, "_configure_context", lambda ctx: contextlib.nullcontext())
    ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with ctx._verification_window():
        ctx.verify_mode = ssl.CERT_REQUIRED
        ctx.check_hostname = True
        assert (ctx._ctx.check_hostname, ctx._ctx.verify_mode) == (True, ssl.CERT_REQUIRED)


# Self-signed leaf for CN=unsloth-test.invalid; no OS store trusts it.
_SELF_SIGNED_DER = base64.b64decode(
    "MIIBlDCCATugAwIBAgIUXEQ8RJeYDgLZpLNQWEqVaXV8NIswCgYIKoZIzj0EAwIwHzEdMBsGA1UEAwwUdW5zbG90"
    "aC10ZXN0LmludmFsaWQwIBcNMjYxMDAxMDczMzI0WhgPMjEyNjA5MDcwNzMzMjRaMB8xHTAbBgNVBAMMFHVuc2xv"
    "dGgtdGVzdC5pbnZhbGlkMFkwEwYHKoZIzj0CAQYIKoZIzj0DAQcDQgAEC3GFqPgFuj4tNTl7hJQID2TU2SS1ycZQ"
    "hjinCUIZwynsjjWHq59sdVOnfzQleHndbEZOndbbx73W13IMGAoZtKNTMFEwHQYDVR0OBBYEFGvPMD1yHK9hNbTR"
    "a6P5I6B2Zv0NMB8GA1UdIwQYMBaAFGvPMD1yHK9hNbTRa6P5I6B2Zv0NMA8GA1UdEwEB/wQFMAMBAf8wCgYIKoZI"
    "zj0EAwIDRwAwRAIgDPyIveRS9sSmo/LG71KofNwbWJxxWhZmwqAZbQeEbycCIHujtZSrOjFRK7vNq7CHzgruUBf8"
    "E765ScsLf2YOhR0/"
)


class _DerCert(bytes):
    def __new__(cls):
        return super().__new__(cls, _SELF_SIGNED_DER)

    def public_bytes(self, _encoding):
        return bytes(self)


@pytest.mark.skipif(
    sys.platform not in ("darwin", "win32"), reason = "the OS verifier only exists on macOS / Windows"
)
def test_the_real_os_verifier_sees_the_policy_while_a_window_is_open(monkeypatch):
    name = "_unsloth_vendored_truststore_os_verifier"
    try:
        module, api = _load_vendored(name, monkeypatch)
        ctx = module.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

        class _Sock:
            context = ctx._ctx

            def get_unverified_chain(self):
                # Python < 3.13 hands back certificate objects rather than DER bytes.
                return [_DerCert()]

        with ctx._verification_window():
            # The shared context is CERT_NONE now; the caller's policy is still CERT_REQUIRED.
            assert ctx._ctx.verify_mode == ssl.CERT_NONE
            with pytest.raises(ssl.SSLCertVerificationError):
                api._verify_peercerts(_Sock(), server_hostname = "unsloth-test.invalid")
    finally:
        for key in [k for k in sys.modules if k.startswith(name)]:
            sys.modules.pop(key, None)
