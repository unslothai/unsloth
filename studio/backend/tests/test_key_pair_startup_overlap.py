# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import inspect
import threading

import pytest

from core.inference import key_exchange


@pytest.fixture(autouse = True)
def _restore_key_pair():
    saved = (
        key_exchange._private_key,
        key_exchange._public_key_pem,
        key_exchange._public_key_fingerprint,
    )
    yield
    (
        key_exchange._private_key,
        key_exchange._public_key_pem,
        key_exchange._public_key_fingerprint,
    ) = saved


def test_install_publishes_the_key_generated_off_thread(monkeypatch):
    seen = []
    real = key_exchange._generate_private_key

    def _generate():
        seen.append(threading.current_thread() is not threading.main_thread())
        return real()

    monkeypatch.setattr(key_exchange, "_generate_private_key", _generate)
    install = key_exchange.start_key_pair()
    install()
    assert seen == [True]
    pem = key_exchange.get_public_key_pem()
    assert pem.startswith("-----BEGIN PUBLIC KEY-----")
    assert key_exchange.get_public_key_fingerprint() == key_exchange._compute_fingerprint(pem)


def test_install_raises_what_generation_raised(monkeypatch):
    def _fail():
        raise RuntimeError("no entropy")

    monkeypatch.setattr(key_exchange, "_generate_private_key", _fail)
    install = key_exchange.start_key_pair()
    with pytest.raises(RuntimeError, match = "no entropy"):
        install()


def test_init_key_pair_still_generates_synchronously():
    key_exchange.init_key_pair()
    first = key_exchange.get_public_key_fingerprint()
    key_exchange.init_key_pair()
    assert first is not None and key_exchange.get_public_key_fingerprint() != first


def test_lifespan_starts_keygen_first_and_installs_it_before_serving():
    import main

    source = inspect.getsource(main.lifespan)
    start = source.index("start_key_pair()")
    assert start < source.index("clear_compiled_cache_unless_shared(app)")
    install = source.index("_install_key_pair()")
    assert start < install < source.index("start_stall_watchdog(")
    assert install < source.index("yield")
