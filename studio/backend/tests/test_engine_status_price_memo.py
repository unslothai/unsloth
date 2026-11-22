# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Engine status polls reuse the install price until the lock, Studio's packages or a job moves."""

import os
import sys

import pytest

from core.inference import engine_install as install

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason = "local engine host is Linux only")


@pytest.fixture
def priced(monkeypatch, tmp_path):
    from core.inference import wsl_host

    monkeypatch.setattr(install, "engine_root", lambda: tmp_path)
    monkeypatch.setattr(install, "support_reason", lambda engine = "vllm", **_: None)
    monkeypatch.setattr(install, "_jobs", {})
    monkeypatch.setattr(install, "_cancels", {})
    monkeypatch.setattr(install, "_priced_plans", {})
    monkeypatch.setattr(wsl_host, "active", lambda: False)
    monkeypatch.setattr(install, "flashinfer_kernel_wheel", lambda engine, wait = True: None)
    packages = {"numpy": "2.3.0"}
    monkeypatch.setattr(install, "_studio_packages", lambda: dict(packages))
    real_plan = install.install_plan
    calls = []

    def counting_plan(engine):
        calls.append(engine)
        return real_plan(engine)

    monkeypatch.setattr(install, "install_plan", counting_plan)
    return calls, packages


def test_repeated_status_polls_price_the_plan_once(priced):
    calls, _ = priced
    first = install.status("vllm")["download_bytes"]
    for _ in range(3):
        assert install.status("vllm")["download_bytes"] == first
    assert first and first > 0
    assert calls == ["vllm"]


def test_studio_package_changes_reprice(priced, monkeypatch):
    calls, packages = priced
    before = install.download_bytes("vllm")
    pins = {name: version for name, (version, _) in install._pins("vllm").items()}
    packages["numpy"] = pins["numpy"]
    assert install.download_bytes("vllm") == before
    assert len(calls) == 2


def test_job_state_changes_and_removal_forget_the_price(priced, monkeypatch):
    calls, _ = priced
    install.download_bytes("vllm")
    install.download_bytes("vllm")
    assert len(calls) == 1
    install._update("vllm", state = "running", phase = "downloading")
    install.download_bytes("vllm")
    assert len(calls) == 2
    install._update("vllm", activity = "50%")
    install.download_bytes("vllm")
    assert len(calls) == 2, "progress updates keep the price"
    install._update("vllm", state = "success")
    install.download_bytes("vllm")
    assert len(calls) == 3
    install.remove("vllm")
    install.download_bytes("vllm")
    assert len(calls) == 4


def test_a_rewritten_lock_reprices(priced, monkeypatch, tmp_path):
    calls, _ = priced
    real = install.requirements("vllm")
    lock = tmp_path / real.name
    lock.write_bytes(real.read_bytes())
    compat = real.with_suffix(".compat.json")
    (tmp_path / compat.name).write_bytes(compat.read_bytes())
    monkeypatch.setattr(install, "requirements", lambda engine: lock)
    before = install.download_bytes("vllm")
    assert install.download_bytes("vllm") == before
    assert len(calls) == 1
    lock.write_bytes(real.read_bytes())
    os.utime(lock, ns = (1, 1))
    # A new file identity reprices; the same bytes price the same.
    assert install.download_bytes("vllm") == before
    assert len(calls) == 2


def test_the_install_itself_never_reads_the_memo():
    import inspect

    source = inspect.getsource(install._install)
    assert "install_plan(engine)" in source
    assert "_priced_plan" not in source
