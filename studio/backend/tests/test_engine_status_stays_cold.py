# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Idle engine discovery must not import the runtime it merely describes."""

import threading
from types import SimpleNamespace

import pytest

from core import _torchao_stub
from core.inference import engine_install as install
from utils.hardware import hardware


@pytest.mark.parametrize("system", ["Linux", "Windows"])
@pytest.mark.parametrize("rocm", [False, True, None])
def test_cold_engine_profiles_use_build_metadata_without_detecting(monkeypatch, system, rocm):
    monkeypatch.setattr(install.platform, "system", lambda: system)
    monkeypatch.setattr(
        install.os.path, "exists", lambda path: path == "/proc/driver/nvidia/version"
    )
    monkeypatch.setattr(hardware, "DETECTION_COMPLETE", threading.Event())
    monkeypatch.delitem(install.sys.modules, "torch", raising = False)
    monkeypatch.setattr(_torchao_stub, "_installed_torch_is_rocm", lambda: rocm)
    monkeypatch.setattr(
        hardware, "ensure_hardware_detected", lambda: pytest.fail("idle profile detected hardware")
    )
    assert install.gpu_platform() == ("rocm" if rocm else "cuda")


def test_loaded_build_and_settled_verdict_take_precedence_over_disk(monkeypatch):
    monkeypatch.setattr(install.platform, "system", lambda: "Windows")
    complete = threading.Event()
    monkeypatch.setattr(hardware, "DETECTION_COMPLETE", complete)
    monkeypatch.setattr(_torchao_stub, "_installed_torch_is_rocm", lambda: False)
    monkeypatch.setitem(
        install.sys.modules,
        "torch",
        SimpleNamespace(__version__ = "2.12.0+rocm7.2", version = SimpleNamespace(hip = "7.2")),
    )
    assert install.gpu_platform() == "rocm"
    monkeypatch.setattr(hardware, "IS_ROCM", False)
    complete.set()
    assert install.gpu_platform() == "cuda"


def test_cold_rocm_architecture_read_does_not_cache_a_provisional_empty_result(monkeypatch):
    monkeypatch.delitem(install.sys.modules, "torch", raising = False)
    monkeypatch.setattr(install, "_rocm_arches", None)
    probes = []
    monkeypatch.setattr(install, "_rocm_gpu_arches", lambda: probes.append(True) or {0: "gfx1100"})
    assert install._known_rocm_gpu_arches(wait = False) == {}
    assert install._rocm_arches is None
    assert probes == []
    assert install._known_rocm_gpu_arches(wait = True) == {0: "gfx1100"}
    assert probes == [True]


def test_cold_rocm_status_reuses_measured_architectures(monkeypatch):
    monkeypatch.delitem(install.sys.modules, "torch", raising = False)
    monkeypatch.setattr(install, "_rocm_arches", {0: "gfx1100"})
    monkeypatch.setattr(install, "_rocm_gpu_arches", lambda: pytest.fail("status imported torch"))
    assert install._known_rocm_gpu_arches(wait = False) == {0: "gfx1100"}


def test_rocm_status_can_resolve_architectures_after_explicit_hardware_use(monkeypatch):
    monkeypatch.setitem(install.sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(install, "_rocm_gpu_arches", lambda: {1: "gfx1200"})
    assert install._known_rocm_gpu_arches(wait = False) == {1: "gfx1200"}
