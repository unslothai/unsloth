# SPDX-License-Identifier: AGPL-3.0-only
"""Windows ROCm APU carve-out credit at the second host-shortfall preflight.

The tests exercise the real LlamaCppBackend method with bounded input probes;
they do not require a 97 GiB GGUF or mutate any device/driver configuration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import core.inference.llama_cpp as llama_cpp_module
from test_llama_cpp_placement import _backend, _restore_host_guard

_GiB = 1024**3


@pytest.fixture
def apu_preflight(tmp_path, monkeypatch):
    """The reporter's GGUF/lazy/device readings with isolated host probes."""
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [(0, 19_000, 96 * 1024)])
    _restore_host_guard(backend)
    monkeypatch.setattr(llama_cpp_module, "sys", SimpleNamespace(platform = "win32"))

    backend._gguf_read_bytes = lambda _path: int(97.12 * _GiB)
    backend._lazy_read_host_bytes = lambda *a, **kw: int(36.4 * _GiB)
    backend._lazy_auto_resolves_off = lambda **kw: False
    backend._shared_heap_budget = lambda *a, **kw: (0, 40 * _GiB)
    backend._amd_apu_wants_unified_memory = lambda _indices: True
    backend.probe_server_capabilities = lambda _binary: {}
    backend._host_offload_warning_opted_out = lambda: False
    backend._host_offload_shortfall_message = (
        lambda offload, _avail: "host shortfall" if offload > 18 * _GiB else None
    )
    backend._available_system_memory_mib = lambda: 20_000
    backend._cgroup_available_memory_mib = lambda: None
    argv = [
        "llama-server",
        "-m",
        str(gguf),
        "--load-mode",
        "none",
        "--lazy-mode",
        "on-direct",
        "-ngl",
        "49",
        "--fit",
        "off",
    ]
    return backend, argv


def test_verified_carveout_prevents_false_shortfall(apu_preflight):
    """Use complete model bytes in the verifier, preserving original argv."""
    backend, argv = apu_preflight
    calls = []
    backend._windows_rocm_apu_carve_out_credit = lambda size, **kw: (
        calls.append((size, kw)),
        61 * _GiB,
    )[1]
    before = argv.copy()
    assert (
        backend._launch_host_shortfall_message(
            argv,
            [(0, 19_000, 96 * 1024)],
            env = {},
            avail_mib = 20_000,
        )
        is None
    )
    assert argv == before
    assert len(calls) == 1
    assert calls[0][0] == int(97.12 * _GiB)
    assert calls[0][1]["gpu_indices"] == [0]
    assert calls[0][1]["extra_args"] == before


def test_carveout_is_max_not_sum(apu_preflight):
    """A verified 35 GiB pool plus 19,000 MiB reading must not be added."""
    backend, argv = apu_preflight
    backend._windows_rocm_apu_carve_out_credit = lambda *a, **kw: 35 * _GiB
    assert (
        backend._launch_host_shortfall_message(
            argv,
            [(0, 19_000, 96 * 1024)],
            env = {},
            avail_mib = 20_000,
        )
        == "host shortfall"
    )


@pytest.mark.parametrize(
    "case",
    [
        "linux",
        "two_gpu",
        "shared_vulkan",
        "integrated_cuda",
        "cpu_child",
        "not_rocm_apu",
    ],
)
def test_ineligible_launch_never_uses_carveout(apu_preflight, monkeypatch, case):
    """No carve-out credit when GPU ownership cannot be verified."""
    backend, argv = apu_preflight
    probes = []
    backend._windows_rocm_apu_carve_out_credit = lambda *a, **kw: (probes.append(True), 61 * _GiB)[
        1
    ]
    kwargs = {"env": {}, "avail_mib": 20_000}
    gpus = [(0, 19_000, 96 * 1024)]
    if case == "linux":
        monkeypatch.setattr(llama_cpp_module, "sys", SimpleNamespace(platform = "linux"))
    elif case == "two_gpu":
        gpus.append((1, 10_000, 20_000))
    elif case == "shared_vulkan":
        kwargs["shared_gpu_ids"] = [0]
    elif case == "integrated_cuda":
        kwargs["integrated_gpu_ids"] = [0]
    elif case == "cpu_child":
        kwargs["child_has_no_gpu"] = True
    elif case == "not_rocm_apu":
        backend._amd_apu_wants_unified_memory = lambda _indices: False
    assert backend._launch_host_shortfall_message(argv, gpus, **kwargs) == "host shortfall"
    assert not probes


def test_failed_verification_keeps_host_guard(apu_preflight):
    """Missing counters or registry attribution cannot suppress a warning."""
    backend, argv = apu_preflight

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("carve-out telemetry unavailable")

    backend._windows_rocm_apu_carve_out_credit = unavailable
    assert (
        backend._launch_host_shortfall_message(
            argv,
            [(0, 19_000, 96 * 1024)],
            env = {},
            avail_mib = 20_000,
        )
        == "host shortfall"
    )
