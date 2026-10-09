# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import subprocess
from types import SimpleNamespace

import pytest

from core.inference import diffusion_attention as att
from core.inference import rocm_sdpa_probe as probe

torch = pytest.importorskip("torch")


@pytest.fixture(autouse = True)
def flags():
    before = [
        (read(), write)
        for read, write in (
            (torch.backends.cuda.flash_sdp_enabled, torch.backends.cuda.enable_flash_sdp),
            (
                torch.backends.cuda.mem_efficient_sdp_enabled,
                torch.backends.cuda.enable_mem_efficient_sdp,
            ),
            (torch.backends.cuda.math_sdp_enabled, torch.backends.cuda.enable_math_sdp),
        )
    ]
    for _, write in before:
        write(True)
    att._SDPA_PROBE_CACHE.clear()
    yield
    for value, write in before:
        write(value)
    att._SDPA_PROBE_CACHE.clear()


def test_isolated_probe_runs_each_backend_in_a_fresh_child(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        status = "failed" if cmd[-1] == "FLASH_ATTENTION" else "available"
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + json.dumps(status))

    monkeypatch.setattr(subprocess, "run", run)
    assert att._probe_rocm_sdpa_kernels("cuda:1", torch.bfloat16) == ("math", "mem_efficient")
    assert [cmd[-1] for cmd, _ in calls] == ["MATH", "FLASH_ATTENTION", "EFFICIENT_ATTENTION"]
    assert all(cmd[-3:-1] == ["cuda:1", "bfloat16"] and kw["timeout"] == 45 for cmd, kw in calls)


@pytest.mark.parametrize("failure", ["timeout", "exit", "garbage", "unknown"])
def test_incomplete_probe_does_not_disable_or_cache_kernels(monkeypatch, failure):
    def run(cmd, **kw):
        if cmd[-1] == "MATH":
            return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + '"available"')
        if failure == "timeout":
            raise subprocess.TimeoutExpired(cmd, 45)
        return SimpleNamespace(
            returncode = int(failure == "exit"),
            stdout = ("garbage" if failure == "garbage" else probe.RESULT_PREFIX + '"unknown"'),
        )

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(torch.version, "hip", "7.14")
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    assert att.guard_rocm_fused_sdpa(target) == ()
    assert att._SDPA_PROBE_CACHE == {}
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


@pytest.mark.parametrize(
    "available,disabled",
    [
        (("math",), ("flash", "mem_efficient")),
        (("math", "flash"), ("mem_efficient",)),
        (("math", "flash", "mem_efficient"), ()),
        ((), ()),
    ],
)
def test_guard_only_disables_failed_backends(monkeypatch, available, disabled):
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(att, "_probe_sdpa_kernels", lambda *a: available)
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    assert att.guard_rocm_fused_sdpa(target) == disabled
    assert torch.backends.cuda.flash_sdp_enabled() == ("flash" not in disabled)
    assert torch.backends.cuda.mem_efficient_sdp_enabled() == ("mem_efficient" not in disabled)
    assert att.guard_rocm_fused_sdpa(target) == ()


def test_cached_capabilities_follow_current_flags(monkeypatch):
    calls = []

    def run(*args):
        calls.append(args)
        return ("math", "flash", "mem_efficient")

    monkeypatch.setattr(att, "_probe_sdpa_kernels", run)
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    assert att.sdpa_subquadratic_confirmed(target)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    assert att.sdpa_math_only(target)
    assert not att.sdpa_subquadratic_confirmed(target)
    torch.backends.cuda.enable_flash_sdp(True)
    assert att.sdpa_subquadratic_confirmed(target)
    assert len(calls) == 1


@pytest.mark.parametrize("rocm,allow", [(False, False), (True, True)])
def test_other_platforms_and_explicit_override_do_not_probe(monkeypatch, rocm, allow):
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: rocm)
    monkeypatch.setenv(att.ROCM_FUSED_SDPA_ALLOW_ENV, "1" if allow else "0")
    monkeypatch.setattr(att, "available_sdpa_kernels", lambda t: pytest.fail("unexpected probe"))
    assert att.guard_rocm_fused_sdpa(SimpleNamespace()) == ()


def test_probe_consumes_deferred_launch_errors(monkeypatch):
    real = torch.nn.functional.scaled_dot_product_attention

    class Pending:
        def float(self):
            raise RuntimeError("hipErrorInvalidValue")

    def sdpa(*args, **kwargs):
        return real(*args, **kwargs) if torch.backends.cuda.math_sdp_enabled() else Pending()

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    assert probe.probe("cpu", "float32", "FLASH_ATTENTION") == "failed"
    assert probe.probe("cpu", "float32", "MATH") == "available"


def test_probe_rejects_numerically_wrong_fused_result(monkeypatch):
    real = torch.nn.functional.scaled_dot_product_attention

    def sdpa(*args, **kwargs):
        result = real(*args, **kwargs)
        return result if torch.backends.cuda.math_sdp_enabled() else result + 1

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    assert probe.probe("cpu", "float32", "FLASH_ATTENTION") == "failed"
