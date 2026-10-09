# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
import os
import subprocess
import threading
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
    att._ROCM_GUARD_DISABLED.clear()
    att._ROCM_PROBE_INCOMPLETE.clear()
    yield
    for value, write in before:
        write(value)
    att._SDPA_PROBE_CACHE.clear()
    att._ROCM_GUARD_DISABLED.clear()
    att._ROCM_PROBE_INCOMPLETE.clear()


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_isolated_probe_runs_each_backend_in_a_fresh_child(monkeypatch, platform):
    from utils import subprocess_compat

    monkeypatch.setattr(subprocess_compat, "sys", SimpleNamespace(platform = platform))
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x08000000, raising = False)
    monkeypatch.setattr(subprocess, "STARTF_USESHOWWINDOW", 1, raising = False)
    monkeypatch.setattr(subprocess, "SW_HIDE", 0, raising = False)
    monkeypatch.setattr(
        subprocess, "STARTUPINFO", lambda: SimpleNamespace(dwFlags = 16), raising = False
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        status = "failed" if cmd[-1] == "FLASH_ATTENTION" else "available"
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + json.dumps(status))

    monkeypatch.setattr(subprocess, "run", run)
    assert att._probe_rocm_sdpa_kernels("cuda:1", torch.bfloat16) == ("math", "mem_efficient")
    assert calls[0][0][-1] == "MATH"
    assert {cmd[-1] for cmd, _ in calls[1:]} == {"FLASH_ATTENTION", "EFFICIENT_ATTENTION"}
    assert all(cmd[-3:-1] == ["cuda:1", "bfloat16"] and kw["timeout"] == 45 for cmd, kw in calls)
    for _, kwargs in calls:
        if platform == "win32":
            assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
            assert kwargs["startupinfo"].dwFlags == 17
            assert kwargs["startupinfo"].wShowWindow == subprocess.SW_HIDE
        else:
            assert "creationflags" not in kwargs and "startupinfo" not in kwargs


def test_fused_children_overlap_only_after_math_succeeds(monkeypatch):
    math_done = threading.Event()
    fused_started = threading.Barrier(2, timeout = 5)

    def run(cmd, **kwargs):
        if cmd[-1] == "MATH":
            math_done.set()
        else:
            assert math_done.is_set()
            # A serial implementation cannot finish either child before the other starts.
            fused_started.wait()
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + '"available"')

    monkeypatch.setattr(subprocess, "run", run)
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == (
        "math",
        "flash",
        "mem_efficient",
    )


@pytest.mark.parametrize("status", ["failed", "unavailable", "unknown", "timeout"])
def test_fused_children_are_not_started_without_working_math(monkeypatch, status):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        if status == "timeout":
            raise subprocess.TimeoutExpired(cmd, 45)
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + json.dumps(status))

    monkeypatch.setattr(subprocess, "run", run)
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == ()
    assert calls == ["MATH"]


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


def test_guard_restores_only_its_own_flags_for_a_healthy_card(monkeypatch):
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    capability = {"cuda:0": ("math",), "cuda:1": ("math", "flash", "mem_efficient")}
    monkeypatch.setattr(att, "_probe_sdpa_kernels", lambda device, dtype: capability[device])
    broken = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    healthy = SimpleNamespace(device = "cuda:1", dtype = torch.bfloat16)
    assert att.guard_rocm_fused_sdpa(broken) == ("flash", "mem_efficient")
    assert att.sdpa_math_only(healthy)
    assert att.guard_rocm_fused_sdpa(healthy) == ()
    assert att.sdpa_subquadratic_confirmed(healthy)
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()

    torch.backends.cuda.enable_flash_sdp(False)
    assert att.guard_rocm_fused_sdpa(broken) == ("mem_efficient",)
    att.guard_rocm_fused_sdpa(healthy)
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


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
    monkeypatch.setattr(att, "_sdpa_capability", lambda t: pytest.fail("unexpected probe"))
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


def test_a_successful_retry_after_an_incomplete_probe_still_disables_failed_kernels(monkeypatch):
    answers = iter([(), ("math",)])
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(att, "_probe_sdpa_kernels", lambda *a: next(answers, ("math",)))
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    att.apply_attention_backend(SimpleNamespace(), None, target = target)
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_probe_children_inherit_windows_rocm_dll_directories(monkeypatch, tmp_path):
    from utils import torch_device_probe

    monkeypatch.setattr(torch_device_probe, "_rocm_dll_directories", lambda: [str(tmp_path)])
    envs = []

    def run(cmd, **kwargs):
        envs.append(kwargs["env"])
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + '"available"')

    monkeypatch.setattr(subprocess, "run", run)
    att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16)
    assert envs and all(e[torch_device_probe.ROCM_DLL_DIRS_ENV_VAR] == str(tmp_path) for e in envs)

    added = []
    monkeypatch.setattr(probe.sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", added.append, raising = False)
    monkeypatch.setenv(torch_device_probe.ROCM_DLL_DIRS_ENV_VAR, str(tmp_path))
    probe._register_rocm_dll_directories()
    assert added == [str(tmp_path)]


def test_an_unanswered_probe_needs_no_torch(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    assert att.available_sdpa_kernels(SimpleNamespace(device = "")) == ()
    assert att.sdpa_math_only(SimpleNamespace(device = "")) is False


def test_an_incomplete_child_probe_is_not_relaunched_by_every_check_of_one_load(monkeypatch):
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        raise subprocess.TimeoutExpired(cmd, 45)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr("core._torchao_stub._module_is_rocm", lambda m: True)
    target = SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16)
    att.apply_attention_backend(SimpleNamespace(), None, target = target)
    assert not att.sdpa_math_only(target)
    assert calls == ["MATH"]
    monkeypatch.setattr(att, "_ROCM_PROBE_RETRY_S", 0.0)
    att.sdpa_math_only(target)
    assert calls == ["MATH", "MATH"]
