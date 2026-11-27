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
def flags(monkeypatch):
    # A card AOTriton's experimental kernels are verified on, unless a test says otherwise.
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1151", raising = False)
    monkeypatch.delenv(att.ROCM_FUSED_SDPA_ALLOW_ENV, raising = False)
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
    getattr(att, "_ROCM_UNVERIFIED_LOGGED", set()).clear()
    getattr(att, "_ROCM_PROBE_FORCED_MATH", set()).clear()
    yield
    for value, write in before:
        write(value)
    att._SDPA_PROBE_CACHE.clear()
    att._ROCM_GUARD_DISABLED.clear()
    att._ROCM_PROBE_INCOMPLETE.clear()
    getattr(att, "_ROCM_UNVERIFIED_LOGGED", set()).clear()
    getattr(att, "_ROCM_PROBE_FORCED_MATH", set()).clear()


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


@pytest.mark.parametrize("dtype,disabled", [("bfloat16", True), ("float32", False)])
def test_any_completed_half_precision_probe_applies_its_answer(monkeypatch, dtype, disabled):
    monkeypatch.setattr(att, "_run_rocm_sdpa_children", lambda device, dtype: ("math",))
    monkeypatch.setattr("core._torchao_stub._module_is_rocm", lambda m: True)
    assert att.sdpa_math_only(SimpleNamespace(device = "cuda:0", dtype = getattr(torch, dtype)))
    assert torch.backends.cuda.flash_sdp_enabled() is not disabled
    assert torch.backends.cuda.mem_efficient_sdp_enabled() is not disabled


def _record_children(
    monkeypatch,
    fused = "available",
    calls = None,
):
    calls = [] if calls is None else calls

    def run(cmd, **kwargs):
        calls.append((cmd[-1], kwargs["env"].get("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL")))
        if cmd[-1] != "MATH" and fused == "timeout":
            raise subprocess.TimeoutExpired(cmd, 45)
        status = "available" if cmd[-1] == "MATH" else fused
        return SimpleNamespace(returncode = 0, stdout = probe.RESULT_PREFIX + json.dumps(status))

    monkeypatch.setattr(subprocess, "run", run)
    return calls


@pytest.mark.parametrize("arch", ["gfx1103", "gfx1152", "gfx1153", "gfx1102", ""])
def test_unverified_arch_probes_without_aotriton_experimental_kernels(monkeypatch, arch):
    """#17985: Studio's experimental opt-in ran gfx1103's experimental AOTriton kernels in the children and hung the GPU."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: arch)
    monkeypatch.setenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    calls = _record_children(monkeypatch, fused = "unavailable")
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == ("math",)
    assert sorted(calls) == [
        ("EFFICIENT_ATTENTION", "0"),
        ("FLASH_ATTENTION", "0"),
        ("MATH", "0"),
    ]
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()


_VERIFIED = ("gfx90a", "gfx942", "gfx950", "gfx1100", "gfx1101", "gfx1151", "gfx1200", "gfx1201")


@pytest.mark.parametrize("arch", _VERIFIED)
def test_verified_arch_keeps_the_experimental_opt_in(monkeypatch, arch):
    """gfx1151 / gfx1100 / gfx1201 need the opt-in on AOTriton < 0.12: their probes and fused kernels are unchanged."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: arch)
    monkeypatch.setenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    calls = _record_children(monkeypatch)
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == (
        "math",
        "flash",
        "mem_efficient",
    )
    assert {env for _, env in calls} == {"1"}
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()
    assert att.ROCM_AOTRITON_EXPERIMENTAL_VERIFIED_ARCHS == frozenset(_VERIFIED)


def test_unverified_arch_keeps_fused_kernels_aotriton_releases(monkeypatch):
    """Only the experimental opt-in is withheld: kernels AOTriton ships as supported still run fused."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1102")
    _record_children(monkeypatch)
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.float16) == (
        "math",
        "flash",
        "mem_efficient",
    )
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


@pytest.mark.parametrize("arch,disabled", [("gfx1103", True), ("gfx1151", False)])
def test_unanswered_probe_fails_closed_only_on_unverified_archs(monkeypatch, arch, disabled):
    """The parent keeps the experimental opt-in, so an unverified card keeps only fused kernels a child confirmed."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: arch)
    _record_children(monkeypatch, fused = "timeout")
    expected = ("math",) if disabled else ()
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == expected
    assert torch.backends.cuda.flash_sdp_enabled() is not disabled
    assert torch.backends.cuda.mem_efficient_sdp_enabled() is not disabled
    # The retry backoff keeps reporting what the parent now runs.
    assert att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16) == expected
    assert att._SDPA_PROBE_CACHE == {}


def test_forced_math_reads_as_math_only_and_is_retried(monkeypatch):
    """Callers bound the score matrix while MATH is forced; a later answer replaces it instead of a memoized MATH."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1103")
    answers = iter([(), ("math", "flash", "mem_efficient")])
    monkeypatch.setattr(att, "_run_rocm_sdpa_children", lambda device, dtype: next(answers))
    monkeypatch.setattr(att, "_probe_sdpa_kernels", att._probe_rocm_sdpa_kernels)
    target = "cuda:0"
    assert att._sdpa_capability(SimpleNamespace(device = target, dtype = torch.float16)) == ("math",)
    assert att._SDPA_PROBE_CACHE == {}
    assert not torch.backends.cuda.flash_sdp_enabled()
    att._ROCM_PROBE_INCOMPLETE.clear()  # backoff elapsed
    assert att._sdpa_capability(SimpleNamespace(device = target, dtype = torch.float16)) == (
        "math",
        "flash",
        "mem_efficient",
    )
    assert att._ROCM_PROBE_FORCED_MATH == set()
    assert torch.backends.cuda.flash_sdp_enabled()


def test_unanswered_read_only_probe_leaves_the_flags(monkeypatch):
    """A candidate card reads as math-only, like a direct probe, but the running card's switches stay as they are."""
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1103")
    _record_children(monkeypatch, fused = "timeout")
    with att.read_only_sdpa_probe():
        assert att._probe_rocm_sdpa_kernels("cuda:1", torch.bfloat16) == ("math",)
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


def _split_cards(monkeypatch, secondary_arch):
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(
        att, "_rocm_device_arch", lambda device: "gfx1151" if device == "cuda:0" else secondary_arch
    )

    def children(device, dtype):
        return ("math", "flash", "mem_efficient") if device == "cuda:0" else ()

    monkeypatch.setattr(att, "_run_rocm_sdpa_children", children)
    monkeypatch.setattr(att, "_probe_sdpa_kernels", att._probe_rocm_sdpa_kernels)


def test_split_guard_disables_fused_for_an_unanswered_unverified_card(monkeypatch):
    """An unanswered card whose experimental kernels are unverified would run them unconfirmed: it counts as math-only."""
    _split_cards(monkeypatch, "gfx1103")
    disabled = att.guard_rocm_fused_sdpa_cards(["cuda:0", "cuda:1"], torch.bfloat16)
    assert set(disabled) == {"flash", "mem_efficient"}
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()


def test_read_only_timeout_keeps_math_through_the_retry_backoff(monkeypatch):
    """A direct check inside the backoff after a read-only timeout must not read as unknown and keep the fused flags."""
    _split_cards(monkeypatch, "gfx1103")
    with att.read_only_sdpa_probe():
        assert att._sdpa_capability(SimpleNamespace(device = "cuda:1", dtype = torch.bfloat16)) == (
            "math",
        )
    assert torch.backends.cuda.flash_sdp_enabled()
    assert att._sdpa_capability(SimpleNamespace(device = "cuda:1", dtype = torch.bfloat16)) == ("math",)
    assert not torch.backends.cuda.flash_sdp_enabled()
    assert not torch.backends.cuda.mem_efficient_sdp_enabled()
    assert att._SDPA_PROBE_CACHE.get(("cuda:1", "torch.bfloat16")) is None


def test_split_guard_keeps_fused_for_an_unanswered_verified_card(monkeypatch):
    """A verified arch's opt-in is measured: an unanswered probe stays unknown and the answering cards decide."""
    _split_cards(monkeypatch, "gfx1100")
    assert att.guard_rocm_fused_sdpa_cards(["cuda:0", "cuda:1"], torch.bfloat16) == ()
    assert torch.backends.cuda.flash_sdp_enabled()
    assert torch.backends.cuda.mem_efficient_sdp_enabled()


def test_allow_env_probes_what_the_parent_runs(monkeypatch):
    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1103")
    monkeypatch.setenv(att.ROCM_FUSED_SDPA_ALLOW_ENV, "1")
    monkeypatch.setenv("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")
    calls = _record_children(monkeypatch, fused = "timeout")
    att._probe_rocm_sdpa_kernels("cuda:0", torch.bfloat16)
    assert {env for _, env in calls} == {"1"}
    assert torch.backends.cuda.flash_sdp_enabled()


@pytest.mark.parametrize(
    "platform,silicon,verified",
    [
        ("linux", ["gfx1103"], False),
        ("linux", None, False),
        ("linux", ["gfx1100"], True),
        ("win32", ["gfx1103"], True),
    ],
)
def test_hsa_override_is_judged_by_the_silicon(monkeypatch, platform, silicon, verified):
    """HSA_OVERRIDE_GFX_VERSION=11.0.0 on a 780M presents gfx1100; the kernels still run on gfx1103."""
    import sys

    monkeypatch.setattr(att, "_rocm_device_arch", lambda device: "gfx1100")
    monkeypatch.setattr(att, "_rocm_kfd_silicon_archs", lambda device: silicon)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("HSA_OVERRIDE_GFX_VERSION", "11.0.0")
    assert att._rocm_aotriton_experimental_verified("cuda:0") is verified
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION")
    assert att._rocm_aotriton_experimental_verified("cuda:0") is True


def test_device_arch_is_read_from_the_indexed_card(monkeypatch):
    seen = []

    def props(index):
        seen.append(index)
        return SimpleNamespace(gcnArchName = "gfx1103:xnack-")

    monkeypatch.undo()
    monkeypatch.setattr(torch.cuda, "get_device_properties", props)
    assert att._rocm_device_arch("cuda:1") == "gfx1103"
    assert seen == [1]
    assert att._rocm_aotriton_experimental_verified("cuda:1") is False


def test_auto_flash_still_runs_on_integrated_gpus(monkeypatch):
    """Strix Halo (gfx1151) is an APU too: auto flash is decided by the flash probe, not by host memory sharing."""
    monkeypatch.setattr(att, "_is_cuda_rocm", lambda t: True)
    monkeypatch.setattr(att, "_flash_attn_installed", lambda: True)
    monkeypatch.setattr(att, "_rocm_flash_attn_runs", lambda t: True)
    monkeypatch.delenv(att.ROCM_FLASH_ENV, raising = False)
    from core.inference import host_pinning

    monkeypatch.setattr(host_pinning, "device_is_integrated", lambda device = None: True)
    assert att._rocm_auto_flash(SimpleNamespace(device = "cuda:0", dtype = torch.bfloat16))


@pytest.mark.parametrize(
    "env,index,expected",
    [
        ({}, 1, ["gfx1103"]),
        ({"HIP_VISIBLE_DEVICES": "0"}, 0, ["gfx1100"]),
        ({"HIP_VISIBLE_DEVICES": "1"}, 0, ["gfx1103"]),
        ({"CUDA_VISIBLE_DEVICES": "1,0"}, 1, ["gfx1100"]),
        ({"ROCR_VISIBLE_DEVICES": "1,0", "HIP_VISIBLE_DEVICES": "1"}, 0, ["gfx1100"]),
        ({"HIP_VISIBLE_DEVICES": "GPU-abc"}, 0, ["gfx1100", "gfx1103"]),
        ({"HIP_VISIBLE_DEVICES": "0"}, 3, ["gfx1100", "gfx1103"]),
        ({"GPU_DEVICE_ORDINAL": "1"}, 0, ["gfx1100", "gfx1103"]),
    ],
)
def test_kfd_silicon_is_read_for_the_requested_device(monkeypatch, env, index, expected):
    """A gfx1100 dGPU selected by a visibility mask is judged by its own node, not an unused gfx1103 beside it."""
    from utils.hardware import amd

    import sys

    monkeypatch.setattr(sys, "platform", "linux")
    for name in (
        "ROCR_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(name, raising = False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(amd, "amd_kfd_gpu_gfx_targets", lambda: ["gfx1100", "gfx1103"])
    assert att._rocm_kfd_silicon_archs(f"cuda:{index}") == expected
