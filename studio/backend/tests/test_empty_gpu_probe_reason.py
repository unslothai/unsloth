# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""_explain_empty_gpu_probe: why _get_gpu_memory came back empty.

From a real report. A user's model ran on the CPU and their whole log said:

    GGUF size: 16.4 GB, ... GPUs free: [], selected: None, --fit: on
    Load mode: the whole load (34.3 GB) fits without paging

which records the verdict and none of the reason, so the question actually being
asked -- why did my model go to system RAM -- could not be answered from it. The
probe returns [] for at least six unrelated reasons and every one lands on the CPU;
these tests pin that each names itself, and that the explainer never creates the HIP
context the amd-smi branch exists to avoid.
"""

import sys
import types

import pytest

from core.inference import llama_cpp as mod
from core.inference.llama_cpp import LlamaCppBackend


def _fake_torch(
    *,
    available = True,
    count = 2,
    rocm = True,
):
    cuda = types.SimpleNamespace(
        is_available = lambda: available,
        device_count = lambda: count,
        mem_get_info = lambda i: (_ for _ in ()).throw(
            AssertionError("mem_get_info must not be called: it creates a HIP context")
        ),
    )
    return types.SimpleNamespace(
        cuda = cuda,
        version = types.SimpleNamespace(hip = "6.2" if rocm else None, cuda = None if rocm else "12.4"),
    )


@pytest.fixture(autouse = True)
def _not_vulkan(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: False))
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/opt/llama-server")
    )


def _clear_masks(monkeypatch):
    for var in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
    ):
        monkeypatch.delenv(var, raising = False)


def _load_site_guard() -> str:
    """The `if` statement guarding the empty-probe warning, by indentation.

    Structural rather than a fixed window of characters: the guard carries a long
    comment, and a window wide enough today silently stops covering the condition the
    moment that comment grows.
    """
    import inspect
    import textwrap

    from core.inference import llama_cpp as mod

    lines = inspect.getsource(mod.LlamaCppBackend.load_model).splitlines()
    hit = next((i for i, line in enumerate(lines) if "could not enumerate any GPU" in line), None)
    assert hit is not None, "the empty-probe warning is gone"
    start = hit
    while start >= 0 and not lines[start].lstrip().startswith("if "):
        start -= 1
    assert start >= 0, "no enclosing if statement"
    # Forward to the END of the block, not just to the marker line. Stopping at the
    # marker cut the message's own continuation literal out of the extract, so a mutant
    # that rewrote it went undetected -- the test read a line that had not changed.
    base = len(lines[start]) - len(lines[start].lstrip())
    end = hit
    while end + 1 < len(lines):
        nxt = lines[end + 1]
        if nxt.strip() and (len(nxt) - len(nxt.lstrip())) <= base:
            break
        end += 1
    return textwrap.dedent("\n".join(lines[start : end + 1]))


class TestItNamesTheCause:
    def test_a_vulkan_build_says_so(self, monkeypatch):
        monkeypatch.setattr(
            LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda b = None: True)
        )
        assert "Vulkan" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_no_torch_at_all(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", None)
        assert "torch is not importable" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_torch_without_a_usable_device(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "no usable CUDA or HIP device" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_zero_devices_enumerated(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(count = 0))
        assert "enumerated 0 devices" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_the_arch_gate_dropping_every_card(self, monkeypatch):
        """The #7624 case: a build with no kernels for the cards actually present."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda b = None: frozenset({"gfx1100"})),
        )
        monkeypatch.setattr(
            LlamaCppBackend,
            "_rocm_arch_by_physical_id",
            staticmethod(lambda: {0: "gfx1030", 1: "gfx1030"}),
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "arch gate dropped every device" in msg
        assert "gfx1100" in msg and "gfx1030" in msg, "name both sides or it is not actionable"

    def test_rocm_devices_present_but_both_probes_declined(self, monkeypatch):
        """The reported host: cards visible to torch, amd-smi and the fallback empty."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda b = None: None)
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "both declined" in msg and "3" not in msg
        assert "2 ROCm device" in msg

    def test_a_covered_arch_does_not_blame_the_gate(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend,
            "_installed_llama_gfx_archs",
            staticmethod(lambda b = None: frozenset({"gfx1030"})),
        )
        monkeypatch.setattr(
            LlamaCppBackend, "_rocm_arch_by_physical_id", staticmethod(lambda: {0: "gfx1030"})
        )
        assert "arch gate" not in LlamaCppBackend._explain_empty_gpu_probe()


class TestItReportsTheVisibilityMask:
    @pytest.mark.parametrize(
        "var", ["CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "GPU_DEVICE_ORDINAL"]
    )
    def test_an_empty_mask_is_called_out(self, monkeypatch, var):
        """A container or a stale mask hides every device, and looks identical to
        having no GPU unless the mask is printed."""
        _clear_masks(monkeypatch)
        monkeypatch.setenv(var, "")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert f"{var} is empty" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_a_set_mask_is_quoted(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setenv("HIP_VISIBLE_DEVICES", "3")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "HIP_VISIBLE_DEVICES='3'" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_no_mask_adds_nothing(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "VISIBLE_DEVICES" not in LlamaCppBackend._explain_empty_gpu_probe()


class TestItIsSafeToCallOnAnyPath:
    def test_it_never_touches_mem_get_info(self, monkeypatch):
        """Reading memory creates a HIP primary context the backend never gives
        back (~700 MiB), which is the whole reason the amd-smi branch exists. The
        fake raises if it is called, so reaching a verdict here proves it was not."""
        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(
            LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(lambda b = None: None)
        )
        assert LlamaCppBackend._explain_empty_gpu_probe()

    def test_a_raising_helper_still_yields_a_reason(self, monkeypatch):
        def boom(binary = None):
            raise RuntimeError("marker unreadable")

        _clear_masks(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", _fake_torch())
        monkeypatch.setattr(LlamaCppBackend, "_installed_llama_gfx_archs", staticmethod(boom))
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "could not be determined" in msg and "RuntimeError" in msg

    def test_it_is_never_empty(self, monkeypatch):
        """An unexplained empty list is the defect; an empty explanation restores it."""
        _clear_masks(monkeypatch)
        for torch_mod in (None, _fake_torch(available = False), _fake_torch(count = 0)):
            monkeypatch.setitem(sys.modules, "torch", torch_mod)
            assert LlamaCppBackend._explain_empty_gpu_probe().strip()


class TestMacOSIsNotACpuVerdict:
    """This probe reads CUDA and HIP only, so it is empty on every Mac -- including an
    Apple Silicon one whose model is about to run entirely on the Metal GPU. There is a
    whole "No GPU is enumerated on Metal" branch in load_model for that state. Warning
    there would tell every Mac user the opposite of what happens."""

    @pytest.mark.parametrize("torch_state", ["absent", "no_cuda", "rocm_like"])
    def test_the_reason_names_metal_rather_than_blaming_torch(self, monkeypatch, torch_state):
        _clear_masks(monkeypatch)
        monkeypatch.setattr(mod, "_metal_capable_host", lambda: True)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setitem(
            sys.modules,
            "torch",
            None if torch_state == "absent" else _fake_torch(available = torch_state != "no_cuda"),
        )
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "Metal" in msg
        assert "torch" not in msg, "on macOS the probe's blind spot is the answer, not torch"

    def test_the_load_site_exempts_only_metal_capable_macs(self):
        """Source-level, because load_model cannot be driven from a unit test. The
        simulation in temp/simgpu executes this same guard across the platform matrix."""
        guard = _load_site_guard()
        assert "_metal_capable_host()" in guard, (
            "the exemption must be Apple Silicon, not every Mac: an Intel Mac with an "
            "empty probe really is CPU-only and wants this line"
        )
        assert "not _arch_gate_forced_cpu" in guard, "the arch gate warns better; do not double up"


class TestManualModeIsNotAnAbsentGpu:
    """Both Manual modes set ``gpus = []`` on purpose, to stand the Python planner down
    and let ``--fit`` or an explicit ``--gpu-layers`` place the model. The GPU is fully
    used there. The reporting user's own log shows ``GPUs free: []`` directly beneath
    "Manual mode with Auto layers hands memory management to llama.cpp --fit", so an
    empty list there is this, not a probe that failed."""

    def test_the_warning_reads_detected_gpus_not_gpus(self):
        assert (
            "not _detected_gpus" in _load_site_guard()
        ), "without this every Manual-mode load warns that it is running on the CPU"

    def test_detected_gpus_is_captured_before_manual_empties_the_pool(self):
        """The distinction only holds if the capture precedes the emptying."""
        import inspect

        from core.inference import llama_cpp as mod

        src = inspect.getsource(mod.LlamaCppBackend.load_model)
        capture = src.find("_detected_gpus = list(gpus)")
        emptied = src.find('if gpu_memory_mode == "manual" and gpu_layers < 0:')
        assert capture != -1 and emptied != -1
        assert capture < emptied, "_detected_gpus must hold the probe result, not the emptied one"


class TestItDoesNotAssertTheLaunchOutcome:
    """llama.cpp enumerates devices itself and still receives ``--fit on``, so it can
    place on a GPU this probe never saw. An explicit ``gpu_ids`` pick is pinned for the
    child further down. Claiming "will run on the CPU" from an empty probe therefore
    states an outcome this code does not know."""

    def test_the_message_reports_the_probe_not_the_placement(self):
        # Comments stripped: the guard's own note explains why it must not claim CPU,
        # and matching that would pass while the message said the opposite.
        body = "\n".join(
            line for line in _load_site_guard().splitlines() if not line.lstrip().startswith("#")
        )
        assert "could not enumerate any GPU" in body
        for claim in ("will run on the CPU", "live in system RAM", "left placement to"):
            assert claim not in body, f"the message must not assert {claim!r}"

    def test_an_explicit_gpu_pick_is_excluded(self):
        """gpu_ids is restored into gpu_indices after this block and pinned for the
        child, so those loads reach the GPU the user chose."""
        assert "not gpu_ids" in _load_site_guard()

    def test_the_pick_really_is_restored_after_the_warning(self):
        import inspect

        from core.inference import llama_cpp as mod

        src = inspect.getsource(mod.LlamaCppBackend.load_model)
        warn = src.find("could not enumerate any GPU")
        restore = src.find("gpu_indices = sorted(gpu_ids)")
        assert warn != -1 and restore != -1
        assert warn < restore, "if the pin moved above the warning, re-derive this guard"


class TestIntelMacGetsItsRealReason:
    """The load site warns on an Intel Mac, since auto_detect_backend resolves it to CPU.
    The explainer has to agree: a blanket Darwin branch would answer "macOS offloads
    through Metal" on a host that offloads through nothing."""

    def test_an_intel_mac_does_not_get_the_metal_answer(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setattr(mod, "_metal_capable_host", lambda: False)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        msg = LlamaCppBackend._explain_empty_gpu_probe()
        assert "Metal" not in msg, "an Intel Mac is not offloading to Metal"
        assert "no usable CUDA or HIP" in msg

    def test_apple_silicon_still_gets_it(self, monkeypatch):
        _clear_masks(monkeypatch)
        monkeypatch.setattr(mod, "_metal_capable_host", lambda: True)
        monkeypatch.setattr(sys, "platform", "darwin")
        monkeypatch.setitem(sys.modules, "torch", _fake_torch(available = False))
        assert "Metal" in LlamaCppBackend._explain_empty_gpu_probe()

    def test_the_explainer_and_the_load_site_use_the_same_check(self):
        """Two different notions of "is this Mac fine" is how they drifted apart."""
        import inspect

        src = inspect.getsource(mod.LlamaCppBackend._explain_empty_gpu_probe)
        assert "_metal_capable_host()" in src
        assert 'sys.platform == "darwin"' not in src
        assert "_metal_capable_host()" in _load_site_guard()


class TestItDoesNotClaimTheFitterRuns:
    """use_fit is the internal default, read before extra_args are appended to the
    command; a last-wins ``--fit off`` there wins. The later code asks
    fit_is_effectively_on for exactly this distinction, and cmd does not exist yet at
    the warning, so the message must not mention the fitter at all."""

    def test_the_message_does_not_mention_fit(self):
        body = "\n".join(
            line for line in _load_site_guard().splitlines() if not line.lstrip().startswith("#")
        )
        assert "--fit" not in body

    def test_the_command_is_built_after_the_warning(self):
        import inspect

        src = inspect.getsource(mod.LlamaCppBackend.load_model)
        assert src.find("could not enumerate any GPU") < src.find("_fitter_runs = fit_is_effectively_on"), (
            "if the effective fitter state became available above the warning, re-derive this"
        )
