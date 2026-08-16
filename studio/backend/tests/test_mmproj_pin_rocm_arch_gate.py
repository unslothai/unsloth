# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The vision-projector CPU pin against a ROCm arch gate that actually runs.

``test_mmproj_pin_platform_matrix.py`` claims the ROCm arch gate "neither
suppresses the pin nor conjures one", but its ``rocm`` cells only pin
``_host_torch_is_rocm() == True``: the harness replaces ``_get_gpu_memory`` on
the backend instance, and ``_rocm_arch_gate_keep`` is applied INSIDE
``_get_gpu_memory``, so the stub hands back its ``memory`` verbatim and the gate
never executes. The load-level "every device gated out" branch is guarded on an
EMPTY probe, which a non-empty stub short-circuits before ``_host_torch_is_rocm``
is even consulted.

So the gate is stubbed BELOW ``_get_gpu_memory`` here, at the two sources it
reads: a fake ROCm torch (``sys.modules["torch"]``, as
``test_gpu_arch_gate_7624.py`` does) and a real ``UNSLOTH_PREBUILT_INFO.json`` on
disk. ``_installed_llama_gfx_archs``, ``_rocm_arch_by_physical_id``,
``_rocm_classify_unified_memory`` and ``_rocm_arch_gate_keep`` all run for real,
and ``_get_gpu_memory`` returns whatever they decide.

#7670 (on main) made the gated state reachable in the field: it fails OPEN
whenever coverage is unknown or the arch is in ``mapped_targets``, so a gated
pool is a real host shape rather than a hypothetical one.

Three pool shapes, each asked what the pin does:
  * the gate keeps every device -- the pin must decide exactly as it would
    ungated;
  * the gate drops SOME (mixed pool) -- the pin must decide on the SURVIVORS,
    not on the pool the probe started from, in both directions;
  * the gate drops ALL -- the load is forced to CPU, and there is nothing left
    to pin to.

``_discrete_vram`` is asked the same question: it is computed from the SURVIVING
pool, so a gate that removes the shared-memory APU and leaves a discrete card
must flip it on, and one that removes the discrete card and leaves the APU must
flip it off.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

# Reuses the placement harness (and its module stubs). Import before anything
# from core.inference.
from test_llama_cpp_placement import _backend  # noqa: E402,F401

from core.inference import llama_cpp  # noqa: E402
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_REAL_POPEN = subprocess.Popen

_MIB = 1024 * 1024
_PIN = "--no-mmproj-offload"

# Same sizes as the platform matrix, so the two suites ask the planner the same
# question and only the gate varies. A card the model alone nearly fills: the
# projector's 900 MiB is what tips it over.
_PROJECTOR_BYTES = 900 * _MIB
_TIGHT_MODEL_BYTES = 4_500 * _MIB
_TIGHT_FREE_MIB = 6_000
_TIGHT_TOTAL_MIB = 8_000


class _Device:
    """One ROCm device as the two probes below see it.

    ``total_mib`` 0 marks a unified-memory APU: the arch alone decides that (via
    ``_rocm_classify_unified_memory``), and ``_get_gpu_memory`` is what reports
    the 0, so it is derived here rather than passed.
    """

    def __init__(self, arch: str, free_mib: int, total_mib: int):
        self.arch = arch
        self.free_mib = free_mib
        self.total_mib = total_mib


# gfx1101 / gfx1100: discrete RDNA3. gfx900: a Vega the modern prebuilts do not
# carry kernels for. gfx1151 (Strix Halo) is classified unified-memory by
# _rocm_classify_unified_memory, so its "VRAM" is shared system RAM.
_DGPU_TIGHT = _Device("gfx1101", _TIGHT_FREE_MIB, _TIGHT_TOTAL_MIB)
_DGPU_ROOMY = _Device("gfx1100", 40_000, 48_000)
_OLD_ROOMY = _Device("gfx900", 40_000, 48_000)
_APU = _Device("gfx1151", 40_000, 128_000)
# A shared pool sized to offer the SAME usable budget as _DGPU_TIGHT once
# _get_gpu_memory has taken the 1024 MiB host reserve off it. Without that the
# "no pin on a shared pool" case is untestable: an APU the model does not fit on
# either way answers "no pin" whatever _discrete_vram says, and the assertion
# cannot fail. Derived below and pinned by an assertion in the test.
_APU_TIGHT = _Device("gfx1151", 5_940 + llama_cpp._IGPU_HOST_RESERVE_MIB, 128_000)


def _discrete_usable_mib(free_mib: float, total_mib: float) -> float:
    return llama_cpp._vram_usable_mib(free_mib, total_mib, llama_cpp._CTX_FIT_VRAM_FRACTION)


def _apu_usable_mib(free_mib: float) -> float:
    # A shared pool reports total 0, so the budget comes off free alone.
    return llama_cpp._vram_usable_mib(free_mib, 0, llama_cpp._CTX_FIT_VRAM_FRACTION)


class _FakeProps:
    """hipDeviceProp_t stand-in: the canonical arch attribute plus the total the
    unified-memory classifier and ``_rocm_total_memory_mib_by_physical_id`` read."""

    def __init__(self, device: _Device):
        self.gcnArchName = device.arch
        self.total_memory = device.total_mib * _MIB
        self.name = "AMD Radeon Graphics"


def _fake_rocm_torch(devices: list[_Device]):
    torch = types.ModuleType("torch")
    torch.version = types.SimpleNamespace(hip = "6.4.0")
    torch.__version__ = "2.6.0+rocm6.4"
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(devices),
        mem_get_info = lambda o: (devices[o].free_mib * _MIB, devices[o].total_mib * _MIB),
        get_device_properties = lambda o: _FakeProps(devices[o]),
    )
    return torch


@pytest.fixture
def rocm_host(tmp_path, monkeypatch):
    """Make this box a ROCm host for the duration, without touching the gate.

    Everything patched here is BELOW ``_get_gpu_memory``: the nvidia-smi child
    (this runner has a real NVIDIA card and would otherwise answer first), the
    binary path the install marker is read relative to, and the inherited
    visibility masks. Returns a callable that plants the marker.
    """
    _real_run = subprocess.run

    def _no_nvidia_smi(cmd, *args, **kwargs):
        # Only nvidia-smi is denied. Blanket-patching subprocess.run would also
        # silence the llama-server capability probe, and the pin reads its answer
        # through _paravirtual_mmproj_pinnable.
        if cmd and str(cmd[0]) == "nvidia-smi":
            raise FileNotFoundError("nvidia-smi")
        return _real_run(cmd, *args, **kwargs)

    monkeypatch.setattr(llama_cpp.subprocess, "run", _no_nvidia_smi)
    binary = str(tmp_path / "build" / "bin" / "llama-server")
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda **_kw: binary)
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b = None: False))
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising = False)

    def _install(devices: list[_Device], coverage):
        """Plant the fake torch and, when ``coverage`` is not None, the install
        marker whose ``mapped_targets`` the gate reads. None means no marker at
        all -- a source build, where coverage is unknown and the gate fails open."""
        monkeypatch.setitem(sys.modules, "torch", _fake_rocm_torch(devices))
        import utils.llama_cpp_freshness as freshness

        freshness._marker_cache.clear()
        if coverage is not None:
            (tmp_path / "UNSLOTH_PREBUILT_INFO.json").write_text(
                json.dumps({"mapped_targets": list(coverage)}), encoding = "utf-8"
            )
        return binary

    return _install


def _vision_backend(tmp_path, *, model_bytes = _TIGHT_MODEL_BYTES):
    """A vision GGUF backend whose GPU probe is NOT stubbed.

    ``_backend`` replaces ``_get_gpu_memory`` on the instance; that is the hole
    this file exists to close, so both probe stubs are removed and the real
    staticmethods answer from the fake torch above.
    """
    backend, gguf = _backend(tmp_path, vulkan = False, memory = [])
    del backend._get_gpu_memory
    del backend._get_gpu_free_memory
    del backend._is_vulkan_backend
    del backend._find_llama_server_binary
    backend._get_gguf_size_bytes = lambda _path: model_bytes
    # The real estimator returns several GB against the stub GGUF, which swamps
    # every other term and makes nothing fit on any card -- a pin test built on
    # it would pass whatever the policy did.
    backend._estimate_compute_buffer_bytes = lambda *a, **k: 100 * _MIB
    mmproj = tmp_path / "model-mmproj.gguf"
    mmproj.write_bytes(b"\x00" * 16)
    backend._resolve_launch_mmproj_path = lambda **kwargs: str(mmproj)
    backend._mmproj_vram_bytes = lambda _path: _PROJECTOR_BYTES
    backend._mmproj_matches_model_family = lambda *a, **k: True
    return backend, gguf


def _probe(binary, *, for_llama_server: bool):
    return LlamaCppBackend._get_gpu_memory(binary, for_llama_server = for_llama_server)


def _launch(backend, gguf, **load_kwargs):
    """``test_llama_cpp_placement._launch`` with the binary matched by name.

    That harness keys its fake Popen on the literal ``/fake/llama-server``, but
    the install marker is read by walking UP from the binary, so the binary has
    to live under ``tmp_path`` here for the gate to find any coverage at all.
    """
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or not str(cmd[0]).endswith("llama-server"):
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        captured["env"] = kwargs.get("env") or dict(os.environ)
        return types.SimpleNamespace(
            pid = 123,
            stdout = (),
            poll = lambda: None,
            terminate = lambda: None,
            wait = lambda timeout = None: 0,
            kill = lambda: None,
        )

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(
            GgufLoadIntent(
                gguf_path = str(gguf),
                model_identifier = "test",
                **load_kwargs,
            )
        )
    return captured


# --------------------------------------------------------------------------
# The gate really runs: the probe is what proves it
# --------------------------------------------------------------------------


class TestTheGateActuallyExecutes:
    """Before asking what the pin does with a gated pool, prove the pool IS gated.

    Each case reads the probe both ways: ``for_llama_server=True`` opts into the
    gate, the default does not. A gate that never executed would return the same
    list twice.
    """

    def test_keeps_every_covered_device(self, tmp_path, rocm_host):
        binary = rocm_host([_DGPU_TIGHT, _DGPU_ROOMY], ["gfx1100", "gfx1101"])
        gated = _probe(binary, for_llama_server = True)
        assert gated == [(0, _TIGHT_FREE_MIB, _TIGHT_TOTAL_MIB), (1, 40_000, 48_000)]
        assert gated == _probe(binary, for_llama_server = False)

    def test_drops_the_uncovered_device_only(self, tmp_path, rocm_host):
        binary = rocm_host([_DGPU_TIGHT, _OLD_ROOMY], ["gfx1101"])
        assert _probe(binary, for_llama_server = True) == [(0, _TIGHT_FREE_MIB, _TIGHT_TOTAL_MIB)]
        # The ungated probe still sees both: the gate answers "what can
        # llama-server run on", not "what GPUs exist".
        assert len(_probe(binary, for_llama_server = False)) == 2

    def test_drops_every_device_when_nothing_is_covered(self, tmp_path, rocm_host):
        binary = rocm_host([_DGPU_TIGHT, _OLD_ROOMY], ["gfx1030"])
        assert _probe(binary, for_llama_server = True) == []
        assert len(_probe(binary, for_llama_server = False)) == 2

    def test_unknown_coverage_keeps_everything(self, tmp_path, rocm_host):
        # No marker: a source build or a custom link. #7670 fails open here, which
        # is why the gated cases above are not the only reachable ones.
        binary = rocm_host([_DGPU_TIGHT, _OLD_ROOMY], None)
        assert len(_probe(binary, for_llama_server = True)) == 2


# --------------------------------------------------------------------------
# What the pin does with a gated pool
# --------------------------------------------------------------------------


class TestPinAgainstAGatedPool:
    def test_gate_keeps_all_and_the_roomy_card_still_answers(self, tmp_path, rocm_host):
        """Gate keeps all: a covered roomy card is in the pool, so no pin.

        The tight card alone would be pinned (the case below proves it), so this
        is not a placement that would come out unpinned whatever the pool was.
        """
        rocm_host([_DGPU_TIGHT, _DGPU_ROOMY], ["gfx1100", "gfx1101"])
        backend, gguf = _vision_backend(tmp_path)

        cmd = _launch(backend, gguf, is_vision = True)["cmd"]

        assert _PIN not in cmd
        assert "--mmproj" in cmd
        assert backend.vision_on_cpu is False
        assert backend._arch_gate_forced_cpu is False

    def test_gate_drops_some_and_the_pin_follows_the_survivors(self, tmp_path, rocm_host):
        """Mixed pool: the roomy card is uncovered, so the pin is decided on the
        tight survivor and fires.

        This is the assertion the platform matrix could not make. The pool the
        probe starts from would not be pinned at all (see the companion case
        below, which is the same host with the gate failing open); the pool the
        gate leaves is pinned. The pin is following the survivors.
        """
        rocm_host([_DGPU_TIGHT, _OLD_ROOMY], ["gfx1101"])
        backend, gguf = _vision_backend(tmp_path)

        result = _launch(backend, gguf, is_vision = True)
        cmd = result["cmd"]

        assert cmd.count(_PIN) == 1
        # Pinned, not disabled: the projector still loads, on the CPU.
        assert "--mmproj" in cmd
        assert backend.is_vision is True
        assert backend.vision_on_cpu is True
        assert backend.vision_disabled_by_user is False
        # Nothing was forced to CPU: a survivor remained.
        assert backend._arch_gate_forced_cpu is False

    def test_the_same_host_is_not_pinned_when_coverage_is_unknown(self, tmp_path, rocm_host):
        """The control for the case above: identical hardware, no install marker.

        The gate fails open, the roomy gfx900 is back in the pool, and the model
        fits with the projector on the GPU. So the pin in the previous test is
        the gate's doing and not the fixture's.
        """
        rocm_host([_DGPU_TIGHT, _OLD_ROOMY], None)
        backend, gguf = _vision_backend(tmp_path)

        cmd = _launch(backend, gguf, is_vision = True)["cmd"]

        assert _PIN not in cmd
        assert backend.vision_on_cpu is False

    def test_gate_drops_all_forces_cpu_and_pins_nothing(self, tmp_path, rocm_host):
        """Empty pool: there is no GPU to free VRAM on, so no pin, and the child
        is masked off every card it would otherwise enumerate and abort on.

        The model is deliberately the one the pin WOULD fire for on a covered
        card, so a policy that keyed off "does it fit" rather than off an
        enumerated device would pin here.
        """
        rocm_host([_DGPU_TIGHT, _OLD_ROOMY], ["gfx1030"])
        backend, gguf = _vision_backend(tmp_path)

        result = _launch(backend, gguf, is_vision = True)
        cmd, env = result["cmd"], result["env"]

        assert _PIN not in cmd
        # Vision is untouched: the projector loads and runs on llama.cpp's CPU
        # backend. Forcing CPU is not a reason to drop it silently.
        assert "--mmproj" in cmd
        assert backend.vision_on_cpu is False
        assert backend.vision_disabled_by_user is False
        assert backend._arch_gate_forced_cpu is True
        assert env.get("CUDA_VISIBLE_DEVICES") == "-1"
        assert env.get("HIP_VISIBLE_DEVICES") == "-1"


# --------------------------------------------------------------------------
# _discrete_vram is computed from the SURVIVING pool
# --------------------------------------------------------------------------


class TestDiscreteVramFollowsTheSurvivors:
    """The pin is gated on ``_discrete_vram`` -- every device in the pool reports
    a total, i.e. none of them is a shared system-RAM pool. That is derived from
    the pool AFTER the gate, so the gate can flip it either way.

    Not observable directly (it is a local), so it is read through the pin, which
    is the only consumer whose answer differs on it for these pools.
    """

    def test_dropping_the_apu_leaves_a_discrete_pool_and_pins(self, tmp_path, rocm_host):
        # gfx1151 Strix Halo shares system RAM, so _get_gpu_memory reports total 0
        # for it and _discrete_vram is False for the ungated pool. Gate it out and
        # the survivor is a discrete card: the pin becomes available and fires.
        binary = rocm_host([_APU, _DGPU_TIGHT], ["gfx1101"])
        ungated = _probe(binary, for_llama_server = False)
        assert [g[2] for g in ungated] == [
            0,
            _TIGHT_TOTAL_MIB,
        ], "the APU must report a total of 0, or this case is not about _discrete_vram"
        assert _probe(binary, for_llama_server = True) == [(1, _TIGHT_FREE_MIB, _TIGHT_TOTAL_MIB)]

        backend, gguf = _vision_backend(tmp_path)
        cmd = _launch(backend, gguf, is_vision = True)["cmd"]

        assert cmd.count(_PIN) == 1
        assert backend.vision_on_cpu is True

    def test_the_same_pool_is_not_pinned_when_the_apu_survives(self, tmp_path, rocm_host):
        """Control for the case above: identical hardware, no install marker.

        The gate fails open, the roomy APU is back in the pool, and the model
        fits with the projector on the GPU, so the pin in the previous test is
        the gate's doing. This one is about the GATE, not about _discrete_vram
        (the roomy pool fits either way); the single-APU case below is where
        _discrete_vram is the only thing answering.
        """
        rocm_host([_APU, _DGPU_TIGHT], None)
        backend, gguf = _vision_backend(tmp_path)

        cmd = _launch(backend, gguf, is_vision = True)["cmd"]

        assert _PIN not in cmd
        assert backend.vision_on_cpu is False

    def test_a_surviving_apu_alone_is_still_not_discrete(self, tmp_path, rocm_host):
        """Gate drops the discrete card and leaves only the shared pool.

        The surviving APU is deliberately as tight as the discrete card that gets
        dropped, so a pin keyed on "does it fit" would fire; ``_discrete_vram``
        is what stops it.
        """
        binary = rocm_host([_APU_TIGHT, _DGPU_TIGHT], ["gfx1151"])
        gated = _probe(binary, for_llama_server = True)
        assert [g[0] for g in gated] == [0]
        assert gated[0][2] == 0, "a shared pool must report no total"
        # The survivor offers the SAME budget as the discrete card the gate just
        # dropped -- which is a card the pin does fire for (see
        # test_gate_drops_some_and_the_pin_follows_the_survivors). So the only
        # thing left to answer differently here is _discrete_vram.
        assert _apu_usable_mib(gated[0][1]) == pytest.approx(
            _discrete_usable_mib(_TIGHT_FREE_MIB, _TIGHT_TOTAL_MIB), abs = 8
        )

        backend, gguf = _vision_backend(tmp_path)
        cmd = _launch(backend, gguf, is_vision = True)["cmd"]

        assert _PIN not in cmd
        assert "--mmproj" in cmd
        assert backend.vision_on_cpu is False
