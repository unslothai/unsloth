# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Regression tests for issue #9549 -- "Studio loads into RAM despite the VRAM-only checks".

Reporter: the #7072 / #7452 host again. Windows 10, ROCm 7.13, two DISCRETE cards
(Radeon PRO W7900 45 GiB + W7500 7.98 GiB), both Model Memory toggles on, and the
model still accounted to system RAM.

The companion report #9550 supplies the launch line from the same box:

    -ngl -1 --fit off ... GPUs free: [(0, 45914), (1, 8032)]

so the weights ARE fully offloaded and the fitter DOES read VRAM. What is missing
from that argv is any memory flag at all -- and that is by design:
``_weights_in_host_memory`` is False for a discrete full offload, so
``apply_model_memory_policy`` skips the lock (mlock would pin a second full copy in
host RAM for weights that are not there). Residency is then carried by the
idle-unload veto alone.

These tests pin that behaviour end to end on a FAKED version of the reporter's host,
driven from torch device properties rather than by stubbing the APU predicate, so the
whole chain is exercised: props -> _rocm_classify_unified_memory ->
_amd_apu_wants_unified_memory -> _weights_in_host_memory -> apply_model_memory_policy.

The last test covers #9551 (W7900 shown as 45 rather than 48) on the same fixture,
because it is the same two cards and the same probe.

Mocks throughout: torch, the Windows performance counter and the platform are faked;
this repository has no AMD hardware and no Windows or ROCm CI. Fixtures are imported
from test_rocm_windows_vram_7072, following the idiom test_rocm_windows_vram_7452
already uses for the same pair of cards.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from utils.hardware import hardware as hw

_BACKEND = Path(__file__).resolve().parent.parent
_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

from test_rocm_windows_vram_7072 import (  # noqa: E402, F401  (win_rocm is a fixture)
    GB,
    _adapter_output,
    _subprocess_run,
    win_rocm,
)

# Load llama_server_args directly, as test_model_memory_settings.py does: importing
# the package would drag in the whole inference chain.
_spec = importlib.util.spec_from_file_location(
    "_lsa_9549_test_only", _BACKEND / "core" / "inference" / "llama_server_args.py"
)
_lsa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lsa)
apply_model_memory_policy = _lsa.apply_model_memory_policy
memory_state_satisfies_settings = _lsa.memory_state_satisfies_settings


# The reporter's cards. Discrete RDNA3: is_integrated is 0 and the arch is gfx1100,
# neither of which the unified-memory classifier accepts.
W7900 = {"name": "AMD Radeon PRO W7900", "total": 46080 * 1024 * 1024, "arch": "gfx1100"}
W7500 = {"name": "AMD Radeon PRO W7500", "total": 8176 * 1024 * 1024, "arch": "gfx1102"}
# The control: one Strix Halo APU, where the same toggles SHOULD lock.
STRIX_HALO = {
    "name": "AMD Radeon 8060S Graphics",
    "total": 64 * GB,
    "arch": "gfx1151",
    "is_integrated": 1,
}


def _props_torch(devices):
    """A ROCm torch whose device properties carry arch + the integrated flag.

    Driving the APU predicate from real properties is the point: stubbing
    ``_amd_apu_wants_unified_memory`` would assume the very classification this
    issue turns on.
    """

    class _Props:
        def __init__(self, spec):
            self.name = spec["name"]
            self.total_memory = spec["total"]
            self.is_integrated = spec.get("is_integrated", 0)
            self.gcnArchName = spec["arch"]

    torch = types.ModuleType("torch")
    torch.__version__ = "2.11.0+rocm7.13"
    torch.version = types.SimpleNamespace(hip = "7.13", cuda = None)
    torch.cuda = types.SimpleNamespace(
        is_available = lambda: True,
        device_count = lambda: len(devices),
        current_device = lambda: 0,
        get_device_properties = lambda i: _Props(devices[i]),
        # The WDDM over-report: free comes back at total on a card that is full.
        mem_get_info = lambda i: (devices[i]["total"], devices[i]["total"]),
        memory_allocated = lambda i = None: 0,
        memory_reserved = lambda i = None: 0,
    )
    return torch


@pytest.fixture
def reporter_host(win_rocm, monkeypatch):
    """The reporter's box: Windows ROCm, two discrete cards, amd-smi unavailable."""
    monkeypatch.setitem(sys.modules, "torch", _props_torch([W7900, W7500]))
    return monkeypatch


@pytest.fixture
def apu_host(monkeypatch):
    """The control: Linux ROCm, one unified-memory APU."""
    monkeypatch.setattr(hw, "IS_ROCM", True)
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("ROCR_VISIBLE_DEVICES", raising = False)
    monkeypatch.setitem(sys.modules, "torch", _props_torch([STRIX_HALO]))
    return monkeypatch


def _gate(**kwargs):
    """``_weights_in_host_memory`` on a minimal backend stand-in.

    Unlike test_model_memory_settings.TestHostMemoryGate this does NOT stub the
    APU predicate: it runs the real one against the faked torch above.
    """
    from core.inference.llama_cpp import LlamaCppBackend

    backend = type(
        "_B",
        (),
        {
            "n_layers": 48,
            "_n_cpu_moe": 0,
            "_weights_in_host_memory": LlamaCppBackend._weights_in_host_memory,
            "_offloads_every_layer": LlamaCppBackend._offloads_every_layer,
            "_amd_apu_wants_unified_memory": staticmethod(
                LlamaCppBackend._amd_apu_wants_unified_memory
            ),
            "_vulkan_targets_are_igpus": staticmethod(lambda binary, idx = None: False),
        },
    )()
    params = {
        "fully_gpu_offloaded": True,
        "gpu_memory_mode": "auto",
        "gpu_layers": None,
        "extra_args": None,
    }
    params.update(kwargs)
    return backend._weights_in_host_memory(**params)


def _policy(monkeypatch, keep_resident, no_ram_reserve, extras, *, host_resident):
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_keep_resident", lambda: keep_resident)
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: no_ram_reserve)
    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: (keep_resident, no_ram_reserve))
    monkeypatch.setattr(mm, "should_mlock", lambda: keep_resident and not no_ram_reserve)
    return apply_model_memory_policy(
        extras,
        supports_load_mode = True,
        weights_in_host_memory = host_resident,
    )


# ----------------------------------------------------------------------------- #
# The reproduction
# ----------------------------------------------------------------------------- #
def test_discrete_windows_full_offload_is_not_host_resident(reporter_host):
    """Two discrete RDNA3 cards are not unified memory, so a full offload puts no
    weights in pageable host RAM and there is nothing for mlock to pin."""
    assert _gate() is False


def test_keep_resident_emits_no_lock_on_the_reporters_host(reporter_host, monkeypatch):
    """#9549 as a unit test.

    "Keep model in GPU memory" is ON, "Don't reserve system RAM" is OFF -- the
    combination that is supposed to lock -- and the launch still carries no memory
    flag whatsoever, because the weights are on the cards.
    """
    host_resident = _gate()
    managed, extras = _policy(monkeypatch, True, False, [], host_resident = host_resident)
    assert managed == []
    assert extras == []
    assert "--mlock" not in managed
    assert "--load-mode" not in managed


@pytest.mark.parametrize(
    "keep_resident,no_ram_reserve",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_every_toggle_pair_produces_the_same_argv_on_a_discrete_offload(
    reporter_host, monkeypatch, keep_resident, no_ram_reserve
):
    """All four cells are byte-identical here, which is exactly what the reporter
    observes when he toggles the boxes and reloads. Nothing is being ignored: with
    the weights on the cards there is no flag either toggle can contribute."""
    host_resident = _gate()
    managed, extras = _policy(
        monkeypatch, keep_resident, no_ram_reserve, [], host_resident = host_resident
    )
    assert (managed, extras) == ([], [])


def test_no_ram_reserve_still_strips_a_user_supplied_mlock(reporter_host, monkeypatch):
    """The one thing that IS observable on this host: a hand-typed --mlock is
    removed. Without a user extra to strip there is nothing to see, which is why a
    bare 2x2 looks like a no-op."""
    host_resident = _gate()
    managed, extras = _policy(monkeypatch, False, True, ["--mlock"], host_resident = host_resident)
    assert managed == []
    assert "--mlock" not in extras


def test_the_settings_api_reports_satisfied_while_the_toggle_reads_on(reporter_host):
    """The UX half of #9549: keep_resident is ON, the launch holds no lock, and the
    reload check still says the launch matches the settings -- so nothing in the UI
    ever tells the user the toggle did not apply. ``mlock_applicable`` False is the
    deliberate excuse (llama_server_args.memory_state_satisfies_settings)."""
    import utils.model_memory_settings as mm
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(mm, "get_keep_resident", lambda: True)
        patch.setattr(mm, "get_no_ram_reserve", lambda: False)
        assert (
            memory_state_satisfies_settings(
                state = (False, False),
                policy_active = False,
                mlock_applicable = False,
            )
            is True
        )


# ----------------------------------------------------------------------------- #
# The control: the branch the AMD CI runner exercises for real
# ----------------------------------------------------------------------------- #
def test_a_unified_memory_apu_does_lock(apu_host, monkeypatch):
    """On gfx1151 the "VRAM" is system RAM, so a full offload IS host resident and
    the same toggle pair emits the lock. This is the assertion the live gfx1151 run
    corroborates; a divergence between the two is informative."""
    host_resident = _gate()
    assert host_resident is True
    managed, _extras = _policy(monkeypatch, True, False, [], host_resident = host_resident)
    assert managed == ["--load-mode", "mmap+mlock"]


def test_no_ram_reserve_wins_over_keep_resident_on_the_apu(apu_host, monkeypatch):
    """Both on: mlock is itself a full RAM reservation, so no-reserve takes it away
    even where it would otherwise apply."""
    managed, _extras = _policy(monkeypatch, True, True, [], host_resident = _gate())
    assert managed == []


# ----------------------------------------------------------------------------- #
# #9551 on the same two cards
# ----------------------------------------------------------------------------- #
def test_the_displayed_total_is_the_driver_total_untouched(reporter_host):
    """No budget fraction, reserve or headroom is applied to the TOTAL.

    #9551 reports the W7900 as "45" rather than 48. Whatever that is, it is not
    Studio subtracting anything: the number published for the System tab is the
    driver's own total divided by 1024**3, and 46080 MiB is 45.0 GiB exactly.
    """
    devices, _aggregate = hw._rocm_windows_per_device_vram([0, 1])
    assert devices, "the Windows ROCm per-device probe returned nothing"
    totals = {int(dev["index"]): dev["total_gb"] for dev in devices}
    # The only transform is bytes -> GiB, rounded to 2 dp for display.
    assert totals[0] == round(W7900["total"] / (1024**3), 2)
    assert totals[1] == round(W7500["total"] / (1024**3), 2)
    # 45.0 and 7.98 -- the two figures in the reporter's screenshots.
    assert totals[0] == pytest.approx(45.0, abs = 0.01)
    assert totals[1] == pytest.approx(7.98, abs = 0.01)
