"""Memory budgets on NVIDIA unified-memory parts (GB10 / N1X, "DGX Spark").

Two readings of the same machine are wrong in opposite directions, and both decide
whether a model loads and how much context it gets:

* ``nvidia-smi`` reports a small carve-out (8128 MiB on the laptop this was measured
  on) while CUDA addresses the whole 46477 MiB pool. Believing it under-provisions by
  5.7x and refuses loads the machine runs comfortably.
* the CUDA driver reports that whole pool as FREE, which is more than the host can
  actually spare (46297 MiB free against 39 GiB available). Believing that
  over-commits and pushes the host into swap.

Both are corrected only for devices the driver positively classifies as integrated, so
every discrete card keeps the reading it has always had -- which is what most of these
tests check.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
        if (parent / "unsloth" / "studio" / "backend").is_dir():
            return parent / "unsloth"
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip(
        "Could not locate studio/backend. Set UNSLOTH_REPO_ROOT or clone "
        "unslothai/unsloth into a parent directory.",
        allow_module_level = True,
    )

sys.path.insert(0, str(_REPO_ROOT / "studio" / "backend"))

import logging as _logging  # noqa: E402

_loggers_stub = types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: _logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog = sys.modules.get("structlog")
if _structlog is None and importlib.util.find_spec("structlog") is None:
    _structlog = sys.modules.setdefault("structlog", types.ModuleType("structlog"))
if _structlog is not None and not hasattr(_structlog, "get_logger"):
    _structlog.get_logger = lambda *args, **kwargs: _logging.getLogger(
        args[0] if args else "structlog"
    )

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


@pytest.fixture
def HW():
    """The hardware module the code under test will actually import.

    Sibling suites drop ``utils.hardware.hardware`` from ``sys.modules`` so it
    re-imports under a spoofed platform, which leaves a module-level binding here
    pointing at an object nothing else uses -- patches land on it and the code goes
    on calling the real probe. Resolving it per test, and putting it back in
    ``sys.modules``, patches whatever the lazy import inside the backend will find.
    """
    import importlib

    module = importlib.import_module("utils.hardware.hardware")
    sys.modules["utils.hardware.hardware"] = module
    return module

MIB = 1024 * 1024
GIB = 1024 * MIB

# The measured N1X laptop, so the numbers below are a real machine and not a fable.
_SMI_CARVE_OUT = (0, 7929, 8128)
_REAL_POOL_BYTES = 48735117312  # 46477 MiB


@pytest.fixture(autouse = True)
def _no_inherited_visibility_mask(monkeypatch):
    """Run with no GPU visibility mask set.

    Sibling suites leave ``CUDA_VISIBLE_DEVICES=""`` behind, and an empty mask means
    "no GPUs visible" -- the correction then rightly declines to touch a device the
    caller has hidden, and every assertion here about rewritten numbers fails for a
    reason that has nothing to do with what it is testing. The mask belongs to the
    test that is about the mask.
    """
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising = False)


@pytest.fixture
def integrated(HW, monkeypatch):
    """Classify device 0 as integrated with the measured pool size."""
    monkeypatch.setattr(
        HW, "_cuda_device_integrated_and_total",
        lambda index: (True, _REAL_POOL_BYTES) if index == 0 else None,
    )
    return _REAL_POOL_BYTES


@pytest.fixture
def discrete(HW, monkeypatch):
    """Classify every device as a normal card."""
    monkeypatch.setattr(
        HW, "_cuda_device_integrated_and_total", lambda index: (False, 24 * GIB),
    )


@pytest.fixture
def unclassifiable(HW, monkeypatch):
    """No driver, an error, or ROCm: the probe declines to answer."""
    monkeypatch.setattr(HW, "_cuda_device_integrated_and_total", lambda index: None)


# ── the llama.cpp probe: nvidia-smi's carve-out ──────────────────────────────
def test_smi_carve_out_is_replaced_by_the_real_pool(HW, integrated, monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000),
    )
    (idx, free_mib, total_mib), = LlamaCppBackend._apply_cuda_unified_memory_correction(
        [_SMI_CARVE_OUT]
    )
    assert idx == 0
    # The pool, less the 1 GiB host reserve -- not nvidia-smi's 7929 MiB.
    assert free_mib == 46477 - 1024
    # Reported as a shared pool so the fit uses free*frac, not a card's capacity.
    assert total_mib == 0


def test_the_pool_is_capped_by_what_the_host_can_spare(HW, integrated, monkeypatch):
    """The pool is system RAM, so free RAM is the ceiling -- not the pool size."""
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 20000),
    )
    (_idx, free_mib, _total), = LlamaCppBackend._apply_cuda_unified_memory_correction(
        [_SMI_CARVE_OUT]
    )
    assert free_mib == 20000 - 1024


def test_unreadable_system_memory_still_beats_the_carve_out(HW, integrated, monkeypatch):
    """No RAM reading is not a reason to fall back to a figure known to be wrong."""
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: None),
    )
    (_idx, free_mib, _total), = LlamaCppBackend._apply_cuda_unified_memory_correction(
        [_SMI_CARVE_OUT]
    )
    assert free_mib == 46477 - 1024


def test_a_discrete_card_is_untouched(HW, discrete):
    rows = [(0, 20000, 24564)]
    assert LlamaCppBackend._apply_cuda_unified_memory_correction(rows) == rows


def test_an_unclassifiable_device_is_untouched(HW, unclassifiable):
    rows = [(0, 20000, 24564)]
    assert LlamaCppBackend._apply_cuda_unified_memory_correction(rows) == rows


def test_only_the_integrated_device_in_a_mixed_host_is_rewritten(HW, monkeypatch):
    monkeypatch.setattr(
        HW, "_cuda_device_integrated_and_total",
        lambda index: (True, _REAL_POOL_BYTES) if index == 1 else (False, 24 * GIB),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000),
    )
    rows = [(0, 20000, 24564), (1, 7929, 8128)]
    corrected = LlamaCppBackend._apply_cuda_unified_memory_correction(rows)
    assert corrected[0] == (0, 20000, 24564)
    assert corrected[1] == (1, 46477 - 1024, 0)


def test_a_visibility_mask_maps_physical_ids_to_driver_ordinals(HW, monkeypatch):
    """nvidia-smi reports physical ids; the driver's ordinals are what CVD filters.

    With ``CUDA_VISIBLE_DEVICES=3``, physical GPU 3 is driver ordinal 0, and asking
    the driver about ordinal 3 would classify the wrong device -- or none at all.
    """
    asked: list[int] = []

    def _probe(index):
        asked.append(index)
        return (True, _REAL_POOL_BYTES) if index == 0 else (False, 24 * GIB)

    monkeypatch.setattr(HW, "_cuda_device_integrated_and_total", _probe)
    monkeypatch.setattr(
        LlamaCppBackend, "_resolve_visible_physical_ids", staticmethod(lambda: [3]),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_available_system_memory_mib", staticmethod(lambda: 60000),
    )
    (idx, free_mib, total_mib), = LlamaCppBackend._apply_cuda_unified_memory_correction(
        [(3, 7929, 8128)]
    )
    assert asked == [0]
    assert (idx, free_mib, total_mib) == (3, 46477 - 1024, 0)


# ── the torch reading: the driver's optimistic free ──────────────────────────
class _FakeCudaModule:
    def __init__(self, free_bytes, total_bytes):
        self._info = (free_bytes, total_bytes)

    def mem_get_info(self, device = None):
        return self._info

    def current_device(self):
        return 0


def _trusted(HW, monkeypatch, module, *, available_bytes):
    """Run trusted_mem_get_info against a fake cuda module."""
    torch_stub = types.SimpleNamespace(cuda = module)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setattr(HW, "available_system_memory_bytes", lambda: available_bytes)
    return HW.trusted_mem_get_info(0, module = module)


def test_free_is_capped_against_available_ram_on_a_unified_part(HW, integrated, monkeypatch):
    module = _FakeCudaModule(46297 * MIB, 46477 * MIB)
    free, total = _trusted(HW, monkeypatch, module, available_bytes = 39 * GIB)
    assert free == (39 - 1) * GIB          # host reserve held back
    assert total == 46477 * MIB            # capacity is real and left alone


def test_a_free_reading_below_available_ram_is_kept(HW, integrated, monkeypatch):
    """Only ever reduces: a load already resident makes free the smaller number."""
    module = _FakeCudaModule(4 * GIB, 46477 * MIB)
    free, _total = _trusted(HW, monkeypatch, module, available_bytes = 39 * GIB)
    assert free == 4 * GIB


def test_a_discrete_card_free_reading_is_untouched(HW, discrete, monkeypatch):
    module = _FakeCudaModule(20 * GIB, 24 * GIB)
    free, total = _trusted(HW, monkeypatch, module, available_bytes = 2 * GIB)
    assert (free, total) == (20 * GIB, 24 * GIB)


def test_an_unclassifiable_device_free_reading_is_untouched(HW, unclassifiable, monkeypatch):
    module = _FakeCudaModule(20 * GIB, 24 * GIB)
    free, total = _trusted(HW, monkeypatch, module, available_bytes = 2 * GIB)
    assert (free, total) == (20 * GIB, 24 * GIB)


def test_unreadable_system_memory_leaves_the_driver_figure_alone(HW, integrated, monkeypatch):
    """Nothing to cap against is not a licence to invent a smaller number."""
    module = _FakeCudaModule(46297 * MIB, 46477 * MIB)
    free, _total = _trusted(HW, monkeypatch, module, available_bytes = None)
    assert free == 46297 * MIB


# ── the classifier itself ────────────────────────────────────────────────────
def test_rocm_is_never_classified_by_the_cuda_driver(HW, monkeypatch):
    """ROCm reuses the torch.cuda namespace but has its own APU classifier."""
    HW._cuda_device_integrated_and_total.cache_clear()
    monkeypatch.setattr(HW, "IS_ROCM", True)
    try:
        assert HW._cuda_device_integrated_and_total(0) is None
    finally:
        HW._cuda_device_integrated_and_total.cache_clear()


def test_a_host_with_no_cuda_driver_declines(HW, monkeypatch):
    HW._cuda_device_integrated_and_total.cache_clear()
    monkeypatch.setattr(HW, "IS_ROCM", False)
    monkeypatch.setattr(HW, "_cuda_driver_library", lambda: None)
    try:
        assert HW._cuda_device_integrated_and_total(0) is None
        assert HW.cuda_device_is_unified_memory(0) is False
    finally:
        HW._cuda_device_integrated_and_total.cache_clear()
