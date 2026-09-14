# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The shared contract across the platform x GPU-vendor product.

The consolidation is pure arithmetic and renaming, with no platform-specific
branch anywhere in it, which is precisely the claim worth testing rather than
asserting: a contract that quietly depends on the host is the kind of thing that
is only discovered by the one user who has that host.

The matrix is the four platform keys the repo already parametrises over
(``test_diffusion_predownload_guard_platforms.py``) crossed with the placements
a GGUF load can end up in: everything on one card, split across two, partly on
the host, entirely on the host, and nothing probed at all.

Two properties hold in every cell:

* the WIRE SHAPE of both legacy routes is identical everywhere, so no client
  needs a per-platform branch
* ``weights_bytes`` keeps its own meaning on each route in every cell, which is
  the compatibility boundary the whole consolidation rests on

No GPU, no network, no model load. Pure functions.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import test_kv_cache_estimation  # noqa: E402,F401 -- process-wide stubs

import pytest  # noqa: E402

from core.inference.memory_contract import (  # noqa: E402
    EMPTY_BREAKDOWN,
    build_memory_estimate,
    project_estimate_memory_response,
    project_kv_cache_estimate,
)
from models.inference import EstimateMemoryResponse  # noqa: E402
from test_memory_estimate_contract_freeze import _KV_CACHE_ESTIMATE_KEYS  # noqa: E402

PLATFORMS = ("linux", "wsl", "win32", "darwin")

# The vendor decides where bytes LAND, which is the only thing that varies here.
# Named for the host they model so a failure says which machine it is about.
PLACEMENTS = {
    "nvidia-single": {"gpu": 8_100_000_000, "total": 8_700_000_000},
    "nvidia-dual": {"gpu": 8_700_000_000, "total": 8_700_000_000},
    "amd-rocm-discrete": {"gpu": 6_000_000_000, "total": 8_700_000_000},
    # An APU or Apple part: one pool, so everything is "on the GPU" and also all
    # of it is host memory. The route reports the placement, not the topology.
    "unified-apu": {"gpu": 8_700_000_000, "total": 8_700_000_000},
    # --n-cpu-moe or a layer split: some of it is off the card.
    "partial-offload": {"gpu": 3_000_000_000, "total": 8_700_000_000},
    # LLAMA_ARG_DEVICE=none. Zero is a REAL answer here, not a missing one.
    "cpu-only": {"gpu": 0, "total": 8_700_000_000},
}

_QUANT_FILE_BYTES = 4_100_000_000
_RESIDENT_FILES_BYTES = 5_000_000_000


def _breakdown(gpu: int, total: int) -> SimpleNamespace:
    return SimpleNamespace(
        weights_bytes = _RESIDENT_FILES_BYTES,
        kv_bytes = 3_000_000_000,
        compute_bytes = 700_000_000,
        drafter_runtime_bytes = 0,
        drafter_runtime_gpu_bytes = 0,
        projector_runtime_bytes = 0,
        drafter_kv_unsized = False,
        adapters_unsized = False,
        total_bytes = total,
        gpu_bytes = gpu,
        kv_estimable = True,
        kv_on_gpu = gpu > 0,
        n_ctx = 32768,
        cache_type_kv = "f16",
        n_parallel = 1,
        layer_count = 28,
        gpu_layers = 28 if gpu > 0 else 0,
    )


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("placement", sorted(PLACEMENTS))
class TestTheContractIsPlatformIndependent:
    def test_both_wire_shapes_are_identical_everywhere(self, platform, placement, monkeypatch):
        monkeypatch.setattr(sys, "platform", platform, raising = False)
        p = PLACEMENTS[placement]
        est = build_memory_estimate(
            _breakdown(p["gpu"], p["total"]), quant_file_bytes = _QUANT_FILE_BYTES
        )
        assert set(project_kv_cache_estimate(est)) == set(
            _KV_CACHE_ESTIMATE_KEYS
        ), f"{platform}/{placement}: the models route's key set moved"
        assert set(project_estimate_memory_response(est)) == set(
            EstimateMemoryResponse.model_fields
        ), f"{platform}/{placement}: the inference route's field set moved"

    def test_the_two_meanings_stay_apart_everywhere(self, platform, placement, monkeypatch):
        monkeypatch.setattr(sys, "platform", platform, raising = False)
        p = PLACEMENTS[placement]
        est = build_memory_estimate(
            _breakdown(p["gpu"], p["total"]), quant_file_bytes = _QUANT_FILE_BYTES
        )
        panel = project_estimate_memory_response(est)
        bar = project_kv_cache_estimate(est)
        assert panel["weights_bytes"] == _RESIDENT_FILES_BYTES
        assert bar["weights_bytes"] == _QUANT_FILE_BYTES
        assert panel["weights_bytes"] != bar["weights_bytes"], (
            f"{platform}/{placement}: the two routes agreed on weights_bytes, which "
            "silently changes the number under one set of callers"
        )

    def test_a_cpu_only_launch_reports_zero_rather_than_nothing(
        self, platform, placement, monkeypatch
    ):
        monkeypatch.setattr(sys, "platform", platform, raising = False)
        if placement != "cpu-only":
            pytest.skip("only the CPU-only placement asserts this")
        est = build_memory_estimate(
            _breakdown(0, 8_700_000_000), quant_file_bytes = _QUANT_FILE_BYTES
        )
        bar = project_kv_cache_estimate(est)
        assert bar["gpu_bytes"] == 0, (
            f"{platform}: a launch that touches no card reported {bar['gpu_bytes']!r} "
            "instead of 0. None means 'the planner never ran', and a caller that "
            "cannot tell them apart draws VRAM pressure for a CPU load."
        )
        assert bar["gpu_bytes"] is not None


@pytest.mark.parametrize("platform", PLATFORMS)
class TestTheAbsentPlannerIsTheSameEverywhere:
    def test_a_planner_that_never_ran_is_null_not_zero(self, platform, monkeypatch):
        monkeypatch.setattr(sys, "platform", platform, raising = False)
        est = build_memory_estimate(EMPTY_BREAKDOWN, quant_file_bytes = _QUANT_FILE_BYTES)
        bar = project_kv_cache_estimate(est)
        assert bar["gpu_bytes"] is None, (
            f"{platform}: an absent planner reported {bar['gpu_bytes']!r}. 0 would mean "
            "'measured, nothing on the card', which is a different claim."
        )
        # The quant size is known independently of the planner, so it survives.
        assert bar["weights_bytes"] == _QUANT_FILE_BYTES
        # And the shape is still complete, so a caller needs no branch for it.
        assert set(bar) == set(_KV_CACHE_ESTIMATE_KEYS)


class TestOldClientsAreUnaffected:
    """Forwards compatibility: what a client written before this PR still sees."""

    def test_every_field_an_old_client_read_is_still_present_and_typed(self):
        # The fields #7880's frontend reads off /kv-cache-estimate, by name, as a
        # stand-in for any third-party client pinned to that shape.
        est = build_memory_estimate(
            _breakdown(8_100_000_000, 8_700_000_000), quant_file_bytes = _QUANT_FILE_BYTES
        )
        bar = project_kv_cache_estimate(
            est, kv_bytes = 3_000_000_000, spec_bytes = None, projector_bytes = None
        )
        for name in (
            "kv_bytes",
            "weights_bytes",
            "native_context",
            "spec_bytes",
            "n_ctx",
            "gpu_bytes",
            "compute_bytes",
            "total_bytes",
            "gpu_floor_bytes",
            "context_is_pinned",
            "inherited_device_pin",
            "spec_unpriced",
        ):
            assert name in bar, f"{name} disappeared from a shipped contract"
        for name in ("kv_bytes", "weights_bytes", "n_ctx"):
            assert bar[name] is None or isinstance(
                bar[name], int
            ), f"{name} changed type, which breaks a strict deserializer"
        for name in ("context_is_pinned", "inherited_device_pin", "spec_unpriced"):
            assert isinstance(bar[name], bool)

    def test_the_canonical_model_never_grows_the_ambiguous_name(self):
        from models.inference import MemoryEstimate
        assert "weights_bytes" not in MemoryEstimate.model_fields, (
            "MemoryEstimate has grown a weights_bytes field. That name means two "
            "different things on the two legacy routes and belongs on neither."
        )
