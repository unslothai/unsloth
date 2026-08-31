# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The canonical MemoryEstimate and the two legacy shapes projected from it.

``test_memory_estimate_contract_freeze.py`` pins what the two ROUTES emit today.
This file pins the shared layer they are being moved onto: that the canonical
model separates the two meanings ``weights_bytes`` used to carry, and that each
projection puts the right one back on the wire.

The single most important assertion here is
``test_the_two_projections_disagree_about_weights_bytes``. The whole point of
routing both surfaces through one model is that their arithmetic cannot drift;
the whole point of projecting back out is that their CONTRACTS still differ
where they always did. Getting the first without the second is a silent
regression for every existing caller of one route.

Pure functions, no I/O.
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

# Imported for its side effects: installs the process-wide loggers/structlog/httpx
# stubs that models.inference needs on a runner without the real packages. Same
# pattern as test_kv_cache_estimate_route.py.
import test_kv_cache_estimation  # noqa: E402,F401

import pytest  # noqa: E402

from core.inference.memory_contract import (  # noqa: E402
    build_memory_estimate,
    project_estimate_memory_response,
    project_kv_cache_estimate,
)

# A planner breakdown where every term is a distinct number, so a projection
# that reads the wrong field cannot coincidentally look right.
_BREAKDOWN = SimpleNamespace(
    weights_bytes = 5_000_000_000,  # resident: quant + projector + drafter
    kv_bytes = 3_000_000_000,
    compute_bytes = 700_000_000,
    drafter_runtime_bytes = 400_000_000,
    drafter_runtime_gpu_bytes = 250_000_000,
    projector_runtime_bytes = 120_000_000,
    drafter_kv_unsized = False,
    adapters_unsized = False,
    total_bytes = 8_700_000_000,
    gpu_bytes = 8_100_000_000,
    kv_estimable = True,
    kv_on_gpu = True,
    n_ctx = 32768,
    cache_type_kv = "f16",
    n_parallel = 4,
    layer_count = 28,
    gpu_layers = 28,
)

# The quant file alone: strictly smaller than the resident total above, which is
# the relationship the two fields exist to express.
_QUANT_FILE_BYTES = 4_100_000_000


@pytest.fixture
def estimate():
    return build_memory_estimate(
        _BREAKDOWN,
        quant_file_bytes = _QUANT_FILE_BYTES,
        native_context = 131072,
        gpu_floor_bytes = 900_000_000,
        context_is_pinned = False,
        inherited_device_pin = True,
        spec_unpriced = True,
    )


class TestTheCanonicalModel:
    def test_the_ambiguous_name_is_gone(self, estimate):
        # Absent rather than redefined: a field named weights_bytes on the shared
        # model would be picked up and guessed at by exactly the readers this
        # separation exists to protect.
        assert not hasattr(estimate, "weights_bytes"), (
            "MemoryEstimate has grown a weights_bytes field. That name means two "
            "different things on the two legacy routes and belongs on neither the "
            "shared model nor any new caller."
        )

    def test_the_two_meanings_are_separate_and_ordered(self, estimate):
        assert estimate.quant_file_bytes == _QUANT_FILE_BYTES
        assert estimate.resident_files_bytes == _BREAKDOWN.weights_bytes
        # The quant file is one of the resident files, so this ordering is a
        # property of the model rather than of this fixture.
        assert estimate.quant_file_bytes < estimate.resident_files_bytes

    def test_the_planner_weights_field_is_the_resident_meaning(self, estimate):
        # The planner's own field name is weights_bytes and it carries the
        # AGGREGATE. Mapping it to quant_file_bytes would be the easy mistake.
        assert estimate.resident_files_bytes == _BREAKDOWN.weights_bytes
        assert estimate.quant_file_bytes != _BREAKDOWN.weights_bytes

    def test_the_two_figures_are_reported_as_given(self):
        """Neither figure is adjusted to make the other look consistent.

        The quant file is by definition one of the resident files, so a quant
        larger than the resident total is impossible in production. It is still
        reported unchanged, because the two numbers come from different places
        and "fixing" one against the other replaces a caller's real value with
        an unrelated one rather than catching anything. An earlier draft clamped
        here and truncated a 4.1 GB quant to a 373 byte synthetic header.
        """
        out = build_memory_estimate(_BREAKDOWN, quant_file_bytes = 9_999_999_999)
        assert out.quant_file_bytes == 9_999_999_999
        assert out.resident_files_bytes == _BREAKDOWN.weights_bytes

    def test_optional_terms_default_without_being_invented(self):
        out = build_memory_estimate(_BREAKDOWN, quant_file_bytes = _QUANT_FILE_BYTES)
        # None means "not computed" and must not become 0, which is a real value
        # meaning "nothing is pinned to the card at the shortest context".
        assert out.gpu_floor_bytes is None
        assert out.native_context is None


class TestTheRouteOverrides:
    """The figures /kv-cache-estimate computes itself, passed in not assigned on."""

    def test_gpu_bytes_carries_three_distinct_states(self):
        # A number, a real None (planner never ran), and "use the breakdown's".
        # None cannot express the third, which is why there is a sentinel.
        assert build_memory_estimate(_BREAKDOWN, quant_file_bytes = 1, gpu_bytes = 0).gpu_bytes == 0
        assert (
            build_memory_estimate(_BREAKDOWN, quant_file_bytes = 1, gpu_bytes = None).gpu_bytes is None
        )
        # Omitted entirely: falls through to whatever the breakdown had.
        assert (
            build_memory_estimate(_BREAKDOWN, quant_file_bytes = 1).gpu_bytes == _BREAKDOWN.gpu_bytes
        )

    def test_the_other_overrides_apply(self):
        out = build_memory_estimate(
            _BREAKDOWN,
            quant_file_bytes = 1,
            compute_bytes = 11,
            total_bytes = 22,
            n_ctx = 33,
        )
        assert (out.compute_bytes, out.total_bytes, out.n_ctx) == (11, 22, 33)
        # And None collapses to 0 for the three that are declared non-optional,
        # matching what the route's `or None` handling produced before.
        none_out = build_memory_estimate(
            _BREAKDOWN,
            quant_file_bytes = 1,
            compute_bytes = None,
            total_bytes = None,
            n_ctx = None,
        )
        assert (none_out.compute_bytes, none_out.total_bytes, none_out.n_ctx) == (0, 0, 0)

    def test_the_route_does_not_mutate_the_model_after_building_it(self):
        """Pydantic does not validate assignment, so a post-build write is unchecked.

        Measured rather than assumed: on pydantic 2.13,
        ``m.gpu_bytes = "not an int"`` succeeds and puts that string on the wire,
        and a declared ``int`` field accepts None the same way. The route used to
        assign these four fields after construction; this pins that it does not
        go back to doing so.
        """
        from pathlib import Path

        source = (Path(__file__).resolve().parent.parent / "routes" / "models.py").read_text(
            encoding = "utf-8"
        )
        for field in ("gpu_bytes", "compute_bytes", "total_bytes", "n_ctx"):
            assert f"_estimate.{field} =" not in source, (
                f"routes/models.py assigns {field} onto a built MemoryEstimate. "
                "Pass it to build_memory_estimate instead; assignment skips "
                "validation entirely."
            )

    def test_assignment_really_is_unvalidated(self):
        # The premise of the guard above. If a future pydantic or a model_config
        # change makes assignment validate, this fails and the guard can relax.
        from models.inference import MemoryEstimate

        m = MemoryEstimate(available = True)
        m.gpu_bytes = "not an int"
        assert m.gpu_bytes == "not an int", (
            "assignment now validates; the no-mutation guard above is no longer "
            "load-bearing and its comment should be updated"
        )


class TestTheLegacyProjections:
    def test_estimate_memory_gets_the_resident_total(self, estimate):
        out = project_estimate_memory_response(estimate)
        assert (
            out["weights_bytes"] == _BREAKDOWN.weights_bytes
        ), "the Load Model panel itemizes weights_bytes as every resident file"

    def test_kv_cache_estimate_gets_the_quant_file(self, estimate):
        out = project_kv_cache_estimate(estimate)
        assert out["weights_bytes"] == _QUANT_FILE_BYTES, (
            "the Hub bar draws its weights segment from weights_bytes and prints it "
            "beside the download size on the same row"
        )

    def test_the_two_projections_disagree_about_weights_bytes(self, estimate):
        """The compatibility boundary, asserted directly.

        If this fails, the two routes have been made to agree on a key whose
        meaning was never shared, and one set of callers is now silently reading
        a different number through an unchanged JSON shape.
        """
        panel = project_estimate_memory_response(estimate)
        bar = project_kv_cache_estimate(estimate)
        assert panel["weights_bytes"] != bar["weights_bytes"], (
            "both routes now report the same weights_bytes. See the module "
            "docstring: these two meanings are different by design."
        )

    def test_the_kv_projection_keeps_none_rather_than_zero(self):
        # This route uses None for "no such term" throughout, and the frontend's
        # estimateIsUnsized() distinguishes null from 0. Coercing to 0 would make
        # an unsizable model look like a free one.
        empty = SimpleNamespace(**{**_BREAKDOWN.__dict__, "kv_bytes": 0})
        out = project_kv_cache_estimate(
            build_memory_estimate(empty, quant_file_bytes = 0), kv_bytes = 0
        )
        assert out["kv_bytes"] is None
        assert out["weights_bytes"] is None

    def test_a_zero_gpu_share_survives_as_zero(self):
        """The one field where 0 must NOT become None.

        An inherited LLAMA_ARG_DEVICE=none makes the launch entirely CPU
        resident. Folding that into None sends the caller back to summing
        segments and drawing VRAM pressure for a load that touches no card --
        a bug this route already had once and fixed.
        """
        cpu_only = SimpleNamespace(**{**_BREAKDOWN.__dict__, "gpu_bytes": 0})
        out = project_kv_cache_estimate(
            build_memory_estimate(cpu_only, quant_file_bytes = _QUANT_FILE_BYTES),
            kv_bytes = _BREAKDOWN.kv_bytes,
        )
        assert out["gpu_bytes"] == 0, "a real zero GPU share was folded into 'no answer'"

    def test_a_missing_planner_leaves_the_gpu_share_null(self):
        # The other side of the same coin: never ran is not the same as ran and
        # found nothing.
        absent = SimpleNamespace(**{**_BREAKDOWN.__dict__, "gpu_bytes": None})
        out = project_kv_cache_estimate(
            build_memory_estimate(absent, quant_file_bytes = _QUANT_FILE_BYTES),
            kv_bytes = _BREAKDOWN.kv_bytes,
        )
        assert out["gpu_bytes"] is None

    def test_the_kv_projection_does_not_borrow_the_planners_kv(self, estimate):
        # This route prices the target cache itself; the planner's figure is not
        # interchangeable. Passing no kv_bytes must yield None rather than
        # silently substituting the planner's.
        assert project_kv_cache_estimate(estimate)["kv_bytes"] is None

    def test_the_kv_projection_passes_its_own_itemization_through(self, estimate):
        # These four are the models route's own terms; the planner does not model
        # them separately, so they must survive the round trip untouched.
        out = project_kv_cache_estimate(
            estimate,
            spec_bytes = 111,
            spec_fixed_bytes = 22,
            projector_bytes = 333,
            kv_checkpoint_bytes = 44,
        )
        assert (out["spec_bytes"], out["spec_fixed_bytes"]) == (111, 22)
        assert (out["projector_bytes"], out["kv_checkpoint_bytes"]) == (333, 44)

    def test_both_projections_carry_every_key_their_route_promises(self, estimate):
        # Cross-checked against the frozen key sets, so the projections and the
        # freeze cannot drift apart from each other.
        from test_memory_estimate_contract_freeze import _KV_CACHE_ESTIMATE_KEYS

        assert set(project_kv_cache_estimate(estimate)) == set(_KV_CACHE_ESTIMATE_KEYS)

        from models.inference import EstimateMemoryResponse

        assert set(project_estimate_memory_response(estimate)) == set(
            EstimateMemoryResponse.model_fields
        )
