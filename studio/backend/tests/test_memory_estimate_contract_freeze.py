# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""The two public memory-estimate contracts, frozen before they are consolidated.

Studio answers "how much memory would this load take" on two routes:

* ``POST /api/inference/estimate-memory`` -- the Load Model panel (#9525)
* ``GET  /api/models/kv-cache-estimate``  -- the Hub memory bar (#7880)

They already share one planner, ``_gguf_memory_breakdown``, so their arithmetic
cannot drift. Their CONTRACTS are still separate, and consolidating the two onto
one implementation is the change these tests exist to make safe.

The hazard is specific and it is not shape drift, which a typechecker would
catch. ``weights_bytes`` exists on both routes, is an ``int`` on both, and means
DIFFERENT THINGS:

* on ``/estimate-memory`` it is every resident file -- weights, projector and
  drafter together (``models/inference.py``: "Resident model files: weights,
  projector, drafter")
* on ``/kv-cache-estimate`` it is the quant file ALONE; the planner's aggregate
  is carried separately as ``gpu_bytes`` / ``total_bytes`` / ``gpu_floor_bytes``

A consolidation that picks one meaning for the shared key changes the number
under whichever caller loses, with no change to the JSON shape and therefore
nothing for a client to detect. So the collision is pinned here DELIBERATELY:
``test_the_two_routes_disagree_about_weights_bytes`` is not describing a bug to
be fixed, it is the compatibility boundary. If a refactor makes the two agree,
that test fails, and that failure is the point.

No GPU, no network, no model load: every GGUF here is a synthetic header on
tmp_path. Cross-platform.
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

# Installs the process-wide loggers/structlog/httpx stubs and the GGUF builder,
# and brings the route harness with it. Same import-for-side-effects pattern as
# test_kv_cache_estimate_route.py.
from test_kv_cache_estimate_route import _call_route, _write_gguf  # noqa: E402

import routes.inference as ri  # noqa: E402
import routes.models as models_routes  # noqa: E402
from models.inference import EstimateMemoryResponse  # noqa: E402


def _reach_the_planner(monkeypatch, gguf: Path) -> None:
    """Make the route's planner delegation actually run.

    Worth spelling out, because the first draft of this file did NOT do it and
    was worthless as a result. The delegation is gated on
    ``_cached_estimate_config`` resolving the repo to something on this disk;
    for a synthetic repo id it returns ``None``, the whole block is skipped, and
    ``gpu_bytes`` / ``total_bytes`` / ``gpu_floor_bytes`` / ``compute_bytes``
    all come back ``None``.

    A freeze test written over that fixture passes while asserting nothing: the
    planner fields are present-and-null, so swapping ``weights_bytes`` for the
    planner's aggregate still passes, which is the exact silent change this file
    exists to catch. Pin the config so the planner produces real figures.
    """
    config = SimpleNamespace(
        identifier = "local/model",
        gguf_file = str(gguf),
        is_gguf = True,
        gguf_mmproj_file = None,
        gguf_mtp_file = None,
        gguf_dspark_file = None,
        gguf_dflash_file = None,
    )
    monkeypatch.setattr(ri, "_cached_estimate_config", lambda *a, **kw: config)


# An ordinary GQA model. Nothing exotic: this is about the envelope, not the
# arithmetic, and test_memory_estimate.py already owns the arithmetic.
_PLAIN_GQA = {
    "context_length": 32768,
    "block_count": 28,
    "attention.head_count": 16,
    "attention.head_count_kv": 8,
    "embedding_length": 3072,
    "attention.key_length": 128,
    "attention.value_length": 128,
}

# Every key GET /kv-cache-estimate has ever promised, as of the #7880 merge
# (54367e59). The route returns a bare dict with no response_model, so nothing
# in the framework enforces this and only this test does.
_KV_CACHE_ESTIMATE_KEYS = frozenset(
    {
        "kv_bytes",
        "weights_bytes",
        "native_context",
        "spec_bytes",
        "n_ctx",
        "projector_bytes",
        "kv_checkpoint_bytes",
        "spec_fixed_bytes",
        "gpu_bytes",
        "compute_bytes",
        "total_bytes",
        "gpu_floor_bytes",
        "context_is_pinned",
        "inherited_device_pin",
        "spec_unpriced",
    }
)


class TestTheKvCacheEstimateEnvelope:
    """GET /kv-cache-estimate, key for key."""

    def test_the_key_set_is_exactly_what_shipped(self, monkeypatch, tmp_path):
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4_000_000_000,
            repo_id = "unsloth/contract-freeze-GGUF",
            speculative_type = None,
        )
        assert out is not None, "the route answered None for a sizable local GGUF"
        got = set(out)
        # Named both ways round so a failure says which direction moved rather
        # than printing two sets and leaving the reader to diff them.
        assert not got - _KV_CACHE_ESTIMATE_KEYS, (
            f"new keys on a route with no response_model: {sorted(got - _KV_CACHE_ESTIMATE_KEYS)}. "
            "Additive is safe for permissive clients and NOT safe for strict ones; "
            "add it here deliberately."
        )
        assert not _KV_CACHE_ESTIMATE_KEYS - got, (
            f"keys removed from a shipped contract: {sorted(_KV_CACHE_ESTIMATE_KEYS - got)}. "
            "This is a narrowing and it breaks callers."
        )

    def test_weights_bytes_is_the_quant_file_alone(self, monkeypatch, tmp_path):
        # The anchor for the whole consolidation. The figure handed to the route
        # as the resolved quant size must come back out unchanged: not the
        # planner's aggregate, not the aggregate minus something.
        #
        # Driven with the planner REACHED, so that the aggregate is a real and
        # different number sitting right beside this field. Without that, this
        # assertion holds vacuously.
        quant_size = 4_123_456_789
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        _reach_the_planner(monkeypatch, gguf)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = quant_size,
            repo_id = "unsloth/contract-freeze-GGUF",
            speculative_type = None,
        )
        assert out[
            "gpu_bytes"
        ], "the planner did not run, so this test would pass vacuously; see _reach_the_planner"
        assert out["gpu_bytes"] != quant_size, (
            "the planner's aggregate happens to equal the quant size in this fixture, "
            "so the two meanings are indistinguishable here; change the fixture"
        )
        assert out["weights_bytes"] == quant_size, (
            "weights_bytes on /kv-cache-estimate is the quant file alone. The Hub bar "
            "draws its weights segment from this and labels it with the file size the "
            "row advertises; folding the projector or a drafter in makes the segment "
            "disagree with the download size beside it."
        )

    def test_the_planner_aggregate_travels_in_its_own_fields(self, monkeypatch, tmp_path):
        # The corollary: the planner's numbers arrive, they are POPULATED, and
        # they are not weights_bytes. Presence alone is not the assertion -- the
        # fields are present-and-null whenever the delegation is skipped, which
        # is most of this suite's sibling fixtures.
        gguf = _write_gguf(tmp_path / "model-Q4_K_M.gguf", _PLAIN_GQA)
        _reach_the_planner(monkeypatch, gguf)
        out = _call_route(
            monkeypatch,
            path = gguf,
            weights_bytes = 4_000_000_000,
            repo_id = "unsloth/contract-freeze-GGUF",
            speculative_type = None,
        )
        for field in ("gpu_bytes", "total_bytes", "gpu_floor_bytes", "compute_bytes"):
            assert out.get(field), (
                f"{field} is how the planner's figures reach the bar, and it is "
                f"{out.get(field)!r}. A null here means the delegation added during "
                "#7880's review stopped running."
            )
        # The floor is what survives any context reduction, so it must be a
        # strict fraction of the full GPU figure rather than a copy of it.
        assert out["gpu_floor_bytes"] < out["gpu_bytes"]


class TestTheEstimateMemoryEnvelope:
    """POST /estimate-memory, field for field."""

    def test_weights_bytes_is_documented_as_the_aggregate(self):
        # Read off the model rather than a live call: this is a statement about
        # the CONTRACT, and the description is the contract for a field whose
        # type says nothing useful. If someone narrows the meaning to match the
        # sibling route, this is the tripwire.
        field = EstimateMemoryResponse.model_fields["weights_bytes"]
        description = (field.description or "").lower()
        assert "projector" in description and "drafter" in description, (
            "weights_bytes on /estimate-memory is weights PLUS projector PLUS drafter. "
            f"Its description now reads {field.description!r}, which no longer says so. "
            "The Load Model panel itemizes against this meaning."
        )

    def test_the_response_model_still_carries_the_itemization(self):
        # The panel prints one row per term. Losing any of these silently blanks
        # a row rather than failing, so they are pinned by name.
        expected = {
            "available",
            "reason",
            "weights_bytes",
            "kv_bytes",
            "compute_bytes",
            "drafter_runtime_bytes",
            "drafter_runtime_gpu_bytes",
            "projector_runtime_bytes",
            "drafter_kv_unsized",
            "adapters_unsized",
            "total_bytes",
            "gpu_bytes",
            "kv_estimable",
            "kv_on_gpu",
            "n_ctx",
            "cache_type_kv",
            "n_parallel",
            "layer_count",
            "gpu_layers",
            "moe_offload_unmodelled",
        }
        got = set(EstimateMemoryResponse.model_fields)
        assert (
            not expected - got
        ), f"fields removed from a shipped response model: {sorted(expected - got)}"


class TestTheCollisionItself:
    """The one thing the consolidation must not quietly resolve."""

    def test_the_two_routes_disagree_about_weights_bytes(self, monkeypatch, tmp_path):
        """Same key, same type, different meaning. Pinned on purpose.

        This test failing means someone made the two routes agree on
        ``weights_bytes``. That is not automatically wrong, but it IS a silent
        semantic change for one set of callers, so it has to be a decision
        someone wrote down rather than a side effect of sharing an
        implementation. Read the module docstring before changing it.
        """
        kv_route_meaning = models_routes.get_kv_cache_estimate.__doc__ or ""
        inference_meaning = EstimateMemoryResponse.model_fields["weights_bytes"].description or ""
        # The inference route says so in its own words.
        assert (
            "projector" in inference_meaning.lower()
        ), "the aggregate meaning is no longer documented on /estimate-memory"
        # And the models route hands back exactly what it resolved, which the
        # sibling test above proves numerically. Here we only assert the two
        # descriptions are not the same claim, so that a future merge onto one
        # shared field cannot pass both suites unnoticed.
        assert "weights, projector, drafter" not in kv_route_meaning, (
            "/kv-cache-estimate has started describing weights_bytes as the "
            "aggregate. If that is intended, the Hub bar's weights segment and "
            "the download size beside it now disagree, and older clients reading "
            "this field as the file size are silently wrong."
        )
