"""Reingest endpoint tests (Backfill UX).

Full end-to-end reingest needs a running studio + a real embedder; that's
covered manually via the curl smoke flow in the plan. Here we cover the
parts that are testable without external models: payload validation and
the (multimodal, late) constraint propagation.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STUDIO_BACKEND = REPO_ROOT / "studio" / "backend"
if str(STUDIO_BACKEND) not in sys.path:
    sys.path.insert(0, str(STUDIO_BACKEND))


def test_reingest_request_accepts_all_optional_fields():
    from routes.rag import ReingestKBRequest

    empty = ReingestKBRequest()
    assert empty.chunking_strategy is None
    assert empty.mode is None
    assert empty.embedding_model is None

    partial = ReingestKBRequest(chunking_strategy = "late")
    assert partial.chunking_strategy == "late"
    assert partial.mode is None


def test_reingest_request_rejects_unknown_strategy():
    from routes.rag import ReingestKBRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReingestKBRequest(chunking_strategy = "telekinetic")


def test_reingest_request_rejects_unknown_mode():
    from routes.rag import ReingestKBRequest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ReingestKBRequest(mode = "augmented")


def test_constraint_still_enforced_for_reingest_combos():
    """The combination guard is shared with create — verify it still bites."""
    from fastapi import HTTPException

    from routes.rag import _validate_mode_combo

    with pytest.raises(HTTPException) as excinfo:
        _validate_mode_combo("multimodal", "late")
    assert excinfo.value.status_code == 400
