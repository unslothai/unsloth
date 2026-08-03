# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A deleted upload must report itself, not fail as a HuggingFace repo lookup."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402


def test_missing_local_dataset_file_reports_404():
    from utils.paths import dataset_uploads_root

    missing = dataset_uploads_root() / "deleted-upload-test-fixture.jsonl"
    assert not missing.exists()
    request = datasets_route.CheckFormatRequest(dataset_name = str(missing))

    with pytest.raises(HTTPException) as exc:
        datasets_route.check_format(request, current_subject = "test")

    assert exc.value.status_code == 404
    assert "no longer exists" in exc.value.detail


def test_hf_repo_id_is_not_treated_as_a_missing_file():
    request = datasets_route.CheckFormatRequest(dataset_name = "unsloth/does-not-exist")

    with pytest.raises(HTTPException) as exc:
        datasets_route.check_format(request, current_subject = "test")

    assert exc.value.status_code != 404
