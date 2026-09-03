# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from models.inference import (
    DiffusionLoadRequest,
    DiffusionStatusResponse,
    VideoLoadRequest,
    VideoStatusResponse,
)


def test_media_requests_and_statuses_keep_load_and_display_identities_separate():
    snapshot = "/cache/models--Org--Opaque/snapshots/abc"
    logical = "Org/Opaque"

    for request_type in (DiffusionLoadRequest, VideoLoadRequest):
        request = request_type(model_path = snapshot, display_repo_id = logical)
        assert request.model_path == snapshot
        assert request.display_repo_id == logical

    for status_type in (DiffusionStatusResponse, VideoStatusResponse):
        status = status_type(loaded = True, repo_id = snapshot, display_repo_id = logical)
        assert status.repo_id == snapshot
        assert status.display_repo_id == logical
