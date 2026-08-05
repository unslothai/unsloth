# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A deleted upload must report itself, without swallowing other failures."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from routes import datasets as datasets_route  # noqa: E402


@pytest.fixture(autouse = True)
def isolated_studio_home(tmp_path, monkeypatch):
    """Keep fixtures out of the developer's real Studio uploads directory."""
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def no_hub(monkeypatch):
    """Fail every Hub lookup in-process, recording what was asked for.

    HF_HUB_OFFLINE is read into huggingface_hub.constants at import time, so
    setting it from a test is a no-op and the lookups still dial out.
    """
    import huggingface_hub

    attempts: list[str] = []

    class _NoHubApi:
        def __init__(self, *args, **kwargs):
            pass

        def list_repo_files(self, repo_id, **kwargs):
            attempts.append(repo_id)
            raise ConnectionError("Hub is unavailable in tests")

    def _no_load_dataset(*args, **kwargs):
        path = kwargs.get("path", args[0] if args else None)
        attempts.append(str(path))
        raise ConnectionError("Hub is unavailable in tests")

    monkeypatch.setattr(huggingface_hub, "HfApi", _NoHubApi)
    monkeypatch.setattr("datasets.load_dataset", _no_load_dataset)
    return attempts


def _check(dataset_name: str) -> HTTPException:
    request = datasets_route.CheckFormatRequest(dataset_name = dataset_name)
    with pytest.raises(HTTPException) as exc:
        datasets_route.check_format(request, hf_token = None, current_subject = "test")
    return exc.value


def test_missing_upload_reports_404(no_hub):
    from utils.paths import dataset_uploads_root

    missing = dataset_uploads_root() / "deleted-upload-test-fixture.jsonl"
    assert not missing.exists()
    error = _check(str(missing))
    assert error.status_code == 404
    assert "no longer on disk" in error.detail
    assert no_hub == [], "a local path must never be sent to the Hub as a repo id"


@pytest.mark.parametrize(
    "spelling",
    [
        pytest.param("{path}", id = "as-sent-by-the-ui"),
        pytest.param(" {path}", id = "leading-whitespace"),
        pytest.param("{path} ", id = "trailing-whitespace"),
    ],
)
def test_every_spelling_of_a_missing_upload_reports_404(spelling, no_hub):
    """resolve_dataset_path strips before testing absoluteness, so a guard on the
    raw string disagrees here and ships the local path to the Hub as a repo id."""
    from utils.paths import dataset_uploads_root

    missing = dataset_uploads_root() / "deleted-upload-test-fixture.jsonl"
    error = _check(spelling.format(path = missing))
    assert error.status_code == 404
    assert no_hub == []


def test_a_tilde_spelling_is_still_a_local_file(tmp_path, monkeypatch, no_hub):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / ".unsloth" / "studio"))

    error = _check("~/.unsloth/studio/assets/datasets/uploads/deleted.jsonl")
    assert error.status_code == 404
    assert no_hub == []


def test_corrupt_local_file_keeps_its_own_error(no_hub):
    from utils.paths import dataset_uploads_root

    corrupt = dataset_uploads_root() / "corrupt-test-fixture.jsonl"
    corrupt.parent.mkdir(parents = True, exist_ok = True)
    corrupt.write_text("{not valid json at all\n")

    error = _check(str(corrupt))
    assert error.status_code == 500, "an unreadable file is not a missing one"
    assert "no longer on disk" not in str(error.detail)


def test_hub_repo_id_never_reports_a_deleted_file(no_hub):
    """A relative reference stays a Hub lookup, however local it looks."""
    error = _check("uploads/private-or-unreachable")

    assert error.status_code != 404
    assert no_hub, "the Hub branch is where a relative reference belongs"


@pytest.mark.parametrize(
    ("dataset_name", "expected"),
    [
        pytest.param("{anchor}not-a-dataset.jsonl", "under a dataset root", id = "outside-roots"),
        pytest.param("{uploads}/../escape.jsonl", "'..' segments", id = "traversal"),
        pytest.param("uploads/nul\x00byte.jsonl", "null bytes", id = "null-byte"),
    ],
)
def test_rejected_paths_are_client_errors(dataset_name, expected, isolated_studio_home, no_hub):
    """A path resolve_dataset_path refuses is the caller's mistake, not a server
    fault, and must never reach the Hub. Matches the hub check-format twin."""
    from utils.paths import dataset_uploads_root

    # Anchor of the studio home: absolute on POSIX ("/x") and Windows ("C:\\x").
    # A hardcoded "/etc/x" is relative on Windows and misses the branch entirely.
    error = _check(
        dataset_name.format(
            anchor = isolated_studio_home.anchor,
            uploads = dataset_uploads_root(),
        )
    )

    assert error.status_code == 400
    assert expected in str(error.detail)
    assert no_hub == []
