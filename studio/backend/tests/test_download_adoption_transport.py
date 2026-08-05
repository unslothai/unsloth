# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A start that adopts a running job reports the transport that job is on."""

from hub.utils.download_registry import DownloadRegistry


def _claim(registry, key, transport):
    accepted, state = registry.claim(
        key,
        transport,
        repo_type = "model",
        repo_id = key.split("::", 1)[0],
    )
    return accepted, state


def test_a_running_job_reports_the_transport_it_started_on():
    registry = DownloadRegistry()
    assert _claim(registry, "unsloth/Qwen3-4B-GGUF", "xet") == (True, "running")

    # The second client asked for HTTP; the claim is refused and it adopts.
    accepted, state = _claim(registry, "unsloth/Qwen3-4B-GGUF", "http")
    assert accepted is False and state == "running"
    assert registry.adoptable("unsloth/Qwen3-4B-GGUF") is True
    # What it must be told, rather than the http it asked for: pausing a Xet
    # run promises a resume that does not exist.
    assert registry.job_transport("unsloth/Qwen3-4B-GGUF") == "xet"


def test_an_unknown_job_has_no_transport_to_report():
    registry = DownloadRegistry()
    assert registry.job_transport("unsloth/never-started") is None


def test_a_job_claimed_without_metadata_reports_nothing():
    registry = DownloadRegistry()
    assert registry.claim("unsloth/bare", "http")[0] is True
    assert registry.job_transport("unsloth/bare") is None
