# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Repo resolution for Spark-TTS bases. A dependency-light leaf so the mapping is
unit-testable without the Unsloth inference stack (see this package's __init__)."""

from __future__ import annotations


def spark_tts_base_repo(base_model: str) -> str:
    """Hub repo holding the BiCodec assets for a Spark-TTS base.

    A merged export records its base as the registry alias "Spark-TTS-0.5B/LLM", which
    names a load subdirectory rather than a repo, so passing it to snapshot_download
    rejected the export and it could not be deployed to Create. Mirrors the mapping in
    core/training/trainer.py; BiCodec lives at the repo root, not under LLM/.
    """
    if not base_model.endswith("/LLM"):
        return base_model
    parent = base_model.rsplit("/", 1)[0]
    # Already qualified ("unsloth/Spark-TTS-0.5B/LLM") vs bare alias ("Spark-TTS-0.5B/LLM").
    return parent if "/" in parent else f"unsloth/{parent}"
