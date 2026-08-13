# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A curated registry alias must be resolved before asking whether it is an audio model.

"Spark-TTS-0.5B/LLM" names a load subdirectory, not a repository. Probing it fetched a
repo that does not exist, got a 404 on every candidate path, and read that as a
DEFINITIVE "not an audio model" rather than "not a repo id". Spark-TTS then presented as
a text model, so choosing it with an audio dataset hit the modality gate and Start
Training stayed disabled (reported on Windows against PR 7984).
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from routes.models import _audio_probe_target  # noqa: E402


def test_a_registry_alias_resolves_to_the_repo_that_exists():
    assert _audio_probe_target("Spark-TTS-0.5B/LLM") == "unsloth/Spark-TTS-0.5B"


def test_a_plain_repo_id_is_unchanged():
    assert _audio_probe_target("unsloth/Spark-TTS-0.5B") == "unsloth/Spark-TTS-0.5B"
    assert _audio_probe_target("unsloth/gemma-3-270m-it") == "unsloth/gemma-3-270m-it"


def test_a_local_path_is_never_rewritten(tmp_path):
    # A trained checkpoint is a directory, and the registry knows nothing about it.
    assert _audio_probe_target(str(tmp_path)) == str(tmp_path)


def test_an_unresolvable_name_falls_through_rather_than_failing():
    assert _audio_probe_target("nobody/not-in-any-registry") == "nobody/not-in-any-registry"


def test_the_merged_export_load_path_resolves_the_alias_the_same_way():
    """One resolver, not two. The BiCodec export path used to carry its own copy of the
    "Spark-TTS-0.5B/LLM" -> "unsloth/Spark-TTS-0.5B" mapping; it now shares load_scan_target
    with the capability probe here and with the trainer preflight in routes/training.py, so
    the three cannot drift."""
    # Read rather than import: core.inference.inference pulls the whole Unsloth stack,
    # which is what made a second, dependency-light copy of this mapping tempting.
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[1] / "core" / "inference" / "inference.py"
    ).read_text(encoding = "utf-8")
    assert "load_scan_target(" in source
    assert "spark_tts_base_repo" not in source

    from utils.security import load_scan_target
    from utils.utils import canonical_model_repo_id

    repo, subdirs = load_scan_target(canonical_model_repo_id("Spark-TTS-0.5B/LLM"), ())
    assert repo == "unsloth/Spark-TTS-0.5B"
    # BiCodec lives at the repo root; LLM is where the language model half sits.
    assert subdirs == ("LLM",)
