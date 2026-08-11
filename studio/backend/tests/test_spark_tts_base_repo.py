# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The base repo a merged Spark-TTS export downloads for its BiCodec assets.

A merged export records its base as the registry alias "Spark-TTS-0.5B/LLM". That names
a load subdirectory, not a repo, so passing it straight to snapshot_download rejected the
export and it could not be deployed to Create.
"""

from __future__ import annotations

from core.inference.spark_tts_paths import spark_tts_base_repo


def test_the_bare_registry_alias_becomes_a_real_repo():
    assert spark_tts_base_repo("Spark-TTS-0.5B/LLM") == "unsloth/Spark-TTS-0.5B"


def test_an_already_qualified_alias_is_not_prefixed_twice():
    assert spark_tts_base_repo("unsloth/Spark-TTS-0.5B/LLM") == "unsloth/Spark-TTS-0.5B"


def test_a_plain_repo_id_is_left_alone():
    assert spark_tts_base_repo("unsloth/Spark-TTS-0.5B") == "unsloth/Spark-TTS-0.5B"
    assert spark_tts_base_repo("someone/their-own-spark") == "someone/their-own-spark"


def test_a_repo_whose_name_merely_ends_in_llm_is_not_mangled():
    # The suffix is a path segment, not a substring: "org/MyLLM" is a repo in its own right.
    assert spark_tts_base_repo("org/MyLLM") == "org/MyLLM"
