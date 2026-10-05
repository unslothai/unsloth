# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Spark-TTS's tokenizer lives under LLM/, so the pre-detect load needs the subfolder.

unsloth/Spark-TTS-0.5B keeps only BiCodec/, config.yaml, src/ and wav2vec2-* at its repo
root. AutoTokenizer on the root finds no vocab and raises "Couldn't instantiate the
backend tokenizer ... You need to have sentencepiece or tiktoken installed", which sends
the reader after a dependency that is installed and irrelevant. _load_model already reads
weights from LLM/; the tokenizer pre-detect has to agree.

Source-level: the real call needs the network and a 2 GB download.
"""

from __future__ import annotations

import os
import typing
from pathlib import Path

import pytest


def _find_repo_root() -> Path | None:
    env = os.environ.get("UNSLOTH_REPO_ROOT")
    if env:
        p = Path(env).resolve()
        if (p / "studio" / "backend").is_dir():
            return p
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "studio" / "backend").is_dir():
            return parent
    return None


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT is None:
    pytest.skip("Could not locate studio/backend.", allow_module_level = True)


def _helper():
    """Exec just the helper: importing trainer.py pulls in the whole torch stack."""
    src = (_REPO_ROOT / "studio/backend/core/training/trainer.py").read_text(encoding = "utf-8")
    start = src.index("def _spark_tts_tokenizer_kwargs")
    end = src.index("class UnslothTrainer:")
    namespace: dict = {"os": os, "Optional": typing.Optional}
    exec(src[start:end], namespace)
    return namespace["_spark_tts_tokenizer_kwargs"]


def test_a_spark_repo_root_reads_the_llm_subfolder():
    assert _helper()("bicodec", "unsloth/Spark-TTS-0.5B") == {"subfolder": "LLM"}


@pytest.mark.parametrize(
    "lookup_name",
    ["Spark-TTS-0.5B/LLM", r"C:\models\Spark-TTS-0.5B\LLM", "/srv/Spark-TTS-0.5B/LLM"],
)
def test_a_name_already_pointing_at_llm_is_left_alone(lookup_name: str):
    assert _helper()("bicodec", lookup_name) == {}


@pytest.mark.parametrize("audio_type", ["snac", "csm", "dac", "whisper", "audio_vlm", None])
def test_every_other_codec_is_untouched(audio_type):
    assert _helper()(audio_type, "unsloth/orpheus-3b-0.1-ft") == {}
