# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from typing import Optional


SNAC_CODEC_REPOSITORY = "hubertsiuzdak/snac_24khz"
BICODEC_CODEC_REPOSITORY = "unsloth/Spark-TTS-0.5B"
DAC_CODEC_REPOSITORY = "ibm-research/DAC.speech.v1.0"

_CODEC_REPOSITORIES = {
    "snac": (SNAC_CODEC_REPOSITORY,),
    "bicodec": (BICODEC_CODEC_REPOSITORY,),
    "dac": (DAC_CODEC_REPOSITORY,),
}


def audio_codec_repositories(audio_type: Optional[str]) -> tuple[str, ...]:
    return _CODEC_REPOSITORIES.get(audio_type or "", ())
