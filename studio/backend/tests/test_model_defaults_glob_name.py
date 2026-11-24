# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A model name reached rglob as a pattern, so a crafted name ([...]-heavy) made the match backtrack for
hours and hung the backend for every account. The name is a literal file name."""

import time

from utils.models import model_config


def test_a_glob_heavy_model_name_is_matched_literally():
    name = "*?" * 9 + "*!"
    start = time.monotonic()
    model_config.load_model_defaults(name)  # falls back to the default config
    from utils.inference import inference_config

    inference_config._has_specific_yaml(name)
    assert time.monotonic() - start < 5


def test_known_model_defaults_still_load():
    assert model_config.load_model_defaults("unsloth/Qwen3-4B-Instruct-2507") is not None
