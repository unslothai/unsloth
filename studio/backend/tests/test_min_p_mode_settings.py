# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import pytest

from routes import chat_history


@pytest.mark.parametrize("mode", [None, "server-default", "custom"])
@pytest.mark.parametrize("value", [0, 0.01])
def test_min_p_mode_round_trip_and_numeric_patch(mode, value):
    params = {"minP": value, **({"minPMode": mode} if mode else {})}
    payload = {
        "inferenceParams": params,
        "inferenceParamsByModel": {"model": params},
        "customPresets": [{"name": "Saved", "params": params}],
    }
    chat_history.put_settings(payload, current_subject="test")
    assert chat_history.get_settings(current_subject="test").settings == payload
    assert chat_history.ChatThreadSettings(**params).model_dump(exclude_unset=True) == params
    chat_history.put_settings({"inferenceParams": {"minP": 0.2}}, current_subject="test")
    assert chat_history.get_settings(current_subject="test").settings["inferenceParams"] == {
        **params,
        "minP": 0.2,
    }
