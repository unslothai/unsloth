# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Min P intent survives the settings routes and SQLite without inferred defaults."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from routes import chat_history
from storage import studio_db


def _payload(params):
    return {
        "inferenceParams": dict(params),
        "inferenceParamsByModel": {"vllm/model": dict(params)},
        "customPresets": [{"name": "Saved", "params": dict(params)}],
    }


def _records(settings):
    return [
        settings["inferenceParams"],
        settings["inferenceParamsByModel"]["vllm/model"],
        settings["customPresets"][0]["params"],
    ]


@pytest.mark.parametrize("mode", ["server-default", "custom"])
@pytest.mark.parametrize("value", [0, 0.01, 0.2])
def test_mode_round_trip_and_partial_merge(mode, value):
    params = {"minPMode": mode, "minP": value, "temperature": 0.7, "topK": 20}
    chat_history.put_settings(_payload(params), current_subject = "test")
    chat_history.put_settings(
        {
            "inferenceParams": {"temperature": 0.4},
            "inferenceParamsByModel": {"vllm/model": {"temperature": 0.4}},
        },
        current_subject = "test",
    )
    records = _records(chat_history.get_settings(current_subject = "test").settings)
    assert records == [{**params, "temperature": 0.4}, {**params, "temperature": 0.4}, params]
    # A numeric-only patch has no evidence that the user changed the mode.
    chat_history.put_settings({"inferenceParams": {"minP": 0.3}}, current_subject = "test")
    assert chat_history.get_settings(current_subject = "test").settings["inferenceParams"] == {
        **params,
        "temperature": 0.4,
        "minP": 0.3,
    }


@pytest.mark.parametrize("value", [0, 0.01, 0.2])
def test_legacy_numbers_remain_exact_without_fabricated_mode(value):
    params = {"minP": value, "temperature": 0.7}
    chat_history.put_settings(_payload(params), current_subject = "test")
    assert _records(chat_history.get_settings(current_subject = "test").settings) == [params] * 3
    assert (
        chat_history.ChatThreadSettings.model_validate(params).model_dump(exclude_unset = True)
        == params
    )


@pytest.mark.parametrize("mode", ["server-default", "custom"])
def test_thread_mode_patch_persists_number_and_siblings(mode):
    params = {"minP": 0.01, "temperature": 0.7, "topK": 20}
    studio_db.upsert_chat_thread(
        {
            "id": "minp-thread",
            "title": "Test",
            "modelType": "base",
            "modelId": "vllm/model",
            "createdAt": 1_700_000_000_000,
            "settings": params,
        }
    )
    legacy = chat_history.thread_from_row(studio_db.get_chat_thread("minp-thread"))
    assert legacy.model_dump()["settings"] == params
    chat_history.patch_thread(
        "minp-thread",
        chat_history.ChatThreadPatch(settingsPatch = {"minPMode": mode}),
        current_subject = "test",
    )
    reloaded = chat_history.thread_from_row(studio_db.get_chat_thread("minp-thread"))
    assert reloaded.model_dump()["settings"] == {**params, "minPMode": mode}


@pytest.mark.parametrize("scope", ["inferenceParams", "inferenceParamsByModel", "customPresets"])
def test_invalid_mode_rejects_entire_settings_write(scope):
    chat_history.put_settings({"inferenceParams": {"minP": 0.2}}, current_subject = "test")
    payload = _payload({"minPMode": "automatic"})
    with pytest.raises(HTTPException) as error:
        chat_history.put_settings({scope: payload[scope]}, current_subject = "test")
    assert error.value.status_code == 400
    assert chat_history.get_settings(current_subject = "test").settings == {
        "inferenceParams": {"minP": 0.2}
    }


def test_invalid_thread_mode_rejected_and_empty_settings_stay_empty():
    with pytest.raises(ValidationError):
        chat_history.ChatThreadPatch(settingsPatch = {"minPMode": "automatic"})
    assert chat_history.ChatThreadSettings().model_dump(exclude_unset = True) == {}
    assert chat_history.ChatInferenceSettings().model_dump(exclude_unset = True) == {}
