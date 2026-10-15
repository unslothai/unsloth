# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""PUT /api/chat/settings forbids extra fields, so one bad toggle 400s the whole save."""

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from routes.chat_history import ChatSettingsPayload  # noqa: E402
from storage.studio_db import (  # noqa: E402
    CorruptSettingsError,
    _deep_merge_settings,
    get_connection,
    upsert_chat_settings_merge,
)


def test_mirrored_settings_round_trip():
    payload = ChatSettingsPayload.model_validate(
        {
            "toolsEnabled": True,
            "deepResearchEnabled": False,
            "permissionMode": "ask",
            "ragSource": {"type": "kb", "kbId": "notes"},
            "ragMode": "dense",
            "ragTopK": 12,
            "ragAutoInject": "on",
            "ragAutoInjectMinScore": 0.42,
            "researchWebsitePolicy": {
                "allowedDomains": ["unsloth.ai"],
                "blockedDomains": [],
            },
            "researchModelTimeoutSeconds": 0,
            "speculativeType": "ngram",
            "gpuMemoryMode": "manual",
            "fitOnDeviceOnly": True,
        }
    )

    assert payload.model_dump(exclude_unset = True) == {
        "toolsEnabled": True,
        "deepResearchEnabled": False,
        "permissionMode": "ask",
        "ragSource": {"type": "kb", "kbId": "notes"},
        "ragMode": "dense",
        "ragTopK": 12,
        "ragAutoInject": "on",
        "ragAutoInjectMinScore": 0.42,
        "researchWebsitePolicy": {
            "allowedDomains": ["unsloth.ai"],
            "blockedDomains": [],
        },
        "researchModelTimeoutSeconds": 0,
        "speculativeType": "ngram",
        "gpuMemoryMode": "manual",
        "fitOnDeviceOnly": True,
    }


def test_thread_rag_source_keeps_its_shape():
    payload = ChatSettingsPayload.model_validate({"ragSource": {"type": "thread"}})
    assert payload.model_dump(exclude_unset = True) == {"ragSource": {"type": "thread"}}


def test_research_mcp_sources_round_trip_and_clear():
    sources = [{"serverId": "notes", "tool": "search_notes"}]
    payload = ChatSettingsPayload.model_validate({"researchMcpSources": sources})
    assert payload.model_dump(exclude_unset = True) == {"researchMcpSources": sources}
    cleared = ChatSettingsPayload.model_validate({"researchMcpSources": []})
    assert _deep_merge_settings(
        payload.model_dump(exclude_unset = True), cleared.model_dump(exclude_unset = True)
    ) == {"researchMcpSources": []}


def test_rag_source_replaces_rather_than_merges():
    """A thread ragSource replaces a kb pick rather than merging, since the thread variant forbids kbId."""
    merged = _deep_merge_settings(
        {"ragSource": {"type": "kb", "kbId": "notes"}},
        {"ragSource": {"type": "thread"}},
    )
    assert merged["ragSource"] == {"type": "thread"}
    ChatSettingsPayload.model_validate(merged)


def _corrupt_stored_setting(key: str) -> None:
    conn = get_connection()
    try:
        conn.execute("UPDATE chat_settings SET value_json = ? WHERE key = ?", ("{not json", key))
        conn.commit()
    finally:
        conn.close()


def test_a_valid_rag_source_repairs_a_corrupt_row():
    """Atomic keys carry their whole value, so a valid write can repair a quarantined row."""
    upsert_chat_settings_merge({"ragSource": {"type": "kb", "kbId": "notes"}})
    _corrupt_stored_setting("ragSource")

    merged = upsert_chat_settings_merge({"ragSource": {"type": "thread"}})
    assert merged["ragSource"] == {"type": "thread"}


def test_a_partial_patch_onto_a_corrupt_row_still_conflicts():
    upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.7, "topP": 0.9}})
    _corrupt_stored_setting("inferenceParams")

    with pytest.raises(CorruptSettingsError):
        upsert_chat_settings_merge({"inferenceParams": {"temperature": 0.2}})


def test_other_nested_settings_still_merge():
    merged = _deep_merge_settings(
        {"inferenceParams": {"temperature": 0.7, "topP": 0.9}},
        {"inferenceParams": {"temperature": 0.2}},
    )
    assert merged["inferenceParams"] == {"temperature": 0.2, "topP": 0.9}


def test_unset_fields_stay_out_of_the_merge():
    payload = ChatSettingsPayload.model_validate({"ragTopK": 5})
    assert payload.model_dump(exclude_unset = True) == {"ragTopK": 5}


@pytest.mark.parametrize("level", ["high", "low"])
def test_sandbox_level_survives_the_payload(level):
    payload = ChatSettingsPayload.model_validate({"sandboxLevel": level})
    assert payload.model_dump(exclude_unset = True) == {"sandboxLevel": level}


def test_max_tool_calls_off_survives_the_payload():
    payload = ChatSettingsPayload.model_validate({"maxToolCallsPerMessage": 0})
    assert payload.model_dump(exclude_unset = True) == {"maxToolCallsPerMessage": 0}


@pytest.mark.parametrize(
    "payload",
    [
        {"permissionMode": "full"},
        {"sandboxLevel": "medium"},
        {"ragTopK": 0},
        {"ragTopK": 51},
        {"ragAutoInjectMinScore": 2},
        {"ragMode": "vector"},
        {"speculativeType": "mtp"},
        {"gpuMemoryMode": ""},
        {"ragSource": {"type": "kb"}},
        {"ragSource": {"type": "kb", "kbId": ""}},
        {"researchWebsitePolicy": {"allowedDomains": "unsloth.ai"}},
        # The run route takes 0 or at least 10, so a persisted 1..9 would 400 every run.
        {"researchModelTimeoutSeconds": 1},
        {"researchModelTimeoutSeconds": 9},
        {"researchModelTimeoutSeconds": -1},
        {"researchModelTimeoutSeconds": 365 * 24 * 3600 + 1},
        # bool subclasses int, so False would persist as the 0 sentinel.
        {"researchModelTimeoutSeconds": False},
        {"maxToolCallsPerMessage": False},
        {"maxToolCallsPerMessage": True},
        {"maxToolCallsPerMessage": -1},
        {"toolCallTimeout": 0},
        {"unknownSetting": True},
    ],
)
def test_out_of_contract_values_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate(payload)


# json.loads accepts bare NaN/Infinity, so non-browser clients can send them.


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    "payload_for",
    [
        lambda v: {"ragAutoInjectMinScore": v},
        lambda v: {"inferenceParams": {"temperature": v}},
        lambda v: {"customPresets": [{"name": "p", "params": {"topP": v}}]},
    ],
)
def test_non_finite_numbers_are_refused_rather_than_stored(payload_for, value):
    """Non-finite numbers are refused, since a bare NaN token is invalid JSON for non-Python readers."""
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate(payload_for(value))


def test_the_rejection_detail_can_be_rendered_as_json():
    """The 400 detail must not echo NaN input, since JSONResponse with allow_nan=False would 500 on it."""
    import json

    from fastapi import HTTPException

    from routes.chat_history import put_settings

    with pytest.raises(HTTPException) as excinfo:
        put_settings({"ragAutoInjectMinScore": float("nan")}, current_subject = "t")
    assert excinfo.value.status_code == 400
    json.dumps(excinfo.value.detail, allow_nan = False)


def test_auto_compact_settings_round_trip():
    payload = ChatSettingsPayload.model_validate(
        {
            "autoCompactEnabled": False,
            "contextPolicy": "rolling",
            "compactionHeadroomRatio": 0.05,
        }
    )
    assert payload.model_dump(exclude_unset = True) == {
        "autoCompactEnabled": False,
        "contextPolicy": "rolling",
        "compactionHeadroomRatio": 0.05,
    }


def test_auto_compact_settings_can_inherit_the_server_policy():
    payload = ChatSettingsPayload.model_validate({"contextPolicy": "inherit"})
    assert payload.model_dump(exclude_unset = True) == {"contextPolicy": "inherit"}


def test_compaction_headroom_ratio_is_bounded():
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate({"compactionHeadroomRatio": 1.5})


def test_a_sampling_seed_survives_the_payload():
    payload = ChatSettingsPayload.model_validate({"inferenceParams": {"seed": 3407}})
    assert payload.model_dump(exclude_unset = True) == {"inferenceParams": {"seed": 3407}}


def test_clearing_the_seed_reaches_the_merge_as_null():
    """A cleared seed is an explicit null, not an omission: the merge overwrites per
    key and never removes one, so an omitted seed would leave the old pin in place."""
    payload = ChatSettingsPayload.model_validate({"inferenceParams": {"seed": None}})
    updates = payload.model_dump(exclude_unset = True)
    assert updates == {"inferenceParams": {"seed": None}}

    merged = _deep_merge_settings({"inferenceParams": {"seed": 3407, "topP": 0.9}}, updates)
    assert merged["inferenceParams"] == {"seed": None, "topP": 0.9}


@pytest.mark.parametrize(
    "seed",
    [
        # bool subclasses int, so lax mode would store either as a pin the user never set.
        True,
        False,
        -1,
        2**32 - 1,  # llama.cpp's "draw one" sentinel, not a value a pin can name.
        2**32,
        1e40,
    ],
)
def test_out_of_range_seeds_are_refused(seed):
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate({"inferenceParams": {"seed": seed}})


@pytest.mark.parametrize("seed", [0, 3407, 2**32 - 2])
def test_the_whole_uint32_pin_range_is_accepted(seed):
    payload = ChatSettingsPayload.model_validate({"inferenceParams": {"seed": seed}})
    assert payload.model_dump(exclude_unset = True) == {"inferenceParams": {"seed": seed}}
