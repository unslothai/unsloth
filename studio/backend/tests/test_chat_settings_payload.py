# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The mirrored chat toggles PUT /api/chat/settings accepts.

The payload is extra="forbid" and one bad field 400s the whole save, so these pin
the contract the client sanitises against before sending.
"""

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


def test_rag_source_replaces_rather_than_merges():
    """A thread pick over a stored kb pick must not keep kbId.

    The union's thread variant forbids extra fields, so a merged
    {"type": "thread", "kbId": ...} is out of contract the moment it is read back.
    """
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
    """An atomic key carries its whole value, so it can replace a quarantined row.

    The corrupt-key guard exists because a partial patch would merge onto a base
    that is no longer there. Applying it to ragSource would 409 the user's pick and
    leave the selection unsaved.
    """
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


@pytest.mark.parametrize(
    "payload",
    [
        # Full access disables the sandbox, so it is re-accepted each session.
        {"permissionMode": "full"},
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
        # bool subclasses int, so False would persist as the 0 "unlimited" sentinel.
        {"researchModelTimeoutSeconds": False},
        {"unknownSetting": True},
    ],
)
def test_out_of_contract_values_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate(payload)


# ---------------------------------------------------------------------------
# Non-finite numbers
# ---------------------------------------------------------------------------
#
# json.loads accepts bare NaN and Infinity, so both reach the payload from any
# client that is not a browser (JSON.stringify emits null for them). Two things
# then went wrong, and each needs its own guard.


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
    """A stored NaN is written to value_json as a bare `NaN` token.

    Python reads it back, so the row is never quarantined, and the response model
    renders it as null: the value is silently lost and the row on disk is not
    valid JSON for any reader that is not Python.
    """
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate(payload_for(value))


def test_the_rejection_detail_can_be_rendered_as_json():
    """The 400 must be renderable, or the caller gets a 500 instead.

    Starlette's JSONResponse dumps with allow_nan = False, so echoing the
    offending input back inside `detail` turned a correctly refused request into
    an unhandled ValueError in the response renderer.
    """
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
