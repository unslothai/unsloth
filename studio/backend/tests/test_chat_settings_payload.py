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
        "speculativeType": "ngram",
        "gpuMemoryMode": "manual",
        "fitOnDeviceOnly": True,
    }


def test_thread_rag_source_keeps_its_shape():
    payload = ChatSettingsPayload.model_validate({"ragSource": {"type": "thread"}})
    assert payload.model_dump(exclude_unset = True) == {"ragSource": {"type": "thread"}}


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
        {"unknownSetting": True},
    ],
)
def test_out_of_contract_values_are_rejected(payload):
    with pytest.raises(ValidationError):
        ChatSettingsPayload.model_validate(payload)
