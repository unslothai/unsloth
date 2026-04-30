# SPDX-License-Identifier: AGPL-3.0-only

from models.inference import WikiDeletePreviewRequest


def test_wiki_delete_preview_request_accepts_single_entry_payload_shape():
    payload = WikiDeletePreviewRequest.model_validate(
        {
            "entry_type": "SOURCE",
            "entry": "sources/alpha",
            "cascade_orphan_knowledge": True,
        }
    )

    assert payload.entry_type == "source"
    assert payload.entries == ["sources/alpha"]


def test_wiki_delete_preview_request_accepts_string_entries_value():
    payload = WikiDeletePreviewRequest.model_validate(
        {
            "entry_type": "analysis",
            "entries": "analysis/alpha-summary",
            "cascade_orphan_knowledge": False,
        }
    )

    assert payload.entry_type == "analysis"
    assert payload.entries == ["analysis/alpha-summary"]
