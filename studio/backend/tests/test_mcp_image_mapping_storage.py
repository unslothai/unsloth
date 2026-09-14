# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.inference.mcp_image_disclosure import stored_image_input_mappings


@pytest.mark.parametrize("stored", [None, "", "broken", "null", "true", "5", "{}", '"text"', {}])
def test_optional_mapping_display_tolerates_missing_or_invalid_storage(stored):
    assert stored_image_input_mappings({"image_input_mappings_json": stored}) == []
    assert stored_image_input_mappings({}) == []


def test_mapping_reader_preserves_list_members_for_caller_validation():
    entries = [None, 5, "bad", {}, {"tool": "inspect", "field": "image", "encoding": "base64"}]
    assert (
        stored_image_input_mappings({"image_input_mappings_json": json.dumps(entries)}) == entries
    )


@pytest.mark.parametrize("stored", ["broken", "null", "{}", "[null]", "[5]", '["bad"]'])
def test_approval_mapping_lookup_rejects_corrupt_storage(monkeypatch, stored):
    from core.inference.mcp_image_disclosure import McpImageDisclosureError
    from core.inference.mcp_image_tool_loop import _mapping_for_name
    from storage import mcp_servers_db

    monkeypatch.setattr(
        mcp_servers_db,
        "get_server_for_tool",
        lambda _: {
            "id": "server",
            "is_enabled": True,
            "allow_image_attachments": True,
            "image_input_mappings_json": stored,
        },
    )
    with pytest.raises(McpImageDisclosureError, match = "configuration is invalid"):
        _mapping_for_name("mcp__server__inspect")
