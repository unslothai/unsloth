# Copyright 2026-present the Unforgettable contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json

from unforgettable.store.records import get_record
from unforgettable.tools.handlers import dispatch
from unforgettable.tools.specs import MEMORY_TOOL_NAMES


def test_tool_names_are_stable():
    assert MEMORY_TOOL_NAMES == {
        "memory_write",
        "memory_search",
        "memory_get",
        "memory_supersede",
        "memory_deprecate",
    }


def test_write_search_get_supersede_deprecate(db_path):
    written = json.loads(
        dispatch(
            "memory_write",
            {
                "kind": "directive",
                "title": "Always cite ids",
                "body": "Ground answers in returned memory ids.",
                "provenance": "human",
            },
            db_path=db_path,
        )
    )
    assert written["status"] == "active"
    search = json.loads(
        dispatch("memory_search", {"query": "cite ids"}, db_path=db_path)
    )
    assert search[0]["id"] == written["id"]
    got = json.loads(dispatch("memory_get", {"id": written["id"]}, db_path=db_path))
    assert got["title"] == "Always cite ids"
    superseded = json.loads(
        dispatch(
            "memory_supersede",
            {"id": written["id"], "body": "Cite ids and provenance."},
            db_path=db_path,
        )
    )
    assert get_record(written["id"], db_path=db_path)["status"] == "superseded"
    deprecated = json.loads(
        dispatch(
            "memory_deprecate",
            {"id": superseded["id"], "reason": "replaced later"},
            db_path=db_path,
        )
    )
    assert deprecated["status"] == "deprecated"
    assert "No matching" in dispatch(
        "memory_search", {"query": "cite ids"}, db_path=db_path
    )
