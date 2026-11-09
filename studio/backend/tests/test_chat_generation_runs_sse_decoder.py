# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json

import pytest

from core.inference.chat_generation_runs import _SSEDecoder


def _chunk(content: str) -> str:
    # model_dump_json and llama-server both write U+2028, U+2029 and U+0085 unescaped.
    payload = {"choices": [{"index": 0, "delta": {"content": content}}]}
    return "data: " + json.dumps(payload, ensure_ascii = False) + "\n\n"


@pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\x85"])
@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_a_delta_holding_a_unicode_line_separator_is_kept(separator, newline):
    content = f"first{separator}second"
    wire = _chunk(content).replace("\n", newline)

    values = _SSEDecoder().feed(wire)

    assert len(values) == 1
    assert json.loads(values[0])["choices"][0]["delta"]["content"] == content


def test_multi_line_data_fields_are_still_joined_with_newlines():
    values = _SSEDecoder().feed('data: {"a":\ndata: 1}\n\n')

    assert values == ['{"a":\n1}']
