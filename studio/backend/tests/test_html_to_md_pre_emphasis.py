# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference._html_to_md import html_to_markdown
from core.inference.tools import _render_page


@pytest.mark.parametrize(
    "pre, text",
    [
        (
            "<pre>SELECT * FROM <em>t</em> WHERE <b>x</b> = 1</pre>",
            "SELECT * FROM t WHERE x = 1",
        ),
        (
            "<pre>func append(slice []<i>T</i>, elements ...<i>T</i>) []<i>T</i></pre>",
            "func append(slice []T, elements ...T) []T",
        ),
        (
            "<pre><strong>grep</strong> [<em>OPTION</em>...] <em>PATTERNS</em></pre>",
            "grep [OPTION...] PATTERNS",
        ),
    ],
)
def test_code_block_keeps_text_as_written(pre, text):
    assert html_to_markdown(pre) == f"```\n{text}\n```"


def test_fetched_page_code_block_keeps_text_as_written():
    body = (
        "<html><head><title>SELECT</title></head><body><main><h1>SELECT</h1>"
        "<p>Retrieve <b>rows</b> from a table or view.</p>"
        "<pre class='synopsis'>SELECT [ ALL | DISTINCT ]\n"
        "    [ * | <em class='replaceable'>expression</em> [ [ AS ] "
        "<em class='replaceable'>output_name</em> ] ]\n"
        "    [ FROM <em class='replaceable'>from_item</em> ]</pre>"
        "</main></body></html>"
    )
    page = _render_page(("html", "https://example.com/sql-select.html", body), 16000)
    assert "Retrieve **rows** from a table or view." in page
    assert (
        "SELECT [ ALL | DISTINCT ]\n"
        "    [ * | expression [ [ AS ] output_name ] ]\n"
        "    [ FROM from_item ]"
    ) in page
