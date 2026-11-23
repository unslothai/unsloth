# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from playwright.sync_api import sync_playwright

LEAF = (
    Path(__file__).resolve().parents[2] / "studio/frontend/src/features/browser/agent/page-agent.ts"
)
ABSTRACT = " ".join(
    f"Sentence {i} of a long abstract about efficient finetuning." for i in range(40)
)
CODE = """from unsloth import FastLanguageModel

# Load the model in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    "unsloth/Qwen3-4B",
    load_in_4bit = True,
)"""


def shipped() -> str:
    js = f"import({json.dumps(LEAF.as_uri())}).then((m) => console.log(JSON.stringify(m.pageAgentSource())))"
    cmd = ["node", "--experimental-strip-types", "--no-warnings", "-e", js]
    return json.loads(subprocess.run(cmd, check = True, capture_output = True, text = True).stdout)


@pytest.fixture(scope = "module")
def page():
    source = shipped()
    with sync_playwright() as p:
        with getattr(p, os.environ.get("PW_BROWSER", "chromium")).launch(headless = True) as browser:
            with browser.new_page(viewport = {"width": 1280, "height": 900}) as page:

                def read(html, **command):
                    page.set_content(html)
                    page.evaluate(source)
                    return page.evaluate(
                        "(c) => window[Symbol.for('unsloth.agent')].run(c)",
                        {"op": "snapshot", **command},
                    )["text"]

                yield read


@pytest.mark.parametrize("full_page", [False, True])
def test_long_paragraph_reaches_the_model_whole(page, full_page):
    text = page(f"<h1>Paper</h1><p>{ABSTRACT}</p><p>After</p>", fullPage = full_page)
    assert ABSTRACT in text.splitlines()
    assert text.splitlines()[-1] == "After"


def test_code_block_keeps_its_lines(page):
    html = f"<p>Example:</p><pre><code>{CODE}</code></pre><p>Done</p>"
    text = page(html)
    assert f"```\n{CODE}\n```" in text
    assert text.endswith("Done")


def test_wrapped_source_text_still_reads_as_one_line(page):
    html = '<p>Install with\n      <code style="white-space: pre">pip install\nunsloth</code>\n      today.</p>'
    assert page(html).splitlines() == ["Install with pip install", "unsloth today."]


def test_page_text_stays_within_the_budget(page):
    text = page(f"<p>{'word ' * 5000}</p><p>{'tail ' * 5000}</p>")
    assert 7000 < len(text) < 8200
    assert "tail" not in text
    assert text.endswith("(more below: use browser_scroll, browser_find, or a snapshot ref)")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--confcutdir", str(Path(__file__).parent)]))
