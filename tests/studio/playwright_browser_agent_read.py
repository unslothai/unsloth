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

                def read(
                    html,
                    scroll = 0,
                    **command,
                ):
                    page.set_content(html)
                    page.evaluate(source)
                    page.evaluate(f"window.scrollTo(0, {scroll})")
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


def test_cut_code_block_still_closes_its_fence(page):
    code = "\n".join(f"line_{i} = {i}" for i in range(2000))
    text = page(f"<p>Example:</p><pre><code>{code}</code></pre>")
    body = text.rsplit("\n…", 1)[0]
    assert body.endswith("…\n```")
    assert body.count("```") == 2


def test_fence_outlasts_backticks_in_the_code(page):
    code = "Write:\n```python\nprint(1)\n```"
    text = page(f"<pre>{code}</pre><p>Done</p>")
    assert f"````\n{code}\n````" in text


def test_scrolling_moves_through_a_block_taller_than_the_budget(page):
    code = "\n".join(f"line_{i:04d} = {i}" for i in range(3000))
    html = f'<pre style="margin: 0; font: 15px/20px monospace">{code}</pre>'
    top = [l for l in page(html).splitlines() if l.startswith("line_")]
    lower = [l for l in page(html, scroll = 8 + 20 * 1000).splitlines() if l.startswith("line_")]
    assert top[0] == "line_0000 = 0"
    assert lower[0] == "line_1000 = 1000"
    assert page(html, scroll = 20 * 1000, fullPage = True).splitlines()[1] == "line_0000 = 0"


def test_highlighter_lines_stay_one_fenced_block(page):
    lines = "".join(
        f'<span style="display: block">{line or chr(10)}</span>' for line in CODE.split("\n")
    )
    text = page(f"<p>Example:</p><pre>&gt;&gt;&gt; x<code>{lines}</code></pre><p>Done</p>")
    assert f"```\n>>> x\n{CODE}\n```" in text


def test_scrolling_moves_through_highlighter_lines(page):
    spans = "".join(f'<span style="display: block">line_{i:04d} = {i}</span>' for i in range(3000))
    html = f'<pre style="margin: 0; font: 15px/20px monospace">{spans}</pre>'
    text = page(html, scroll = 8 + 20 * 1000)
    assert text.splitlines()[1] == "```"
    assert [line for line in text.splitlines() if line.startswith("line_")][0] == "line_1000 = 1000"


def test_many_backtick_runs_still_read(page):
    code = "\n".join(f"console.log(`v{i}`)" for i in range(70000))
    text = page(f"<pre>{code}</pre>", fullPage = True)
    assert text.startswith("```\nconsole.log(`v0`)")
    assert text.endswith("(more below: use browser_scroll, browser_find, or a snapshot ref)")


def test_spacing_between_inline_elements_does_not_cut_early(page):
    items = "".join(f"<span> w{i:04d} </span> " for i in range(1200))
    text = page(f"<p>{items}</p>")
    assert "w1199" in text
    assert "more below" not in text


def test_hidden_code_line_stays_out(page):
    line = '<span style="display: block{}">{}</span>'
    html = (
        "<pre>"
        + line.format("", "one()")
        + line.format("; height: 0; overflow: hidden", "hidden()")
        + line.format("", "two()")
        + "</pre>"
    )
    assert page(html) == "```\none()\ntwo()\n```"


def test_boxless_wrapper_in_code_keeps_its_text(page):
    assert (
        page('<pre>a = <span style="display: contents">value</span> + 1</pre>')
        == "```\na = value + 1\n```"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "--confcutdir", str(Path(__file__).parent)]))
