# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
"""Run with Playwright's Python and PLAYWRIGHT_BROWSERS_PATH set to its Chromium cache."""

import re
import subprocess
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[4]
source = subprocess.check_output(
    [
        "node",
        "--experimental-strip-types",
        "--input-type=module",
        "-e",
        "import {pageAgentSource} from './studio/frontend/src/features/browser/agent/page-agent.ts'; console.log(pageAgentSource())",
    ],
    cwd = ROOT,
    text = True,
)

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless = True, args = ["--disable-gpu"])
    page = browser.new_page()
    page.set_content(
        '<div id="background" inert><iframe id="a"></iframe></div><iframe id="b"></iframe>'
    )
    a, b = page.frames[1:]
    markup = '<form><input aria-label="first"><textarea aria-label="second"></textarea><div contenteditable="true" aria-label="editor"></div></form>'
    for frame in (a, b):
        frame.set_content(markup)
        frame.evaluate(
            '(source) => { window.agent = eval(source); window.events = []; window.focuses = []; window.submits = 0; for (const kind of ["focus","focusin","blur","focusout"]) document.addEventListener(kind, e => focuses.push(kind+":"+e.target.getAttribute("aria-label")),true); document.addEventListener("input", e => events.push(e.target.getAttribute("aria-label"))); document.addEventListener("submit", e => { e.preventDefault(); submits++; }); }',
            source,
        )
    b.locator("input").focus()
    a_refs = dict(
        re.findall(
            r'\[(r\d+)\] textbox "([^"]+)"',
            a.evaluate('agent.run({op:"snapshot",fullPage:true}).text'),
        )
    )
    a_refs = {name: ref for ref, name in a_refs.items()}
    b_refs = dict(
        re.findall(
            r'\[(r\d+)\] textbox "([^"]+)"',
            b.evaluate('agent.run({op:"snapshot",fullPage:true}).text'),
        )
    )
    b_refs = {name: ref for ref, name in b_refs.items()}

    def run(frame, command):
        result = frame.evaluate("(c) => agent.run(c)", command)
        assert result["ok"], result

    def check_focus():
        assert page.evaluate("document.activeElement.id") == "b"
        assert b.evaluate('document.activeElement.getAttribute("aria-label")') == "first"

    run(a, dict(op = "type", ref = a_refs["first"], text = "alpha", submit = False, virtualInput = True))
    run(a, dict(op = "press", key = "Backspace", virtualInput = True))
    run(a, dict(op = "press", key = "Tab", virtualInput = True))
    assert a.evaluate("focuses.slice(0,6)") == [
        "focus:first",
        "focusin:first",
        "blur:first",
        "focusout:first",
        "focus:second",
        "focusin:second",
    ]
    run(a, dict(op = "press", key = "Enter", virtualInput = True))
    assert a.locator("textarea").input_value() == "\n"
    run(a, dict(op = "type", ref = a_refs["second"], text = "line", submit = False, virtualInput = True))
    run(a, dict(op = "press", key = "Enter", virtualInput = True))
    run(a, dict(op = "type", ref = a_refs["editor"], text = "hello", submit = False, virtualInput = True))
    run(a, dict(op = "press", key = "Backspace", virtualInput = True))
    check_focus()
    assert a.locator("input").input_value() == "alph"
    assert a.locator("textarea").input_value() == "line\n"
    assert a.locator("[contenteditable]").inner_text() == "hell"
    assert b.locator("input").input_value() == ""
    assert a.evaluate("events") == [
        "first",
        "first",
        "second",
        "second",
        "second",
        "editor",
        "editor",
    ]
    run(a, dict(op = "type", ref = a_refs["first"], text = "submitted", submit = True, virtualInput = True))
    assert a.evaluate("submits") == 1 and b.evaluate("submits") == 0
    check_focus()
    run(b, dict(op = "type", ref = b_refs["first"], text = "beta", submit = False))
    check_focus()
    assert b.locator("input").input_value() == "beta"
    assert a.locator("input").input_value() == "submitted"
    run(a, dict(op = "type", ref = a_refs["first"], text = "resume", submit = False, virtualInput = True))
    check_focus()
    page.evaluate(
        'document.querySelector("#background").inert = false; document.querySelector("#b").inert = true'
    )
    run(a, dict(op = "status"))
    run(a, dict(op = "describe", ref = ""))
    run(a, dict(op = "press", key = "Backspace"))
    assert a.locator("input").input_value() == "resum"
    assert a.evaluate('document.activeElement.getAttribute("aria-label")') == "first"
    assert b.locator("input").input_value() == "beta"
    print(
        "PASS: inert A input, textarea, contenteditable, press and Tab; B focus preserved; A resumes its target after becoming visible"
    )
    browser.close()
