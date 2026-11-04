# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from pathlib import Path
import subprocess

from playwright.sync_api import sync_playwright

FRONTEND = Path(__file__).resolve().parents[2]


def test_preview_scroll_and_inert_boundary():
    source = """<!doctype html><html><head><style>
      body { margin:0; height:3000px } #fixed {position:fixed;top:0} #target {position:absolute;top:900px}
      </style></head><body><div id="fixed">fixed</div><input id="input" autofocus><div id="target">target</div>
      <script>document.body.dataset.compromised='yes'</script><img src="https://example.invalid/tracker"></body></html>"""
    code = """import {observationDocument} from './src/features/browser/observer-state.ts';
      console.log(JSON.stringify(observationDocument(process.argv[1], 'a'.repeat(32),
        {scrollX:0, scrollY:800, viewportWidth:600, viewportHeight:400})));"""
    result = subprocess.run(
        ["node", "--experimental-strip-types", "--input-type=module", "-e", code, source],
        cwd = FRONTEND,
        check = True,
        capture_output = True,
        text = True,
    )
    document = json.loads(result.stdout)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless = True, args = ["--disable-gpu"])
        page = browser.new_page(viewport = {"width": 800, "height": 600})
        responses = []
        failures = []
        page.on("response", lambda response: responses.append(response.url))
        page.on("requestfailed", lambda request: failures.append((request.url, request.failure)))
        page.set_content(
            '<button id="back">Back</button><div inert><iframe id="view" sandbox="allow-scripts" tabindex="-1" style="width:600px;height:400px;pointer-events:none"></iframe></div>'
        )
        page.locator("#back").focus()
        page.locator("#view").evaluate("(frame, doc) => frame.srcdoc = doc", document)
        frame = page.frame_locator("#view")
        frame.locator("#target").wait_for()
        page.frames[-1].wait_for_function(
            "Math.round(document.querySelector('#target').getBoundingClientRect().top) === 100"
        )
        assert (
            frame.locator("#target").evaluate("el => Math.round(el.getBoundingClientRect().top)")
            == 100
        )
        assert (
            frame.locator("#fixed").evaluate("el => Math.round(el.getBoundingClientRect().top)")
            == 0
        )
        assert frame.locator("body").get_attribute("data-compromised") is None
        assert page.evaluate("document.activeElement.id") == "back"
        page.locator("#back").focus()
        page.keyboard.press("Tab")
        assert page.locator("#view").evaluate("el => document.activeElement === el") is False
        page.mouse.click(80, 80)
        page.keyboard.type("must not enter page")
        assert frame.locator("#input").input_value() == ""
        assert not any("example.invalid" in url for url in responses)
        assert any(
            "example.invalid" in url and "csp" in (reason or "").lower() for url, reason in failures
        ), failures
        browser.close()
