# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import json
from playwright.sync_api import sync_playwright
import ast
import os
from pathlib import Path

source = (Path(__file__).resolve().parents[4] / "backend/routes/browser.py").read_text()
module = ast.parse(source)
shell = next(
    ast.literal_eval(node.value)
    for node in module.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_FRAME_HTML" for target in node.targets)
)
annotator = next(
    ast.literal_eval(node.value)
    for node in module.body
    if isinstance(node, ast.Assign)
    and any(isinstance(target, ast.Name) and target.id == "_ANNOTATE_JS" for target in node.targets)
)
html = "<!doctype html><html><body><input id='value'><p id='mark'>Text to annotate here</p><script>window.marker = Math.random()</script></body></html>"
with sync_playwright() as p:
    engine = os.environ.get("RUNTIME_BROWSER", "chromium")
    browser = getattr(p, "chromium" if engine == "edge" else engine).launch(
        headless = True,
        args = ["--disable-gpu"] if engine in ["chromium", "edge"] else [],
        **({"channel": "msedge"} if engine == "edge" else {}),
    )
    page = browser.new_page()
    page.route(
        "**/api/browser/fetch",
        lambda r: r.fulfill(
            status = 200,
            headers = {"X-Unsloth-Browser-Kind": "html"},
            content_type = "application/json",
            body = json.dumps(
                {
                    "html": (
                        "<!doctype html><html><body><input id='value' autofocus><script>document.querySelector('input').focus()</script></body></html>"
                        if "c.example" in r.request.post_data_json["url"]
                        else html
                    ),
                    "url": r.request.post_data_json["url"],
                    "base": None,
                }
            ),
        ),
    )
    page.route("**/api/browser/frame?*", lambda r: r.fulfill(content_type = "text/html", body = shell))
    page.route(
        "**/api/browser/annotate.js",
        lambda r: r.fulfill(content_type = "application/javascript", body = annotator),
    )
    page.goto("http://127.0.0.1:15429/tests/fixtures/parallel-web-runtime/")
    page.wait_for_function("window.store != null")
    a = page.evaluate(
        """() => { const s=window.store.getState(); s.showChat('single:a'); s.openUrl('https://a.example/'); return window.store.getState().activeTabId; }"""
    )
    assert page.evaluate("(id) => window.whenWebPageReady(id)", a)
    shell_frame = next(f for f in page.frames if "/api/browser/frame?" in f.url)
    f = shell_frame.child_frames[0]
    f.locator("#value").fill("kept")
    marker = f.evaluate("window.marker")
    page.evaluate(
        '(id) => window.agents.setState({chats: {"single:a": {tabId:id, shared:true}}})', a
    )
    b = page.evaluate(
        """() => { const s=window.store.getState(); s.showChat('single:b'); s.openUrl('https://b.example/'); return window.store.getState().activeTabId; }"""
    )
    assert page.evaluate("(id) => window.whenWebPageReady(id)", b)
    page.wait_for_function("f => f.parentElement.inert", arg = shell_frame.frame_element())
    assert not page.evaluate(
        "([id, frame]) => window.browserScopeContains(document.querySelector('#slot'), id, frame)",
        [b, shell_frame.frame_element()],
    )
    assert page.evaluate(
        "id => window.requestFrameAgent(id, {op: 'status'}, '({run: op => ({virtualInput: op.virtualInput === true})})')",
        a,
    )["result"]["virtualInput"]
    page.locator("#foreground").focus()
    page.evaluate(
        "document.querySelector('#foreground').addEventListener('blur', () => window.blurs = (window.blurs || 0) + 1)"
    )
    assert f.evaluate("document.querySelector('#value').value") == "kept"
    assert f.evaluate("window.marker") == marker
    f.evaluate("document.querySelector('#value').focus()")
    assert page.evaluate("document.activeElement.id") == "foreground"
    assert page.evaluate("window.blurs || 0") == 0
    b_shell = next(
        frame
        for frame in page.frames
        if "/api/browser/frame?" in frame.url and frame != shell_frame
    )
    b_frame = b_shell.child_frames[0]
    page.wait_for_function("f => !f.parentElement.inert", arg = b_shell.frame_element())
    assert page.evaluate(
        "([id, frame]) => window.browserScopeContains(document.querySelector('#slot'), id, frame)",
        [b, b_shell.frame_element()],
    )
    assert not page.evaluate(
        "([id, frame]) => window.browserScopeContains(document.querySelector('#slot'), id, frame)",
        [a, b_shell.frame_element()],
    )
    b_frame.locator("#value").focus()
    b_frame.evaluate(
        "document.querySelector('#value').addEventListener('blur', () => window.blurs = (window.blurs || 0) + 1)"
    )
    f.evaluate("document.querySelector('#value').focus()")
    assert b_frame.evaluate("document.activeElement.id") == "value"
    assert b_frame.evaluate("window.blurs || 0") == 0
    c = page.evaluate("window.store.getState().openTabIn('single:c', 'https://c.example/')")
    assert not page.evaluate("(id) => window.whenWebPageReady(id, undefined, 250)", c)
    page.evaluate(
        """id => window.agents.setState(state => ({chats: {...state.chats, "single:c": {tabId:id, working:true}}}))""",
        c,
    )
    assert page.evaluate("(id) => window.whenWebPageReady(id)", c)
    c_shell = next(
        frame
        for frame in page.frames
        if "/api/browser/frame?" in frame.url and frame not in [shell_frame, b_shell]
    )
    c_frame = c_shell.child_frames[0]
    c_identity = c_frame.evaluate("window.fixtureIdentity = Math.random()")
    page.evaluate(
        """id => window.agents.setState(state => ({chats: {...state.chats, "single:c": {tabId:id, shared:false, working:false, runKey:null}}}))""",
        c,
    )
    page.wait_for_timeout(250)
    assert c_frame.evaluate("window.fixtureIdentity") == c_identity
    assert page.evaluate("(id) => window.whenWebPageReady(id)", c)
    assert page.evaluate("window.store.getState().chat") == "single:b"
    assert b_frame.evaluate("document.activeElement.id") == "value"
    assert b_frame.evaluate("window.blurs || 0") == 0
    page.evaluate("window.store.getState().showChat('single:a')")
    assert f.evaluate("window.marker") == marker
    assert f.evaluate("document.querySelector('#value').value") == "kept"
    page.wait_for_function("f => !f.parentElement.inert", arg = shell_frame.frame_element())
    page.wait_for_function("""() => {
      const target = [...document.querySelectorAll('iframe')].find(f => !f.parentElement.inert);
      return target && target.offsetWidth > 0;
    }""")
    f.evaluate("document.querySelector('#value').focus()")
    assert f.evaluate("document.activeElement.id") == "value"
    page.evaluate("window.showAnnotation(true)")
    f.locator("unsloth-annotate").wait_for(state = "attached")
    f.locator("#mark").click()
    note = page.locator("[data-browser-annotation-portal] form input")
    note.wait_for(state = "visible")
    note.fill("Keep this annotation")
    assert note.evaluate(
        "(e, id) => window.browserScopeContains(document.querySelector('#slot'), id, e)", a
    )
    assert note.evaluate(
        "e => document.elementFromPoint(e.getBoundingClientRect().left + 5, e.getBoundingClientRect().top + 5) === e"
    )
    page.evaluate("window.showMenu(true)")
    page.locator('[data-slot="dialog-overlay"]').wait_for(state = "visible")
    assert note.evaluate(
        "e => !!document.elementFromPoint(e.getBoundingClientRect().left + 5, e.getBoundingClientRect().top + 5).closest('[data-slot=dialog-overlay], [data-slot=dialog-content]')"
    )
    page.evaluate("window.showMenu(false)")
    page.evaluate("window.observe(true)")
    portal = page.locator("[data-browser-annotation-portal]")
    assert portal.is_hidden()
    assert portal.evaluate("e => e.inert")
    assert not portal.evaluate("e => e.contains(document.activeElement)")
    page.evaluate("window.observe(false)")
    note.wait_for(state = "visible")
    assert note.input_value() == "Keep this annotation"
    page.evaluate("window.store.getState().closePanel()")
    assert f.evaluate("window.marker") == marker
    print(
        "PASS: Chromium retained A iframe and form state through B and panel hide; background focus did not steal B"
    )
    browser.close()
