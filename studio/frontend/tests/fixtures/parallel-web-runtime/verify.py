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
html = "<!doctype html><html><head><link rel='stylesheet' href='https://assets.example/site.css'><style>#desktop-control{display:none}@media(min-width:500px){#desktop-control{display:block}}</style></head><body><button id='desktop-control' onclick='window.desktopClicks=(window.desktopClicks||0)+1'>Desktop action</button><div style='height:600px'></div><input id='value'><p id='mark'>Text to annotate here</p><img id='asset' src='https://assets.example/site.svg'><div style='height:1000px'></div><script>window.marker = Math.random()</script></body></html>"
with sync_playwright() as p:
    engine = os.environ.get("RUNTIME_BROWSER", "chromium")
    browser = getattr(p, "chromium" if engine == "edge" else engine).launch(
        headless = True,
        args = ["--disable-gpu"] if engine in ["chromium", "edge"] else [],
        **({"channel": "msedge"} if engine == "edge" else {}),
    )
    page = browser.new_page()
    assets = []

    def asset(route):
        assets.append(route.request.url)
        css = route.request.url.endswith(".css")
        route.fulfill(
            content_type = "text/css" if css else "image/svg+xml",
            headers = {
                "Access-Control-Allow-Origin": "*",
                "Cross-Origin-Resource-Policy": "cross-origin",
            },
            body = "#value{color:rgb(14,97,43)}"
            if css
            else '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16"><rect width="16" height="16" fill="red"/></svg>',
        )

    page.route("https://assets.example/**", asset)
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
    page.goto(
        os.environ.get("RUNTIME_BASE_URL", "http://127.0.0.1:15429")
        + "/tests/fixtures/parallel-web-runtime/"
    )
    page.wait_for_function("window.store != null")
    a = page.evaluate(
        """() => { const s=window.store.getState(); s.showChat('single:a'); s.openUrl('https://a.example/'); return window.store.getState().activeTabId; }"""
    )
    assert page.evaluate("(id) => window.whenWebPageReady(id)", a)
    shell_frame = next(f for f in page.frames if "/api/browser/frame?" in f.url)
    f = shell_frame.child_frames[0]
    f.locator("#value").fill("kept")
    f.wait_for_function(
        "getComputedStyle(document.querySelector('#value')).color === 'rgb(14, 97, 43)'"
    )
    f.wait_for_function("document.querySelector('#asset').naturalWidth === 16")
    f.evaluate("scrollTo(0,400)")
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
    b_frame.locator("#value").fill("second")
    before_float = b_frame.evaluate("({width:innerWidth,marker:window.marker})")
    page.evaluate("window.setPreviewScale(true)")
    page.wait_for_function("document.querySelector('#slot').getBoundingClientRect().width === 300")
    page.wait_for_function(
        "frame => Math.abs(frame.getBoundingClientRect().width - 300) < 1",
        arg = b_shell.frame_element(),
    )
    print(
        "FLOATING_VIEWPORT",
        b_frame.evaluate(
            "({width:innerWidth,desktopActionVisible:getComputedStyle(document.querySelector('#desktop-control')).display !== 'none'})"
        ),
        flush = True,
    )
    assert b_frame.locator(
        "#desktop-control"
    ).is_visible(), "Floating hid the desktop action by shrinking the page viewport"
    assert b_frame.evaluate("innerWidth") == before_float["width"]
    b_frame.locator("#desktop-control").scroll_into_view_if_needed()
    button_box = b_frame.locator("#desktop-control").evaluate(
        "e => e.getBoundingClientRect().toJSON()"
    )
    shell_box = b_shell.frame_element().bounding_box()
    assert shell_box
    inner_box = b_frame.frame_element().evaluate("e => e.getBoundingClientRect().toJSON()")
    # Edge's Playwright locator omits the outer iframe scale.
    scale = shell_box["width"] / b_shell.evaluate("innerWidth")
    page.mouse.click(
        shell_box["x"] + (inner_box["x"] + button_box["x"] + button_box["width"] / 2) * scale,
        shell_box["y"] + (inner_box["y"] + button_box["y"] + button_box["height"] / 2) * scale,
    )
    assert b_frame.evaluate("window.desktopClicks") == 1
    assert b_frame.evaluate("window.marker") == before_float["marker"]
    page.evaluate("window.setPreviewScale(false)")
    page.wait_for_function(
        "frame => Math.abs(frame.getBoundingClientRect().width - 600) < 1",
        arg = b_shell.frame_element(),
    )
    assert b_frame.evaluate("innerWidth") == before_float["width"]
    assert b_frame.locator("#value").input_value() == "second"
    asset_count = len(assets)
    viewport = f.evaluate("[innerWidth,innerHeight,scrollY]")
    page.evaluate("(id) => window.watchTab(id)", a)
    readonly = page.locator(f'[data-web-readonly="{a}"] iframe')
    readonly.wait_for(state = "visible")
    assert readonly.evaluate("frame => frame.parentElement.inert")
    assert f.evaluate("[innerWidth,innerHeight,scrollY]") == viewport
    assert (
        f.locator("#value").evaluate("element => getComputedStyle(element).color")
        == "rgb(14, 97, 43)"
    )
    assert f.locator("#asset").evaluate("image => image.naturalWidth") == 16
    box = f.locator("#value").bounding_box()
    assert box
    page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
    page.keyboard.type("must not enter A")
    page.keyboard.press("Enter")
    assert f.locator("#value").input_value() == "kept"
    assert b_frame.locator("#value").input_value() == "second"
    assert f.evaluate("window.marker") == marker
    assert f.evaluate("[innerWidth,innerHeight,scrollY]") == viewport
    assert len(assets) == asset_count
    assert page.evaluate("window.store.getState().activeTabId") == b
    assert page.evaluate("window.store.getState().chat") == "single:b"
    assert not page.evaluate(
        '([id,frame]) => window.browserScopeContains(document.querySelector("#slot"),id,frame)',
        [b, shell_frame.frame_element()],
    )
    assert page.evaluate("id => window.requestFrameAgent(id, {op:'status'}, '')", a)["result"][
        "virtualInput"
    ]
    page.evaluate("window.watchTab(null)")
    page.wait_for_function("frame => !frame.parentElement.inert", arg = b_shell.frame_element())
    b_frame.locator("#value").focus()
    b_frame.evaluate("window.blurs=0")
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
    page.locator("#slot").evaluate("element => element.style.width='300px'")
    page.wait_for_function(
        "frame => frame.parentElement.style.clipPath.includes('300px')",
        arg = shell_frame.frame_element(),
    )
    slot = page.locator("#slot").bounding_box()
    assert slot
    assert page.evaluate(
        "([x,y,frame]) => document.elementFromPoint(x,y) !== frame",
        [slot["x"] + slot["width"] + 30, slot["y"] + 150, shell_frame.frame_element()],
    )
    page.locator("#slot").evaluate("element => element.style.width='600px'")
    page.evaluate("window.store.getState().closePanel()")
    assert f.evaluate("window.marker") == marker
    print(
        f"PASS: {engine} retained A iframe, assets, scroll and form state through B and read-only viewing; input isolation, annotations and clipping passed"
    )
    browser.close()
