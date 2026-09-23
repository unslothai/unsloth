# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Exercise the real queue view: menus, editing, ordering, drag, touch and file drops.

Run after npm ci and Playwright browser installation:
    python tests/studio/playwright_prompt_queue_actions.py
PW_ENGINE=webkit selects WebKit. No backend or inference is required.
"""

import os
import re
from playwright.sync_api import expect, sync_playwright
from _playwright_robust import start_vite, stop_process, wait_for_smoke_page

PAGE = "/smoke-prompt-queue-actions.html"
ENTRY = "/smoke-prompt-queue-actions-main.tsx"


def check_form_actions(page):
    controls = [
        ("Reorder queued prompt 1 of 3", False),
        ("Steer with queued prompt 1", False),
        ("Remove queued prompt 1", False),
        ("More options for queued prompt 1", False),
        ("Steer with queued prompt 1", True),
    ]
    for name, reject in controls:
        for keyboard in (False, True):
            page.get_by_role("button", name="Reset fixture", exact=True).click()
            if reject:
                page.get_by_role("button", name="Simulate dispatch race", exact=True).click()
            button = page.get_by_role("button", name=name, exact=True)
            if keyboard:
                button.focus()
                button.press("Enter")
            else:
                button.click()
            expect(page.get_by_label("Composer submissions", exact=True)).to_have_text("0")
            expect(page.get_by_role("textbox", name="Composer draft", exact=True)).to_have_value(
                "Unsent composer draft"
            )
            page.keyboard.press("Escape")
    print("PASS: queue controls preserve the unsent composer draft without submitting", flush=True)


def check_actions(page):
    rows = page.locator("[data-queue-item-id]")

    def order(ids):
        expect(rows).to_have_count(len(ids))
        page.wait_for_function(
            "ids => JSON.stringify([...document.querySelectorAll('[data-queue-item-id]')].map(e => e.dataset.queueItemId)) === JSON.stringify(ids)",
            arg=ids,
        )

    def menu(position):
        expect(page.get_by_role("menu")).to_have_count(0)
        page.get_by_role(
            "button", name=f"More options for queued prompt {position}", exact=True
        ).click()
        expect(page.get_by_role("menu")).to_be_visible()

    def action(name):
        page.get_by_role("menuitem", name=name, exact=True).click()
        expect(page.get_by_role("menu")).to_have_count(0)

    order(["q0", "q1", "q2"])
    behavior = page.get_by_label("Follow-up behavior", exact=True)
    expect(behavior).to_have_text("queue")
    expect(page.get_by_role("button", name=re.compile(r"^Steer with queued prompt"))).to_have_count(
        3
    )
    menu(3)
    expect(page.get_by_role("menuitem")).to_have_text(
        ["Edit message", "Copy message", "Turn off queueing"]
    )
    expect(page.get_by_role("menuitem", name=re.compile(r"^Move"))).to_have_count(0)
    page.keyboard.press("Escape")

    menu(1)
    action("Turn off queueing")
    expect(behavior).to_have_text("steer")
    order(["q0", "q1", "q2"])
    expect(page.get_by_text("Paused", exact=True)).to_have_count(0)
    expect(page.get_by_label("Steered prompt", exact=True)).to_be_empty()
    page.reload()
    expect(behavior).to_have_text("steer")
    menu(2)
    action("Turn on queueing")
    expect(behavior).to_have_text("queue")
    order(["q0", "q1", "q2"])
    expect(page.get_by_text("Paused", exact=True)).to_have_count(0)
    expect(page.get_by_label("Steered prompt", exact=True)).to_be_empty()

    menu(2)
    action("Edit message")
    editor = page.get_by_role("textbox", name="Edit queued prompt 2", exact=True)
    expect(editor).to_be_focused()
    editor.fill(" ")
    expect(page.get_by_role("button", name="Save", exact=True)).to_be_disabled()
    editor.fill("Changed second prompt\nWith another line")
    editor.press("Control+Enter")
    expect(rows.nth(1)).to_contain_text("Changed second prompt")
    edit_button = page.get_by_role("button", name="More options for queued prompt 2", exact=True)
    expect(edit_button).to_be_focused()
    edit_button.click()
    action("Edit message")
    editor.fill("Discard this edit")
    editor.press("Escape")
    expect(rows.nth(1)).to_contain_text("Changed second prompt")
    expect(edit_button).to_be_focused()

    handle = page.get_by_role("button", name="Reorder queued prompt 3 of 3", exact=True)
    handle.focus()
    page.keyboard.press("Home")
    order(["q2", "q0", "q1"])
    page.keyboard.press("ArrowDown")
    order(["q0", "q2", "q1"])
    page.keyboard.press("End")
    order(["q0", "q1", "q2"])
    source = handle.bounding_box()
    target = rows.first.bounding_box()
    page.mouse.move(source["x"] + source["width"] / 2, source["y"] + source["height"] / 2)
    page.mouse.down()
    page.mouse.move(target["x"] + target["width"] / 2, target["y"] + target["height"] / 2, steps=5)
    page.keyboard.press("Escape")
    page.mouse.up()
    order(["q0", "q1", "q2"])
    handle.drag_to(rows.first)
    order(["q2", "q0", "q1"])
    expect(rows.first).not_to_have_class(re.compile(r"opacity-40"))

    # Dragging a file must remain available to the outer attachment dropzone.
    assert rows.first.evaluate("""row => {
      const transfer = new DataTransfer();
      transfer.items.add(new File(['file'], 'example.txt', {type:'text/plain'}));
      const event = new DragEvent('dragover', {bubbles:true, cancelable:true, dataTransfer:transfer});
      row.dispatchEvent(event);
      return !event.defaultPrevented;
    }""")

    page.get_by_role("button", name="Simulate paused queue", exact=True).click()
    expect(page.get_by_text("Paused", exact=True)).to_be_visible()
    menu(1)
    action("Turn off queueing")
    expect(behavior).to_have_text("steer")
    expect(page.get_by_text("Paused", exact=True)).to_be_visible()
    order(["q2", "q0", "q1"])
    menu(1)
    action("Resume queue")
    expect(page.get_by_text("Paused", exact=True)).to_have_count(0)
    expect(behavior).to_have_text("steer")
    page.get_by_role("button", name="Simulate dispatch race").click()
    page.get_by_role("button", name="Reorder queued prompt 3 of 3", exact=True).focus()
    page.keyboard.press("Home")
    order(["q2", "q0", "q1"])
    expect(page.get_by_role("status").filter(has_text="queue changed")).to_be_attached()
    page.get_by_role("button", name="Steer with queued prompt 3", exact=True).click()
    order(["q2", "q0", "q1"])
    expect(page.get_by_role("status").filter(has_text="could not steer")).to_be_attached()
    page.get_by_role("button", name="Remove queued prompt 2", exact=True).click()
    order(["q2", "q1"])
    page.get_by_role("button", name="Remove queued prompt 2", exact=True).click()
    order(["q2"])
    expect(
        page.get_by_role("button", name="Reorder queued prompt 1 of 1", exact=True)
    ).to_be_disabled()
    menu(1)
    expect(page.get_by_role("menuitem", name=re.compile(r"^Move"))).to_have_count(0)
    page.keyboard.press("Escape")
    expect(
        page.get_by_role("button", name="More options for queued prompt 1", exact=True)
    ).to_be_focused()
    page.get_by_role("button", name="Reset fixture", exact=True).click()
    page.get_by_role("button", name="Simulate paused queue", exact=True).click()
    page.get_by_role("button", name="Steer with queued prompt 2", exact=True).click()
    order(["q0", "q2"])
    expect(page.get_by_label("Steered prompt", exact=True)).to_have_text("Second prompt")
    expect(page.get_by_text("Paused", exact=True)).to_have_count(0)
    page.get_by_role("button", name="Lock prompts", exact=True).click()
    expect(
        page.get_by_role("button", name="Steer with queued prompt 1", exact=True)
    ).to_be_disabled()
    expect(
        page.get_by_role("button", name="Steer with queued prompt 2", exact=True)
    ).to_be_disabled()
    print(
        "PASS: compact menu, persistent queueing preference, steer, edit, keyboard, mouse drag, file drop, resume, races and locked states",
        flush=True,
    )


def check_motion(page, reduced=False):
    rows = page.locator("[data-queue-item-id]")
    queue = page.locator('[aria-label^="Prompt queue,"]')
    attempts = page.get_by_label("Move attempts", exact=True)

    def reset():
        page.get_by_role("button", name="Reset fixture", exact=True).click()
        expect(rows).to_have_count(3)

    def row(item):
        return page.locator(f'[data-queue-item-id="{item}"]')

    def order(ids):
        assert rows.evaluate_all("rows => rows.map(row => row.dataset.queueItemId)") == ids

    def settle():
        rows.evaluate_all(
            "rows => Promise.all(rows.flatMap(row => row.getAnimations()).map(animation => animation.finished.catch(() => {})))"
        )

    def grab(item):
        handle = row(item).get_by_role("button", name=re.compile("^Reorder queued prompt"))
        box = handle.bounding_box()
        point = (box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.mouse.move(*point)
        page.mouse.down()
        return point

    reset()
    first = row("q0").bounding_box()
    third = row("q2").bounding_box()
    x, y = grab("q2")
    page.mouse.move(x, y - 14, steps=3)
    expect(row("q2")).to_have_attribute("data-queue-dragging", "true")
    page.wait_for_function(
        "top => Math.abs(document.querySelector('[data-queue-item-id=\"q2\"]').getBoundingClientRect().top - top) < 1",
        arg=third["y"] - 14,
    )
    order(["q0", "q1", "q2"])
    expect(attempts).to_have_text("0")

    page.mouse.move(x, first["y"] + first["height"] / 2 + 8, steps=8)
    page.wait_for_function(
        "top => Math.abs(document.querySelector('[data-queue-item-id=\"q0\"]').getBoundingClientRect().top - top) < 1",
        arg=first["y"] + third["height"],
    )
    if reduced:
        assert row("q0").evaluate("row => row.style.transition") == "none"
    order(["q0", "q1", "q2"])
    expect(attempts).to_have_text("0")
    page.mouse.up()
    expect(attempts).to_have_text("1")
    order(["q2", "q0", "q1"])
    if reduced:
        assert rows.evaluate_all(
            "rows => rows.every(row => row.getAnimations().every(animation => animation.effect.getKeyframes().every(frame => frame.transform === undefined)))"
        )
    settle()
    assert abs(row("q2").bounding_box()["y"] - first["y"]) < 1
    expect(page.locator('[data-queue-dragging="true"]')).to_have_count(0)

    if reduced:
        print("PASS: reduced motion keeps direct dragging without settling animations", flush=True)
        return

    # Cancelled gestures never reach the queue engine.
    for cancel in ("escape", "outside", "pointercancel"):
        reset()
        target = row("q2").bounding_box()
        x, y = grab("q0")
        page.mouse.move(x, target["y"] + target["height"] / 2, steps=6)
        expect(row("q0")).to_have_attribute("data-queue-dragging", "true")
        if cancel == "escape":
            page.keyboard.press("Escape")
        elif cancel == "outside":
            page.mouse.move(queue.bounding_box()["x"] - 30, y)
        else:
            row("q0").get_by_role(
                "button", name=re.compile("^Reorder queued prompt")
            ).dispatch_event("pointercancel")
        page.mouse.up()
        settle()
        order(["q0", "q1", "q2"])
        expect(attempts).to_have_text("0")
        expect(page.locator('[data-queue-dragging="true"]')).to_have_count(0)

    reset()
    page.get_by_role("button", name="Simulate dispatch race", exact=True).click()
    target = row("q0").bounding_box()
    x, y = grab("q2")
    page.mouse.move(x, target["y"] + target["height"] / 2, steps=6)
    page.mouse.up()
    expect(attempts).to_have_text("1")
    settle()
    order(["q0", "q1", "q2"])
    assert abs(row("q0").bounding_box()["y"] - target["y"]) < 1
    expect(page.get_by_role("status").filter(has_text="queue changed")).to_be_attached()

    reset()
    x, y = grab("q0")
    page.mouse.move(x, y + 60, steps=5)
    expect(row("q0")).to_have_attribute("data-queue-dragging", "true")
    page.get_by_role("button", name="Dispatch first", exact=True).evaluate(
        "button => button.click()"
    )
    expect(row("q0")).to_have_count(0)
    page.mouse.up()
    order(["q1", "q2"])
    expect(attempts).to_have_text("0")
    expect(page.locator('[data-queue-dragging="true"]')).to_have_count(0)

    # Holding at the edge scrolls without further pointer events.
    reset()
    page.get_by_role("button", name="Long queue", exact=True).click()
    expect(rows).to_have_count(12)
    bounds = queue.bounding_box()
    x, y = grab("long-0")
    page.mouse.move(x, bounds["y"] + bounds["height"] - 5, steps=10)
    page.wait_for_function(
        "() => { const list = document.querySelector('[aria-label^=\"Prompt queue,\"]'); return list.scrollTop >= list.scrollHeight - list.clientHeight - 1; }"
    )
    expect(attempts).to_have_text("0")
    page.mouse.up()
    expect(attempts).to_have_text("1")
    settle()
    order([f"long-{i}" for i in range(1, 12)] + ["long-0"])
    print(
        "PASS: continuous drag, sliding rows, deferred commit, cancellation, dispatch race and edge scrolling",
        flush=True,
    )


def check_touch(browser, url):
    context = browser.new_context(
        viewport={"width": 320, "height": 812}, is_mobile=True, has_touch=True
    )
    try:
        page = context.new_page()
        page.goto(url)
        source = page.get_by_role("button", name="Reorder queued prompt 3 of 3", exact=True)
        expect(source).to_be_visible()
        a = source.bounding_box()
        b = page.get_by_role(
            "button", name="Reorder queued prompt 1 of 3", exact=True
        ).bounding_box()
        session = context.new_cdp_session(page)
        x, y = a["x"] + a["width"] / 2, a["y"] + a["height"] / 2
        session.send(
            "Input.dispatchTouchEvent", {"type": "touchStart", "touchPoints": [{"x": x, "y": y}]}
        )
        for step in range(1, 7):
            session.send(
                "Input.dispatchTouchEvent",
                {
                    "type": "touchMove",
                    "touchPoints": [{"x": x, "y": y + (b["y"] - a["y"]) * step / 6}],
                },
            )
        session.send("Input.dispatchTouchEvent", {"type": "touchEnd", "touchPoints": []})
        expect(page.locator("[data-queue-item-id]").first).to_have_attribute(
            "data-queue-item-id", "q2"
        )
        page.get_by_role("button", name="More options for queued prompt 1", exact=True).tap()
        expect(page.get_by_role("menu")).to_be_visible()
        page.get_by_role("menuitem", name="Edit message", exact=True).tap()
        expect(page.get_by_role("textbox", name="Edit queued prompt 1", exact=True)).to_be_visible()
        page.get_by_role("button", name="Cancel", exact=True).tap()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        print("PASS: touch drag and menu editing at 320px", flush=True)
    finally:
        context.close()


def main():
    engine = os.environ.get("PW_ENGINE", "chromium")
    port = int(os.environ.get("PW_PORT", "5421"))
    server = None
    base = os.environ.get("BASE_URL")
    try:
        if not base:
            server = start_vite(port)
            base = f"http://127.0.0.1:{port}"
        url = base + PAGE
        wait_for_smoke_page(url, ENTRY, proc=server)
        with sync_playwright() as pw:
            options = {"headless": True}
            if os.environ.get("PW_EXECUTABLE"):
                options["executable_path"] = os.environ["PW_EXECUTABLE"]
            if os.environ.get("PW_CHANNEL"):
                options["channel"] = os.environ["PW_CHANNEL"]
            browser = getattr(pw, engine).launch(**options)
            print(f"Browser: {os.environ.get('PW_CHANNEL', engine)} {browser.version}", flush=True)
            try:
                page = browser.new_page(viewport={"width": 1100, "height": 800})
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                check_form_actions(page)
                page.get_by_role("button", name="Reset fixture", exact=True).click()
                check_actions(page)
                check_motion(page)
                from _prompt_queue_edge_cases import check_edge_cases

                check_edge_cases(page)
                assert not errors, errors
                reduced_context = browser.new_context(
                    reduced_motion="reduce", viewport={"width": 1100, "height": 800}
                )
                try:
                    reduced_page = reduced_context.new_page()
                    reduced_page.goto(url)
                    check_motion(reduced_page, reduced=True)
                finally:
                    reduced_context.close()
                if engine == "chromium":
                    check_touch(browser, url)
                from _prompt_queue_edge_cases import check_performance

                check_performance(browser, url)
            finally:
                browser.close()
    finally:
        if server:
            stop_process(server)


if __name__ == "__main__":
    main()
