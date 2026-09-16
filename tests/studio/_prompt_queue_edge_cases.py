# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

"""Additional queue gesture and input regressions, shared by browser engines."""

import re
import json
import time
from playwright.sync_api import expect


def check_edge_cases(page):
    rows = page.locator("[data-queue-item-id]")
    attempts = page.get_by_label("Move attempts", exact = True)

    def reset():
        page.get_by_role("button", name = "Reset fixture", exact = True).click()
        expect(rows).to_have_count(3)
        expect(rows.first).to_have_attribute("data-queue-item-id", "q0")
        expect(attempts).to_have_text("0")
        rows.evaluate_all(
            "rows => Promise.all(rows.flatMap(row => row.getAnimations()).map(a => a.finished.catch(() => {})))"
        )

    def grab():
        handle = rows.first.get_by_role("button", name = re.compile("^Reorder"))
        box = handle.bounding_box()
        x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.mouse.down()
        page.mouse.move(x, y + 60, steps = 6)
        expect(rows.first).to_have_attribute("data-queue-dragging", "true")
        return handle

    for cancel in ("blur", "resize", "lostcapture", "lock", "remove"):
        reset()
        handle = grab()
        if cancel in ("blur", "resize"):
            page.evaluate("name => window.dispatchEvent(new Event(name))", cancel)
        elif cancel == "lostcapture":
            handle.dispatch_event("lostpointercapture")
        else:
            name = "Lock prompts" if cancel == "lock" else "Dispatch first"
            page.get_by_role("button", name = name, exact = True).evaluate("el => el.click()")
        page.mouse.up()
        expect(page.locator('[data-queue-dragging="true"]')).to_have_count(0)
        expect(attempts).to_have_text("0")
        assert rows.evaluate_all("rows => rows.every(row => !row.style.transform)")

    # A short click and a secondary pointer must not reorder anything.
    reset()
    handle = rows.first.get_by_role("button", name = re.compile("^Reorder"))
    handle.click()
    handle.click(button = "right")
    handle.dispatch_event("pointerdown", {"pointerId": 99, "isPrimary": False, "button": 0})
    expect(attempts).to_have_text("0")
    expect(page.locator('[data-queue-dragging="true"]')).to_have_count(0)

    # Mixed keyboard and pointer input commits at most one move.
    reset()
    grab()
    page.keyboard.press("End")
    page.mouse.up()
    expect(attempts).to_have_text("1")
    expect(rows.last).to_have_attribute("data-queue-item-id", "q0")

    # Some IMEs report legacy keyCode 229 after isComposing becomes false.
    reset()
    rows.first.get_by_role("button", name = re.compile("^More options")).click()
    page.get_by_role("menuitem", name = "Edit message", exact = True).click()
    editor = page.get_by_role("textbox", name = "Edit queued prompt 1", exact = True)
    editor.fill("日本語の下書き")
    editor.dispatch_event("keydown", {"key": "Enter", "ctrlKey": True, "keyCode": 229})
    expect(editor).to_be_visible()
    expect(editor).to_have_value("日本語の下書き")
    editor.press("Escape")
    expect(rows.first).to_contain_text("First prompt")

    for direction in ("ltr", "rtl"):
        for width, zoom in ((320, 1), (768, 0.8), (768, 1.25), (1440, 2)):
            print(f"LAYOUT: direction={direction}, width={width}, zoom={zoom}", flush = True)
            page.set_viewport_size({"width": width, "height": 1000})
            page.evaluate(
                "([dir, zoom]) => { document.documentElement.dir = dir; document.body.style.zoom = zoom; }",
                [direction, zoom],
            )
            reset()
            first = rows.first
            handle = first.get_by_role("button", name = re.compile("^Reorder"))
            box = handle.bounding_box()
            before = first.bounding_box()["y"]
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x, y + 14, steps = 4)
            expect(first).to_have_attribute("data-queue-dragging", "true")
            page.evaluate(
                "() => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)))"
            )
            displacement = first.bounding_box()["y"] - before
            assert abs(displacement - 14) < 1, (direction, zoom, displacement)
            target = rows.last.bounding_box()
            page.mouse.move(x, target["y"] + target["height"] / 2, steps = 6)
            expect(attempts).to_have_text("0")
            page.mouse.up()
            expect(attempts).to_have_text("1")
            expect(rows.last).to_have_attribute("data-queue-item-id", "q0")
            rows.evaluate_all(
                "rows => Promise.all(rows.flatMap(row => row.getAnimations()).map(a => a.finished.catch(() => {})))"
            )
            assert abs(rows.last.bounding_box()["y"] - target["y"]) < 1
            reset()
            last = rows.last.get_by_role("button", name = re.compile("^Reorder"))
            last.focus()
            last.press("Home")
            expect(rows.first).to_have_attribute("data-queue-item-id", "q2")
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), (
                direction,
                width,
                zoom,
            )
            rows.first.get_by_role("button", name = re.compile("^More options")).click()
            expect(page.get_by_role("menu")).to_be_visible()
            page.keyboard.press("Escape")
    page.evaluate("document.documentElement.dir = 'ltr'; document.body.style.zoom = ''")
    page.set_viewport_size({"width": 1100, "height": 800})
    print(
        "PASS: lifecycle cancellation, mixed inputs, IME, RTL, zoom and responsive queue layouts",
        flush = True,
    )


def check_performance(browser, url):
    for size in (3, 25, 100, 500):
        page = browser.new_page(viewport = {"width": 1100, "height": 850})
        try:
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            started = time.monotonic()
            page.goto(f"{url}?size={size}")
            rows = page.locator("[data-queue-item-id]")
            # Cold Vite imports are outside the drag frame measurement.
            expect(rows).to_have_count(size, timeout = 30000)
            print(
                f"MOUNT: rows={size}, cold_page_ms={round((time.monotonic() - started) * 1000)}",
                flush = True,
            )
            handle = rows.first.get_by_role("button", name = re.compile("^Reorder"))
            box = handle.bounding_box()
            x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x, y + 60, steps = 6)
            expect(rows.first).to_have_attribute("data-queue-dragging", "true")
            frames = page.evaluate("""() => new Promise(resolve => {
              const deltas = []; let previous;
              const sample = time => {
                if (previous !== undefined) deltas.push(time - previous);
                previous = time;
                if (deltas.length === 60) resolve(deltas);
                else requestAnimationFrame(sample);
              };
              requestAnimationFrame(sample);
            })""")
            page.mouse.up()
            expect(page.get_by_label("Move attempts", exact = True)).to_have_text("1")
            rows.evaluate_all(
                "rows => Promise.all(rows.flatMap(row => row.getAnimations()).map(a => a.finished.catch(() => {})))"
            )
            idle_mutations = page.evaluate("""() => new Promise(resolve => {
              let count = 0;
              const observer = new MutationObserver(records => {
                count += records.filter(r => r.target.matches('[data-queue-item-id]') && r.attributeName === 'style').length;
              });
              observer.observe(document.querySelector('[aria-label="Queued prompts"]'), {subtree:true, attributes:true});
              setTimeout(() => { observer.disconnect(); resolve(count); }, 300);
            })""")
            assert idle_mutations == 0, "idle queue continues updating row styles"
            assert rows.evaluate_all("rows => rows.every(row => !row.style.transform)")
            ordered = sorted(frames)
            metrics = {
                "rows": size,
                "p95_frame_ms": round(ordered[int(len(ordered) * 0.95) - 1], 2),
                "max_frame_ms": round(max(frames), 2),
                "idle_style_updates": idle_mutations,
            }
            print("PERF: " + json.dumps(metrics), flush = True)
            assert metrics["p95_frame_ms"] < 250, "drag stalls for more than 250ms per frame"
            assert not errors, errors
        finally:
            page.close()
