# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows players end to end against a mock backend (studio/frontend/smoke-flows-player.html).

Story: begin, read, pick choices, save, and reach the ending the compiled Lantern Picnic example promises.
World: start the Phaser village, walk to Old Tom, start his quest, pick up the rod, return it and see the
quest complete. World 3D: the same quest in the first-person three.js view, keyboard walking and turning, and
the map / 3D toggle keeping the player where they stand. Screenshots go to $FLOWS_SHOTS (default ./outputs/flows)."""

from __future__ import annotations

import json
import os
import struct
import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _playwright_robust import (  # noqa: E402
    chromium_launch_args,
    start_vite,
    stop_process,
    wait_for_smoke_page,
)

PORT = int(os.environ.get("SMOKE_PORT", "5431"))
BASE = f"http://127.0.0.1:{PORT}/smoke-flows-player.html"
SHOTS = Path(os.environ.get("FLOWS_SHOTS", "outputs/flows"))


def shot(page: Page, name: str) -> None:
    SHOTS.mkdir(parents = True, exist_ok = True)
    page.screenshot(path = str(SHOTS / f"player_{name}.png"))


def play_story(page: Page) -> None:
    page.goto(f"{BASE}?flow=story")
    page.wait_for_selector("[data-testid=story-begin]", timeout = 120_000)
    arm(page)
    page.wait_for_timeout(600)
    shot(page, "story_title")
    page.click("[data-testid=story-begin]")
    page.wait_for_selector("[data-testid=story-dialogue]")
    page.wait_for_timeout(1200)
    shot(page, "story_dialogue")
    chose = False
    for _ in range(200):
        if page.locator("[data-testid=story-ending]").count():
            break
        choices = page.locator("[data-testid=story-choice]")
        if choices.count():
            if not chose:
                page.wait_for_timeout(500)
                shot(page, "story_choice")
                page.click("button[aria-label=Save]")
                page.get_by_text("Slot 1").first.click()
                page.wait_for_function("window.__flowsPlayerSmoke.sessions.has('story-1')")
            creek = choices.filter(has_text = "creek")
            (creek.first if creek.count() else choices.first).click()
            chose = True
        else:
            page.keyboard.press("Space")
        page.wait_for_timeout(40)
    page.wait_for_selector("[data-testid=story-ending]", timeout = 10_000)
    title = page.locator("[data-testid=story-ending] h2").inner_text()
    assert title == "The Cake Incident", title
    page.wait_for_timeout(900)
    shot(page, "story_ending")
    page.get_by_test_id("story-ending").get_by_role("button", name = "Back to the editor").click()
    assert page.evaluate("window.__flowsPlayerSmoke.exited") == 1


def walk_to(page: Page, kind: str, entity: str, label: str) -> None:
    prompt = page.get_by_text(label)
    for _ in range(30):
        if kind == "npcs":
            tile = page.evaluate(f"window.__flowsPlayerSmoke.world.npcTile({entity!r})")
        else:
            tile = page.evaluate(f"window.__flowsPlayerSmoke.map.items[{entity!r}]")
        page.evaluate(
            f"window.__flowsPlayerSmoke.world.walkTo({{x: {tile['x']}, y: {tile['y'] + 1}}})"
        )
        try:
            prompt.wait_for(timeout = 1_500)
            return
        except Exception:  # noqa: BLE001 - keep chasing
            continue
    prompt.wait_for(timeout = 1_000)


def talk(page: Page, message: str, expect: str) -> None:
    page.keyboard.press("e")
    page.wait_for_selector("[data-testid=world-dialogue]")
    page.fill("[data-testid=world-say]", message)
    page.keyboard.press("Enter")
    page.get_by_text(expect).first.wait_for(timeout = 10_000)


def play_world(page: Page) -> None:
    page.goto(f"{BASE}?flow=world")
    page.wait_for_selector("[data-testid=world-start]", timeout = 120_000)
    arm(page)
    page.wait_for_timeout(400)
    shot(page, "world_intro")
    page.click("[data-testid=world-start]")
    page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
    page.wait_for_timeout(1200)
    shot(page, "world_explore")
    before = page.evaluate("window.__flowsPlayerSmoke.world.playerTile()")
    moved = None
    for key in ("d", "a", "s", "w"):
        page.keyboard.down(key)
        page.wait_for_timeout(1_200)
        page.keyboard.up(key)
        after = page.evaluate("window.__flowsPlayerSmoke.world.playerTile()")
        if after != before:
            moved = key
            break
    assert moved, f"the keyboard never moved the player from {before}"

    walk_to(page, "npcs", "tom", "Talk to Old Tom")
    talk(page, "Hello there!", "(tom) You said: Hello there!")
    page.wait_for_timeout(400)
    shot(page, "world_dialogue")
    page.get_by_role("button", name = "Bye").click()

    walk_to(page, "items", "rod", "Pick up Lucky Rod")
    page.keyboard.press("e")
    page.get_by_text("Picked up Lucky Rod").wait_for()

    walk_to(page, "npcs", "tom", "Talk to Old Tom")
    talk(page, "I found your rod", "My lucky rod! You found it!")
    page.get_by_text("Quest complete: The Lucky Rod", exact = False).wait_for()
    page.get_by_role("button", name = "Bye").click()
    page.keyboard.press("q")
    page.wait_for_timeout(500)
    shot(page, "world_quest_done")
    calls = page.evaluate("window.__flowsPlayerSmoke.calls")
    assert [c["entry"] for c in calls] == ["reply", "reply"], calls
    assert calls[1]["inputs"]["state"]["inventory"] == ["rod"], calls[1]


def play_world_3d(page: Page) -> None:
    page.goto(f"{BASE}?flow=world&render=3d&time=dusk")
    page.wait_for_selector("[data-testid=world-start]", timeout = 120_000)
    arm(page)
    page.wait_for_selector("[data-testid=world3d-canvas]")
    page.click("[data-testid=world-start]")
    page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
    page.wait_for_timeout(1500)
    shot(page, "world3d_explore")
    still = page.screenshot()
    page.keyboard.down("ArrowLeft")
    page.wait_for_timeout(700)
    page.keyboard.up("ArrowLeft")
    page.wait_for_timeout(200)
    assert page.screenshot() != still, "turning did not change the 3D view"
    before = page.evaluate("window.__flowsPlayerSmoke.world.playerTile()")
    moved = None
    for key in ("w", "s", "a", "d"):
        page.keyboard.down(key)
        page.wait_for_timeout(1_000)
        page.keyboard.up(key)
        if page.evaluate("window.__flowsPlayerSmoke.world.playerTile()") != before:
            moved = key
            break
    assert moved, f"the keyboard never moved the player from {before}"

    walk_to(page, "npcs", "fern", "Talk to Grandma Fern")
    for _ in range(2):
        page.keyboard.down("s")
        page.wait_for_timeout(300)
        page.keyboard.up("s")
    page.evaluate("window.__flowsPlayerSmoke.world.setNpcStatus('fern', 'zz')")
    page.evaluate("window.__flowsPlayerSmoke.world.say('fern', 'Have you eaten? Sit, sit.', 6000)")
    page.wait_for_timeout(500)
    shot(page, "world3d_capsule")
    walk_to(page, "npcs", "tom", "Talk to Old Tom")
    page.wait_for_timeout(300)
    shot(page, "world3d_npc")
    talk(page, "Hello there!", "(tom) You said: Hello there!")
    page.wait_for_timeout(600)
    shot(page, "world3d_dialogue")
    page.get_by_role("button", name = "Bye").click()

    walk_to(page, "items", "rod", "Pick up Lucky Rod")
    page.wait_for_timeout(300)
    shot(page, "world3d_item")
    page.keyboard.press("e")
    page.get_by_text("Picked up Lucky Rod").wait_for()

    walk_to(page, "npcs", "tom", "Talk to Old Tom")
    talk(page, "I found your rod", "My lucky rod! You found it!")
    page.get_by_text("Quest complete: The Lucky Rod", exact = False).wait_for()
    page.get_by_role("button", name = "Bye").click()

    here = page.evaluate("window.__flowsPlayerSmoke.world.playerTile()")
    page.click("[data-testid=world-view-toggle]")
    page.wait_for_function("!document.querySelector('[data-testid=world3d-canvas]')")
    page.wait_for_function(
        "window.__flowsPlayerSmoke.world && document.querySelector('[data-testid=world-canvas] canvas')"
    )
    page.wait_for_timeout(900)
    assert page.evaluate("window.__flowsPlayerSmoke.world.playerTile()") == here
    shot(page, "world3d_to_map")
    page.click("[data-testid=world-view-toggle]")
    page.wait_for_selector("[data-testid=world3d-canvas]")
    page.wait_for_timeout(900)
    assert page.evaluate("window.__flowsPlayerSmoke.world.playerTile()") == here
    calls = page.evaluate("window.__flowsPlayerSmoke.calls")
    assert [c["entry"] for c in calls] == ["reply", "reply"], calls

    for t in ("dawn", "day", "night"):
        page.goto(f"{BASE}?flow=world&render=3d&time={t}")
        arm(page)
        page.click("[data-testid=world-start]")
        page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
        page.wait_for_timeout(1200)
        shot(page, f"world3d_{t}")


def box_glb(extra: dict | None = None) -> bytes:
    """A small coloured box as a binary glTF: what a Blender export of a prop looks like."""
    pos = [(x, y, z) for x in (-0.5, 0.5) for y in (0, 1) for z in (-0.5, 0.5)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    idx = [i for a, b, c, d in quads for i in (a, b, c, a, c, d)]
    vbytes = b"".join(struct.pack("<3f", *p) for p in pos)
    ibytes = b"".join(struct.pack("<H", i) for i in idx)
    ibytes += b"\0" * (-len(ibytes) % 4)
    doc = {
        "asset": {"version": "2.0"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [0.55, 0.35, 0.9, 1]}}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "buffers": [{"byteLength": len(vbytes) + len(ibytes)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(vbytes)},
            {"buffer": 0, "byteOffset": len(vbytes), "byteLength": len(idx) * 2},
        ],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3",
             "min": [-0.5, 0, -0.5], "max": [0.5, 1, 0.5]},
            {"bufferView": 1, "componentType": 5123, "count": len(idx), "type": "SCALAR"},
        ],
        **(extra or {}),
    }  # fmt: skip
    text = json.dumps(doc).encode()
    text += b" " * (-len(text) % 4)
    body = vbytes + ibytes
    chunks = (
        struct.pack("<II", len(text), 0x4E4F534A)
        + text
        + struct.pack("<II", len(body), 0x004E4942)
        + body
    )
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks


def edit_world_3d(page: Page) -> None:
    """The scene editor: add props, undo / redo, import a model (and refuse a bad one), delete, export, save,
    then see the scenery in the 2D map and again after a reload."""
    page.goto(f"{BASE}?flow=world&render=3d&time=day&props=reset")
    page.wait_for_selector("[data-testid=world-start]", timeout = 120_000)
    arm(page)
    page.click("[data-testid=world-start]")
    page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
    page.click("[data-testid=world-edit-toggle]")
    page.wait_for_selector("[data-testid=world-editor]")
    items = page.locator("[data-testid=world-editor-item]")
    for kind in ("house", "well", "stall", "lamp", "tree"):
        page.click(f"[data-testid=world-editor-add-{kind}]")
    assert items.count() == 5, items.count()
    page.get_by_role("button", name = "Undo").click()
    assert items.count() == 4
    page.get_by_role("button", name = "Redo").click()
    assert items.count() == 5
    SHOTS.mkdir(parents = True, exist_ok = True)
    good, bad = SHOTS / "boat.glb", SHOTS / "remote.glb"
    good.write_bytes(box_glb())
    bad.write_bytes(box_glb({"images": [{"uri": "https://example.com/t.png"}]}))
    page.set_input_files("[data-testid=world-editor-file]", str(bad))
    page.get_by_role("alert").filter(has_text = "outside files").wait_for()
    page.set_input_files("[data-testid=world-editor-file]", str(good))
    page.wait_for_function(
        "document.querySelectorAll('[data-testid=world-editor-item]').length === 6"
    )
    assert "boat" in items.nth(5).inner_text()
    page.wait_for_timeout(800)
    shot(page, "world3d_editor")
    items.nth(0).click()
    page.locator("[data-testid=world3d-canvas]").hover()
    page.keyboard.press("x")
    assert items.count() == 5
    page.keyboard.press("Control+z")
    assert items.count() == 6
    with page.expect_download() as dl:
        page.get_by_role("button", name = "Export .glb").click()
    exported = Path(dl.value.path()).read_bytes()
    assert exported[:4] == b"glTF" and len(exported) > 10_000, len(exported)
    page.click("[data-testid=world-editor-done]")
    page.wait_for_selector("[data-testid=world-editor]", state = "detached")
    saved = page.evaluate("JSON.parse(sessionStorage.getItem('flows-smoke-props3d'))")
    assert [p["kind"] for p in saved] == ["house", "well", "stall", "lamp", "tree", "model"], saved
    page.click("[data-testid=world-view-toggle]")
    page.wait_for_function(
        "window.__flowsPlayerSmoke.world && document.querySelector('[data-testid=world-canvas] canvas')"
    )
    page.wait_for_timeout(900)
    shot(page, "world_props_2d")
    page.goto(f"{BASE}?flow=world&render=3d&time=dusk")
    arm(page)
    page.click("[data-testid=world-start]")
    page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
    page.click("[data-testid=world-edit-toggle]")
    page.wait_for_selector("[data-testid=world-editor]")
    assert items.count() == 6
    page.wait_for_timeout(1200)
    shot(page, "world3d_props_reloaded")
    page.click("[data-testid=world-editor-done]")


STAMP: dict[str, float] = {}


def arm(page: Page) -> None:
    STAMP["value"] = page.evaluate("window.__flowsStamp = Math.random()")


def reloaded(page: Page) -> bool:
    return "value" in STAMP and page.evaluate("window.__flowsStamp") != STAMP["value"]


def retry_on_reload(
    page: Page,
    scenario,
    attempts: int = 3,
) -> None:
    """The dev server reloads the page when source files change under it (other edits, Tailwind's scan).
    A reload wipes the harness state; rerun the scenario then, and only then."""
    for attempt in range(attempts):
        STAMP.clear()
        try:
            scenario(page)
            return
        except Exception:
            if not reloaded(page) or attempt == attempts - 1:
                raise
            print(f"{scenario.__name__}: the dev server reloaded the page; retrying")


def main() -> int:
    proc = start_vite(PORT)
    try:
        wait_for_smoke_page(BASE, "smoke-flows-player-main.tsx", proc = proc)
        with sync_playwright() as p:
            # Software WebGL for the 3D view on GPU-less runners.
            browser = p.chromium.launch(
                args = chromium_launch_args() + ["--enable-unsafe-swiftshader"]
            )
            page = browser.new_page(viewport = {"width": 1280, "height": 800})
            errors: list[str] = []
            # Warm-up: the first visit makes Vite optimize Phaser and reload the page; play after that.
            page.goto(f"{BASE}?flow=world")
            page.wait_for_selector("[data-testid=world-start]", timeout = 180_000)
            page.wait_for_timeout(3_000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            only = os.environ.get("FLOWS_SCENARIOS")
            scenarios = (play_story, play_world, play_world_3d, edit_world_3d)
            for scenario in scenarios:
                if only and scenario.__name__ not in only.split(","):
                    continue
                retry_on_reload(page, scenario)
            browser.close()
            assert not errors, errors
    finally:
        stop_process(proc)
    print("flows player smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
