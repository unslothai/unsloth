# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Flows players end to end against a mock backend (studio/frontend/smoke-flows-player.html).

Story: begin, read, pick choices, save, and reach the ending the compiled Lantern Picnic example promises.
World: start the Phaser village, walk to Old Tom, start his quest, pick up the rod, return it and see the
quest complete. Screenshots go to $FLOWS_SHOTS (default ./outputs/flows)."""

from __future__ import annotations

import os
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
                # Save mid-story through the real menu.
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
    """Walk next to an item, or chase an NPC (they wander) until the interaction prompt names it."""
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
    # Keyboard movement works (the player leaves spawn).
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


STAMP: dict[str, float] = {}


def arm(page: Page) -> None:
    """Mark the page right after a scenario's own navigation; a later reload loses the mark."""
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
            browser = p.chromium.launch(args = chromium_launch_args())
            page = browser.new_page(viewport = {"width": 1280, "height": 800})
            errors: list[str] = []
            # Warm-up: the first visit makes Vite optimize Phaser and reload the page; play after that.
            page.goto(f"{BASE}?flow=world")
            page.wait_for_selector("[data-testid=world-start]", timeout = 180_000)
            page.wait_for_timeout(3_000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            for scenario in (play_story, play_world):
                retry_on_reload(page, scenario)
            browser.close()
            assert not errors, errors
    finally:
        stop_process(proc)
    print("flows player smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
