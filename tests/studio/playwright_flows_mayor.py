# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Mayor mode end to end against the mock backend (smoke-flows-player.html?flow=world&city=1).

Zone a block (R, C, I) by clicking the 2D map, lay a road, build a power plant and connect it with a power line, run
months until houses grow, and see the advisor's monthly note and a resident petition. Then switch to 3D and back:
the city stays. Screenshots go to $FLOWS_SHOTS (default ./outputs/flows)."""

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

PORT = int(os.environ.get("SMOKE_PORT", "5432"))
BASE = f"http://127.0.0.1:{PORT}/smoke-flows-player.html"
SHOTS = Path(os.environ.get("FLOWS_SHOTS", "outputs/flows"))
VIEW = {"width": 1400, "height": 1300}
# Relative layout: R, C and I zones (3x3) on top, a road under them, the plant below the I zone, wired up
# through a road crossing.
W, H = 9, 8
ZONES = [("R", 1, 1), ("C", 4, 1), ("I", 7, 1)]
ROAD = [(x, 3) for x in range(0, 9)]
PLANT = ("P", 7, 6)
WIRE = [(7, 4), (7, 3)]

FIND = """([w, h, pad]) => {
  const s = window.__flowsPlayerSmoke;
  const m = s.map;
  const solid = new Set([2, 6, 7, 8]);
  const me = s.world.playerTile();
  const canvas = document.querySelector('[data-testid=world-canvas] canvas');
  const box = canvas.getBoundingClientRect();
  const T = 32;
  const zoom = Math.min(3, Math.max(1, Math.round(box.width / (T * 18))));
  const vw = box.width / zoom, vh = box.height / zoom;
  const left = Math.min(Math.max(0, me.x * T + T / 2 - vw / 2), m.width * T - vw);
  const top = Math.min(Math.max(0, me.y * T + T / 2 - vh / 2), m.height * T - vh);
  const screen = (x, y) => [box.x + ((x + 0.5) * T - left) * zoom, box.y + ((y + 0.5) * T - top) * zoom];
  let best = null;
  for (let oy = 0; oy + h <= m.height; oy++)
    for (let ox = 0; ox + w <= m.width; ox++) {
      const [sx0, sy0] = screen(ox, oy);
      const [sx1, sy1] = screen(ox + w - 1, oy + h - 1);
      if (sx0 < pad.left || sy0 < pad.top || sx1 > box.x + box.width - 40 || sy1 > box.y + box.height - pad.bottom)
        continue;
      let ok = true;
      for (let y = oy; y < oy + h && ok; y++)
        for (let x = ox; x < ox + w && ok; x++) {
          if (solid.has(m.tiles[y * m.width + x])) ok = false;
          if (x === me.x && y === me.y) ok = false;
        }
      if (!ok) continue;
      const d = Math.abs(ox + w / 2 - me.x) + Math.abs(oy + h / 2 - me.y);
      if (!best || d < best.d) best = { ox, oy, d };
    }
  if (!best) return { miss: true, box: [box.x, box.y, box.width, box.height], zoom, left, top, me, w: m.width, h: m.height };
  const pts = {};
  for (let y = best.oy; y < best.oy + h; y++)
    for (let x = best.ox; x < best.ox + w; x++) pts[`${x - best.ox},${y - best.oy}`] = screen(x, y);
  return { ox: best.ox, oy: best.oy, zoom, pts };
}"""


def shot(page: Page, name: str) -> None:
    SHOTS.mkdir(parents = True, exist_ok = True)
    page.screenshot(path = str(SHOTS / f"mayor_{name}.png"))


def panel(page: Page, key: str) -> int:
    return int(page.get_attribute("[data-testid=mayor-panel]", f"data-{key}") or "0")


def click_tile(page: Page, spot: dict, x: int, y: int) -> None:
    px, py = spot["pts"][f"{x},{y}"]
    page.mouse.click(px, py)
    page.wait_for_timeout(60)


def play_mayor(page: Page) -> None:
    page.goto(f"{BASE}?flow=world&city=1")
    page.wait_for_selector("[data-testid=world-start]", timeout = 120_000)
    page.click("[data-testid=world-start]")
    page.wait_for_function("window.__flowsPlayerSmoke.world !== null")
    page.wait_for_selector("[data-testid=mayor-panel]")
    page.wait_for_timeout(1500)
    shot(page, "start")
    spot = page.evaluate(FIND, [W, H, {"left": 340, "top": 90, "bottom": 120}])
    assert spot and not spot.get("miss"), f"no clear 9x8 block on screen: {spot}"
    funds = panel(page, "funds")
    for kind, x, y in ZONES:
        page.click(f"[data-testid=mayor-tool-{kind}]")
        click_tile(page, spot, x, y)
    page.click("[data-testid=mayor-tool-road]")
    for x, y in ROAD:
        click_tile(page, spot, x, y)
    page.click(f"[data-testid=mayor-tool-{PLANT[0]}]")
    click_tile(page, spot, PLANT[1], PLANT[2])
    page.click("[data-testid=mayor-tool-wire]")
    for x, y in WIRE:
        click_tile(page, spot, x, y)
    page.click("[data-testid=mayor-tool-walk]")
    assert panel(page, "zones") == 4, page.get_attribute("[data-testid=mayor-panel]", "data-zones")
    spent = 3 * 100 + 9 * 10 + 3000 + 2 * 5
    assert panel(page, "funds") == funds - spent, (funds, panel(page, "funds"))
    page.wait_for_timeout(300)
    shot(page, "zoned")
    page.click("[data-testid=mayor-overlay-power]")
    page.wait_for_timeout(300)
    shot(page, "power_overlay")
    page.click("[data-testid=mayor-overlay-power]")
    months = 0
    for _ in range(8):
        page.click("[data-testid=mayor-next-month]")
        months += 1
        page.wait_for_timeout(700)
        if panel(page, "res") > 0:
            break
    assert panel(page, "res") > 0, "no houses grew"
    page.get_by_test_id("mayor-advice").get_by_text("report", exact = False).wait_for(timeout = 70_000)
    page.wait_for_timeout(500)
    shot(page, "grown_advisor")
    calls = page.evaluate(
        "window.__flowsPlayerSmoke.calls.filter(c => c.inputs.state.task === 'mayor').length"
    )
    month = panel(page, "month")
    assert 1 <= calls <= month, (calls, month)
    print(f"mayor: {month} months, {calls} model calls, residents {panel(page, 'res')}")
    page.click("[data-testid=world-view-toggle]")
    page.wait_for_timeout(2500)
    shot(page, "in_3d")
    page.click("[data-testid=world-view-toggle]")
    page.wait_for_timeout(2500)
    assert panel(page, "zones") == 4, "the city survives a 2D / 3D switch"
    shot(page, "back_to_2d")


def main() -> int:
    proc = start_vite(PORT)
    try:
        wait_for_smoke_page(BASE, "smoke-flows-player-main.tsx", proc = proc)
        with sync_playwright() as p:
            browser = p.chromium.launch(
                args = chromium_launch_args() + ["--enable-unsafe-swiftshader"]
            )
            page = browser.new_page(viewport = VIEW)
            errors: list[str] = []
            page.goto(f"{BASE}?flow=world&city=1")
            page.wait_for_selector("[data-testid=world-start]", timeout = 180_000)
            page.wait_for_timeout(3_000)
            page.on("pageerror", lambda e: errors.append(str(e)))
            play_mayor(page)
            browser.close()
            assert not errors, errors
    finally:
        stop_process(proc)
    print("flows mayor smoke: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
