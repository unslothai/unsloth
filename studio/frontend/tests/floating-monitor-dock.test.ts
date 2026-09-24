// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  FLOATING_MONITOR_EDGE_INSET,
  FLOATING_MONITOR_HANDLE_HALF_WIDTH,
  dockedMonitorFits,
  floatingMonitorConstraintStyle,
  floatingMonitorHandleClearance,
  getFloatingMonitorLayout,
} from "../src/components/floating-monitor-layout.ts";
import { UI_FONT_SIZE_RANGE } from "../src/features/settings/stores/appearance-custom-store.ts";

const source = readFileSync(
  new URL("../src/components/floating-monitor.tsx", import.meta.url),
  "utf8",
);
const desktop = {
  isOpen: true,
  isMobile: false,
  isChatRoute: true,
  settingsPanelOpen: true,
  settingsWidth: 272,
  sidebarWidth: 280,
  viewportWidth: 1440,
};
const docked = { visible: true, suppressed: false, dockedBesideRunSettings: true };
const resting = { visible: true, suppressed: false, dockedBesideRunSettings: false };
const yielded = { visible: false, suppressed: true, dockedBesideRunSettings: false };

test("only a visible desktop monitor on the chat route docks beside Run settings", () => {
  assert.deepEqual(getFloatingMonitorLayout(desktop), docked);
  for (const [change, expected] of [
    [{ isOpen: false }, { ...resting, visible: false }],
    [{ isMobile: true }, yielded],
    [{ settingsPanelOpen: false }, resting],
    [{ isChatRoute: false }, resting],
    [{ viewportWidth: 768 }, yielded],
  ] as const) {
    assert.deepEqual(getFloatingMonitorLayout({ ...desktop, ...change }), expected);
  }
});

test("docking reserves the sidebar, scaled clearance, and rendered monitor width", () => {
  const room = {
    viewportWidth: 800,
    sidebarWidth: 280,
    settingsWidth: 290,
    monitorWidth: 205,
    uiSpaceScale: 0.8,
  };
  assert.equal(dockedMonitorFits(room), true);
  assert.equal(dockedMonitorFits({ ...room, monitorWidth: 220 }), false);
  assert.equal(dockedMonitorFits({ ...room, sidebarWidth: 300 }), false);
  assert.equal(dockedMonitorFits({ ...room, viewportWidth: 0 }), true);
  assert.deepEqual(getFloatingMonitorLayout({ ...desktop, ...room }), docked);
  assert.deepEqual(
    getFloatingMonitorLayout({ ...desktop, ...room, monitorWidth: 220 }),
    yielded,
  );
  // A hand-resized monitor must not cover the collapsed web icon rail.
  assert.equal(
    dockedMonitorFits({ viewportWidth: 804, sidebarWidth: 48, settingsWidth: 272, monitorWidth: 500 }),
    false,
  );
});

test("the dock inset follows every panel width and supported UI font scale", () => {
  const { min, default: base, max } = UI_FONT_SIZE_RANGE;
  for (const settingsWidth of [248, 272, 320, 420, 560]) {
    for (const uiSpaceScale of [min / base, 1, max / base]) {
      const clearance = FLOATING_MONITOR_HANDLE_HALF_WIDTH * uiSpaceScale;
      const style = floatingMonitorConstraintStyle({
        zIndex: 9100,
        dockedBesideRunSettings: true,
        settingsWidth,
        uiSpaceScale,
      });
      assert.deepEqual(style, { zIndex: 9100, right: settingsWidth + clearance });
      assert.equal(floatingMonitorHandleClearance(uiSpaceScale), clearance);
      // The monitor ends before the panel and its outward-facing resize handle.
      assert.ok(1440 - style.right < 1440 - settingsWidth);
    }
  }
  assert.ok(floatingMonitorHandleClearance(max / base) > FLOATING_MONITOR_HANDLE_HALF_WIDTH);
});

test("undocking restores a scaled corner inset, including for invalid scales", () => {
  for (const [scale, expected] of [
    [0.8, FLOATING_MONITOR_EDGE_INSET * 0.8],
    [undefined, FLOATING_MONITOR_EDGE_INSET],
    [0, FLOATING_MONITOR_EDGE_INSET],
    [Number.NaN, FLOATING_MONITOR_EDGE_INSET],
  ] as const) {
    assert.equal(
      floatingMonitorConstraintStyle({
        zIndex: 9100,
        dockedBesideRunSettings: false,
        settingsWidth: 560,
        uiSpaceScale: scale,
      }).right,
      expected,
    );
    if (!scale || Number.isNaN(scale)) {
      assert.equal(floatingMonitorHandleClearance(scale), FLOATING_MONITOR_HANDLE_HALF_WIDTH);
    }
  }
});
test("a monitor fits at the exact dock boundary, but not one pixel below", () => {
  const capacity = {
    sidebarWidth: 280,
    settingsWidth: 272,
    monitorWidth: 256,
    uiSpaceScale: 1,
  };
  // Both the left inset (16px) and the resize-handle clearance (4px)
  // count toward the required available width.
  const boundary = 280 + 272 + 256 + 16 + 4;
  assert.equal(dockedMonitorFits({ ...capacity, viewportWidth: boundary }), true);
  assert.equal(dockedMonitorFits({ ...capacity, viewportWidth: boundary - 1 }), false);
});

test("the monitor reads the panel's painted width and keeps its floating layer", () => {
  const container = source.slice(source.indexOf("ref={setConstraintsElement}"));
  assert.match(source, /const \{ width: committedSettingsWidth \} = useChatSettingsWidth\(\)/);
  assert.match(source, /paintedSettingsWidth > 0 \? paintedSettingsWidth : committedSettingsWidth/);
  assert.match(source, /new ResizeObserver\(measure\)[\s\S]*observer\.observe\(panel\)/);
  assert.match(source, /const uiSpaceScale = useUiSpaceScale\(\)/);
  assert.match(container, /floatingMonitorConstraintStyle\(\{[\s\S]*zIndex,[\s\S]*settingsWidth,[\s\S]*uiSpaceScale,/);
  assert.match(source, /useFloatingPanelZIndex\("resource-monitor"\)/);
  assert.doesNotMatch(container.slice(0, container.indexOf("<motion.div")), /\bz-\d+\b|\bright-\[\d+rem\]/);
});

test("responsive changes reattach geometry observers and recalculate capacity", () => {
  assert.match(source, /\}, \[isChatRoute, settingsPanelOpen, isMobile\]\);/);
  assert.match(source, /\}, \[isMobile\]\);/);
  assert.match(source, /paintedSidebarWidth > 0 \? paintedSidebarWidth : committedSidebarWidth/);
  assert.match(source, /const sidebarWidth = pinned \? pinnedSidebarWidth : unpinnedSidebarWidth/);
  assert.match(source, /const viewportWidth = useSyncExternalStore\(/);
  assert.match(source, /window\.addEventListener\("resize", onChange\)/);
  assert.match(source, /onRenderedWidth=\{setMonitorWidth\}/);
});

test("suppression retains layout but does not publish a phantom frame", () => {
  assert.match(source, /\(visible \|\| suppressed\) &&/);
  assert.match(source, /suppressed && "invisible"/);
  assert.match(source, /if \(hidden\) \{\s*useMonitorFrameStore\.getState\(\)\.clearFrame\(publisher\);/);
  assert.match(source, /if \(!hiddenRef\.current\) \{\s*useMonitorFrameStore\.getState\(\)\.setFrame/);
  assert.match(source, /\}, \[layout, constraintsElement, publisher, hidden\]\);/);
});

test("a net-zero docked drag must not discard the saved full-width X", () => {
  const update = source.slice(source.indexOf("function updateDrag"), source.indexOf("function finishDrag"));
  const finish = source.slice(source.indexOf("function finishDrag"));
  // Intermediate pointer moves are provisional. Only a released horizontal
  // displacement replaces the user's earlier full-width placement.
  assert.doesNotMatch(update, /chosenLeftRef\.current = null/);
  assert.match(finish, /if \(left !== baseLeft\) \{\s*hasDraggedLeftRef\.current = true;\s*chosenLeftRef\.current = narrowedRef\.current \? null : left;/);
  assert.match(source, /restoreLeftRef\.current = chosenLeftRef\.current/);
  assert.match(source, /if \(!narrowed\) \{\s*restoreLeftRef\.current = chosenLeftRef\.current;\s*reconcileRef\.current\?\.\(\);/);
  assert.match(source, /place\(hasDraggedTopRef\.current, currentTop, maxTop\)/);
});

test("the panel width clamps to the same bounds used by docking", async () => {
  const { clampChatSettingsWidth } = await import("../src/hooks/use-chat-settings-width.ts");
  for (const width of [248, 272, 320, 420, 560]) {
    const settingsWidth = clampChatSettingsWidth(width);
    assert.equal(settingsWidth, width);
    assert.equal(
      floatingMonitorConstraintStyle({
        zIndex: 9100,
        dockedBesideRunSettings: true,
        settingsWidth,
      }).right,
      width + FLOATING_MONITOR_HANDLE_HALF_WIDTH,
    );
  }
  assert.equal(clampChatSettingsWidth(200), 248);
  assert.equal(clampChatSettingsWidth(600), 560);
});

test("spacing and the resize handle use the same live UI scale", () => {
  const css = readFileSync(new URL("../src/index.css", import.meta.url), "utf8");
  const handle = readFileSync(
    new URL("../src/components/ui/panel-resize-handle.tsx", import.meta.url),
    "utf8",
  );
  assert.match(css, /--spacing:\s*calc\(0\.25rem \* var\(--ui-space-scale, 1\)\)/);
  assert.match(css, /--ui-space-scale:\s*calc\(var\(--ui-font-scale, 1\) \/ 0\.9375\)/);
  assert.match(handle, /edge === "left" \? "-left-1" : "-right-1"/);
  assert.match(source, /const uiSpaceScale = useUiSpaceScale\(\)/);
});

test("docking never remounts the monitor or loses its drag geometry", () => {
  assert.doesNotMatch(source, /key=\{`\$\{panelKey\}-/);
  assert.match(source, /useLayoutEffect\(\(\) => \{\s*if \(narrowedRef\.current === narrowed\)/);
  assert.match(source, /const restoreTo = restoreLeftRef\.current;/);
  assert.match(source, /place\(hasDraggedLeftRef\.current, currentLeft, maxLeft\)/);
  assert.match(source, /const unpinnedSidebarWidth = sidebarHoldsRail \? paintedSidebarWidth : 0/);
});
