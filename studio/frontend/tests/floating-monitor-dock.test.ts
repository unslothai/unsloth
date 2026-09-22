// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The Live resource monitor and the Run settings panel share the bottom-right
 * corner, so they collide by construction: the monitor is anchored to that
 * corner and the panel opens in front of it at 248-560 px wide.
 *
 * Two things are pinned here, because both were measurably wrong in the
 * original attempt at this fix:
 *
 * 1. The docking geometry. The monitor has to clear the panel at every width
 *    the panel can be dragged to, not only at the ~272 px default. A fixed
 *    18rem (288 px) inset is short of the panel at its default and 272 px short
 *    at the 560 px maximum, where the monitor ends up entirely behind it. The
 *    inset therefore comes from the panel's own clamped width, which is driven
 *    here through the real store.
 *
 * 2. The source contracts that decide whether any of that is on screen: the
 *    container keeps the shared floating-panel layer (a Tailwind `z-40` puts
 *    the bottom-right notification stack at 9000 back over the monitor), and
 *    the dock state must not change the panel key, which would remount the
 *    panel and discard the position and size the user set.
 */

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

import {
  FLOATING_MONITOR_EDGE_INSET,
  FLOATING_MONITOR_WIDTH,
  floatingMonitorConstraintStyle,
  getFloatingMonitorLayout,
} from "../src/components/floating-monitor-layout.ts";

const MONITOR = new URL(
  "../src/components/floating-monitor.tsx",
  import.meta.url,
);
const source = readFileSync(MONITOR, "utf8");

/** The JSX attributes of the monitor's fixed constraint container. */
function containerAttributes(): string {
  const start = source.indexOf("ref={setConstraintsElement}");
  assert.notEqual(start, -1, "the monitor's constraint container is gone");
  const end = source.indexOf("<motion.div", start);
  assert.notEqual(end, -1, "the constraint container has no child panel");
  return source.slice(start, end);
}

// Every width the panel can be dragged to, from use-chat-settings-width.
const SETTINGS_WIDTHS = [248, 272, 320, 420, 560];
const VIEWPORT_WIDTH = 1440;
const Z_INDEX = 9100;

test("closed monitor stays hidden", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: false,
      isMobile: false,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: false, dockedBesideRunSettings: false },
  );
});

test("desktop monitor docks beside open run settings", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: false,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: true, dockedBesideRunSettings: true },
  );
});

test("mobile monitor yields to the run-settings sheet", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: true,
      isChatRoute: true,
      settingsPanelOpen: true,
    }),
    { visible: false, dockedBesideRunSettings: false },
  );
});

test("monitor remains visible when run settings are closed", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: true,
      isChatRoute: true,
      settingsPanelOpen: false,
    }),
    { visible: true, dockedBesideRunSettings: false },
  );
});

test("stale chat settings state does not dock the monitor off-route", () => {
  assert.deepEqual(
    getFloatingMonitorLayout({
      isOpen: true,
      isMobile: false,
      isChatRoute: false,
      settingsPanelOpen: true,
    }),
    { visible: true, dockedBesideRunSettings: false },
  );
});

test("an undocked monitor keeps the resting inset", () => {
  const style = floatingMonitorConstraintStyle({
    zIndex: Z_INDEX,
    dockedBesideRunSettings: false,
    settingsWidth: 560,
  });
  assert.deepEqual(style, { zIndex: Z_INDEX, right: FLOATING_MONITOR_EDGE_INSET });
});

test("the docked constraint clears the panel at every draggable width", () => {
  for (const settingsWidth of SETTINGS_WIDTHS) {
    const { right } = floatingMonitorConstraintStyle({
      zIndex: Z_INDEX,
      dockedBesideRunSettings: true,
      settingsWidth,
    });
    // The container ends exactly where the panel begins: the same inset from
    // the viewport's right edge.
    assert.equal(
      right,
      settingsWidth,
      `the docked inset must follow the ${settingsWidth}px panel`,
    );
    const monitorRight = VIEWPORT_WIDTH - right;
    const panelLeft = VIEWPORT_WIDTH - settingsWidth;
    assert.ok(
      monitorRight <= panelLeft,
      `monitor right edge ${monitorRight} overlaps the panel left edge ` +
        `${panelLeft} at a ${settingsWidth}px panel`,
    );
    // And there still has to be room for the monitor itself, not just a sliver.
    assert.ok(
      monitorRight - FLOATING_MONITOR_WIDTH >= FLOATING_MONITOR_EDGE_INSET,
      `a ${settingsWidth}px panel leaves no ${FLOATING_MONITOR_WIDTH}px ` +
        `monitor on screen`,
    );
  }
});

test("a stored 248-560px panel width reaches the monitor's constraint", async () => {
  // The panel paints from useChatSettingsWidth and the monitor reads the same
  // store, so the two meet on the same clamped number. Drive the real store at
  // a wide viewport and prove the number that reaches `right` is the panel's
  // painted width.
  const stored = new Map<string, string>();
  (globalThis as unknown as { window: unknown }).window = {
    innerWidth: VIEWPORT_WIDTH,
    addEventListener: () => {},
    removeEventListener: () => {},
    localStorage: {
      getItem: (key: string) => stored.get(key) ?? null,
      setItem: (key: string, value: string) => void stored.set(key, value),
    },
  };
  try {
    const { clampChatSettingsWidth } = await import(
      "../src/hooks/use-chat-settings-width.ts"
    );
    for (const width of SETTINGS_WIDTHS) {
      const settingsWidth = clampChatSettingsWidth(width);
      assert.equal(
        settingsWidth,
        width,
        `a ${width}px preference is painted as ${width}px on a ` +
          `${VIEWPORT_WIDTH}px viewport`,
      );
      const { right } = floatingMonitorConstraintStyle({
        zIndex: Z_INDEX,
        dockedBesideRunSettings: true,
        settingsWidth,
      });
      assert.equal(
        right,
        settingsWidth,
        `the ${settingsWidth}px panel must set a ${settingsWidth}px inset`,
      );
    }
  } finally {
    delete (globalThis as unknown as { window?: unknown }).window;
  }
});

test("the container keeps the shared floating-panel layer", () => {
  const attributes = containerAttributes();
  // The notification stack paints at Z_LAYER.OVERLAY_STACK (9000) and the
  // floating panels at FLOATING_PANEL (9100); a Tailwind z-* class is 50 or
  // below, which puts the stack back over the monitor.
  assert.match(
    source,
    /useFloatingPanelZIndex\("resource-monitor"\)/,
    "the monitor must read its z-index from lib/floating-panel-order",
  );
  assert.match(
    attributes,
    /floatingMonitorConstraintStyle\(\{[\s\S]*zIndex,/,
    "the container must apply that z-index",
  );
  assert.doesNotMatch(
    attributes,
    /\bz-\d+\b|\bz-\[/,
    "the constraint container must not hardcode a z class",
  );
});

test("the docked offset comes from the panel's own width", () => {
  const attributes = containerAttributes();
  assert.match(
    attributes,
    /floatingMonitorConstraintStyle\(\{[\s\S]*settingsWidth,/,
    "the container must be fed the panel width, not a constant",
  );
  assert.doesNotMatch(
    attributes,
    /\bright-\[\d+(\.\d+)?rem\]/,
    "the docked offset must not be a fixed rem distance",
  );
  // The panel resolves its width in one place; the monitor has to read it.
  assert.match(
    source,
    /import \{ useChatSettingsWidth \} from "@\/hooks\/use-chat-settings-width"/,
    "the monitor must read the shared panel-width hook",
  );
  assert.match(
    source,
    /const \{ width: settingsWidth \} = useChatSettingsWidth\(\)/,
    "the monitor must bind the hook's live width",
  );
  assert.match(
    source,
    /settingsWidth=\{settingsWidth\}/,
    "that width must reach the panel that paints the container",
  );
});

test("docking does not remount the panel", () => {
  const keyed = source.slice(
    source.indexOf("export function FloatingMonitor()"),
  );
  assert.doesNotMatch(
    keyed,
    /key=\{`\$\{panelKey\}-/,
    "the panel key must not change with the dock state",
  );
});
