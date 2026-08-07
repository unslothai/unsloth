// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// Every localStorage key written by a panel width store.
const PANEL_WIDTH_KEYS = ["sidebar_width", "chat_settings_width"];

// The store reads window at import time, so stub it before importing.
const stubWindow = {
  innerWidth: 1440,
  localStorage: {
    getItem: () => null,
    setItem: () => {},
  },
  addEventListener: () => {},
  removeEventListener: () => {},
};
(globalThis as { window?: unknown }).window = stubWindow;

const {
  clampSidebarWidth,
  SIDEBAR_WIDTH_DEFAULT,
  SIDEBAR_WIDTH_MAX,
  SIDEBAR_WIDTH_MIN,
} = await import("../src/hooks/use-sidebar-width.ts");

test("clamps to the absolute range on a roomy window", () => {
  stubWindow.innerWidth = 1440;
  assert.equal(clampSidebarWidth(320), 320);
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX + 200), SIDEBAR_WIDTH_MAX);
  assert.equal(clampSidebarWidth(10), SIDEBAR_WIDTH_MIN);
  assert.equal(clampSidebarWidth(Number.NaN), SIDEBAR_WIDTH_DEFAULT);
});

test("caps at 40% of a narrow window", () => {
  stubWindow.innerWidth = 800;
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX), 320);
  assert.equal(clampSidebarWidth(300), 300);
});

test("the floor still wins when 40% falls below it", () => {
  stubWindow.innerWidth = 500;
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX), SIDEBAR_WIDTH_MIN);
});

test("re-evaluates the cap per call, so a resize can re-clamp", () => {
  stubWindow.innerWidth = 1440;
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX), SIDEBAR_WIDTH_MAX);
  stubWindow.innerWidth = 900;
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX), 360);
  stubWindow.innerWidth = 1440;
  assert.equal(clampSidebarWidth(SIDEBAR_WIDTH_MAX), SIDEBAR_WIDTH_MAX);
});

// The reset action promises to clear every stored preference, so a persisted
// panel width that is missing from the list survives the reload.
test("persisted panel widths are cleared by the preference reset", async () => {
  const source = await readFile(
    new URL("../src/features/settings/tabs/general-tab.tsx", import.meta.url),
    "utf8",
  );
  const keys = source.slice(
    source.indexOf("const PREFS_KEYS"),
    source.indexOf("];", source.indexOf("const PREFS_KEYS")),
  );
  for (const key of PANEL_WIDTH_KEYS) {
    assert.ok(keys.includes(`"${key}"`), `${key} missing from PREFS_KEYS`);
  }
});
