// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { installLocalStorageFake, readSrc, registerBundlerResolver } from "./helpers/kit.ts";

registerBundlerResolver();
const { store } = installLocalStorageFake();
const {
  PLUS_MENU_ORDER,
  PLUS_MENU_PINS_STORAGE_KEY,
  normalizePlusMenuOrder,
  splitPlusMenuItems,
  usePlusMenuPrefsStore,
} = await import("../src/features/chat/stores/plus-menu-prefs-store.ts");

test("deep research and browser use are menu items, shown by default", () => {
  assert.ok(PLUS_MENU_ORDER.includes("deepResearch"));
  assert.ok(PLUS_MENU_ORDER.includes("browserUse"));
  const { pins } = usePlusMenuPrefsStore.getInitialState();
  assert.equal(pins.deepResearch, true);
  assert.equal(pins.browserUse, true);
});

test("a saved order drops unknown ids and slots new ones after their default neighbour", () => {
  // Saved before this release: no deep research or browser use, one retired id.
  const saved = ["skills", "chatWithFiles", "retired", "mcp", "skills"];
  const order = normalizePlusMenuOrder(saved);
  assert.equal(order.length, PLUS_MENU_ORDER.length);
  assert.equal(new Set(order).size, order.length);
  assert.ok(!order.includes("retired" as never));
  // The user's own items keep their relative order.
  const mine = order.filter((id) => ["skills", "chatWithFiles", "mcp"].includes(id));
  assert.deepEqual(mine, ["skills", "chatWithFiles", "mcp"]);
  // Browser use follows chat with files, as it does by default.
  assert.equal(order[order.indexOf("chatWithFiles") + 1], "browserUse");
  assert.deepEqual(normalizePlusMenuOrder(undefined), PLUS_MENU_ORDER);
});

test("the menu shows pinned items in the saved order and the rest under More in that order", () => {
  const pins = Object.fromEntries(PLUS_MENU_ORDER.map((id) => [id, false])) as Record<
    (typeof PLUS_MENU_ORDER)[number],
    boolean
  >;
  pins.skills = true;
  pins.deepResearch = true;
  const order = normalizePlusMenuOrder(["skills", "mcp", "deepResearch"]);
  const { pinned, overflow } = splitPlusMenuItems(order, pins);
  assert.deepEqual(pinned, ["skills", "deepResearch"]);
  // More keeps the saved order too: mcp was placed before chat with files.
  assert.ok(overflow.indexOf("mcp") < overflow.indexOf("chatWithFiles"));
  assert.equal(pinned.length + overflow.length, PLUS_MENU_ORDER.length);
});

test("an install that saved pins before ordering existed gets the default order and new pins", async () => {
  store.set(
    PLUS_MENU_PINS_STORAGE_KEY,
    JSON.stringify({ state: { pins: { chatWithFiles: false, skills: true } }, version: 1 }),
  );
  await usePlusMenuPrefsStore.persist.rehydrate();
  const state = usePlusMenuPrefsStore.getState();
  assert.deepEqual(state.order, PLUS_MENU_ORDER);
  assert.equal(state.pins.chatWithFiles, false);
  assert.equal(state.pins.browserUse, true);
  state.setOrder(["projects", ...PLUS_MENU_ORDER]);
  assert.equal(usePlusMenuPrefsStore.getState().order[0], "projects");
  usePlusMenuPrefsStore.getState().resetMenu();
  assert.deepEqual(usePlusMenuPrefsStore.getState().order, PLUS_MENU_ORDER);
});

test("both + menus render from the saved order, with no hard-coded browser use slot", () => {
  for (const path of ["components/assistant-ui/thread.tsx", "features/chat/shared-composer.tsx"]) {
    const src = readSrc(path);
    assert.match(src, /splitPlusMenuItems\(plusOrder, plusPins\)/, path);
    assert.match(src, /browserUse: <BrowserAgentMenuItem \/>/, path);
    assert.ok(!src.includes('pinnedPlusItems.includes("chatWithFiles")'), path);
  }
});

test("dragging a chat menu row stays inside the list and saves the order once, on drop", () => {
  const src = readSrc("features/settings/components/chat-menu-order-list.tsx");
  // A hard stop at the list: the row no longer slides over the fixed rows or the heading.
  assert.match(src, /dragConstraints=\{bounds\}\s*dragElastic=\{0\}/);
  // Swaps go to a local draft; the persisted store (read by both + menus) only takes the drop.
  assert.match(src, /onReorder=\{\(ids: PlusMenuItemId\[\]\) => setDraft\(ids\)\}/);
  assert.match(src, /onDragEnd=\{onDrop\}/);
  assert.match(src, /if \(draftRef\.current\) \{\s*setOrder\(draftRef\.current\);/);
  // layoutScroll on the settings pane would double-count its scroll against the constraints.
  assert.ok(!readSrc("features/settings/settings-dialog.tsx").includes("layoutScroll"));
});
