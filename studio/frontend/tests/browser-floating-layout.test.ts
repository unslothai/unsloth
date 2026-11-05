// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import { test } from "node:test";

register("./helpers/browser-store-resolver.mjs", import.meta.url);
const { useBrowserStore } = await import("../src/features/browser/store.ts");

test("floating, full view and the split are one layout at a time, and floating outlasts a close", () => {
  const store = () => useBrowserStore.getState();
  store().openUrl("https://example.com");
  store().setFullView(true);
  store().setFloating(true);
  assert.deepEqual([store().floating, store().fullView], [true, false]);
  store().setFullView(true);
  assert.deepEqual([store().floating, store().fullView], [false, true]);
  store().setFloating(true);
  store().splitWithChatOn("right");
  assert.deepEqual([store().floating, store().fullView, store().chatSide], [false, false, "right"]);
  store().setFloating(true);
  store().closePanel();
  store().openPanel();
  assert.deepEqual([store().open, store().floating], [true, true]);
});
