// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { DEFAULT_CUSTOMIZATION } from "../src/features/settings/stores/appearance-custom-store.ts";

import { readText } from "./helpers/kit.ts";

// Records saved before an item existed get the backend's defaults, so they must match; read the real constant.
test("the backend sidebar menu defaults match the frontend", () => {
  const source = readText("../../backend/routes/settings.py");
  const block = /SIDEBAR_MENU_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(source);
  assert.ok(block, "could not find SIDEBAR_MENU_ITEM_DEFAULTS in settings.py");
  // Menu ids are camelCase (darkMode, guidedTour), unlike the lowercase nav ids.
  const backend = [...block[1].matchAll(/"([A-Za-z]+)":\s*(True|False)/g)].map((m) => ({
    id: m[1],
    visible: m[2] === "True",
  }));
  // Order matters: the backend appends missing ids in this order.
  assert.deepEqual(backend, DEFAULT_CUSTOMIZATION.sidebarMenu);
});
