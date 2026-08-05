// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { DEFAULT_CUSTOMIZATION } from "../src/features/settings/stores/appearance-custom-store.ts";

// A record predating sidebarNav is served the backend's own defaults, so a drift there
// hands the user a layout this side never shipped. settings.py says the two must match;
// the backend's parity test compares against a hand-copied list, which cannot catch a
// frontend-only change. Read the real constant instead.
test("the backend sidebar nav defaults match the frontend", async () => {
  const source = await readFile(
    new URL("../../backend/routes/settings.py", import.meta.url),
    "utf8",
  );
  const block = /SIDEBAR_NAV_ITEM_DEFAULTS = \{([\s\S]*?)^\}/m.exec(source);
  assert.ok(block, "could not find SIDEBAR_NAV_ITEM_DEFAULTS in settings.py");
  const backend = [...block[1].matchAll(/"([a-z]+)":\s*(True|False)/g)].map((m) => ({
    id: m[1],
    pinned: m[2] === "True",
  }));
  // Order matters too: the backend appends its missing ids in this order.
  assert.deepEqual(backend, DEFAULT_CUSTOMIZATION.sidebarNav);
});
