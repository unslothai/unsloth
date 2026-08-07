// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { createSettingsSearchIndex } from "../src/features/settings/settings-search.ts";

const UPDATE_ENTRY = "settings.about.updates";

test("desktop update searches route to General", () => {
  const index = createSettingsSearchIndex(true);

  assert.ok(index.general.includes(UPDATE_ENTRY));
  assert.ok(!index.about.includes(UPDATE_ENTRY));
});

test("browser update searches keep routing to About", () => {
  const index = createSettingsSearchIndex(false);

  assert.ok(!index.general.includes(UPDATE_ENTRY));
  assert.ok(index.about.includes(UPDATE_ENTRY));
});

const STARTUP_ENTRIES = [
  "settings.general.startup.sectionTitle",
  "settings.general.startup.launchAtLogin",
] as const;

test("startup entries are searchable on desktop only", () => {
  const desktop = createSettingsSearchIndex(true);
  const browser = createSettingsSearchIndex(false);

  for (const entry of STARTUP_ENTRIES) {
    assert.ok(desktop.general.includes(entry));
    assert.ok(!browser.general.includes(entry));
  }
});
