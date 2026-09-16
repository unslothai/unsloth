// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import {
  OWNER_ONLY_SETTINGS_TABS,
  resolveSettingsTab,
  settingsTabVisible,
} from "../src/features/settings/settings-tab-visibility.ts";
import { SETTINGS_TABS } from "../src/features/settings/stores/settings-dialog-store.ts";

test("the owner sees every settings tab", () => {
  for (const tab of SETTINGS_TABS) {
    assert.equal(settingsTabVisible(tab, true), true, tab);
    assert.equal(resolveSettingsTab(tab, true), tab);
  }
});

test("a managed account never sees a tab whose routes are all owner-only", () => {
  const hidden = SETTINGS_TABS.filter((tab) => !settingsTabVisible(tab, false));
  assert.deepEqual(new Set(hidden), OWNER_ONLY_SETTINGS_TABS);
  for (const tab of ["accounts", "resources", "remote-lan", "agents", "debugging"] as const) {
    assert.ok(OWNER_ONLY_SETTINGS_TABS.has(tab), tab);
    assert.equal(resolveSettingsTab(tab, false), "general");
  }
  for (const tab of ["general", "api-keys", "data", "chat", "connections"] as const) {
    assert.equal(resolveSettingsTab(tab, false), tab);
  }
});
