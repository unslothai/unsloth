// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SETTINGS_SEARCH_KEYWORDS,
  createSettingsSearchIndex,
} from "../src/features/settings/settings-search.ts";
import { en } from "../src/i18n/locales/en.ts";

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

// The words a user types for this feature are not substrings of any of its
// labels, so without keywords the rows it is named after were unfindable.
test("model memory rows are reachable by the terms the feature is about", () => {
  const index = createSettingsSearchIndex(false);
  const rows = [
    "settings.resources.modelMemory.title",
    "settings.resources.modelMemory.keepResident",
    "settings.resources.modelMemory.noRamReserve",
  ] as const;

  for (const row of rows) {
    assert.ok(index.resources.includes(row), `${row} is indexed under Resources`);
    assert.equal(
      SETTINGS_SEARCH_KEYWORDS[row],
      "settings.resources.modelMemory.modelMemoryKeywords",
      `${row} has synonyms`,
    );
  }

  for (const term of ["mlock", "vram", "ulimit", "memlock", "pin"]) {
    assert.ok(
      en.settings.resources.modelMemory.modelMemoryKeywords.includes(term),
      `search matches "${term}"`,
    );
  }
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
