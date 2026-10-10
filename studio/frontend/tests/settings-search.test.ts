// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SETTINGS_SEARCH_KEYWORDS,
  createSettingsSearchIndex,
  renderedSearchEntries,
} from "../src/features/settings/settings-search.ts";
import { en } from "../src/i18n/locales/en.ts";

const UPDATE_ENTRY = "settings.about.updates";
const INTERFACE_SCALE_ENTRY = "settings.appearance.custom.interfaceScale.label";

test("desktop update searches route to General", () => {
  const index = createSettingsSearchIndex({ desktop: true, closeToTray: true });

  assert.ok(index.general.includes(UPDATE_ENTRY));
  assert.ok(!index.about.includes(UPDATE_ENTRY));
});

test("browser update searches keep routing to About", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(!index.general.includes(UPDATE_ENTRY));
  assert.ok(index.about.includes(UPDATE_ENTRY));
});

test("interface scale is searchable on every build", () => {
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(desktop.appearance.includes(INTERFACE_SCALE_ENTRY));
  assert.ok(browser.appearance.includes(INTERFACE_SCALE_ENTRY));
  assert.equal(
    desktop.appearance.filter((key) => key === INTERFACE_SCALE_ENTRY).length,
    1,
  );
});

// The feature's search terms are not substrings of its labels.
test("model memory rows are reachable by the terms the feature is about", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });
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

const DESKTOP_STARTUP_ENTRIES = [
  "settings.general.startup.sectionTitle",
  "settings.general.startup.launchAtLogin",
] as const;
const CLOSE_TO_TRAY_ENTRY = "settings.general.startup.closeToTray";
const CURRENT_DATE_ENTRY = "settings.chat.currentDate.label";

test("the current date prompt setting is searchable under Chat", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(index.chat.includes(CURRENT_DATE_ENTRY));
});

test("desktop startup entries are absent from browser search", () => {
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  for (const entry of DESKTOP_STARTUP_ENTRIES) {
    assert.ok(desktop.general.includes(entry));
    assert.ok(!browser.general.includes(entry));
  }
});

test("the repair row is searchable on the desktop, where it exists", () => {
  // Desktop only: DesktopRepairControl renders nothing in a browser.
  const desktop = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(desktop.general.includes("settings.general.repairInstall.label"));
  assert.ok(!browser.general.includes("settings.general.repairInstall.label"));
});

test("close to tray is searchable only on supported desktops", () => {
  const supported = createSettingsSearchIndex({ desktop: true, closeToTray: true });
  const mac = createSettingsSearchIndex({ desktop: true, closeToTray: false });
  const browser = createSettingsSearchIndex({ desktop: false, closeToTray: false });

  assert.ok(supported.general.includes(CLOSE_TO_TRAY_ENTRY));
  assert.ok(!mac.general.includes(CLOSE_TO_TRAY_ENTRY));
  assert.ok(!browser.general.includes(CLOSE_TO_TRAY_ENTRY));
});

test("the endpoint rows are searchable only while Hugging Face serves, as they render", () => {
  const index = createSettingsSearchIndex({ desktop: false, closeToTray: false });
  const endpoint = ["settings.general.hub.endpoint", "settings.general.hub.datasetsServer"] as const;
  const modelScope = renderedSearchEntries(index, "general", "modelscope");
  const huggingFace = renderedSearchEntries(index, "general", "huggingface");
  assert.deepEqual(endpoint.map((key) => [huggingFace.includes(key), modelScope.includes(key)]), [[true, false], [true, false]]);
  assert.ok(modelScope.includes("settings.general.hub.source"));
});

const MCP_ENTRY = "settings.apiKeys.mcp.title";

test("agent access (MCP) is found by the protocol and the agents' names, by the owner only", () => {
  const index = createSettingsSearchIndex({
    desktop: false,
    closeToTray: false,
  });
  assert.ok(index["api-keys"].includes(MCP_ENTRY));
  const keywordsKey = SETTINGS_SEARCH_KEYWORDS[MCP_ENTRY];
  assert.equal(keywordsKey, "settings.apiKeys.mcp.keywords");
  const haystack =
    `${en.settings.apiKeys.mcp.title} ${en.settings.apiKeys.mcp.keywords}`.toLowerCase();
  for (const term of ["mcp", "claude", "codex", "model context protocol"]) {
    assert.ok(haystack.includes(term), `search matches "${term}"`);
  }
  assert.ok(
    renderedSearchEntries(index, "api-keys", "huggingface", true).includes(
      MCP_ENTRY,
    ),
  );
  assert.ok(
    !renderedSearchEntries(index, "api-keys", "huggingface", false).includes(
      MCP_ENTRY,
    ),
  );
});
