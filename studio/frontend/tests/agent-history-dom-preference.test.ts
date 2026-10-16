// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";
import { agentHistoryLevelFor, agentHistoryStorageModeForUi, agentHistoryUiModeFor, agentHistoryWindowFor, normalizeAgentHistoryDomMode } from "../src/features/chat/stores/agent-history-preference-policy.ts";
import type { AgentHistoryPreferenceSnapshot } from "../src/features/chat/stores/agent-history-preference-policy.ts";

const legacy: AgentHistoryPreferenceSnapshot = {
  globalEnabled: false, globalWindowRows: 5,
  enabledThreads: { olderChat: true }, disabledThreads: {},
  windowRowsByThreadId: {},
};
test("v1 legacy settings remain enabled without migration", () => {
  assert.equal(agentHistoryLevelFor(legacy, "olderChat"), "legacy");
  assert.equal(agentHistoryLevelFor(legacy, "new"), "off");
});
test("explicit chat levels override global levels in both directions", () => {
  const prefs = { ...legacy, globalDomLevel: "low" as const,
    domLevelByThreadId: { custom: "high" as const, disabled: "off" as const } };
  assert.equal(agentHistoryLevelFor(prefs, "custom"), "high");
  assert.equal(agentHistoryLevelFor(prefs, "disabled"), "off");
  assert.equal(agentHistoryLevelFor(prefs, "new"), "low");
  assert.equal(agentHistoryLevelFor({ ...prefs, globalDomLevel: "off" }, "custom"), "high");
});

test("existing explicit v1 exceptions retain precedence when the global mode changes", () => {
  const prefs = { ...legacy, globalDomLevel: "off" as const,
    disabledThreads: { blocked: true as const } };
  assert.equal(agentHistoryLevelFor(prefs, "olderChat"), "legacy");
  assert.equal(agentHistoryLevelFor(prefs, "blocked"), "off");
  assert.equal(agentHistoryLevelFor(prefs, "new"), "off");
  const enabled = { ...prefs, globalDomLevel: "high" as const };
  assert.equal(agentHistoryLevelFor(enabled, "olderChat"), "high");
  assert.equal(agentHistoryLevelFor(enabled, "blocked"), "off");
  assert.equal(agentHistoryLevelFor({ ...enabled,
    domLevelByThreadId: { blocked: "low" as const } }, "blocked"), "low");
});


test("invalid or absent explicit modes retain the v1 behavior", () => {
  const state = { ...legacy, globalEnabled: true,
    domLevelByThreadId: { bogus: "unsupported" as "off" } };
  // Persisted preferences are filtered by the zustand-store migration;
  // the pure policy must also treat unexpected values as absent.
  assert.equal(agentHistoryLevelFor(state, "bogus"), "legacy");
  assert.equal(agentHistoryLevelFor({ ...state, globalDomLevel: undefined }, "olderChat"), "legacy");
});

test("an explicit per-chat Off is never overridden by global High", () => {
  const prefs = { ...legacy, globalEnabled: true, globalDomLevel: "high" as const,
    domLevelByThreadId: { parked: "off" as const } };
  assert.equal(agentHistoryLevelFor(prefs, "parked"), "off");
  assert.equal(agentHistoryLevelFor(prefs, "another"), "high");
});

test("four explicit modes respect message caps and per-chat overrides", () => {
  for (const [mode, limit] of [
    ["dynamic-low", 5], ["dynamic-high", 32],
    ["fixed-low", 5], ["fixed-high", 32],
  ] as const) {
    const prefs = { ...legacy, globalEnabled: true, globalDomLevel: mode };
    assert.equal(agentHistoryLevelFor(prefs, "new"), mode);
    assert.equal(agentHistoryWindowFor(prefs, "new"), limit);
    assert.equal(agentHistoryLevelFor({ ...prefs, globalEnabled: false }, "new"), "off");
    assert.equal(agentHistoryLevelFor({
      ...prefs, domLevelByThreadId: { one: "fixed-high" as const },
    }, "one"), "fixed-high");
  }
});

test("old numeric per-chat preferences remain readable until explicitly changed", () => {
  const old = { ...legacy, globalEnabled: true, globalWindowRows: 8 as const,
    windowRowsByThreadId: { olderChat: 16 as const } };
  assert.equal(agentHistoryWindowFor(old, "olderChat"), 16);
  assert.equal(agentHistoryWindowFor(old, "new"), 8);
  assert.equal(agentHistoryWindowFor({ ...old,
    domLevelByThreadId: { olderChat: "fixed-low" as const } }, "olderChat"), 5);
});

test("UI collapses legacy High/Low into three modes without changing persisted semantics", () => {
  for (const mode of ["low", "high", "dynamic-low", "dynamic-high"] as const) {
    assert.equal(agentHistoryUiModeFor(mode), "dynamic");
  }
  for (const mode of ["legacy", "fixed-low", "fixed-high"] as const) {
    assert.equal(agentHistoryUiModeFor(mode), "fixed");
  }
  assert.equal(agentHistoryUiModeFor("off"), "off");
  assert.equal(agentHistoryStorageModeForUi("dynamic"), "dynamic");
  assert.equal(agentHistoryStorageModeForUi("fixed"), "fixed-low");
  assert.equal(agentHistoryStorageModeForUi("off"), "off");

  const global = readSrc("features/settings/tabs/chat-tab.tsx");
  const section = global.split('\n          label={t("settings.chat.agentHistoryGlobal")}')[1]
    ?.split('\n          label={t("settings.chat.webSearch.images")}')[0] ?? "";
  const options = [...section.matchAll(/<SelectItem value="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(options, ["dynamic", "fixed", "off"]);
  assert.doesNotMatch(section, /<Switch/);
  const menu = readSrc("features/chat/components/chat-header-menu.tsx");
  const submenu = menu.split('data-test-id="agent-dom-level-trigger"')[1]
    ?.split('<Item\n        glyph={<HugeiconsIcon icon={ForkIcon}')[0] ?? "";
  assert.match(submenu, /agent-dom-level-\$\{value\}/);
  assert.doesNotMatch(submenu, /agent-window-size-|setEnabled\(/);
});


test("legacy mode preferences hydrate to one medium dynamic or five-row fixed profile", () => {
  for (const mode of ["low", "high", "dynamic-low", "dynamic-high", "dynamic"]) {
    assert.equal(normalizeAgentHistoryDomMode(mode), "dynamic");
  }
  for (const mode of ["fixed-low", "fixed-high"]) {
    assert.equal(normalizeAgentHistoryDomMode(mode), "fixed-low");
  }
  assert.equal(normalizeAgentHistoryDomMode("off"), "off");
  assert.equal(normalizeAgentHistoryDomMode("nonsense"), undefined);
  const state = { ...legacy, globalEnabled: true, globalDomLevel: "dynamic" as const };
  assert.equal(agentHistoryWindowFor(state, "new"), 16);
  assert.equal(agentHistoryWindowFor({
    ...state, domLevelByThreadId: { fixed: "fixed-low" as const },
  }, "fixed"), 5);
});
