// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { chatMemoryBytes } from "../src/features/settings/lib/chat-memory-size.ts";
import {
  agentsMdEditCount,
  forgetAgentsMd,
} from "../src/features/chat/utils/agents-md.ts";
import { createSettingsSearchIndex } from "../src/features/settings/settings-search.ts";
import { readSrcAsync } from "./helpers/kit.ts";

const chatTab = await readSrcAsync("features/settings/tabs/chat-tab.tsx");
const section = await readSrcAsync(
  "features/settings/components/chat-memory-section.tsx",
);
const adapter = await readSrcAsync("features/chat/api/chat-adapter.ts");

test("memory size is counted the way the backend saves it", () => {
  assert.equal(chatMemoryBytes(""), 0);
  assert.equal(chatMemoryBytes("  \n\n "), 0);
  assert.equal(chatMemoryBytes(" a\r\nb \n\n"), "a\nb\n".length);
  assert.equal(chatMemoryBytes("é"), 3);
});

test("the Chat tab shows the memory section and search finds it", () => {
  assert.match(chatTab, /<ChatMemorySection \/>/);
  assert.match(chatTab, /lazy\(\(\) =>\s*import\("..\/components\/chat-memory-section"\)/);
  const index = createSettingsSearchIndex({
    desktop: false,
    closeToTray: false,
    menuBarIcon: false,
  });
  assert.ok(index.chat.includes("settings.chat.memory.title"));
});

test("a saved edit is not hidden by the AGENTS.md read cache", () => {
  const before = agentsMdEditCount();
  forgetAgentsMd();
  assert.equal(agentsMdEditCount(), before + 1);
  assert.match(section, /forgetAgentsMd\(\);/);
  assert.match(adapter, /hit\.edits === edits/);
});

test("the memory switch and editor have accessible names", () => {
  assert.match(section, /aria-label=\{t\("settings\.chat\.memory\.enable"\)\}/);
  assert.match(section, /htmlFor=\{editorId\}/);
  assert.match(section, /role="alert"/);
});
