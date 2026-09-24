// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0


import assert from "node:assert/strict";
import test from "node:test";

import {
  anyToolPillOn,
  resolveEnableTools,
  type ToolPillFlags,
} from "../src/features/chat/api/all-tools-off.ts";
import { readSrc } from "./helpers/kit.ts";

const ALL_OFF: ToolPillFlags = {
  toolsEnabled: false,
  codeToolsEnabled: false,
  artifactsEnabled: false,
  mcpEnabledForChat: false,
  ragOn: false,
  deepResearchArmed: false,
  hasEnabledSkills: false,
};

test("resolveEnableTools: master-off always wins", () => {
  assert.equal(resolveEnableTools(true, true), false);
  assert.equal(resolveEnableTools(false, true), false);
  assert.equal(resolveEnableTools(true, false), true);
  assert.equal(resolveEnableTools(false, false), false);
});

test("reporter case: Search off alone still enables tools when Code is on", () => {
  const pills = { ...ALL_OFF, codeToolsEnabled: true };
  assert.equal(anyToolPillOn(pills), true);
  // Without master-off → tools on (this is the bug the issue reported)
  assert.equal(resolveEnableTools(anyToolPillOn(pills), false), true);
  // With master-off → tools hard off
  assert.equal(resolveEnableTools(anyToolPillOn(pills), true), false);
});

test("maintainer claim: Search+Code off is only enough when every other pill is off", () => {
  assert.equal(anyToolPillOn(ALL_OFF), false);
  assert.equal(resolveEnableTools(anyToolPillOn(ALL_OFF), false), false);

  for (const key of Object.keys(ALL_OFF) as (keyof ToolPillFlags)[]) {
    const pills = { ...ALL_OFF, [key]: true };
    assert.equal(
      anyToolPillOn(pills),
      true,
      `${key} alone must light the tool loop`,
    );
    assert.equal(
      resolveEnableTools(anyToolPillOn(pills), true),
      false,
      `master-off must veto ${key}`,
    );
    assert.equal(
      resolveEnableTools(anyToolPillOn(pills), false),
      true,
      `${key} alone enables tools when master-off is unset`,
    );
  }
});

test("hosted-only branch is also vetoed by master-off", () => {
  // Mirrors the external hosted arm: web_search / web_fetch / code_execution / image_generation
  const hostedWanted = true;
  assert.equal(resolveEnableTools(hostedWanted, true), false);
  assert.equal(resolveEnableTools(hostedWanted, false), true);
});

test("default allToolsOff is false — existing pill behavior unchanged", () => {
  // Default store loads false; with Search on, tools stay on.
  assert.equal(
    resolveEnableTools(anyToolPillOn({ ...ALL_OFF, toolsEnabled: true }), false),
    true,
  );
});

test("adapter and composer wire the shared gate (no missed site)", () => {
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(adapter, /from "\.\/all-tools-off"/);
  // Every former enable_tools:true decision goes through resolveEnableTools
  const resolves = adapter.match(/resolveEnableTools\(/g) ?? [];
  assert.ok(
    resolves.length >= 4,
    `expected >=4 resolveEnableTools call sites, got ${resolves.length}`,
  );

  const composer = readSrc("features/chat/shared-composer.tsx");
  assert.match(composer, /Disable tools/);
  assert.match(composer, /setAllToolsOff/);

  // Single-chat + menu lives here (shared-composer only covers compare mode).
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /Disable tools/);
  assert.match(thread, /setAllToolsOff/);

  const store = readSrc("features/chat/stores/chat-runtime-store.ts");
  assert.match(store, /allToolsOff: loadBool\(CHAT_ALL_TOOLS_OFF_KEY,\s*false\)/);
});
