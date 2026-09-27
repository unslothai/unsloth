// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const SKILLS_API_SOURCE = readFileSync(
  new URL("../src/features/chat/api/skills-api.ts", import.meta.url),
  "utf8",
);
const CHAT_ADAPTER_SOURCE = readFileSync(
  new URL("../src/features/chat/api/chat-adapter.ts", import.meta.url),
  "utf8",
);

test("token counting waits for the initial skills discovery", () => {
  assert.match(SKILLS_API_SOURCE, /initialized: boolean;/);
  assert.match(SKILLS_API_SOURCE, /initialized: false,/);
  assert.match(SKILLS_API_SOURCE, /initialized: true,/);
  assert.match(
    SKILLS_API_SOURCE,
    /if \(pending\) await Promise\.race\(\[pending\.catch\(\(\) => undefined\), deadline\]\);/,
  );
  assert.match(SKILLS_API_SOURCE, /let stale = !snapshot\.initialized;/);
  assert.match(CHAT_ADAPTER_SOURCE, /await settleSkillsForText\(""\);/);
  assert.match(
    CHAT_ADAPTER_SOURCE,
    /const hasEnabledSkills = getSkillsSnapshot\(\)\.skills\.some\(/,
  );
});

test("a pre-send catalog re-read cannot hold the request past the settle deadline", () => {
  assert.match(SKILLS_API_SOURCE, /const SETTLE_TIMEOUT_MS = 3000;/);
  assert.match(
    SKILLS_API_SOURCE,
    /await Promise\.race\(\[listSkills\(true\)\.catch\(\(\) => undefined\), deadline\]\);/,
  );
});

test("token counting lists the skill tools only when the completion would", () => {
  const countExtras = CHAT_ADAPTER_SOURCE.slice(
    CHAT_ADAPTER_SOURCE.indexOf("export async function buildLocalTokenCountExtras"),
    CHAT_ADAPTER_SOURCE.indexOf("mcp_enabled: mcpEnabledForChat"),
  );
  assert.match(
    countExtras,
    /\.\.\.\(hasEnabledSkills \? \["read_skill", "create_skill"\] : \[\]\),/,
  );
  assert.doesNotMatch(countExtras, /^\s*"read_skill",$/m);
  assert.doesNotMatch(countExtras, /^\s*"create_skill",$/m);
});

test("request building waits for skills and preserves the launcher tool catalog", () => {
  const payloadBuilder = CHAT_ADAPTER_SOURCE.slice(
    CHAT_ADAPTER_SOURCE.indexOf("const buildRequestPayload = async"),
    CHAT_ADAPTER_SOURCE.indexOf(
      "while (true)",
      CHAT_ADAPTER_SOURCE.indexOf("const buildRequestPayload = async"),
    ),
  );
  assert.match(
    payloadBuilder,
    /if \(supportsStudioToolsForThisTurn\) \{\s*await settleSkillsForText\(lastUserText\(outboundMessages\)\);/,
  );
  assert.match(
    payloadBuilder,
    /resolveEnableTools\(\s*supportsTools &&\s*\([\s\S]*hasEnabledSkills[\s\S]*\),\s*allToolsOff,\s*\)\s*\? \{\s*enable_tools: true/,
  );
  const localToolCatalog = payloadBuilder.slice(
    payloadBuilder.indexOf("// Sent for every local chat"),
  );
  assert.doesNotMatch(localToolCatalog, /^\s*"read_skill",$/m);
  assert.doesNotMatch(localToolCatalog, /^\s*"create_skill",$/m);
});

test("a sign-out drops the module-level skills snapshot", () => {
  assert.ok(SKILLS_API_SOURCE.includes("window.addEventListener(AUTH_SESSION_CLEARED_EVENT"));
  assert.ok(SKILLS_API_SOURCE.includes("publish(EMPTY_SNAPSHOT)"));
});
