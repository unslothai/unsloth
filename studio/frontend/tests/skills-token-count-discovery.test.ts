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
  assert.match(SKILLS_API_SOURCE, /if \(pending\) await pending\.catch/);
  assert.match(SKILLS_API_SOURCE, /let stale = !snapshot\.initialized;/);
  assert.match(CHAT_ADAPTER_SOURCE, /await settleSkillsForText\(""\);/);
  assert.match(
    CHAT_ADAPTER_SOURCE,
    /const hasEnabledSkills = getSkillsSnapshot\(\)\.skills\.some\(/,
  );
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
    /supportsTools &&\s*\([\s\S]*hasEnabledSkills[\s\S]*\)\s*\? \{\s*enable_tools: true/,
  );
  const localToolCatalog = payloadBuilder.slice(
    payloadBuilder.indexOf("// Sent for every local chat"),
  );
  assert.doesNotMatch(localToolCatalog, /^\s*"read_skill",$/m);
  assert.doesNotMatch(localToolCatalog, /^\s*"create_skill",$/m);
});
