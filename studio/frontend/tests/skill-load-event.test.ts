// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { skillLoadCardEvent } from "../src/features/chat/api/skill-load-event.ts";
import { readSrc } from "./helpers/kit.ts";

const event = { type: "skill_load", load_id: "load-1", name: "skill-creator" };

test("backend preload has a distinct UI-only card identity, not a model read_skill call", () => {
  const start = skillLoadCardEvent({ ...event, status: "loading" });
  assert.equal(start.tool_name, "studio_load_skill");
  assert.equal(start.type, "tool_start");
  assert.equal(start.tool_call_id, "load-1");
  const loaded = skillLoadCardEvent({
    ...event,
    status: "loaded",
    detail: "Complete SKILL.md read",
  });
  assert.equal(loaded.type, "tool_end");
  assert.equal(loaded.result, "Complete SKILL.md read");
  assert.equal(loaded.tool_call_id, start.tool_call_id);
  const denied = skillLoadCardEvent({
    ...event,
    status: "unavailable",
    detail: "Not loaded",
  });
  assert.equal(denied.result, "Not loaded");
});

test("Ask reuses the real scoped confirmation channel", () => {
  const approval = skillLoadCardEvent({
    ...event,
    status: "awaiting_approval",
    approval_id: "approval-1",
  });
  assert.equal(approval.awaiting_confirmation, true);
  assert.equal(approval.approval_id, "approval-1");
});

test("legacy and durable streams normalize preload, and history omits UI evidence", () => {
  const legacy = readSrc("features/chat/api/chat-api.ts");
  const durable = readSrc("features/chat/api/chat-generation-api.ts");
  assert.match(legacy, /parsed.type === "skill_load"/);
  assert.match(durable, /frameType === "skill_load"/);
  const adapter = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    adapter,
    /if \(toolPart.toolName === "studio_load_skill"\) continue;/,
  );
  assert.match(
    adapter,
    /!chunk.choices && toolEvent.tool_name !== "studio_load_skill"/,
  );
  const thread = readSrc("components/assistant-ui/thread.tsx");
  assert.match(thread, /studio_load_skill: ReadSkillToolUIConfirmable/);
  assert.match(thread, /enabled=\{supportsTools && codeToolsEffective\}/);
  assert.match(thread, /const codeToolsEffective = useChatRuntimeStore\(codeToolsOn\);/);
  const ui = readSrc("components/assistant-ui/tool-ui-read-skill.tsx");
  assert.match(ui, /Skill not loaded/);
  assert.match(ui, /Loaded \$\{name\}/);
});
