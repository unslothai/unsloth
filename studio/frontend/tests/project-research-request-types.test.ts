// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { OpenAIChatCompletionsRequest } from "../src/features/chat/types/api.ts";
import type { ResearchRun } from "../src/features/chat/types/research.ts";

test("chat request types retain deep research and immutable project context transport", () => {
  const request = {
    model: "local-model",
    messages: [{ role: "user", content: "Research this project" }],
    stream: true,
    max_tokens: 1024,
    deep_research_armed: true,
    session_id: "project-project-1",
    project_context_snapshot_id: "a".repeat(32),
    thread_id: "thread-1",
  } satisfies OpenAIChatCompletionsRequest;

  const transported = JSON.parse(JSON.stringify(request));
  assert.equal(transported.deep_research_armed, true);
  assert.equal(transported.session_id, "project-project-1");
  assert.equal(transported.project_context_snapshot_id, "a".repeat(32));
  assert.equal(transported.thread_id, "thread-1");

  const run = {
    config: { projectContextSnapshotId: "a".repeat(32) },
  } satisfies Pick<ResearchRun, "config">;
  assert.equal(run.config.projectContextSnapshotId, "a".repeat(32));
});
