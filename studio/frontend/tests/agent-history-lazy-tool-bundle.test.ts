// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.
import assert from "node:assert/strict";
import test from "node:test";
import { readSrc } from "./helpers/kit.ts";

test("opt-in Agent History renderer remains lazy while current media tools remain registered", () => {
  const source = readSrc("components/assistant-ui/thread.tsx");
  assert.match(source, /const AgentHistoryScrollMessages = lazy\(\(\) =>/);
  assert.match(source, /import\("@\/components\/assistant-ui\/agent-history-scroll-messages"\)/);
  assert.match(source, /agentHistoryEnabled\s*\?\s*\(/);
  assert.match(source, /<Suspense fallback=\{null\}>/);
  assert.match(source, /<AgentHistoryScrollMessages\b/);
  assert.match(source, /<ProgressiveMessages\b/);
  // Upstream added VideoGenerationToolUI to the shared media card module after R6V3.
  // Preserve the current image/video implementation instead of moving only one export
  // behind an ineffective lazy boundary while VideoGenerationToolUI eagerly imports it.
  for (const tool of ["ImageGenerationToolUI", "VideoGenerationToolUI", "RenderHtmlToolUI"]) {
    assert.match(source, new RegExp(tool + "Confirmable"));
  }
});
