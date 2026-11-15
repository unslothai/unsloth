// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";

const read = (path: string) => readFileSync(new URL(path, import.meta.url), "utf8");
const root = read("../src/app/routes/__root.tsx");
const page = read("../src/features/chat/chat-page.tsx");

test("leaving the chat route pauses voice mode and coming back resumes it", () => {
  // The page stays mounted off its route, so its unmount cleanup never runs and nothing on the
  // other pages shows the loop: left listening, the mic kept sending turns unseen. Ending the
  // session instead unloaded every voice model, so it pauses (frames dropped, models kept).
  assert.match(root, /<ChatPageSlot search=\{chatSearch\} active=\{isChatRoute\} \/>/);
  assert.match(page, /if \(!active && !paused\) \{\s*pausedOffRouteRef\.current = true;\s*requestVoicePauseToggle\(\);/);
  assert.match(page, /else if \(active && pausedOffRouteRef\.current\) \{\s*pausedOffRouteRef\.current = false;\s*if \(paused\) requestVoicePauseToggle\(\);/);
});
