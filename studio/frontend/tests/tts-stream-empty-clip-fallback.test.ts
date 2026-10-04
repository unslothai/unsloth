// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const source = readSrc("features/chat/hooks/use-tts-player.ts");

test("a 200 stream that carried no PCM is a miss, not a played sentence", () => {
  // The server ends a failed synthesis as a clean 200 with an empty body; resolving true
  // there skipped the blob fallback and dropped the sentence without a word.
  assert.match(source, /openGate\(\);\s*(?:\/\/[^\n]*\n\s*)*finish\(scheduledAny \|\| requestIdRef\.current !== reqId\);/);
  // The explicit stop path still resolves true: nothing to fall back to.
  assert.match(source, /const unwind = \(\) => finish\(true\);/);
});
