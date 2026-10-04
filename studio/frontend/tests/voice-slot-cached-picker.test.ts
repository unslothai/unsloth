// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the voice picker offers no cached CSM GGUF, which /voice/load refuses", () => {
  const list = readSrc("features/chat/chat-page.tsx").match(
    /const TTS_REPO_KEYWORDS = \[([^\]]*)\]/,
  );
  assert.ok(list, "TTS_REPO_KEYWORDS not found");
  assert.doesNotMatch(list[1], /"csm"/);
});
