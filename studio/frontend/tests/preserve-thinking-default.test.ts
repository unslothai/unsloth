// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { preserveThinkingDefaultFromLoad } from "../src/features/chat/lib/resolve-preserve-thinking-default.ts";

test("an advertised supported preserve-thinking default is enabled", () => {
  assert.equal(
    preserveThinkingDefaultFromLoad({
      supports_preserve_thinking: true,
      preserve_thinking_default: true,
    }),
    true,
  );
});

test("unsupported and older responses keep preserve thinking off", () => {
  assert.equal(
    preserveThinkingDefaultFromLoad({
      supports_preserve_thinking: false,
      preserve_thinking_default: true,
    }),
    false,
  );
  assert.equal(
    preserveThinkingDefaultFromLoad({ supports_preserve_thinking: true }),
    false,
  );
  assert.equal(preserveThinkingDefaultFromLoad({}), false);
});
