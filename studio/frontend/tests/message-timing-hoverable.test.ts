// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

test("the reply stats card stays open while the pointer moves onto it", () => {
  // TooltipProvider defaults to non-hoverable content; the stats card opts back in so its
  // prompt and generation figures can be selected and copied.
  const timing = readSrc("components/assistant-ui/message-timing.tsx");
  const root = timing.match(/<Tooltip\b[^>]*>/);
  assert.ok(root, "message timing renders a Tooltip");
  assert.match(root[0], /disableHoverableContent=\{false\}/);
});
