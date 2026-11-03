// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { prepareBackgroundImage } from "../src/features/chat-background/image.ts";

// Rejected before any decoding, so no DOM is needed.
test("uploads that fail validation reject with 'file'", async () => {
  await assert.rejects(prepareBackgroundImage(new Blob([new Uint8Array(4)], { type: "image/gif" })), { message: "file" });
  await assert.rejects(prepareBackgroundImage(new Blob([], { type: "image/png" })), { message: "file" });
});
