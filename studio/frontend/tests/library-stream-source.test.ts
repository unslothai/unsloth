// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Audio and video with a file of their own play from a signed link the element range-requests,
// so a long file is never buffered whole.

import assert from "node:assert/strict";
import test from "node:test";

import { streamsPreview } from "../src/features/library/file-name.ts";

test("audio and video with a file of their own stream", () => {
  for (const id of ["upload:abc", "audio:a1", "video:v1", "sandbox:t-1:out/song.mp3"]) {
    assert.equal(streamsPreview(id, "audio"), true, id);
    assert.equal(streamsPreview(id, "video"), true, id);
  }
});

test("everything else keeps its blob", () => {
  // Attachments live inside their message; images and PDFs are small enough to buffer.
  assert.equal(streamsPreview("attachment:m:a", "video"), false);
  assert.equal(streamsPreview("upload:abc", "image"), false);
  assert.equal(streamsPreview("upload:abc", "pdf"), false);
  assert.equal(streamsPreview("upload:abc", null), false);
  assert.equal(streamsPreview("model:training:/x", "video"), false);
  assert.equal(streamsPreview("upload", "video"), false);
  assert.equal(streamsPreview(":upload", "video"), false);
});
