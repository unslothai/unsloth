// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  SttModelNotDownloadedError,
  sttRequestError,
} from "../src/features/chat/adapters/stt-errors.ts";

test("a not-downloaded 409 asks for the download, from either call", () => {
  // A segment can be the first to see this, when a short recording ends before
  // the fire-and-forget preload rejects.
  const error = sttRequestError(409, "STT model 'small' is not downloaded.");
  assert.ok(error instanceof SttModelNotDownloadedError);
});

test("the other 409s stay ordinary errors", () => {
  // A load cancelled for training, and a model switch mid-request.
  for (const detail of [
    "Dictation model loading was cancelled so training could start.",
    "The dictation model changed while this recording was being prepared.",
  ]) {
    const error = sttRequestError(409, detail);
    assert.ok(!(error instanceof SttModelNotDownloadedError));
    assert.equal(error.message, detail);
  }
});

test("another status is never mistaken for it", () => {
  assert.ok(
    !(
      sttRequestError(500, "something is not downloaded") instanceof
      SttModelNotDownloadedError
    ),
  );
});
