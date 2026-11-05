// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A failed generation's toast shows the backend's reason, the exception class and first line.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const MISMATCH =
  "Video generation failed. RuntimeError: Input type (CPUBFloat16Type) and weight type " +
  "(CUDABFloat16Type) should be the same";

test("the backend's failure reason is the toast text", async () => {
  const { videoFailureToastText } = await import("../src/features/video/generation-failure.ts");
  assert.equal(videoFailureToastText(MISMATCH), MISMATCH);
  assert.equal(videoFailureToastText(null), "Video generation failed");
});

test("a failure whose cause mentions cancelled still shows", async () => {
  const { videoFailureToastText } = await import("../src/features/video/generation-failure.ts");
  const reason = "Video generation failed. CancelledError: the offload stream was cancelled";
  assert.equal(videoFailureToastText(reason), reason);
});

test("the user's own Cancel stays silent", async () => {
  const { videoFailureToastText } = await import("../src/features/video/generation-failure.ts");
  assert.equal(videoFailureToastText("Video generation was cancelled."), null);
});

test("both failure paths on the Video page toast through it", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.equal(page.match(/const msg = videoFailureToastText\([pg]\.error\);/g)?.length, 2);
  assert.doesNotMatch(page, /if \(!msg\.toLowerCase\(\)\.includes\("cancelled"\)\)/);
});
