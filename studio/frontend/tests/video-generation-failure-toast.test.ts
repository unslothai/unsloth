// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A failed generation's toast shows the backend's reason, the exception class and first line.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const MISMATCH =
  "Video generation failed. RuntimeError: Input type (CPUBFloat16Type) and weight type " +
  "(CUDABFloat16Type) should be the same";

test("the backend's failure reason becomes the toast description", async () => {
  const { videoFailureToast } = await import("../src/features/video/generation-failure.ts");
  assert.deepEqual(videoFailureToast(MISMATCH), {
    title: "Video generation failed.",
    errorType: "RuntimeError",
    detail: "Input type (CPUBFloat16Type) and weight type (CUDABFloat16Type) should be the same",
  });
  assert.deepEqual(videoFailureToast("Video generation failed. KeyError"), {
    title: "Video generation failed.",
    errorType: "KeyError",
    detail: null,
  });
});

test("fixed backend text and an empty error keep the logged-failure title", async () => {
  const { videoFailureToast } = await import("../src/features/video/generation-failure.ts");
  const oom =
    "Video generation failed. The renderer crashed. See Settings > Logs.";
  assert.deepEqual(videoFailureToast(oom), {
    title: "Video generation failed.",
    errorType: null,
    detail: "The renderer crashed. See Settings > Logs.",
  });
  for (const empty of [null, undefined, "", "  "]) {
    assert.deepEqual(videoFailureToast(empty), {
      title: "Video generation failed.",
      errorType: null,
      detail: null,
    });
  }
});

test("a failure whose cause mentions cancelled still shows", async () => {
  const { videoFailureToast } = await import("../src/features/video/generation-failure.ts");
  const failure = videoFailureToast(
    "Video generation failed. CancelledError: the offload stream was cancelled",
  );
  assert.equal(failure?.errorType, "CancelledError");
});

test("the user's own Cancel stays silent", async () => {
  const { videoFailureToast } = await import("../src/features/video/generation-failure.ts");
  assert.equal(videoFailureToast("Video generation was cancelled."), null);
});

test("every failure toast on the Video page goes through one helper", () => {
  const page = readSrc("features/video/video-page.tsx");
  assert.equal(page.match(/showVideoFailureToast\([pg]\.error\);/g)?.length, 2);
  assert.match(page, /if \(!showVideoFailureToast\(refusal\) && refusal\) toast\.error\(refusal\);/);
  assert.match(page, /action: generationFailureLogsAction\(title\)/);
  assert.doesNotMatch(page, /if \(!msg\.toLowerCase\(\)\.includes\("cancelled"\)\)/);
});
