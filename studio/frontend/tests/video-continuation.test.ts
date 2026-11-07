// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  CONTINUATION_KEEP_OPTIONS,
  CONTINUATION_MAX_SECONDS,
  continuationDurationWarning,
  continuationLengthError,
  continuationRequest,
  supportsVideoContinuation,
} from "../src/features/video/video-continuation.ts";

const page = await readFile(
  new URL("../src/features/video/video-page.tsx", import.meta.url),
  "utf8",
);

test("only a status that declares it offers continuation (Cosmos3-Nano); older backends do not", () => {
  assert.equal(supportsVideoContinuation({ supports_video_continuation: true }), true);
  for (const status of [
    { supports_video_continuation: false },
    {},
    null,
    undefined,
  ]) {
    assert.equal(supportsVideoContinuation(status), false);
  }
});

test("continuing from the end of the clip is the default choice", () => {
  assert.deepEqual(
    CONTINUATION_KEEP_OPTIONS.map((option) => option.value),
    ["last", "first"],
  );
  assert.match(page, /useState<VideoContinuationKeep>\("last"\)/);
});

test("the request carries the clip only when the model takes one and one is staged", () => {
  assert.deepEqual(continuationRequest(true, "data:video/mp4;base64,AA", "first"), {
    continuation_video: "data:video/mp4;base64,AA",
    continuation_keep: "first",
  });
  assert.deepEqual(continuationRequest(false, "data:video/mp4;base64,AA", "last"), {});
  assert.deepEqual(continuationRequest(true, null, "last"), {});
  assert.deepEqual(continuationRequest(true, undefined, "last"), {});
});

test("a length that is all kept frames is refused before Generate", () => {
  assert.match(continuationLengthError(5) ?? "", /keeps 5 of its frames/);
  assert.equal(continuationLengthError(9), null);
  assert.equal(continuationLengthError(189), null);
});

test("the clip section renders only for a family that declares it, and a clip and a keyframe exclude each other", () => {
  const start = page.indexOf('data-testid="video-continuation"');
  assert.ok(start > 0);
  const head = page.slice(0, start);
  assert.match(head.slice(head.lastIndexOf("{supports")), /^\{supportsContinuation && \(/);
  const section = page.slice(start, page.indexOf("{cropPictureIndex !== null", start));
  assert.match(section, /setFirstFrame\(null\);\s*setLastFrame\(null\);/);
  const keyframes = page.slice(
    page.indexOf('data-testid="video-keyframes"'),
    page.indexOf("{supportsReferences && ("),
  );
  assert.equal(keyframes.match(/if \(next\) dropContinuationClip\(/g)?.length, 2);
  // Late uploads read the live inputs, not the closure they were started with.
  assert.match(section, /liveInputs\.current\.firstFrame \|\| liveInputs\.current\.lastFrame/);
  assert.match(page, /if \(!liveInputs\.current\.continuationClip\) return;\s*setContinuationClip\(null\);/);
  assert.match(page, /if \(status\?\.loaded && !supportsContinuation\) setContinuationClip\(null\);/);
});

test("the payload spreads the continuation fields and the gallery labels a continued clip", () => {
  assert.match(
    page,
    /\.\.\.continuationRequest\(supportsContinuation, stagedClip\?\.dataUrl, continuationKeep\)/,
  );
  assert.match(page, /v2v: "Continued a clip"/);
});

test("a picked clip past the 15 s cap is flagged before Generate; unknown or shorter clips are not", () => {
  assert.equal(CONTINUATION_MAX_SECONDS, 15);
  assert.equal(continuationDurationWarning(undefined), null);
  assert.equal(continuationDurationWarning(3.9), null);
  assert.equal(continuationDurationWarning(15), null);
  assert.match(continuationDurationWarning(32.04) ?? "", /32\.0 s.*up to 15 s/);
  assert.match(page, /continuationDurationWarning\(continuationClip\.durationSeconds\)/);
});

test("a start frame and a clip replace each other, and the user is told which one went", () => {
  assert.match(page, /Removed the start frame: the clip to continue replaces it/);
  assert.match(page, /Removed the clip to continue: \$\{why\}/);
  assert.match(page, /dropContinuationClip\("a start frame replaces it"\)/);
  assert.match(page, /dropContinuationClip\("an end frame replaces it"\)/);
});

test("a staged clip sizes the canvas through Match source, as a start frame does", () => {
  assert.match(
    page,
    /const canvasClip = canvasKeyframe \? null : supportsContinuation \? \(continuationClip\?\.dataUrl \?\? null\) : null;/,
  );
  assert.match(page, /const canvasSource = canvasKeyframe \?\? canvasClip;/);
  assert.match(page, /media\.onloadedmetadata = \(\) => done\(\[media\.videoWidth, media\.videoHeight\]\);/);
  assert.match(page, /const sourceSized = matchSource && canvasSource != null && matchedResolution != null;/);
  assert.match(page, /canvasSource && matchedResolution \? \(\s*<SelectItem value=\{MATCH_SOURCE\}>Match source<\/SelectItem>/);
});
