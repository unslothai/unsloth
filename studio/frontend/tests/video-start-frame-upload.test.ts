// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

// A hidden frame must never be sent: the user cannot see what conditioned the clip.

const page = await readFile(
  new URL("../src/features/video/video-page.tsx", import.meta.url),
  "utf8",
);

function block(after: string, until: string): string {
  const start = page.indexOf(after);
  assert.ok(start >= 0, `missing: ${after}`);
  const end = page.indexOf(until, start);
  assert.ok(end > start, `missing after ${after}: ${until}`);
  return page.slice(start, end);
}

test("support is read per frame from the status, with the legacy flag as fallback", () => {
  assert.match(
    page,
    /const \{ first: supportsFirstFrame, last: supportsLastFrame \} = keyframeSupport\(status\);/,
  );
});

test("each dropzone renders only where its frame is supported", () => {
  const controls = block(
    'data-testid="video-keyframes"',
    "{supportsReferences && (",
  );
  const start = controls.indexOf('removeLabel="Remove start frame"');
  const end = controls.indexOf('removeLabel="Remove end frame"');
  assert.ok(start > 0 && end > 0);
  const gateBefore = (index: number) => {
    const head = controls.slice(0, index);
    return head.slice(head.lastIndexOf("{supports")).split(" ")[0];
  };
  assert.equal(gateBefore(start), "{supportsFirstFrame");
  assert.equal(gateBefore(end), "{supportsLastFrame");
  assert.match(
    controls,
    /supportsLastFrame \? "grid grid-cols-2 gap-2" : "grid gap-2"/,
  );
});

test("the section and its wording follow the loaded family", () => {
  assert.match(
    page,
    /const supportsKeyframes = supportsFirstFrame \|\| supportsLastFrame;/,
  );
  assert.match(
    page,
    /\{supportsLastFrame \? "Start and end frame" : "Start frame"\}/,
  );
  assert.match(
    page,
    /"Generate a video from a prompt, or animate a start frame"/,
  );
});

test("the request carries each frame only where it is supported", () => {
  assert.match(
    page,
    /first_frame: supportsFirstFrame \? firstFrame \?\? undefined : undefined,/,
  );
  assert.match(
    page,
    /last_frame: supportsLastFrame \? lastFrame \?\? undefined : undefined,/,
  );
});

test("a frame the newly loaded model cannot take is dropped, each on its own", () => {
  assert.match(
    page,
    /if \(status\?\.loaded && !supportsFirstFrame\) setFirstFrame\(null\);/,
  );
  assert.match(
    page,
    /if \(status\?\.loaded && !supportsLastFrame\) setLastFrame\(null\);/,
  );
});

test("the 1:4 to 4:1 warning is MiniMax-H3's alone", () => {
  // A preset family centre-crops any picture onto its nearest preset, so it never refuses one.
  assert.match(
    page,
    /const canvasRefusesAspect = !status\?\.defaults\?\.canvas_from_presets;/,
  );
  assert.match(
    page,
    /\{canvasKeyframe && !matchedResolution && canvasRefusesAspect && \(/,
  );
});
