// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import type { VideoGenerationDefaults } from "../src/features/video/api.ts";
import { matchedCanvas, requestCanvas } from "../src/features/video/keyframe-canvas.ts";

const h3: VideoGenerationDefaults = {
  steps: 20,
  guidance: 1,
  num_frames: 124,
  fps: 24,
  frame_step: 17,
  frame_offset: 5,
  duration_presets: [5, 10, 14.4],
  resolution_multiple: 32,
  resolution_presets: [[1344, 768], [768, 1344], [960, 544], [544, 960], [864, 480], [480, 864]],
  canvas_short_edge: 768,
  canvas_max_pixels: 768 * 1344,
  keyframe_keeps_canvas: true,
};

const pagePromise = readFile(new URL("../src/features/video/video-page.tsx", import.meta.url), "utf8");

function hook(page: string, name: string, kind: "useCallback" | "useMemo") {
  const start = page.indexOf(`const ${name} = ${kind}(`);
  assert.ok(start >= 0, `video-page.tsx no longer declares ${name} with ${kind}`);
  let depth = 0;
  let end = -1;
  for (let i = page.indexOf("(", start); i < page.length; i++) {
    const ch = page[i];
    if (ch === "(") depth++;
    else if (ch === ")" && --depth === 0) {
      end = i;
      break;
    }
  }
  const call = page.slice(start, end + 1);
  const depsStart = call.lastIndexOf("[");
  const deps = call
    .slice(depsStart + 1, call.lastIndexOf("]"))
    .split(",")
    .map((d) => d.trim())
    .filter(Boolean);
  return { body: call.slice(0, depsStart), deps };
}

const IDENT = /[A-Za-z_$][\w$]*(?:\?\.[A-Za-z_$][\w$]*)*/g;

test("Match source on MiniMax-H3 sends the portrait canvas it shows", () => {
  // 720x1280 portrait start image: the page shows 768x1344 and must send it, not the held 864x480.
  const matched = matchedCanvas(720, 1280, h3);
  assert.deepEqual(matched, [768, 1344]);
  assert.deepEqual(requestCanvas(true, matched, [864, 480], true), { width: 768, height: 1344 });
  assert.deepEqual(requestCanvas(true, matchedCanvas(1000, 1000, h3), [864, 480], true), { width: 768, height: 768 });
  assert.deepEqual(requestCanvas(true, matchedCanvas(1920, 1080, h3), [864, 480], true), { width: 1344, height: 768 });
  assert.deepEqual(requestCanvas(false, matched, [544, 960], true), { width: 544, height: 960 });
});

test("Generate sends the size the page shows, never one captured before the start frame loaded", async () => {
  const page = await pagePromise;
  const generate = hook(page, "handleGenerate", "useCallback");
  const spread = generate.body.match(/\.\.\.(requestCanvas\([^)]*\)|[A-Za-z_$][\w$]*),/);
  assert.ok(spread, "handleGenerate no longer spreads the request canvas into the body");
  const inputs = (spread[1].replace(/^requestCanvas/, "").match(IDENT) ?? []).filter(Boolean);
  assert.ok(inputs.length > 0);
  for (const input of inputs) {
    assert.ok(
      generate.deps.includes(input),
      `handleGenerate reads ${input} for the request size but does not list it as a dependency`,
    );
  }
  if (inputs.length === 1 && !spread[1].startsWith("requestCanvas")) {
    const memo = hook(page, inputs[0], "useMemo");
    const memoInputs = new Set(memo.body.slice(memo.body.indexOf("requestCanvas(")).match(IDENT) ?? []);
    for (const name of ["sourceSized", "matchedResolution", "videoSize", "autoMatchSource"]) {
      assert.ok(memoInputs.has(name), `${inputs[0]} no longer reads ${name}`);
      assert.ok(memo.deps.includes(name), `${inputs[0]} reads ${name} but does not list it`);
    }
  }
});
