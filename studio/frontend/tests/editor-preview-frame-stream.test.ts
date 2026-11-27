// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFrames } from "../src/features/chat/editor-preview/frame-stream.ts";

function stream(...chunks: number[][]) {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(Uint8Array.from(chunk));
      controller.close();
    },
  });
}

test("decodes split frame headers and bodies and discards heartbeats", async () => {
  const frames: number[][] = [];
  await readFrames(stream([0, 0], [0, 3, 10], [20, 30, 0, 0, 0], [0, 0, 0, 0, 2, 40, 50]),
    (frame) => frames.push([...frame]));
  assert.deepEqual(frames, [[10, 20, 30], [40, 50]]);
});

test("rejects oversized frames before allocating their buffers", async () => {
  await assert.rejects(readFrames(stream([0, 64, 0, 1]), () => assert.fail()), /too large/);
});
