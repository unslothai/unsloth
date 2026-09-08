// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readSseJsonEvents } from "../src/lib/sse-json-events.ts";

function streamOf(
  chunks: string[],
  { gapMs = 0, trailingSilence = false } = {},
): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  let index = 0;
  return new ReadableStream({
    async pull(controller) {
      if (index >= chunks.length) {
        if (!trailingSilence) controller.close();
        return;
      }
      if (gapMs) await new Promise((resolve) => setTimeout(resolve, gapMs));
      controller.enqueue(encoder.encode(chunks[index++]));
    },
  });
}

async function collect<T>(events: AsyncGenerator<T>): Promise<T[]> {
  const out: T[] = [];
  for await (const event of events) out.push(event);
  return out;
}

test("frames split on any CRLF or LF blank-line separator", async () => {
  const events = await collect(
    readSseJsonEvents<{ n: number }>(
      streamOf([
        'data: {"n": 1}\n\n',
        'data: {"n": 2}\r\n\r\n',
        'data: {"n"',
        ': 3}\n\n',
      ]),
    ),
  );
  assert.deepEqual(events, [{ n: 1 }, { n: 2 }, { n: 3 }]);
});

test("[DONE] ends the stream and unparseable frames are skipped", async () => {
  const events = await collect(
    readSseJsonEvents<{ n: number }>(
      streamOf([
        "data: not json\n\n",
        'data: {"n": 9}\n\n',
        "data: [DONE]\n\n",
        'data: {"n": 10}\n\n',
      ]),
    ),
  );
  assert.deepEqual(events, [{ n: 9 }]);
});

test("a stream that goes silent ends instead of hanging, keeping frames already seen", async () => {
  const started = Date.now();
  const events = await collect(
    readSseJsonEvents<{ n: number }>(
      streamOf(['data: {"n": 1}\n\n'], { trailingSilence: true }),
      60,
    ),
  );
  assert.deepEqual(events, [{ n: 1 }]);
  assert.ok(Date.now() - started < 2000);
});

test("without a stall bound a slow stream still delivers every frame", async () => {
  const events = await collect(
    readSseJsonEvents<{ n: number }>(
      streamOf(['data: {"n": 1}\n\n', 'data: {"n": 2}\n\n'], { gapMs: 40 }),
    ),
  );
  assert.deepEqual(events, [{ n: 1 }, { n: 2 }]);
});
