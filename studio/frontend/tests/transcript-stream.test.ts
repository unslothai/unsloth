// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readTranscriptStream } from "../src/features/audio/transcript-stream.ts";

function stream(text: string, chunkSize = 3) {
  const bytes = new TextEncoder().encode(text);
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (let offset = 0; offset < bytes.length; offset += chunkSize)
        controller.enqueue(bytes.slice(offset, offset + chunkSize));
      controller.close();
    },
  });
}

test("transcription stream handles fragmented Unicode and retains the saved result", async () => {
  const updates: string[] = [];
  const result = await readTranscriptStream(
    stream(
      [
        { type: "progress", text: "你好" },
        { type: "heartbeat" },
        {
          type: "complete",
          text: "你好 world",
          model: "tiny",
          record: { id: "saved" },
        },
      ]
        .map((event) => JSON.stringify(event))
        .join("\n"),
    ),
    (update) => updates.push(update.text),
  );
  assert.deepEqual(updates, ["你好"]);
  assert.equal(result.text, "你好 world");
  assert.equal(result.record?.id, "saved");
});

test("a broken stream does not treat partial text as a completed transcript", async () => {
  await assert.rejects(
    readTranscriptStream(
      stream('{"type":"progress","text":"partial"}\n'),
      () => {},
    ),
    /before the result arrived/,
  );
});

test("server failures reach the caller", async () => {
  await assert.rejects(
    readTranscriptStream(
      stream('{"type":"error","message":"Model unavailable"}\n'),
      () => {},
    ),
    /Model unavailable/,
  );
});

test("a failed save still delivers the complete transcript", async () => {
  const result = await readTranscriptStream(
    stream(
      '{"type":"complete","text":"keep me","model":"tiny","record":null}\n',
    ),
    () => {},
  );
  assert.equal(result.text, "keep me");
  assert.equal(result.record, null);
});
