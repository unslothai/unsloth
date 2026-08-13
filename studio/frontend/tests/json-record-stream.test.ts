// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Covers JSON framing and UTF-8 boundaries without reading the entire export.

import assert from "node:assert/strict";
import test from "node:test";

import {
  fileImportSource,
  streamJsonRecords,
  type TextChunk,
} from "../src/features/chat/utils/json-record-stream.ts";

async function* asChunks(text: string, size: number): AsyncGenerator<TextChunk> {
  for (let index = 0; index < text.length; index += size) {
    const slice = text.slice(index, index + size);
    yield { text: slice, bytes: Buffer.byteLength(slice) };
  }
}

async function collect(text: string, size: number): Promise<unknown[]> {
  const out: unknown[] = [];
  for await (const record of streamJsonRecords(asChunks(text, size))) out.push(record);
  return out;
}

const TRICKY = [
  { id: 1, title: 'braces } and ] inside a string' },
  { id: 2, title: 'an escaped quote \\" then a brace {' },
  { id: 3, nested: { deep: [{ deeper: ["{", "}", "[", "]"] }] } },
  { id: 4, title: "backslash at the end \\\\" },
];

test("an array is cut into records at every chunk size, including one character at a time", async () => {
  const text = JSON.stringify(TRICKY);
  for (const size of [1, 2, 7, 64, text.length, text.length * 2]) {
    assert.deepEqual(await collect(text, size), TRICKY, `chunk size ${size}`);
  }
});

test("pretty-printed and single-line arrays parse the same", async () => {
  assert.deepEqual(await collect(JSON.stringify(TRICKY, null, 2), 13), TRICKY);
});

test("JSONL, NDJSON with blank lines, and a bare object all yield their records", async () => {
  const jsonl = TRICKY.map((record) => JSON.stringify(record)).join("\n");
  assert.deepEqual(await collect(jsonl, 9), TRICKY);
  assert.deepEqual(await collect(`\n\n${jsonl}\n\n`, 9), TRICKY);
  assert.deepEqual(await collect(JSON.stringify(TRICKY[0]), 3), [TRICKY[0]]);
});

test("an empty file and an empty array yield nothing rather than throwing", async () => {
  assert.deepEqual(await collect("", 4), []);
  assert.deepEqual(await collect("[]", 1), []);
  assert.deepEqual(await collect("   \n  ", 2), []);
});

test("a record far larger than the chunk size is reassembled whole", async () => {
  const big = { id: "big", blob: "x".repeat(500_000) };
  const [record] = await collect(JSON.stringify([big, { id: "after" }]), 4096);
  assert.deepEqual(record, big);
});

test("truncated JSON fails loudly instead of importing a half-read chat", async () => {
  await assert.rejects(collect('[{"id":1},{"id":', 4), SyntaxError);
});

test("one mangled record is skipped, not treated as the end of the file", async () => {
  // Balanced framing lets the scanner skip only the damaged record.
  const text = '[{"id":1},{bad},{"id":3}]';
  for (const size of [1, 5, 64, text.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const record of streamJsonRecords(asChunks(text, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(record);
    }
    assert.deepEqual(out, [{ id: 1 }, { id: 3 }], `chunk size ${size}`);
    assert.deepEqual(malformed, ["{bad}"], `chunk size ${size}`);
  }

  // Same for JSONL, which is where the old importer's leniency was visible.
  const jsonl = ['{"id":1}', "{bad}", '{"id":3}'].join("\n");
  assert.deepEqual(await collect(jsonl, 3), [{ id: 1 }, { id: 3 }]);
});

test("a row that loses its closing brace does not swallow the rows after it", async () => {
  // Recover later rows using JSONL boundaries.
  const jsonl = ['{"id":1}', '{"id":2', '{"id":3}', '{"id":4}'].join("\n");
  for (const size of [1, 6, 64, jsonl.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const record of streamJsonRecords(asChunks(jsonl, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(record);
    }
    assert.deepEqual(out, [{ id: 1 }, { id: 3 }, { id: 4 }], `chunk size ${size}`);
    assert.deepEqual(malformed, ['{"id":2'], `chunk size ${size}`);
  }

  // Windows line endings frame records the same way.
  const crlf = ['{"id":1}', '{"id":2', '{"id":3}'].join("\r\n");
  assert.deepEqual(await collect(crlf, 5), [{ id: 1 }, { id: 3 }]);
});

test("a broken first row is recovered even though nothing framed the file yet", async () => {
  // With no prior record, only unindented objects establish line framing.
  const jsonl = ['{"id":1', '{"id":2}', '{"id":3}'].join("\n");
  const out: unknown[] = [];
  const malformed: string[] = [];
  for await (const record of streamJsonRecords(asChunks(jsonl, 5), {
    onMalformed: (bad) => malformed.push(bad),
  })) {
    out.push(record);
  }
  assert.deepEqual(out, [{ id: 2 }, { id: 3 }]);
  assert.deepEqual(malformed, ['{"id":1']);
});

test("a multi-line record still frames by nesting, so a pretty-printed file is not shredded", async () => {
  // Recovery must not fire on input that is merely formatted across lines.
  const pretty = JSON.stringify({ id: 1, nested: { rows: [1, 2, 3] } }, null, 2);
  assert.deepEqual(await collect(pretty, 7), [{ id: 1, nested: { rows: [1, 2, 3] } }]);
  assert.deepEqual(await collect(JSON.stringify(TRICKY, null, 4), 11), TRICKY);
});

test("a pretty-printed record whose inner lines parse on their own stays one record", async () => {
  // Parseable inner values must not be mistaken for JSONL records.
  const record = {
    id: "c1",
    tags: ["kept-tag"],
    messages: [
      { role: "user", content: `data:image/png;base64,${"A".repeat(400)}` },
      { role: "assistant", content: "done" },
    ],
  };
  const pretty = JSON.stringify(record, null, 2);
  for (const size of [1, 16, 512, pretty.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const parsed of streamJsonRecords(asChunks(pretty, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(parsed);
    }
    assert.deepEqual(out, [record], `chunk size ${size}`);
    assert.deepEqual(malformed, [], `chunk size ${size}`);
  }

  // Damaged rows after such a record are still recovered at end of input.
  const mixed = `${pretty}\n{"id":"after"\n{"id":"tail"}`;
  const after: unknown[] = [];
  const afterBad: string[] = [];
  for await (const parsed of streamJsonRecords(asChunks(mixed, 16), {
    onMalformed: (bad) => afterBad.push(bad),
  })) {
    after.push(parsed);
  }
  assert.deepEqual(after, [record, { id: "tail" }]);
  assert.deepEqual(afterBad, ['{"id":"after"']);
});

test("a file that stops inside one record reports it once, not as a pile of fragments", async () => {
  // Indented inner lines, one of which parses alone, must not become records.
  const truncated = '{\n  "id": 1,\n  "tags": [\n    "kept-tag"\n  ]';
  const malformed: string[] = [];
  const out: unknown[] = [];
  for await (const record of streamJsonRecords(asChunks(truncated, 5), {
    onMalformed: (bad) => malformed.push(bad),
  })) {
    out.push(record);
  }
  assert.deepEqual(out, []);
  assert.deepEqual(malformed, [truncated]);
});

test("byte counts are reported for progress, not decoded character counts", async () => {
  // Four bytes each in UTF-8, two UTF-16 units each in the decoded string.
  const text = JSON.stringify([{ id: "🦥🦥🦥" }]);
  let bytes = 0;
  for await (const _ of streamJsonRecords(asChunks(text, 8), {
    onBytes: (n) => {
      bytes += n;
    },
  })) {
    // drain
  }
  assert.equal(bytes, Buffer.byteLength(text));
  assert.ok(bytes > text.length);
});

/** A File stub that emits fixed-size byte chunks. */
function chunkedFile(text: string, name: string, chunkBytes: number): File {
  const bytes = Buffer.from(text, "utf-8");
  return {
    name,
    size: bytes.byteLength,
    stream: () =>
      new ReadableStream<Uint8Array>({
        start(controller) {
          for (let at = 0; at < bytes.byteLength; at += chunkBytes) {
            controller.enqueue(new Uint8Array(bytes.subarray(at, at + chunkBytes)));
          }
          controller.close();
        },
      }),
  } as unknown as File;
}

test("multi-byte characters split across File reads decode intact, not as replacement chars", async () => {
  // Every read boundary lands mid-character: 3-byte characters, chunks of 64 bytes + 1.
  const title = "日本語のプロンプト設計".repeat(400);
  const records = [{ id: "unicode", title }, { id: "after", title: "🦥 sloth" }];
  const file = chunkedFile(JSON.stringify(records), "chat-export.json", 65);

  const out: unknown[] = [];
  let bytes = 0;
  for await (const record of streamJsonRecords(fileImportSource(file).chunks(), {
    onBytes: (n) => {
      bytes += n;
    },
  })) {
    out.push(record);
  }

  assert.deepEqual(out, records);
  assert.equal(bytes, file.size);
  assert.ok(!JSON.stringify(out).includes("�"));
});
