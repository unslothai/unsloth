// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeTextChunks,
  fileImportSource,
  readAllText,
  streamJsonRecords,
  type TextChunk,
} from "../src/features/chat/utils/json-record-stream.ts";

async function* asChunks(text: string, size: number): AsyncGenerator<TextChunk> {
  for (let index = 0; index < text.length; index += size) {
    const slice = text.slice(index, index + size);
    yield { text: slice, bytes: Buffer.byteLength(slice) };
  }
}

async function drain(chunks: AsyncIterable<unknown>): Promise<void> {
  for await (const chunk of chunks) void chunk;
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

  const jsonl = ['{"id":1}', "{bad}", '{"id":3}'].join("\n");
  assert.deepEqual(await collect(jsonl, 3), [{ id: 1 }, { id: 3 }]);
});

test("a row that loses its closing brace does not swallow the rows after it", async () => {
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

  const crlf = ['{"id":1}', '{"id":2', '{"id":3}'].join("\r\n");
  assert.deepEqual(await collect(crlf, 5), [{ id: 1 }, { id: 3 }]);
});

test("a broken first row is recovered even though nothing framed the file yet", async () => {
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
  const pretty = JSON.stringify({ id: 1, nested: { rows: [1, 2, 3] } }, null, 2);
  assert.deepEqual(await collect(pretty, 7), [{ id: 1, nested: { rows: [1, 2, 3] } }]);
  assert.deepEqual(await collect(JSON.stringify(TRICKY, null, 4), 11), TRICKY);
});

test("a pretty-printed record whose inner lines parse on their own stays one record", async () => {
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
  await drain(
    streamJsonRecords(asChunks(text, 8), {
      onBytes: (n) => {
        bytes += n;
      },
    }),
  );
  assert.equal(bytes, Buffer.byteLength(text));
  assert.ok(bytes > text.length);
});

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

test("an array that ends between records fails instead of reporting a complete import", async () => {
  for (const truncated of ['[{"id":1}', '[{"id":1},', '[{"id":1},\n', "[", '[{"id":1},{"id":2}']) {
    await assert.rejects(collect(truncated, 3), SyntaxError, truncated);
  }

  for (const size of [1, 4, 64]) {
    assert.deepEqual(await collect('[{"id":1},{"id":2}]', size), [{ id: 1 }, { id: 2 }]);
    assert.deepEqual(await collect('[{"id":1}]\n', size), [{ id: 1 }]);
  }

  assert.deepEqual(await collect('{"id":1}\n{"id":2}', 5), [{ id: 1 }, { id: 2 }]);
  assert.deepEqual(await collect('{"id":1}\n{"id":2}\n', 5), [{ id: 1 }, { id: 2 }]);
});

test("invalid UTF-8 is rejected on the desktop path and tolerated on the browser path", async () => {
  async function* bytes(): AsyncGenerator<Uint8Array> {
    yield new Uint8Array([0x7b, 0x22, 0x61, 0x22, 0x3a, 0x22]); // {"a":"
    yield new Uint8Array([0xff]); // not a UTF-8 sequence
    yield new Uint8Array([0x22, 0x7d]); // "}
  }

  const lenient: string[] = [];
  for await (const chunk of decodeTextChunks(bytes())) lenient.push(chunk.text);
  assert.equal(lenient.join(""), '{"a":"�"}');

  await assert.rejects(drain(decodeTextChunks(bytes(), true)), TypeError);

  const cutShort = (async function* () {
    yield new Uint8Array([0xe6, 0x97]); // first two bytes of a 3-byte character
  })();
  await assert.rejects(drain(decodeTextChunks(cutShort, true)), TypeError);
});

test("the whole-file read is bounded by bytes, not by decoded string length", async () => {
  // Three bytes per character, one UTF-16 unit each: a length check would pass 3x the limit.
  const text = "日".repeat(64);
  const source = {
    name: "chats.csv",
    async *chunks() {
      for (const chunk of [text, text]) {
        yield { text: chunk, bytes: Buffer.byteLength(chunk) };
      }
    },
  };

  assert.equal((await readAllText(source, 512, "CSV")).length, 128);
  await assert.rejects(readAllText(source, 300, "CSV"), /chats\.csv is too large to import as CSV/);
});

test("a pretty-printed record following a single-line one survives every chunk boundary", async () => {
  const record = { id: 2, tags: ["a", "b"], nested: { rows: [1, 2] } };
  const mixed = `{"id":1}\n${JSON.stringify(record, null, 2)}\n`;
  for (const size of [1, 5, 12, 64, mixed.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const parsed of streamJsonRecords(asChunks(mixed, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(parsed);
    }
    assert.deepEqual(out, [{ id: 1 }, record], `chunk size ${size}`);
    assert.deepEqual(malformed, [], `chunk size ${size}`);
  }
});

test("a broken first row is recovered while the file streams, not held until the end", async () => {
  // With nothing emitted yet there is no proven framing; waiting for EOF would buffer the export.
  const rows = `{"id":1\n${Array.from({ length: 60 }, (_, i) => `{"id":${i + 2}}`).join("\n")}\n`;
  let pulled = 0;
  async function* counted(): AsyncGenerator<TextChunk> {
    for (let at = 0; at < rows.length; at += 16) {
      pulled++;
      const slice = rows.slice(at, at + 16);
      yield { text: slice, bytes: Buffer.byteLength(slice) };
    }
  }

  const out: unknown[] = [];
  const malformed: string[] = [];
  let pulledAtFirstRecord = Number.POSITIVE_INFINITY;
  for await (const record of streamJsonRecords(counted(), {
    onMalformed: (bad) => malformed.push(bad),
  })) {
    if (out.length === 0) pulledAtFirstRecord = pulled;
    out.push(record);
  }

  assert.equal(out.length, 60);
  assert.deepEqual(malformed, ['{"id":1']);
  assert.ok(
    pulledAtFirstRecord < pulled,
    `first record arrived only after ${pulledAtFirstRecord} of ${pulled} chunks`,
  );
});

test("a nested value at column 0 does not end the record that contains it", async () => {
  // After `{"nested":` only a value can follow, so the brace below it is nesting.
  const nested = '{"nested":\n{"id":2}\n}';
  for (const size of [1, 4, 9, 17, nested.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const record of streamJsonRecords(asChunks(`{"id":1}\n${nested}\n`, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(record);
    }
    assert.deepEqual(out, [{ id: 1 }, { nested: { id: 2 } }], `chunk size ${size}`);
    assert.deepEqual(malformed, [], `chunk size ${size}`);
  }
});

test("a pretty-printed conversation after a damaged row is framed, not shredded into lines", async () => {
  const conversation = {
    id: "c2",
    messages: [
      { role: "user", content: "hi" },
      { role: "assistant", content: "hello" },
    ],
  };
  const text = `{"id":1\n${JSON.stringify(conversation, null, 2)}\n{"id":3}\n`;
  for (const size of [1, 7, 40, 200, text.length]) {
    const out: unknown[] = [];
    const malformed: string[] = [];
    for await (const record of streamJsonRecords(asChunks(text, size), {
      onMalformed: (bad) => malformed.push(bad),
    })) {
      out.push(record);
    }
    assert.deepEqual(out, [conversation, { id: 3 }], `chunk size ${size}`);
    assert.deepEqual(malformed, ['{"id":1'], `chunk size ${size}`);
  }
});

test("an array cut mid-record explains itself instead of quoting the JSON engine", async () => {
  const truncated = '[{"id":1},{"id":2},{"title":"half a chat';
  for (const size of [1, 5, 64, 4096]) {
    const seen: unknown[] = [];
    let message = "";
    try {
      for await (const record of streamJsonRecords(asChunks(truncated, size))) seen.push(record);
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    assert.deepEqual(seen, [{ id: 1 }, { id: 2 }], `records lost at chunk ${size}`);
    assert.match(message, /ends in the middle of a record/);
    assert.doesNotMatch(message, /Unterminated string|position \d+/);
  }
});

test("an array cut between records still names the missing bracket", async () => {
  for (const truncated of ['[{"id":1},{"id":2},', '[{"id":1},{"id":2}']) {
    let message = "";
    try {
      await drain(streamJsonRecords(asChunks(truncated, 3)));
    } catch (error) {
      message = error instanceof Error ? error.message : String(error);
    }
    assert.match(message, /ends before its closing bracket/);
  }
});

test("a complete array is unaffected by the truncation handling", async () => {
  const records = await collect('[{"id":1},{"id":2}]', 2);
  assert.deepEqual(records, [{ id: 1 }, { id: 2 }]);
});

test("a record after the array's closing bracket is refused, not imported", async () => {
  for (const size of [1, 6, 64]) {
    await assert.rejects(collect('[{"id":1}]\n{"id":2}\n', size), SyntaxError, `chunk size ${size}`);
    await assert.rejects(collect('[{"id":1}] garbage', size), SyntaxError, `chunk size ${size}`);
    assert.deepEqual(await collect('[{"id":1}]  \n\t\n', size), [{ id: 1 }], `chunk size ${size}`);
  }
});
