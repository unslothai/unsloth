// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  decodeTextChunks,
  readAllText,
  streamJsonRecords,
  type ImportSource,
} from "../src/features/chat/utils/json-record-stream.ts";

function utf16le(text: string): Uint8Array {
  const bytes = new Uint8Array(text.length * 2);
  for (let i = 0; i < text.length; i++) {
    bytes[i * 2] = text.charCodeAt(i) & 0xff;
    bytes[i * 2 + 1] = text.charCodeAt(i) >> 8;
  }
  return bytes;
}

function utf16be(text: string): Uint8Array {
  const bytes = new Uint8Array(text.length * 2);
  for (let i = 0; i < text.length; i++) {
    bytes[i * 2] = text.charCodeAt(i) >> 8;
    bytes[i * 2 + 1] = text.charCodeAt(i) & 0xff;
  }
  return bytes;
}

function utf16leBom(text: string): Uint8Array {
  const body = utf16le(text);
  const out = new Uint8Array(body.length + 2);
  out.set([0xff, 0xfe], 0);
  out.set(body, 2);
  return out;
}

async function* asBytes(bytes: Uint8Array, size: number): AsyncGenerator<Uint8Array> {
  for (let index = 0; index < bytes.length; index += size) {
    yield bytes.subarray(index, index + size);
  }
}

async function collectText(bytes: Uint8Array, size: number): Promise<string> {
  let text = "";
  for await (const chunk of decodeTextChunks(asBytes(bytes, size))) text += chunk.text;
  return text;
}

const EXPORT = '[{"id":1,"title":"first"},{"id":2,"title":"second"}]';

test("a UTF-16LE export with a BOM decodes instead of importing NUL-laden noise", async () => {
  for (const size of [1, 3, 64, 4096]) {
    assert.equal(await collectText(utf16leBom(EXPORT), size), EXPORT, `chunk size ${size}`);
  }
});

test("a UTF-16 export without a BOM is detected from its NUL-heavy pairs", async () => {
  for (const bytes of [utf16le(EXPORT), utf16be(EXPORT)]) {
    const records: unknown[] = [];
    for await (const record of streamJsonRecords(decodeTextChunks(asBytes(bytes, 7)))) {
      records.push(record);
    }
    assert.equal(records.length, 2);
  }
});

test("readAllText reports a UTF-16 import instead of a parse failure", async () => {
  const bytes = utf16leBom(EXPORT);
  const source: ImportSource = {
    name: "chats.json",
    async *chunks() {
      yield* decodeTextChunks(asBytes(bytes, 5));
    },
  };
  assert.equal(await readAllText(source, 1 << 20, "JSON"), EXPORT);
});

test("UTF-8 text with multi-byte characters still decodes as UTF-8", async () => {
  const text = '[{"t":"日本語 ✓ é"}]';
  const bytes = new TextEncoder().encode(text);
  assert.equal(await collectText(bytes, 4), text);
  assert.equal(await collectText(bytes, 4096), text);
});

test("the desktop fatal path accepts UTF-16 instead of throwing on NULs", async () => {
  const bytes = utf16leBom(EXPORT);
  let text = "";
  for await (const chunk of decodeTextChunks(asBytes(bytes, 11), true)) text += chunk.text;
  assert.equal(text, EXPORT);
});

test("BOM-less UTF-16 is detected even when non-ASCII text follows the JSON opener", async () => {
  const text = '\n["日本語日本語日本語"]';
  for (const bytes of [utf16le(text), utf16be(text)]) {
    for (const size of [1, 8, 4096]) {
      assert.equal(await collectText(bytes, size), text);
    }
  }
});

test("stopping text iteration closes the byte iterator", async () => {
  let closed = false;
  async function* input(): AsyncGenerator<Uint8Array> {
    try {
      yield utf16le(EXPORT);
      assert.fail("the consumer should not request another chunk");
    } finally {
      closed = true;
    }
  }
  for await (const _ of decodeTextChunks(input())) break;
  assert.equal(closed, true);
});

test("a fatal decode error closes the byte iterator", async () => {
  let closed = false;
  async function* input(): AsyncGenerator<Uint8Array> {
    try {
      yield Uint8Array.from([0x7b, 0xff, 0x7d]);
    } finally {
      closed = true;
    }
  }
  await assert.rejects(async () => {
    for await (const _ of decodeTextChunks(input(), true)) {
      assert.fail("malformed UTF-8 should not emit text");
    }
  }, TypeError);
  assert.equal(closed, true);
});

test("UTF-16BE with a split BOM reports the original byte count", async () => {
  const bytes = Uint8Array.from([0xfe, 0xff, ...utf16be(EXPORT)]);
  const output = [];
  for await (const chunk of decodeTextChunks(asBytes(bytes, 1), true)) output.push(chunk);
  assert.equal(output.map((chunk) => chunk.text).join(""), EXPORT);
  assert.equal(output.reduce((total, chunk) => total + chunk.bytes, 0), bytes.length);
});
