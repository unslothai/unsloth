// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createContext, runInContext } from "node:vm";

const read = (relative: string): string =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const HTML_COMMENT = /<!--[\s\S]*?-->/g;
const POLYFILL_TAG = /<script\b[^>]*src="\/crypto-boot\.js"[^>]*>/;
const ENTRY_TAG = /<script\b[^>]*src="\/src\/main\.tsx"[^>]*>/;
const ASYNC_ATTR = /\basync\b/;
const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

// Markup, not a parsed DOM: catches the tag being dropped, moved or made async,
// not every way of making it inert. Comments would otherwise satisfy that.
const MARKUP = read("../index.html").replace(HTML_COMMENT, "");
const BOOT_SCRIPT = read("../public/crypto-boot.js");

function boot(cryptoStub: unknown): { randomUUID?: () => string } {
  const sandbox = { crypto: cryptoStub } as {
    crypto: { randomUUID?: () => string };
  };
  runInContext(BOOT_SCRIPT, createContext(sandbox));
  return sandbox.crypto;
}

test("index.html loads the polyfill before the module entry", () => {
  const polyfill = MARKUP.search(POLYFILL_TAG);
  const entry = MARKUP.search(ENTRY_TAG);
  assert.ok(polyfill !== -1, "index.html must load /crypto-boot.js");
  assert.ok(entry !== -1, "index.html must load the module entry");
  assert.ok(
    polyfill < entry,
    "the polyfill must be parsed before the module entry",
  );
});

test("neither the polyfill nor the entry opts out of ordered execution", () => {
  // The entry is deferred, so anything above it runs first and `defer` here
  // would too. `async` leaves that ordering and lets the entry win.
  assert.doesNotMatch(POLYFILL_TAG.exec(MARKUP)?.[0] ?? "", ASYNC_ATTR);
  assert.doesNotMatch(ENTRY_TAG.exec(MARKUP)?.[0] ?? "", ASYNC_ATTR);
});

// Arbitrary and non-sequential, so no substitute source reproduces it.
const STREAM = [
  0x9e, 0x37, 0x79, 0xb9, 0x7f, 0x4a, 0x7c, 0x15, 0xf3, 0x9c, 0xc0, 0x60, 0x5c,
  0xed, 0xc8, 0x34,
];

let cursor = 0;
const generate = boot({
  getRandomValues: (array: Uint8Array) => {
    for (let i = 0; i < array.length; i += 1) {
      array[i] = STREAM[cursor % STREAM.length];
      cursor += 1;
    }
    return array;
  },
}).randomUUID;

test("the drawn bytes are what the UUID is built from", () => {
  // The generator is pure given its bytes, so a golden value catches masking,
  // folding, or an ignored stream. It certifies no entropy — no unit test can —
  // and a generator rewrite is expected to change it.
  cursor = 0;
  const uuid = generate?.() ?? "";
  assert.equal(uuid, "f799fac5-2c00-4cd8-8e79-8fac53c00cd8");
  assert.match(uuid, UUID_V4);
});

test("the polyfill leaves a real randomUUID alone", () => {
  const native = () => "native";
  const patched = boot({
    randomUUID: native,
    getRandomValues: () => undefined,
  });
  assert.equal(patched.randomUUID, native);
});
