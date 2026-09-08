// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync, readdirSync } from "node:fs";
import test from "node:test";
import { fileURLToPath } from "node:url";
import { createContext, runInContext } from "node:vm";

const read = (relative: string): string =>
  readFileSync(fileURLToPath(new URL(relative, import.meta.url)), "utf8");

const HTML_COMMENT = /<!--[\s\S]*?-->/g;
const POLYFILL_TAG = /<script\b[^>]*src="\/crypto-boot\.js"[^>]*>/;
const ENTRY_TAG = /<script\b[^>]*\btype="module"[^>]*>/;
const APP_ENTRY = /<script\b[^>]*src="\/src\/main\.tsx"[^>]*>/;
const ASYNC_ATTR = /\basync\b/;
const UUID_V4 =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

// Check every HTML entry as raw markup so comments cannot satisfy the patterns.
const PAGES = readdirSync(new URL("../", import.meta.url))
  .filter((name) => name.endsWith(".html"))
  .map((name) => [name, read(`../${name}`).replace(HTML_COMMENT, "")] as const);
const BOOT_SCRIPT = read("../public/crypto-boot.js");

function boot(cryptoStub: unknown): { randomUUID?: () => string } {
  const sandbox = { crypto: cryptoStub } as {
    crypto: { randomUUID?: () => string };
  };
  runInContext(BOOT_SCRIPT, createContext(sandbox));
  return sandbox.crypto;
}

test("every page loads the polyfill before its module entry", () => {
  assert.ok(PAGES.length > 0, "no HTML entries found");
  const index = PAGES.find(([name]) => name === "index.html")?.[1];
  assert.match(index ?? "", APP_ENTRY, "index.html must keep the app entry");
  for (const [name, markup] of PAGES) {
    const polyfill = markup.search(POLYFILL_TAG);
    const entry = markup.search(ENTRY_TAG);
    assert.ok(polyfill !== -1, `${name} must load /crypto-boot.js`);
    assert.ok(entry !== -1, `${name} must load a module entry`);
    assert.ok(polyfill < entry, `${name} loads the polyfill too late`);
  }
});

test("no page lets the polyfill or its entry opt out of ordered execution", () => {
  for (const [name, markup] of PAGES) {
    assert.doesNotMatch(POLYFILL_TAG.exec(markup)?.[0] ?? "", ASYNC_ATTR, name);
    assert.doesNotMatch(ENTRY_TAG.exec(markup)?.[0] ?? "", ASYNC_ATTR, name);
  }
});

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
  // A fixed byte stream catches changes to UUID masking and folding.
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
