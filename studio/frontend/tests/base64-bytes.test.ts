// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { randomBytes } from "node:crypto";
import { test } from "node:test";

import { decodeBase64Bytes } from "../src/lib/base64-bytes.ts";

// The decoder both native readers used before.
function atobLoop(base64: string): Uint8Array {
  const binary = globalThis.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return bytes;
}

function outcome(decode: (value: string) => Uint8Array, value: string): string {
  try {
    return `ok:${Buffer.from(decode(value)).toString("hex")}`;
  } catch (error) {
    return `err:${(error as Error).name}`;
  }
}

const edgeInputs = [
  "",
  "=",
  "==",
  "A",
  "AA",
  "AAA",
  "AAAA",
  "AA=",
  "AA==",
  "AAA=",
  "AB==",
  "AF==",
  "AAA==",
  "A===",
  "AAAA=",
  "AAAA==",
  " AAAA ",
  "A A\nA\tA",
  "AA==AA==",
  "QQ",
  "QQ=",
  "QQ==",
  "QUI",
  "QUI=",
  "QUJD",
  "-_-_",
  "AAA.",
  "%%%%",
  "ÿÿ",
];

function randomInputs(): string[] {
  const alphabet = "AB/+=-_ \n\tz09.%Q";
  const inputs: string[] = [];
  for (let index = 0; index < 5000; index += 1) {
    let value = "";
    const length = Math.floor(Math.random() * 9);
    for (let j = 0; j < length; j += 1) {
      value += alphabet[Math.floor(Math.random() * alphabet.length)];
    }
    inputs.push(value);
  }
  for (const length of [0, 1, 2, 3, 4, 5, 255, 256, 257, 65_537]) {
    inputs.push(randomBytes(length).toString("base64"));
  }
  return inputs;
}

function withFromBase64<T>(impl: unknown, run: () => T): T {
  const target = Uint8Array as unknown as { fromBase64?: unknown };
  const had = Object.prototype.hasOwnProperty.call(target, "fromBase64");
  const previous = target.fromBase64;
  target.fromBase64 = impl;
  try {
    return run();
  } finally {
    if (had) target.fromBase64 = previous;
    else delete target.fromBase64;
  }
}

test("decodeBase64Bytes matches the atob loop without fromBase64", () => {
  withFromBase64(undefined, () => {
    for (const value of [...edgeInputs, ...randomInputs()]) {
      assert.equal(
        outcome(decodeBase64Bytes, value),
        outcome(atobLoop, value),
        JSON.stringify(value),
      );
    }
  });
});

test("decodeBase64Bytes round-trips padded payloads byte for byte", () => {
  for (const length of [0, 1, 2, 3, 4, 1023, 1024, 1025, 1 << 20]) {
    const source = randomBytes(length);
    const decoded = decodeBase64Bytes(source.toString("base64"));
    assert.ok(decoded instanceof Uint8Array);
    assert.equal(decoded.byteLength, length);
    assert.ok(Buffer.from(decoded).equals(source));
  }
});

test("decodeBase64Bytes uses fromBase64 and keeps atob's error when it throws", () => {
  const calls: string[] = [];
  withFromBase64(
    function (this: unknown, value: string) {
      assert.equal(this, Uint8Array);
      calls.push(value);
      return new Uint8Array(Buffer.from(value, "base64"));
    },
    () => {
      assert.deepEqual(Array.from(decodeBase64Bytes("QUJD")), [65, 66, 67]);
    },
  );
  assert.deepEqual(calls, ["QUJD"]);

  withFromBase64(
    () => {
      throw new SyntaxError("bad base64");
    },
    () => {
      assert.equal(outcome(decodeBase64Bytes, "A"), outcome(atobLoop, "A"));
      assert.equal(
        outcome(decodeBase64Bytes, "A"),
        "err:InvalidCharacterError",
      );
    },
  );
});
