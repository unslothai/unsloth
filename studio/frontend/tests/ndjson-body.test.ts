// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { ndjsonBody } from "../src/features/chat/utils/ndjson.ts";

test("terminates a single record with a newline", () => {
  assert.equal(ndjsonBody(['{"messages":[]}']), '{"messages":[]}\n');
});

test("separates and terminates every record", () => {
  assert.equal(ndjsonBody(["{\"a\":1}", "{\"b\":2}"]), "{\"a\":1}\n{\"b\":2}\n");
});

test("concatenated bodies stay parseable line by line", () => {
  const combined = ndjsonBody(['{"a":1}']) + ndjsonBody(['{"b":2}']);
  const parsed = combined
    .split("\n")
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as Record<string, number>);
  assert.deepEqual(parsed, [{ a: 1 }, { b: 2 }]);
});

test("returns an empty body when there are no records", () => {
  assert.equal(ndjsonBody([]), "");
});
