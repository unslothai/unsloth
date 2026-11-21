// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import ts from "@typescript/typescript6";

const source = readFileSync(
  new URL("../src/features/hub/hooks/use-hub-model-search.ts", import.meta.url),
  "utf8",
);
const start = source.indexOf("const UNSLOTH_PREFETCH = ");
const fnStart = source.indexOf("function unslothPrefetchLimit(");
assert.notEqual(start, -1);
assert.notEqual(fnStart, -1);
const end = source.indexOf("\n}\n", fnStart) + 2;
const unslothPrefetchLimit = new Function(
  ts.transpileModule(source.slice(start, end), { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText +
    "; return unslothPrefetchLimit;",
)() as (query: string, sortBy: string) => number;

// author=unsloth&search=qwen3.8-27b&sort=createdAt, as the Hub returned it.
const NEWEST_FIRST = [
  "unsloth/Qwen3.8-27B-FP8",
  "unsloth/Qwen3.8-27B-unsloth-bnb-4bit",
  "unsloth/Qwen3.8-27B-NVFP4",
  "unsloth/Qwen3.8-27B-GGUF",
  "unsloth/Qwen3.8-27B",
];

test("a newest-first search floats every Unsloth upload of the typed model", () => {
  for (const sort of ["fresh", "createdAt", "lastModified"]) {
    const floated = NEWEST_FIRST.slice(0, unslothPrefetchLimit("qwen3.8-27b", sort));
    assert.ok(floated.includes("unsloth/Qwen3.8-27B-GGUF"), sort);
  }
});

test("a popularity sort keeps the short Unsloth head", () => {
  for (const sort of ["downloads", "likes", "trendingScore"]) {
    assert.equal(unslothPrefetchLimit("qwen3.8-27b", sort), 3, sort);
  }
});

test("browsing with no query reads the full Unsloth pool", () => {
  assert.equal(unslothPrefetchLimit("", "downloads"), unslothPrefetchLimit("", "fresh"));
  assert.ok(unslothPrefetchLimit("  ", "downloads") > 3);
});
