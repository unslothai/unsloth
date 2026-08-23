// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const GET_METHOD_RE = /method:\s*"GET"/;
const POST_METHOD_RE = /method:\s*"POST"/;
const AUTH_FETCH_RE = /authFetch\s*\(/;

/** Comments are dropped first: a stale "// method: GET" note next to the call would
 * otherwise fail the test, and a commented-out POST would pass it. Line comments only
 * when they own the line, so a "https://" inside a string survives. */
function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[ \t]*\/\/[^\n]*/gm, "");
}

/** Brace-balanced, so a nested object literal closing at column 0 cannot end the body
 * early and hide the request that follows it. The parameter list is skipped first,
 * because an inline options type opens a brace before the body does. */
function functionBody(source: string, name: string): string {
  const start = source.search(
    new RegExp(`(async )?function\\*?\\s+${name}\\b`),
  );
  if (start < 0) {
    throw new Error(`${name} is gone or was renamed`);
  }
  let parens = 0;
  let i = source.indexOf("(", start);
  for (; i < source.length; i++) {
    if (source[i] === "(") parens++;
    else if (source[i] === ")" && --parens === 0) break;
  }
  let braces = 0;
  for (i = source.indexOf("{", i); i < source.length; i++) {
    if (source[i] === "{") braces++;
    else if (source[i] === "}" && --braces === 0)
      return source.slice(start, i + 1);
  }
  throw new Error(`could not find the end of ${name}`);
}

for (const [relativePath, functionName] of [
  ["features/training/api/train-api.ts", "streamTrainingProgress"],
  ["features/export/api/export-api.ts", "streamExportLogs"],
  ["features/rag/api/rag-api.ts", "openEventStream"],
  ["features/recipe-studio/api/index.ts", "streamRecipeJobEvents"],
] as const) {
  test(`${functionName} opens its event stream over POST`, async () => {
    const source = await readFile(
      new URL(`../src/${relativePath}`, import.meta.url),
      "utf8",
    );
    const body = functionBody(stripComments(source), functionName);
    // The body must still be the one that opens the stream, or the assertions below
    // would pass on any function that happens to carry the right spelling.
    assert.match(body, AUTH_FETCH_RE);
    assert.match(body, POST_METHOD_RE);
    assert.doesNotMatch(body, GET_METHOD_RE);
  });
}
