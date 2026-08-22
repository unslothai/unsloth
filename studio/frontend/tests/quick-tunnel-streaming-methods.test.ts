// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const GET_METHOD_RE = /method:\s*"GET"/;
const POST_METHOD_RE = /method:\s*"POST"/;

function functionBody(source: string, name: string): string {
  const start = source.search(
    new RegExp(`(async )?function\\*?\\s+${name}\\b`),
  );
  if (start < 0) {
    throw new Error(`${name} is gone or was renamed`);
  }
  const end = source.indexOf("\n}\n", start);
  if (end <= start) {
    throw new Error(`could not find the end of ${name}`);
  }
  return source.slice(start, end);
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
    const body = functionBody(source, functionName);
    assert.match(body, POST_METHOD_RE);
    assert.doesNotMatch(body, GET_METHOD_RE);
  });
}
