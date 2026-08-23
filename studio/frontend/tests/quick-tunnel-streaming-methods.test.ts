// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { openStreamResponse } from "../src/lib/open-stream-response.ts";

const GET_METHOD_RE = /method:\s*"GET"/;
const OPEN_STREAM_RE = /openStreamResponse\s*\(/;

/** Drop comments first: a stale "// method: GET" note would fail the test and a
 * commented-out call would pass it. Own-the-line only, so "https://" in a string lives. */
function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^[ \t]*\/\/[^\n]*/gm, "");
}

/** Brace-balanced, so an object literal closing at column 0 cannot end the body early.
 * Skips the parameter list first: an inline options type opens a brace before the body. */
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
  test(`${functionName} opens its event stream through openStreamResponse`, async () => {
    const source = await readFile(
      new URL(`../src/${relativePath}`, import.meta.url),
      "utf8",
    );
    const body = functionBody(stripComments(source), functionName);
    assert.match(body, OPEN_STREAM_RE);
    // A caller that pinned GET itself would silently opt out of the tunnel fix.
    assert.doesNotMatch(body, GET_METHOD_RE);
  });
}

function recorder(statuses: number[]) {
  const calls: Array<{ url: string; method?: string; init: RequestInit }> = [];
  const fetcher = (url: string, init: RequestInit) => {
    calls.push({ url, method: init.method, init });
    const status = statuses[calls.length - 1] ?? 200;
    return Promise.resolve(new Response(null, { status }));
  };
  return { calls, fetcher };
}

test("openStreamResponse asks for POST first", async () => {
  const { calls, fetcher } = recorder([200]);
  const response = await openStreamResponse(fetcher, "/api/train/progress");
  assert.equal(response.status, 200);
  assert.deepEqual(
    calls.map((c) => c.method),
    ["POST"],
  );
});

test("openStreamResponse retries as GET on 405, which is the old-backend reply", async () => {
  const { calls, fetcher } = recorder([405, 200]);
  const response = await openStreamResponse(fetcher, "/api/train/progress");
  assert.equal(response.status, 200);
  assert.deepEqual(
    calls.map((c) => c.method),
    ["POST", "GET"],
  );
  assert.equal(calls[0].url, calls[1].url);
});

for (const status of [400, 401, 403, 404, 409, 500]) {
  test(`openStreamResponse does not retry on ${status}`, async () => {
    const { calls, fetcher } = recorder([status, 200]);
    const response = await openStreamResponse(
      fetcher,
      "/api/rag/jobs/x/events",
    );
    // 404 is a real answer here (unknown job); retrying would double every miss.
    assert.equal(response.status, status);
    assert.deepEqual(
      calls.map((c) => c.method),
      ["POST"],
    );
  });
}

test("openStreamResponse forwards headers, signal and fetch options to both attempts", async () => {
  const { calls, fetcher } = recorder([405, 200]);
  const controller = new AbortController();
  const headers = new Headers({ "Last-Event-ID": "7" });
  await openStreamResponse(
    fetcher,
    "/api/export/logs/stream?since=7",
    { headers, signal: controller.signal },
    { retryNetworkErrors: false },
  );
  for (const call of calls) {
    assert.equal((call.init.headers as Headers).get("Last-Event-ID"), "7");
    assert.equal(call.init.signal, controller.signal);
  }
});

test("openStreamResponse retries at most once", async () => {
  const { calls, fetcher } = recorder([405, 405]);
  const response = await openStreamResponse(fetcher, "/api/train/progress");
  assert.equal(response.status, 405);
  assert.equal(calls.length, 2);
});
