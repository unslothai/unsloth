// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { test } from "node:test";
import type { PageResponse } from "../src/features/browser/api.ts";
import type { PageRequest } from "../src/features/browser/frame-message.ts";
import { pageRequests } from "../src/features/browser/page-requests.ts";

const request: PageRequest = { url: "https://site.test/api", method: "GET", headers: [], body: null, redirect: "follow" };
const answer: PageResponse = { status: 200, statusText: "OK", url: "https://site.test/api", headers: [], body: new ArrayBuffer(0) };
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

function setup(page: string | null = "https://site.test/models") {
  const calls: Array<{ page: string; signal: AbortSignal; finish: (ok: boolean) => void }> = [];
  const replies: Array<[string, PageResponse | null]> = [];
  const requests = pageRequests(
    (page, _request, signal) =>
      new Promise((resolve, reject) => calls.push({ page, signal, finish: (ok) => (ok ? resolve(answer) : reject(new Error("failed"))) })),
    (id, reply) => replies.push([id, reply]),
    () => page,
  );
  return { requests, calls, replies };
}

test("requests go out for the panel's page address and are answered once each", async () => {
  const t = setup();
  t.requests.request("a", request);
  t.requests.request("b", request);
  assert.equal(t.calls[0].page, "https://site.test/models");
  t.calls[0].finish(true);
  t.calls[1].finish(false);
  await tick();
  assert.deepEqual(t.replies, [["a", answer], ["b", null]]);
});

test("three run at once across every page and the rest wait their turn", async () => {
  const t = setup();
  const other = setup();
  for (let i = 0; i < 2; i++) t.requests.request(`r${i}`, request);
  for (let i = 0; i < 3; i++) other.requests.request(`o${i}`, request);
  assert.equal(t.calls.length + other.calls.length, 3);
  assert.equal(other.calls.length, 1);
  t.calls[0].finish(true);
  await tick();
  assert.equal(other.calls.length, 2);
  t.requests.stop();
  other.requests.stop();
  for (const call of [...t.calls, ...other.calls]) call.finish(false);
  await tick();
});

test("one the panel won't make, or with no page address to make it for, fails at once", () => {
  const t = setup();
  t.requests.request("bad", null);
  assert.deepEqual(t.replies, [["bad", null]]);
  const local = setup(null);
  local.requests.request("file", request);
  assert.equal(local.calls.length, 0);
  assert.deepEqual(local.replies, [["file", null]]);
});

test("an abort cancels the backend call and needs no answer", async () => {
  const t = setup();
  t.requests.request("a", request);
  t.requests.abort("a");
  assert.equal(t.calls[0].signal.aborted, true);
  t.calls[0].finish(false);
  await tick();
  assert.deepEqual(t.replies, []);
});

test("a page that goes ends everything it asked for, queued or running", async () => {
  const t = setup();
  for (let i = 0; i < 8; i++) t.requests.request(`r${i}`, request);
  t.requests.stop();
  assert.ok(t.calls.every((call) => call.signal.aborted));
  for (const call of t.calls) call.finish(false);
  await tick();
  assert.equal(t.calls.length, 3);
  assert.deepEqual(t.replies, []);
});
