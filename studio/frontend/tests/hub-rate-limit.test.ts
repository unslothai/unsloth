// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** A Hub 429 is answered locally until its window ends, per origin and rate-limit bucket. */

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

register("./bundler-resolver.mjs", import.meta.url);

const { fetchHub, resetHubRateLimits } = await import(
  "../src/lib/hub-fetch.ts"
);

const API = "https://huggingface.co/api/models?search=qwen";
const RESOLVE =
  "https://huggingface.co/unsloth/Qwen3-4B/resolve/main/config.json";

function stubHub(
  status: (url: string) => number,
  headers: Record<string, string> = {},
) {
  const sent: string[] = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = (async (input: string | Request) => {
    const url = input instanceof Request ? input.url : String(input);
    sent.push(url);
    return new Response("{}", { status: status(url), headers });
  }) as typeof fetch;
  return { sent, restore: () => (globalThis.fetch = realFetch) };
}

function withClock(start: number) {
  const realNow = Date.now;
  let now = start;
  Date.now = () => now;
  return {
    advance: (ms: number) => (now += ms),
    restore: () => (Date.now = realNow),
  };
}

test("a 429 is not re-sent until its Retry-After passes", async () => {
  resetHubRateLimits();
  const clock = withClock(1_000_000);
  const hub = stubHub(() => 429, { "Retry-After": "30" });
  try {
    assert.equal((await fetchHub(API)).status, 429);
    const local = await fetchHub(API);
    assert.equal(local.status, 429);
    assert.equal(local.headers.get("Retry-After"), "30");
    assert.equal(local.url, API);
    assert.equal(hub.sent.length, 1);
    clock.advance(31_000);
    await fetchHub(API);
    assert.equal(hub.sent.length, 2);
  } finally {
    hub.restore();
    clock.restore();
    resetHubRateLimits();
  }
});

test("the API bucket's 429 leaves file downloads alone", async () => {
  resetHubRateLimits();
  const hub = stubHub((url) => (url.includes("/api/") ? 429 : 200));
  try {
    await fetchHub(API);
    assert.equal((await fetchHub(RESOLVE)).status, 200);
    assert.equal((await fetchHub(API)).status, 429);
    assert.deepEqual(hub.sent, [API, RESOLVE]);
  } finally {
    hub.restore();
    resetHubRateLimits();
  }
});

test("HF's RateLimit reset sets the window when Retry-After is absent", async () => {
  resetHubRateLimits();
  const clock = withClock(2_000_000);
  const hub = stubHub(() => 429, { RateLimit: '"api";r=0;t=120' });
  try {
    await fetchHub(API);
    clock.advance(119_000);
    await fetchHub(API);
    assert.equal(hub.sent.length, 1);
    clock.advance(2_000);
    await fetchHub(API);
    assert.equal(hub.sent.length, 2);
  } finally {
    hub.restore();
    clock.restore();
    resetHubRateLimits();
  }
});

test("other failures are not remembered", async () => {
  resetHubRateLimits();
  const hub = stubHub(() => 503);
  try {
    await fetchHub(API);
    await fetchHub(API);
    assert.equal(hub.sent.length, 2);
  } finally {
    hub.restore();
    resetHubRateLimits();
  }
});
