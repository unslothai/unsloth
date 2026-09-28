// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { register } from "node:module";
import test from "node:test";

// A Hub that answers 401 is reachable and refusing the saved token (an expired
// or revoked OAuth token makes every read 401, public listings included). The
// panel used to call that "Couldn't reach Hugging Face", which sends the user to
// check a connection that works.
register("./store-stub-resolver.mjs", import.meta.url);
const {
  classifyFetchFailure,
  fetchWithTimeout,
  getHubPhase,
  getLastHubFailure,
  hubAuthFailure,
  markRemoteNetworkOnline,
} = await import("../src/features/hub/lib/network.ts");

import { readText } from "./helpers/kit.ts";

const HF = "https://huggingface.co";

function installWindow() {
  const listeners = new Map<string, Set<() => void>>();
  (globalThis as Record<string, unknown>).window = {
    addEventListener(type: string, fn: () => void) {
      if (!listeners.has(type)) listeners.set(type, new Set());
      listeners.get(type)?.add(fn);
    },
    removeEventListener(type: string, fn: () => void) {
      listeners.get(type)?.delete(fn);
    },
    dispatchEvent(event: { type: string }) {
      for (const fn of listeners.get(event.type) ?? []) fn();
      return true;
    },
    location: { href: "http://127.0.0.1:8888/hub" },
  };
}

test("a 401 is a refused token; 403, 404, 429 and 5xx are not", () => {
  const rejected = hubAuthFailure({ status: 401 }, HF);
  assert.equal(rejected?.kind, "auth-rejected");
  assert.equal(rejected?.status, 401);
  assert.equal(rejected?.origin, HF);
  assert.match(rejected?.message ?? "", /refused the saved Hugging Face token/);
  // A gated or private repo answers 403/404 to a token that is perfectly valid.
  for (const status of [403, 404, 429, 500, 502, 503]) {
    assert.equal(hubAuthFailure({ status }, HF), null, `status ${status}`);
  }
});

test("the SDK's error text is enough when the status was not kept", () => {
  // What @huggingface/hub's createApiError leaves in `message` for a 401: the
  // Hub's JSON error when there is one, else "Api error with status 401".
  for (const message of [
    "OAuth token verification failed: Invalid Compact JWS",
    "Invalid credentials in Authorization header",
    "Api error with status 401",
    "Unauthorized",
  ]) {
    assert.equal(hubAuthFailure({ message }, HF)?.kind, "auth-rejected", message);
  }
  for (const message of [
    "Api error with status 403",
    "Api error with status 404",
    "Api error with status 429",
    "You have exceeded our hourly quotas for action: api. Too many requests",
    "Api error with status 503",
    "Failed to fetch",
    "Unexpected token < in JSON at position 0",
    "",
  ]) {
    assert.equal(hubAuthFailure({ message }, HF), null, message);
  }
  assert.equal(hubAuthFailure({ message: null }, HF), null);
});

test("an explicit status wins over the text", () => {
  // "Unauthorized" in a 403 body is still a 403: a gated repo, not a dead token.
  assert.equal(hubAuthFailure({ status: 403, message: "Unauthorized" }, HF), null);
});

test("network failures keep their own kinds", () => {
  assert.equal(
    classifyFetchFailure(new TypeError("Failed to fetch"), HF).kind,
    "network-opaque",
  );
  assert.equal(classifyFetchFailure(new Error("x"), HF, { timedOut: true }).kind, "timeout");
});

test("a 401 answer does not back the Hub off or record an outage", async () => {
  installWindow();
  markRemoteNetworkOnline();
  const original = globalThis.fetch;
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ error: "OAuth token verification failed" }), {
      status: 401,
      headers: { "Content-Type": "application/json" },
    })) as typeof fetch;
  try {
    const response = await fetchWithTimeout(`${HF}/api/models?search=qwen`, {}, 1_000);
    assert.equal(response.status, 401);
    // The Hub is reachable: on-device paths, the offline fallbacks and the
    // backoff all key off this phase, and a refused token must not trip them.
    assert.equal(getHubPhase(HF), "available");
    assert.equal(getLastHubFailure(HF), null);
  } finally {
    globalThis.fetch = original;
  }
});

test("the discovery feed names a refused token instead of an unreachable Hub", async () => {
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  assert.match(
    search,
    /isDiscoverTab\s*\?\s*\(failure \?\? hubAuthFailure\(\{ message: rawSearchError \}\)\)\s*:\s*null,/,
  );
  const states = await readText("../src/features/hub/catalog/catalog-states.tsx");
  const describe = states.slice(
    states.indexOf("function describeFailure"),
    states.indexOf("\nfunction UseModelScopeButton"),
  );
  const branch = describe.slice(describe.indexOf('case "auth-rejected":'));
  assert.match(branch, /rejected your token/);
  assert.match(branch, /offlineLike: false/);
  assert.match(branch, /tokenRejected: true/);
  // Both panels recover the refusal from an SDK error the network layer never saw.
  assert.equal(states.match(/hubAuthFailure\(\{ message \}\)/g)?.length, 2);
  // The fix is the token: offer it, and do not suggest switching hubs over it.
  assert.equal(states.match(/\{tokenRejected \? <UpdateTokenButton \/> : null\}/g)?.length, 2);
  assert.equal(states.match(/!offlineLike && !tokenRejected \? <UseModelScopeButton \/>/g)?.length, 2);
  assert.match(states, /openSettings\("general"\)/);
});
