// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// hf-readme.ts reaches import.meta.env through @/config/env, so it cannot be
// imported outside vite. These read the source instead, the same way the panel
// and dead-iterator suites do.

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

test("the card is fetched through the backend whenever the browser cannot", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  // Three ways to know the raw URL will not work: a mirror, a listing already
  // served by the proxy, and a detail-pane lookup that fell back on its own.
  // The last one is the deep link / pinned publisher case, which never runs a
  // listing and so has no other proof.
  const at = src.indexOf("export function readmeViaBackend");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\n}", at));
  for (const term of ["hubProxyFirst()", "isHubProxyServing()", "isDirectHubBlocked()"]) {
    assert.ok(body.includes(term), `readmeViaBackend must consider ${term}`);
  }
  assert.match(src, /if \(readmeViaBackend\(\)\) return fetchReadmeViaBackend\(/);
});

test("relative assets resolve against the configured endpoint", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  const at = src.indexOf("export function readmeBaseUrl");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\n}", at));
  assert.ok(
    body.includes("hubEndpointOrigin()"),
    "a hardcoded Hub sends a mirror user's repo path to the public one",
  );
  // No literal scheme at all, which also rules out any other hardcoded host.
  assert.ok(!body.includes("://"), "the base must come from the endpoint");
});

test("the card's load effect is not gated off when the backend can serve it", async () => {
  const src = await read("../src/features/hub/catalog/model-readme.tsx");
  // online is !isHuggingFaceOffline(), which the proxy sets true precisely when
  // the backend route is the one that works. Gating on it alone showed the
  // unavailable card over a card we could fetch.
  assert.match(src, /if \(!online && !readmeViaBackend\(\)\) return;/);
});

test("the cached result is scoped to the route that produced it", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  const at = src.indexOf("export function fetchReadme(");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("cache.set(key, entry)", at));
  // A direct attempt that failed first caches a 30s rejection. Without the
  // route in the key, every backend-capable retry gets that rejection back.
  assert.ok(body.includes("readmeViaBackend()"));
  assert.match(body, /const key = `\$\{kind\}::\$\{repoId\}::\$\{fingerprintToken\(token\)\}::\$\{via\}`;/);
});
