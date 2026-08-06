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
    body.includes("hubEndpointBase()"),
    "a hardcoded Hub sends a mirror user's repo path to the public one",
  );
  // No literal scheme at all, which also rules out any other hardcoded host.
  assert.ok(!body.includes("://"), "the base must come from the endpoint");
});

test("the card's load effect is not gated off when the backend can serve it", async () => {
  const src = await read("../src/features/hub/catalog/model-readme.tsx");
  // online is !isDirectHubOffline(), which the proxy sets true precisely when
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

test("the component's own cache is scoped to the route as well", async () => {
  const src = await read("../src/features/hub/catalog/model-readme.tsx");
  // model-readme keeps a second cache holding the in-flight promise. Without
  // the route here, a direct attempt still running when the browser turns out
  // to be blocked is handed back and the backend route is never reached, so
  // scoping only fetchReadme's key fixes nothing in that ordering.
  const at = src.indexOf("const stateKey = useMemo");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("  );", at));
  assert.ok(body.includes("::${via}`"), "the route belongs in the key");
  assert.ok(body.includes("via]"), "and in the memo's dependencies");
  assert.match(src, /const via = online && !readmeViaBackend\(\)/);
});

test("a subpath mirror keeps its prefix in the asset base", async () => {
  const src = await read("../src/features/hub/lib/hub-endpoint.ts");
  const at = src.indexOf("export function hubEndpointBase");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\n}", at));
  // The backend builds its URLs from the whole of HF_ENDPOINT, so dropping the
  // path here resolved every relative asset one directory too high on a mirror
  // mounted at, say, https://mirror.example/hf.
  assert.ok(!body.includes(".origin"), "the path is part of the endpoint");
  assert.ok(body.includes(".href"));
});

test("a direct card failure falls back to the backend on its own", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  const at = src.indexOf("async function fetchReadmeOnce");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\nexport function fetchReadme(", at));
  // Neither proxy flag is set by a /raw block, so without this the card is
  // permanently unavailable whenever only that path is filtered.
  const transientAt = body.indexOf("if (transient)");
  assert.notEqual(transientAt, -1);
  assert.ok(
    body.slice(transientAt).includes("fetchReadmeViaBackend("),
    "the transient path must try the same route the listing falls back to",
  );
});

test("a served card records that the backend is the working route", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  // lastIndexOf: fetchReadmeViaBackend has its own transient check earlier.
  const at = src.lastIndexOf("if (transient) {");
  assert.notEqual(at, -1);
  const branch = src.slice(at, src.indexOf("\n  return null;", at));
  // The direct attempt already opened an "other" backoff, so without recording
  // this the gate shuts on the next card and none load until it lapses.
  assert.ok(
    branch.includes("readmeFallbackProven = true"),
    "a successful fallback has to promote, or it only helps the first card",
  );
  // And readmeViaBackend has to read it back, or the flag is decorative.
  const gateAt = src.indexOf("export function readmeViaBackend");
  const gate = src.slice(gateAt, src.indexOf("\n}", gateAt));
  assert.ok(gate.includes("readmeFallbackProven"));
});

test("a blocked card does not declare the whole origin unreachable", async () => {
  const src = await read("../src/features/hub/lib/hf-readme.ts");
  // A filter on /raw says nothing about /api, the avatar CDN or the dataset
  // size lookup. Promoting to the origin-wide flag pushed all of them onto the
  // proxy and, through isDirectHubOffline, showed the listing as unreachable
  // when only the card was.
  assert.ok(
    !src.includes("setDirectHubBlocked"),
    "card evidence stays scoped to cards",
  );
  // Read-only is fine and is what keeps the listing's own diagnosis honoured.
  assert.ok(src.includes("isDirectHubBlocked()"));
});
