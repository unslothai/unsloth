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
  // A mirror must not be bypassed, and a browser the proxy is already covering
  // cannot fetch the raw URL either. Both take the same route.
  assert.match(
    src,
    /export function readmeViaBackend\(\): boolean \{\s*\n\s*return hubProxyFirst\(\) \|\| isHubProxyServing\(\);/,
  );
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
  assert.ok(!body.includes("https://huggingface.co"));
});

test("the card's load effect is not gated off when the backend can serve it", async () => {
  const src = await read("../src/features/hub/catalog/model-readme.tsx");
  // online is !isHuggingFaceOffline(), which the proxy sets true precisely when
  // the backend route is the one that works. Gating on it alone showed the
  // unavailable card over a card we could fetch.
  assert.match(src, /if \(!online && !readmeViaBackend\(\)\) return;/);
});
