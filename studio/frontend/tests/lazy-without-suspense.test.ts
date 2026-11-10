// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { onceLoader } from "../src/lib/once-loader.ts";

test("onceLoader loads once, shares the pending load and exposes the value", async () => {
  let calls = 0;
  const loader = onceLoader(async () => {
    calls += 1;
    return "panel";
  });
  assert.equal(loader.current(), null);
  const [a, b] = await Promise.all([loader.load(), loader.load()]);
  assert.deepEqual([a, b, calls, loader.current()], ["panel", "panel", 1, "panel"]);
  assert.equal(await loader.load(), "panel");
  assert.equal(calls, 1);
});

test("onceLoader retries after a failed load", async () => {
  let calls = 0;
  const loader = onceLoader(async () => {
    calls += 1;
    if (calls === 1) throw new Error("chunk failed");
    return 7;
  });
  await assert.rejects(loader.load(), /chunk failed/);
  assert.equal(loader.current(), null);
  assert.equal(await loader.load(), 7);
  assert.equal(calls, 2);
});

test("the browser pane is code-split without Suspense, so its first open is not held 300 ms", () => {
  const source = readFileSync(new URL("../src/features/chat/chat-page.tsx", import.meta.url), "utf8");
  assert.match(source, /const BrowserPanel = lazyWithoutSuspense\(\(\) =>\s*import\("@\/features\/browser\/browser-panel"\)/);
  assert.doesNotMatch(source, /lazy\(\(\) =>\s*import\("@\/features\/browser\/browser-panel"\)/);
  assert.match(source, /useEffect\(\(\) => preloadWhenIdle\(BrowserPanel\.preload\), \[\]\);/);
  for (const name of ["FullViewChatBar", "FullViewChatButton"]) {
    assert.match(source, new RegExp(`const ${name} = lazyWithoutSuspense\\(`));
  }
  const helper = readFileSync(new URL("../src/lib/lazy-without-suspense.tsx", import.meta.url), "utf8");
  assert.doesNotMatch(helper.match(/import \{[^}]*\} from "react";/)?.[0] ?? "", /\bSuspense\b|\blazy\b/);
  assert.match(helper, /if \(failure\) throw failure\.error;/);
  // A load that resolves between render and the effect must still re-render: the effect keys on what render saw.
  const effect = helper.match(/useEffect\(\(\) => \{[\s\S]*?\}, \[renderedLoaded\]\);/)?.[0];
  assert.ok(effect, "the load effect depends on the render-time loaded flag");
  assert.doesNotMatch(effect, /loader\.current\(\)/);
});
