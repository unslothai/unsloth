// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// The Discover footer renders on `hasMore`, which useHubPaginatedSearch keeps
// after a page fails, so the button outlives the outage that broke it. Auto-fill
// must still stand down while the Hub is only probing, but a click is an
// explicit request and has to reach fetchMore.

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

// The callback body only, so the dependency array (which names both switches)
// cannot satisfy the assertions below. Comments go too: this is a claim about
// what the code reads, not about how it is described.
function body(source: string, start: string, what: string): string {
  const at = source.indexOf(start);
  assert.notEqual(at, -1, `could not find ${what}`);
  const open = source.indexOf("{", at);
  const close = source.indexOf("\n  }", open);
  assert.ok(open !== -1 && close > open, `could not read the body of ${what}`);
  return source.slice(open, close).replace(/\/\/.*$/gm, "");
}

test("a manual Load more does not ride on the auto-fill switch", async () => {
  const hook = await read("../src/features/hub/hooks/use-hub-infinite-scroll.ts");
  const manual = body(
    hook,
    "const fetchMoreManually",
    "fetchMoreManually in use-hub-infinite-scroll.ts",
  );
  assert.ok(
    manual.includes("manualEnabledRef"),
    "the click path must consult its own switch",
  );
  // Anchored, since manualEnabledRef ends in the name we are ruling out.
  assert.doesNotMatch(
    manual,
    /(?<![A-Za-z])enabledRef/,
    "gating the click on the auto-fill switch makes a visible button a no-op",
  );
});

test("the Discover feed keeps Load more live while probing", async () => {
  const page = await read("../src/features/hub/hub-page.tsx");
  assert.match(
    page,
    /const canProbe = online \|\| hubPhase === "probing";/,
    "probing means the backoff lapsed and we learned nothing, not that we gave up",
  );
  assert.match(
    page,
    /enabled: online && isDiscoverTab && hasMore,\s*\n\s*manualEnabled: canProbe && isDiscoverTab && hasMore,/,
    "auto-fill stays on `online` while the button follows canProbe",
  );
});
