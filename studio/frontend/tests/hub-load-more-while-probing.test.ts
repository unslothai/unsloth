// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// The Discover footer renders on `hasMore`, which useHubPaginatedSearch keeps
// after a page fails, so the button outlives the outage that broke it. Auto-fill
// must still stand down, but a click has to reach fetchMore and re-probe rather
// than sit inert for the whole backoff window.

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

// The callback body only, so a dependency array cannot satisfy an assertion.
// Comments go too: this is a claim about what the code reads.
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
  assert.doesNotMatch(
    manual,
    /(?<![A-Za-z])enabledRef/,
    "gating the click on the auto-fill switch makes a visible button a no-op",
  );
  assert.ok(
    manual.includes("manualFetchMoreRef"),
    "the click must be able to run a different fetch from auto-fill's",
  );
});

test("the click path re-probes instead of waiting out the backoff", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  const manual = body(
    search,
    "const fetchMoreManual",
    "fetchMoreManual in use-discover-search.ts",
  );
  assert.ok(
    manual.includes("clearRemoteBackoff()"),
    "same contract as Retry: an explicit click tests the network now",
  );
  // The auto path must NOT clear it, or scrolling would defeat the backoff.
  const auto = body(search, "const fetchMore =", "fetchMore in use-discover-search.ts");
  assert.ok(!auto.includes("clearRemoteBackoff"));
  assert.ok(auto.includes("canProbe"));
});

test("the Discover feed wires the click path through, ungated by phase", async () => {
  const page = await read("../src/features/hub/hub-page.tsx");
  assert.match(page, /manualEnabled: isDiscoverTab && hasMore,/);
  assert.match(page, /manualFetchMore: fetchMoreDiscoverManual,/);
  assert.match(
    page,
    /enabled: online && isDiscoverTab && hasMore,/,
    "auto-fill stays on `online`",
  );
});

test("an answered status is not treated as a connectivity loss", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  // A 401 or 429 proves the origin answered. Folding it into the offline branch
  // raised "Can't reach Hugging Face" and then announced "Back online" after.
  assert.match(
    search,
    /const online =\s*\n?\s*phase === "available" \|\| failure\?\.kind === "http";/,
  );
});
