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

test("only a proven success counts as available", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  // A lapsed backoff is "probing", not "available". Treating it as available is
  // what announced "Back online", retried, failed and looped.
  assert.match(search, /const online = phase === "available";/);
});

test("the feed does not re-run the request that just proved recovery", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  // Retry -> clearRemoteBackoff -> retrySearch -> the response resolves ->
  // markRemoteNetworkOnline -> online flips true -> this effect fired and called
  // retrySearch a second time, discarding the results that had just arrived and
  // re-issuing the same call. Only another surface's recovery should refresh us.
  assert.match(search, /const selfProbed = selfProbedRef\.current;/);
  assert.match(search, /if \(!selfProbed\) retrySearch\(\);/);
  // The announcement is unconditional: the recovery is real either way, so the
  // toast has to sit between the latch read and the guarded retry, ungated.
  const marker = "selfProbedRef.current = false;";
  const from = search.indexOf(marker, search.indexOf("const selfProbed ="));
  const to = search.indexOf("if (!selfProbed) retrySearch();");
  assert.ok(from !== -1 && to > from);
  const announce = search.slice(from + marker.length, to);
  assert.ok(announce.includes('toast.success("Back online"'));
  assert.ok(!announce.includes("selfProbed"), "the toast must not be gated on it");
  assert.ok(!/\bif\s*\(/.test(announce), "nor on anything else");
});

test("the probing flag tracks this feed's own requests", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  const at = search.indexOf("const selfProbedRef = useRef(false);");
  assert.notEqual(at, -1);
  const effect = search.slice(at, search.indexOf("}, [online, isLoading", at));
  // Latch first, reset second. Reversing them loses a request that was already
  // in flight when the outage was recorded, and its success then triggers the
  // redundant retry this exists to stop.
  const latch = effect.indexOf("selfProbedRef.current = true;");
  const reset = effect.indexOf("selfProbedRef.current = false;");
  assert.notEqual(latch, -1);
  assert.notEqual(reset, -1);
  assert.ok(latch < reset, "an in-flight request outranks the idle reset");
  assert.match(effect, /if \(isLoading \|\| isLoadingMore\)/);
  assert.match(effect, /else if \(!online\)/);
});
