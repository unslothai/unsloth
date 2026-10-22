// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { readText } from "./helpers/kit.ts";

// "probing" only clears when a listing succeeds, so other gated paths need their own way back.

/** A callback body, so a dependency array cannot satisfy an assertion. */
function body(source: string, start: string, end: string): string {
  const at = source.indexOf(start);
  assert.notEqual(at, -1, `could not find ${start}`);
  const to = source.indexOf(end, at);
  assert.ok(to > at, `could not find ${end} after ${start}`);
  return source.slice(at, to);
}

test("clients with their own request keep their own reachability", async () => {
  const page = await readText("../src/features/hub/hub-page.tsx");
  assert.match(page, /const online = useOnlineStatus\(\);/);
  assert.ok(
    !/useHubAvailability\(\)\.phase/.test(page),
    "the feed's phase must not gate a client that never runs a listing",
  );
});

test("the panel still reads the classified cause, not that boolean", async () => {
  const page = await readText("../src/features/hub/hub-page.tsx");
  assert.match(page, /searchFailure,/);
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  assert.match(search, /const \{ phase, failure \} = useHubAvailability\(\);/);
  assert.match(search, /const online = phase === "available";/);
});

test("a dead feed is restarted by Load more, not left inert", async () => {
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  const fn = body(search, "const fetchMore = useCallback", "\n  }, [");
  assert.ok(fn.includes("needsRestart()"), "it has to notice the dead iterator");
  assert.ok(fn.includes("retrySearch()"), "and rebuild rather than resume");
  // Not on `online`: a lapsed backoff never reaches "available" without a successful listing.
  assert.ok(fn.includes("canProbe"), "a lapsed backoff must be allowed to probe");
  assert.match(search, /const canProbe = phase !== "unavailable";/);
});

test("the restart is not allowed to defeat the backoff", async () => {
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  const fn = body(search, "const fetchMore = useCallback", "\n  }, [");
  // Only the explicit Retry clears the window; the scroll path would re-probe a dead origin.
  assert.ok(!fn.includes("clearRemoteBackoff"), "the auto path must not clear it");
  const retry = body(search, "const handleRetrySearch = useCallback", "\n  }, [");
  assert.ok(retry.includes("clearRemoteBackoff()"), "an explicit click does");
});

test("rows on screen do not hide that the feed failed", async () => {
  const lists = await readText("../src/features/hub/catalog/models-catalog-lists.tsx");
  assert.match(lists, /\{\(hasMore \|\| searchError \|\| searchFailure\) && \(/);
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(footer.includes("failed={Boolean(searchError || searchFailure)}"));
  assert.ok(footer.includes("onRetry={onRetry}"));

  const states = await readText("../src/features/hub/catalog/catalog-states.tsx");
  // To the next declaration: the prop destructuring contains a "\n}" of its own.
  const fn = body(
    states,
    "export function DiscoverFetchMoreFooter",
    "\nexport function InventoryErrorState",
  );
  assert.ok(fn.includes('failed && onRetry ? onRetry : onFetchMore'), "it must retry");
  assert.ok(fn.includes('failed ? "Try again" : "Load more"'), "and say so");
});

test("a live backoff is not bypassed by typing", async () => {
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  // Each ungated attempt failed and re-armed the 30s window, so it never elapsed.
  for (const m of search.matchAll(/enabled: ([^,\n]+),/g)) {
    const gate = m[1].includes("canProbe") || search.includes("paused: !canProbe");
    assert.ok(gate, `an automatic search must respect the live backoff: ${m[1]}`);
  }
  assert.match(search, /const canProbe = phase !== "unavailable";/);
});

test("gating the search again does not re-hide the error", async () => {
  const paginated = await readText(
    "../src/features/hub/hooks/use-hub-paginated-search.ts",
  );
  const disabled = body(paginated, "if (!enabled) {", "\n    // Same query");
  assert.ok(!/error: null/.test(disabled), "disabling must not erase the cause");
  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  assert.match(search, /const searchError = isDiscoverTab \? rawSearchError : null;/);
});

test("a footer retained over an outage can still act", async () => {
  const lists = await readText("../src/features/hub/catalog/models-catalog-lists.tsx");
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(
    footer.includes("failed={Boolean(searchError || searchFailure)}"),
    "unreachable is as good a reason to offer a re-probe as a failed page",
  );
  assert.ok(
    lists.includes("{(hasMore || searchError || searchFailure) && ("),
    "an exhausted listing still has to show the outage",
  );
});

test("the retained footer names the cause, not just the staleness", async () => {
  const lists = await readText("../src/features/hub/catalog/models-catalog-lists.tsx");
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(
    footer.includes("failureText={searchFailure?.message ?? searchError ?? \"\"}"),
    "the classified cause has to reach the one control that persists",
  );
  const states = await readText("../src/features/hub/catalog/catalog-states.tsx");
  const fn = body(
    states,
    "export function DiscoverFetchMoreFooter",
    "\nexport function InventoryErrorState",
  );
  assert.ok(
    fn.includes('{failureText || "These results may be out of date."}'),
    "shown when there is one, with the generic line only as a fallback",
  );
});

test("the notice outlives the backoff window, not the other way round", async () => {
  const lists = await readText("../src/features/hub/catalog/models-catalog-lists.tsx");
  // `online` is a 30s TTL that flips back on a timer; the notice keys on the cause instead.
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(!/failed=\{[^}]*!online/.test(footer), "a timer must not retire it");
  assert.ok(
    !/\{\(hasMore \|\| searchError \|\| !online\) && \(/.test(lists),
    "nor take the whole footer off screen",
  );
  const network = await readText("../src/features/hub/lib/network.ts");
  const online = body(network, "export function markRemoteNetworkOnline", "\nexport function markRemoteNetworkOffline");
  assert.ok(online.includes("lastFailureByOrigin.delete(origin)"), "success clears it");
  const fetchFn = body(network, "export async function fetchWithTimeout", "\n  } catch");
  assert.ok(fetchFn.includes("markRemoteNetworkOnline(origin)"), "on a response");
});

test("a row the mapper rejects is not treated as an outage", async () => {
  const paginated = await readText(
    "../src/features/hub/hooks/use-hub-paginated-search.ts",
  );
  // A mapper throw is not a dead iterator: next() already handed the item over.
  const pull = body(paginated, "  let scanned = 0;", "\n  return { items, done: false");
  assert.ok(pull.includes("try {"), "the mapper call has to be guarded");
  assert.ok(pull.includes("mapped = mapItem(result.value);"), "inside the loop");
  assert.ok(pull.includes("continue;"), "and a bad row skipped, like a null one");
  // Only the mapper is inside the try; covering the await would swallow real network errors.
  const guarded = body(pull, "try {", "} catch");
  assert.ok(!guarded.includes("iter.next()"), "next() stays outside the guard");
  assert.ok(!guarded.includes("result.done"), "and so does the done check");
});

test("pausing dataset fetches leaves the rendered rows alone", async () => {
  const datasets = await readText(
    "../src/features/hub/hooks/use-hub-dataset-search.ts",
  );
  // This hook's `enabled` also empties the rendered rows, so it cannot be gated on the backoff.
  assert.match(datasets, /if \(!enabled\) return \[\];/);
  assert.ok(
    datasets.includes("enabled: enabled && !paused"),
    "the pause has to reach the request and stop there",
  );

  const search = await readText("../src/features/hub/hooks/use-discover-search.ts");
  const call = body(search, "const datasetSearch = useHubDatasetSearch", "\n  });");
  assert.ok(
    call.includes("enabled: isDiscoverTab && isDatasetMode"),
    "visibility is what `enabled` means here",
  );
  assert.ok(call.includes("paused: !canProbe"), "the backoff goes to `paused`");
  const model = body(search, "const modelSearch = useHubModelSearch", "\n  });");
  assert.ok(model.includes("enabled: canProbe &&"), "models are gated as before");
});
