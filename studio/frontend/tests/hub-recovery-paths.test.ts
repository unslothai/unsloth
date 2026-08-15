// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// Making a lapsed backoff mean "probing" fixed the flapping, but "probing" only
// clears when a *listing* succeeds. Anything else gated on it, or on the dead
// iterator that a failed page leaves behind, has to keep a way back on its own.

function read(path: string): Promise<string> {
  return readFile(new URL(path, import.meta.url), "utf8");
}

/** A callback body, so a dependency array cannot satisfy an assertion. */
function body(source: string, start: string, end: string): string {
  const at = source.indexOf(start);
  assert.notEqual(at, -1, `could not find ${start}`);
  const to = source.indexOf(end, at);
  assert.ok(to > at, `could not find ${end} after ${start}`);
  return source.slice(at, to);
}

test("clients with their own request keep their own reachability", async () => {
  const page = await read("../src/features/hub/hub-page.tsx");
  // This value reaches useHubFeed and, through useModelsSelection, the selected
  // model's metadata lookup. Those are independent requests: on the discovery
  // phase they stayed blocked at "probing" until a listing succeeded, and the
  // Downloaded tab has no Retry to make one happen.
  assert.match(page, /const online = useOnlineStatus\(\);/);
  assert.ok(
    !/useHubAvailability\(\)\.phase/.test(page),
    "the feed's phase must not gate a client that never runs a listing",
  );
});

test("the panel still reads the classified cause, not that boolean", async () => {
  const page = await read("../src/features/hub/hub-page.tsx");
  // Reverting `online` must not quietly revert the diagnosis with it: the panel
  // is driven by searchFailure, which useDiscoverSearch reads off the phase.
  assert.match(page, /searchFailure,/);
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  assert.match(search, /const \{ phase, failure \} = useHubAvailability\(\);/);
  assert.match(search, /const online = phase === "available";/);
});

test("a dead feed is restarted by Load more, not left inert", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  const fn = body(search, "const fetchMore = useCallback", "\n  }, [");
  // hasMore is kept so the footer survives the failed page, so the button stays
  // on screen. Without this it called a fetchMore that returned false forever.
  assert.ok(fn.includes("needsRestart()"), "it has to notice the dead iterator");
  assert.ok(fn.includes("retrySearch()"), "and rebuild rather than resume");
  // Not on `online`: that is `phase === "available"`, which a lapsed backoff
  // never reaches without a successful listing, so the restart could never run.
  assert.ok(fn.includes("canProbe"), "a lapsed backoff must be allowed to probe");
  assert.match(search, /const canProbe = phase !== "unavailable";/);
});

test("the restart is not allowed to defeat the backoff", async () => {
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  const fn = body(search, "const fetchMore = useCallback", "\n  }, [");
  // Only the explicit Retry clears the window. Clearing it from a path the
  // scroll observer can reach would re-probe a dead origin on every scroll.
  assert.ok(!fn.includes("clearRemoteBackoff"), "the auto path must not clear it");
  const retry = body(search, "const handleRetrySearch = useCallback", "\n  }, [");
  assert.ok(retry.includes("clearRemoteBackoff()"), "an explicit click does");
});

test("rows on screen do not hide that the feed failed", async () => {
  const lists = await read("../src/features/hub/catalog/models-catalog-lists.tsx");
  // A cached feed substituted after a failed refresh renders rows with
  // hasMore false, which skipped the footer and the panel alike: once the toast
  // went there was nothing on screen saying so and nothing left to click.
  assert.match(lists, /\{\(hasMore \|\| searchError \|\| searchFailure\) && \(/);
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(footer.includes("failed={Boolean(searchError || searchFailure)}"));
  assert.ok(footer.includes("onRetry={onRetry}"));

  const states = await read("../src/features/hub/catalog/catalog-states.tsx");
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
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  // Ungating `enabled` is what let the error reach the panel, but it also let a
  // changed query, sort or channel build a new iterator and fire immediately.
  // Each attempt failed and re-armed the 30s window, so it never elapsed.
  // Either term does it. The dataset hook splits them because its `enabled`
  // also empties the rendered rows; `paused` holds the request on its own.
  for (const m of search.matchAll(/enabled: ([^,\n]+),/g)) {
    const gate = m[1].includes("canProbe") || search.includes("paused: !canProbe");
    assert.ok(gate, `an automatic search must respect the live backoff: ${m[1]}`);
  }
  // Only "unavailable" holds it: gating on `online` would never let a lapsed
  // window re-probe, since that requires a listing to have already succeeded.
  assert.match(search, /const canProbe = phase !== "unavailable";/);
});

test("gating the search again does not re-hide the error", async () => {
  // This is only safe because the disabled path stopped clearing it. Pin both
  // halves together: restoring the null would silently undo the whole fix.
  const paginated = await read(
    "../src/features/hub/hooks/use-hub-paginated-search.ts",
  );
  const disabled = body(paginated, "if (!enabled) {", "\n    // Same query");
  assert.ok(!/error: null/.test(disabled), "disabling must not erase the cause");
  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  assert.match(search, /const searchError = isDiscoverTab \? rawSearchError : null;/);
});

test("a footer retained over an outage can still act", async () => {
  const lists = await read("../src/features/hub/catalog/models-catalog-lists.tsx");
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  // An avatar or card failure marks the same origin, so the listing keeps its
  // rows and no searchError, and useHubInfiniteScroll is gated on `online`:
  // the button rendered enabled and did nothing for the whole window.
  assert.ok(
    footer.includes("failed={Boolean(searchError || searchFailure)}"),
    "unreachable is as good a reason to offer a re-probe as a failed page",
  );
  // And the same condition has to reach the render, or the prop is decorative:
  // with pagination exhausted, hasMore is false and no searchError is recorded,
  // so on `(hasMore || searchError)` the footer never appeared at all.
  assert.ok(
    lists.includes("{(hasMore || searchError || searchFailure) && ("),
    "an exhausted listing still has to show the outage",
  );
});

test("the retained footer names the cause, not just the staleness", async () => {
  const lists = await read("../src/features/hub/catalog/models-catalog-lists.tsx");
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  // This footer outlives the toast, so it is where the diagnosis has to live.
  // Reducing it to "out of date" threw away the thing this change exists for.
  assert.ok(
    footer.includes("failureText={searchFailure?.message ?? searchError ?? \"\"}"),
    "the classified cause has to reach the one control that persists",
  );
  const states = await read("../src/features/hub/catalog/catalog-states.tsx");
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
  const lists = await read("../src/features/hub/catalog/models-catalog-lists.tsx");
  // `online` is the 30s TTL. It flips back on a timer with nothing proven, so
  // the notice and its Retry disappeared while getHubPhase() still said
  // "probing" and searchFailure still held the cause. Keying on the cause ties
  // the notice to what clears it: a request that worked.
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(!/failed=\{[^}]*!online/.test(footer), "a timer must not retire it");
  assert.ok(
    !/\{\(hasMore \|\| searchError \|\| !online\) && \(/.test(lists),
    "nor take the whole footer off screen",
  );
  // And the cause is only cleared by markRemoteNetworkOnline, which only
  // fetchWithTimeout calls, and only on a resolved response.
  const network = await read("../src/features/hub/lib/network.ts");
  const online = body(network, "export function markRemoteNetworkOnline", "\nexport function markRemoteNetworkOffline");
  assert.ok(online.includes("lastFailureByOrigin.delete(origin)"), "success clears it");
  const fetchFn = body(network, "export async function fetchWithTimeout", "\n  } catch");
  assert.ok(fetchFn.includes("markRemoteNetworkOnline(origin)"), "on a response");
});

test("a row the mapper rejects is not treated as an outage", async () => {
  const paginated = await read(
    "../src/features/hub/hooks/use-hub-paginated-search.ts",
  );
  // next() already handed the item over, so the generator is fine. Letting the
  // throw out reached the same catch as a network error, which set iterDeadRef
  // and made needsRestart() true; every restart then re-read the same page and
  // hit the same row, so the feed could not get past it.
  const pull = body(paginated, "  let scanned = 0;", "\n  return { items, done: false");
  assert.ok(pull.includes("try {"), "the mapper call has to be guarded");
  assert.ok(pull.includes("mapped = mapItem(result.value);"), "inside the loop");
  assert.ok(pull.includes("continue;"), "and a bad row skipped, like a null one");
  // Only the mapper is inside it: widening the try to cover the await would
  // swallow the real network error this dead-iterator machinery exists for.
  const guarded = body(pull, "try {", "} catch");
  assert.ok(!guarded.includes("iter.next()"), "next() stays outside the guard");
  assert.ok(!guarded.includes("result.done"), "and so does the done check");
});

test("pausing dataset fetches leaves the rendered rows alone", async () => {
  const datasets = await read(
    "../src/features/hub/hooks/use-hub-dataset-search.ts",
  );
  // This hook's `enabled` does double duty: it gates the request AND returns []
  // from the results memo, so gating it on the backoff blanked every visible
  // dataset row for the window. The model hook has no such line, which is why
  // only datasets went blank.
  assert.match(datasets, /if \(!enabled\) return \[\];/);
  assert.ok(
    datasets.includes("enabled: enabled && !paused"),
    "the pause has to reach the request and stop there",
  );

  const search = await read("../src/features/hub/hooks/use-discover-search.ts");
  const call = body(search, "const datasetSearch = useHubDatasetSearch", "\n  });");
  assert.ok(
    call.includes("enabled: isDiscoverTab && isDatasetMode"),
    "visibility is what `enabled` means here",
  );
  assert.ok(call.includes("paused: !canProbe"), "the backoff goes to `paused`");
  // The model hook keeps the single gate: without the blanking memo there is
  // nothing to split, and folding it in would be a behaviour change.
  const model = body(search, "const modelSearch = useHubModelSearch", "\n  });");
  assert.ok(model.includes("enabled: canProbe &&"), "models are gated as before");
});
