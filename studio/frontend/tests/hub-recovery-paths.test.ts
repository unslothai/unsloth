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
  assert.match(lists, /\{\(hasMore \|\| searchError\) && \(/);
  const footer = body(lists, "<DiscoverFetchMoreFooter", "/>");
  assert.ok(footer.includes("failed={Boolean(searchError)"));
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
  for (const m of search.matchAll(/enabled: ([^,\n]+),/g)) {
    assert.ok(
      m[1].includes("canProbe"),
      `an automatic search must respect the live backoff: ${m[1]}`,
    );
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
    footer.includes("failed={Boolean(searchError) || !online}"),
    "unreachable is as good a reason to offer a re-probe as a failed page",
  );
});
