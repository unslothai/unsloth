// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

// @huggingface/hub's listModels/listDatasets await the fetch inside the
// generator body, so a failed page throws out of it and the generator is
// finished. This is the language rule the Load more path has to respect.

test("a generator that threw yields done, not another request", async () => {
  let requests = 0;
  async function* listing() {
    for (const page of [1, 2]) {
      requests += 1;
      if (page === 2) throw new Error("Failed to fetch");
      yield page;
    }
  }

  const iter = listing();
  assert.deepEqual(await iter.next(), { value: 1, done: false });
  await assert.rejects(iter.next());
  const after = await iter.next();
  assert.equal(after.done, true, "the generator is finished once it throws");
  assert.equal(
    requests,
    2,
    "reusing it issues no further request, so pagination silently ends",
  );
});

test("a failed page marks the feed as needing a restart", async () => {
  const src = await readFile(
    new URL("../src/features/hub/hooks/use-hub-paginated-search.ts", import.meta.url),
    "utf8",
  );
  // Set on the fetchMore failure path and cleared only where a new generator is
  // built, so the flag tracks the live generator rather than the last error.
  assert.match(src, /iterDeadRef\.current = true;/);
  assert.match(src, /iterRef\.current = iter;\s*\n\s*iterDeadRef\.current = false;/);
  // Word-bounded: a bare substring is satisfied by any longer identifier, so
  // renaming this to needsRestartSomething left the check green.
  assert.match(src, /\bneedsRestart\b/);
});

test("the manual Load more restarts rather than resuming a dead feed", async () => {
  const src = await readFile(
    new URL("../src/features/hub/hooks/use-discover-search.ts", import.meta.url),
    "utf8",
  );
  const at = src.indexOf("const fetchMoreManual");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("\n  }, [", at)).replace(/\/\/.*$/gm, "");
  assert.ok(body.includes("clearRemoteBackoff()"), "an explicit click re-probes");
  assert.ok(
    body.includes("needsRestart()") && body.includes("retrySearch()"),
    "resuming a finished generator would end pagination instead of probing",
  );
});

test("the auto-fill refuses a dead feed instead of swallowing its error", async () => {
  const src = await readFile(
    new URL("../src/features/hub/hooks/use-hub-paginated-search.ts", import.meta.url),
    "utf8",
  );
  const at = src.indexOf("const fetchMore = useCallback");
  assert.notEqual(at, -1);
  const body = src.slice(at, src.indexOf("const iter = iterRef.current;", at));
  // hasMore stays true so the footer and its error survive, so the observer
  // fires again the moment the row scrolls into view. That pull returns done,
  // which sets error: null and hasMore: false: the failure disappears and the
  // list silently ends. Only retry(), which builds a new generator, may resume.
  assert.match(body, /if \(iterDeadRef\.current\) \{/);
  const guard = body.slice(body.indexOf("if (iterDeadRef.current) {"));
  assert.ok(guard.includes("return false;"), "a dead feed starts nothing");
});

test("the success path is the only thing that clears the error", async () => {
  const src = await readFile(
    new URL("../src/features/hub/hooks/use-hub-paginated-search.ts", import.meta.url),
    "utf8",
  );
  // Pins why the guard matters: a done page really does blank the error, so
  // without the guard above the observer overwrites the diagnosis with nothing.
  const at = src.indexOf("pullBatch(iter, mapItem, BATCH)");
  assert.notEqual(at, -1);
  const then = src.slice(at, src.indexOf(".catch(", at));
  assert.ok(then.includes("error: null"));
  assert.ok(then.includes("hasMore: !done"));
});
