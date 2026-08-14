// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// A project source mutation invalidates before and after itself, and each
// invalidation starts a list request. Those can complete out of order, and an
// earlier one landing last would put the pre-mutation list back: the composer's
// bar would show no sources and, worse, report nothing indexing, so a send could
// go out before the file it is meant to use finished indexing.
//
// This pins the latest-request rule useRagDocuments.refresh applies.

import assert from "node:assert/strict";
import test from "node:test";

type Row = { id: string; status: string };

/** The publish gate as refresh applies it: take a ticket, drop the result if a
 * newer request has started since. */
function makeRefresher(published: Row[][]) {
  let seq = 0;
  return async function refresh(list: () => Promise<Row[]>) {
    const requestId = ++seq;
    const rows = await list();
    if (seq !== requestId) return;
    published.push(rows);
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  return { promise, resolve };
}

const EMPTY: Row[] = [];
const INDEXING: Row[] = [{ id: "doc-1", status: "running" }];

test("a stale list response cannot replace a newer one", async () => {
  const published: Row[][] = [];
  const refresh = makeRefresher(published);
  const before = deferred<Row[]>();
  const after = deferred<Row[]>();

  // Both fired by the same upload: the pre-mutation refresh first.
  const first = refresh(() => before.promise);
  const second = refresh(() => after.promise);

  // The post-mutation request wins the race back.
  after.resolve(INDEXING);
  await second;
  // The pre-mutation request lands afterwards, carrying the empty list.
  before.resolve(EMPTY);
  await first;

  assert.deepEqual(
    published,
    [INDEXING],
    "only the newest request publishes, so the indexing row survives",
  );
});

test("responses arriving in order still publish the newest", async () => {
  const published: Row[][] = [];
  const refresh = makeRefresher(published);
  const before = deferred<Row[]>();
  const after = deferred<Row[]>();

  const first = refresh(() => before.promise);
  const second = refresh(() => after.promise);

  before.resolve(EMPTY);
  await first;
  after.resolve(INDEXING);
  await second;

  assert.deepEqual(published, [INDEXING]);
});

test("a lone refresh still publishes", async () => {
  const published: Row[][] = [];
  const refresh = makeRefresher(published);
  await refresh(async () => INDEXING);
  assert.deepEqual(published, [INDEXING]);
});
