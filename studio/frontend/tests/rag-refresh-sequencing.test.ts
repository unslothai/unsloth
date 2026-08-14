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
import { readFileSync } from "node:fs";
import test from "node:test";

type Row = { id: string; status: string };

/** The publish gate as refresh applies it: take a ticket, drop the result if a
 * newer request has started since. `clearScope` is the scope-change effect,
 * which takes a ticket without issuing a request of its own. */
function makeRefresher(published: Row[][]) {
  let seq = 0;
  async function refresh(list: () => Promise<Row[]>) {
    const requestId = ++seq;
    const rows = await list();
    if (seq !== requestId) return;
    published.push(rows);
  }
  refresh.clearScope = () => {
    seq += 1;
  };
  return refresh;
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

// Leaving a project for a chat with no project clears the scope, and a cleared
// scope issues no replacement request. Without a ticket taken on the way out,
// the project's own response lands afterwards and puts its sources back, in a
// chat that is not in that project.
test("a response for a scope that has been cleared does not publish", async () => {
  const published: Row[][] = [];
  const refresh = makeRefresher(published);
  const inFlight = deferred<Row[]>();

  const pending = refresh(() => inFlight.promise);
  refresh.clearScope();
  inFlight.resolve(INDEXING);
  await pending;

  assert.deepEqual(published, [], "the old scope's sources stay gone");
});

test("the scope-change effect takes a ticket on the way out", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /prev !== null && prev !== scopeKey\)[\s\S]{0,400}?refreshSeq\.current \+= 1;/,
    "clearing the scope must outrank a refresh already in flight",
  );
});

// A superseded request's failure describes a scope that is no longer shown, and
// a host where the vector extension cannot load answers 503 to every project
// source request: neither is worth a toast per composer opened.
test("a failure is only reported for the request still being awaited", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /refreshSeq\.current === requestId &&\s*!useRagAvailabilityStore\.getState\(\)\.isUnavailable\(\)/,
  );
});

// The composer mounts a project scope for every project chat, so a host that
// cannot run RAG would issue and fail one request per chat opened.
test("no project scope is opened where RAG cannot run", () => {
  const source = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    source,
    /projectId && !ragUnavailable \? \{ type: "project", projectId \} : null/,
  );
});

// A project's sources are uploaded from the Sources panel as well as the
// composer, and the panel invalidates before the POST as well as after. The
// composer holds no row for a file the panel is still uploading, so re-listing
// alone left it reporting nothing indexing for the length of the upload, and a
// send could go out that the file could not reach.
test("an upload in the other instance counts as indexing", () => {
  const inFlight = new Map<string, number>();
  const note = (projectId: string, delta: number) => {
    const next = (inFlight.get(projectId) ?? 0) + delta;
    if (next > 0) {
      inFlight.set(projectId, next);
    } else {
      inFlight.delete(projectId);
    }
  };
  // What the composer's instance reports: its own rows, plus uploads elsewhere.
  const composerIndexing = (rows: Row[]) =>
    (inFlight.get("proj-1") ?? 0) > 0 ||
    rows.some((row) => row.status === "pending" || row.status === "running");

  assert.equal(composerIndexing(EMPTY), false, "nothing happening");

  note("proj-1", 1);
  assert.equal(
    composerIndexing(EMPTY),
    true,
    "the panel's POST gates the composer before any row exists",
  );

  // The upload finishes and the row the panel created is now listed.
  note("proj-1", -1);
  assert.equal(composerIndexing(INDEXING), true, "still indexing");
  assert.equal(composerIndexing(EMPTY), false);
});

test("the composer reads indexing from the hooks, not the listed rows", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /noteProjectUpload\(uploadingProjectId, 1\)/);
  assert.match(hook, /noteProjectUpload\(uploadingProjectId, -1\)/);
  assert.match(hook, /uploadsElsewhere > 0 \|\|/);
  const bar = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(bar, /const hasIndexing = threadIndexing \|\| projectIndexing;/);
});
