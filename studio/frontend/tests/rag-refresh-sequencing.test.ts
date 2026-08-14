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

// A project's sources change from the Sources panel as well as the composer:
// an upload the panel invalidates before as well as after, and a folder sync
// that only reports at start and completion. The composer holds no row for
// either while it runs, so re-listing alone left it reporting nothing indexing,
// and a send could go out that those sources could not reach.
test("work in the other instance counts as indexing", () => {
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
  assert.match(hook, /noteProjectWork\(uploadingProjectId, 1\)/);
  assert.match(hook, /noteProjectWork\(uploadingProjectId, -1\)/);
  assert.match(hook, /workElsewhere > 0 \|\|/);
  // A folder sync reports at start and completion only, so the rows it
  // creates land with nothing gating the composer in between.
  const folders = readFileSync(
    new URL(
      "../src/features/rag/components/use-linked-folders.ts",
      import.meta.url,
    ),
    "utf8",
  );
  // Tied to the job, not to the component that started it: leaving the Sources
  // tab aborts its event stream, the sync carries on.
  assert.match(folders, /watchProjectFolderJob\(scopeId, initial\.id\)/);
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /noteProjectWork\(projectId, 1\)/);
  assert.match(api, /noteProjectWork\(projectId, -1\)/);
  const bar = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    bar,
    /const hasIndexing =\s*threadIndexing \|\| projectIndexing \|\| projectListLoading;/,
  );
});

// A list slower than the poll interval used to retire itself: each tick took a
// newer ticket before the previous response landed, so every response was
// dropped. The row being watched never reached completed and a queued send
// waited on it forever.
test("a poll tick is skipped while one is still out", () => {
  let inFlight = false;
  let started = 0;
  const tick = () => {
    if (inFlight) return;
    inFlight = true;
    started += 1;
  };

  tick();
  tick();
  tick();
  assert.equal(started, 1, "one request, however many ticks pass");

  inFlight = false;
  tick();
  assert.equal(started, 2, "the next tick goes out once it has landed");
});

test("the poll and the initial list are wired that way", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /if \(!refreshInFlight\.current\) \{\s*void refresh\(\{ quiet: true \}\);/,
  );
  // Reopening a project whose job is already running: nothing is listed yet and
  // no upload of ours is counted, so the gate has to hold for the first list.
  const bar = readFileSync(
    new URL(
      "../src/features/rag/components/thread-documents-bar.tsx",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    bar,
    /threadIndexing \|\| projectIndexing \|\| projectListLoading/,
  );
});

// Two tabs on the same project share its sources, and a CustomEvent reaches
// only the tab that fired it.
test("an invalidation crosses tabs", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    api,
    /getProjectChannel\(\)\?\.postMessage\(\{ kind: "sources", projectId \}\)/,
  );
  // Work in flight crosses too, or the other tab stays sendable through it.
  assert.match(
    api,
    /getProjectChannel\(\)\?\.postMessage\(\{ kind: "work", projectId, delta \}\)/,
  );
  assert.match(api, /new BroadcastChannel\(PROJECT_SOURCES_CHANGED_EVENT\)/);
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /subscribeProjectSourcesBroadcast\(\);/);
});

// The tab that started the work is the only one that can report it finished,
// and it may be closed or reloaded first. A count taken on its word alone would
// gate the project for the session, so what it reports lapses.
test("work reported by another tab lapses", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /const REMOTE_WORK_TTL_MS = 120_000;/);
  assert.match(api, /until: Date\.now\(\) \+ REMOTE_WORK_TTL_MS/);
  // Local and remote add up; a remote count past its deadline is dropped.
  assert.match(
    api,
    /remote && remote\.until > Date\.now\(\) \? remote\.count : 0;\s*return \(projectWorkInFlight\.get\(projectId\) \?\? 0\) \+ remoteCount;/,
  );
  // One pending wake-up per project, however chatty the other tab is.
  assert.match(api, /clearTimeout\(timer\)/);

  const TTL = 120_000;
  const remote = new Map<string, { count: number; until: number }>();
  let now = 1_000;
  const note = (projectId: string, delta: number) => {
    const entry = remote.get(projectId) ?? { count: 0, until: 0 };
    const count = Math.max(0, entry.count + delta);
    if (count === 0) {
      remote.delete(projectId);
    } else {
      remote.set(projectId, { count, until: now + TTL });
    }
  };
  const counted = (projectId: string) => {
    const entry = remote.get(projectId);
    return entry && entry.until > now ? entry.count : 0;
  };

  note("proj-1", 1);
  assert.equal(counted("proj-1"), 1, "the other tab's upload gates this one");

  note("proj-1", -1);
  assert.equal(counted("proj-1"), 0, "and releases when it says so");

  // The other tab goes away mid-upload and never reports the end.
  note("proj-1", 1);
  assert.equal(counted("proj-1"), 1);
  now += TTL + 1;
  assert.equal(counted("proj-1"), 0, "the gate does not outlive the tab");
});

// Two uploads overlapping in the other tab: the first to finish must not
// release the gate the second is still holding.
test("overlapping remote work is counted, not flagged", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /const count = Math\.max\(0, entry\.count \+ delta\);/);

  const remote = new Map<string, { count: number; until: number }>();
  const note = (projectId: string, delta: number) => {
    const entry = remote.get(projectId) ?? { count: 0, until: 0 };
    const count = Math.max(0, entry.count + delta);
    if (count === 0) {
      remote.delete(projectId);
    } else {
      remote.set(projectId, { count, until: Date.now() + 120_000 });
    }
  };

  note("proj-1", 1);
  note("proj-1", 1);
  note("proj-1", -1);
  assert.equal(remote.get("proj-1")?.count, 1, "one still running");
  note("proj-1", -1);
  assert.equal(remote.has("proj-1"), false);
});

// The composer says the source is gone the moment it is clicked, but it is
// still there until the DELETE returns, and the sources probe is only
// invalidated after that.
test("a project delete is work on the project", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /noteProjectWork\(removingProjectId, 1\)/);
  assert.match(hook, /noteProjectWork\(removingProjectId, -1\)/);
});

// A superseded request clearing the flag would report the list as known while
// the request that will publish is still out.
test("the newest request owns the loading flag", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /if \(refreshSeq\.current === requestId\) \{\s*refreshInFlight\.current = false;\s*setLoading\(false\);/,
  );
});
