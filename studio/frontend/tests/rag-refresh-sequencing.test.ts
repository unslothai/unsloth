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
    /if \(refreshSeq\.current !== requestId\) return true;\s*if \(\s*!opts\?\.silentErrors &&\s*!useRagAvailabilityStore\.getState\(\)\.isUnavailable\(\)/,
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
    /getProjectChannel\(\)\?\.postMessage\(\{\s*kind: "work",\s*projectId,\s*delta,\s*from: TAB_ID,/,
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
    /if \(entry\.until > now\) remoteCount \+= entry\.count;\s*\}\s*return \(projectWorkInFlight\.get\(projectId\) \?\? 0\) \+ remoteCount;/,
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
  assert.match(
    api,
    /setRemoteProjectWork\(projectId, from, Math\.max\(0, current \+ delta\)\);/,
  );

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

// The mutation releases its own work lease when its POST returns, and the
// invalidation it fires afterwards triggers a quiet refresh, which takes no
// loading gate. Between the two the composer would report nothing indexing
// while the rows it is about to receive are still being indexed.
test("the refresh an invalidation triggers is counted as work", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /noteProjectWork\(projectScopeId, 1\);\s*void \(async \(\) => \{/,
  );
  assert.match(hook, /\} finally \{\s*noteProjectWork\(projectScopeId, -1\);/);
});

// One failed read is not a finished job. A backend restart answers a tick or
// two while the durable sync runs on, and releasing there lets the composer
// send through sources that are still indexing.
test("a folder job watcher rides out a failed read", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  // The catch is inside the loop, so a failure does not reach the finally.
  assert.match(
    api,
    /catch \{\s*consecutiveFailures \+= 1;\s*if \(consecutiveFailures >= MAX_FOLDER_JOB_READ_FAILURES\) \{\s*break;/,
  );
  // A read that comes back clears the streak, so only a run of them gives up.
  assert.match(api, /consecutiveFailures = 0;/);

  // The same loop, run against reads that fail and then recover.
  const reads = ["fail", "fail", "fail", "running", "fail", "completed"];
  let failures = 0;
  let released = -1;
  for (let i = 0; i < reads.length; i += 1) {
    if (reads[i] === "fail") {
      failures += 1;
      if (failures >= 20) {
        released = i;
        break;
      }
      continue;
    }
    failures = 0;
    if (reads[i] === "completed") {
      released = i;
      break;
    }
  }
  assert.equal(
    released,
    5,
    "released by the terminal status, not by a failure",
  );
});

// An upload larger than the deadline sends no delta in between, so without a
// renewal the other tab stops counting it and becomes sendable mid-upload.
test("work in flight renews the deadline other tabs put on it", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(api, /const WORK_HEARTBEAT_MS = 45_000;/);
  assert.match(
    api,
    /channel\.postMessage\(\{ kind: "work", projectId, delta: 0, from: TAB_ID \}\)/,
  );
  // Started and stopped by the count itself, so an idle tab posts nothing.
  assert.match(
    api,
    /if \(projectWorkInFlight\.size === 0\) \{\s*if \(workHeartbeat !== null\) \{\s*clearInterval\(workHeartbeat\)/,
  );

  // A zero delta moves the deadline and leaves the count alone.
  const remote = new Map<string, { count: number; until: number }>();
  let now = 1_000;
  const note = (projectId: string, delta: number) => {
    const entry = remote.get(projectId) ?? { count: 0, until: 0 };
    const count = Math.max(0, entry.count + delta);
    if (count === 0) remote.delete(projectId);
    else remote.set(projectId, { count, until: now + 120_000 });
  };
  const counted = (projectId: string) => {
    const entry = remote.get(projectId);
    return entry && entry.until > now ? entry.count : 0;
  };

  note("proj-1", 1);
  now += 90_000;
  note("proj-1", 0); // heartbeat at 45s and 90s
  now += 90_000;
  assert.equal(counted("proj-1"), 1, "still gated three minutes in");

  // The tab goes away, the heartbeats stop, and the gate lapses as before.
  now += 120_001;
  assert.equal(counted("proj-1"), 0);
});

// Clearing to a null scope outranks the refresh still in flight but starts no
// replacement, so nothing reaches the sequence guard that would clear the
// flags. The composer reads the list as still unknown and holds every send.
test("dropping the scope clears the flags no request will", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /if \(scope\) \{[\s\S]{0,160}?void refresh\(\);\s*\} else \{[\s\S]{0,300}?refreshInFlight\.current = false;[\s\S]{0,200}?setLoading\(false\);/,
  );

  // The guard that leaves them set: the ticket has already moved on.
  let seq = 0;
  let loading = false;
  const start = () => {
    loading = true;
    return ++seq;
  };
  const settle = (requestId: number) => {
    if (seq === requestId) loading = false;
  };
  const ticket = start();
  seq += 1; // the scope change stands the request down
  settle(ticket);
  assert.equal(
    loading,
    true,
    "the request cannot clear it after being outranked",
  );
});

// The rows a folder sync writes are new sources, and the probe caches its
// answer for 30s. The watcher is the only observer once the panel unmounts, so
// a send released by it would still read the cached "no sources".
test("a folder job drops the cached answer before the gate", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    api,
    /invalidateProjectSources\(projectId\);\s*noteProjectWork\(projectId, -1\);/,
  );
});

// BroadcastChannel does not replay, so a tab opened mid-upload hears nothing
// until the next delta, which for an upload is its completion.
test("a tab that opens mid-upload asks what is already running", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  // Asked once, on the way in.
  assert.match(api, /askForWorkInFlight\(\);\s*return projectChannel;/);
  assert.match(api, /postMessage\(\{ kind: "work-query" \}\)/);
  assert.match(
    api,
    /channel\.postMessage\(\{ kind: "work-state", projectId, count, from: TAB_ID \}\)/,
  );

  // Per sender, and a floor within it: the answer can race a delta from the
  // same tab that is already counted, and must not lower it.
  const remote = new Map<string, { count: number; until: number }>();
  const seed = (from: string, count: number) => {
    if (count <= 0) return;
    const entry = remote.get(from);
    if (entry && entry.until > Date.now() && entry.count >= count) return;
    remote.set(from, { count, until: Date.now() + 120_000 });
  };

  seed("tab-a", 2);
  seed("tab-a", 1);
  assert.equal(
    remote.get("tab-a")?.count,
    2,
    "a smaller answer does not lower it",
  );
  seed("tab-a", 3);
  assert.equal(remote.get("tab-a")?.count, 3);
  seed("tab-b", 0);
  assert.equal(remote.has("tab-b"), false, "an idle tab seeds nothing");
});

// Two tabs uploading to one project are two operations. Merged into a single
// project-wide count, the first to finish clears the gate the second is still
// holding, and a send goes out mid-upload.
test("work is counted per reporting tab, not per project", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    api,
    /const remoteProjectWork = new Map<\s*string,\s*Map<string, \{ count: number; until: number \}>\s*>\(\);/,
  );
  // Every message says who sent it, or the counts cannot be kept apart.
  assert.match(api, /kind: "work",\s*projectId,\s*delta,\s*from: TAB_ID,/);
  assert.match(
    api,
    /postMessage\(\{ kind: "work-state", projectId, count, from: TAB_ID \}\)/,
  );
  assert.match(api, /if \(entry\.until > now\) remoteCount \+= entry\.count;/);

  // The reported sequence, per sender.
  const TTL = 120_000;
  const now = 1_000;
  const byProject = new Map<
    string,
    Map<string, { count: number; until: number }>
  >();
  const set = (from: string, count: number) => {
    const bySender = byProject.get("proj-1") ?? new Map();
    if (count <= 0) bySender.delete(from);
    else bySender.set(from, { count, until: now + TTL });
    byProject.set("proj-1", bySender);
  };
  const total = () => {
    let sum = 0;
    for (const entry of byProject.get("proj-1")?.values() ?? []) {
      if (entry.until > now) sum += entry.count;
    }
    return sum;
  };

  // Both existing tabs answer a late tab's query.
  set("tab-a", 1);
  set("tab-b", 1);
  assert.equal(total(), 2, "two uploads, not one");

  // One finishes; the other still holds the gate.
  set("tab-a", 0);
  assert.equal(total(), 1);
  set("tab-b", 0);
  assert.equal(total(), 0);
});

// The gate is released by the mutation's reconciling refresh, so a refresh that
// fails releases it with the composer holding no rows at all.
test("a failed reconciling refresh is retried before the gate drops", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /const REFRESH_RETRIES = 3;/);
  assert.match(
    hook,
    /if \(await refresh\(\{ quiet: true, silentErrors: !last \}\)\) return;/,
  );
  // The release is still guaranteed, retries or not.
  assert.match(hook, /\} finally \{\s*noteProjectWork\(projectScopeId, -1\);/);
  // A superseded request reports the list as known: the newer one owns it.
  assert.match(hook, /if \(refreshSeq\.current !== requestId\) return true;/);
});

// The backend creates and starts the job before it answers, so the request
// itself is time the project is changing with nothing gating on it.
test("a folder mutation takes the gate before its request", () => {
  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-linked-folders.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(
    hook,
    /noteProjectWork\(projectWorkScopeId, 1\);\s*try \{\s*return await run\(\);\s*\} finally \{\s*noteProjectWork\(projectWorkScopeId, -1\);/,
  );
  // Linking, syncing, rebuilding and unlinking all go through it.
  assert.match(
    hook,
    /withProjectWork\(async \(\) => \{\s*const created = await createLinkedFolder\(/,
  );
  assert.match(
    hook,
    /withProjectWork\(async \(\) => \{\s*const started =\s*mode === "rebuild"/,
  );
  assert.match(
    hook,
    /withProjectWork\(\(\) => deleteLinkedFolder\(folderId, removeIndex\)\)/,
  );
  // The job's own lease is taken inside the request's, so a scope change
  // between the response and trackJob cannot leave the project uncounted.
  assert.match(hook, /watchStartedJob\(created\.job\.id\);\s*return created;/);
  assert.match(hook, /watchStartedJob\(started\.job\.id\);\s*return started;/);
});

// A folder sync outlives the tab that started it. After a reload the watcher is
// gone, and the backend scans the folder before writing any rows, so the
// composer's own list is legitimately empty. Only the Sources panel lists
// linked folders, and a project opens on Chats, so the composer has to ask.
test("a project composer picks up a folder sync already running", () => {
  const api = readFileSync(
    new URL("../src/features/rag/api/rag-api.ts", import.meta.url),
    "utf8",
  );
  assert.match(
    api,
    /export async function reconcileProjectFolderJobs\(\s*projectId: string,\s*\): Promise<void>/,
  );
  assert.match(
    api,
    /if \(folder\.activeJobId\) \{\s*watchProjectFolderJob\(projectId, folder\.activeJobId\);/,
  );
  // Asked once per project, and a failed look does not count as an answer.
  assert.match(
    api,
    /if \(reconciledFolderProjects\.has\(projectId\)\) return;\s*reconciledFolderProjects\.add\(projectId\);/,
  );
  assert.match(api, /reconciledFolderProjects\.delete\(projectId\);/);

  const hook = readFileSync(
    new URL(
      "../src/features/rag/components/use-rag-documents.ts",
      import.meta.url,
    ),
    "utf8",
  );
  assert.match(hook, /void reconcileProjectFolderJobs\(workScopeId\);/);
});

// A queued prompt waiting on a project source outlives the bar that watched it.
// isIndexing() only answers while that bar is mounted and current, and project
// sources are retrieved whatever the Docs pill says, so the queue has to ask
// for the project itself rather than reading the thread scope alone.
test("a background prompt queue checks the project it will send to", () => {
  const thread = readFileSync(
    new URL("../src/components/assistant-ui/thread.tsx", import.meta.url),
    "utf8",
  );
  // The thread-scope check no longer returns early past the project one.
  assert.doesNotMatch(
    thread,
    /if \(!item\.target\.usesThreadDocuments\) \{\s*return false;/,
  );
  assert.match(thread, /const projectId = await resolveProjectId\(threadId\);/);
  // Work in flight counts as well as rows: an upload has no row until it lands.
  assert.match(
    thread,
    /if \(projectWorkCount\(projectId\) > 0\) \{\s*return true;/,
  );
  assert.match(
    thread,
    /const projectDocuments = await listProjectDocuments\(projectId\);\s*return projectDocuments\.some\(indexingDocument\);/,
  );
});
