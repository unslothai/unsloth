// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Invalidation list requests can complete out of order; pins the latest-request rule.

import assert from "node:assert/strict";
import test from "node:test";

import { readSrc } from "./helpers/kit.ts";

const THREAD = readSrc("components/assistant-ui/thread.tsx");
const CHAT_ADAPTER = readSrc("features/chat/api/chat-adapter.ts");
const RAG_API = readSrc("features/rag/api/rag-api.ts");
const THREAD_DOCUMENTS_BAR = readSrc("features/rag/components/thread-documents-bar.tsx");
const USE_LINKED_FOLDERS = readSrc("features/rag/components/use-linked-folders.ts");
const USE_RAG_DOCUMENTS = readSrc("features/rag/components/use-rag-documents.ts");

type Row = { id: string; status: string };

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

  const first = refresh(() => before.promise);
  const second = refresh(() => after.promise);

  after.resolve(INDEXING);
  await second;
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

// Clearing scope issues no request, so without a ticket the old response would republish.
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
  assert.match(
    USE_RAG_DOCUMENTS,
    /prev !== null && prev !== scopeKey\)[\s\S]{0,400}?refreshSeq\.current \+= 1;/,
    "clearing the scope must outrank a refresh already in flight",
  );
});

// Nothing that reports may come before the supersession check.
test("a failure is only reported for the request still being awaited", () => {
  assert.match(
    USE_RAG_DOCUMENTS,
    /if \(refreshSeq\.current !== requestId\) return true;(?:(?!toast\.error)[\s\S]){0,400}?if \(\s*!opts\?\.silentErrors &&\s*!useRagAvailabilityStore\.getState\(\)\.isUnavailable\(\)/,
  );
});

// Without RAG every project chat would fail one request, so no project scope is opened.
test("no project scope is opened where RAG cannot run", () => {
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const projectId =\s*\(ragEnabled && ragSource\.type === "kb"\) \|\| ragUnavailable\s*\? null\s*: \(threadProjectId \?\? null\);/,
  );
  assert.match(THREAD_DOCUMENTS_BAR, /projectId \? \{ type: "project", projectId \} : null/);
});

// Sources panel work has no composer row while running, so it must count as indexing.
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

  note("proj-1", -1);
  assert.equal(composerIndexing(INDEXING), true, "still indexing");
  assert.equal(composerIndexing(EMPTY), false);
});

test("the composer reads indexing from the hooks, not the listed rows", () => {
  assert.match(USE_RAG_DOCUMENTS, /noteProjectWork\(uploadingProjectId, 1\)/);
  assert.match(USE_RAG_DOCUMENTS, /noteProjectWork\(uploadingProjectId, -1\)/);
  assert.match(USE_RAG_DOCUMENTS, /workElsewhere > 0 \|\|/);
  // Tied to the job, not the component: leaving the Sources tab does not stop the sync.
  assert.match(USE_LINKED_FOLDERS, /watchProjectFolderJob\(scopeId, initial\.id\)/);
  assert.match(RAG_API, /noteProjectWork\(projectId, 1\)/);
  assert.match(RAG_API, /noteProjectWork\(projectId, -1\)/);
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const hasIndexing =\s*threadIndexing \|\| threadListLoading \|\| projectIndexing \|\| projectListLoading;/,
  );
});

// Each tick used to take a newer ticket, so a slow list never published.
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
  assert.match(
    USE_RAG_DOCUMENTS,
    /if \(!refreshInFlight\.current\) \{\s*void refresh\(\{ quiet: true \}\);/,
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /threadIndexing \|\| threadListLoading \|\| projectIndexing \|\| projectListLoading/,
  );
});

// A CustomEvent reaches only the tab that fired it.
test("an invalidation crosses tabs", () => {
  assert.match(
    RAG_API,
    /getProjectChannel\(\)\?\.postMessage\(\{ kind: "sources", projectId \}\)/,
  );
  assert.match(
    RAG_API,
    /getProjectChannel\(\)\?\.postMessage\(\{\s*kind: "work",\s*projectId,\s*delta,\s*from: TAB_ID,/,
  );
  assert.match(RAG_API, /new BroadcastChannel\(PROJECT_SOURCES_CHANGED_EVENT\)/);
  assert.match(USE_RAG_DOCUMENTS, /subscribeProjectSourcesBroadcast\(\);/);
});

// The reporting tab may close first, so remote work lapses instead of gating forever.
test("work reported by another tab lapses", () => {
  assert.match(RAG_API, /const REMOTE_WORK_TTL_MS = 120_000;/);
  assert.match(RAG_API, /until: Date\.now\(\) \+ REMOTE_WORK_TTL_MS/);
  assert.match(
    RAG_API,
    /if \(entry\.until > now\) remoteCount \+= entry\.count;\s*\}\s*return \(projectWorkInFlight\.get\(projectId\) \?\? 0\) \+ remoteCount;/,
  );
  assert.match(RAG_API, /clearTimeout\(timer\)/);

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

  note("proj-1", 1);
  assert.equal(counted("proj-1"), 1);
  now += TTL + 1;
  assert.equal(counted("proj-1"), 0, "the gate does not outlive the tab");
});

test("overlapping remote work is counted, not flagged", () => {
  assert.match(
    RAG_API,
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

test("a project delete is work on the project", () => {
  assert.match(USE_RAG_DOCUMENTS, /noteProjectWork\(removingProjectId, 1\)/);
  assert.match(USE_RAG_DOCUMENTS, /noteProjectWork\(removingProjectId, -1\)/);
});

test("the newest request owns the loading flag", () => {
  assert.match(
    USE_RAG_DOCUMENTS,
    /if \(refreshSeq\.current === requestId\) \{\s*refreshInFlight\.current = false;\s*setLoading\(false\);/,
  );
});

// Between the lease release and the quiet refresh, the composer would report nothing indexing.
test("the refresh an invalidation triggers is counted as work", () => {
  assert.match(
    USE_RAG_DOCUMENTS,
    /void loadProjectSources\(projectScopeId, \{ quiet: true \}\);/,
  );
  assert.match(
    USE_RAG_DOCUMENTS,
    /noteProjectWork\(projectId, 1\);\s*try \{[\s\S]{0,900}?\} finally \{\s*noteProjectWork\(projectId, -1\);/,
  );
});

// A backend restart misses a tick or two while the durable sync runs on.
test("a folder job watcher rides out a failed read", () => {
  assert.match(
    RAG_API,
    /if \(isRagClientError\(error\)\) break;\s*consecutiveFailures \+= 1;\s*if \(consecutiveFailures >= MAX_FOLDER_JOB_READ_FAILURES\) \{\s*break;/,
  );
  assert.match(RAG_API, /consecutiveFailures = 0;/);

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

// A long upload sends no delta, so without renewal the other tab stops counting it.
test("work in flight renews the deadline other tabs put on it", () => {
  assert.match(RAG_API, /const WORK_HEARTBEAT_MS = 45_000;/);
  // Send the absolute count: a delta cannot revive an entry the receiver already let lapse.
  assert.match(RAG_API, /setInterval\(answerWorkQuery, WORK_HEARTBEAT_MS\)/);
  assert.match(
    RAG_API,
    /if \(projectWorkInFlight\.size === 0\) \{\s*if \(workHeartbeat !== null\) \{\s*clearInterval\(workHeartbeat\)/,
  );

  const remote = new Map<string, { count: number; until: number }>();
  let now = 1_000;
  const counted = (projectId: string) => {
    const entry = remote.get(projectId);
    return entry && entry.until > now ? entry.count : 0;
  };
  const seed = (projectId: string, count: number) => {
    if (count <= 0) return;
    remote.set(projectId, {
      count: Math.max(counted(projectId), count),
      until: now + 120_000,
    });
  };

  seed("proj-1", 1);
  now += 90_000;
  seed("proj-1", 1); // heartbeat inside the deadline
  now += 90_000;
  assert.equal(counted("proj-1"), 1, "still gated three minutes in");

  now += 120_001;
  assert.equal(counted("proj-1"), 0, "lapsed while nothing was heard");
  seed("proj-1", 1);
  assert.equal(counted("proj-1"), 1, "revived by the heartbeat after it lapsed");
});

// Clearing to null starts no request, so nothing else would clear the flags.
test("dropping the scope clears the flags no request will", () => {
  assert.match(
    USE_RAG_DOCUMENTS,
    /if \(scope\) \{[\s\S]{0,300}?: refresh\(\)\);\s*\} else \{[\s\S]{0,300}?refreshInFlight\.current = false;[\s\S]{0,200}?setLoading\(false\);/,
  );

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

// The probe caches for 30s, so the watcher must drop it before releasing a send.
test("a folder job drops the cached answer before the gate", () => {
  assert.match(
    RAG_API,
    /announceProjectSourcesUpdated\(projectId\);\s*noteProjectWork\(projectId, -1\);/,
  );
});

// BroadcastChannel does not replay, so a new tab must ask what is running.
test("a tab that opens mid-upload asks what is already running", () => {
  assert.match(RAG_API, /askForWorkInFlight\(\);\s*return projectChannel;/);
  assert.match(RAG_API, /postMessage\(\{ kind: "work-query" \}\)/);
  assert.match(
    RAG_API,
    /channel\.postMessage\(\{ kind: "work-state", projectId, count, from: TAB_ID \}\)/,
  );

  // Floor per sender: the answer can race an already-counted delta from the same tab.
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

// A single project-wide count would let the first upload to finish release the second.
test("work is counted per reporting tab, not per project", () => {
  assert.match(
    RAG_API,
    /const remoteProjectWork = new Map<\s*string,\s*Map<string, \{ count: number; until: number \}>\s*>\(\);/,
  );
  assert.match(RAG_API, /kind: "work",\s*projectId,\s*delta,\s*from: TAB_ID,/);
  assert.match(
    RAG_API,
    /postMessage\(\{ kind: "work-state", projectId, count, from: TAB_ID \}\)/,
  );
  assert.match(RAG_API, /if \(entry\.until > now\) remoteCount \+= entry\.count;/);

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

  set("tab-a", 1);
  set("tab-b", 1);
  assert.equal(total(), 2, "two uploads, not one");

  set("tab-a", 0);
  assert.equal(total(), 1);
  set("tab-b", 0);
  assert.equal(total(), 0);
});

test("a failed reconciling refresh is retried before the gate drops", () => {
  assert.match(USE_RAG_DOCUMENTS, /const REFRESH_RETRIES = 3;/);
  assert.match(
    USE_RAG_DOCUMENTS,
    /if \(await refresh\(\{ quiet: opts\?\.quiet, silentErrors: !last \}\)\) return;/,
  );
  assert.match(
    USE_RAG_DOCUMENTS,
    /\} finally \{\s*noteProjectWork\(projectId, -1\);/,
  );
  assert.match(
    USE_RAG_DOCUMENTS,
    /scope\.type === "project"\s*\? loadProjectSources\(scope\.projectId\)\s*: refresh\(\)/,
  );
  assert.match(USE_RAG_DOCUMENTS, /if \(refreshSeq\.current !== requestId\) return true;/);
});

// The backend starts the job before answering, so the gate is taken before the request.
test("a folder mutation takes the gate before its request", () => {
  assert.match(
    USE_LINKED_FOLDERS,
    /noteProjectWork\(projectWorkScopeId, 1\);\s*try \{\s*return await run\(\);\s*\} finally \{\s*noteProjectWork\(projectWorkScopeId, -1\);/,
  );
  assert.match(
    USE_LINKED_FOLDERS,
    /withProjectWork\(async \(\) => \{\s*const created = await createLinkedFolder\(/,
  );
  assert.match(
    USE_LINKED_FOLDERS,
    /withProjectWork\(async \(\) => \{\s*const started =\s*mode === "rebuild"/,
  );
  assert.match(
    USE_LINKED_FOLDERS,
    /withProjectWork\(\(\) => deleteLinkedFolder\(folderId, removeIndex\)\)/,
  );
  // Job lease is taken inside the request's, so a scope change cannot drop the count.
  assert.match(USE_LINKED_FOLDERS, /watchStartedJob\(created\.job\.id\);\s*return created;/);
  assert.match(USE_LINKED_FOLDERS, /watchStartedJob\(started\.job\.id\);\s*return started;/);
});

// After a reload the backend scans before writing rows, so the composer must ask for running syncs.
test("a project composer picks up a folder sync already running", () => {
  assert.match(
    RAG_API,
    /export async function reconcileProjectFolderJobs\(\s*projectId: string,\s*\): Promise<void>/,
  );
  assert.match(
    RAG_API,
    /if \(folder\.activeJobId\) \{\s*watchProjectFolderJob\(projectId, folder\.activeJobId\);/,
  );
  assert.match(
    RAG_API,
    /if \(\(folderReconcileNotBefore\.get\(projectId\) \?\? 0\) > now\) return;[\s\S]{0,400}?folderReconcileNotBefore\.set\(projectId, now \+ FOLDER_RECONCILE_MIN_GAP_MS\);/,
  );
  assert.match(RAG_API, /folderReconcileNotBefore\.delete\(projectId\);/);

  assert.match(USE_RAG_DOCUMENTS, /void reconcileProjectFolderJobs\(workScopeId\);/);
  // The backend schedules auto-sync jobs on its own timer, so one mount-time look misses later ones.
  assert.match(
    USE_RAG_DOCUMENTS,
    /const reconcile = setInterval\(\(\) => \{\s*void reconcileProjectFolderJobs\(workScopeId\);\s*\}, FOLDER_RECONCILE_INTERVAL_MS\);/,
  );
  // The lookup takes a lease before its first await, so the listener must register first.
  assert.match(
    USE_RAG_DOCUMENTS,
    /window\.addEventListener\(PROJECT_WORK_CHANGED_EVENT, read\);[\s\S]{0,400}?void reconcileProjectFolderJobs\(workScopeId\);/,
  );
  assert.match(USE_RAG_DOCUMENTS, /clearInterval\(reconcile\);/);
});

// Unlinking and history prune delete job rows, so a watcher can poll an id that never answers.
test("a folder job watcher stops on an answered 4xx", () => {
  const retryBudget = Number(
    /const MAX_FOLDER_JOB_READ_FAILURES = (\d+);/.exec(RAG_API)?.[1],
  );
  assert.ok(retryBudget > 0);

  const run = (clientError: boolean) => {
    let failures = 0;
    for (let tick = 0; tick < 600; tick += 1) {
      if (clientError) return tick;
      failures += 1;
      if (failures >= retryBudget) return tick;
    }
    return -1;
  };
  assert.equal(run(true), 0, "a 404 releases the gate on the first read");
  assert.equal(run(false), retryBudget - 1, "a network failure still rides out");
});

// isIndexing() only answers while the bar is mounted, so the queue checks the project itself.
test("a background prompt queue checks the project it will send to", () => {
  assert.doesNotMatch(
    THREAD,
    /if \(!item\.target\.usesThreadDocuments\) \{\s*return false;/,
  );
  assert.match(THREAD, /\? await resolveProjectId\(threadId, undefined, \{/);
  assert.match(THREAD, /composerProjectId: queueProjectId,/);
  // Work in flight counts as well as rows: an upload has no row until it lands.
  assert.match(
    THREAD,
    /if \(projectWorkCount\(projectId\) > 0\) \{\s*return true;/,
  );
  assert.match(
    THREAD,
    /const projectDocuments = await listProjectDocuments\(projectId\);\s*return projectDocuments\.some\(indexingDocument\);/,
  );
});

// Without a row, the store holds whichever project is on screen when the poll lands.
test("a queue in a chat with no row still waits on its project", () => {
  assert.match(
    THREAD,
    /const projectIdAtQueueStart = incognitoAtQueueStart\s*\?\s*null\s*:\s*\(chatStateAtQueueStart\.activeProjectId \?\? null\);/,
  );
  assert.match(THREAD, /getQueueProjectId: \(\) => projectIdAtQueueStart,/);
  assert.match(
    THREAD,
    /const queueProjectId = item\.target\.getQueueProjectId\(\);\s*const projectId = threadId\s*\?\s*await resolveProjectId\(threadId, undefined, \{\s*rethrowReadFailure: true,\s*composerProjectId: queueProjectId,\s*\}\)\s*:\s*queueProjectId;/,
  );
  assert.match(THREAD, /if \(threadId && item\.target\.usesThreadDocuments\) \{/);
});

// Unlinking deletes rows regardless of current scope, so the announcement must not be gated on it.
test("unlinking a folder announces for the project it was for", () => {
  assert.match(USE_LINKED_FOLDERS, /const unlinkedProjectId = projectWorkScopeId;/);
  assert.match(
    USE_LINKED_FOLDERS,
    /if \(unlinkedProjectId\) announceProjectSourcesUpdated\(unlinkedProjectId\);\s*if \(currentScopeKey\.current !== operationScopeKey\) return;/,
  );
});

// A failed read is not proof of no project; recording null misfiles the next attachment.
test("a failed project lookup leaves the scope unresolved", () => {
  assert.match(THREAD_DOCUMENTS_BAR, /const PROJECT_LOOKUP_RETRIES = 3;/);
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /for \(let attempt = 0; attempt < PROJECT_LOOKUP_RETRIES; attempt \+= 1\)/,
  );
  assert.doesNotMatch(
    THREAD_DOCUMENTS_BAR,
    /setResolved\(\{ threadId, trigger: activeProjectId, projectId: null \}\);/,
  );
  assert.match(THREAD_DOCUMENTS_BAR, /const projectUnresolved = threadProjectId === undefined;/);
  assert.match(THREAD_DOCUMENTS_BAR, /uploading \|\| projectUploading \|\| projectUnresolved/);
});

// Answering null on a failed read would dispatch the queued prompt and skip the retry.
test("a failed row read holds a queued prompt instead of releasing it", () => {
  assert.match(CHAT_ADAPTER, /opts\?: \{ rethrowReadFailure\?: boolean;/);
  assert.match(
    CHAT_ADAPTER,
    /\} catch \(error\) \{[\s\S]{0,200}?if \(opts\?\.rethrowReadFailure\) throw error;\s*return null;/,
  );
  assert.match(
    THREAD,
    /await resolveProjectId\(threadId, undefined, \{\s*rethrowReadFailure: true,/,
  );
  // Other callers fail soft on the send path, where they must not adopt the on-screen project.
  assert.equal(
    (THREAD.match(/rethrowReadFailure/g) ?? []).length,
    1,
    "only the queue probe rethrows",
  );
});

// A knowledge base replaces every other scope in rag_scope.
test("a knowledge-base queue does not wait on project sources", () => {
  assert.match(
    THREAD,
    /const usesKnowledgeBaseAtQueueStart =\s*chatStateAtQueueStart\.ragEnabled &&\s*chatStateAtQueueStart\.ragSource\.type === "kb";/,
  );
  assert.match(THREAD, /usesKnowledgeBase: usesKnowledgeBaseAtQueueStart,/);
  assert.match(
    THREAD,
    /if \(item\.target\.usesKnowledgeBase\) \{\s*return false;\s*\}[\s\S]{0,900}?const projectId = threadId/,
  );
  assert.match(
    CHAT_ADAPTER,
    /ragEnabled && ragSource\.type === "kb"\s*\? \{ kb_id: ragSource\.kbId \}/,
  );
});

// A retry resumes with the old project's lister and would publish into another composer.
test("a retry stops when the scope it started for is gone", () => {
  assert.match(USE_RAG_DOCUMENTS, /const startedFor = `project:\$\{projectId\}`;/);
  assert.match(
    USE_RAG_DOCUMENTS,
    /liveScopeKeyRef\.current = scopeKey;\s*\}, \[scopeKey\]\);/,
  );
  // The scope effect starting this load runs before the one recording the live scope.
  assert.match(
    USE_RAG_DOCUMENTS,
    /await new Promise\(\(resolve\) =>\s*setTimeout\(resolve, 1000 \* \(attempt \+ 1\)\),\s*\);[\s\S]{0,400}?if \(liveScopeKeyRef\.current !== startedFor\) return;/,
  );
});

// A thread gets its id before its row is written, and the store names the on-screen project.
test("a queue with no row yet falls back to its own project, not the store", () => {
  assert.match(
    CHAT_ADAPTER,
    /opts\?: \{ rethrowReadFailure\?: boolean; composerProjectId\?: string \| null \}/,
  );
  assert.match(
    CHAT_ADAPTER,
    /const composerProjectId =\s*opts\?\.composerProjectId !== undefined\s*\?\s*opts\.composerProjectId\s*:\s*useChatRuntimeStore\.getState\(\)\.activeProjectId;/,
  );
  assert.match(CHAT_ADAPTER, /if \(thread\) \{\s*composerProjectByPendingThread\.delete\(threadId\);/);
});

// Re-arming a full TTL per update pushes past an earlier sender's deadline.
test("the work timer is armed for the earliest sender deadline", () => {
  assert.match(
    RAG_API,
    /let earliest = Number\.POSITIVE_INFINITY;\s*for \(const entry of bySender\.values\(\)\) \{\s*earliest = Math\.min\(earliest, entry\.until\);/,
  );
  assert.match(RAG_API, /Math\.max\(0, earliest - Date\.now\(\)\)/);
  assert.match(RAG_API, /if \(entry\.until <= now\) live\.delete\(sender\);/);
  assert.match(
    RAG_API,
    /publishProjectWorkChanged\(projectId\);\s*armRemoteWorkExpiry\(projectId\);/,
  );

  const TTL = 120_000;
  const senders = new Map<string, number>();
  const armFor = () => Math.min(...senders.values());
  senders.set("tab-a", 1_000 + TTL);
  assert.equal(armFor(), 1_000 + TTL);
  senders.set("tab-b", 31_000 + TTL);
  assert.equal(armFor(), 1_000 + TTL, "A's deadline still owns the timer");
  senders.delete("tab-a");
  assert.equal(armFor(), 31_000 + TTL);
});
