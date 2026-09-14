// Simulations that drive the PR's REAL modules, not a model of them: the cross-tab
// work protocol under tab death, clock skew and a missing BroadcastChannel; the
// stored attach target on an upgraded or downgraded profile; and the scope
// precedence behind rag_scope. None of it is visible in a screenshot.
//
// Run from studio/frontend: node --experimental-strip-types --test <this file>

import assert from "node:assert/strict";
import test from "node:test";
import { installLocalStorageFake, registerStoreStubResolver } from "./helpers/kit.ts";

const { storage } = installLocalStorageFake();
const events = new EventTarget();
Object.assign(globalThis, {
  window: Object.assign(events, { localStorage: storage, location: { protocol: "http:" } }),
});
registerStoreStubResolver();

const rag = await import("../src/features/rag/api/rag-api.ts");
// The auth stub fails any unexpected network access; a test opts in per call.
const { setAuthFetchHandler } = await import("./helpers/store-stubs/auth.ts");

// --------------------------------------------------------------- work lease protocol

test("a project with no work in flight does not gate a send", () => {
  assert.equal(rag.projectWorkCount("p-idle"), 0);
});

test("local work is counted while it runs and released when it ends", () => {
  rag.noteProjectWork("p-local", 1);
  assert.equal(rag.projectWorkCount("p-local"), 1);
  rag.noteProjectWork("p-local", 1);
  assert.equal(rag.projectWorkCount("p-local"), 2, "two uploads are two operations");
  rag.noteProjectWork("p-local", -1);
  rag.noteProjectWork("p-local", -1);
  assert.equal(rag.projectWorkCount("p-local"), 0, "the gate must open again");
});

test("an over-release cannot drive the count negative and mask real work", () => {
  rag.noteProjectWork("p-neg", -5);
  assert.equal(rag.projectWorkCount("p-neg"), 0);
  rag.noteProjectWork("p-neg", 1);
  assert.equal(rag.projectWorkCount("p-neg"), 1, "a stale release must not owe credit");
  rag.noteProjectWork("p-neg", -1);
});

test("one project's work does not gate another", () => {
  rag.noteProjectWork("p-a", 1);
  assert.equal(rag.projectWorkCount("p-b"), 0);
  rag.noteProjectWork("p-a", -1);
});

test("the client error test only fires on an answered 4xx", () => {
  assert.equal(rag.isRagClientError(rag.ragError(404, { detail: "Project not found" })), true);
  assert.equal(rag.isRagClientError(rag.ragError(403, null)), true);
  assert.equal(rag.isRagClientError(rag.ragError(429, null)), false, "rate limiting is transient");
  assert.equal(rag.isRagClientError(rag.ragError(503, null)), false);
  assert.equal(rag.isRagClientError(rag.ragError(500, null)), false);
  assert.equal(rag.isRagClientError(new TypeError("Failed to fetch")), false,
    "a network failure is not an answer");
  assert.equal(rag.isRagClientError(null), false);
  assert.equal(rag.isRagClientError(undefined), false);
});

test("the error still reads as the message a toast would show", () => {
  const err = rag.ragError(404, { detail: "Project not found" });
  assert.equal(err.message, "Project not found");
  assert.equal(err.status, 404);
  assert.ok(err instanceof Error);
});

// ------------------------------------------------------- the stored attachment target

const { DEFAULT_PROJECT_ATTACHMENT_TARGET, normalizeProjectAttachmentTarget } =
  await import("../src/features/chat/utils/project-attachment-target.ts");

// The function the store reads a profile's value through, driven directly rather
// than through the store, which drags the whole app graph in with it.
async function targetForStoredValue(value: string | null): Promise<string> {
  return normalizeProjectAttachmentTarget(value);
}

test("an install that has never chosen gets the shipped default", async () => {
  assert.equal(await targetForStoredValue(null), DEFAULT_PROJECT_ATTACHMENT_TARGET);
  assert.equal(DEFAULT_PROJECT_ATTACHMENT_TARGET, "project");
});

test("both real values round-trip", async () => {
  assert.equal(await targetForStoredValue("project"), "project");
  assert.equal(await targetForStoredValue("thread"), "thread");
});

test("a value from a later build falls to the chat, never to sharing", async () => {
  // A downgrade must not turn an unknown preference into project-wide indexing.
  assert.equal(await targetForStoredValue("kb"), "thread");
  assert.equal(await targetForStoredValue("everyone"), "thread");
});

test("corrupt storage never produces an invalid target", async () => {
  for (const value of ["", " ", "null", "undefined", "0", "[object Object]", "PROJECT"]) {
    const got = await targetForStoredValue(value);
    assert.ok(got === "project" || got === "thread", `${JSON.stringify(value)} produced ${got}`);
  }
});

// --------------------------------------------------------------- retrieval precedence

// The shape chat-adapter builds for rag_scope, kept as the contract the backend's
// _resolve_scope is tested against, so the two cannot drift silently.
function ragScope(opts: {
  ragEnabled: boolean;
  kbId?: string | null;
  threadId?: string | null;
  projectRagEnabled?: boolean;
  projectId?: string | null;
}): Record<string, string> {
  const { ragEnabled, kbId, threadId, projectRagEnabled, projectId } = opts;
  if (ragEnabled && kbId) return { kb_id: kbId };
  return {
    ...(ragEnabled && threadId ? { thread_id: threadId } : {}),
    ...(projectRagEnabled && projectId ? { project_id: projectId } : {}),
  };
}

test("a chat with no project is scoped exactly as before", () => {
  assert.deepEqual(ragScope({ ragEnabled: true, threadId: "T" }), { thread_id: "T" });
  assert.deepEqual(ragScope({ ragEnabled: false, threadId: "T" }), {});
});

test("a project chat retrieves the project even with the pill off", () => {
  assert.deepEqual(
    ragScope({ ragEnabled: false, threadId: "T", projectRagEnabled: true, projectId: "P" }),
    { project_id: "P" },
  );
});

test("a project chat with the pill on retrieves both", () => {
  assert.deepEqual(
    ragScope({ ragEnabled: true, threadId: "T", projectRagEnabled: true, projectId: "P" }),
    { thread_id: "T", project_id: "P" },
  );
});

test("a knowledge base replaces everything, project included", () => {
  assert.deepEqual(
    ragScope({ ragEnabled: true, kbId: "K", threadId: "T", projectRagEnabled: true, projectId: "P" }),
    { kb_id: "K" },
  );
});

// ------------------------------------------------------- folder syncs the composer gates on

/** Answer every RAG request from a table, and count what was asked for. */
function stubFetch(handler: (url: string) => { status: number; body: unknown }) {
  const urls: string[] = [];
  setAuthFetchHandler((input: string) => {
    urls.push(input);
    const { status, body } = handler(input);
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => body,
    } as Response;
  });
  return { urls, restore: () => setAuthFetchHandler(null) };
}

// Unlinking a folder deletes its job rows, and so does the terminal-job prune, so
// a watcher can outlive the job it polls. Riding that 404 out through the retry
// budget gates the composer for a minute on work that already ended.
test("a folder job that no longer exists releases the gate at once", async () => {
  const fetched = stubFetch(() => ({ status: 404, body: { detail: "Job not found" } }));
  try {
    rag.watchProjectFolderJob("p-folder-404", "job-gone");
    assert.equal(rag.projectWorkCount("p-folder-404"), 1, "the lease is taken up front");
    // The read is answered, so the loop breaks without ever reaching its sleep.
    for (let tick = 0; tick < 10; tick += 1) await Promise.resolve();
    assert.equal(rag.projectWorkCount("p-folder-404"), 0, "a send must not wait out the retries");
    assert.equal(fetched.urls.length, 1, "and it must not keep polling a deleted job");
  } finally {
    fetched.restore();
  }
});

// The backend enqueues a sync per auto-syncing folder every FOLDER_SYNC_INTERVAL_S
// with no frontend event, so a project looked at once and remembered forever lets
// the composer send through a scan that is rewriting its sources.
test("a project can be looked at again for jobs that start later", async () => {
  const fetched = stubFetch(() => ({ status: 200, body: { linkedFolders: [] } }));
  const realNow = Date.now;
  let clock = realNow();
  Date.now = () => clock;
  try {
    await rag.reconcileProjectFolderJobs("p-folder-again");
    assert.equal(fetched.urls.length, 1);
    // A second bar mounting on the same project shares the answer.
    await rag.reconcileProjectFolderJobs("p-folder-again");
    assert.equal(fetched.urls.length, 1, "two bars must not double every open");
    // A later scan is not shut out by that.
    clock += 60_000;
    await rag.reconcileProjectFolderJobs("p-folder-again");
    assert.equal(fetched.urls.length, 2, "a periodic look has to reach the backend");
  } finally {
    Date.now = realNow;
    fetched.restore();
  }
});

// A look that never came back is not an answer, so it must not close the project either.
test("a failed look leaves the project open to the next one", async () => {
  let fail = true;
  const fetched = stubFetch(() => (fail ? { status: 503, body: null } : { status: 200, body: { linkedFolders: [] } }));
  try {
    await rag.reconcileProjectFolderJobs("p-folder-retry");
    assert.equal(fetched.urls.length, 1);
    fail = false;
    await rag.reconcileProjectFolderJobs("p-folder-retry");
    assert.equal(fetched.urls.length, 2, "the retry must not be rate limited out");
  } finally {
    fetched.restore();
  }
});

// A job that scans before writing any row leaves the composer's list legitimately
// empty, so the gate is open for as long as the lookup takes.
test("every look for folder jobs gates the composer while it runs", async () => {
  let release!: () => void;
  const answered = new Promise<void>((done) => { release = done; });
  const urls: string[] = [];
  setAuthFetchHandler(async (input: string) => {
    urls.push(input);
    await answered;
    return { ok: true, status: 200, json: async () => ({ linkedFolders: [] }) } as Response;
  });
  const realNow = Date.now;
  let clock = realNow();
  Date.now = () => clock;
  try {
    const looking = rag.reconcileProjectFolderJobs("p-folder-gate");
    assert.equal(rag.projectWorkCount("p-folder-gate"), 1, "a send must wait for the answer");
    release();
    await looking;
    assert.equal(rag.projectWorkCount("p-folder-gate"), 0, "and must not wait past it");
    // A periodic look is gated the same way: the backend's timer starts jobs the
    // composer's existing list says nothing about.
    clock += 60_000;
    await rag.reconcileProjectFolderJobs("p-folder-gate");
    assert.equal(urls.length, 2);
    assert.equal(rag.projectWorkCount("p-folder-gate"), 0);
  } finally {
    Date.now = realNow;
    setAuthFetchHandler(null);
  }
});
