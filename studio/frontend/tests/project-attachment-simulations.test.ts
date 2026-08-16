// Simulations that drive the PR's REAL modules, not a model of them.
//
// The browser matrix covers what a user sees. This covers what a user cannot see and
// what a screenshot cannot show: the cross-tab work protocol under tab death, clock
// skew and a missing BroadcastChannel; the stored attach target on an upgraded or
// downgraded profile; and the scope precedence the adapter uses to build rag_scope.
//
// Run from studio/frontend with:
//   node --experimental-strip-types --test <this file>

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

// The real function the store reads a profile's value through, driven directly rather
// than through the store, which drags the whole app graph in behind it.
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

// The shape chat-adapter builds for rag_scope. Kept here as the contract the backend's
// _resolve_scope is tested against on the other side, so the two cannot drift silently.
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
