// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// The attach target can be chosen before the chat exists, so it is parked under a
// pending key. A chat is created by sending as often as by attaching, and only the
// attach path went through ensureThreadId: the choice was dropped and inherited.

import assert from "node:assert/strict";
import test from "node:test";

import { CHAT_PROJECT_ATTACHMENT_TARGET_KEY } from "../src/features/chat/utils/project-attachment-target.ts";

import { readSrc } from "./helpers/kit.ts";

const CHAT_RUNTIME_STORE = readSrc(
  "features/chat/stores/chat-runtime-store.ts",
);
const THREAD_DOCUMENTS_BAR = readSrc(
  "features/rag/components/thread-documents-bar.tsx",
);

const PENDING = "__pending__";
type Target = "project" | "chat";

/** adoptPendingProjectAttachmentTarget, as the store reducer applies it. */
function adopt(
  byThread: Record<string, Target>,
  threadId: string,
): Record<string, Target> {
  const pending = byThread[PENDING];
  if (pending === undefined || threadId in byThread) {
    return byThread;
  }
  const next = { ...byThread };
  delete next[PENDING];
  next[threadId] = pending;
  return next;
}

test("the chat that gets an id takes the choice made before it existed", () => {
  const after = adopt({ [PENDING]: "chat" }, "thread-1");
  assert.equal(after["thread-1"], "chat");
  assert.equal(PENDING in after, false, "and the next new chat starts clean");
});

test("a chat that made its own choice keeps it", () => {
  const before: Record<string, Target> = {
    [PENDING]: "chat",
    "thread-1": "project",
  };
  assert.equal(adopt(before, "thread-1")["thread-1"], "project");
});

test("nothing pending leaves the chat on the saved default", () => {
  const before: Record<string, Target> = { "thread-2": "chat" };
  assert.equal(adopt(before, "thread-1"), before, "no entry, so no override");
});

/** clearPendingProjectAttachmentTarget, as the store reducer applies it. */
function clearPending(
  byThread: Record<string, Target>,
): Record<string, Target> {
  if (!(PENDING in byThread)) {
    return byThread;
  }
  const next = { ...byThread };
  delete next[PENDING];
  return next;
}

// A composer abandoned without sending or attaching left its choice parked, and
// the next fresh chat read it as its own.
test("a choice made in a composer that never became a chat is dropped", () => {
  const after = clearPending({ [PENDING]: "chat", "thread-1": "project" });
  assert.equal(PENDING in after, false);
  assert.equal(after["thread-1"], "project", "real chats are untouched");
});

test("clearing with nothing pending changes nothing", () => {
  const before: Record<string, Target> = { "thread-1": "chat" };
  assert.equal(clearPending(before), before);
});

// Adoption has to run on both paths that turn a fresh composer into a chat.
test("both chat-creating paths adopt the pending choice", () => {
  // Attaching a file first: ensureThreadId materializes the thread.
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /initialize\(\)[\s\S]{0,300}?adoptPendingProjectAttachmentTarget\(remoteId, claim\)/,
  );
  // Sending a message first: the id arrives as a prop.
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /if \(!hadThreadId\) \{[\s\S]{0,200}?adoptPendingProjectAttachmentTarget\(threadId\)/,
  );
});

test("the composer clears its pending choice when it goes away", () => {
  assert.match(THREAD_DOCUMENTS_BAR, /clearPendingProjectAttachmentTarget\(\)/);
});

// Membership is read from the chat's own row, so there is a window where it is
// unknown. Attaching in it would file the file by guess.
test("attaching is held until the chat's project is known", () => {
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const projectUnresolved = threadProjectId === undefined;/,
    "unresolved has to be distinguishable from no project",
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /uploading \|\| projectUploading \|\| projectUnresolved/,
    "the attach controls hold",
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /if \(projectUnresolved\) \{\s*return;\s*\}/,
    "and a desktop drop stays in the store rather than draining",
  );
});

// A send outlives navigation, and hydration and a model load both run before the
// project is first resolved.
test("the run keeps the project it started in", () => {
  const source = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    source,
    // Whichever await comes first: the property is that the read is the run's
    // last synchronous statement, not which call follows it. The send-time stamp
    // takes precedence over the store (#9129: a document send lands after the
    // user may have navigated), but both reads stay on this side of the await.
    /const composerProjectIdAtSend = creationClaim\s*\? creationClaim\.projectId\s*: \(useChatRuntimeStore\.getState\(\)\.activeProjectId \?\? null\);\s*(?:\/\/[^\n]*\n\s*)*await /,
    "captured before the first await",
  );
  assert.match(
    source,
    /rememberComposerProjectForRun\(resolvedThreadId, composerProjectIdAtSend\)/,
  );
});

// A composer abandoned mid-materialization has its own pending entry dropped by
// its cleanup, and the next composer's lands under the same key: the abandoned
// promise resolving would hand that to the dead chat and delete it.
test("an abandoned composer cannot consume the next composer's choice", () => {
  let claim = 0;
  let byThread: Record<string, Target> = {};
  const setPending = (target: Target) => {
    claim += 1;
    byThread = { ...byThread, [PENDING]: target };
  };
  const clearPending = () => {
    if (!(PENDING in byThread)) return;
    claim += 1;
    const next = { ...byThread };
    delete next[PENDING];
    byThread = next;
  };
  const adoptWithClaim = (threadId: string, seen: number) => {
    if (seen !== claim) return;
    byThread = adopt(byThread, threadId);
  };

  // A picks "This chat" and starts materializing.
  setPending("chat");
  const seenByA = claim;
  // A is abandoned, B mounts and picks for itself.
  clearPending();
  setPending("chat");

  // Unguarded, this is the reported failure: the dead chat takes B's entry.
  const stolen = adopt(byThread, "thread-A");
  assert.equal(PENDING in stolen, false, "B's choice would be consumed");
  assert.equal(stolen["thread-A"], "chat");

  adoptWithClaim("thread-A", seenByA);
  assert.equal(byThread[PENDING], "chat", "B still has its choice");
  assert.equal(
    byThread["thread-A"],
    undefined,
    "and the dead chat took nothing",
  );

  // The ordinary case is untouched: nothing intervened, so B's own resolve adopts.
  adoptWithClaim("thread-B", claim);
  assert.equal(byThread["thread-B"], "chat");
  assert.equal(PENDING in byThread, false);
});

// The counter has to move on both ways the entry changes hands, or a claim
// taken before one of them still looks current afterwards.
test("both writers of the pending entry move the claim", () => {
  assert.equal(
    CHAT_RUNTIME_STORE.match(/pendingAttachmentTargetClaim \+= 1;/g)?.length,
    2,
    "set-pending and clear-pending both bump it",
  );
  assert.match(
    CHAT_RUNTIME_STORE,
    /if \(claim !== undefined && claim !== pendingAttachmentTargetClaim\) \{\s*return state;/,
  );

  // Read before initialize(), not after it resolves.
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const claim = readPendingAttachmentTargetClaim\(\);[\s\S]{0,200}?\.initialize\(\)/,
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /adoptPendingProjectAttachmentTarget\(remoteId, claim\)/,
  );
});

// Sending a normal message in a project composer creates the chat, and the page
// swaps ProjectComposer for Thread on the same state change. The bar holding
// the choice unmounts without ever seeing the new id, and the Thread's own bar
// mounts with the id already set, so neither of the bar's adopt paths runs.
test("the project composer's choice survives the swap to a thread", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  // The swap: this is what unmounts the composer.
  assert.match(page, /\{pendingNewThreadId \? \(/);
  // Adopted before it, in the path that learns the id, and only for the choice
  // this composer recorded: a later composer's choice is refused, not consumed.
  assert.match(
    page,
    /adoptPendingProjectAttachmentTarget\(\s*activeThreadId,\s*captured\?\.nonce === newThreadNonce \? captured\.claim : NO_SUCH_CLAIM,\s*\);\s*setPendingNewThreadId\(activeThreadId\);/,
  );
  assert.match(page, /const NO_SUCH_CLAIM = -1;/);
  // Captured by claim, so re-picking the same destination is still this
  // composer's choice rather than an unrecognised one.
  assert.match(page, /const claim = readPendingAttachmentTargetClaim\(\);/);
  assert.match(
    page,
    /if \(captured\?\.nonce === newThreadNonce && captured\.claim === claim\) \{\s*return;/,
  );

  // The claim the store hands out changes on every pending write, value or not.
  assert.match(
    CHAT_RUNTIME_STORE,
    /if \(threadId === null\) \{\s*pendingAttachmentTargetClaim \+= 1;/,
  );

  // Why the bar cannot cover it: the Thread's bar starts with an id, so the
  // first-id branch never fires for it.
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const hadThreadIdRef = useRef\(threadId !== null\);/,
  );
});

// A browser-local preference that a reset leaves behind outlives the reset: new
// attachments keep going to the scope the user just asked to forget.
test("the attach-target preference is cleared by the preferences reset", () => {
  const tab = readSrc("features/settings/tabs/general-tab.tsx");
  const start = tab.indexOf("const PREFS_KEYS");
  const keys = tab.slice(start, tab.indexOf("];", start));
  assert.ok(start >= 0, "PREFS_KEYS moved");
  assert.match(keys, /CHAT_PROJECT_ATTACHMENT_TARGET_KEY,/);
  // By the constant the store writes under, so the two cannot drift apart.
  assert.equal(
    CHAT_PROJECT_ATTACHMENT_TARGET_KEY,
    "unsloth_chat_project_attachment_target",
  );
});
