// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Sending also creates a chat, not just attaching, so both paths must adopt the parked choice.

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

test("a choice made in a composer that never became a chat is dropped", () => {
  const after = clearPending({ [PENDING]: "chat", "thread-1": "project" });
  assert.equal(PENDING in after, false);
  assert.equal(after["thread-1"], "project", "real chats are untouched");
});

test("clearing with nothing pending changes nothing", () => {
  const before: Record<string, Target> = { "thread-1": "chat" };
  assert.equal(clearPending(before), before);
});

test("both chat-creating paths adopt the pending choice", () => {
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /initialize\(\)[\s\S]{0,300}?adoptPendingProjectAttachmentTarget\(remoteId, claim\)/,
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /if \(!hadThreadId\) \{[\s\S]{0,200}?adoptPendingProjectAttachmentTarget\(threadId\)/,
  );
});

test("the composer clears its pending choice when it goes away", () => {
  assert.match(THREAD_DOCUMENTS_BAR, /clearPendingProjectAttachmentTarget\(\)/);
});

// Membership is unknown until the chat's row loads; attaching then would guess.
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

test("the run keeps the project it started in", () => {
  const source = readSrc("features/chat/api/chat-adapter.ts");
  assert.match(
    source,
    // The read must be the run's last synchronous statement; the send-time stamp beats the store.
    /const composerProjectIdAtSend = creationClaim\s*\? creationClaim\.projectId\s*: \(useChatRuntimeStore\.getState\(\)\.activeProjectId \?\? null\);\s*(?:\/\/[^\n]*\n\s*)*await /,
    "captured before the first await",
  );
  assert.match(
    source,
    /rememberComposerProjectForRun\(resolvedThreadId, composerProjectIdAtSend\)/,
  );
});

// An abandoned composer's promise must not consume the next composer's pending entry.
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

  setPending("chat");
  const seenByA = claim;
  clearPending();
  setPending("chat");

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

  adoptWithClaim("thread-B", claim);
  assert.equal(byThread["thread-B"], "chat");
  assert.equal(PENDING in byThread, false);
});

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

  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const claim = readPendingAttachmentTargetClaim\(\);[\s\S]{0,200}?\.initialize\(\)/,
  );
  assert.match(
    THREAD_DOCUMENTS_BAR,
    /adoptPendingProjectAttachmentTarget\(remoteId, claim\)/,
  );
});

// The page swaps ProjectComposer for Thread on send, so neither bar's adopt path runs.
test("the project composer's choice survives the swap to a thread", () => {
  const page = readSrc("features/chat/chat-page.tsx");
  assert.match(page, /\{pendingNewThreadId \? \(/);
  assert.match(
    page,
    /adoptPendingProjectAttachmentTarget\(\s*activeThreadId,\s*captured\?\.nonce === newThreadNonce \? captured\.claim : NO_SUCH_CLAIM,\s*\);\s*setPendingNewThreadId\(activeThreadId\);/,
  );
  assert.match(page, /const NO_SUCH_CLAIM = -1;/);
  assert.match(page, /const claim = readPendingAttachmentTargetClaim\(\);/);
  assert.match(
    page,
    /if \(captured\?\.nonce === newThreadNonce && captured\.claim === claim\) \{\s*return;/,
  );

  assert.match(
    CHAT_RUNTIME_STORE,
    /if \(threadId === null\) \{\s*pendingAttachmentTargetClaim \+= 1;/,
  );

  assert.match(
    THREAD_DOCUMENTS_BAR,
    /const hadThreadIdRef = useRef\(threadId !== null\);/,
  );
});

test("the attach-target preference is cleared by the preferences reset", () => {
  const tab = readSrc("features/settings/tabs/general-tab.tsx");
  const start = tab.indexOf("const PREFS_KEYS");
  const keys = tab.slice(start, tab.indexOf("];", start));
  assert.ok(start >= 0, "PREFS_KEYS moved");
  assert.match(keys, /CHAT_PROJECT_ATTACHMENT_TARGET_KEY,/);
  assert.equal(
    CHAT_PROJECT_ATTACHMENT_TARGET_KEY,
    "unsloth_chat_project_attachment_target",
  );
});
