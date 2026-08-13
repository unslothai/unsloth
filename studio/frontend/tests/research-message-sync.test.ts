// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import type { MessageRecord } from "../src/features/chat/types.ts";
import {
  hasResearchMetadata,
  reconcileServerManagedMessages,
} from "../src/features/chat/utils/research-message-sync.ts";

const STORED_PROMPT = {
  id: "user-1",
  threadId: "thread-1",
  parentId: null,
  role: "user",
  content: [{ type: "text", text: "What changed?" }],
  createdAt: 1,
} as unknown as MessageRecord;

const STORED_REPORT = {
  id: "assistant-1",
  threadId: "thread-1",
  parentId: "user-1",
  role: "assistant",
  content: [{ type: "text", text: "# Report", researchRunId: "run-1" }],
  metadata: { researchRunId: "run-1", serverManaged: true },
  createdAt: 2,
} as unknown as MessageRecord;

const STORED = [STORED_PROMPT, STORED_REPORT];

function drifted(record: MessageRecord, patch: object): MessageRecord {
  return { ...record, ...patch } as MessageRecord;
}

test("the client's drifted copy of a research report is replaced by the stored one", () => {
  // The live run object and serverRevision only exist client-side, so sending them back reads
  // as an edit to a server-managed message and the backend 409s the whole payload.
  const records = [
    STORED_PROMPT,
    drifted(STORED_REPORT, {
      content: [{ type: "text", text: "# Report (streaming)" }],
      metadata: {
        researchRunId: "run-1",
        serverManaged: true,
        serverRevision: 4,
        researchRun: { id: "run-1", status: "running" },
      },
    }),
  ];

  assert.deepEqual(reconcileServerManagedMessages(records, STORED), STORED);
});

test("the research prompt is protected as the parent of the report", () => {
  // The prompt carries no metadata of its own; the backend still refuses to let it change.
  const records = [
    drifted(STORED_PROMPT, { createdAt: 999, metadata: { model: "local" } }),
    STORED_REPORT,
  ];

  assert.deepEqual(reconcileServerManagedMessages(records, STORED), STORED);
});

test("ordinary messages in the same thread keep the client's edits", () => {
  const stored = [
    ...STORED,
    {
      id: "user-2",
      threadId: "thread-1",
      parentId: "assistant-1",
      role: "user",
      content: [{ type: "text", text: "old" }],
      createdAt: 3,
    } as unknown as MessageRecord,
  ];
  const edited = drifted(stored[2], {
    content: [{ type: "text", text: "new" }],
  });

  const synced = reconcileServerManagedMessages(
    [STORED_PROMPT, STORED_REPORT, edited],
    stored,
  );

  assert.deepEqual(synced, [STORED_PROMPT, STORED_REPORT, edited]);
});

test("a thread with no research turn is passed through untouched", () => {
  const plain = [drifted(STORED_PROMPT, { content: [] })];
  assert.equal(reconcileServerManagedMessages(plain, [STORED_PROMPT]), plain);
});

test("research ownership is detected from any of the backend's link keys", () => {
  assert.equal(hasResearchMetadata({ researchStatus: "completed" }), true);
  assert.equal(hasResearchMetadata({ researchPlanRevision: 1 }), true);
  assert.equal(hasResearchMetadata({ model: "local" }), false);
  assert.equal(hasResearchMetadata(null), false);
  assert.equal(hasResearchMetadata(undefined), false);
});

test("a relinked research prompt keeps the client's parent, not the pruned one", () => {
  // Deleting the message a research prompt hung off relinks it to the grandparent; echoing the
  // stored parentId would persist a link to the row the same pruning sync then deletes.
  const stored = [
    {
      id: "user-0",
      threadId: "thread-1",
      parentId: null,
      role: "user",
      content: [{ type: "text", text: "hello" }],
      createdAt: 0,
    } as unknown as MessageRecord,
    drifted(STORED_PROMPT, { parentId: "user-0" }),
    STORED_REPORT,
  ];
  const afterDelete = [
    drifted(STORED_PROMPT, { parentId: null }),
    STORED_REPORT,
  ];

  const synced = reconcileServerManagedMessages(afterDelete, stored);

  assert.equal(synced[0].parentId, null);
  assert.deepEqual(synced[0].content, STORED_PROMPT.content);
  assert.deepEqual(synced[1], STORED_REPORT);
});
