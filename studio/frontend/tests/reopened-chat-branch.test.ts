// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import {
  orderBySelectedBranch,
  resolveSavedBranchHead,
} from "../src/features/chat/utils/message-order.ts";

// Three turns, then a retry of the first reply: the retry is the newest row.
const RETRIED_FIRST_REPLY = [
  { id: "u1", parentId: null, createdAt: 1, role: "user" },
  { id: "a1", parentId: "u1", createdAt: 2, role: "assistant" },
  { id: "u2", parentId: "a1", createdAt: 3, role: "user" },
  { id: "a2", parentId: "u2", createdAt: 4, role: "assistant" },
  { id: "u3", parentId: "a2", createdAt: 5, role: "user" },
  { id: "a3", parentId: "u3", createdAt: 6, role: "assistant" },
  { id: "a1-retry", parentId: "u1", createdAt: 7, role: "assistant" },
];

const ids = (rows: { id: string }[]) => rows.map((row) => row.id);

test("the branch left open is reopened, not the newest row's", () => {
  const headId = resolveSavedBranchHead(RETRIED_FIRST_REPLY, "a3");
  assert.equal(headId, "a3");
  assert.deepEqual(ids(orderBySelectedBranch(RETRIED_FIRST_REPLY, headId)), [
    "u1",
    "a1",
    "u2",
    "a2",
    "u3",
    "a3",
  ]);
});

test("with nothing saved the newest row's branch is shown, as before", () => {
  const headId = resolveSavedBranchHead(RETRIED_FIRST_REPLY, null);
  assert.equal(headId, undefined);
  assert.deepEqual(ids(orderBySelectedBranch(RETRIED_FIRST_REPLY, headId)), [
    "u1",
    "a1-retry",
  ]);
});

test("a saved head whose row is gone falls back to the newest row", () => {
  assert.equal(resolveSavedBranchHead(RETRIED_FIRST_REPLY, "deleted"), undefined);
});

test("turns added under the saved head since are part of its branch", () => {
  const extended = [
    ...RETRIED_FIRST_REPLY,
    { id: "u4", parentId: "a3", createdAt: 8, role: "user" },
    { id: "a4", parentId: "u4", createdAt: 9, role: "assistant" },
  ];
  assert.equal(resolveSavedBranchHead(extended, "a3"), "a4");
});

test("a saved head with retried children follows the newest child", () => {
  const retriedLast = [
    ...RETRIED_FIRST_REPLY,
    { id: "a3-retry", parentId: "u3", createdAt: 8, role: "assistant" },
  ];
  assert.equal(resolveSavedBranchHead(retriedLast, "u3"), "a3-retry");
});

test("the newest turn under the saved head wins over a newer sibling", () => {
  // Another device retried a3 under u3, then went back to a3 and continued from it.
  const continuedOlder = [
    ...RETRIED_FIRST_REPLY,
    { id: "a3-retry", parentId: "u3", createdAt: 8, role: "assistant" },
    { id: "u4", parentId: "a3", createdAt: 9, role: "user" },
    { id: "a4", parentId: "u4", createdAt: 10, role: "assistant" },
  ];
  assert.equal(resolveSavedBranchHead(continuedOlder, "u3"), "a4");
});

test("legacy rows without parents chain through to the newest", () => {
  const legacy = [
    { id: "u1", createdAt: 1, role: "user" },
    { id: "a1", createdAt: 2, role: "assistant" },
    { id: "u2", createdAt: 3, role: "user" },
  ];
  assert.equal(resolveSavedBranchHead(legacy, "a1"), "u2");
});
