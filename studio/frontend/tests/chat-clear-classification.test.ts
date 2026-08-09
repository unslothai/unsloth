// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import assert from "node:assert/strict";
import test from "node:test";

import { classifyChatClearThreads } from "../src/features/chat/utils/chat-clear-classification.ts";

const classifyPendingThread = ({
  backendCleared = false,
  pendingCleanupConfirmed = false,
}: {
  backendCleared?: boolean;
  pendingCleanupConfirmed?: boolean;
}) =>
  classifyChatClearThreads({
    allThreadIds: ["pending"],
    backendThreadIds: new Set(),
    legacyThreadIds: new Set(),
    pendingThreadIds: new Set(["pending"]),
    backendInventoryLoaded: true,
    backendCleared,
    legacyCleared: true,
    pendingCleanupConfirmed,
  });

test("does not classify an absent pending create after a failed clear", () => {
  assert.deepEqual(classifyPendingThread({}), {
    deletedThreadIds: [],
    failedThreadIds: ["pending"],
  });
});

test("classifies a pending create after its cleanup confirms", () => {
  assert.deepEqual(classifyPendingThread({ pendingCleanupConfirmed: true }), {
    deletedThreadIds: ["pending"],
    failedThreadIds: [],
  });
});

test("classifies a pending create after the backend clear confirms", () => {
  assert.deepEqual(classifyPendingThread({ backendCleared: true }), {
    deletedThreadIds: ["pending"],
    failedThreadIds: [],
  });
});

test("still trusts absence for ids with no pending backend write", () => {
  assert.deepEqual(
    classifyChatClearThreads({
      allThreadIds: ["stable"],
      backendThreadIds: new Set(),
      legacyThreadIds: new Set(),
      pendingThreadIds: new Set(),
      backendInventoryLoaded: true,
      backendCleared: false,
      legacyCleared: true,
      pendingCleanupConfirmed: false,
    }),
    { deletedThreadIds: ["stable"], failedThreadIds: [] },
  );
});
