// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ParentLinkedMessage, resolveSavedBranchHead } from "./message-order";

// The last message of the branch on screen, per thread. The rows record every branch but not
// which one was open, so without this a reload shows the newest row's branch: after a retry of an
// early reply and a switch back, that is the retry and nothing after it. The prefix puts it in the
// account-switch purge.
const BRANCH_HEAD_PREFIX = "unsloth_chat_branch_head:";

function branchHeadKey(threadId: string): string {
  return `${BRANCH_HEAD_PREFIX}${threadId}`;
}

// Best-effort like the composer drafts: storage throws when blocked or full.
function readBranchHead(threadId: string): string | null {
  try {
    return window.localStorage.getItem(branchHeadKey(threadId));
  } catch {
    return null;
  }
}

export function writeBranchHead(threadId: string, messageId: string): void {
  try {
    window.localStorage.setItem(branchHeadKey(threadId), messageId);
  } catch {
    // ignore write failures
  }
}

export function clearBranchHead(threadId: string): void {
  try {
    window.localStorage.removeItem(branchHeadKey(threadId));
  } catch {
    // ignore
  }
}

/** The head of the branch `threadId` was left on, among its stored rows. Undefined means the
 *  newest row's branch, as for a chat never opened in this browser. */
export function savedBranchHead(
  threadId: string,
  messages: ParentLinkedMessage[],
): string | undefined {
  return resolveSavedBranchHead(messages, readBranchHead(threadId));
}
