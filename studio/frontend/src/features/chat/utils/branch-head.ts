// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { type ParentLinkedMessage, resolveSavedBranchHead } from "./message-order";

// storage keeps every branch but not which was open; this prefix joins account-switch purging.
const BRANCH_HEAD_PREFIX = "unsloth_chat_branch_head:";

function branchHeadKey(threadId: string): string {
  return `${BRANCH_HEAD_PREFIX}${threadId}`;
}

// best effort like composer drafts because storage can be blocked or full.
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
    // storage failures are non-fatal
  }
}

export function clearBranchHead(threadId: string): void {
  try {
    window.localStorage.removeItem(branchHeadKey(threadId));
  } catch {
    // storage failures are non-fatal
  }
}

/** saved branch head for `threadId`; undefined selects the newest stored row. */
export function savedBranchHead(
  threadId: string,
  messages: ParentLinkedMessage[],
): string | undefined {
  return resolveSavedBranchHead(messages, readBranchHead(threadId));
}
