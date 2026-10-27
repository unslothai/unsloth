// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** Creation inputs captured at Send, since attachments and view switches can delay
 *  initialize(). Per send, because a blank thread id can be reused across views. */
export type ThreadCreationClaim = {
  projectId: string | null;
  incognito: boolean;
  modelId: string;
  modelGgufVariant: string | null;
  createdAt: number;
};

const claimsByThreadId = new Map<string, ThreadCreationClaim & { claimedAt: number }>();

/** TTL, not consume-on-read: initialize() and the run adapter both read it in any order. */
const CLAIM_TTL_MS = 10 * 60 * 1000;
const MAX_CLAIMS = 64;

function gc(now: number): void {
  for (const [threadId, claim] of claimsByThreadId) {
    if (now - claim.claimedAt > CLAIM_TTL_MS) {
      claimsByThreadId.delete(threadId);
    }
  }
  while (claimsByThreadId.size > MAX_CLAIMS) {
    const oldest = claimsByThreadId.keys().next();
    if (oldest.done) break;
    claimsByThreadId.delete(oldest.value);
  }
}

export function claimThreadCreation(
  threadIds: Iterable<string | null | undefined>,
  claim: ThreadCreationClaim,
): void {
  const now = Date.now();
  for (const threadId of threadIds) {
    if (!threadId) continue;
    claimsByThreadId.set(threadId, { ...claim, claimedAt: now });
  }
  gc(now);
}

/** Returns the whole claim so a null/false value stays distinct from no claim. */
export function readThreadCreationClaim(
  threadId: string,
): ThreadCreationClaim | undefined {
  const claim = claimsByThreadId.get(threadId);
  if (!claim) return undefined;
  if (Date.now() - claim.claimedAt > CLAIM_TTL_MS) {
    claimsByThreadId.delete(threadId);
    return undefined;
  }
  return {
    projectId: claim.projectId,
    incognito: claim.incognito,
    modelId: claim.modelId,
    modelGgufVariant: claim.modelGgufVariant,
    createdAt: claim.createdAt,
  };
}

export function __resetThreadCreationClaimsForTests(): void {
  claimsByThreadId.clear();
}
