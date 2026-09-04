// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/** The creation inputs a first send was made under, captured when the user pressed Send.
 *  `initialize()` snapshots project, temporary status, model and creation time when the thread
 *  MATERIALIZES, which used to be the same tick as the send. Two things separated them:
 *  `BaseComposerRuntimeCore.send()` awaits every incomplete attachment before `handleSend`, and
 *  the shared provider (#8908) survives a view switch. Navigating meanwhile moves all four, so a
 *  document sent from a Temporary Chat, or from one project into another, was created under
 *  wherever the user ended up. Per SEND, not per thread creation: `switchToNewThread()` reuses an
 *  untouched blank thread, so one local id can be the current new thread in two views in a row. */
export type ThreadCreationClaim = {
  projectId: string | null;
  incognito: boolean;
  modelId: string;
  modelGgufVariant: string | null;
  createdAt: number;
};

const claimsByThreadId = new Map<string, ThreadCreationClaim & { claimedAt: number }>();

/** Bounded rather than consumed on read: `initialize()` and the run adapter both read a claim with
 *  no ordering guarantee between them, so releasing on the first starves the second. Expiry is
 *  safe: a persisted row makes `ensureThreadRecord` early-return, a fresh thread always gets a
 *  fresh local id, and every send overwrites the ids it stamps. */
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

/** The composer, before `send()` starts awaiting; the queue, before it initializes. */
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

/** The whole claim, so a value OF null or false (no project, not temporary) stays distinguishable
 *  from no claim: the first wins over the store, the second defers to it. */
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
