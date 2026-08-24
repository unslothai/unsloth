// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * The creation inputs a first send was made under, captured when the user pressed Send.
 *
 * `initialize()` snapshots project, temporary status, model and creation time when the
 * thread MATERIALIZES, which used to be the same tick as the send. Two things separated
 * them: `BaseComposerRuntimeCore.send()` awaits every incomplete attachment before
 * `handleSend` and Studio's PDF/DOCX/text adapters extract there, and the shared provider
 * (#8908) now survives a view switch, so that late send is still alive to land. Navigating
 * meanwhile moves all four: the adapter is rebuilt with the new project, and ChatPage's
 * view effect clears `incognito` for anything that is not a fresh single chat.
 *
 * So a document sent from a Temporary Chat, or from one project into another, is created
 * and persisted under wherever the user ended up. `ensureThreadRecord` already takes these
 * as snapshot parameters; this is what lets the caller supply them from the send.
 *
 * Per SEND, not per thread creation: `switchToNewThread()` reuses an untouched blank thread,
 * so one local id can be the current new thread in two views in a row.
 */
export type ThreadCreationClaim = {
  projectId: string | null;
  incognito: boolean;
  modelId: string;
  createdAt: number;
};

const claimsByThreadId = new Map<string, ThreadCreationClaim & { claimedAt: number }>();

/**
 * Bounded rather than consumed on read. A claim has two readers with no ordering guarantee
 * between them -- `initialize()`, which files the row, and the run adapter, which resolves
 * the project the RUN uses -- so releasing on the first read starves the second. Leaving it
 * to expire is safe: a persisted row makes `ensureThreadRecord` early-return, a fresh thread
 * always gets a fresh local id, and every send overwrites the ids it stamps.
 */
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

/**
 * Called from the send paths that can initialize a fresh thread -- the composer, before
 * `send()` starts awaiting, and the prompt queue, before it initializes on dispatch.
 */
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

/**
 * Returns the whole claim so a value OF null or false (no project, not temporary) is
 * distinguishable from no claim: the first wins over what the store holds now, the second
 * defers to it.
 */
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
    createdAt: claim.createdAt,
  };
}

export function __resetThreadCreationClaimsForTests(): void {
  claimsByThreadId.clear();
}
