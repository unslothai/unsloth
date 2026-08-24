// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which project a first send was made from, captured when the user pressed Send.
 *
 * `initialize()` files a materializing thread under the adapter's `projectId`, and
 * `useRemoteThreadListRuntime` hands the core a fresh adapter every render
 * (`__internal_setOptions`), so that is the project on screen AT MATERIALIZATION TIME.
 * Two things separated it from the one Send was pressed in: `BaseComposerRuntimeCore.send()`
 * awaits every incomplete attachment before `handleSend`, and Studio's PDF/DOCX/text
 * adapters extract there; and the shared provider (#8908) now survives a project switch, so
 * that late send is still alive to land. Send a document, switch projects while it
 * converts, and the chat is created under the project switched TO.
 *
 * Per SEND, not per thread creation: `switchToNewThread()` reuses an untouched blank thread,
 * so one local id can be the current new thread in two projects in a row.
 */

/** Every id the sending thread was known by, so the claim survives id re-keying. */
type ProjectClaim = {
  projectId: string | null;
  claimedAt: number;
};

const claimsByThreadId = new Map<string, ProjectClaim>();

/** Bounded so an abandoned send cannot leak; a claim outlives only its own conversion. */
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

/** Called from the composer's send path, synchronously, before `send()` starts awaiting. */
export function claimThreadProject(
  threadIds: Iterable<string | null | undefined>,
  projectId: string | null,
): void {
  const now = Date.now();
  for (const threadId of threadIds) {
    if (!threadId) continue;
    claimsByThreadId.set(threadId, { projectId, claimedAt: now });
  }
  gc(now);
}

/**
 * Returns the claim rather than the id so a claim OF null (a send from outside any project)
 * is distinguishable from no claim: the first wins over the adapter's current project, the
 * second defers to it.
 */
export function readThreadProjectClaim(
  threadId: string,
): { projectId: string | null } | undefined {
  const claim = claimsByThreadId.get(threadId);
  if (!claim) return undefined;
  if (Date.now() - claim.claimedAt > CLAIM_TTL_MS) {
    claimsByThreadId.delete(threadId);
    return undefined;
  }
  return { projectId: claim.projectId };
}

/** Drop a claim once the thread it belongs to has been filed. */
export function releaseThreadProjectClaim(threadId: string): void {
  claimsByThreadId.delete(threadId);
}

export function __resetThreadProjectClaimsForTests(): void {
  claimsByThreadId.clear();
}
