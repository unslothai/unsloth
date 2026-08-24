// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Which project a first send was made from, captured when the user pressed Send.
 *
 * `createStudioDbAdapter` files a materializing thread under the `projectId` its closure
 * holds, and `useRemoteThreadListRuntime` hands the core a fresh adapter on every render
 * (`__internal_setOptions`), so `initialize()` reads whatever project the provider is
 * showing AT MATERIALIZATION TIME, not the one Send was pressed in.
 *
 * Those used to be the same moment. They are not any more, in two independent ways:
 *
 *   - assistant-ui's `BaseComposerRuntimeCore.send()` empties the composer and then
 *     `await`s every incomplete attachment through the adapter before calling
 *     `handleSend`. Studio's PDF, DOCX and text adapters do their extraction there, so a
 *     first send carrying a document reaches the runtime seconds after the click.
 *   - the shared provider (#8908) survives a project switch, so that late send is still
 *     alive to land. Before it, switching projects unmounted the provider and the pending
 *     send died with it.
 *
 * Together: send a document into a new project chat, switch projects while it converts,
 * and the chat is created in, and persisted under, the project the user switched TO.
 *
 * The claim is per SEND rather than per thread creation on purpose. `switchToNewThread()`
 * reuses an untouched blank thread rather than minting one, so the same local id can be
 * the current new thread in two projects in a row; pinning at creation would file a
 * legitimate later send under the project the thread was first shown in.
 */

/** Every id the sending thread was known by, so the claim survives id re-keying. */
type ProjectClaim = {
  projectId: string | null;
  claimedAt: number;
};

const claimsByThreadId = new Map<string, ProjectClaim>();

/**
 * Bounded so an abandoned send cannot leak. A claim outlives only its own conversion, and
 * `initialize()` consumes it; anything older than this is a send that never landed.
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
 * Record the project a send belongs to, against every id its thread is currently known by.
 *
 * Called from the composer's send path, synchronously, before `send()` starts awaiting.
 */
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
 * The project a materializing thread's send was made from, if one was recorded.
 *
 * Returns the claim rather than the id so a claim OF null (a send from outside any
 * project) is distinguishable from no claim at all: the first must win over the adapter's
 * current project, the second must defer to it.
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
