// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { rendersAsRow } from "@/components/assistant-ui/thread-message-slot";
import type { ThreadMessageRole } from "@/components/assistant-ui/thread-message-slot";

/** Where a fork's inherited history ends, and the chat it came from. */
export interface ForkBoundary {
  /**
   * This thread's own copies of every inherited message.
   *
   * A set rather than one id because the divider is a fact about the branch on screen, not
   * about a message: editing an inherited turn starts a sibling branch and keeps the originals,
   * so the last inherited message on the new branch is an earlier one, and switching back has
   * to restore the old one.
   */
  messageIds: ReadonlySet<string>;
  /** The chat the divider links back to, null once it is gone. */
  sourceThreadId: string | null;
}

// Published on load: message rows render from a propless slot, so they cannot be handed the
// thread record. A thread missing here shows no divider.
export interface ForkBoundaryState {
  boundaryByThreadId: Record<string, ForkBoundary>;
  /** The message the divider currently follows, resolved against the branch on screen. */
  anchorByThreadId: Record<string, string>;
  setForkBoundary: (threadId: string, boundary: ForkBoundary | null) => void;
  setForkBoundaryAnchor: (threadId: string, anchor: string | undefined) => void;
}

function sameIds(a: ReadonlySet<string>, b: ReadonlySet<string>): boolean {
  if (a === b) return true;
  if (a.size !== b.size) return false;
  for (const id of a) if (!b.has(id)) return false;
  return true;
}

function same(a: ForkBoundary | undefined, b: ForkBoundary): boolean {
  return (
    a !== undefined &&
    a.sourceThreadId === b.sourceThreadId &&
    sameIds(a.messageIds, b.messageIds)
  );
}

/**
 * The inherited message the divider follows on this branch, or undefined for none.
 *
 * The inherited chain runs from the root to the fork point, so it is a prefix of whatever
 * branch is selected: this stops at the divergence rather than walking a thread that has grown
 * past it. Only a message that paints a row can carry the divider, and `isEditing` is false
 * here because the branch alone does not say which message is being edited; an edit composer
 * is a row either way, so the only effect is on a system message mid-edit.
 */
export function forkBoundaryAnchor(
  messages: readonly { id: string; role: ThreadMessageRole }[],
  inherited: ReadonlySet<string> | undefined,
): string | undefined {
  if (inherited === undefined || inherited.size === 0) return undefined;
  let anchor: string | undefined;
  for (const message of messages) {
    if (!inherited.has(message.id)) break;
    if (rendersAsRow(message.role, false)) anchor = message.id;
  }
  return anchor;
}

export const useForkBoundaryStore = create<ForkBoundaryState>()((set) => ({
  boundaryByThreadId: {},
  anchorByThreadId: {},
  setForkBoundary: (threadId, boundary) =>
    set((state) => {
      const current = state.boundaryByThreadId[threadId];
      if (boundary === null) {
        if (current === undefined) return state;
        const next = { ...state.boundaryByThreadId };
        delete next[threadId];
        return { boundaryByThreadId: next };
      }
      // Same values keep the identity, so the rows subscribed to this do not re-render.
      if (same(current, boundary)) return state;
      return {
        boundaryByThreadId: { ...state.boundaryByThreadId, [threadId]: boundary },
      };
    }),
  setForkBoundaryAnchor: (threadId, anchor) =>
    set((state) => {
      const current = state.anchorByThreadId[threadId];
      if (current === anchor) return state;
      if (anchor === undefined) {
        if (current === undefined) return state;
        const next = { ...state.anchorByThreadId };
        delete next[threadId];
        return { anchorByThreadId: next };
      }
      return {
        anchorByThreadId: { ...state.anchorByThreadId, [threadId]: anchor },
      };
    }),
}));

export function setForkBoundary(
  threadId: string,
  messageIds: Iterable<string> | null | undefined,
  sourceThreadId?: string | null,
): void {
  const ids = messageIds ? new Set(messageIds) : null;
  useForkBoundaryStore
    .getState()
    .setForkBoundary(
      threadId,
      ids && ids.size > 0
        ? { messageIds: ids, sourceThreadId: sourceThreadId ?? null }
        : null,
    );
}

export function setForkBoundaryAnchor(
  threadId: string | null,
  anchor: string | undefined,
): void {
  if (threadId === null) return;
  useForkBoundaryStore.getState().setForkBoundaryAnchor(threadId, anchor);
}
