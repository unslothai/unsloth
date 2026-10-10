// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { create } from "zustand";

import { rendersAsRow } from "@/components/assistant-ui/thread-message-slot";
import type { ThreadMessageRole } from "@/components/assistant-ui/thread-message-slot";

export interface ForkBoundary {
  /** This thread's copies of inherited messages; a set because edits branch and switching restores. */
  messageIds: ReadonlySet<string>;
  sourceThreadId: string | null;
}

// Published on load: message rows render from a propless slot and cannot get the thread record.
export interface ForkBoundaryState {
  boundaryByThreadId: Record<string, ForkBoundary>;
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

/** Inherited message the divider follows on this branch; the inherited chain is a branch prefix. */
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
