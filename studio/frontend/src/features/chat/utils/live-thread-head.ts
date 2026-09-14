// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type LiveThreadView = {
  threadListItem: () => { getState: () => { remoteId?: string | null } };
  thread: () => {
    getState: () => { messages: ReadonlyArray<{ id: string }> };
  };
};

const views = new Set<LiveThreadView>();

export function registerLiveThreadView(view: LiveThreadView): () => void {
  views.add(view);
  return () => {
    views.delete(view);
  };
}

// The branch picker moves the head only in memory, so storage alone cannot say which branch is on screen.
export function liveThreadHeadId(threadId: string): string | null {
  for (const view of views) {
    try {
      if (view.threadListItem().getState().remoteId !== threadId) continue;
      const headId = view.thread().getState().messages.at(-1)?.id;
      if (headId) return headId;
    } catch {
      // A view torn down mid-switch has no thread to read.
    }
  }
  return null;
}
