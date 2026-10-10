// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type LiveThreadView = {
  threads: () => { getState: () => { mainThreadId?: string } };
  threadListItem: () => {
    getState: () => { id?: string; remoteId?: string | null };
  };
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

// The branch picker moves the head only in memory, so storage cannot say which branch is shown.
export function liveThreadBranch(threadId: string): string[] | null {
  for (const view of views) {
    try {
      const item = view.threadListItem().getState();
      if (item.remoteId !== threadId) continue;
      if (item.id !== view.threads().getState().mainThreadId) continue;
      return view.thread().getState().messages.map((message) => message.id);
    } catch {
      // A view torn down mid-switch has no thread to read.
    }
  }
  return null;
}
