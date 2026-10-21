// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type LiveThreadView = {
  subscribe?: (listener: () => void) => () => void;
  threads: () => { getState: () => { mainThreadId?: string } };
  threadListItem: () => {
    getState: () => { id?: string; remoteId?: string | null };
  };
  thread: () => {
    getState: () => { messages: ReadonlyArray<{ id: string }>; isRunning?: boolean };
  };
};

const views = new Set<LiveThreadView>();
const listeners = new Set<() => void>();

function notifyViewsChanged(): void {
  for (const listener of [...listeners]) listener();
}

export function subscribeLiveThreadViews(listener: () => void): () => void {
  listeners.add(listener);
  return () => { listeners.delete(listener); };
}

export function registerLiveThreadView(view: LiveThreadView): () => void {
  views.add(view);
  const unsubscribe = view.subscribe?.(notifyViewsChanged);
  notifyViewsChanged();
  return () => {
    unsubscribe?.();
    views.delete(view);
    notifyViewsChanged();
  };
}

// The branch picker moves the head only in memory, so storage alone cannot say which branch is on screen.
function liveThreadState(threadId: string) {
  for (const view of views) {
    try {
      const item = view.threadListItem().getState();
      if (item.remoteId !== threadId) continue;
      // A chat opened earlier keeps its runtime alive, but its thread() reads the chat on screen.
      if (item.id !== view.threads().getState().mainThreadId) continue;
      return view.thread().getState();
    } catch {
      // A view torn down mid-switch has no thread to read.
    }
  }
  return null;
}

export function liveThreadBranch(threadId: string): string[] | null {
  return liveThreadState(threadId)?.messages.map((message) => message.id) ?? null;
}

export function liveThreadRunning(threadId: string): boolean {
  return Boolean(liveThreadState(threadId)?.isRunning);
}
