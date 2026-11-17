// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

// Another device writes straight to the server, so nothing local fires: poll the change counters.
export const REMOTE_HISTORY_POLL_MS = 4_000;

export type RemoteHistoryRevision = {
  revision: number;
  threads: Record<string, number>;
};

export type RemoteHistoryWatcher = {
  threadId?: () => string | null | undefined;
  /** False while the view must not be rewritten (a run here, an open edit): the change waits. */
  isIdle?: () => boolean;
  /** Resolving false (the read failed) keeps the change pending for the next poll. */
  onThreadChanged?: () => Promise<boolean> | undefined;
};

export type RemoteHistorySyncDeps = {
  fetchRevision: (
    threadIds: readonly string[],
  ) => Promise<RemoteHistoryRevision | null>;
  onHistoryChanged: () => void;
  isVisible: () => boolean;
};

export type RemoteHistorySync = {
  watch: (watcher: RemoteHistoryWatcher) => () => void;
  poll: () => Promise<void>;
  watching: () => boolean;
};

export function createRemoteHistorySync(
  deps: RemoteHistorySyncDeps,
): RemoteHistorySync {
  const watchers = new Map<
    RemoteHistoryWatcher,
    { threadId: string | null; revision: number | null }
  >();
  let historyRevision: number | null = null;
  let inFlight: Promise<void> | null = null;
  // An older server has no endpoint; stop asking for the rest of the page's life.
  let unsupported = false;

  const run = async () => {
    const ids = new Set<string>();
    for (const watcher of watchers.keys()) {
      const id = watcher.threadId?.();
      if (id) ids.add(id);
    }
    let result: RemoteHistoryRevision | null;
    try {
      result = await deps.fetchRevision([...ids]);
    } catch {
      return;
    }
    if (result === null) {
      unsupported = true;
      return;
    }
    if (historyRevision !== null && result.revision !== historyRevision) {
      deps.onHistoryChanged();
    }
    historyRevision = result.revision;
    for (const [watcher, seen] of watchers) {
      const id = watcher.threadId?.() ?? null;
      const revision = id ? result.threads[id] : undefined;
      if (!id || revision === undefined) {
        watchers.set(watcher, { threadId: id, revision: null });
        continue;
      }
      // A thread just opened was loaded fresh; this read is its baseline.
      if (seen.threadId !== id || seen.revision === null) {
        watchers.set(watcher, { threadId: id, revision });
        continue;
      }
      if (seen.revision === revision) continue;
      if (watcher.isIdle && !watcher.isIdle()) continue;
      watchers.set(watcher, { threadId: id, revision });
      const applied = watcher.onThreadChanged?.();
      // Not awaited: a hung read must not stall the polls that drive the sidebar.
      void applied
        ?.catch(() => false)
        .then((ok) => {
          const now = watchers.get(watcher);
          if (!ok && now?.threadId === id && now.revision === revision) {
            watchers.set(watcher, seen);
          }
        });
    }
  };

  const poll = () => {
    if (unsupported || watchers.size === 0 || !deps.isVisible()) {
      return Promise.resolve();
    }
    inFlight ??= run().finally(() => {
      inFlight = null;
    });
    return inFlight;
  };

  return {
    watch(watcher) {
      watchers.set(watcher, { threadId: null, revision: null });
      return () => {
        watchers.delete(watcher);
        if (watchers.size === 0) historyRevision = null;
      };
    },
    poll,
    watching: () => watchers.size > 0 && !unsupported,
  };
}
