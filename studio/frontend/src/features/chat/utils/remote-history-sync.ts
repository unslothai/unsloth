// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

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

/** Revisions read inside one of this tab's write transactions, before and after it. */
export type OwnHistoryWrite = {
  threadId: string;
  thread: readonly [before: number, after: number];
  history: readonly [before: number, after: number];
};

export type RemoteHistorySync = {
  watch: (watcher: RemoteHistoryWatcher) => () => void;
  poll: () => Promise<void>;
  watching: () => boolean;
  /** A write from this tab: the baseline moves past it only if nothing else wrote since. */
  noteOwnWrite: (write: OwnHistoryWrite) => void;
};

type OwnSteps = Map<number, number>;

function addStep(steps: OwnSteps, [before, after]: readonly [number, number]) {
  if (after > before) steps.set(before, after);
}

/** Follows own writes from the baseline; stops at the first step another writer made. */
function advance(steps: OwnSteps | undefined, from: number): number {
  let at = from;
  for (let next = steps?.get(at); next !== undefined; next = steps?.get(at)) {
    at = next;
  }
  if (steps) {
    // Counters only grow: steps behind the baseline can never be followed again.
    for (const before of steps.keys()) if (before < at) steps.delete(before);
  }
  return at;
}

export function createRemoteHistorySync(
  deps: RemoteHistorySyncDeps,
): RemoteHistorySync {
  // reading: revision an in-flight re-read was started for.
  const watchers = new Map<
    RemoteHistoryWatcher,
    {
      threadId: string | null;
      revision: number | null;
      reading?: number;
      // A baseline taken by a poll after the view loaded may already hold another device's write
      // the view lacks; own writes skip the re-read only once a re-read has synced the view.
      synced?: boolean;
    }
  >();
  let historyRevision: number | null = null;
  // Same for the sidebar: own writes skip its refresh only once a change event refreshed it.
  let historySynced = false;
  const historySteps: OwnSteps = new Map();
  const threadSteps = new Map<string, OwnSteps>();
  // Bumped per own write: a poll sent before one may answer with counters it already passed.
  let ownWrites = 0;
  let inFlight: Promise<void> | null = null;
  // An older server has no endpoint: stop asking.
  let unsupported = false;

  const run = async () => {
    const ids = new Set<string>();
    for (const watcher of watchers.keys()) {
      const id = watcher.threadId?.();
      if (id) ids.add(id);
    }
    const writesAtSend = ownWrites;
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
    const overtaken = ownWrites !== writesAtSend;
    if (historyRevision !== null && historySynced) {
      historyRevision = advance(historySteps, historyRevision);
    }
    if (
      overtaken &&
      historyRevision !== null &&
      result.revision < historyRevision
    ) {
      // Stale snapshot: an own write already moved past it.
    } else {
      if (historyRevision !== null && result.revision !== historyRevision) {
        historySynced = true;
        deps.onHistoryChanged();
      }
      historyRevision = result.revision;
    }
    for (const [watcher, read] of watchers) {
      let seen = read;
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
      const own = seen.synced
        ? advance(threadSteps.get(id), seen.revision)
        : seen.revision;
      if (own !== seen.revision) {
        seen = { ...seen, revision: own };
        watchers.set(watcher, seen);
      }
      if (seen.revision === revision) continue;
      if (overtaken && revision < own) continue;
      if (watcher.isIdle && !watcher.isIdle()) continue;
      const before = seen.revision;
      watchers.set(watcher, {
        threadId: id,
        revision,
        reading: revision,
        synced: seen.synced,
      });
      const applied = watcher.onThreadChanged?.();
      // Not awaited: a hung read must not stall the polls that drive the sidebar.
      void applied
        ?.catch(() => false)
        .then((ok) => {
          const now = watchers.get(watcher);
          if (now?.threadId !== id || now.reading !== revision) return;
          watchers.set(watcher, {
            threadId: id,
            revision: ok ? now.revision : before,
            synced: ok || now.synced,
          });
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
        if (watchers.size === 0) {
          historyRevision = null;
          historySynced = false;
          historySteps.clear();
          threadSteps.clear();
        }
      };
    },
    poll,
    watching: () => watchers.size > 0 && !unsupported,
    noteOwnWrite({ threadId, thread, history }) {
      if (watchers.size === 0) return;
      ownWrites++;
      addStep(historySteps, history);
      if (historyRevision !== null && historySynced) {
        historyRevision = advance(historySteps, historyRevision);
      }
      const open = new Set<string>();
      for (const watcher of watchers.keys()) {
        const id = watcher.threadId?.();
        if (id) open.add(id);
      }
      // Only open chats keep steps; one opened later takes a fresh baseline.
      for (const id of threadSteps.keys()) {
        if (!open.has(id)) threadSteps.delete(id);
      }
      if (!open.has(threadId)) return;
      let steps = threadSteps.get(threadId);
      if (!steps) threadSteps.set(threadId, (steps = new Map()));
      addStep(steps, thread);
      for (const [watcher, seen] of watchers) {
        if (
          seen.threadId !== threadId ||
          seen.revision === null ||
          !seen.synced
        ) {
          continue;
        }
        watchers.set(watcher, {
          ...seen,
          revision: advance(steps, seen.revision),
        });
      }
    },
  };
}
