// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

type HighlightEntry = {
  step: () => boolean;
  finish: () => void;
};

export type FenceHighlightQueueOptions = {
  schedule: (callback: () => void) => () => void;
  blocked: () => boolean;
  commit: (callback: () => void) => void;
};

export function createFenceHighlightQueue(options: FenceHighlightQueueOptions): {
  add: (step: () => boolean, finish: () => void) => () => void;
  flush: () => void;
  pending: () => number;
} {
  const entries = new Set<HighlightEntry>();
  let scheduled: { cancel: (() => void) | null } | null = null;
  let running = false;
  let flushing = false;

  const cancelScheduled = (): void => {
    const previous = scheduled;
    scheduled = null;
    previous?.cancel?.();
  };

  const schedule = (): void => {
    if (scheduled || running || flushing || entries.size === 0) return;
    const ticket = { cancel: null as (() => void) | null };
    scheduled = ticket;
    try {
      ticket.cancel = options.schedule(() => {
        if (scheduled !== ticket) return;
        scheduled = null;
        run();
      });
    } catch (error) {
      if (scheduled === ticket) scheduled = null;
      entries.clear();
      throw error;
    }
  };

  const run = (): void => {
    running = true;
    let entry: HighlightEntry | undefined;
    try {
      if (options.blocked()) return;
      entry = entries.values().next().value;
      if (!entry) return;
      const current = entry;
      let complete = false;
      options.commit(() => {
        if (entries.has(current)) complete = current.step();
      });
      if (entries.delete(current) && !complete) entries.add(current);
    } catch (error) {
      if (entry) entries.delete(entry);
      else entries.clear();
      throw error;
    } finally {
      running = false;
      schedule();
    }
  };

  const flush = (): void => {
    if (flushing) return;
    cancelScheduled();
    flushing = true;
    let failure: { error: unknown } | null = null;
    try {
      while (entries.size > 0) {
        try {
          options.commit(() => {
            while (entries.size > 0) {
              const entry = entries.values().next().value as HighlightEntry;
              entries.delete(entry);
              try {
                entry.finish();
              } catch (error) {
                failure ??= { error };
              }
            }
          });
        } catch (error) {
          entries.clear();
          failure ??= { error };
        }
      }
    } finally {
      flushing = false;
      schedule();
    }
    if (failure) throw failure.error;
  };

  return {
    add: (step, finish) => {
      const entry = { step, finish };
      entries.add(entry);
      schedule();
      return () => {
        entries.delete(entry);
        if (entries.size === 0) cancelScheduled();
      };
    },
    flush,
    pending: () => entries.size,
  };
}
