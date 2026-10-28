// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

export type SettleQueue<T> = {
  add: (row: T) => void;
  forget: (row: T) => void;
};

export const createRemeasureQueue = <T>(
  frame: (callback: () => void) => void,
  remeasure: (row: T) => void,
  paused: () => boolean,
): SettleQueue<T> => {
  const queued = new Set<T>();
  let scheduled = false;
  const run = (): void => {
    scheduled = false;
    const next = queued.values().next();
    if (!next.done && !paused()) {
      queued.delete(next.value);
      remeasure(next.value);
    }
    if (queued.size !== 0) schedule();
  };
  const schedule = (): void => {
    if (scheduled) return;
    scheduled = true;
    frame(run);
  };
  return {
    add: (row) => {
      queued.add(row);
      schedule();
    },
    forget: (row) => {
      queued.delete(row);
    },
  };
};

export const createSettleQueue = <T>(
  frame: (callback: () => void) => void,
  settle: (row: T) => void,
): SettleQueue<T> => {
  let queued = new Set<T>();
  let ripe = new Set<T>();
  let scheduled = false;
  const run = (): void => {
    scheduled = false;
    const due = ripe;
    ripe = queued;
    queued = new Set();
    for (const row of due) settle(row);
    if (ripe.size !== 0) schedule();
  };
  const schedule = (): void => {
    if (scheduled) return;
    scheduled = true;
    frame(run);
  };
  return {
    add: (row) => {
      ripe.delete(row);
      queued.add(row);
      schedule();
    },
    forget: (row) => {
      queued.delete(row);
      ripe.delete(row);
    },
  };
};
